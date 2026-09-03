"""Parity and compatibility tests for manual-optimization policies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.core.optimizer import LightningOptimizer

from stable_pretraining import Module
from stable_pretraining.geodro_lejepa import (
    AdversaryScope,
    CoherentHardnessGeoDROLeJEPALoss,
    GeoDROOptimizerStepPolicy,
    GraphTransportGeoDROJEPALoss,
)
from stable_pretraining.optimization_policy import (
    StandardOptimizationPolicy,
    resolve_optimization_policy,
)


EXAMPLES_CONFIG_DIR = Path(__file__).resolve().parents[3] / "examples"


class _StandardHarness:
    def __init__(self, *, accum_steps: int = 2, mock: bool = False):
        self.trainer = SimpleNamespace(accumulate_grad_batches=accum_steps)
        self.events = []
        self.mock = mock
        self.state = {
            "loss": torch.tensor(6.0, requires_grad=True),
            "callback_loss": torch.tensor(2.0),
        }

    def __call__(self, batch, *, stage):
        self.events.append(("forward", stage, batch["batch_idx"]))
        return self.state

    def _manual_optimization_handles(self):
        self.events.append(("handles",))
        return ([], True) if self.mock else (["optimizer"], False)

    def manual_backward(self, loss):
        self.events.append(("backward", float(loss.detach())))

    def after_manual_backward(self):
        self.events.append(("after_backward",))

    def _step_manual_optimizers(self, optimizers, batch_idx, accum_steps):
        self.events.append(("step", optimizers, batch_idx, accum_steps))


@pytest.mark.unit
def test_standard_policy_preserves_accumulation_and_callback_state():
    policy = StandardOptimizationPolicy()
    module = _StandardHarness(accum_steps=2)
    first_batch = {}

    first = policy.training_step(module, first_batch, 0)

    assert first is module.state
    assert first_batch["batch_idx"] == 0
    assert module.events == [
        ("forward", "fit", 0),
        ("handles",),
        ("backward", 3.0),
        ("after_backward",),
    ]
    assert first["callback_loss"].item() == 2.0

    module.events.clear()
    second = policy.training_step(module, {}, 1)
    assert second is module.state
    assert module.events == [
        ("forward", "fit", 1),
        ("handles",),
        ("backward", 3.0),
        ("after_backward",),
        ("step", ["optimizer"], 1, 2),
    ]


@pytest.mark.unit
def test_standard_policy_mock_optimizer_returns_without_backward():
    module = _StandardHarness(mock=True)
    result = StandardOptimizationPolicy().training_step(module, {}, 0)
    assert result is module.state
    assert module.events == [("forward", "fit", 0), ("handles",)]


@pytest.mark.unit
def test_standard_policy_rejects_non_dict_batch():
    with pytest.raises(ValueError, match="expected to be a dict"):
        StandardOptimizationPolicy().training_step(_StandardHarness(), [], 0)


class _RecordingOptimizer(LightningOptimizer):
    def __init__(self, parameter: nn.Parameter, name: str, events: list[tuple]):
        super().__init__(torch.optim.SGD([parameter], lr=0.1))
        self._name = name
        self._events = events

    def step(self, closure=None, **kwargs):
        self._events.append(("step", self._name))

    def zero_grad(self, set_to_none=False):
        self._events.append(("zero", self._name, set_to_none))


@pytest.mark.unit
def test_step_helper_preserves_frequency_clipping_scheduler_and_zero_order():
    events = []
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizers = [
        _RecordingOptimizer(parameter, "first", events),
        _RecordingOptimizer(parameter, "second", events),
    ]

    class Harness:
        _optimizer_index_to_name = {0: "first", 1: "second"}
        _optimizer_frequencies = {"first": 1, "second": 2}
        _optimizer_gradient_clip_val = {"first": 0.5, "second": None}
        _optimizer_gradient_clip_algorithm = {"first": "norm", "second": "norm"}
        trainer = SimpleNamespace(
            lr_scheduler_configs=[
                SimpleNamespace(interval="step", frequency=2),
                SimpleNamespace(interval="epoch", frequency=1),
            ]
        )

        def clip_gradients(self, optimizer, **kwargs):
            events.append(("clip", optimizer._name, kwargs))

        def _step_scheduler(self, scheduler_cfg):
            events.append(("scheduler", scheduler_cfg.frequency))

    Module._step_manual_optimizers(
        Harness(),
        optimizers,
        batch_idx=3,
        accum_steps=2,
    )
    assert events == [
        (
            "clip",
            "first",
            {"gradient_clip_val": 0.5, "gradient_clip_algorithm": "norm"},
        ),
        ("step", "first"),
        ("scheduler", 2),
        ("step", "second"),
        ("zero", "first", True),
        ("zero", "second", True),
    ]


class _Provider(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.calls = 0

    def create_optimization_policy(self):
        self.calls += 1
        return self.policy


class _ProviderOwner(nn.Module):
    def __init__(self, *providers):
        super().__init__()
        for index, provider in enumerate(providers):
            self.add_module(f"provider_{index}", provider)


@pytest.mark.unit
def test_resolver_precedence_validation_and_ambiguity():
    explicit = StandardOptimizationPolicy()
    provider = _Provider(GeoDROOptimizerStepPolicy())
    owner = _ProviderOwner(provider)
    assert resolve_optimization_policy(owner, explicit) is explicit
    assert provider.calls == 0

    selected = resolve_optimization_policy(owner)
    assert isinstance(selected, GeoDROOptimizerStepPolicy)
    assert provider.calls == 1

    with pytest.raises(ValueError, match="Multiple optimization policy providers"):
        resolve_optimization_policy(
            _ProviderOwner(
                _Provider(StandardOptimizationPolicy()),
                _Provider(StandardOptimizationPolicy()),
            )
        )
    with pytest.raises(TypeError, match="non-empty string name"):
        resolve_optimization_policy(owner, SimpleNamespace(training_step=lambda: None))


@pytest.mark.unit
@pytest.mark.parametrize(
    "loss_cls",
    [CoherentHardnessGeoDROLeJEPALoss, GraphTransportGeoDROJEPALoss],
)
def test_module_explicit_and_legacy_paths_delegate_identically(
    loss_cls,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_training_step(module, batch, batch_idx):
        calls.append((module, batch, batch_idx))
        return {"loss": torch.tensor(2.0), "batch_idx": batch_idx}

    monkeypatch.setattr(
        "stable_pretraining.geodro_lejepa.optimizer_step.optimizer_step_training_step",
        fake_training_step,
    )
    legacy = Module(
        geodro_lejepa_loss=loss_cls(adversary_scope=AdversaryScope.OPTIMIZER_STEP.value)
    )
    explicit = Module(
        geodro_lejepa_loss=loss_cls(
            adversary_scope=AdversaryScope.OPTIMIZER_STEP.value
        ),
        optimization_policy=GeoDROOptimizerStepPolicy(),
    )

    with pytest.warns(DeprecationWarning, match="Implicit optimizer-policy"):
        legacy_result = legacy.training_step({"sample": 1}, 3)
    explicit_result = explicit.training_step({"sample": 1}, 3)
    assert legacy_result.keys() == explicit_result.keys()
    assert torch.equal(legacy_result["loss"], explicit_result["loss"])
    assert legacy_result["batch_idx"] == explicit_result["batch_idx"] == 3
    assert len(calls) == 2

    legacy.training_step({"sample": 2}, 4)
    assert len(calls) == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    "loss_cls",
    [CoherentHardnessGeoDROLeJEPALoss, GraphTransportGeoDROJEPALoss],
)
def test_legacy_loss_provider_is_stateless_and_warns_once(loss_cls):
    loss_fn = loss_cls(adversary_scope=AdversaryScope.OPTIMIZER_STEP.value)
    keys_before = set(loss_fn.state_dict())
    with pytest.warns(DeprecationWarning, match="Implicit optimizer-policy"):
        first = loss_fn.create_optimization_policy()
    second = loss_fn.create_optimization_policy()
    assert isinstance(first, GeoDROOptimizerStepPolicy)
    assert isinstance(second, GeoDROOptimizerStepPolicy)
    assert set(loss_fn.state_dict()) == keys_before

    microbatch = loss_cls(adversary_scope=AdversaryScope.MICROBATCH.value)
    assert microbatch.create_optimization_policy() is None


@pytest.mark.unit
def test_policy_object_adds_no_module_checkpoint_keys():
    def forward(self, batch, stage):
        return {"loss": self.backbone(batch["x"]).sum()}

    torch.manual_seed(3)
    standard = Module(forward=forward, backbone=nn.Linear(2, 2))
    torch.manual_seed(3)
    explicit = Module(
        forward=forward,
        backbone=nn.Linear(2, 2),
        optimization_policy=GeoDROOptimizerStepPolicy(),
    )
    assert set(standard.state_dict()) == set(explicit.state_dict())


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_name",
    [
        "geodro/geodro_lejepa_imagenet100ctrl_v1_optstep_accum",
        "geodro/geodro_jepa_v2_coherent_hardness_batch_memory_optstep_imagenet100ctrl",
        "geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl",
    ],
)
def test_canonical_configs_compose_explicit_policy(config_name):
    with initialize_config_dir(version_base=None, config_dir=str(EXAMPLES_CONFIG_DIR)):
        cfg = compose(config_name=config_name)
    policy = instantiate(cfg.module.optimization_policy)
    assert isinstance(policy, GeoDROOptimizerStepPolicy)


@pytest.mark.unit
def test_generic_module_has_no_method_specific_dispatch():
    source = (Path(__file__).resolve().parents[2] / "module.py").read_text(
        encoding="utf-8"
    )
    lowered = source.lower().replace("_", "")
    assert "geodro" not in source.lower()
    assert "adversaryscope" not in lowered
