"""Method-agnostic manual-optimization policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .module import Module


class OptimizationPolicy(Protocol):
    """One manual-optimization training-step strategy."""

    name: str

    def training_step(
        self,
        module: Module,
        batch: Any,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Execute one training batch."""


class OptimizationPolicyProvider(Protocol):
    """Child-module compatibility hook for selecting a policy."""

    def create_optimization_policy(self) -> OptimizationPolicy | None:
        """Return a policy when this child owns specialized orchestration."""


class StandardOptimizationPolicy:
    """The default joint-loss manual-optimization behavior."""

    name = "standard"

    def training_step(
        self,
        module: Module,
        batch: Any,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Run the framework's standard manual-optimization step."""
        if type(batch) is not dict:
            msg = f"batch is expected to be a dict! Not as {type(batch)}"
            raise ValueError(msg)
        batch["batch_idx"] = batch_idx
        state = module(batch, stage="fit")

        optimizers, has_mock_optimizer = module._manual_optimization_handles()
        if has_mock_optimizer:
            return state

        accum_steps = max(
            int(
                getattr(
                    module.trainer,
                    "accumulate_grad_batches_",
                    getattr(module.trainer, "accumulate_grad_batches", 1),
                )
            ),
            1,
        )
        loss = state["loss"]
        if accum_steps > 1:
            loss = loss / accum_steps

        module.manual_backward(loss)
        module.after_manual_backward()

        if (batch_idx + 1) % accum_steps != 0:
            return state

        module._step_manual_optimizers(optimizers, batch_idx, accum_steps)
        return state


def validate_optimization_policy(policy: Any, *, source: str) -> OptimizationPolicy:
    """Validate the small structural policy contract."""
    if policy is None:
        raise TypeError(f"{source} returned no optimization policy.")
    if not isinstance(getattr(policy, "name", None), str) or not policy.name:
        raise TypeError(f"{source} policy must define a non-empty string name.")
    if not callable(getattr(policy, "training_step", None)):
        raise TypeError(f"{source} policy must define callable training_step().")
    return policy


def resolve_optimization_policy(
    module: Module,
    explicit_policy: Any = None,
) -> OptimizationPolicy:
    """Resolve explicit, unique provider, then standard policy."""
    if explicit_policy is not None:
        return validate_optimization_policy(explicit_policy, source="explicit")

    providers: list[tuple[str, OptimizationPolicy]] = []
    for child_name, child in module.named_modules():
        if child is module:
            continue
        factory = getattr(child, "create_optimization_policy", None)
        if not callable(factory):
            continue
        policy = factory()
        if policy is not None:
            providers.append(
                (
                    child_name or child.__class__.__name__,
                    validate_optimization_policy(
                        policy,
                        source=f"provider {child_name or child.__class__.__name__}",
                    ),
                )
            )
    if len(providers) > 1:
        names = ", ".join(name for name, _ in providers)
        raise ValueError(f"Multiple optimization policy providers matched: {names}.")
    if providers:
        return providers[0][1]
    return StandardOptimizationPolicy()
