import json
from pathlib import Path

import pytest

from stable_pretraining.callbacks.run_manifest import (
    RunManifestCallback,
    expected_signal_contract,
)


class _Config(dict):
    def update(self, other, allow_val_change=False):  # noqa: ARG002
        super().update(other)


class _Experiment:
    def __init__(self):
        self.config = _Config()


class _Logger:
    name = "test-run"
    version = "abc123"

    def __init__(self, metadata):
        self._wandb_init = {"config": metadata}
        self._experiment = _Experiment()

    @property
    def experiment(self):
        return self._experiment


class _Trainer:
    def __init__(self, root_dir: Path, metadata):
        self.default_root_dir = str(root_dir)
        self.logger = _Logger(metadata)
        self.is_global_zero = True
        self.max_epochs = 400
        self.global_step = 0
        self.current_epoch = 0
        self.num_nodes = 2
        self.num_devices = 4
        self.accumulate_grad_batches = 1
        self.accelerator = "cuda"
        self.strategy = "ddp"
        self.precision = "bf16-mixed"


@pytest.mark.unit
def test_run_manifest_writes_trace_and_updates_wandb_config(tmp_path, monkeypatch):
    import stable_pretraining.callbacks.run_manifest as manifest_module

    monkeypatch.setattr(manifest_module, "_hydra_output_dir", lambda: None)
    monkeypatch.setenv(
        "GEODRO_CONFIG_NAME", "geodro/geodro_lejepa_imagenet100ctrl_v1"
    )
    monkeypatch.setenv("GEODRO_GRAPH_BATCH", "2048")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "mcml-hgx-a100-80x4")
    monkeypatch.setenv("SLURM_JOB_QOS", "mcml")
    metadata = {
        "method": "geodro_v1_1",
        "dataset": "imagenet100ctrl",
        "model": "vits8",
    }
    trainer = _Trainer(tmp_path, metadata)
    callback = RunManifestCallback()

    callback.setup(trainer, object(), "fit")

    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "started"
    for key, value in metadata.items():
        assert manifest["metadata"][key] == value
    assert (
        manifest["metadata"]["config_name"]
        == "geodro/geodro_lejepa_imagenet100ctrl_v1"
    )
    assert manifest["runtime"]["env"]["GEODRO_GRAPH_BATCH"] == "2048"
    assert manifest["slurm"]["SLURM_JOB_PARTITION"] == "mcml-hgx-a100-80x4"
    assert "train/geodro/Weight_alpha" in manifest["expected_signals"]["geodro_v1"]
    assert trainer.logger.experiment.config["run_trace"]["manifest_path"] == str(
        manifest_path
    )


@pytest.mark.unit
def test_expected_signal_contract_separates_erm_and_geodro():
    erm = expected_signal_contract({"method": "lejepa_erm"})
    geodro = expected_signal_contract({"method": "geodro_v1_1"})

    assert "erm_control" in erm
    assert "geodro_v1" not in erm
    assert "geodro_v1" in geodro
    assert "train/pred_erm_loss" in geodro["geodro_v1"]


@pytest.mark.unit
def test_run_manifest_infers_metadata_from_geodro_config_env(tmp_path, monkeypatch):
    import stable_pretraining.callbacks.run_manifest as manifest_module

    monkeypatch.setattr(manifest_module, "_hydra_output_dir", lambda: None)
    monkeypatch.setenv(
        "GEODRO_CONFIG_NAME", "geodro/geodro_lejepa_imagenet100ctrl_erm"
    )
    trainer = _Trainer(tmp_path, {})
    callback = RunManifestCallback()

    callback.setup(trainer, object(), "fit")

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["metadata"]["method"] == "lejepa_erm"
    assert manifest["metadata"]["dataset"] == "imagenet100ctrl"
    assert manifest["metadata"]["model"] == "vits8"
