"""Run manifest callback for experiment traceability."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as pl
from lightning.pytorch.callbacks import Callback
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf


class RunManifestCallback(Callback):
    """Write a compact local run manifest and mirror trace metadata to W&B."""

    def __init__(self, filename: str = "run_manifest.json") -> None:
        super().__init__()
        self.filename = filename
        self._manifest_path: Path | None = None

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        if stage != "fit" or not _is_rank_zero(trainer):
            return
        self._write_manifest(trainer, status="started")

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not _is_rank_zero(trainer):
            return
        self._write_manifest(trainer, status="completed")

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        if not _is_rank_zero(trainer):
            return
        self._write_manifest(trainer, status="failed", exception=exception)

    def _write_manifest(
        self,
        trainer: pl.Trainer,
        *,
        status: str,
        exception: BaseException | None = None,
    ) -> None:
        output_dir = _hydra_output_dir() or Path(trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = output_dir / self.filename

        metadata = _metadata_with_env_fallback(_logger_metadata(trainer.logger))
        manifest = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "expected_signals": expected_signal_contract(metadata),
            "trainer": _trainer_trace(trainer),
            "slurm": _slurm_trace(),
            "runtime": _runtime_trace(output_dir),
            "git": _git_trace(),
        }
        if exception is not None:
            manifest["exception"] = {
                "type": exception.__class__.__name__,
                "message": str(exception),
            }

        with self._manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")

        _update_wandb_config(trainer.logger, manifest, self._manifest_path)
        logger.info(f"Run manifest written to {self._manifest_path}")


def expected_signal_contract(metadata: dict[str, Any]) -> dict[str, list[str]]:
    """Return the metrics expected for the run method."""
    common = [
        "train/loss",
        "train/pred_loss",
        "train/sigreg_loss",
        "eval/linear_probe_MulticlassAccuracy",
        "eval/knn_probe_MulticlassAccuracy",
    ]
    method = str(metadata.get("method", ""))
    if method.startswith("geodro"):
        return {
            "common": common,
            "geodro_v1": [
                "train/pred_erm_loss",
                "train/geodro/Weight_alpha",
                "train/geodro/Weight_warmup_multiplier",
                "train/geodro/Weight_fallback",
                "train/geodro/Weight_ess_ratio",
                "train/geodro/Weight_entropy",
                "train/geodro/Weight_max_p",
                "train/geodro/Weight_min_p",
                "train/geodro/Graph_num_nodes",
                "train/geodro/Graph_num_edges",
                "train/geodro/Graph_num_components",
                "train/geodro/Graph_singleton_fraction",
                "train/geodro/Utility_utility_mean",
                "train/geodro/Utility_view_disp_mean",
                "train/geodro/Utility_view_reliability_mean",
                "train/geodro/Flow_clamp_activation_ratio",
                "train/geodro/Flow_nan_or_inf_seen",
                "train/geodro/Controlled_coherent_fraction",
                "train/geodro/Controlled_coherent_mass",
                "train/geodro/Controlled_coherent_mass_lift",
                "train/geodro/Controlled_isolated_fraction",
                "train/geodro/Controlled_isolated_mass",
                "train/geodro/Controlled_isolated_mass_lift",
            ],
        }
    return {"common": common, "erm_control": ["train/pred_loss"]}


def _metadata_with_env_fallback(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)
    config_name = os.environ.get("GEODRO_CONFIG_NAME")
    if not config_name:
        return metadata

    metadata.setdefault("config_name", config_name)
    config_base = config_name.rsplit("/", 1)[-1]
    if config_base.startswith("geodro_lejepa_imagenet100ctrl_"):
        variant = config_base.removeprefix("geodro_lejepa_imagenet100ctrl_")
        metadata.setdefault("dataset", "imagenet100ctrl")
        metadata.setdefault("model", "vits8")
        if variant == "erm" or variant.startswith("erm_"):
            metadata.setdefault("method", "lejepa_erm")
            metadata.setdefault("run_role", f"main_{variant}")
        elif variant == "v1" or variant.startswith("v1_"):
            metadata.setdefault("method", "geodro_v1_1")
            metadata.setdefault("run_role", f"main_{variant}")
        else:
            metadata.setdefault("method", f"geodro_v1_1_{variant}")
            metadata.setdefault("run_role", f"main_{variant}")
    return metadata


def _is_rank_zero(trainer: pl.Trainer) -> bool:
    return bool(
        getattr(trainer, "is_global_zero", getattr(trainer, "global_rank", 0) == 0)
    )


def _hydra_output_dir() -> Path | None:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return None
    return None


def _logger_metadata(logger_obj: Any) -> dict[str, Any]:
    if logger_obj is None or logger_obj is False:
        return {}

    metadata: Any = None
    wandb_init = getattr(logger_obj, "_wandb_init", None)
    if isinstance(wandb_init, dict):
        metadata = wandb_init.get("config")

    if metadata is None:
        experiment = getattr(logger_obj, "_experiment", None)
        config = getattr(experiment, "config", None)
        if config is not None:
            try:
                metadata = dict(config)
            except (TypeError, ValueError):
                metadata = None

    return _jsonable(metadata) if isinstance(metadata, dict | DictConfig) else {}


def _trainer_trace(trainer: pl.Trainer) -> dict[str, Any]:
    return {
        "max_epochs": getattr(trainer, "max_epochs", None),
        "global_step": getattr(trainer, "global_step", None),
        "current_epoch": getattr(trainer, "current_epoch", None),
        "num_nodes": getattr(trainer, "num_nodes", None),
        "num_devices": getattr(trainer, "num_devices", None),
        "accelerator": str(getattr(trainer, "accelerator", "")),
        "strategy": str(getattr(trainer, "strategy", "")),
        "precision": str(getattr(trainer, "precision", "")),
        "accumulate_grad_batches": getattr(trainer, "accumulate_grad_batches", None),
        "default_root_dir": str(getattr(trainer, "default_root_dir", "")),
        "logger_name": getattr(getattr(trainer, "logger", None), "name", None),
        "logger_version": getattr(getattr(trainer, "logger", None), "version", None),
    }


def _slurm_trace() -> dict[str, str | None]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_NODELIST",
        "SLURM_SUBMIT_DIR",
    ]
    return {key: os.environ.get(key) for key in keys}


def _runtime_trace(output_dir: Path) -> dict[str, Any]:
    keys = [
        "GEODRO_CONFIG_NAME",
        "GEODRO_RUN_ID",
        "GEODRO_RUN_LABEL",
        "GEODRO_BATCH_SIZE",
        "GEODRO_VAL_BATCH_SIZE",
        "GEODRO_MAX_EPOCHS",
        "GEODRO_ACCUM_GRAD_BATCHES",
        "GEODRO_GRAPH_BATCH",
        "GEODRO_ADVERSARY_SCOPE",
        "GEODRO_MEMORY_UPDATE_SCOPE",
        "GEODRO_PREFLIGHT_ENABLED",
        "GEODRO_PREFLIGHT_STEPS",
        "GEODRO_K_OVERRIDE",
        "GEODRO_INNER_STEPS",
        "GEODRO_TAU_FLOW",
        "GEODRO_BETA",
        "GEODRO_ALPHA_MAX",
        "GEODRO_CLAMP_ACTIVATION_FAIL",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "WANDB_MODE",
    ]
    return {
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "env": {key: os.environ.get(key) for key in keys},
    }


def _git_trace() -> dict[str, Any]:
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git(["status", "--short"]),
    }


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _update_wandb_config(
    logger_obj: Any,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    if logger_obj is None or logger_obj is False:
        return

    experiment = getattr(logger_obj, "experiment", None)
    config = getattr(experiment, "config", None)
    if config is None or not hasattr(config, "update"):
        return

    trace = {
        "manifest_path": str(manifest_path),
        "status": manifest["status"],
        "slurm": manifest["slurm"],
        "runtime": manifest["runtime"],
        "expected_signals": manifest["expected_signals"],
    }
    try:
        config.update({"run_trace": _jsonable(trace)}, allow_val_change=True)
    except TypeError:
        config.update({"run_trace": _jsonable(trace)})
    except Exception as exc:
        logger.warning(f"Could not update W&B run_trace config: {exc}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
