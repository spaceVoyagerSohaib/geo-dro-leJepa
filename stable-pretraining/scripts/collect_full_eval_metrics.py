#!/usr/bin/env python3
"""Collect and compare downstream eval metrics for completed 400-epoch pretrain runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
RESEARCH_DIR = PROJECT_ROOT.parent / "research" / "curated_results"
JSON_PATH = RESEARCH_DIR / "full_pretrain_eval_matrix.json"
EVAL_DOC_PATH = RESEARCH_DIR / "geodro_lejepa_eval_results.md"
AUTO_START = "<!-- AUTO:full-eval-matrix:START -->"
AUTO_END = "<!-- AUTO:full-eval-matrix:END -->"

REQUIRED_MODES = (
    "imagenet100ctrl",
    "imagenet100c",
    "waterbirds",
    "imagenet_sketch",
    "imagenet_r",
    "imagenet_a",
    "imagenet_o",
    "celeba",
    "camelyon17",
)

# Slurm log name fragments -> eval mode.
LOG_MODE_HINTS: dict[str, str] = {
    "im100ctrl": "imagenet100ctrl",
    "im100c": "imagenet100c",
    "waterbirds": "waterbirds",
    "imsketch": "imagenet_sketch",
    "imr": "imagenet_r",
    "ima": "imagenet_a",
    "imo": "imagenet_o",
    "celeba": "celeba",
    "cam17": "camelyon17",
}

PENDING_STATES = frozenset({"PENDING", "CONFIGURING"})
RUNNING_STATES = frozenset({"RUNNING", "COMPLETING"})
FAILED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)

Direction = Literal["higher", "lower"]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PRETRAIN_METRIC_PATTERNS = {
    "weight_alpha_epoch": re.compile(
        r"train/geodro/Weight_alpha_epoch\s*\|\s*([0-9.eE+-]+)"
    ),
    "weight_fallback_epoch": re.compile(
        r"train/geodro/Weight_fallback_epoch\s*\|\s*([0-9.eE+-]+)"
    ),
    "pred_loss_minus_erm_epoch": re.compile(
        r"train/pred_loss_minus_erm_epoch\s*\|\s*([0-9.eE+-]+)"
    ),
    "linear_probe_epoch": re.compile(
        r"eval/linear_probe_MulticlassAccuracy_epoch\s*\|\s*([0-9.eE+-]+)"
    ),
    "linear_probe_best": re.compile(
        r"eval/linear_probe_MulticlassAccuracy['\"].*reached\s+([0-9.]+)\s+\(best"
    ),
    "knn_probe_epoch": re.compile(
        r"eval/knn_probe_MulticlassAccuracy_epoch\s*\|\s*([0-9.eE+-]+)"
    ),
}

# Headline comparison columns (key in flat metrics dict, human label, direction).
HEADLINE_COLUMNS: list[tuple[str, str, Direction, str]] = [
    ("imagenet100ctrl/val/top1_acc", "IN100ctrl top1", "higher", "ID"),
    ("imagenet100ctrl/val/top5_acc", "IN100ctrl top5", "higher", "ID"),
    ("imagenet100ctrl/val/worst_subset_acc", "IN100ctrl worst-subset", "higher", "ID"),
    ("imagenet100ctrl/val/knn_top1_acc", "IN100ctrl kNN", "higher", "ID"),
    ("imagenet100ctrl/val/ece", "IN100ctrl ECE", "lower", "ID"),
    ("imagenet100c/mCE", "IN100C mCE", "lower", "corruption"),
    ("imagenet100c/clean_vs_corrupted_gap", "IN100C clean-corr gap", "lower", "corruption"),
    ("imagenet_sketch/val/top1_acc", "IN-Sketch top1", "higher", "shift"),
    ("imagenet_sketch/clean_vs_shifted_gap", "IN-Sketch gap", "lower", "shift"),
    ("imagenet_r/val/top1_acc", "IN-R top1", "higher", "shift"),
    ("imagenet_r/clean_vs_shifted_gap", "IN-R gap", "lower", "shift"),
    ("imagenet_a/val/top1_acc", "IN-A top1", "higher", "shift"),
    ("imagenet_a/clean_vs_shifted_gap", "IN-A gap", "lower", "shift"),
    ("imagenet_o/auroc_max_softmax", "IN-O AUROC", "higher", "OOD"),
    ("waterbirds/test/worst_group_acc", "Waterbirds test worst-grp", "higher", "subpop"),
    ("celeba/test/worst_group_acc", "CelebA test worst-grp", "higher", "subpop"),
    ("camelyon17/ood_val/acc", "Camelyon17 OOD-val", "higher", "domain"),
    ("camelyon17/ood_test/acc", "Camelyon17 OOD-test", "higher", "domain"),
]

ERM_BASELINE_RUN_ID = "erm-full-b128-a4-e400-cont-5663763"

# Completed runs kept for archival/diagnostics but omitted from comparison tables.
EXCLUDED_FROM_COMPARISON_RUN_IDS: frozenset[str] = frozenset(
    {
        # Strict-fallback GeoDRO (alpha=0, fallback=1); not an ERM-equivalent comparator.
        "geo-dro-v2-full-k16-bs128-is20-ac4-5649375",
    }
)
EXCLUDED_FROM_COMPARISON_REASONS: dict[str, str] = {
    "geo-dro-v2-full-k16-bs128-is20-ac4-5649375": (
        "strict-fallback GeoDRO (alpha=0, fallback=1); excluded from comparisons"
    ),
}


def is_comparison_eligible(run_id: str) -> bool:
    return run_id not in EXCLUDED_FROM_COMPARISON_RUN_IDS

# Family-grouped table specs for the canonical eval doc (title, columns).
FAMILY_TABLES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "In-distribution (ImageNet-100 controlled)",
        [
            ("imagenet100ctrl/val/top1_acc", "top1"),
            ("imagenet100ctrl/val/top5_acc", "top5"),
            ("imagenet100ctrl/val/worst_subset_acc", "worst-subset"),
            ("imagenet100ctrl/val/knn_top1_acc", "kNN"),
            ("imagenet100ctrl/val/ece", "ECE"),
        ],
    ),
    (
        "Corruption (ImageNet-100-C)",
        [
            ("imagenet100c/mCE", "mCE"),
            ("imagenet100c/clean_vs_corrupted_gap", "clean-corr gap"),
        ],
    ),
    (
        "Covariate shift (IN-Sketch / IN-R / IN-A)",
        [
            ("imagenet_sketch/val/top1_acc", "sketch top1"),
            ("imagenet_sketch/clean_vs_shifted_gap", "sketch gap"),
            ("imagenet_r/val/top1_acc", "IN-R top1"),
            ("imagenet_r/clean_vs_shifted_gap", "IN-R gap"),
            ("imagenet_a/val/top1_acc", "IN-A top1"),
            ("imagenet_a/clean_vs_shifted_gap", "IN-A gap"),
        ],
    ),
    (
        "OOD detection (ImageNet-O)",
        [("imagenet_o/auroc_max_softmax", "AUROC")],
    ),
    (
        "Subpopulation shift (Waterbirds, CelebA)",
        [
            ("waterbirds/test/worst_group_acc", "WB test worst-grp"),
            ("celeba/test/worst_group_acc", "CelebA test worst-grp"),
        ],
    ),
    (
        "Domain shift (WILDS Camelyon17)",
        [
            ("camelyon17/ood_val/acc", "OOD-val acc"),
            ("camelyon17/ood_test/acc", "OOD-test acc"),
        ],
    ),
]


@dataclass
class EvalRecord:
    """Collected status and metrics for one eval mode/job."""

    mode: str
    job_id: str
    metrics_path: str
    status: str
    schema_version: int | None = None
    method: str = ""
    checkpoint: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)


@dataclass
class PretrainRun:
    """Collected pretraining run metadata plus downstream eval records."""

    run_id: str
    job_id: str
    run_label: str
    method: str
    config_name: str
    status: str
    current_epoch: int | None
    max_epochs: int | None
    output_dir: str
    manifest_path: str
    checkpoint: str
    pretrain_metrics: dict[str, float] = field(default_factory=dict)
    evals: dict[str, EvalRecord] = field(default_factory=dict)
    eval_complete_count: int = 0
    missing_modes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def load_sacct() -> dict[str, str]:
    try:
        user = subprocess.check_output(["id", "-un"], text=True).strip()
        end_date = (date.today() + timedelta(days=1)).isoformat()
        out = subprocess.check_output(
            [
                "sacct",
                "-u",
                user,
                "--format=JobID,State",
                "-S",
                "2026-05-01",
                "-E",
                end_date,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    states: dict[str, str] = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and "." not in parts[0]:
            states[parts[0]] = parts[-1]
    return states


def scrape_pretrain_metrics(log_path: Path) -> dict[str, float]:
    if not log_path.is_file():
        return {}
    text = ANSI_RE.sub("", log_path.read_text(errors="replace"))
    tail = text[-800_000:] if len(text) > 800_000 else text
    out: dict[str, float] = {}
    for name, pattern in PRETRAIN_METRIC_PATTERNS.items():
        hits = pattern.findall(tail)
        if hits:
            try:
                out[name] = float(hits[-1])
            except ValueError:
                pass
    return out


def find_checkpoint(run_dir: Path) -> str:
    ckpt_dir = run_dir / "checkpoints"
    for name in ("best.ckpt", "last.ckpt"):
        candidate = ckpt_dir / name
        if candidate.is_file():
            return str(candidate.relative_to(PROJECT_ROOT))
    if ckpt_dir.is_dir():
        epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt"))
        if epoch_ckpts:
            return str(epoch_ckpts[-1].relative_to(PROJECT_ROOT))
    return ""


def parse_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_completed_400ep(data: dict[str, Any]) -> bool:
    trainer = data.get("trainer") or {}
    max_ep = trainer.get("max_epochs")
    cur_ep = trainer.get("current_epoch")
    status = data.get("status", "")
    return (
        max_ep == 400
        and cur_ep == 400
        and status == "completed"
    )


def is_400ep_attempt(data: dict[str, Any]) -> bool:
    trainer = data.get("trainer") or {}
    env = (data.get("runtime") or {}).get("env") or {}
    max_ep = trainer.get("max_epochs")
    if max_ep == 400:
        return True
    return str(env.get("GEODRO_MAX_EPOCHS", "")) == "400"


def extract_run_label(run_id: str, data: dict[str, Any]) -> str:
    env = (data.get("runtime") or {}).get("env") or {}
    if env.get("GEODRO_RUN_LABEL"):
        return str(env["GEODRO_RUN_LABEL"])
    # Strip trailing -jobid from run_id.
    m = re.match(r"^(.+)-(\d{7})$", run_id)
    return m.group(1) if m else run_id


def extract_job_id(run_id: str, data: dict[str, Any]) -> str:
    slurm = data.get("slurm") or {}
    if slurm.get("SLURM_JOB_ID"):
        return str(slurm["SLURM_JOB_ID"])
    m = re.search(r"-(\d{7})$", run_id)
    return m.group(1) if m else ""


def parse_eval_dir_name(dirname: str) -> tuple[str, str] | None:
    for mode in REQUIRED_MODES:
        prefix = f"{mode}-"
        if dirname.startswith(prefix):
            job_id = dirname[len(prefix) :]
            if job_id.isdigit():
                return mode, job_id
    return None


def load_eval_metrics(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def flatten_mode_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge mode metrics; normalize imagenet100c keys onto imagenet100c/ prefix."""
    metrics = dict(payload.get("metrics") or {})
    mode = payload.get("mode", "")
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        flat[key] = value
    if mode == "imagenet100c":
        for key, value in list(metrics.items()):
            if key.startswith("imagenetc/"):
                flat[f"imagenet100c/{key[len('imagenetc/'):]}"] = value
    return flat


def eval_status_without_metrics(
    job_id: str,
    sacct: dict[str, str],
    *,
    failed: bool = False,
) -> str:
    """Classify an eval record that has no metrics.json artifact."""
    sacct_state = sacct.get(job_id, "")
    state_base = sacct_state.split("+", maxsplit=1)[0]
    if state_base in PENDING_STATES:
        return "pending"
    if state_base in RUNNING_STATES:
        return "running"
    if state_base in FAILED_STATES or failed:
        return "failed"
    if state_base == "COMPLETED":
        return "missing_metrics"
    return "unknown_no_metrics"


def collect_evals_for_run(
    run_dir: Path,
    run_id: str,
    sacct: dict[str, str],
    eval_log_index: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, EvalRecord]:
    eval_root = run_dir / "eval"
    by_mode: dict[str, list[EvalRecord]] = {m: [] for m in REQUIRED_MODES}

    if eval_root.is_dir():
        for eval_dir in sorted(eval_root.iterdir()):
            if not eval_dir.is_dir():
                continue
            parsed = parse_eval_dir_name(eval_dir.name)
            if not parsed:
                continue
            mode, job_id = parsed
            metrics_path = eval_dir / "metrics.json"
            args_path = eval_dir / "args.json"
            if metrics_path.is_file():
                payload = load_eval_metrics(metrics_path)
                if payload is None:
                    rec = EvalRecord(
                        mode=mode,
                        job_id=job_id,
                        metrics_path=str(metrics_path),
                        status="corrupt_json",
                    )
                else:
                    rec = EvalRecord(
                        mode=mode,
                        job_id=job_id,
                        metrics_path=str(metrics_path),
                        status="completed",
                        schema_version=payload.get("schema_version"),
                        method=str(payload.get("method", "")),
                        checkpoint=str(payload.get("checkpoint", "")),
                        metrics=flatten_mode_metrics(payload),
                    )
            elif args_path.is_file():
                rec = EvalRecord(
                    mode=mode,
                    job_id=job_id,
                    metrics_path="",
                    status=eval_status_without_metrics(job_id, sacct),
                )
            else:
                continue
            by_mode[mode].append(rec)

    run_logs = eval_log_index.get(run_id, {})

    # Augment from eval logs referencing this pretrain run.
    for mode, entries in run_logs.items():
        for entry in entries:
            job_id = entry["job_id"]
            if any(r.job_id == job_id for r in by_mode.get(mode, [])):
                continue
            if entry.get("metrics_path"):
                payload = load_eval_metrics(Path(entry["metrics_path"]))
                if payload:
                    by_mode[mode].append(
                        EvalRecord(
                            mode=mode,
                            job_id=job_id,
                            metrics_path=entry["metrics_path"],
                            status="completed",
                            schema_version=payload.get("schema_version"),
                            method=str(payload.get("method", "")),
                            checkpoint=str(payload.get("checkpoint", "")),
                            metrics=flatten_mode_metrics(payload),
                        )
                    )
                    continue
            by_mode[mode].append(
                EvalRecord(
                    mode=mode,
                    job_id=job_id,
                    metrics_path=entry.get("metrics_path", ""),
                    status=eval_status_without_metrics(
                        job_id,
                        sacct,
                        failed=bool(entry.get("failed")),
                    ),
                )
            )

    chosen: dict[str, EvalRecord] = {}
    for mode in REQUIRED_MODES:
        records = by_mode.get(mode) or []
        if not records:
            continue
        completed = [r for r in records if r.status == "completed"]
        pool = completed if completed else records
        pool.sort(key=lambda r: int(r.job_id), reverse=True)
        best = pool[0]
        best.provenance = [r.metrics_path or f"eval/{mode}-{r.job_id}" for r in records]
        chosen[mode] = best
    return chosen


def build_eval_log_index() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Map pretrain_run_id -> mode -> list of eval log entries."""
    index: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if not LOGS_DIR.is_dir():
        return index

    for path in sorted(LOGS_DIR.glob("slurm-geodro-*-eval-*.out")):
        mode = ""
        name = path.name.lower()
        for hint, mode_name in LOG_MODE_HINTS.items():
            if f"geodro-{hint}-eval" in name or f"geodro-{hint.replace('_', '')}-eval" in name:
                mode = mode_name
                break
        if not mode:
            continue
        job_m = re.search(r"-(\d{7})\.out$", path.name)
        job_id = job_m.group(1) if job_m else ""
        text = path.read_text(errors="replace")
        pretrain_run = ""
        for pat in (
            r"Pretrain run(?: ID)?:\s*(\S+)",
            r"PRETRAIN_RUN_ID[=:\s]+(\S+)",
            r"Run ID:\s*(\S+)",
        ):
            m = re.search(pat, text)
            if m:
                pretrain_run = m.group(1).strip()
                break
        metrics_path = ""
        m = re.search(r"Wrote metrics to (\S+)", text)
        if m:
            metrics_path = m.group(1)
        failed = "ERROR" in text and not metrics_path
        if not pretrain_run and metrics_path:
            payload = load_eval_metrics(Path(metrics_path))
            if payload:
                pretrain_run = str(payload.get("pretrain_run_id", ""))
        if not pretrain_run:
            continue
        index.setdefault(pretrain_run, {}).setdefault(mode, []).append(
            {
                "job_id": job_id,
                "log_path": str(path),
                "metrics_path": metrics_path,
                "failed": failed,
            }
        )
    return index


def discover_runs(sacct: dict[str, str]) -> tuple[list[PretrainRun], list[PretrainRun]]:
    eval_log_index = build_eval_log_index()
    completed: list[PretrainRun] = []
    incomplete: list[PretrainRun] = []

    for manifest_path in sorted(OUTPUTS_DIR.glob("*/run_manifest.json")):
        if "/preflight/" in str(manifest_path):
            continue
        data = parse_manifest(manifest_path)
        if not data or not is_400ep_attempt(data):
            continue

        run_dir = manifest_path.parent
        run_id = run_dir.name
        trainer = data.get("trainer") or {}
        metadata = data.get("metadata") or {}

        run = PretrainRun(
            run_id=run_id,
            job_id=extract_job_id(run_id, data),
            run_label=extract_run_label(run_id, data),
            method=str(metadata.get("method", "")),
            config_name=str(metadata.get("config_name", "")),
            status=str(data.get("status", "unknown")),
            current_epoch=trainer.get("current_epoch"),
            max_epochs=trainer.get("max_epochs"),
            output_dir=str(run_dir),
            manifest_path=str(manifest_path),
            checkpoint=find_checkpoint(run_dir),
            pretrain_metrics=scrape_pretrain_metrics(run_dir / "logs" / "pretrain.log"),
            evals=collect_evals_for_run(run_dir, run_id, sacct, eval_log_index),
        )
        run.eval_complete_count = sum(
            1 for m in REQUIRED_MODES if run.evals.get(m) and run.evals[m].status == "completed"
        )
        run.missing_modes = [
            m
            for m in REQUIRED_MODES
            if m not in run.evals or run.evals[m].status != "completed"
        ]
        if run.missing_modes:
            status_to_modes: dict[str, list[str]] = {}
            for mode in run.missing_modes:
                rec = run.evals.get(mode)
                if rec:
                    status_to_modes.setdefault(rec.status, []).append(mode)
            pending = status_to_modes.get("pending", []) + status_to_modes.get(
                "running", []
            )
            if pending:
                run.notes.append(f"eval pending/running: {', '.join(pending)}")
            unknown = status_to_modes.get("unknown_no_metrics", [])
            if unknown:
                run.notes.append(
                    f"eval status unknown/no metrics: {', '.join(unknown)}"
                )
            failed = status_to_modes.get("failed", [])
            if failed:
                run.notes.append(f"eval failed/no metrics: {', '.join(failed)}")
            missing_metrics = status_to_modes.get("missing_metrics", [])
            if missing_metrics:
                run.notes.append(
                    f"eval completed but metrics missing: {', '.join(missing_metrics)}"
                )

        if is_completed_400ep(data):
            completed.append(run)
        else:
            incomplete.append(run)

    completed.sort(key=lambda r: r.run_id)
    incomplete.sort(key=lambda r: r.run_id)
    return completed, incomplete


def metric_get(flat: dict[str, Any], key: str) -> float | None:
    val = flat.get(key)
    if val is None and key.startswith("imagenet100c/"):
        val = flat.get("imagenetc/" + key.split("/", 1)[1])
    if val is None and key == "imagenet100ctrl/val/worst_subset_acc":
        val = flat.get("imagenet100ctrl/val/worst_group_acc")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def build_headline_row(run: PretrainRun) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "run_label": run.run_label,
        "method": run.method,
        "eval_complete": f"{run.eval_complete_count}/9",
        "eval_complete_count": run.eval_complete_count,
        "paper_ready": run.eval_complete_count == 9,
        "comparison_eligible": is_comparison_eligible(run.run_id),
    }
    merged: dict[str, Any] = {}
    for mode, rec in run.evals.items():
        if rec.status == "completed":
            merged.update(rec.metrics)
    for key, label, direction, family in HEADLINE_COLUMNS:
        val = metric_get(merged, key)
        row[key] = val
        row[f"{key}__label"] = label
        row[f"{key}__direction"] = direction
        row[f"{key}__family"] = family
    return row


def compute_deltas(
    rows: list[dict[str, Any]], baseline_run_id: str = ERM_BASELINE_RUN_ID
) -> dict[str, dict[str, float | None]]:
    baseline = next((r for r in rows if r["run_id"] == baseline_run_id), None)
    if not baseline:
        return {}
    deltas: dict[str, dict[str, float | None]] = {}
    for row in rows:
        if row["run_id"] == baseline_run_id:
            continue
        run_deltas: dict[str, float | None] = {}
        for key, _, direction, _ in HEADLINE_COLUMNS:
            b = baseline.get(key)
            v = row.get(key)
            if b is None or v is None:
                run_deltas[key] = None
            elif direction == "higher":
                run_deltas[key] = float(v) - float(b)
            else:
                run_deltas[key] = float(b) - float(v)  # positive = improvement
        deltas[row["run_id"]] = run_deltas
    return deltas


def interpret_run(row: dict[str, Any], deltas: dict[str, float | None] | None) -> list[str]:
    notes: list[str] = []
    if row["eval_complete_count"] == 9:
        notes.append("eval suite complete (9/9)")
    elif row["eval_complete_count"] == 0:
        notes.append("no completed eval modes yet")
    else:
        notes.append(f"partial eval ({row['eval_complete']})")

    if deltas:
        wb = deltas.get("waterbirds/test/worst_group_acc")
        if wb is not None:
            if wb > 0.01:
                notes.append(f"Waterbirds worst-group +{wb:.3f} vs ERM-cont")
            elif wb < -0.01:
                notes.append(f"Waterbirds worst-group {wb:.3f} vs ERM-cont")
        mce = deltas.get("imagenet100c/mCE")
        if mce is not None:
            if mce > 0.01:
                notes.append(f"IN100C mCE improved by {mce:.3f} vs ERM-cont (lower is better)")
            elif mce < -0.01:
                notes.append(f"IN100C mCE worse by {abs(mce):.3f} vs ERM-cont")
    return notes


def find_best_per_column(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    bests: dict[str, str | None] = {}
    eligible = [r for r in rows if r.get("comparison_eligible", True)]
    for key, _, direction, _ in HEADLINE_COLUMNS:
        candidates = [
            (r["run_id"], r.get(key)) for r in eligible if r.get(key) is not None
        ]
        if not candidates:
            bests[key] = None
            continue
        if direction == "higher":
            bests[key] = max(candidates, key=lambda x: x[1])[0]
        else:
            bests[key] = min(candidates, key=lambda x: x[1])[0]
    return bests


def fmt_val(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def fmt_delta(val: float | None) -> str:
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.4f}"


def short_run_id(run_id: str) -> str:
    m = re.search(r"-(\d{7})$", run_id)
    return m.group(1) if m else run_id


def render_family_table(
    title: str,
    columns: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    deltas: dict[str, dict[str, float | None]],
    *,
    include_delta: bool = True,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"### {title}")
    lines.append("")
    header_cols = ["Run"]
    for _, label in columns:
        header_cols.append(label)
        if include_delta:
            header_cols.append(f"Δ {label}")
    lines.append("| " + " | ".join(header_cols) + " |")
    align = ["---"] + ["---:"] * (len(header_cols) - 1)
    lines.append("| " + " | ".join(align) + " |")
    for row in rows:
        cells = [f"`{short_run_id(row['run_id'])}`"]
        run_deltas = deltas.get(row["run_id"], {})
        for key, _ in columns:
            cells.append(fmt_val(row.get(key)))
            if include_delta:
                if row["run_id"] == ERM_BASELINE_RUN_ID:
                    cells.append("—")
                else:
                    cells.append(fmt_delta(run_deltas.get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_auto_block(payload: dict[str, Any]) -> str:
    """Render the injectable auto-generated section for geodro_lejepa_eval_results.md."""
    lines: list[str] = []
    lines.append(f"_Auto-generated: {payload['generated_at']}_")
    lines.append("")
    summary = payload["summary"]
    lines.append(
        f"**Snapshot:** {summary['completed_400ep_count']} completed 400-epoch runs · "
        f"{summary['comparison_eligible_paper_ready_count']} comparison-eligible paper-ready "
        f"({summary['eval_complete_9_of_9']} total 9/9) · "
        f"{summary['eval_partial']} partial · {summary['eval_none']} no eval yet"
    )
    lines.append("")

    lines.append("## Run roster (400-epoch completed)")
    lines.append("")
    lines.append("| Run | Paper-ready | Eval | Method | Notes |")
    lines.append("|-----|:-----------:|------|--------|-------|")
    for run in payload["completed_runs"]:
        paper_ready = "yes" if run.get("paper_ready") else "no"
        notes = "; ".join(run.get("notes") or []) or "—"
        lines.append(
            f"| `{run['run_id']}` | {paper_ready} | {run['eval_complete_count']}/9 | "
            f"{run.get('method', '—')} | {notes} |"
        )
    lines.append("")

    lines.append("## Eval completeness (9 modes)")
    lines.append("")
    header = "| Run | " + " | ".join(REQUIRED_MODES) + " |"
    sep = "|-----|" + "|".join(["---"] * len(REQUIRED_MODES)) + "|"
    lines.append(header)
    lines.append(sep)
    for run in payload["completed_runs"]:
        cells = []
        for mode in REQUIRED_MODES:
            ev = (run.get("evals") or {}).get(mode)
            if not ev:
                cells.append("—")
            elif ev.get("status") == "completed":
                cells.append("ok")
            else:
                cells.append(ev.get("status", "?"))
        lines.append(f"| `{short_run_id(run['run_id'])}` | " + " | ".join(cells) + " |")
    lines.append("")

    headline_rows = payload["headline_rows"]
    deltas = payload["delta_vs_erm"]
    paper_ready_rows = [
        r
        for r in headline_rows
        if r.get("paper_ready") and r.get("comparison_eligible", True)
    ]
    excluded_comparison = [
        r for r in headline_rows if not r.get("comparison_eligible", True)
    ]

    lines.append("## Paper-ready comparison (9/9 eval)")
    lines.append("")
    if not paper_ready_rows:
        lines.append("_No runs with a complete 9-mode eval suite yet._")
        lines.append("")
    else:
        lines.append(
            f"Δ columns are vs true ERM `{payload['erm_baseline_run_id']}` "
            "(positive = improvement; lower-is-better metrics use ERM−run)."
        )
        lines.append("")
        for title, columns in FAMILY_TABLES:
            lines.extend(
                render_family_table(title, columns, paper_ready_rows, deltas)
            )

    if excluded_comparison:
        lines.append("## Excluded from comparison (archival only)")
        lines.append("")
        lines.append(
            "These completed runs are kept in the matrix for diagnostics but are "
            "not included in paper-ready comparison tables."
        )
        lines.append("")
        lines.append("| Run | Eval | Reason |")
        lines.append("|-----|------|--------|")
        for row in excluded_comparison:
            reason = EXCLUDED_FROM_COMPARISON_REASONS.get(
                row["run_id"], "excluded from comparison"
            )
            lines.append(
                f"| `{row['run_id']}` | {row['eval_complete']} | {reason} |"
            )
        lines.append("")

    comparison_rows = [
        r for r in headline_rows if r.get("comparison_eligible", True)
    ]
    lines.append("## All comparison-eligible runs (includes partial eval)")
    lines.append("")
    for title, columns in FAMILY_TABLES:
        lines.extend(render_family_table(title, columns, comparison_rows, deltas))

    lines.append("## Per-run readout")
    lines.append("")
    lines.append("| Run | Eval | Readout |")
    lines.append("|-----|------|---------|")
    for item in payload["interpretations"]:
        lines.append(
            f"| `{short_run_id(item['run_id'])}` | {item['eval_complete']} | "
            f"{'; '.join(item['notes']) or '—'} |"
        )
    lines.append("")

    incomplete = payload.get("incomplete_400ep_runs") or []
    if incomplete:
        lines.append("## Incomplete 400-epoch attempts (excluded)")
        lines.append("")
        lines.append("| Run | Status | Epoch | Eval |")
        lines.append("|-----|--------|-------|------|")
        for run in incomplete:
            ep = run.get("current_epoch")
            max_ep = run.get("max_epochs")
            epoch_str = f"{ep}/{max_ep}" if ep is not None else "—"
            lines.append(
                f"| `{run['run_id']}` | {run.get('status', '?')} | {epoch_str} | "
                f"{run.get('eval_complete_count', 0)}/9 |"
            )
        lines.append("")

    lines.append("## Missing eval actions")
    lines.append("")
    missing_any = False
    for run in payload["completed_runs"]:
        missing = run.get("missing_modes") or []
        if not missing:
            continue
        missing_any = True
        ckpt = run.get("checkpoint") or "outputs/<run>/checkpoints/last.ckpt"
        lines.append(
            f"- **`{run['run_id']}`** ({len(missing)} missing: {', '.join(missing)}) — "
            f"`bash scripts/slurm/run_full_eval_suite.sh {ckpt} {run['run_id']} "
            f"{run.get('method', 'unknown')}`"
        )
    if not missing_any:
        lines.append("_All completed 400-epoch runs have a full 9-mode eval suite._")
    lines.append("")

    return "\n".join(lines)


def inject_auto_block(doc_path: Path, auto_content: str) -> None:
    if not doc_path.is_file():
        raise FileNotFoundError(
            f"Eval results doc not found: {doc_path}. Create it before running --update-doc."
        )
    text = doc_path.read_text()
    if AUTO_START not in text or AUTO_END not in text:
        raise ValueError(
            f"Missing {AUTO_START} / {AUTO_END} markers in {doc_path}"
        )
    before, rest = text.split(AUTO_START, maxsplit=1)
    _, after = rest.split(AUTO_END, maxsplit=1)
    updated = (
        before.rstrip()
        + "\n\n"
        + AUTO_START
        + "\n\n"
        + auto_content.rstrip()
        + "\n\n"
        + AUTO_END
        + after
    )
    doc_path.write_text(updated)


def run_to_dict(run: PretrainRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "job_id": run.job_id,
        "run_label": run.run_label,
        "method": run.method,
        "config_name": run.config_name,
        "status": run.status,
        "current_epoch": run.current_epoch,
        "max_epochs": run.max_epochs,
        "output_dir": run.output_dir,
        "manifest_path": run.manifest_path,
        "checkpoint": run.checkpoint,
        "pretrain_metrics": run.pretrain_metrics,
        "eval_complete_count": run.eval_complete_count,
        "paper_ready": run.eval_complete_count == 9,
        "comparison_eligible": is_comparison_eligible(run.run_id),
        "comparison_excluded_reason": EXCLUDED_FROM_COMPARISON_REASONS.get(
            run.run_id
        ),
        "missing_modes": run.missing_modes,
        "notes": run.notes,
        "evals": {
            mode: {
                "mode": rec.mode,
                "job_id": rec.job_id,
                "status": rec.status,
                "metrics_path": rec.metrics_path,
                "schema_version": rec.schema_version,
                "method": rec.method,
                "checkpoint": rec.checkpoint,
                "metrics": rec.metrics if rec.status == "completed" else {},
                "provenance": rec.provenance,
            }
            for mode, rec in run.evals.items()
        },
    }


def collect() -> dict[str, Any]:
    sacct = load_sacct()
    completed, incomplete = discover_runs(sacct)
    headline_rows = [build_headline_row(r) for r in completed]
    deltas_by_run = compute_deltas(headline_rows)
    bests = find_best_per_column(headline_rows)

    interpretations = []
    for row in headline_rows:
        interpretations.append(
            {
                "run_id": row["run_id"],
                "eval_complete": row["eval_complete"],
                "notes": interpret_run(row, deltas_by_run.get(row["run_id"])),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "erm_baseline_run_id": ERM_BASELINE_RUN_ID,
        "required_modes": list(REQUIRED_MODES),
        "headline_columns": [
            {"key": k, "label": lbl, "direction": d, "family": f}
            for k, lbl, d, f in HEADLINE_COLUMNS
        ],
        "completed_runs": [run_to_dict(r) for r in completed],
        "incomplete_400ep_runs": [run_to_dict(r) for r in incomplete],
        "headline_rows": headline_rows,
        "delta_vs_erm": deltas_by_run,
        "best_run_per_column": bests,
        "interpretations": interpretations,
        "excluded_from_comparison_run_ids": sorted(EXCLUDED_FROM_COMPARISON_RUN_IDS),
        "summary": {
            "completed_400ep_count": len(completed),
            "eval_complete_9_of_9": sum(1 for r in completed if r.eval_complete_count == 9),
            "paper_ready_count": sum(1 for r in completed if r.eval_complete_count == 9),
            "comparison_eligible_paper_ready_count": sum(
                1
                for r in completed
                if r.eval_complete_count == 9 and is_comparison_eligible(r.run_id)
            ),
            "eval_partial": sum(1 for r in completed if 0 < r.eval_complete_count < 9),
            "eval_none": sum(1 for r in completed if r.eval_complete_count == 0),
            "incomplete_400ep_count": len(incomplete),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESEARCH_DIR,
        help="Directory for JSON output",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Write JSON only; do not update geodro_lejepa_eval_results.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary to stdout without writing files",
    )
    args = parser.parse_args()

    payload = collect()
    summary = payload["summary"]
    print(
        f"Completed 400ep: {summary['completed_400ep_count']} | "
        f"paper-ready: {summary['paper_ready_count']} | "
        f"partial: {summary['eval_partial']} | "
        f"no eval: {summary['eval_none']} | "
        f"incomplete attempts: {summary['incomplete_400ep_count']}"
    )
    for run in payload["completed_runs"]:
        tag = " [paper-ready]" if run.get("paper_ready") else ""
        print(f"  {run['run_id']}: {run['eval_complete_count']}/9 eval{tag}")

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / JSON_PATH.name
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")

    if not args.json_only:
        inject_auto_block(EVAL_DOC_PATH, render_auto_block(payload))
        print(f"Updated {EVAL_DOC_PATH}")


if __name__ == "__main__":
    main()
