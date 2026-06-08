#!/usr/bin/env python3
"""Validate the generated GeoDRO-LeJEPA full eval matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
RESEARCH_DIR = REPO_ROOT / "research" / "curated_results"
MATRIX_PATH = RESEARCH_DIR / "full_pretrain_eval_matrix.json"
EVAL_DOC_PATH = RESEARCH_DIR / "geodro_lejepa_eval_results.md"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_full_eval_metrics as collector  # noqa: E402
from _eval_metrics import (  # noqa: E402
    ALEXNET_IN1K_CE_BASELINES,
    clean_vs_shifted_gap,
    mean_corruption_error,
)


METHOD_ALIASES: dict[str, frozenset[str]] = {
    "geodro_v1_1": frozenset({"geodro_v1_1_r4", "geodro_v1_1_r6"}),
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def comparable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    out.pop("generated_at", None)
    for section in ("completed_runs", "incomplete_400ep_runs"):
        for run in out.get(section, []):
            run.pop("pretrain_metrics", None)
    return out


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def metric_close(a: float | None, b: float | None, *, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def method_matches(run_method: str, eval_method: str) -> bool:
    if run_method == eval_method:
        return True
    return eval_method in METHOD_ALIASES.get(run_method, frozenset())


def mode_headline_keys(mode: str, payload: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for column in payload.get("headline_columns", []):
        key = column.get("key", "")
        if key.startswith(f"{mode}/"):
            keys.append(key)
    return keys


def validate_completed_eval(
    run: dict[str, Any],
    mode: str,
    rec: dict[str, Any],
    payload: dict[str, Any],
    errors: list[str],
    alias_notes: set[str],
) -> None:
    metrics_path = resolve_repo_path(str(rec.get("metrics_path", "")))
    if not metrics_path.is_file():
        errors.append(f"{run['run_id']} {mode}: metrics_path missing: {metrics_path}")
        return

    source = load_json(metrics_path)
    parsed = collector.parse_eval_dir_name(metrics_path.parent.name)
    if parsed != (mode, str(rec.get("job_id", ""))):
        errors.append(
            f"{run['run_id']} {mode}: metrics directory/job mismatch "
            f"{metrics_path.parent.name!r} vs job {rec.get('job_id')!r}"
        )

    expected_fields = {
        "schema_version": 2,
        "mode": mode,
        "pretrain_run_id": run["run_id"],
        "checkpoint": run.get("checkpoint", ""),
    }
    for field, expected in expected_fields.items():
        actual = source.get(field)
        if actual != expected:
            errors.append(
                f"{run['run_id']} {mode}: payload {field}={actual!r}, "
                f"expected {expected!r}"
            )

    eval_method = str(source.get("method", ""))
    run_method = str(run.get("method", ""))
    if not method_matches(run_method, eval_method):
        errors.append(
            f"{run['run_id']} {mode}: eval method {eval_method!r} does not "
            f"match run method {run_method!r}"
        )
    elif eval_method != run_method:
        alias_notes.add(f"{run['run_id']}: {eval_method} -> {run_method}")

    flattened = collector.flatten_mode_metrics(source)
    if rec.get("metrics") != flattened:
        errors.append(f"{run['run_id']} {mode}: matrix metrics differ from source JSON")

    for key in mode_headline_keys(mode, payload):
        if collector.metric_get(flattened, key) is None:
            errors.append(f"{run['run_id']} {mode}: missing headline metric {key}")

    if mode == "imagenet100c":
        validate_imagenetc_derivatives(run["run_id"], flattened, errors)


def validate_imagenetc_derivatives(
    run_id: str, metrics: dict[str, Any], errors: list[str]
) -> None:
    per_corruption: dict[str, float] = {}
    for name in ALEXNET_IN1K_CE_BASELINES:
        value = collector.metric_get(metrics, f"imagenet100c/{name}/top1_acc")
        if value is None:
            errors.append(f"{run_id} imagenet100c: missing {name} top1_acc for mCE")
            continue
        per_corruption[name] = value

    if len(per_corruption) == len(ALEXNET_IN1K_CE_BASELINES):
        expected_mce = mean_corruption_error(per_corruption)["mCE"]
        actual_mce = collector.metric_get(metrics, "imagenet100c/mCE")
        if not metric_close(actual_mce, expected_mce):
            errors.append(
                f"{run_id} imagenet100c: mCE={actual_mce}, "
                f"expected {expected_mce}"
            )

    clean_acc = metrics.get("monitor/best_acc")
    corrupted_acc = metrics.get("imagenetc/mean_acc")
    actual_gap = collector.metric_get(metrics, "imagenet100c/clean_vs_corrupted_gap")
    if isinstance(clean_acc, (int, float)) and isinstance(corrupted_acc, (int, float)):
        expected_gap = clean_vs_shifted_gap(float(clean_acc), float(corrupted_acc))
        if not metric_close(actual_gap, expected_gap):
            errors.append(
                f"{run_id} imagenet100c: clean-vs-corrupted gap={actual_gap}, "
                f"expected {expected_gap}"
            )


def validate_payload_internal(payload: dict[str, Any], errors: list[str]) -> None:
    completed = payload.get("completed_runs", [])
    required_modes = payload.get("required_modes", [])
    headline_rows = payload.get("headline_rows", [])

    row_by_id = {row["run_id"]: row for row in headline_rows}
    for run in completed:
        complete_count = sum(
            1
            for mode in required_modes
            if (run.get("evals") or {}).get(mode, {}).get("status") == "completed"
        )
        if run.get("eval_complete_count") != complete_count:
            errors.append(
                f"{run['run_id']}: eval_complete_count={run.get('eval_complete_count')}, "
                f"expected {complete_count}"
            )

        missing = [
            mode
            for mode in required_modes
            if (run.get("evals") or {}).get(mode, {}).get("status") != "completed"
        ]
        if run.get("missing_modes") != missing:
            errors.append(f"{run['run_id']}: missing_modes={run.get('missing_modes')}, expected {missing}")

        row = row_by_id.get(run["run_id"])
        if not row:
            errors.append(f"{run['run_id']}: missing headline row")
            continue
        merged: dict[str, Any] = {}
        for rec in (run.get("evals") or {}).values():
            if rec.get("status") == "completed":
                merged.update(rec.get("metrics") or {})
        for column in payload.get("headline_columns", []):
            key = column["key"]
            expected = collector.metric_get(merged, key)
            actual = row.get(key)
            if not metric_close(actual, expected):
                errors.append(
                    f"{run['run_id']}: headline {key}={actual}, expected {expected}"
                )

    expected_deltas = collector.compute_deltas(headline_rows)
    if payload.get("delta_vs_erm") != expected_deltas:
        errors.append("delta_vs_erm does not match recomputed deltas")

    expected_bests = collector.find_best_per_column(headline_rows)
    if payload.get("best_run_per_column") != expected_bests:
        errors.append("best_run_per_column does not match recomputed bests")

    summary = payload.get("summary", {})
    expected_summary = {
        "completed_400ep_count": len(completed),
        "eval_complete_9_of_9": sum(
            1 for r in completed if r.get("eval_complete_count") == 9
        ),
        "paper_ready_count": sum(
            1 for r in completed if r.get("eval_complete_count") == 9
        ),
        "comparison_eligible_paper_ready_count": sum(
            1
            for r in completed
            if r.get("eval_complete_count") == 9
            and collector.is_comparison_eligible(r["run_id"])
        ),
        "eval_partial": sum(
            1 for r in completed if 0 < int(r.get("eval_complete_count", 0)) < 9
        ),
        "eval_none": sum(1 for r in completed if r.get("eval_complete_count") == 0),
        "incomplete_400ep_count": len(payload.get("incomplete_400ep_runs", [])),
    }
    if summary != expected_summary:
        errors.append(f"summary={summary}, expected {expected_summary}")


def validate_doc_auto_block(
    payload: dict[str, Any], doc_path: Path, errors: list[str]
) -> None:
    if not doc_path.is_file():
        errors.append(f"eval results doc missing: {doc_path}")
        return
    text = doc_path.read_text()
    if collector.AUTO_START not in text or collector.AUTO_END not in text:
        errors.append(f"{rel(doc_path)} missing auto block markers")
        return
    before, rest = text.split(collector.AUTO_START, maxsplit=1)
    _ = before
    current, _after = rest.split(collector.AUTO_END, maxsplit=1)
    expected = "\n\n" + collector.render_auto_block(payload).rstrip() + "\n\n"
    if current != expected:
        errors.append(f"{rel(doc_path)} auto block differs from matrix render")


def validate_live_freshness(
    payload: dict[str, Any], errors: list[str], *, verbose: bool
) -> None:
    live = collector.collect()
    if comparable_payload(payload) == comparable_payload(live):
        return

    errors.append("matrix JSON is stale relative to current source metrics")
    saved_runs = {r["run_id"]: r for r in payload.get("completed_runs", [])}
    live_runs = {r["run_id"]: r for r in live.get("completed_runs", [])}
    for run_id in sorted(set(saved_runs) | set(live_runs)):
        saved = saved_runs.get(run_id)
        current = live_runs.get(run_id)
        if not saved or not current:
            errors.append(f"{run_id}: presence differs between saved and live matrix")
            continue
        if saved.get("eval_complete_count") != current.get("eval_complete_count"):
            errors.append(
                f"{run_id}: saved eval {saved.get('eval_complete_count')}/9, "
                f"live {current.get('eval_complete_count')}/9"
            )
        if saved.get("missing_modes") != current.get("missing_modes"):
            errors.append(
                f"{run_id}: saved missing {saved.get('missing_modes')}, "
                f"live {current.get('missing_modes')}"
            )
        if verbose:
            for mode in payload.get("required_modes", []):
                saved_ev = (saved.get("evals") or {}).get(mode, {})
                live_ev = (current.get("evals") or {}).get(mode, {})
                if saved_ev.get("status") != live_ev.get("status"):
                    errors.append(
                        f"{run_id} {mode}: saved status {saved_ev.get('status')}, "
                        f"live {live_ev.get('status')}"
                    )


def validate(payload: dict[str, Any], doc_path: Path, *, skip_live: bool, verbose: bool) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    alias_notes: set[str] = set()

    validate_payload_internal(payload, errors)
    for run in payload.get("completed_runs", []):
        for mode, rec in (run.get("evals") or {}).items():
            if rec.get("status") == "completed":
                validate_completed_eval(run, mode, rec, payload, errors, alias_notes)
    validate_doc_auto_block(payload, doc_path, errors)
    if not skip_live:
        validate_live_freshness(payload, errors, verbose=verbose)
    return errors, alias_notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--doc", type=Path, default=EVAL_DOC_PATH)
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Do not compare the saved matrix with a fresh collector pass.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include per-mode live freshness differences.",
    )
    args = parser.parse_args()

    payload = load_json(args.matrix)
    errors, alias_notes = validate(
        payload,
        args.doc,
        skip_live=args.skip_live,
        verbose=args.verbose,
    )
    if errors:
        print(f"Validation failed for {rel(args.matrix)}:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    completed_eval_count = sum(
        int(run.get("eval_complete_count", 0))
        for run in payload.get("completed_runs", [])
    )
    print(
        f"Validation OK: {completed_eval_count} completed eval records checked "
        f"across {len(payload.get('completed_runs', []))} completed 400ep runs."
    )
    if alias_notes:
        print("Approved method aliases:")
        for note in sorted(alias_notes):
            print(f"  - {note}")


if __name__ == "__main__":
    main()
