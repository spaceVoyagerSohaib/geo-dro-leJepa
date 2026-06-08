from __future__ import annotations

import pytest

from scripts.collect_full_eval_metrics import (
    collect_evals_for_run,
    eval_status_without_metrics,
)
from scripts.validate_full_eval_matrix import method_matches


pytestmark = pytest.mark.unit


def test_args_only_eval_without_sacct_is_unknown(tmp_path):
    eval_dir = tmp_path / "eval" / "imagenet100ctrl-1234567"
    eval_dir.mkdir(parents=True)
    (eval_dir / "args.json").write_text("{}")

    records = collect_evals_for_run(tmp_path, "run-a", {}, {})

    assert records["imagenet100ctrl"].status == "unknown_no_metrics"


def test_args_only_completed_eval_without_metrics_is_missing_metrics(tmp_path):
    eval_dir = tmp_path / "eval" / "imagenet100ctrl-1234567"
    eval_dir.mkdir(parents=True)
    (eval_dir / "args.json").write_text("{}")

    records = collect_evals_for_run(
        tmp_path,
        "run-a",
        {"1234567": "COMPLETED"},
        {},
    )

    assert records["imagenet100ctrl"].status == "missing_metrics"


def test_eval_log_without_metrics_uses_unknown_or_failed_status(tmp_path):
    log_index = {
        "run-a": {
            "imagenet100c": [
                {"job_id": "7654321", "metrics_path": "", "failed": False},
                {"job_id": "7654322", "metrics_path": "", "failed": True},
            ]
        }
    }

    records = collect_evals_for_run(tmp_path, "run-a", {}, log_index)

    assert records["imagenet100c"].job_id == "7654322"
    assert records["imagenet100c"].status == "failed"


def test_eval_status_without_metrics_handles_slurm_state_suffix():
    assert eval_status_without_metrics("1", {"1": "TIMEOUT+"}) == "failed"
    assert eval_status_without_metrics("2", {"2": "RUNNING"}) == "running"
    assert eval_status_without_metrics("3", {"3": "PENDING"}) == "pending"


def test_validator_accepts_historical_method_aliases():
    assert method_matches("geodro_v1_1", "geodro_v1_1_r4")
    assert method_matches("geodro_v1_1", "geodro_v1_1_r6")
    assert not method_matches("geodro_v1_1", "lejepa_erm")
