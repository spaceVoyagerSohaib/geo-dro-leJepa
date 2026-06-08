"""Unit tests for the eval-side metric utilities."""

from __future__ import annotations

import math

import pytest
import torch

from scripts._eval_metrics import (
    ALEXNET_IN1K_CE_BASELINES,
    auroc_max_softmax,
    clean_vs_shifted_gap,
    expected_calibration_error,
    knn_probe,
    mean_corruption_error,
    negative_log_likelihood,
    selective_prediction_auc,
    top_k_accuracy,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# top_k_accuracy
# ---------------------------------------------------------------------------
def test_top_k_accuracy_perfect_prediction():
    logits = torch.eye(5) * 10.0  # argmax = i for sample i
    targets = torch.arange(5)
    out = top_k_accuracy(logits, targets, ks=(1, 3, 5))
    assert out["top1_acc"] == pytest.approx(1.0)
    assert out["top3_acc"] == pytest.approx(1.0)
    assert out["top5_acc"] == pytest.approx(1.0)


def test_top_k_accuracy_top5_recovers_when_top1_misses():
    # Sample i: true class is i, but argmax is (i+1) mod 10. True class is in top-2.
    logits = torch.zeros(10, 10)
    for i in range(10):
        logits[i, (i + 1) % 10] = 5.0
        logits[i, i] = 4.0
    targets = torch.arange(10)
    out = top_k_accuracy(logits, targets, ks=(1, 2, 5))
    assert out["top1_acc"] == pytest.approx(0.0)
    assert out["top2_acc"] == pytest.approx(1.0)
    assert out["top5_acc"] == pytest.approx(1.0)


def test_top_k_accuracy_clamps_k_to_num_classes():
    logits = torch.eye(3) * 5.0  # 3 classes, perfect prediction
    targets = torch.arange(3)
    out = top_k_accuracy(logits, targets, ks=(1, 5, 100))  # 5 and 100 > 3
    # k>3 clamps to 3; with perfect prediction all are 1.0
    assert out["top1_acc"] == pytest.approx(1.0)
    assert out["top5_acc"] == pytest.approx(1.0)
    assert out["top100_acc"] == pytest.approx(1.0)


def test_top_k_accuracy_empty_input():
    out = top_k_accuracy(torch.empty(0, 10), torch.empty(0, dtype=torch.long), ks=(1, 5))
    assert out == {"top1_acc": 0.0, "top5_acc": 0.0}


def test_top_k_accuracy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        top_k_accuracy(torch.zeros(5, 3), torch.zeros(4, dtype=torch.long))


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------
def test_ece_zero_for_perfectly_calibrated():
    # Confidence == accuracy in every bin: assign 100% confidence to a correct
    # prediction with two classes.
    n = 1000
    probs = torch.zeros(n, 2)
    probs[:, 0] = 1.0
    probs[:, 1] = 0.0
    targets = torch.zeros(n, dtype=torch.long)  # all correct, conf=1.0
    ece = expected_calibration_error(probs, targets, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_high_for_overconfident_wrong():
    # Model says 0.99 prob for class 0 but is always wrong (true class = 1).
    n = 1000
    probs = torch.zeros(n, 2)
    probs[:, 0] = 0.99
    probs[:, 1] = 0.01
    targets = torch.ones(n, dtype=torch.long)  # always wrong, conf=0.99 vs acc=0
    ece = expected_calibration_error(probs, targets, n_bins=15)
    assert ece == pytest.approx(0.99, abs=1e-2)


def test_ece_handles_empty_input():
    ece = expected_calibration_error(torch.empty(0, 5), torch.empty(0, dtype=torch.long))
    assert ece == 0.0


# ---------------------------------------------------------------------------
# negative_log_likelihood
# ---------------------------------------------------------------------------
def test_nll_matches_torch_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(64, 10)
    targets = torch.randint(0, 10, (64,))
    expected = torch.nn.functional.cross_entropy(logits, targets, reduction="mean").item()
    assert negative_log_likelihood(logits, targets) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# kNN probe
# ---------------------------------------------------------------------------
def test_knn_probe_perfect_when_clusters_separate():
    # Two well-separated clusters in 2D, both train and eval.
    train_a = torch.randn(20, 2) * 0.05 + torch.tensor([10.0, 0.0])
    train_b = torch.randn(20, 2) * 0.05 + torch.tensor([-10.0, 0.0])
    train_x = torch.cat([train_a, train_b])
    train_y = torch.cat([torch.zeros(20), torch.ones(20)]).long()

    eval_a = torch.randn(10, 2) * 0.05 + torch.tensor([10.0, 0.0])
    eval_b = torch.randn(10, 2) * 0.05 + torch.tensor([-10.0, 0.0])
    eval_x = torch.cat([eval_a, eval_b])
    eval_y = torch.cat([torch.zeros(10), torch.ones(10)]).long()

    out = knn_probe(train_x, train_y, eval_x, eval_y, k=5)
    assert out["knn_top1_acc"] == pytest.approx(1.0)


def test_knn_probe_empty_eval():
    train_x = torch.randn(10, 4)
    train_y = torch.zeros(10, dtype=torch.long)
    eval_x = torch.empty(0, 4)
    eval_y = torch.empty(0, dtype=torch.long)
    out = knn_probe(train_x, train_y, eval_x, eval_y, k=5)
    assert out == {"knn_top1_acc": 0.0}


def test_knn_probe_k_larger_than_training_set():
    train_x = torch.eye(3)
    train_y = torch.tensor([0, 1, 2])
    eval_x = torch.eye(3)
    eval_y = torch.tensor([0, 1, 2])
    # k=10 is clamped to 3; majority of all 3 votes will tie, but each row's
    # nearest is itself, so prediction equals target via tie resolution.
    out = knn_probe(train_x, train_y, eval_x, eval_y, k=10)
    # With k=N every query receives all training labels: torch.mode picks
    # the lowest class id on ties. The exact accuracy depends on ties; we
    # only check that it does not error and returns a valid float.
    assert 0.0 <= out["knn_top1_acc"] <= 1.0


def test_knn_probe_dim_mismatch_raises():
    with pytest.raises(ValueError):
        knn_probe(
            torch.randn(5, 3),
            torch.zeros(5, dtype=torch.long),
            torch.randn(2, 4),
            torch.zeros(2, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------
def test_auroc_perfectly_separable_is_one():
    # ID samples have max prob 0.9; OOD have 0.1.
    probs_id = torch.zeros(50, 10)
    probs_id[:, 0] = 0.9
    probs_id[:, 1:] = 0.1 / 9
    probs_ood = torch.zeros(50, 10)
    probs_ood[:, 0] = 0.1
    probs_ood[:, 1:] = 0.9 / 9
    auc = auroc_max_softmax(probs_id, probs_ood)
    assert auc == pytest.approx(1.0)


def test_auroc_identical_distributions_is_half():
    torch.manual_seed(0)
    logits = torch.randn(200, 10)
    probs = torch.softmax(logits, dim=1)
    # Same distribution split in two:
    auc = auroc_max_softmax(probs[:100], probs[100:])
    assert 0.4 <= auc <= 0.6


def test_auroc_inverted_separation_is_zero():
    probs_id = torch.zeros(20, 5)
    probs_id[:, 0] = 0.1
    probs_id[:, 1:] = 0.9 / 4
    probs_ood = torch.zeros(20, 5)
    probs_ood[:, 0] = 0.9
    probs_ood[:, 1:] = 0.1 / 4
    auc = auroc_max_softmax(probs_id, probs_ood)
    assert auc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Selective-prediction AUC
# ---------------------------------------------------------------------------
def test_selective_prediction_auc_zero_when_perfect():
    # Always correct.
    probs = torch.zeros(20, 3)
    probs[:, 0] = 1.0
    targets = torch.zeros(20, dtype=torch.long)
    auc = selective_prediction_auc(probs, targets)
    assert auc == pytest.approx(0.0)


def test_selective_prediction_auc_one_when_always_wrong():
    probs = torch.zeros(20, 3)
    probs[:, 0] = 1.0
    targets = torch.ones(20, dtype=torch.long)  # always wrong
    auc = selective_prediction_auc(probs, targets)
    assert auc == pytest.approx(1.0)


def test_selective_prediction_auc_lower_when_confidence_correlates_with_correctness():
    # Two groups: high-confidence correct, low-confidence wrong (well-calibrated).
    n_each = 50
    probs_correlated = torch.zeros(2 * n_each, 2)
    targets = torch.zeros(2 * n_each, dtype=torch.long)
    # First 50: confident & correct.
    probs_correlated[:n_each, 0] = 0.95
    probs_correlated[:n_each, 1] = 0.05
    # Last 50: low-confidence & wrong.
    probs_correlated[n_each:, 0] = 0.55
    probs_correlated[n_each:, 1] = 0.45
    targets[n_each:] = 1
    auc_correlated = selective_prediction_auc(probs_correlated, targets)

    # Anti-correlated: confidence and correctness are inversely linked.
    # First 50 are wrong but confident; last 50 are correct but unsure.
    probs_anti = torch.zeros(2 * n_each, 2)
    targets_anti = torch.zeros(2 * n_each, dtype=torch.long)
    probs_anti[:n_each, 0] = 0.95
    probs_anti[:n_each, 1] = 0.05
    targets_anti[:n_each] = 1  # confident but wrong
    probs_anti[n_each:, 0] = 0.55
    probs_anti[n_each:, 1] = 0.45
    # targets_anti[n_each:] = 0 (already), so unsure but correct.
    auc_anti = selective_prediction_auc(probs_anti, targets_anti)

    # Well-calibrated AURC must be strictly lower than anti-correlated AURC.
    assert auc_correlated < auc_anti


# ---------------------------------------------------------------------------
# mean_corruption_error
# ---------------------------------------------------------------------------
def test_mce_alexnet_baseline_reproduces_one():
    # If the model's per-corruption error EQUALS AlexNet's baseline error,
    # mCE = 1.0 exactly.
    per_corruption_acc = {
        name: 1.0 - err for name, err in ALEXNET_IN1K_CE_BASELINES.items()
    }
    out = mean_corruption_error(per_corruption_acc)
    assert out["mCE"] == pytest.approx(1.0)


def test_mce_perfect_model_is_zero():
    per_corruption_acc = {name: 1.0 for name in ALEXNET_IN1K_CE_BASELINES}
    out = mean_corruption_error(per_corruption_acc)
    assert out["mCE"] == pytest.approx(0.0)
    assert out["mean_error"] == pytest.approx(0.0)


def test_mce_unknown_corruption_logged_separately():
    per_corruption_acc = {
        "gaussian_noise": 0.5,
        "made_up_corruption": 0.5,
    }
    out = mean_corruption_error(per_corruption_acc)
    # mCE only includes the standard 15 corruptions present.
    assert "gaussian_noise_ce" in out
    assert "unknown_made_up_corruption_ce" in out
    assert out["mCE"] == pytest.approx(0.5 / ALEXNET_IN1K_CE_BASELINES["gaussian_noise"])


def test_mce_empty_input():
    out = mean_corruption_error({})
    assert out == {"mCE": 0.0, "mean_error": 0.0}


def test_mce_unknown_baseline_raises():
    with pytest.raises(ValueError):
        mean_corruption_error({"gaussian_noise": 0.5}, baseline="unknown")


# ---------------------------------------------------------------------------
# clean_vs_shifted_gap
# ---------------------------------------------------------------------------
def test_clean_vs_shifted_gap_basic():
    assert clean_vs_shifted_gap(0.85, 0.60) == pytest.approx(0.25)
    assert clean_vs_shifted_gap(0.50, 0.60) == pytest.approx(-0.10)
