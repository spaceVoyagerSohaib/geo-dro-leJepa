"""Eval-side metric utilities for the GeoDRO-LeJEPA evaluation suite.

These helpers are pure functions over already-computed predictions / features,
so they can be appended to existing dispatchers without re-running any forward
pass. Each function returns either a float or a dict[str, float] suitable for
direct insertion into the metrics JSON written by `geodro_lejepa_eval.py`.

References:
- Top-k accuracy: standard ImageNet convention.
- Expected Calibration Error: Naeini et al. 2015 (`Obtaining Well Calibrated
  Probabilities Using Bayesian Binning`), as used by ProSMin (ICLR 2024).
- Mean Corruption Error: Hendrycks & Dietterich 2019 (`Benchmarking Neural
  Network Robustness to Common Corruptions and Perturbations`); the AlexNet
  baseline below is the published per-corruption error table from that paper.
- AUROC for OOD detection: Hendrycks & Gimpel 2017 (`A Baseline for Detecting
  Misclassified and Out-of-Distribution Examples in Neural Networks`).
- Selective prediction / risk-coverage AUC: El-Yaniv & Wiener 2010.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


# Hendrycks & Dietterich (2019) Table 5 / Appendix: AlexNet per-corruption error
# on ImageNet-C, averaged across the five severities. Used as the denominator
# in mCE = mean over corruptions of (model_error / alexnet_error).
ALEXNET_IN1K_CE_BASELINES: dict[str, float] = {
    "gaussian_noise":    0.886,
    "shot_noise":        0.894,
    "impulse_noise":     0.923,
    "defocus_blur":      0.820,
    "glass_blur":        0.826,
    "motion_blur":       0.786,
    "zoom_blur":         0.798,
    "snow":              0.867,
    "frost":             0.827,
    "fog":               0.819,
    "brightness":        0.565,
    "contrast":          0.853,
    "elastic_transform": 0.646,
    "pixelate":          0.718,
    "jpeg_compression":  0.607,
}


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------
def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int] = (1, 5),
) -> dict[str, float]:
    """Return top-k accuracy for each k in `ks`.

    Args:
        logits: [N, C] real-valued scores.
        targets: [N] integer class indices in [0, C).
        ks: iterable of positive integers. Values larger than C are clamped.

    Returns:
        {"top{k}_acc": float} for each k. Empty input yields 0.0.
    """
    if logits.numel() == 0 or targets.numel() == 0:
        return {f"top{int(k)}_acc": 0.0 for k in ks}
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"top_k_accuracy: expected logits [N, C] and targets [N]; "
            f"got {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    num_classes = int(logits.shape[1])
    targets = targets.to(device=logits.device, dtype=torch.long)
    out: dict[str, float] = {}
    for k in ks:
        k_eff = max(1, min(int(k), num_classes))
        topk = logits.topk(k_eff, dim=1).indices  # [N, k_eff]
        correct = (topk == targets.unsqueeze(1)).any(dim=1).float()
        out[f"top{int(k)}_acc"] = float(correct.mean().detach().cpu())
    return out


# ---------------------------------------------------------------------------
# Calibration: ECE
# ---------------------------------------------------------------------------
def expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Equal-width binning ECE over the model's max-probability predictions.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|.
    """
    if probs.numel() == 0:
        return 0.0
    if probs.ndim != 2 or targets.ndim != 1 or probs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"expected_calibration_error: expected probs [N, C] and targets [N]; "
            f"got {tuple(probs.shape)} and {tuple(targets.shape)}."
        )
    if int(n_bins) <= 0:
        raise ValueError("n_bins must be positive.")

    probs = probs.detach().float()
    targets = targets.to(device=probs.device, dtype=torch.long)
    confidence, prediction = probs.max(dim=1)
    correct = (prediction == targets).float()

    bin_edges = torch.linspace(0.0, 1.0, int(n_bins) + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device, dtype=torch.float32)
    n = float(probs.shape[0])
    for b in range(int(n_bins)):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == int(n_bins) - 1:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence >= lo) & (confidence < hi)
        count = in_bin.float().sum()
        if count.item() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidence[in_bin].mean()
        ece = ece + (count / n) * (bin_acc - bin_conf).abs()
    return float(ece.detach().cpu())


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------
def negative_log_likelihood(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Mean cross-entropy in nats; matches `F.cross_entropy(reduction='mean')`."""
    if logits.numel() == 0:
        return 0.0
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"negative_log_likelihood: expected logits [N, C] and targets [N]; "
            f"got {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    targets = targets.to(device=logits.device, dtype=torch.long)
    return float(F.cross_entropy(logits.float(), targets, reduction="mean").detach().cpu())


# ---------------------------------------------------------------------------
# kNN probe
# ---------------------------------------------------------------------------
def knn_probe(
    features_train: torch.Tensor,
    targets_train: torch.Tensor,
    features_eval: torch.Tensor,
    targets_eval: torch.Tensor,
    *,
    k: int = 20,
    metric: str = "cosine",
    chunk_size: int = 1024,
) -> dict[str, float]:
    """Top-1 kNN classification accuracy with majority vote.

    Cosine metric uses L2-normalized features and ranks by descending
    similarity (== ascending cosine distance). Ties are broken by the
    smallest-class-id rule that `torch.mode` produces deterministically.
    """
    if features_train.ndim != 2 or features_eval.ndim != 2:
        raise ValueError(
            f"knn_probe: expected feature matrices [N, D]; got "
            f"{tuple(features_train.shape)} and {tuple(features_eval.shape)}."
        )
    if features_train.shape[1] != features_eval.shape[1]:
        raise ValueError("knn_probe: train and eval feature dim must match.")
    if metric != "cosine":
        raise ValueError(f"knn_probe: unsupported metric {metric!r}.")
    if features_eval.shape[0] == 0:
        return {"knn_top1_acc": 0.0}

    device = features_train.device
    train = F.normalize(features_train.detach().float(), dim=-1)
    evalf = F.normalize(features_eval.detach().float().to(device=device), dim=-1)
    y_train = targets_train.to(device=device, dtype=torch.long)
    y_eval = targets_eval.to(device=device, dtype=torch.long)
    k_eff = max(1, min(int(k), int(train.shape[0])))

    correct = 0
    total = int(evalf.shape[0])
    for start in range(0, total, int(chunk_size)):
        stop = min(start + int(chunk_size), total)
        sims = evalf[start:stop] @ train.T                     # [B, N_train]
        _, idx = sims.topk(k_eff, dim=1)                       # [B, k]
        votes = y_train.index_select(0, idx.reshape(-1)).reshape(idx.shape)
        # Majority vote per row.
        pred = torch.mode(votes, dim=1).values
        correct += int((pred == y_eval[start:stop]).sum().item())
    return {"knn_top1_acc": correct / total if total else 0.0}


# ---------------------------------------------------------------------------
# AUROC for OOD detection (max-softmax baseline)
# ---------------------------------------------------------------------------
def auroc_max_softmax(
    probs_id: torch.Tensor,
    probs_ood: torch.Tensor,
) -> float:
    """AUROC with score = max-softmax-prob; ID = positive class.

    A model that gives ID samples higher max-softmax than OOD samples gets
    AUROC = 1.0; identical distributions give 0.5.
    """
    if probs_id.numel() == 0 or probs_ood.numel() == 0:
        return 0.0
    if probs_id.ndim != 2 or probs_ood.ndim != 2:
        raise ValueError("auroc_max_softmax: probs must be 2D [N, C].")
    score_id = probs_id.detach().float().max(dim=1).values
    score_ood = probs_ood.detach().float().max(dim=1).values
    scores = torch.cat([score_id, score_ood], dim=0)
    labels = torch.cat([
        torch.ones_like(score_id),
        torch.zeros_like(score_ood),
    ], dim=0)
    return _auroc_from_scores(scores, labels)


def _auroc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC computed from scores (higher = positive) and binary labels."""
    if scores.numel() == 0:
        return 0.0
    pos_mask = labels.bool()
    n_pos = int(pos_mask.sum().item())
    n_neg = int((~pos_mask).sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # Rank-based AUROC formula: (sum of positive ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    # Use average ranks to handle ties correctly.
    order = torch.argsort(scores, descending=False)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=scores.dtype)
    # Average ties.
    sorted_scores = scores[order]
    n = int(scores.numel())
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg = (i + 1 + j) / 2.0  # average of ranks i+1..j (inclusive)
            ranks[order[i:j]] = avg
        i = j
    pos_rank_sum = ranks[pos_mask].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.detach().cpu())


# ---------------------------------------------------------------------------
# Selective-prediction risk-coverage AUC
# ---------------------------------------------------------------------------
def selective_prediction_auc(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Area under the risk-coverage curve (AURC).

    Sorts predictions by max-softmax confidence (descending), forms the
    cumulative error rate as coverage grows from 1/N to 1, and returns the
    Riemann-sum estimate ``AURC = (1/N) * sum_i risk(coverage_i)``. This is
    the standard definition (Geifman & El-Yaniv 2018); a perfect classifier
    gets 0 and an always-wrong classifier gets 1.
    """
    if probs.numel() == 0:
        return 0.0
    if probs.ndim != 2 or targets.ndim != 1 or probs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"selective_prediction_auc: expected probs [N, C] and targets [N]; "
            f"got {tuple(probs.shape)} and {tuple(targets.shape)}."
        )
    probs = probs.detach().float()
    targets = targets.to(device=probs.device, dtype=torch.long)
    conf, pred = probs.max(dim=1)
    err = (pred != targets).float()
    order = torch.argsort(conf, descending=True)
    err_sorted = err[order]
    cum_errors = torch.cumsum(err_sorted, dim=0)
    n = err_sorted.shape[0]
    coverage = torch.arange(1, n + 1, device=err_sorted.device, dtype=torch.float32)
    risk = cum_errors / coverage
    aurc = risk.mean()
    return float(aurc.detach().cpu())


# ---------------------------------------------------------------------------
# Mean Corruption Error
# ---------------------------------------------------------------------------
def mean_corruption_error(
    per_corruption_acc: Mapping[str, float],
    *,
    baseline: str = "alexnet_in1k",
) -> dict[str, float]:
    """Hendrycks-Dietterich mCE = mean over corruptions of (err / baseline_err).

    Args:
        per_corruption_acc: corruption_name -> top-1 accuracy averaged across
            the five severities (already in [0, 1]).
        baseline: one of {"alexnet_in1k"}; selects the published denominator.

    Returns a dict with `mCE`, `mean_error`, and the per-corruption CE values.
    """
    if not per_corruption_acc:
        return {"mCE": 0.0, "mean_error": 0.0}
    if baseline != "alexnet_in1k":
        raise ValueError(f"mean_corruption_error: unknown baseline {baseline!r}.")
    table = ALEXNET_IN1K_CE_BASELINES
    ces: dict[str, float] = {}
    errors: list[float] = []
    for name, acc in per_corruption_acc.items():
        if name not in table:
            # Be permissive — log under a separate prefix so the caller can
            # spot extra corruptions. mCE itself only includes the standard 15.
            ces[f"unknown_{name}_ce"] = float(1.0 - acc)
            continue
        err = float(1.0 - acc)
        errors.append(err)
        ces[f"{name}_ce"] = err / table[name]
    standard_ces = [v for k, v in ces.items() if not k.startswith("unknown_")]
    mce = sum(standard_ces) / len(standard_ces) if standard_ces else 0.0
    mean_err = sum(errors) / len(errors) if errors else 0.0
    return {"mCE": mce, "mean_error": mean_err, **ces}


# ---------------------------------------------------------------------------
# Clean vs shifted gap
# ---------------------------------------------------------------------------
def clean_vs_shifted_gap(acc_clean: float, acc_shifted: float) -> float:
    """Robustness drop = clean accuracy - shifted accuracy.

    Positive values mean the shift hurt the model; negative values are unusual
    and worth flagging in diagnostics.
    """
    return float(acc_clean) - float(acc_shifted)
