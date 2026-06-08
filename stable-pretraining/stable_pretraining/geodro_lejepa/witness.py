"""Sparse memory witness overlap scores for GeoDRO geometry support."""

from __future__ import annotations

import warnings

import torch

from .types import (
    MemoryWitnessBatch,
    MemoryWitnessOverlapScores,
    WitnessScoreMode,
)


def compute_witness_overlap_scores(
    witnesses: MemoryWitnessBatch,
    *,
    mode: WitnessScoreMode | str = WitnessScoreMode.SPECIFICITY_WEIGHTED_HELLINGER,
) -> MemoryWitnessOverlapScores:
    """Compute detached current-current overlaps from sparse memory witnesses."""
    score_mode = WitnessScoreMode(mode)
    if not witnesses.valid_memory_for_witnessing or witnesses.indices.numel() == 0:
        return _empty_overlap_scores(witnesses, score_mode=score_mode)

    batch_size, top_m = witnesses.indices.shape
    if batch_size == 0 or top_m == 0:
        return _empty_overlap_scores(witnesses, score_mode=score_mode)

    indices = witnesses.indices.detach().reshape(-1).long()
    probabilities = witnesses.probabilities.detach().float().reshape(-1)
    unique_indices, inverse = torch.unique(
        indices,
        sorted=True,
        return_inverse=True,
    )
    selected_counts = torch.bincount(
        inverse,
        minlength=unique_indices.numel(),
    ).to(device=probabilities.device, dtype=torch.float32)
    specificity_weights = _specificity_weights(
        selected_counts,
        batch_size=batch_size,
    )
    rows = torch.arange(batch_size, device=indices.device).repeat_interleave(top_m)
    values = probabilities.clamp_min(0.0).sqrt()
    raw_incidence = _sparse_incidence(
        rows,
        inverse,
        values,
        size=(batch_size, unique_indices.numel()),
    )
    specificity_values = values * specificity_weights.index_select(0, inverse).sqrt()
    specificity_incidence = _sparse_incidence(
        rows,
        inverse,
        specificity_values,
        size=(batch_size, unique_indices.numel()),
    )
    raw_overlap = _sparse_overlap_to_dense(raw_incidence)
    specificity_overlap = _sparse_overlap_to_dense(specificity_incidence)
    if score_mode == WitnessScoreMode.RAW_HELLINGER:
        selected_overlap = raw_overlap
    else:
        selected_overlap = specificity_overlap
    return MemoryWitnessOverlapScores(
        raw_overlap=raw_overlap.detach(),
        specificity_overlap=specificity_overlap.detach(),
        selected_overlap=selected_overlap.detach(),
        selected_counts=selected_counts.detach(),
        specificity_weights=specificity_weights.detach(),
        witness_score_mode=score_mode,
        valid_for_scoring=True,
    )


def _empty_overlap_scores(
    witnesses: MemoryWitnessBatch,
    *,
    score_mode: WitnessScoreMode,
) -> MemoryWitnessOverlapScores:
    batch_size = int(witnesses.indices.shape[0])
    device = witnesses.indices.device
    raw = torch.zeros((batch_size, batch_size), device=device, dtype=torch.float32)
    empty = torch.empty((0,), device=device, dtype=torch.float32)
    return MemoryWitnessOverlapScores(
        raw_overlap=raw,
        specificity_overlap=raw.clone(),
        selected_overlap=raw.clone(),
        selected_counts=empty,
        specificity_weights=empty,
        witness_score_mode=score_mode,
        valid_for_scoring=False,
    )


def _specificity_weights(
    selected_counts: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if selected_counts.numel() == 0 or batch_size <= 0:
        return selected_counts.new_empty((0,))
    batch = selected_counts.new_tensor(float(batch_size))
    numerator = ((batch + 1.0) / (selected_counts + 1.0)).log()
    denominator = (batch + 1.0).log().clamp_min(torch.finfo(selected_counts.dtype).eps)
    return (numerator / denominator).clamp(0.0, 1.0)


def _sparse_incidence(
    rows: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    *,
    size: tuple[int, int],
) -> torch.Tensor:
    indices = torch.stack([rows.to(torch.long), cols.to(torch.long)])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled.*",
            category=UserWarning,
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=size,
            device=values.device,
        ).coalesce()


def _sparse_overlap_to_dense(incidence: torch.Tensor) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        return torch.sparse.mm(incidence, incidence.transpose(0, 1)).to_dense()
