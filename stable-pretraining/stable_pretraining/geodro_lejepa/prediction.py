"""LeJEPA prediction-loss decomposition for GeoDRO aggregation."""

from __future__ import annotations

import torch

from .types import PredictionTerms


def select_global_views(
    tensor: torch.Tensor,
    *,
    global_mask: torch.Tensor | None = None,
    global_view_count: int | None = None,
) -> torch.Tensor:
    if global_mask is not None and global_view_count is not None:
        raise ValueError("Provide only one of global_mask or global_view_count.")
    if global_mask is not None:
        selected = tensor[global_mask]
    elif global_view_count is not None:
        selected = tensor[:global_view_count]
    else:
        selected = tensor
    if selected.numel() == 0:
        raise ValueError("No global views provided for center computation.")
    return selected


def compute_prediction_terms(
    proj: torch.Tensor,
    *,
    global_mask: torch.Tensor | None = None,
    global_view_count: int | None = None,
) -> PredictionTerms:
    if proj.ndim != 3:
        raise ValueError(f"Expected proj with shape [V, B, K], got {tuple(proj.shape)}.")

    global_proj = select_global_views(
        proj, global_mask=global_mask, global_view_count=global_view_count
    )
    centers = global_proj.mean(dim=0)
    li_v = (proj - centers.unsqueeze(0)).square().mean(dim=2)
    li_local = li_v.mean(dim=0)
    pred_erm = li_local.mean()
    return PredictionTerms(
        centers=centers,
        li_v=li_v,
        li_local=li_local,
        pred_erm=pred_erm,
    )
