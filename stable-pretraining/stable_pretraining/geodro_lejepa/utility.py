"""Detached adversary utility construction for GeoDRO-LeJEPA."""

from __future__ import annotations

import torch

from .graph import graph_dirichlet_energy, graph_neighbor_average
from .types import GraphData, UtilityDiagnostics, UtilityMode


def build_utility(
    li_global: torch.Tensor,
    li_v_global: torch.Tensor,
    graph: GraphData,
    *,
    mode: UtilityMode | str = UtilityMode.VIEW_GRAPH_COHERENT,
    eps_loss: float = 1e-12,
    eps: float = 1e-12,
    u_clip: float = 5.0,
    eta_view: float = 1.0,
    gamma: float = 0.5,
    li_v_sample_dim: int | None = None,
) -> tuple[torch.Tensor, UtilityDiagnostics]:
    mode = UtilityMode(mode)
    li_global = li_global.detach().float()
    li_v_n_v = _as_sample_view(
        li_v_global.detach().float(),
        li_global.shape[0],
        sample_dim=li_v_sample_dim,
    )

    standardized_loss = _robust_log_iqr_standardize(
        li_global, eps_loss=eps_loss, eps=eps, u_clip=u_clip
    )
    if mode == UtilityMode.RAW_LOSS:
        u_loss = li_global
    else:
        u_loss = standardized_loss

    view_disp = _median_absolute_deviation(li_v_n_v, dim=1)
    view_scale = torch.median(view_disp).clamp_min(eps)
    view_reliability = torch.exp(-float(eta_view) * view_disp / view_scale)

    if mode in {UtilityMode.RAW_LOSS, UtilityMode.STANDARDIZED_LOSS}:
        utility = u_loss
    elif mode == UtilityMode.VIEW_AWARE:
        utility = view_reliability * u_loss
    elif mode == UtilityMode.VIEW_GRAPH_COHERENT:
        u_view = view_reliability * u_loss
        u_neighbor = graph_neighbor_average(u_view, graph)
        utility = (1.0 - float(gamma)) * u_view + float(gamma) * u_neighbor
    else:
        raise ValueError(f"Unsupported utility mode: {mode}")

    nan_or_inf_seen = not torch.isfinite(utility).all().item()
    if nan_or_inf_seen:
        utility = torch.zeros_like(li_global)

    diagnostics = UtilityDiagnostics(
        utility_mean=_float_stat(utility, "mean"),
        utility_std=_float_stat(utility, "std"),
        utility_min=_float_stat(utility, "min"),
        utility_max=_float_stat(utility, "max"),
        view_disp_mean=_float_stat(view_disp, "mean"),
        view_disp_max=_float_stat(view_disp, "max"),
        view_reliability_mean=_float_stat(view_reliability, "mean"),
        view_reliability_min=_float_stat(view_reliability, "min"),
        graph_dirichlet_energy=graph_dirichlet_energy(utility, graph),
        loss_standardized_positive_fraction=_mask_fraction(standardized_loss > 0),
        view_reliability_by_loss_sign=_view_reliability_sign_gap(
            standardized_loss,
            view_reliability,
        ),
        view_reliability_positive_loss_mean=_masked_mean(
            view_reliability,
            standardized_loss > 0,
        ),
        view_reliability_negative_loss_mean=_masked_mean(
            view_reliability,
            standardized_loss < 0,
        ),
        nan_or_inf_seen=nan_or_inf_seen,
    )
    return utility.detach(), diagnostics


def _robust_log_iqr_standardize(
    losses: torch.Tensor,
    *,
    eps_loss: float,
    eps: float,
    u_clip: float,
) -> torch.Tensor:
    log_loss = torch.log(losses.clamp_min(0.0) + eps_loss)
    median = torch.median(log_loss)
    q75 = torch.quantile(log_loss, 0.75)
    q25 = torch.quantile(log_loss, 0.25)
    standardized = (log_loss - median) / (q75 - q25 + eps)
    return standardized.clamp(-u_clip, u_clip)


def _median_absolute_deviation(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    median = values.median(dim=dim, keepdim=True).values
    return (values - median).abs().median(dim=dim).values


def _as_sample_view(
    li_v: torch.Tensor,
    num_samples: int,
    *,
    sample_dim: int | None,
) -> torch.Tensor:
    if li_v.ndim != 2:
        raise ValueError(f"Expected li_v_global as 2D, got {tuple(li_v.shape)}.")

    if sample_dim is not None:
        if sample_dim not in {0, 1}:
            raise ValueError(
                f"Expected li_v_sample_dim to be 0 or 1, got {sample_dim}."
            )
        if li_v.shape[sample_dim] != num_samples:
            raise ValueError(
                "Expected li_v_global sample dimension to match li_global length, "
                f"got li_v={tuple(li_v.shape)}, sample_dim={sample_dim}, "
                f"and N={num_samples}."
            )
        if sample_dim == 0:
            return li_v
        return li_v.T.contiguous()

    sample_dim_0 = li_v.shape[0] == num_samples
    sample_dim_1 = li_v.shape[1] == num_samples
    if sample_dim_0 and sample_dim_1:
        raise ValueError(
            "Ambiguous li_v_global orientation because both dimensions match "
            "li_global length; pass li_v_sample_dim=0 for [N, V] or "
            "li_v_sample_dim=1 for [V, N]."
        )
    if sample_dim_0:
        return li_v
    if sample_dim_1:
        return li_v.T.contiguous()
    raise ValueError(
        "Expected one li_v_global dimension to match li_global length, "
        f"got li_v={tuple(li_v.shape)} and N={num_samples}."
    )


def _float_stat(values: torch.Tensor, stat: str) -> float:
    if values.numel() == 0:
        return 0.0
    if stat == "mean":
        result = values.mean()
    elif stat == "std":
        result = values.std(unbiased=False)
    elif stat == "min":
        result = values.min()
    elif stat == "max":
        result = values.max()
    else:
        raise ValueError(f"Unknown stat: {stat}")
    return float(result.detach().cpu())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or not mask.any().item():
        return 0.0
    return float(values[mask].mean().detach().cpu())


def _mask_fraction(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().detach().cpu())


def _view_reliability_sign_gap(
    standardized_loss: torch.Tensor,
    view_reliability: torch.Tensor,
) -> float:
    positive = _masked_mean(view_reliability, standardized_loss > 0)
    negative = _masked_mean(view_reliability, standardized_loss < 0)
    return float(positive - negative)
