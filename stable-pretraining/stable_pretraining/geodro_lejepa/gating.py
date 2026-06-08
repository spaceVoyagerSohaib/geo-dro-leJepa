"""Reliability-gated uniform mixture for GeoDRO-LeJEPA weights."""

from __future__ import annotations

import torch

from .flow import entropy, ess_ratio
from .types import FlowDiagnostics, GraphDiagnostics, WeightDiagnostics


def reliability_gated_weights(
    p_flow: torch.Tensor,
    graph_diag: GraphDiagnostics,
    flow_diag: FlowDiagnostics,
    *,
    step: int | None = None,
    total_steps: int | None = None,
    alpha_max: float = 0.5,
    warmup_fraction: float = 0.10,
    ramp_fraction: float = 0.05,
    ess_min_ratio: float = 0.25,
    max_p_factor_fail: float = 10.0,
    clamp_activation_fail: float = 0.01,
    singleton_fraction_fail: float = 0.5,
    min_graph_nodes: int | None = None,
    p_cap: float | None = None,
) -> tuple[torch.Tensor, WeightDiagnostics]:
    num_nodes = p_flow.numel()
    uniform = torch.ones_like(p_flow) / max(num_nodes, 1)
    reason = _fallback_reason(
        p_flow,
        graph_diag,
        flow_diag,
        ess_min_ratio=ess_min_ratio,
        max_p_factor_fail=max_p_factor_fail,
        clamp_activation_fail=clamp_activation_fail,
        singleton_fraction_fail=singleton_fraction_fail,
        min_graph_nodes=min_graph_nodes,
    )
    warmup = warmup_ramp(
        step=step,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        ramp_fraction=ramp_fraction,
    )
    graph_gate = 0.0 if reason is not None and reason.startswith("graph") else 1.0
    flow_gate = 0.0 if reason is not None and not reason.startswith("graph") else 1.0
    alpha = 0.0 if reason is not None else float(alpha_max) * warmup

    p_train = (1.0 - alpha) * uniform + alpha * p_flow
    if p_cap is not None and p_cap > 0:
        p_train = p_train.clamp(max=float(p_cap))
        p_train = p_train / p_train.sum().clamp_min(torch.finfo(p_train.dtype).eps)

    diagnostics = WeightDiagnostics(
        alpha=float(alpha),
        warmup_multiplier=float(warmup),
        warmup_step=int(step) if step is not None else None,
        warmup_total_steps=int(total_steps) if total_steps is not None else None,
        graph_gate=float(graph_gate),
        flow_gate=float(flow_gate),
        fallback=reason is not None,
        fallback_reason=reason,
        entropy=float(entropy(p_train).detach().cpu()) if num_nodes else 0.0,
        ess_ratio=float(ess_ratio(p_train).detach().cpu()) if num_nodes else 0.0,
        max_p=float(p_train.max().detach().cpu()) if num_nodes else 0.0,
        min_p=float(p_train.min().detach().cpu()) if num_nodes else 0.0,
    )
    return p_train.detach(), diagnostics


def warmup_ramp(
    *,
    step: int | None,
    total_steps: int | None,
    warmup_fraction: float,
    ramp_fraction: float,
) -> float:
    if step is None or total_steps is None or total_steps <= 0:
        return 1.0
    warmup_steps = int(float(warmup_fraction) * total_steps)
    ramp_steps = max(1, int(float(ramp_fraction) * total_steps))
    if step < warmup_steps:
        return 0.0
    if step >= warmup_steps + ramp_steps:
        return 1.0
    return float(step - warmup_steps) / ramp_steps


def _fallback_reason(
    p_flow: torch.Tensor,
    graph_diag: GraphDiagnostics,
    flow_diag: FlowDiagnostics,
    *,
    ess_min_ratio: float,
    max_p_factor_fail: float,
    clamp_activation_fail: float,
    singleton_fraction_fail: float,
    min_graph_nodes: int | None,
) -> str | None:
    num_nodes = p_flow.numel()
    if num_nodes == 0:
        return "empty_weights"
    if not torch.isfinite(p_flow).all() or not torch.isfinite(p_flow.sum()):
        return "invalid_weights"
    if abs(float(p_flow.sum().detach().cpu()) - 1.0) > 1e-4:
        return "invalid_weight_sum"
    if min_graph_nodes is not None and graph_diag.num_nodes < int(min_graph_nodes):
        return "graph_too_small"
    if graph_diag.degenerate or graph_diag.num_edges == 0:
        return "graph_degenerate"
    if graph_diag.singleton_fraction > singleton_fraction_fail:
        return "graph_singleton_fraction"
    if flow_diag.nan_or_inf_seen:
        return "flow_nan_or_inf"
    if flow_diag.clamp_activation_ratio > clamp_activation_fail:
        return "flow_clamp_activation"
    current_ess = float(ess_ratio(p_flow).detach().cpu())
    if current_ess < ess_min_ratio:
        return "flow_low_ess"
    max_allowed = float(max_p_factor_fail) / num_nodes
    if float(p_flow.max().detach().cpu()) > max_allowed:
        return "flow_max_weight"
    return None
