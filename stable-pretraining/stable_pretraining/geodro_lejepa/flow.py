"""Finite-time upwind graph-flow solver for GeoDRO-LeJEPA."""

from __future__ import annotations

import math

import torch

from .types import FlowDiagnostics, GraphData


def solve_graph_flow(
    utility: torch.Tensor,
    graph: GraphData,
    *,
    inner_steps: int = 10,
    beta: float = 0.2,
    tau_flow: float = 0.05,
    p_floor: float = 1e-12,
    eps_log: float = 1e-12,
) -> tuple[torch.Tensor, FlowDiagnostics]:
    utility = utility.detach().float()
    num_nodes = utility.numel()
    uniform = torch.ones_like(utility) / max(num_nodes, 1)
    if num_nodes == 0:
        return uniform, _diagnostics(uniform, 0, 0.0, False, 0.0, fell_back=True)

    if inner_steps <= 0 or tau_flow == 0 or graph.edge_weight.numel() == 0:
        return uniform, _diagnostics(uniform, 0, 0.0, False, float(uniform.min()))

    p = uniform.clone()
    dt = float(tau_flow) * num_nodes / int(inner_steps)
    row, col = graph.edge_index
    weight = graph.edge_weight.to(device=utility.device, dtype=utility.dtype)
    clamp_hits = 0
    total_entries = 0
    min_p_before_clamp = float("inf")
    nan_or_inf_seen = False
    steps_done = 0

    for _ in range(int(inner_steps)):
        log_p = torch.log(p.clamp_min(eps_log))
        delta_ij = (utility[row] - utility[col]) + float(beta) * (
            log_p[col] - log_p[row]
        )
        kappa = torch.where(delta_ij > 0, p[col], p[row])
        edge_flow = weight * kappa * delta_ij

        update = torch.zeros_like(p)
        update.scatter_add_(0, row, edge_flow)
        update.scatter_add_(0, col, -edge_flow)
        next_p = p + dt * update

        min_p_before_clamp = min(
            min_p_before_clamp, float(next_p.min().detach().cpu())
        )
        clamp_hits += int((next_p < p_floor).sum().detach().cpu())
        total_entries += int(next_p.numel())

        if not torch.isfinite(next_p).all():
            nan_or_inf_seen = True
            break

        p = next_p.clamp_min(p_floor)
        p_sum = p.sum()
        if not torch.isfinite(p_sum) or p_sum <= 0:
            nan_or_inf_seen = True
            break
        p = p / p_sum
        steps_done += 1

    if nan_or_inf_seen:
        p = uniform

    clamp_activation_ratio = clamp_hits / max(total_entries, 1)
    diagnostics = _diagnostics(
        p,
        steps_done,
        clamp_activation_ratio,
        nan_or_inf_seen,
        min_p_before_clamp,
        fell_back=nan_or_inf_seen,
    )
    return p.detach(), diagnostics


def entropy(weights: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return -(weights.clamp_min(eps) * torch.log(weights.clamp_min(eps))).sum()


def ess_ratio(weights: torch.Tensor) -> torch.Tensor:
    num_nodes = weights.numel()
    if num_nodes == 0:
        return torch.tensor(0.0, device=weights.device)
    ess = 1.0 / weights.square().sum().clamp_min(torch.finfo(weights.dtype).eps)
    return ess / num_nodes


def _diagnostics(
    weights: torch.Tensor,
    steps_done: int,
    clamp_activation_ratio: float,
    nan_or_inf_seen: bool,
    min_p_before_clamp: float,
    *,
    fell_back: bool = False,
) -> FlowDiagnostics:
    if not math.isfinite(min_p_before_clamp):
        min_p_before_clamp = float(weights.min().detach().cpu()) if weights.numel() else 0.0
    return FlowDiagnostics(
        clamp_activation_ratio=float(clamp_activation_ratio),
        nan_or_inf_seen=bool(nan_or_inf_seen),
        min_p_before_clamp=float(min_p_before_clamp),
        max_p=float(weights.max().detach().cpu()) if weights.numel() else 0.0,
        entropy=float(entropy(weights).detach().cpu()) if weights.numel() else 0.0,
        ess_ratio=float(ess_ratio(weights).detach().cpu()) if weights.numel() else 0.0,
        flow_num_steps=int(steps_done),
        fell_back_to_uniform=bool(fell_back),
    )
