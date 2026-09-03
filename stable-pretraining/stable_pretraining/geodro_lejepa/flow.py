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
    max_substeps: int | None = None,
    objective_tolerance: float = 1e-7,
) -> tuple[torch.Tensor, FlowDiagnostics]:
    """Solve detached graph flow with positivity and objective backtracking.

    ``tau_flow * num_nodes`` is the requested physical flow horizon.  The
    ``inner_steps`` argument supplies an initial proposed step, not a fixed
    discretisation: a proposal is repeatedly halved until it preserves the
    simplex interior and does not decrease the entropy-regularized objective.
    If the requested horizon cannot be completed within ``max_substeps``, the
    solver fails closed to uniform instead of relying on clamp projection.
    """
    utility = utility.detach().float()
    num_nodes = utility.numel()
    uniform = torch.ones_like(utility) / max(num_nodes, 1)
    if num_nodes == 0:
        return uniform, _diagnostics(uniform, fell_back=True)

    if inner_steps <= 0 or tau_flow <= 0 or graph.edge_weight.numel() == 0:
        return uniform, _diagnostics(
            uniform,
            fell_back=inner_steps <= 0 or tau_flow < 0,
            min_p_before_clamp=float(uniform.min().detach().cpu()),
        )

    p = uniform.clone()
    requested_horizon = float(tau_flow) * num_nodes
    nominal_dt = requested_horizon / int(inner_steps)
    max_substeps = max(int(inner_steps), int(max_substeps or 8192))
    min_dt = requested_horizon / max_substeps
    remaining_horizon = requested_horizon
    time_tolerance = max(requested_horizon * 1e-9, 1e-12)
    dt = nominal_dt
    row, col = graph.edge_index
    weight = graph.edge_weight.to(device=utility.device, dtype=utility.dtype)
    min_p_before_clamp = float(uniform.min().detach().cpu())
    nan_or_inf_seen = False
    accepted_substeps = 0
    rejected_substeps = 0
    minimum_accepted_dt = float("inf")
    completed_horizon = 0.0
    current_objective = _regularized_objective(p, utility, beta, eps_log)
    initial_objective = current_objective
    fell_back = False

    while remaining_horizon > time_tolerance:
        if accepted_substeps >= max_substeps:
            fell_back = True
            break

        dt = min(dt, remaining_horizon)
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

        min_p_before_clamp = min(min_p_before_clamp, float(next_p.min().detach().cpu()))

        if not torch.isfinite(next_p).all():
            nan_or_inf_seen = True
            fell_back = True
            break

        p_sum = next_p.sum()
        if not torch.isfinite(p_sum) or p_sum <= 0:
            nan_or_inf_seen = True
            fell_back = True
            break

        # The edge update is conservative. Normalization only removes
        # roundoff-scale drift and occurs before the objective check.
        next_p = next_p / p_sum
        next_objective = _regularized_objective(next_p, utility, beta, eps_log)
        preserves_positivity = bool((next_p >= float(p_floor)).all().detach().cpu())
        ascends_objective = bool(
            (next_objective >= current_objective - float(objective_tolerance))
            .detach()
            .cpu()
        )
        if preserves_positivity and ascends_objective:
            p = next_p
            current_objective = next_objective
            remaining_horizon -= dt
            completed_horizon += dt
            accepted_substeps += 1
            minimum_accepted_dt = min(minimum_accepted_dt, dt)
            continue

        rejected_substeps += 1
        dt *= 0.5
        if dt < min_dt:
            fell_back = True
            break

    if fell_back or nan_or_inf_seen:
        p = uniform

    diagnostics = _diagnostics(
        p,
        clamp_activation_ratio=0.0,
        nan_or_inf_seen=nan_or_inf_seen,
        min_p_before_clamp=min_p_before_clamp,
        flow_num_steps=accepted_substeps,
        fell_back=fell_back,
        accepted_substeps=accepted_substeps,
        rejected_substeps=rejected_substeps,
        minimum_accepted_dt=(
            0.0 if not math.isfinite(minimum_accepted_dt) else minimum_accepted_dt
        ),
        requested_horizon=requested_horizon,
        completed_horizon=completed_horizon,
        raw_utility_gain=float(
            torch.dot(p, utility).sub(torch.dot(uniform, utility)).detach().cpu()
        ),
        regularized_objective_gain=float(
            current_objective.sub(initial_objective).detach().cpu()
        )
        if not fell_back
        else 0.0,
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


def _regularized_objective(
    weights: torch.Tensor,
    utility: torch.Tensor,
    beta: float,
    eps_log: float,
) -> torch.Tensor:
    return torch.dot(weights, utility) + float(beta) * entropy(weights, eps=eps_log)


def _diagnostics(
    weights: torch.Tensor,
    *,
    clamp_activation_ratio: float = 0.0,
    nan_or_inf_seen: bool = False,
    min_p_before_clamp: float | None = None,
    flow_num_steps: int = 0,
    fell_back: bool = False,
    accepted_substeps: int = 0,
    rejected_substeps: int = 0,
    minimum_accepted_dt: float = 0.0,
    requested_horizon: float = 0.0,
    completed_horizon: float = 0.0,
    raw_utility_gain: float = 0.0,
    regularized_objective_gain: float = 0.0,
) -> FlowDiagnostics:
    if min_p_before_clamp is None or not math.isfinite(min_p_before_clamp):
        min_p_before_clamp = (
            float(weights.min().detach().cpu()) if weights.numel() else 0.0
        )
    return FlowDiagnostics(
        clamp_activation_ratio=float(clamp_activation_ratio),
        nan_or_inf_seen=bool(nan_or_inf_seen),
        min_p_before_clamp=float(min_p_before_clamp),
        max_p=float(weights.max().detach().cpu()) if weights.numel() else 0.0,
        entropy=float(entropy(weights).detach().cpu()) if weights.numel() else 0.0,
        ess_ratio=float(ess_ratio(weights).detach().cpu()) if weights.numel() else 0.0,
        flow_num_steps=int(flow_num_steps),
        fell_back_to_uniform=bool(fell_back),
        accepted_substeps=int(accepted_substeps),
        rejected_substeps=int(rejected_substeps),
        minimum_accepted_dt=float(minimum_accepted_dt),
        requested_horizon=float(requested_horizon),
        completed_horizon=float(completed_horizon),
        raw_utility_gain=float(raw_utility_gain),
        regularized_objective_gain=float(regularized_objective_gain),
    )
