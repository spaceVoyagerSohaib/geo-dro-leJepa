"""Regression tests for the repaired detached GeoDRO graph-flow solver."""

from __future__ import annotations

import pytest
import torch

from stable_pretraining.geodro_lejepa.flow import solve_graph_flow
from stable_pretraining.geodro_lejepa.gating import reliability_gated_weights
from stable_pretraining.geodro_lejepa.graph import build_graph
from stable_pretraining.geodro_lejepa.loss import CoherentHardnessGeoDROLeJEPALoss
from stable_pretraining.geodro_lejepa.types import GraphMode


pytestmark = pytest.mark.unit


def _fully_connected_graph(num_nodes: int):
    return build_graph(torch.eye(num_nodes), mode=GraphMode.FULLY_CONNECTED, k=2)


def test_repaired_flow_ascends_without_routine_clamping():
    graph = _fully_connected_graph(32)
    utility = torch.linspace(-1.0, 1.0, 32)

    weights, diagnostics = solve_graph_flow(
        utility,
        graph,
        inner_steps=20,
        beta=0.2,
        tau_flow=0.025,
    )

    assert torch.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert not diagnostics.fell_back_to_uniform
    assert diagnostics.clamp_activation_ratio == pytest.approx(0.0)
    assert diagnostics.completed_horizon == pytest.approx(diagnostics.requested_horizon)
    assert diagnostics.raw_utility_gain > 0.0
    assert diagnostics.regularized_objective_gain >= 0.0


def test_repaired_flow_is_stable_under_step_refinement():
    graph = _fully_connected_graph(32)
    utility = torch.linspace(-1.0, 1.0, 32)

    coarse, coarse_diag = solve_graph_flow(
        utility, graph, inner_steps=20, beta=0.2, tau_flow=0.025
    )
    refined, refined_diag = solve_graph_flow(
        utility, graph, inner_steps=80, beta=0.2, tau_flow=0.025
    )

    assert not coarse_diag.fell_back_to_uniform
    assert not refined_diag.fell_back_to_uniform
    assert coarse_diag.regularized_objective_gain >= 0.0
    assert refined_diag.regularized_objective_gain >= 0.0
    assert torch.allclose(coarse, refined, atol=2e-3, rtol=2e-3)


def test_repaired_flow_preserves_uniform_weights_for_uniform_utility():
    graph = _fully_connected_graph(32)
    utility = torch.full((32,), 0.5)

    weights, diagnostics = solve_graph_flow(
        utility, graph, inner_steps=20, beta=0.2, tau_flow=0.025
    )

    assert not diagnostics.fell_back_to_uniform
    assert torch.allclose(weights, torch.ones_like(weights) / weights.numel())
    assert diagnostics.raw_utility_gain == pytest.approx(0.0, abs=1e-7)
    assert diagnostics.regularized_objective_gain == pytest.approx(0.0, abs=1e-7)


def test_repaired_flow_ascends_noise_at_short_calibration_horizon():
    torch.manual_seed(19)
    graph = _fully_connected_graph(64)
    utility = torch.randn(64)

    weights, diagnostics = solve_graph_flow(
        utility,
        graph,
        inner_steps=20,
        beta=0.2,
        tau_flow=0.00025,
    )

    assert not diagnostics.fell_back_to_uniform
    assert torch.all(weights >= 0)
    assert diagnostics.clamp_activation_ratio == pytest.approx(0.0)
    assert diagnostics.raw_utility_gain > 0.0
    assert diagnostics.regularized_objective_gain >= 0.0


def test_repaired_flow_is_permutation_equivariant():
    graph = _fully_connected_graph(16)
    utility = torch.linspace(-1.0, 1.0, 16)
    permutation = torch.tensor([3, 10, 0, 15, 1, 9, 4, 14, 2, 11, 5, 13, 6, 12, 7, 8])

    weights, diagnostics = solve_graph_flow(
        utility, graph, inner_steps=40, beta=0.2, tau_flow=0.025
    )
    permuted_weights, permuted_diagnostics = solve_graph_flow(
        utility[permutation],
        _fully_connected_graph(16),
        inner_steps=40,
        beta=0.2,
        tau_flow=0.025,
    )

    inverse = torch.argsort(permutation)
    assert not diagnostics.fell_back_to_uniform
    assert not permuted_diagnostics.fell_back_to_uniform
    assert torch.allclose(weights, permuted_weights[inverse], atol=1e-6)


def test_repaired_flow_fails_closed_when_substep_budget_is_insufficient():
    graph = _fully_connected_graph(32)
    utility = torch.linspace(-1.0, 1.0, 32)

    weights, diagnostics = solve_graph_flow(
        utility,
        graph,
        inner_steps=20,
        beta=0.2,
        tau_flow=0.25,
        max_substeps=20,
    )

    assert diagnostics.fell_back_to_uniform
    assert diagnostics.completed_horizon < diagnostics.requested_horizon
    assert diagnostics.raw_utility_gain == pytest.approx(0.0)
    assert torch.allclose(weights, torch.ones_like(weights) / weights.numel())

    training_weights, weight_diagnostics = reliability_gated_weights(
        weights,
        graph.diagnostics,
        diagnostics,
        alpha_max=0.03,
        ess_min_ratio=0.0,
        clamp_activation_fail=1.0,
        max_p_factor_fail=1_000.0,
    )

    assert weight_diagnostics.fallback
    assert weight_diagnostics.fallback_reason == "flow_solver_fallback"
    assert torch.allclose(training_weights, weights)


def test_loss_exposes_solver_budget_and_tolerance_in_config_logs():
    loss = CoherentHardnessGeoDROLeJEPALoss(
        max_flow_substeps=4096,
        flow_objective_tolerance=1e-6,
    )

    logs = loss._config_logs()

    assert logs["Config/flow_max_substeps"] == pytest.approx(4096.0)
    assert logs["Config/flow_objective_tolerance"] == pytest.approx(1e-6)
