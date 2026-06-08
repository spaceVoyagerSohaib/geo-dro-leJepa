"""Detached graph construction for GeoDRO-LeJEPA."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from .prediction import select_global_views
from .types import (
    GraphData,
    GraphDiagnostics,
    GraphDistanceMetric,
    GraphMode,
    GraphSpace,
    MemoryWitnessOverlapScores,
    MemoryWitnessThresholdMode,
)


@dataclass(frozen=True)
class MemoryWitnessedGraphResult:
    """Current-only graph plus diagnostics for memory-added edges."""

    graph: GraphData
    logs: dict[str, float]
    selected_edges: torch.Tensor
    raw_selected_edges: torch.Tensor
    specificity_selected_edges: torch.Tensor
    specificity_removed_edges: torch.Tensor


@dataclass(frozen=True)
class _MemoryEdgeCandidates:
    """Upper-triangular candidate edges eligible for memory witnessing."""

    row: torch.Tensor
    col: torch.Tensor

    @property
    def numel(self) -> int:
        return int(self.row.numel())


def prepare_graph_features(
    emb: torch.Tensor,
    proj: torch.Tensor,
    *,
    global_mask: torch.Tensor | None = None,
    global_view_count: int | None = None,
    graph_space: GraphSpace | str = GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER,
    normalized: bool = True,
) -> torch.Tensor:
    """Build detached graph features in the configured comparable metric space."""
    if emb.ndim != 3 or proj.ndim != 3:
        raise ValueError("Expected emb and proj with shapes [V, B, H/K].")

    graph_space = GraphSpace(graph_space)
    if graph_space == GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER:
        features = select_global_views(
            emb, global_mask=global_mask, global_view_count=global_view_count
        ).mean(dim=0)
    elif graph_space == GraphSpace.PROJECTOR_GLOBAL_CENTER:
        features = select_global_views(
            proj, global_mask=global_mask, global_view_count=global_view_count
        ).mean(dim=0)
    elif graph_space == GraphSpace.CONSENSUS_PREPROJ_PROJECTOR:
        emb_center = select_global_views(
            emb, global_mask=global_mask, global_view_count=global_view_count
        ).mean(dim=0)
        proj_center = select_global_views(
            proj, global_mask=global_mask, global_view_count=global_view_count
        ).mean(dim=0)
        features = torch.cat(
            [F.normalize(emb_center, dim=-1), F.normalize(proj_center, dim=-1)], dim=-1
        )
    elif graph_space == GraphSpace.RANDOM_FEATURES:
        features = torch.randn_like(emb[0])
    else:
        raise ValueError(f"Unsupported graph_space: {graph_space}")

    features = features.detach()
    if normalized:
        features = F.normalize(features, dim=-1)
    return features


def build_graph_features(
    emb: torch.Tensor,
    proj: torch.Tensor,
    *,
    global_mask: torch.Tensor | None = None,
    global_view_count: int | None = None,
    graph_space: GraphSpace | str = GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER,
) -> torch.Tensor:
    return prepare_graph_features(
        emb,
        proj,
        global_mask=global_mask,
        global_view_count=global_view_count,
        graph_space=graph_space,
        normalized=True,
    )


def build_graph(
    features: torch.Tensor,
    *,
    mode: GraphMode | str = GraphMode.MUTUAL_KNN,
    distance_metric: GraphDistanceMetric | str = GraphDistanceMetric.COSINE,
    k: int = 8,
    eps: float = 1e-12,
) -> GraphData:
    if features.ndim != 2:
        raise ValueError(
            f"Expected graph features [N, D], got {tuple(features.shape)}."
        )

    mode = GraphMode(mode)
    distance_metric = GraphDistanceMetric(distance_metric)
    num_nodes = features.shape[0]
    device = features.device
    if mode == GraphMode.NO_GRAPH_KL:
        raise NotImplementedError(
            "GeoDRO-LeJEPA no_graph_kl is reserved for the future KL-DRO "
            "ablation and is not implemented in the v1.1 core path."
        )
    if num_nodes <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0,), dtype=features.dtype, device=device)
        diagnostics = _diagnostics(num_nodes, edge_index, edge_weight, None, mode)
        return GraphData(num_nodes, edge_index, edge_weight, diagnostics)

    distances, sigma, knn, k_eff = _current_geometry(
        features,
        distance_metric=distance_metric,
        k=k,
        eps=eps,
    )
    adjacency = _base_adjacency(
        knn,
        mode=mode,
        k_eff=k_eff,
        device=device,
    )
    return _graph_from_adjacency(
        num_nodes,
        adjacency,
        distances=distances,
        sigma=sigma,
        dtype=features.dtype,
        mode=mode,
        eps=eps,
    )


def build_memory_witnessed_graph(
    features: torch.Tensor,
    overlap_scores: MemoryWitnessOverlapScores,
    *,
    mode: GraphMode | str = GraphMode.MUTUAL_KNN,
    distance_metric: GraphDistanceMetric | str = GraphDistanceMetric.COSINE,
    k: int = 8,
    memory_k_guard: int = 64,
    memory_witness_score_min: float | None = None,
    memory_extra_edges_per_node_max: int = 2,
    memory_added_edge_ratio_max: float = 0.25,
    fill_ratio: float = 0.0,
    memory_min_fill_ratio: float = 0.25,
    train_warmup: float = 1.0,
    memory_witness_threshold_mode: MemoryWitnessThresholdMode | str = (
        MemoryWitnessThresholdMode.EXPLICIT
    ),
    memory_witness_null_quantile: float = 0.95,
    null_overlap_scores: MemoryWitnessOverlapScores | None = None,
    eps: float = 1e-12,
    return_result: bool = False,
) -> tuple[GraphData, dict[str, float]] | MemoryWitnessedGraphResult:
    """Build a current-only graph with optional memory-witnessed edge additions."""
    build_start = time.perf_counter()
    if features.ndim != 2:
        raise ValueError(
            f"Expected graph features [N, D], got {tuple(features.shape)}."
        )

    mode = GraphMode(mode)
    distance_metric = GraphDistanceMetric(distance_metric)
    threshold_mode = MemoryWitnessThresholdMode(memory_witness_threshold_mode)
    num_nodes = int(features.shape[0])
    device = features.device
    if mode == GraphMode.NO_GRAPH_KL:
        raise NotImplementedError(
            "GeoDRO-LeJEPA no_graph_kl is reserved for the future KL-DRO "
            "ablation and is not implemented in the v1.1 core path."
        )
    if num_nodes <= 1:
        graph = build_graph(
            features,
            mode=mode,
            distance_metric=distance_metric,
            k=k,
            eps=eps,
        )
        logs = _memory_edge_logs(
            num_nodes=num_nodes,
            batch_edges=0,
            added_edges_before_budget=0,
            added_edges_after_budget=0,
            memory_k_guard=memory_k_guard,
            memory_extra_edges_per_node_max=memory_extra_edges_per_node_max,
            memory_added_edge_ratio_max=memory_added_edge_ratio_max,
            fill_ratio=fill_ratio,
            memory_min_fill_ratio=memory_min_fill_ratio,
            train_warmup=train_warmup,
            memory_witness_score_min=memory_witness_score_min,
            threshold_mode=threshold_mode,
        )
        logs["Memory/graph_build_time_ms"] = _elapsed_ms(build_start)
        return _memory_graph_return(
            MemoryWitnessedGraphResult(
                graph=graph,
                logs=logs,
                selected_edges=_empty_edges(device),
                raw_selected_edges=_empty_edges(device),
                specificity_selected_edges=_empty_edges(device),
                specificity_removed_edges=_empty_edges(device),
            ),
            return_result=return_result,
        )

    distances, sigma, knn, k_eff = _current_geometry(
        features,
        distance_metric=distance_metric,
        k=k,
        eps=eps,
    )
    base_adjacency = _base_adjacency(
        knn,
        mode=mode,
        k_eff=k_eff,
        device=device,
    )
    batch_edges = int(torch.triu(base_adjacency, diagonal=1).sum().detach().cpu())
    edge_logs = _memory_edge_logs(
        num_nodes=num_nodes,
        batch_edges=batch_edges,
        added_edges_before_budget=0,
        added_edges_after_budget=0,
        memory_k_guard=memory_k_guard,
        memory_extra_edges_per_node_max=memory_extra_edges_per_node_max,
        memory_added_edge_ratio_max=memory_added_edge_ratio_max,
        fill_ratio=fill_ratio,
        memory_min_fill_ratio=memory_min_fill_ratio,
        train_warmup=train_warmup,
        memory_witness_score_min=memory_witness_score_min,
        threshold_mode=threshold_mode,
    )
    witnessed_adjacency = torch.zeros_like(base_adjacency)
    raw_adjacency = torch.zeros_like(base_adjacency)
    specificity_adjacency = torch.zeros_like(base_adjacency)
    selected_overlap = overlap_scores.selected_overlap.detach().float()
    raw_overlap = overlap_scores.raw_overlap.detach().float()
    specificity_overlap = overlap_scores.specificity_overlap.detach().float()
    if (
        overlap_scores.valid_for_scoring
        and selected_overlap.shape == (num_nodes, num_nodes)
        and edge_logs["MemoryWitness/extra_edges_per_node_budget_eff"] > 0
        and edge_logs["MemoryWitness/global_added_edge_cap_eff"] > 0
        and edge_logs["MemoryWitness/K_guard_eff"] > 0
    ):
        candidates = _candidate_edge_indices(
            distances=distances,
            base_adjacency=base_adjacency,
            memory_k_guard=int(edge_logs["MemoryWitness/K_guard_eff"]),
        )
        threshold_value, threshold_logs = _resolve_witness_threshold(
            candidates=candidates,
            expected_shape=base_adjacency.shape,
            null_overlap_scores=null_overlap_scores,
            memory_witness_score_min=memory_witness_score_min,
            threshold_mode=threshold_mode,
            memory_witness_null_quantile=memory_witness_null_quantile,
        )
        edge_logs |= threshold_logs
        witnessed_adjacency, edge_logs = _select_memory_edges(
            distances=distances,
            sigma=sigma,
            base_adjacency=base_adjacency,
            candidates=candidates,
            selected_overlap=selected_overlap,
            memory_witness_score_min=threshold_value,
            memory_extra_edges_per_node_max=int(
                edge_logs["MemoryWitness/extra_edges_per_node_budget_eff"]
            ),
            memory_added_edge_cap=int(
                edge_logs["MemoryWitness/global_added_edge_cap_eff"]
            ),
            edge_logs=edge_logs,
            eps=eps,
        )
        if threshold_value is not None:
            raw_adjacency = _select_diagnostic_memory_edges(
                distances=distances,
                sigma=sigma,
                base_adjacency=base_adjacency,
                candidates=candidates,
                selected_adjacency=witnessed_adjacency,
                selected_overlap=selected_overlap,
                diagnostic_overlap=raw_overlap,
                memory_witness_score_min=threshold_value,
                memory_extra_edges_per_node_max=int(
                    edge_logs["MemoryWitness/extra_edges_per_node_budget_eff"]
                ),
                memory_added_edge_cap=int(
                    edge_logs["MemoryWitness/global_added_edge_cap_eff"]
                ),
                edge_logs=edge_logs,
                eps=eps,
            )
            specificity_adjacency = _select_diagnostic_memory_edges(
                distances=distances,
                sigma=sigma,
                base_adjacency=base_adjacency,
                candidates=candidates,
                selected_adjacency=witnessed_adjacency,
                selected_overlap=selected_overlap,
                diagnostic_overlap=specificity_overlap,
                memory_witness_score_min=threshold_value,
                memory_extra_edges_per_node_max=int(
                    edge_logs["MemoryWitness/extra_edges_per_node_budget_eff"]
                ),
                memory_added_edge_cap=int(
                    edge_logs["MemoryWitness/global_added_edge_cap_eff"]
                ),
                edge_logs=edge_logs,
                eps=eps,
            )

    final_adjacency = base_adjacency | witnessed_adjacency
    graph = _graph_from_adjacency(
        num_nodes,
        final_adjacency,
        distances=distances,
        sigma=sigma,
        dtype=features.dtype,
        mode=mode,
        eps=eps,
    )
    final_edges = int(graph.edge_weight.numel())
    edge_logs["Graph/final_edges"] = float(final_edges)
    edge_logs["Graph/memory_added_edges"] = float(max(final_edges - batch_edges, 0))
    selected_edges = _edge_tensor_from_adjacency(witnessed_adjacency)
    raw_edges = (
        selected_edges
        if raw_adjacency is witnessed_adjacency
        else _edge_tensor_from_adjacency(raw_adjacency)
    )
    specificity_edges = (
        selected_edges
        if specificity_adjacency is witnessed_adjacency
        else _edge_tensor_from_adjacency(specificity_adjacency)
    )
    specificity_removed_edges = _edge_difference(raw_edges, specificity_edges)
    edge_logs |= _memory_edge_comparison_logs(
        selected_edges=selected_edges,
        raw_selected_edges=raw_edges,
        specificity_selected_edges=specificity_edges,
        specificity_removed_edges=specificity_removed_edges,
        distances=distances,
        num_nodes=num_nodes,
        global_cap=int(edge_logs["MemoryWitness/global_added_edge_cap_eff"]),
    )
    edge_logs["Memory/graph_build_time_ms"] = _elapsed_ms(build_start)
    return _memory_graph_return(
        MemoryWitnessedGraphResult(
            graph=graph,
            logs=edge_logs,
            selected_edges=selected_edges,
            raw_selected_edges=raw_edges,
            specificity_selected_edges=specificity_edges,
            specificity_removed_edges=specificity_removed_edges,
        ),
        return_result=return_result,
    )


def graph_neighbor_average(values: torch.Tensor, graph: GraphData) -> torch.Tensor:
    out = torch.zeros_like(values)
    denom = torch.zeros_like(values)
    if graph.edge_weight.numel() == 0:
        return out

    row, col = graph.edge_index
    weight = graph.edge_weight.to(values.dtype)
    out.scatter_add_(0, row, weight * values[col])
    out.scatter_add_(0, col, weight * values[row])
    denom.scatter_add_(0, row, weight)
    denom.scatter_add_(0, col, weight)
    return torch.where(
        denom > 0, out / denom.clamp_min(torch.finfo(values.dtype).eps), out
    )


def graph_dirichlet_energy(values: torch.Tensor, graph: GraphData) -> float:
    if graph.edge_weight.numel() == 0:
        return 0.0
    row, col = graph.edge_index
    energy = graph.edge_weight.to(values.dtype) * (values[row] - values[col]).square()
    return float(energy.sum().detach().cpu())


def _current_geometry(
    features: torch.Tensor,
    *,
    distance_metric: GraphDistanceMetric,
    k: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_nodes = int(features.shape[0])
    distances = _pairwise_distances(features, metric=distance_metric)
    distances.fill_diagonal_(float("inf"))
    k_eff = max(1, min(int(k), num_nodes - 1))
    knn_dist, knn_idx = torch.topk(distances, k=k_eff, dim=1, largest=False)
    sigma = knn_dist[:, -1].clamp_min(eps)
    knn = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=features.device)
    knn.scatter_(1, knn_idx, True)
    return distances, sigma, knn, k_eff


def _base_adjacency(
    knn: torch.Tensor,
    *,
    mode: GraphMode,
    k_eff: int,
    device: torch.device,
) -> torch.Tensor:
    num_nodes = int(knn.shape[0])
    if mode == GraphMode.MUTUAL_KNN:
        return knn & knn.T
    if mode == GraphMode.MAX_UNION_KNN:
        return knn | knn.T
    if mode == GraphMode.FULLY_CONNECTED:
        adjacency = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=device)
        adjacency.fill_diagonal_(False)
        return adjacency
    if mode == GraphMode.RANDOM_REGULAR:
        return _random_symmetric_knn(num_nodes, k_eff, device)
    raise ValueError(f"Unsupported graph mode: {mode}")


def _graph_from_adjacency(
    num_nodes: int,
    adjacency: torch.Tensor,
    *,
    distances: torch.Tensor,
    sigma: torch.Tensor,
    dtype: torch.dtype,
    mode: GraphMode,
    eps: float,
) -> GraphData:
    row, col = torch.triu(adjacency, diagonal=1).nonzero(as_tuple=True)
    if row.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=adjacency.device)
        edge_weight = torch.empty((0,), dtype=dtype, device=adjacency.device)
    else:
        edge_index = torch.stack([row, col])
        edge_weight = _current_edge_weights(
            distances,
            sigma,
            row,
            col,
            dtype=dtype,
            eps=eps,
        )

    diagnostics = _diagnostics(num_nodes, edge_index, edge_weight, sigma, mode)
    return GraphData(num_nodes, edge_index, edge_weight, diagnostics)


def _current_edge_weights(
    distances: torch.Tensor,
    sigma: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    *,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    if row.numel() == 0:
        return torch.empty((0,), dtype=dtype, device=distances.device)
    edge_dist = distances[row, col].clamp_min(0.0)
    denom = (sigma[row] * sigma[col]).clamp_min(eps)
    return torch.exp(-(edge_dist.square()) / denom).to(dtype=dtype)


def _memory_edge_logs(
    *,
    num_nodes: int,
    batch_edges: int,
    added_edges_before_budget: int,
    added_edges_after_budget: int,
    memory_k_guard: int,
    memory_extra_edges_per_node_max: int,
    memory_added_edge_ratio_max: float,
    fill_ratio: float,
    memory_min_fill_ratio: float,
    train_warmup: float,
    memory_witness_score_min: float | None,
    threshold_mode: MemoryWitnessThresholdMode,
) -> dict[str, float]:
    k_guard_eff = max(0, min(int(memory_k_guard), max(int(num_nodes) - 1, 0)))
    k_guard_fraction = k_guard_eff / max(float(num_nodes - 1), 1.0)
    fill_ramp = _fill_ramp(
        fill_ratio=fill_ratio,
        memory_min_fill_ratio=memory_min_fill_ratio,
    )
    train_warmup = float(max(0.0, min(float(train_warmup), 1.0)))
    budget_scale = fill_ramp * train_warmup
    budget_eff = int(
        torch.floor(
            torch.tensor(
                float(memory_extra_edges_per_node_max) * budget_scale,
                dtype=torch.float64,
            )
        ).item()
    )
    global_cap = int(
        torch.floor(
            torch.tensor(
                float(memory_added_edge_ratio_max) * float(batch_edges),
                dtype=torch.float64,
            )
        ).item()
    )
    threshold_value = (
        -1.0 if memory_witness_score_min is None else float(memory_witness_score_min)
    )
    return {
        "MemoryWitness/K_guard_eff": float(k_guard_eff),
        "MemoryWitness/K_guard_fraction": float(k_guard_fraction),
        f"MemoryWitness/threshold_mode/{threshold_mode.value}": 1.0,
        "MemoryWitness/threshold_value": float(threshold_value),
        "MemoryWitness/null_score_mean": 0.0,
        "MemoryWitness/null_score_p95": 0.0,
        "MemoryWitness/null_score_p99": 0.0,
        "MemoryWitness/added_edges_before_budget": float(added_edges_before_budget),
        "MemoryWitness/added_edges_after_budget": float(added_edges_after_budget),
        "MemoryWitness/added_edges_dropped_by_budget": float(
            max(added_edges_before_budget - added_edges_after_budget, 0)
        ),
        "MemoryWitness/extra_edges_per_node_budget_eff": float(max(budget_eff, 0)),
        "MemoryWitness/global_added_edge_cap_eff": float(max(global_cap, 0)),
        "MemoryWitness/fill_ramp": float(fill_ramp),
        "MemoryWitness/train_warmup": float(train_warmup),
        "MemoryWitness/budget_scale": float(budget_scale),
        "Graph/batch_edges": float(batch_edges),
        "Graph/final_edges": float(batch_edges + added_edges_after_budget),
        "Graph/memory_added_edges": float(added_edges_after_budget),
    }


def _resolve_witness_threshold(
    *,
    candidates: _MemoryEdgeCandidates,
    expected_shape: torch.Size,
    null_overlap_scores: MemoryWitnessOverlapScores | None,
    memory_witness_score_min: float | None,
    threshold_mode: MemoryWitnessThresholdMode,
    memory_witness_null_quantile: float,
) -> tuple[float | None, dict[str, float]]:
    if threshold_mode == MemoryWitnessThresholdMode.EXPLICIT:
        value = (
            None
            if memory_witness_score_min is None
            else float(memory_witness_score_min)
        )
        return value, {
            "MemoryWitness/threshold_value": -1.0 if value is None else value,
            "MemoryWitness/null_score_mean": 0.0,
            "MemoryWitness/null_score_p95": 0.0,
            "MemoryWitness/null_score_p99": 0.0,
        }
    if threshold_mode != MemoryWitnessThresholdMode.SHUFFLED_NULL_QUANTILE:
        raise ValueError(f"Unsupported witness threshold mode: {threshold_mode.value}.")

    if (
        null_overlap_scores is None
        or not null_overlap_scores.valid_for_scoring
        or null_overlap_scores.selected_overlap.shape != expected_shape
        or candidates.numel == 0
    ):
        return 1.0, {
            "MemoryWitness/threshold_value": 1.0,
            "MemoryWitness/null_score_mean": 0.0,
            "MemoryWitness/null_score_p95": 0.0,
            "MemoryWitness/null_score_p99": 0.0,
        }

    null_scores = null_overlap_scores.selected_overlap.to(device=candidates.row.device)[
        candidates.row, candidates.col
    ]
    null_scores = null_scores.detach().float()
    threshold = float(
        torch.quantile(
            null_scores,
            float(memory_witness_null_quantile),
        )
        .detach()
        .cpu()
    )
    return threshold, {
        "MemoryWitness/threshold_value": threshold,
        "MemoryWitness/null_score_mean": _float_mean(null_scores),
        "MemoryWitness/null_score_p95": _float_quantile(null_scores, 0.95),
        "MemoryWitness/null_score_p99": _float_quantile(null_scores, 0.99),
    }


def _candidate_edge_indices(
    *,
    distances: torch.Tensor,
    base_adjacency: torch.Tensor,
    memory_k_guard: int,
) -> _MemoryEdgeCandidates:
    num_nodes = int(base_adjacency.shape[0])
    k_guard_eff = min(int(memory_k_guard), max(num_nodes - 1, 0))
    if k_guard_eff <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=base_adjacency.device)
        return _MemoryEdgeCandidates(row=empty, col=empty)
    _, guard_idx = torch.topk(distances, k=k_guard_eff, dim=1, largest=False)
    row = torch.arange(num_nodes, device=base_adjacency.device).unsqueeze(1)
    row = row.expand(-1, k_guard_eff).reshape(-1)
    col = guard_idx.reshape(-1)
    left = torch.minimum(row, col)
    right = torch.maximum(row, col)
    keep = left != right
    left = left[keep]
    right = right[keep]
    keep = ~base_adjacency[left, right]
    left = left[keep]
    right = right[keep]
    if left.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=base_adjacency.device)
        return _MemoryEdgeCandidates(row=empty, col=empty)

    # Sorted keys preserve the legacy row/column tie-break for stable score sorting.
    keys = torch.unique(left * num_nodes + right, sorted=True)
    return _MemoryEdgeCandidates(
        row=torch.div(keys, num_nodes, rounding_mode="floor"),
        col=torch.remainder(keys, num_nodes),
    )


def _fill_ramp(*, fill_ratio: float, memory_min_fill_ratio: float) -> float:
    denom = max(1.0 - float(memory_min_fill_ratio), torch.finfo(torch.float32).eps)
    ramp = (float(fill_ratio) - float(memory_min_fill_ratio)) / denom
    return float(max(0.0, min(ramp, 1.0)))


def _memory_graph_return(
    result: MemoryWitnessedGraphResult,
    *,
    return_result: bool,
) -> tuple[GraphData, dict[str, float]] | MemoryWitnessedGraphResult:
    if return_result:
        return result
    return result.graph, result.logs


def _empty_edges(device: torch.device) -> torch.Tensor:
    return torch.empty((0, 2), dtype=torch.long, device=device)


def _edge_tensor_from_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    row, col = torch.triu(adjacency, diagonal=1).nonzero(as_tuple=True)
    if row.numel() == 0:
        return _empty_edges(adjacency.device)
    return torch.stack([row, col], dim=1).contiguous()


def _edge_set(edges: torch.Tensor) -> set[tuple[int, int]]:
    if edges.numel() == 0:
        return set()
    return {(int(left), int(right)) for left, right in edges.detach().cpu().tolist()}


def _edge_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    remaining = sorted(_edge_set(left) - _edge_set(right))
    if not remaining:
        return _empty_edges(left.device)
    return torch.tensor(remaining, dtype=torch.long, device=left.device)


def _same_tensor_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.data_ptr() == right.data_ptr()
    )


def _memory_edge_comparison_logs(
    *,
    selected_edges: torch.Tensor,
    raw_selected_edges: torch.Tensor,
    specificity_selected_edges: torch.Tensor,
    specificity_removed_edges: torch.Tensor,
    distances: torch.Tensor,
    num_nodes: int,
    global_cap: int,
) -> dict[str, float]:
    raw_set = _edge_set(raw_selected_edges)
    spec_set = _edge_set(specificity_selected_edges)
    union = raw_set | spec_set
    intersection = raw_set & spec_set
    selected_count = int(selected_edges.shape[0])
    return {
        "MemoryWitness/added_edges": float(selected_count),
        "MemoryWitness/added_edges_per_node_mean": (
            2.0 * selected_count / max(float(num_nodes), 1.0)
        ),
        "MemoryWitness/added_edge_cap_active": float(
            global_cap > 0 and selected_count >= global_cap
        ),
        "MemoryWitness/raw_vs_spec_added_edge_agreement": (
            float(len(intersection) / len(union)) if union else 1.0
        ),
        "MemoryWitness/raw_edges_removed_by_specificity": float(
            int(specificity_removed_edges.shape[0])
        ),
        "MemoryWitness/mean_current_distance_for_kept_edges": _edge_distance_mean(
            distances,
            selected_edges,
        ),
        "MemoryWitness/mean_current_distance_for_removed_edges": _edge_distance_mean(
            distances,
            specificity_removed_edges,
        ),
    }


def _edge_distance_mean(distances: torch.Tensor, edges: torch.Tensor) -> float:
    if edges.numel() == 0:
        return 0.0
    values = distances[edges[:, 0], edges[:, 1]].detach().float()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 0.0
    return float(finite.mean().detach().cpu())


def _float_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.mean().detach().cpu())


def _float_quantile(values: torch.Tensor, quantile: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), quantile).detach().cpu())


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _select_diagnostic_memory_edges(
    *,
    distances: torch.Tensor,
    sigma: torch.Tensor,
    base_adjacency: torch.Tensor,
    candidates: _MemoryEdgeCandidates,
    selected_adjacency: torch.Tensor,
    selected_overlap: torch.Tensor,
    diagnostic_overlap: torch.Tensor,
    memory_witness_score_min: float,
    memory_extra_edges_per_node_max: int,
    memory_added_edge_cap: int,
    edge_logs: dict[str, float],
    eps: float,
) -> torch.Tensor:
    if _same_tensor_storage(diagnostic_overlap, selected_overlap):
        return selected_adjacency
    adjacency, _ = _select_memory_edges(
        distances=distances,
        sigma=sigma,
        base_adjacency=base_adjacency,
        candidates=candidates,
        selected_overlap=diagnostic_overlap,
        memory_witness_score_min=memory_witness_score_min,
        memory_extra_edges_per_node_max=memory_extra_edges_per_node_max,
        memory_added_edge_cap=memory_added_edge_cap,
        edge_logs=dict(edge_logs),
        eps=eps,
    )
    return adjacency


def _select_memory_edges(
    *,
    distances: torch.Tensor,
    sigma: torch.Tensor,
    base_adjacency: torch.Tensor,
    candidates: _MemoryEdgeCandidates,
    selected_overlap: torch.Tensor,
    memory_witness_score_min: float | None,
    memory_extra_edges_per_node_max: int,
    memory_added_edge_cap: int,
    edge_logs: dict[str, float],
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    num_nodes = int(base_adjacency.shape[0])
    device = base_adjacency.device
    if (
        candidates.numel == 0
        or int(memory_extra_edges_per_node_max) <= 0
        or int(memory_added_edge_cap) <= 0
    ):
        return torch.zeros_like(base_adjacency), edge_logs

    if memory_witness_score_min is None:
        raise ValueError(
            "geometry_support=batch_memory with memory-witnessed graph edges "
            "requires explicit memory_witness_score_min."
        )

    row = candidates.row
    col = candidates.col
    witness_score = selected_overlap.to(device=device)[row, col]
    keep = witness_score >= float(memory_witness_score_min)
    row = row[keep]
    col = col[keep]
    witness_score = witness_score[keep]
    if row.numel() == 0:
        edge_logs["MemoryWitness/added_edges_before_budget"] = 0.0
        return torch.zeros_like(base_adjacency), edge_logs

    current_weight = _current_edge_weights(
        distances,
        sigma,
        row,
        col,
        dtype=torch.float32,
        eps=eps,
    )
    rank_score = witness_score.float() * current_weight.float()
    order = _rank_memory_edge_order(rank_score=rank_score)

    edge_logs["MemoryWitness/added_edges_before_budget"] = float(row.numel())
    added_degree = [0 for _ in range(num_nodes)]
    selected_edges: list[tuple[int, int]] = []
    ordered_edges = torch.stack([row[order], col[order]], dim=1).detach().cpu().tolist()
    for left, right in ordered_edges:
        if added_degree[left] >= memory_extra_edges_per_node_max:
            continue
        if added_degree[right] >= memory_extra_edges_per_node_max:
            continue
        selected_edges.append((left, right))
        added_degree[left] += 1
        added_degree[right] += 1
        if len(selected_edges) >= memory_added_edge_cap:
            break

    witnessed_adjacency = torch.zeros_like(base_adjacency)
    if selected_edges:
        selected = torch.tensor(selected_edges, dtype=torch.long, device=device)
        witnessed_adjacency[selected[:, 0], selected[:, 1]] = True
        witnessed_adjacency[selected[:, 1], selected[:, 0]] = True

    added_after = len(selected_edges)
    edge_logs["MemoryWitness/added_edges_after_budget"] = float(added_after)
    edge_logs["MemoryWitness/added_edges_dropped_by_budget"] = float(
        max(int(row.numel()) - added_after, 0)
    )
    edge_logs["Graph/memory_added_edges"] = float(added_after)
    return witnessed_adjacency, edge_logs


def _rank_memory_edge_order(
    *,
    rank_score: torch.Tensor,
) -> torch.Tensor:
    """Order candidates by score desc while preserving row/col tie order."""
    return torch.argsort(-rank_score, stable=True)


def _cosine_distances(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features, dim=-1)
    return (1.0 - normalized @ normalized.T).clamp_min(0.0)


def _pairwise_distances(
    features: torch.Tensor,
    *,
    metric: GraphDistanceMetric,
) -> torch.Tensor:
    if metric == GraphDistanceMetric.COSINE:
        return _cosine_distances(features)
    raise ValueError(f"Unsupported graph distance metric: {metric.value}")


def _random_symmetric_knn(num_nodes: int, k: int, device: torch.device) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    for idx in range(num_nodes):
        candidates = torch.randperm(num_nodes - 1, device=device)[:k]
        candidates = torch.where(candidates >= idx, candidates + 1, candidates)
        adjacency[idx, candidates] = True
    return adjacency | adjacency.T


def _diagnostics(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    sigma: torch.Tensor | None,
    mode: GraphMode,
) -> GraphDiagnostics:
    device = edge_index.device
    degree = torch.zeros((num_nodes,), dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        row, col = edge_index
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=degree.dtype))
        degree.scatter_add_(0, col, torch.ones_like(col, dtype=degree.dtype))

    components, largest = _component_stats(num_nodes, edge_index)
    singleton_fraction = (
        float((degree == 0).float().mean().detach().cpu()) if num_nodes else 1.0
    )
    sigma_stats = _sigma_stats(sigma)
    num_edges = int(edge_weight.numel())
    degenerate = num_nodes <= 1 or num_edges == 0
    mutual_edge_fraction = (
        1.0 if mode == GraphMode.MUTUAL_KNN and num_edges > 0 else 0.0
    )
    if mode in {
        GraphMode.FULLY_CONNECTED,
        GraphMode.MAX_UNION_KNN,
        GraphMode.RANDOM_REGULAR,
    }:
        mutual_edge_fraction = float("nan")
    return GraphDiagnostics(
        num_nodes=int(num_nodes),
        num_edges=num_edges,
        num_components=components,
        largest_component_size=largest,
        singleton_fraction=singleton_fraction,
        degree_mean=float(degree.mean().detach().cpu()) if num_nodes else 0.0,
        degree_std=float(degree.std(unbiased=False).detach().cpu())
        if num_nodes
        else 0.0,
        degree_max=float(degree.max().detach().cpu()) if num_nodes else 0.0,
        mutual_edge_fraction=mutual_edge_fraction,
        sigma_min=sigma_stats[0],
        sigma_mean=sigma_stats[1],
        sigma_max=sigma_stats[2],
        degenerate=degenerate,
    )


def _component_stats(num_nodes: int, edge_index: torch.Tensor) -> tuple[int, int]:
    if num_nodes == 0:
        return 0, 0
    neighbors = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        edges = edge_index.detach().cpu().tolist()
        for left, right in zip(edges[0], edges[1], strict=True):
            neighbors[left].append(right)
            neighbors[right].append(left)

    seen = [False] * num_nodes
    num_components = 0
    largest = 0
    for node in range(num_nodes):
        if seen[node]:
            continue
        num_components += 1
        stack = [node]
        seen[node] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in neighbors[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        largest = max(largest, size)
    return num_components, largest


def _sigma_stats(sigma: torch.Tensor | None) -> tuple[float, float, float]:
    if sigma is None or sigma.numel() == 0:
        return 0.0, 0.0, 0.0
    finite = sigma[torch.isfinite(sigma)]
    if finite.numel() == 0:
        return 0.0, 0.0, 0.0
    return (
        float(finite.min().detach().cpu()),
        float(finite.mean().detach().cpu()),
        float(finite.max().detach().cpu()),
    )
