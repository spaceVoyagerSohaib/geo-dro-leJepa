"""GeoDRO-LeJEPA v1.1 loss wrapper."""

from __future__ import annotations

import time
import warnings
from typing import Any

import torch
import torch.nn as nn

from stable_pretraining.losses.lejepa import SIGRegLoss

from .distributed import (
    detached_all_gather_batch,
    detached_all_gather_batch_with_metadata,
    get_world_size,
    validate_gathered_batch_sizes,
)
from .flow import solve_graph_flow
from .gating import reliability_gated_weights, warmup_ramp
from .graph import build_graph, build_memory_witnessed_graph, prepare_graph_features
from .memory import (
    GeoDROFeatureMemoryQueue,
    retrieve_witnesses_from_memory_features,
    warn_missing_memory_checkpoint,
)
from .prediction import compute_prediction_terms
from .types import (
    AdversaryScope,
    AggregationBehavior,
    GeoDROAdversaryInputs,
    GeoDROAdversaryWeights,
    GeoDROFamily,
    GeoDROLeJEPALossOutput,
    GeometrySupport,
    GraphDistanceMetric,
    GraphMode,
    GraphSpace,
    MemoryUpdateScope,
    MemoryUsageMode,
    MemoryWitnessAblationMode,
    MemoryWitnessBatch,
    MemoryWitnessThresholdMode,
    SSLInstantiation,
    UtilityMode,
    UtilitySmoothingGraph,
    WitnessScoreMode,
)
from .utility import build_utility
from .witness import compute_witness_overlap_scores


class _BaseGeoDROJEPALoss(nn.Module):
    """Shared LeJEPA loss wrapper for GeoDRO aggregation behaviors."""

    def __init__(
        self,
        lambda_: float = 0.05,
        num_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
        graph_space: str | None = None,
        graph_feature_space: str | None = None,
        graph_feature_normalized: bool = True,
        graph_distance_metric: str = GraphDistanceMetric.COSINE.value,
        memory_stores_normalized_features: bool | None = None,
        graph_mode: str = GraphMode.MUTUAL_KNN.value,
        utility_mode: str = UtilityMode.VIEW_GRAPH_COHERENT.value,
        adversary_scope: str = AdversaryScope.MICROBATCH.value,
        k: int = 8,
        eps_loss: float = 1e-12,
        eps: float = 1e-12,
        u_clip: float = 5.0,
        eta_view: float = 1.0,
        gamma: float = 0.5,
        inner_steps: int = 10,
        beta: float = 0.2,
        tau_flow: float = 0.05,
        p_floor: float = 1e-12,
        eps_log: float = 1e-12,
        alpha_max: float = 0.5,
        warmup_fraction: float = 0.10,
        ramp_fraction: float = 0.05,
        ess_min_ratio: float = 0.25,
        max_p_factor_fail: float = 10.0,
        clamp_activation_fail: float = 0.01,
        singleton_fraction_fail: float = 0.5,
        min_graph_nodes: int | None = None,
        p_cap: float | None = None,
        family: str = GeoDROFamily.GEODRO_JEPA.value,
        aggregation: str = AggregationBehavior.COHERENT_HARDNESS.value,
        geometry_support: str = GeometrySupport.BATCH.value,
        memory_usage_mode: str = MemoryUsageMode.NONE.value,
        memory_queue_capacity: int = 0,
        memory_update_scope: str = MemoryUpdateScope.MICROBATCH.value,
        memory_top_m: int = 64,
        memory_k_sigma: int | None = None,
        memory_min_fill_ratio: float = 0.25,
        memory_retrieval_chunk_size: int | None = 8192,
        witness_score_mode: str = (
            WitnessScoreMode.SPECIFICITY_WEIGHTED_HELLINGER.value
        ),
        memory_k_guard: int = 64,
        memory_witness_score_min: float | None = None,
        memory_witness_threshold_mode: str = (
            MemoryWitnessThresholdMode.EXPLICIT.value
        ),
        memory_witness_null_quantile: float = 0.95,
        memory_witness_calibration_steps: int | None = None,
        memory_witness_ablation_mode: str = MemoryWitnessAblationMode.NONE.value,
        memory_witness_null_seed: int = 0,
        memory_extra_edges_per_node_max: int = 2,
        memory_added_edge_ratio_max: float = 0.25,
        utility_smoothing_graph: str = UtilitySmoothingGraph.FINAL.value,
        ssl_instantiation: str = SSLInstantiation.LEJEPA.value,
    ):
        super().__init__()
        self.family = GeoDROFamily(family)
        self.aggregation = AggregationBehavior(aggregation)
        self.geometry_support = GeometrySupport(geometry_support)
        self.memory_usage_mode = MemoryUsageMode(memory_usage_mode)
        self.memory_queue_capacity = int(memory_queue_capacity)
        self.memory_update_scope = MemoryUpdateScope(memory_update_scope)
        self.memory_top_m = int(memory_top_m)
        self.memory_k_sigma = (
            self.memory_top_m if memory_k_sigma is None else int(memory_k_sigma)
        )
        self.memory_min_fill_ratio = float(memory_min_fill_ratio)
        self.memory_retrieval_chunk_size = (
            None
            if memory_retrieval_chunk_size is None
            else int(memory_retrieval_chunk_size)
        )
        self.witness_score_mode = WitnessScoreMode(witness_score_mode)
        self.memory_k_guard = int(memory_k_guard)
        self.memory_witness_score_min = (
            None
            if memory_witness_score_min is None
            else float(memory_witness_score_min)
        )
        self.memory_witness_threshold_mode = MemoryWitnessThresholdMode(
            memory_witness_threshold_mode
        )
        self.memory_witness_null_quantile = float(memory_witness_null_quantile)
        self.memory_witness_calibration_steps = (
            None
            if memory_witness_calibration_steps is None
            else int(memory_witness_calibration_steps)
        )
        self.memory_witness_ablation_mode = MemoryWitnessAblationMode(
            memory_witness_ablation_mode
        )
        self.memory_witness_null_seed = int(memory_witness_null_seed)
        self.memory_extra_edges_per_node_max = int(memory_extra_edges_per_node_max)
        self.memory_added_edge_ratio_max = float(memory_added_edge_ratio_max)
        self.utility_smoothing_graph = UtilitySmoothingGraph(utility_smoothing_graph)
        self.ssl_instantiation = SSLInstantiation(ssl_instantiation)
        self.lambda_ = lambda_
        self.sigreg = SIGRegLoss(
            num_slices=num_slices,
            t_max=t_max,
            n_points=n_points,
        )
        self.graph_space = _resolve_graph_feature_space(
            graph_space=graph_space,
            graph_feature_space=graph_feature_space,
        )
        self.graph_feature_space = self.graph_space
        self.graph_feature_normalized = bool(graph_feature_normalized)
        self.graph_distance_metric = GraphDistanceMetric(graph_distance_metric)
        if memory_stores_normalized_features is None:
            memory_stores_normalized_features = self.graph_feature_normalized
        self.memory_stores_normalized_features = bool(
            memory_stores_normalized_features
        )
        self.graph_mode = GraphMode(graph_mode)
        self.utility_mode = UtilityMode(utility_mode)
        self.adversary_scope = AdversaryScope(adversary_scope)
        _validate_memory_retrieval_config(
            memory_top_m=self.memory_top_m,
            memory_k_sigma=self.memory_k_sigma,
            memory_min_fill_ratio=self.memory_min_fill_ratio,
            memory_retrieval_chunk_size=self.memory_retrieval_chunk_size,
        )
        _validate_memory_edge_config(
            memory_k_guard=self.memory_k_guard,
            memory_witness_score_min=self.memory_witness_score_min,
            memory_witness_threshold_mode=self.memory_witness_threshold_mode,
            memory_witness_null_quantile=self.memory_witness_null_quantile,
            memory_witness_calibration_steps=self.memory_witness_calibration_steps,
            memory_extra_edges_per_node_max=self.memory_extra_edges_per_node_max,
            memory_added_edge_ratio_max=self.memory_added_edge_ratio_max,
        )
        _validate_base_axes(
            family=self.family,
            geometry_support=self.geometry_support,
            memory_usage_mode=self.memory_usage_mode,
            memory_queue_capacity=self.memory_queue_capacity,
            memory_update_scope=self.memory_update_scope,
            adversary_scope=self.adversary_scope,
            ssl_instantiation=self.ssl_instantiation,
        )
        self.feature_memory = (
            GeoDROFeatureMemoryQueue(
                capacity=self.memory_queue_capacity,
                graph_space=self.graph_space,
                graph_distance_metric=self.graph_distance_metric,
                graph_feature_normalized=self.graph_feature_normalized,
                update_scope=self.memory_update_scope,
            )
            if self.geometry_support == GeometrySupport.BATCH_MEMORY
            else None
        )
        self.k = k
        self.eps_loss = eps_loss
        self.eps = eps
        self.u_clip = u_clip
        self.eta_view = eta_view
        self.gamma = gamma
        self.inner_steps = inner_steps
        self.beta = beta
        self.tau_flow = tau_flow
        self.p_floor = p_floor
        self.eps_log = eps_log
        self.alpha_max = alpha_max
        self.warmup_fraction = warmup_fraction
        self.ramp_fraction = ramp_fraction
        self.ess_min_ratio = ess_min_ratio
        self.max_p_factor_fail = max_p_factor_fail
        self.clamp_activation_fail = clamp_activation_fail
        self.singleton_fraction_fail = singleton_fraction_fail
        self.min_graph_nodes = (
            int(min_graph_nodes) if min_graph_nodes is not None else max(4 * self.k, 64)
        )
        self.p_cap = p_cap

    def forward(
        self,
        proj: torch.Tensor,
        emb: torch.Tensor,
        *,
        global_mask: torch.Tensor | None = None,
        global_view_count: int | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        coherent_mask: torch.Tensor | None = None,
        isolated_mask: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | GeoDROLeJEPALossOutput:
        if self.adversary_scope != AdversaryScope.MICROBATCH:
            raise NotImplementedError(
                f"{self.__class__.__name__}.forward runs only microbatch scope "
                "directly. "
                "Use the GeoDRO optimizer-step training loop for "
                "adversary_scope=optimizer_step."
            )
        if proj.ndim != 3 or emb.ndim != 3:
            raise ValueError("Expected proj and emb with shapes [V, B, K/H].")

        adversary_inputs = self.compute_adversary_inputs(
            proj,
            emb,
            global_mask=global_mask,
            global_view_count=global_view_count,
        )
        graph_gather = detached_all_gather_batch_with_metadata(
            adversary_inputs.graph_features,
            batch_dim=0,
        )
        li_gather = detached_all_gather_batch_with_metadata(
            adversary_inputs.li_local,
            batch_dim=0,
        )
        li_v_gather = detached_all_gather_batch_with_metadata(
            adversary_inputs.li_v,
            batch_dim=1,
        )
        validate_gathered_batch_sizes(graph_gather, li_gather, li_v_gather)

        weights = self.solve_adversary_weights(
            graph_gather.tensor.float(),
            li_gather.tensor,
            li_v_gather.tensor,
            step=step,
            total_steps=total_steps,
            coherent_mask=coherent_mask,
            isolated_mask=isolated_mask,
        )

        p_local = weights.p_global[graph_gather.local_slice].to(
            device=adversary_inputs.li_local.device,
            dtype=adversary_inputs.li_local.dtype,
        )
        pred_loss = (
            get_world_size() * (p_local.detach() * adversary_inputs.li_local).sum()
        )
        sigreg_loss = self.sigreg(proj)
        total_loss = (1.0 - self.lambda_) * pred_loss + self.lambda_ * sigreg_loss
        extra_logs = weights.extra_logs
        self._enqueue_memory_after_loss(graph_gather.tensor, step=step)

        output = GeoDROLeJEPALossOutput(
            total_loss=total_loss,
            pred_loss=pred_loss,
            sigreg_loss=sigreg_loss,
            pred_erm=adversary_inputs.pred_erm.detach(),
            li_local=adversary_inputs.li_local,
            p_local=p_local.detach(),
            p_global=weights.p_global.detach(),
            alpha=weights.weight_diagnostics.alpha,
            graph_diagnostics=weights.graph_diagnostics,
            utility_diagnostics=weights.utility_diagnostics,
            flow_diagnostics=weights.flow_diagnostics,
            weight_diagnostics=weights.weight_diagnostics,
            extra_logs=extra_logs,
        )
        if return_output:
            return output
        return total_loss

    def compute_adversary_inputs(
        self,
        proj: torch.Tensor,
        emb: torch.Tensor,
        *,
        global_mask: torch.Tensor | None = None,
        global_view_count: int | None = None,
    ) -> GeoDROAdversaryInputs:
        """Build detached graph inputs and prediction terms for one local batch."""
        if proj.ndim != 3 or emb.ndim != 3:
            raise ValueError("Expected proj and emb with shapes [V, B, K/H].")
        prediction = compute_prediction_terms(
            proj, global_mask=global_mask, global_view_count=global_view_count
        )
        graph_features = prepare_graph_features(
            emb,
            proj,
            global_mask=global_mask,
            global_view_count=global_view_count,
            graph_space=self.graph_space,
            normalized=self.graph_feature_normalized,
        )
        return GeoDROAdversaryInputs(
            graph_features=graph_features,
            li_local=prediction.li_local,
            li_v=prediction.li_v,
            pred_erm=prediction.pred_erm,
        )

    def solve_adversary_weights(
        self,
        graph_features: torch.Tensor,
        li_global: torch.Tensor,
        li_v_global: torch.Tensor,
        *,
        step: int | None = None,
        total_steps: int | None = None,
        coherent_mask: torch.Tensor | None = None,
        isolated_mask: torch.Tensor | None = None,
    ) -> GeoDROAdversaryWeights:
        """Solve GeoDRO weights for a gathered microbatch or optimizer-step batch."""
        (
            witnesses,
            overlap_scores,
            null_overlap_scores,
            memory_witness_logs,
        ) = self._memory_witness_state(
            graph_features,
            step=step,
        )
        flow_graph, utility_graph, graph_logs, graph_context = (
            self._build_adversary_graphs(
                graph_features,
                overlap_scores=overlap_scores,
                null_overlap_scores=null_overlap_scores,
                step=step,
                total_steps=total_steps,
            )
        )
        utility, utility_diag = build_utility(
            li_global,
            li_v_global,
            utility_graph,
            mode=self.utility_mode,
            eps_loss=self.eps_loss,
            eps=self.eps,
            u_clip=self.u_clip,
            eta_view=self.eta_view,
            gamma=self.gamma,
            li_v_sample_dim=1,
        )
        p_flow, flow_diag = solve_graph_flow(
            utility,
            flow_graph,
            inner_steps=self.inner_steps,
            beta=self.beta,
            tau_flow=self.tau_flow,
            p_floor=self.p_floor,
            eps_log=self.eps_log,
        )
        p_global, weight_diag = reliability_gated_weights(
            p_flow,
            flow_graph.diagnostics,
            flow_diag,
            step=step,
            total_steps=total_steps,
            alpha_max=self.alpha_max,
            warmup_fraction=self.warmup_fraction,
            ramp_fraction=self.ramp_fraction,
            ess_min_ratio=self.ess_min_ratio,
            max_p_factor_fail=self.max_p_factor_fail,
            clamp_activation_fail=self.clamp_activation_fail,
            singleton_fraction_fail=self.singleton_fraction_fail,
            min_graph_nodes=self.min_graph_nodes,
            p_cap=self.p_cap,
        )
        extra_logs = (
            _extra_logs(flow_graph.diagnostics, utility_diag, flow_diag, weight_diag)
            | self._config_logs()
            | _graph_feature_logs(
                graph_features,
                graph_space=self.graph_space,
                normalized=self.graph_feature_normalized,
                distance_metric=self.graph_distance_metric,
                memory_stores_normalized_features=(
                    self.memory_stores_normalized_features
                ),
            )
            | _controlled_mass_logs(
                p_flow,
                coherent_mask=coherent_mask,
                isolated_mask=isolated_mask,
                prefix="RawFlowControlled",
            )
            | _controlled_mass_logs(
                p_global,
                coherent_mask=coherent_mask,
                isolated_mask=isolated_mask,
            )
            | self._memory_logs(step=step)
            | memory_witness_logs
            | graph_logs
            | self._edge_witness_support_logs(
                witnesses,
                graph_context,
                step=step,
            )
            | _utility_weight_sign_logs(utility, p_global)
            | _edge_utility_coherence_logs(graph_context, utility)
        )
        self._validate_memory_diagnostics(extra_logs)
        return GeoDROAdversaryWeights(
            p_global=p_global,
            graph_diagnostics=flow_graph.diagnostics,
            utility_diagnostics=utility_diag,
            flow_diagnostics=flow_diag,
            weight_diagnostics=weight_diag,
            extra_logs=extra_logs,
        )

    @torch.no_grad()
    def _enqueue_memory_after_loss(
        self,
        graph_features_global: torch.Tensor,
        *,
        step: int | None,
    ) -> None:
        if self.feature_memory is None:
            return
        if self.memory_update_scope != MemoryUpdateScope.MICROBATCH:
            raise RuntimeError(
                "GeoDRO microbatch forward can only update batch_memory with "
                f"memory_update_scope={MemoryUpdateScope.MICROBATCH.value}; "
                f"got {self.memory_update_scope.value}."
            )
        self.feature_memory.enqueue(graph_features_global, step=step)

    @torch.no_grad()
    def enqueue_memory_after_optimizer_step(
        self,
        graph_features_global: torch.Tensor,
        *,
        step: int | None,
    ) -> None:
        """Append current optimizer-step support after replay/optimizer update."""
        if self.feature_memory is None:
            return
        if self.memory_update_scope != MemoryUpdateScope.OPTIMIZER_STEP_DELAYED:
            raise RuntimeError(
                "GeoDRO optimizer-step batch_memory updates require "
                "memory_update_scope="
                f"{MemoryUpdateScope.OPTIMIZER_STEP_DELAYED.value}; got "
                f"{self.memory_update_scope.value}."
            )
        self.feature_memory.enqueue(graph_features_global, step=step)

    def _memory_logs(self, *, step: int | None) -> dict[str, float]:
        if self.feature_memory is None:
            return {
                "Memory/enabled": 0.0,
                "Memory/size": 0.0,
                "Memory/fill_ratio": 0.0,
                "Memory/effective_horizon_steps": 0.0,
                "Memory/age_min": 0.0,
                "Memory/age_median": 0.0,
                "Memory/age_max": 0.0,
                "Memory/checkpoint_restored": 0.0,
                "Memory/checkpoint_missing_fallback": 0.0,
                "Memory/update_scope/none": 1.0,
                "Memory/update_clock/none": 1.0,
                "Memory/updates_per_optimizer_step": 0.0,
                "Memory/update_count": 0.0,
                "Memory/queue_memory_mb": 0.0,
                "Memory/metadata_memory_mb": 0.0,
                "Memory/retrieval_time_ms": 0.0,
                "Memory/graph_build_time_ms": 0.0,
                "Memory/peak_allocated_mb_optional": 0.0,
                "GraphFeature/norm_mean_memory": 0.0,
                "GraphFeature/norm_std_memory": 0.0,
            }
        return self.feature_memory.diagnostics(step=step)

    def _build_adversary_graphs(
        self,
        graph_features: torch.Tensor,
        *,
        overlap_scores,
        null_overlap_scores,
        step: int | None,
        total_steps: int | None,
    ) -> tuple[Any, Any, dict[str, float], dict[str, torch.Tensor]]:
        build_start = time.perf_counter()
        if (
            self.geometry_support == GeometrySupport.BATCH_MEMORY
            and self.feature_memory is not None
            and overlap_scores is not None
            and overlap_scores.valid_for_scoring
        ):
            train_warmup = warmup_ramp(
                step=step,
                total_steps=total_steps,
                warmup_fraction=self.warmup_fraction,
                ramp_fraction=self.ramp_fraction,
            )
            result = build_memory_witnessed_graph(
                graph_features.float(),
                overlap_scores,
                mode=self.graph_mode,
                distance_metric=self.graph_distance_metric,
                k=self.k,
                memory_k_guard=self.memory_k_guard,
                memory_witness_score_min=self.memory_witness_score_min,
                memory_extra_edges_per_node_max=self.memory_extra_edges_per_node_max,
                memory_added_edge_ratio_max=self.memory_added_edge_ratio_max,
                fill_ratio=self.feature_memory.fill_ratio,
                memory_min_fill_ratio=self.memory_min_fill_ratio,
                train_warmup=train_warmup,
                memory_witness_threshold_mode=self.memory_witness_threshold_mode,
                memory_witness_null_quantile=self.memory_witness_null_quantile,
                null_overlap_scores=null_overlap_scores,
                eps=self.eps,
                return_result=True,
            )
            flow_graph = result.graph
            graph_logs = result.logs
            utility_graph = self._utility_smoothing_graph_for_flow(
                graph_features,
                flow_graph=flow_graph,
            )
            return (
                flow_graph,
                utility_graph,
                graph_logs | self._utility_smoothing_logs(),
                {
                    "selected_edges": result.selected_edges,
                    "raw_selected_edges": result.raw_selected_edges,
                    "specificity_selected_edges": result.specificity_selected_edges,
                    "specificity_removed_edges": result.specificity_removed_edges,
                },
            )
        graph = build_graph(
            graph_features.float(),
            mode=self.graph_mode,
            distance_metric=self.graph_distance_metric,
            k=self.k,
            eps=self.eps,
        )
        graph_build_time_ms = (time.perf_counter() - build_start) * 1000.0
        return graph, graph, {
            "Graph/batch_edges": float(graph.edge_weight.numel()),
            "Graph/final_edges": float(graph.edge_weight.numel()),
            "Graph/memory_added_edges": 0.0,
            "MemoryWitness/K_guard_eff": 0.0,
            "MemoryWitness/K_guard_fraction": 0.0,
            f"MemoryWitness/threshold_mode/{self.memory_witness_threshold_mode.value}": 1.0,
            "MemoryWitness/threshold_value": (
                -1.0
                if self.memory_witness_score_min is None
                else float(self.memory_witness_score_min)
            ),
            "MemoryWitness/null_score_mean": 0.0,
            "MemoryWitness/null_score_p95": 0.0,
            "MemoryWitness/null_score_p99": 0.0,
            "MemoryWitness/added_edges_before_budget": 0.0,
            "MemoryWitness/added_edges_after_budget": 0.0,
            "MemoryWitness/added_edges_dropped_by_budget": 0.0,
            "MemoryWitness/extra_edges_per_node_budget_eff": 0.0,
            "MemoryWitness/global_added_edge_cap_eff": 0.0,
            "MemoryWitness/fill_ramp": 0.0,
            "MemoryWitness/train_warmup": 0.0,
            "MemoryWitness/budget_scale": 0.0,
            "MemoryWitness/added_edges": 0.0,
            "MemoryWitness/added_edges_per_node_mean": 0.0,
            "MemoryWitness/added_edge_cap_active": 0.0,
            "MemoryWitness/raw_vs_spec_added_edge_agreement": 1.0,
            "MemoryWitness/added_edges_supported_by_top_hubs_fraction": 0.0,
            "MemoryWitness/mean_witness_age_for_added_edges": 0.0,
            "MemoryWitness/raw_edges_removed_by_specificity": 0.0,
            "MemoryWitness/mean_current_distance_for_removed_edges": 0.0,
            "MemoryWitness/mean_current_distance_for_kept_edges": 0.0,
            "MemoryWitness/utility_coherence_for_removed_edges": 0.0,
            "MemoryWitness/utility_coherence_for_kept_edges": 0.0,
            "Memory/graph_build_time_ms": graph_build_time_ms,
        } | self._utility_smoothing_logs(), {
            "selected_edges": torch.empty((0, 2), dtype=torch.long, device=graph_features.device),
            "raw_selected_edges": torch.empty((0, 2), dtype=torch.long, device=graph_features.device),
            "specificity_selected_edges": torch.empty((0, 2), dtype=torch.long, device=graph_features.device),
            "specificity_removed_edges": torch.empty((0, 2), dtype=torch.long, device=graph_features.device),
        }

    def _utility_smoothing_graph_for_flow(
        self,
        graph_features: torch.Tensor,
        *,
        flow_graph,
    ):
        if self.utility_mode != UtilityMode.VIEW_GRAPH_COHERENT:
            return flow_graph
        if self.utility_smoothing_graph != UtilitySmoothingGraph.BATCH:
            return flow_graph
        return build_graph(
            graph_features.float(),
            mode=self.graph_mode,
            distance_metric=self.graph_distance_metric,
            k=self.k,
            eps=self.eps,
        )

    def _utility_smoothing_logs(self) -> dict[str, float]:
        return {
            f"Utility/smoothing_graph/{self.utility_smoothing_graph.value}": 1.0,
            "Utility/smoothing_graph_active": float(
                self.utility_mode == UtilityMode.VIEW_GRAPH_COHERENT
            ),
        }

    def _memory_witness_state(
        self,
        graph_features: torch.Tensor,
        *,
        step: int | None,
    ):
        if self.feature_memory is None:
            return None, None, None, {
                "Memory/retrieval_time_ms": 0.0,
                "MemoryWitness/valid_memory_for_witnessing": 0.0,
                (
                    "MemoryWitness/witness_score_mode/"
                    f"{self.witness_score_mode.value}"
                ): 1.0,
                f"MemoryWitness/ablation_mode/{self.memory_witness_ablation_mode.value}": 1.0,
                "MemoryWitness/null_type/none": 1.0,
            }
        witnesses = self.feature_memory.retrieve_witnesses(
            graph_features,
            top_m=self.memory_top_m,
            k_sigma=self.memory_k_sigma,
            min_fill_ratio=self.memory_min_fill_ratio,
            eps=self.eps,
            chunk_size=self.memory_retrieval_chunk_size,
        )
        active_witnesses, ablation_logs = self._apply_witness_ablation(
            graph_features,
            witnesses,
            step=step,
        )
        overlap_scores = compute_witness_overlap_scores(
            active_witnesses,
            mode=self.witness_score_mode,
        )
        null_witnesses = _shuffle_witness_identities(
            active_witnesses,
            seed=self._null_seed(step, offset=1),
        )
        null_overlap_scores = compute_witness_overlap_scores(
            null_witnesses,
            mode=self.witness_score_mode,
        )
        return (
            active_witnesses,
            overlap_scores,
            null_overlap_scores,
            active_witnesses.diagnostics(eps=self.eps)
            | overlap_scores.diagnostics()
            | ablation_logs,
        )

    def _apply_witness_ablation(
        self,
        graph_features: torch.Tensor,
        witnesses: MemoryWitnessBatch,
        *,
        step: int | None,
    ) -> tuple[MemoryWitnessBatch, dict[str, float]]:
        logs = {
            f"MemoryWitness/ablation_mode/{self.memory_witness_ablation_mode.value}": 1.0,
            "MemoryWitness/null_type/none": 1.0,
        }
        if self.memory_witness_ablation_mode == MemoryWitnessAblationMode.NONE:
            return witnesses, logs
        if self.memory_witness_ablation_mode == MemoryWitnessAblationMode.SHUFFLED_MEMORY:
            logs = {
                f"MemoryWitness/ablation_mode/{self.memory_witness_ablation_mode.value}": 1.0,
                "MemoryWitness/null_type/shuffled_memory": 1.0,
            }
            return (
                _shuffle_witness_identities(
                    witnesses,
                    seed=self._null_seed(step, offset=0),
                ),
                logs,
            )
        if self.memory_witness_ablation_mode == MemoryWitnessAblationMode.RANDOM_MEMORY:
            logs = {
                f"MemoryWitness/ablation_mode/{self.memory_witness_ablation_mode.value}": 1.0,
                "MemoryWitness/null_type/random_memory": 1.0,
            }
            return self._random_memory_witnesses(graph_features, step=step), logs
        raise ValueError(
            f"Unsupported memory witness ablation mode: "
            f"{self.memory_witness_ablation_mode.value}."
        )

    def _random_memory_witnesses(
        self,
        graph_features: torch.Tensor,
        *,
        step: int | None,
    ) -> MemoryWitnessBatch:
        if self.feature_memory is None or self.feature_memory.valid_size == 0:
            return self.feature_memory.retrieve_witnesses(
                graph_features,
                top_m=self.memory_top_m,
                k_sigma=self.memory_k_sigma,
                min_fill_ratio=self.memory_min_fill_ratio,
                eps=self.eps,
                chunk_size=self.memory_retrieval_chunk_size,
            )
        memory = self.feature_memory.valid_features().detach().float()
        steps = self.feature_memory.valid_insertion_steps().detach()
        generator = torch.Generator(device=memory.device)
        generator.manual_seed(self._null_seed(step, offset=17))
        random_memory = torch.randn(
            memory.shape,
            device=memory.device,
            dtype=torch.float32,
            generator=generator,
        )
        norms = memory.norm(dim=1, keepdim=True)
        random_memory = torch.nn.functional.normalize(random_memory, dim=-1) * norms
        return retrieve_witnesses_from_memory_features(
            graph_features,
            random_memory,
            steps,
            metric=self.graph_distance_metric,
            top_m=self.memory_top_m,
            k_sigma=self.memory_k_sigma,
            min_fill_ratio=self.memory_min_fill_ratio,
            fill_ratio=self.feature_memory.fill_ratio,
            eps=self.eps,
            chunk_size=self.memory_retrieval_chunk_size,
        )

    def _null_seed(self, step: int | None, *, offset: int) -> int:
        base = int(self.memory_witness_null_seed)
        step_value = 0 if step is None else int(step)
        return base + step_value + int(offset)

    def _config_logs(self) -> dict[str, float]:
        return {
            f"Config/family/{self.family.value}": 1.0,
            f"Config/aggregation/{self.aggregation.value}": 1.0,
            f"Config/geometry_support/{self.geometry_support.value}": 1.0,
            f"Config/memory_usage_mode/{self.memory_usage_mode.value}": 1.0,
            "Config/queue_capacity": float(self.memory_queue_capacity),
            "Config/top_m": float(self.memory_top_m),
            "Config/k_sigma": float(self.memory_k_sigma),
            "Config/K_guard": float(self.memory_k_guard),
            f"Config/witness_score_mode/{self.witness_score_mode.value}": 1.0,
            f"Config/memory_witness_threshold_mode/{self.memory_witness_threshold_mode.value}": 1.0,
            "Config/memory_witness_null_quantile": float(
                self.memory_witness_null_quantile
            ),
            "Config/memory_witness_calibration_steps": (
                -1.0
                if self.memory_witness_calibration_steps is None
                else float(self.memory_witness_calibration_steps)
            ),
            "Config/memory_witness_score_min": (
                -1.0
                if self.memory_witness_score_min is None
                else float(self.memory_witness_score_min)
            ),
            f"Config/memory_witness_ablation_mode/{self.memory_witness_ablation_mode.value}": 1.0,
            "Config/memory_witness_null_seed": float(self.memory_witness_null_seed),
            "Config/memory_extra_edges_per_node_max": float(
                self.memory_extra_edges_per_node_max
            ),
            "Config/memory_added_edge_ratio_max": float(
                self.memory_added_edge_ratio_max
            ),
            "Config/memory_min_fill_ratio": float(self.memory_min_fill_ratio),
            f"Config/memory_update_scope/{self.memory_update_scope.value}": 1.0,
            f"Config/utility_smoothing_graph/{self.utility_smoothing_graph.value}": 1.0,
            f"Config/graph_feature_space/{self.graph_space.value}": 1.0,
            "Config/graph_feature_normalized": float(self.graph_feature_normalized),
            f"Config/graph_distance_metric/{self.graph_distance_metric.value}": 1.0,
            "Config/memory_stores_normalized_features": float(
                self.memory_stores_normalized_features
            ),
        }

    def _edge_witness_support_logs(
        self,
        witnesses: MemoryWitnessBatch | None,
        graph_context: dict[str, torch.Tensor],
        *,
        step: int | None,
    ) -> dict[str, float]:
        selected_edges = graph_context.get("selected_edges")
        if witnesses is None or selected_edges is None or selected_edges.numel() == 0:
            return {
                "MemoryWitness/added_edges_supported_by_top_hubs_fraction": 0.0,
                "MemoryWitness/mean_witness_age_for_added_edges": 0.0,
            }
        hub_threshold = max(2, int(torch.ceil(torch.tensor(0.10 * witnesses.indices.shape[0])).item()))
        flat_indices = witnesses.indices.reshape(-1)
        unique_indices, inverse = torch.unique(
            flat_indices,
            sorted=True,
            return_inverse=True,
        )
        counts = torch.bincount(inverse, minlength=unique_indices.numel()).to(
            device=witnesses.indices.device,
            dtype=torch.float32,
        )
        current_step = int(step) if step is not None else 0
        hub_supported = []
        ages = []
        for left, right in selected_edges.detach().cpu().tolist():
            left_indices = witnesses.indices[left]
            right_indices = witnesses.indices[right]
            shared = left_indices[
                (left_indices.unsqueeze(1) == right_indices.unsqueeze(0)).any(dim=1)
            ]
            if shared.numel() == 0:
                hub_supported.append(0.0)
                continue
            match = (unique_indices.unsqueeze(1) == shared.unsqueeze(0)).any(dim=1)
            shared_counts = counts[match]
            hub_supported.append(float((shared_counts >= hub_threshold).any().item()))
            for node in (left, right):
                node_shared = (
                    witnesses.indices[node].unsqueeze(1) == shared.unsqueeze(0)
                ).any(dim=1)
                if node_shared.any().item():
                    node_steps = witnesses.insertion_steps[node][node_shared]
                    ages.append(
                        (current_step - node_steps.float()).clamp_min(0.0).mean()
                    )
        age_value = 0.0
        if ages:
            age_value = float(torch.stack(ages).mean().detach().cpu())
        return {
            "MemoryWitness/added_edges_supported_by_top_hubs_fraction": (
                float(sum(hub_supported) / len(hub_supported))
                if hub_supported
                else 0.0
            ),
            "MemoryWitness/mean_witness_age_for_added_edges": age_value,
        }

    def _validate_memory_diagnostics(self, logs: dict[str, float]) -> None:
        if self.geometry_support != GeometrySupport.BATCH_MEMORY:
            return
        required_logs = set(_REQUIRED_BATCH_MEMORY_LOGS) | {
            f"Memory/update_scope/{self.memory_update_scope.value}",
            f"Memory/update_clock/{self.memory_update_scope.value}",
        }
        missing = [key for key in required_logs if key not in logs]
        if missing:
            raise RuntimeError(
                "Missing required GeoDRO batch_memory diagnostics: "
                + ", ".join(sorted(missing))
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.feature_memory is None:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        has_memory_state = any(str(key).startswith("feature_memory.") for key in state_dict)
        if has_memory_state:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        warn_missing_memory_checkpoint()
        self.feature_memory.mark_checkpoint_missing_fallback()
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict:
            non_memory_missing = [
                key
                for key in result.missing_keys
                if not str(key).startswith("feature_memory.")
            ]
            if non_memory_missing or result.unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading state_dict for "
                    f"{self.__class__.__name__}: missing_keys={non_memory_missing}, "
                    f"unexpected_keys={result.unexpected_keys}."
                )
        return result

    def weighted_replay_loss(
        self,
        proj: torch.Tensor,
        emb: torch.Tensor,
        p_local: torch.Tensor,
        *,
        sigreg_scale: float = 1.0,
        global_mask: torch.Tensor | None = None,
        global_view_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute a gradient-bearing replay loss using fixed detached weights."""
        if proj.ndim != 3 or emb.ndim != 3:
            raise ValueError("Expected proj and emb with shapes [V, B, K/H].")
        prediction = compute_prediction_terms(
            proj, global_mask=global_mask, global_view_count=global_view_count
        )
        p_local = p_local.to(
            device=prediction.li_local.device,
            dtype=prediction.li_local.dtype,
        )
        pred_loss = get_world_size() * (p_local.detach() * prediction.li_local).sum()
        sigreg_loss = self.sigreg(proj)
        total_loss = (1.0 - self.lambda_) * pred_loss + self.lambda_ * float(
            sigreg_scale
        ) * sigreg_loss
        return total_loss, pred_loss, sigreg_loss, prediction.pred_erm.detach()


class CoherentHardnessGeoDROLeJEPALoss(_BaseGeoDROJEPALoss):
    """LeJEPA loss with reliability-gated coherent-hardness aggregation."""

    def __init__(
        self,
        *args,
        aggregation: str = AggregationBehavior.COHERENT_HARDNESS.value,
        **kwargs,
    ):
        aggregation_enum = AggregationBehavior(aggregation)
        if aggregation_enum != AggregationBehavior.COHERENT_HARDNESS:
            raise ValueError(
                "CoherentHardnessGeoDROLeJEPALoss requires "
                f"aggregation={AggregationBehavior.COHERENT_HARDNESS.value}, "
                f"got {aggregation_enum.value}."
            )
        super().__init__(*args, aggregation=aggregation_enum.value, **kwargs)


class GraphTransportGeoDROJEPALoss(_BaseGeoDROJEPALoss):
    """LeJEPA loss with finite-time graph-transport robust aggregation."""

    def __init__(
        self,
        *args,
        aggregation: str = AggregationBehavior.GRAPH_TRANSPORT.value,
        utility_mode: str = UtilityMode.VIEW_AWARE.value,
        **kwargs,
    ):
        aggregation_enum = AggregationBehavior(aggregation)
        utility_enum = UtilityMode(utility_mode)
        if aggregation_enum != AggregationBehavior.GRAPH_TRANSPORT:
            raise ValueError(
                "GraphTransportGeoDROJEPALoss requires "
                f"aggregation={AggregationBehavior.GRAPH_TRANSPORT.value}, "
                f"got {aggregation_enum.value}."
            )
        if utility_enum != UtilityMode.VIEW_AWARE:
            raise ValueError(
                "GraphTransportGeoDROJEPALoss requires the canonical "
                f"utility_mode={UtilityMode.VIEW_AWARE.value}, "
                f"got {utility_enum.value}."
            )
        super().__init__(
            *args,
            aggregation=aggregation_enum.value,
            utility_mode=utility_enum.value,
            **kwargs,
        )


class GeoDROLeJEPALoss(CoherentHardnessGeoDROLeJEPALoss):
    """Deprecated compatibility alias for coherent-hardness GeoDRO-LeJEPA."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GeoDROLeJEPALoss is deprecated and now aliases "
            "CoherentHardnessGeoDROLeJEPALoss. Update configs to target "
            "stable_pretraining.geodro_lejepa.CoherentHardnessGeoDROLeJEPALoss.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _resolve_graph_feature_space(
    *,
    graph_space: str | None,
    graph_feature_space: str | None,
) -> GraphSpace:
    if graph_space is not None and graph_feature_space is not None:
        legacy_space = GraphSpace(graph_space)
        canonical_space = GraphSpace(graph_feature_space)
        if legacy_space != canonical_space:
            raise ValueError(
                "graph_space and graph_feature_space must agree when both are "
                f"set, got {legacy_space.value} and {canonical_space.value}."
            )
        return canonical_space
    if graph_feature_space is not None:
        return GraphSpace(graph_feature_space)
    if graph_space is not None:
        return GraphSpace(graph_space)
    return GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER


def _validate_base_axes(
    *,
    family: GeoDROFamily,
    geometry_support: GeometrySupport,
    memory_usage_mode: MemoryUsageMode,
    memory_queue_capacity: int,
    memory_update_scope: MemoryUpdateScope,
    adversary_scope: AdversaryScope,
    ssl_instantiation: SSLInstantiation,
) -> None:
    if family != GeoDROFamily.GEODRO_JEPA:
        raise ValueError(f"Unsupported GeoDRO family: {family.value}.")
    if geometry_support == GeometrySupport.BATCH:
        if memory_usage_mode != MemoryUsageMode.NONE:
            raise ValueError(
                "geometry_support=batch requires memory_usage_mode=none, "
                f"got {memory_usage_mode.value}."
            )
    elif geometry_support == GeometrySupport.BATCH_MEMORY:
        if memory_usage_mode != MemoryUsageMode.MEMORY_WITNESSED:
            raise ValueError(
                "geometry_support=batch_memory requires "
                f"memory_usage_mode={MemoryUsageMode.MEMORY_WITNESSED.value}, "
                f"got {memory_usage_mode.value}."
            )
        if adversary_scope == AdversaryScope.MICROBATCH:
            if memory_update_scope != MemoryUpdateScope.MICROBATCH:
                raise ValueError(
                    "batch_memory with adversary_scope=microbatch requires "
                    f"memory_update_scope={MemoryUpdateScope.MICROBATCH.value}; "
                    f"got {memory_update_scope.value}."
                )
        elif adversary_scope == AdversaryScope.OPTIMIZER_STEP:
            if memory_update_scope != MemoryUpdateScope.OPTIMIZER_STEP_DELAYED:
                raise ValueError(
                    "batch_memory with adversary_scope=optimizer_step requires "
                    "memory_update_scope="
                    f"{MemoryUpdateScope.OPTIMIZER_STEP_DELAYED.value}; got "
                    f"{memory_update_scope.value}."
                )
        else:
            raise ValueError(
                f"Unsupported adversary_scope for batch_memory: "
                f"{adversary_scope.value}."
            )
        if int(memory_queue_capacity) <= 0:
            raise ValueError(
                "geometry_support=batch_memory requires a positive "
                "memory_queue_capacity."
            )
    else:
        raise ValueError(
            f"Unsupported geometry_support: {geometry_support.value}."
        )
    if ssl_instantiation != SSLInstantiation.LEJEPA:
        raise ValueError(f"Unsupported SSL instantiation: {ssl_instantiation.value}.")


def _validate_memory_retrieval_config(
    *,
    memory_top_m: int,
    memory_k_sigma: int,
    memory_min_fill_ratio: float,
    memory_retrieval_chunk_size: int | None,
) -> None:
    if int(memory_top_m) <= 0:
        raise ValueError(f"memory_top_m must be positive, got {memory_top_m}.")
    if int(memory_k_sigma) <= 0:
        raise ValueError(f"memory_k_sigma must be positive, got {memory_k_sigma}.")
    if not 0.0 <= float(memory_min_fill_ratio) <= 1.0:
        raise ValueError(
            "memory_min_fill_ratio must be in [0, 1], got "
            f"{float(memory_min_fill_ratio)}."
        )
    if memory_retrieval_chunk_size is not None and int(memory_retrieval_chunk_size) <= 0:
        raise ValueError(
            "memory_retrieval_chunk_size must be positive when set, got "
            f"{memory_retrieval_chunk_size}."
        )


def _validate_memory_edge_config(
    *,
    memory_k_guard: int,
    memory_witness_score_min: float | None,
    memory_witness_threshold_mode: MemoryWitnessThresholdMode,
    memory_witness_null_quantile: float,
    memory_witness_calibration_steps: int | None,
    memory_extra_edges_per_node_max: int,
    memory_added_edge_ratio_max: float,
) -> None:
    if int(memory_k_guard) <= 0:
        raise ValueError(f"memory_k_guard must be positive, got {memory_k_guard}.")
    if int(memory_extra_edges_per_node_max) < 0:
        raise ValueError(
            "memory_extra_edges_per_node_max must be non-negative, got "
            f"{memory_extra_edges_per_node_max}."
        )
    if float(memory_added_edge_ratio_max) < 0.0:
        raise ValueError(
            "memory_added_edge_ratio_max must be non-negative, got "
            f"{memory_added_edge_ratio_max}."
        )
    if memory_witness_score_min is not None and not (
        0.0 <= float(memory_witness_score_min) <= 1.0
    ):
        raise ValueError(
            "memory_witness_score_min must be in [0, 1] when set, got "
            f"{memory_witness_score_min}."
        )
    if not 0.0 <= float(memory_witness_null_quantile) <= 1.0:
        raise ValueError(
            "memory_witness_null_quantile must be in [0, 1], got "
            f"{memory_witness_null_quantile}."
        )
    if (
        memory_witness_calibration_steps is not None
        and int(memory_witness_calibration_steps) <= 0
    ):
        raise ValueError(
            "memory_witness_calibration_steps must be positive when set, got "
            f"{memory_witness_calibration_steps}."
        )
    if (
        memory_witness_threshold_mode == MemoryWitnessThresholdMode.EXPLICIT
        and memory_witness_score_min is None
    ):
        # Phase 8 enforces this only once memory edges can actually be selected.
        return


def _extra_logs(*diagnostics: Any) -> dict[str, float]:
    logs: dict[str, float] = {}
    for diagnostic in diagnostics:
        prefix = diagnostic.__class__.__name__.removesuffix("Diagnostics")
        for key, value in vars(diagnostic).items():
            if isinstance(value, bool):
                logs[f"{prefix}/{key}"] = float(value)
            elif isinstance(value, int | float) and value == value:
                logs[f"{prefix}/{key}"] = float(value)
            elif key == "fallback_reason":
                reason = value if isinstance(value, str) else "none"
                logs[f"{prefix}/{key}/{reason}"] = 1.0
    return logs


def _graph_feature_logs(
    graph_features: torch.Tensor,
    *,
    graph_space: GraphSpace,
    normalized: bool,
    distance_metric: GraphDistanceMetric,
    memory_stores_normalized_features: bool,
) -> dict[str, float]:
    if graph_features.numel() == 0:
        norm_mean = 0.0
        norm_std = 0.0
    else:
        norms = graph_features.detach().float().norm(dim=-1)
        norm_mean = float(norms.mean().detach().cpu())
        norm_std = float(norms.std(unbiased=False).detach().cpu())
    return {
        "GraphFeature/norm_mean_current": norm_mean,
        "GraphFeature/norm_std_current": norm_std,
        "GraphFeature/normalized": float(normalized),
        f"GraphFeature/distance_metric/{distance_metric.value}": 1.0,
        f"GraphFeature/feature_space/{graph_space.value}": 1.0,
        "GraphFeature/memory_stores_normalized_features": float(
            memory_stores_normalized_features
        ),
    }


def _controlled_mass_logs(
    p_global: torch.Tensor,
    *,
    coherent_mask: torch.Tensor | None,
    isolated_mask: torch.Tensor | None,
    prefix: str = "Controlled",
) -> dict[str, float]:
    logs: dict[str, float] = {}
    for name, mask in (
        ("coherent", coherent_mask),
        ("isolated", isolated_mask),
    ):
        if mask is None:
            continue
        mask_global = mask.to(device=p_global.device, dtype=p_global.dtype)
        if mask_global.numel() != p_global.numel():
            mask_global = detached_all_gather_batch(mask_global, batch_dim=0)
        if mask_global.numel() != p_global.numel():
            raise ValueError(
                f"Expected {name} diagnostic mask to gather to {p_global.numel()} "
                f"entries, got {mask_global.numel()}."
            )
        mask_global = mask_global.reshape_as(p_global)
        fraction = mask_global.mean()
        mass = (p_global.detach() * mask_global).sum()
        uniform_mass = fraction.clamp_min(torch.finfo(p_global.dtype).eps)
        logs[f"{prefix}/{name}_fraction"] = float(fraction.detach().cpu())
        logs[f"{prefix}/{name}_mass"] = float(mass.detach().cpu())
        logs[f"{prefix}/{name}_mass_lift"] = float(
            (mass / uniform_mass).detach().cpu()
        )
    return logs


def _shuffle_witness_identities(
    witnesses: MemoryWitnessBatch,
    *,
    seed: int,
) -> MemoryWitnessBatch:
    if witnesses.indices.numel() == 0 or witnesses.indices.shape[0] <= 1:
        return witnesses
    generator = torch.Generator(device=witnesses.indices.device)
    generator.manual_seed(int(seed))
    permutation = torch.randperm(
        witnesses.indices.shape[0],
        device=witnesses.indices.device,
        generator=generator,
    )
    return MemoryWitnessBatch(
        indices=witnesses.indices.index_select(0, permutation).detach(),
        distances=witnesses.distances.detach(),
        probabilities=witnesses.probabilities.detach(),
        sigma=witnesses.sigma.detach(),
        insertion_steps=witnesses.insertion_steps.index_select(
            0,
            permutation,
        ).detach(),
        valid_memory_for_witnessing=witnesses.valid_memory_for_witnessing,
        retrieval_time_ms=witnesses.retrieval_time_ms,
        top_m=witnesses.top_m,
        k_sigma=witnesses.k_sigma,
    )


def _utility_weight_sign_logs(
    utility: torch.Tensor,
    p_global: torch.Tensor,
) -> dict[str, float]:
    utility = utility.detach().float()
    weights = p_global.detach().float().to(device=utility.device)
    positive = utility > 0
    negative = utility < 0
    return {
        "Utility/mean_weight_for_positive_utility": _masked_tensor_mean(
            weights,
            positive,
        ),
        "Utility/mean_weight_for_negative_utility": _masked_tensor_mean(
            weights,
            negative,
        ),
    }


def _edge_utility_coherence_logs(
    graph_context: dict[str, torch.Tensor],
    utility: torch.Tensor,
) -> dict[str, float]:
    utility = utility.detach().float()
    return {
        "MemoryWitness/utility_coherence_for_kept_edges": _edge_utility_gap_mean(
            graph_context.get("selected_edges"),
            utility,
        ),
        "MemoryWitness/utility_coherence_for_removed_edges": _edge_utility_gap_mean(
            graph_context.get("specificity_removed_edges"),
            utility,
        ),
    }


def _edge_utility_gap_mean(
    edges: torch.Tensor | None,
    utility: torch.Tensor,
) -> float:
    if edges is None or edges.numel() == 0:
        return 0.0
    edge_device = edges.to(device=utility.device)
    gaps = (utility[edge_device[:, 0]] - utility[edge_device[:, 1]]).abs()
    return float(gaps.mean().detach().cpu()) if gaps.numel() else 0.0


def _masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or not mask.any().item():
        return 0.0
    return float(values[mask].mean().detach().cpu())


_REQUIRED_BATCH_MEMORY_LOGS = {
    "Config/queue_capacity",
    "Config/top_m",
    "Config/k_sigma",
    "Config/K_guard",
    "Config/memory_witness_null_quantile",
    "Config/memory_witness_calibration_steps",
    "Config/memory_witness_score_min",
    "Config/memory_extra_edges_per_node_max",
    "Config/memory_added_edge_ratio_max",
    "Config/memory_min_fill_ratio",
    "Config/graph_feature_normalized",
    "Config/memory_stores_normalized_features",
    "Memory/enabled",
    "Memory/size",
    "Memory/fill_ratio",
    "Memory/update_count",
    "Memory/updates_per_optimizer_step",
    "Memory/queue_memory_mb",
    "Memory/metadata_memory_mb",
    "Memory/retrieval_time_ms",
    "Memory/graph_build_time_ms",
    "Memory/peak_allocated_mb_optional",
    "Graph/batch_edges",
    "Graph/final_edges",
    "Graph/memory_added_edges",
    "Weight/alpha",
    "Weight/fallback",
    "Weight/ess_ratio",
    "Weight/max_p",
    "GraphFeature/norm_mean_current",
    "GraphFeature/norm_std_current",
    "GraphFeature/norm_mean_memory",
    "GraphFeature/norm_std_memory",
    "GraphFeature/normalized",
    "MemoryWitness/valid_memory_for_witnessing",
    "MemoryWitness/threshold_value",
    "MemoryWitness/null_score_mean",
    "MemoryWitness/null_score_p95",
    "MemoryWitness/null_score_p99",
    "MemoryWitness/K_guard_eff",
    "MemoryWitness/K_guard_fraction",
    "MemoryWitness/extra_edges_per_node_budget_eff",
    "MemoryWitness/global_added_edge_cap_eff",
    "MemoryWitness/added_edges_before_budget",
    "MemoryWitness/added_edges_after_budget",
    "MemoryWitness/added_edges_dropped_by_budget",
    "MemoryWitness/fill_ramp",
    "MemoryWitness/train_warmup",
    "MemoryWitness/budget_scale",
    "MemoryWitness/selected_count_mean",
    "MemoryWitness/selected_count_p95",
    "MemoryWitness/selected_count_max",
    "MemoryWitness/hub_fraction",
    "MemoryWitness/spec_weight_mean",
    "MemoryWitness/spec_weight_min",
    "MemoryWitness/spec_weight_max",
    "MemoryWitness/raw_overlap_mean",
    "MemoryWitness/spec_overlap_mean",
    "MemoryWitness/added_edges",
    "MemoryWitness/added_edges_per_node_mean",
    "MemoryWitness/added_edge_cap_active",
    "MemoryWitness/raw_vs_spec_added_edge_agreement",
    "MemoryWitness/added_edges_supported_by_top_hubs_fraction",
    "MemoryWitness/mean_witness_age_for_added_edges",
    "MemoryWitness/raw_edges_removed_by_specificity",
    "MemoryWitness/mean_current_distance_for_removed_edges",
    "MemoryWitness/mean_current_distance_for_kept_edges",
    "MemoryWitness/utility_coherence_for_removed_edges",
    "MemoryWitness/utility_coherence_for_kept_edges",
    "MemoryWitness/distribution_entropy_mean",
    "MemoryWitness/distribution_entropy_p95",
    "MemoryWitness/distribution_perplexity_mean",
    "MemoryWitness/top1_mass_mean",
    "MemoryWitness/top8_mass_mean",
    "MemoryWitness/sigma_memory_mean",
    "MemoryWitness/sigma_memory_p95",
    "Utility/loss_standardized_positive_fraction",
    "Utility/view_reliability_by_loss_sign",
    "Utility/mean_weight_for_negative_utility",
    "Utility/mean_weight_for_positive_utility",
    "Utility/smoothing_graph_active",
}
