"""Shared types for GeoDRO-LeJEPA."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class GraphSpace(str, Enum):
    """Feature space used for graph construction."""

    PRE_PROJECTOR_GLOBAL_CENTER = "pre_projector_global_center"
    PROJECTOR_GLOBAL_CENTER = "projector_global_center"
    CONSENSUS_PREPROJ_PROJECTOR = "consensus_preproj_projector"
    RANDOM_FEATURES = "random_features"


class GraphDistanceMetric(str, Enum):
    """Distance metric used for graph geometry."""

    COSINE = "cosine"


class GraphMode(str, Enum):
    """Graph construction mode."""

    MUTUAL_KNN = "mutual_knn"
    MAX_UNION_KNN = "max_union_knn"
    FULLY_CONNECTED = "fully_connected"
    RANDOM_REGULAR = "random_regular"
    NO_GRAPH_KL = "no_graph_kl"


class UtilityMode(str, Enum):
    """Detached utility used by the adversary."""

    RAW_LOSS = "raw_loss"
    STANDARDIZED_LOSS = "standardized_loss"
    VIEW_AWARE = "view_aware"
    VIEW_GRAPH_COHERENT = "view_graph_coherent"


class UtilitySmoothingGraph(str, Enum):
    """Graph used for coherent-hardness utility smoothing."""

    FINAL = "final"
    BATCH = "batch"


class GeoDROFamily(str, Enum):
    """Top-level GeoDRO method family."""

    GEODRO_JEPA = "geodro_jepa"


class AggregationBehavior(str, Enum):
    """Robust aggregation behavior."""

    COHERENT_HARDNESS = "coherent_hardness"
    GRAPH_TRANSPORT = "graph_transport"


class GeometrySupport(str, Enum):
    """Representation support used to construct graph geometry."""

    BATCH = "batch"
    BATCH_MEMORY = "batch_memory"


class MemoryUsageMode(str, Enum):
    """How optional memory support is used."""

    NONE = "none"
    MEMORY_WITNESSED = "memory_witnessed"


class MemoryUpdateScope(str, Enum):
    """Clock used to update optional memory support."""

    MICROBATCH = "microbatch"
    OPTIMIZER_STEP_DELAYED = "optimizer_step_delayed"


class WitnessScoreMode(str, Enum):
    """Sparse memory witness overlap scoring mode."""

    SPECIFICITY_WEIGHTED_HELLINGER = "specificity_weighted_hellinger"
    RAW_HELLINGER = "raw_hellinger"


class MemoryWitnessThresholdMode(str, Enum):
    """How the memory witness score threshold is selected."""

    EXPLICIT = "explicit"
    SHUFFLED_NULL_QUANTILE = "shuffled_null_quantile"


class MemoryWitnessAblationMode(str, Enum):
    """Optional null ablation applied to memory witnesses."""

    NONE = "none"
    SHUFFLED_MEMORY = "shuffled_memory"
    RANDOM_MEMORY = "random_memory"


class SSLInstantiation(str, Enum):
    """SSL objective family that GeoDRO wraps."""

    LEJEPA = "lejepa"


class AdversaryScope(str, Enum):
    """Batch scope over which adversarial weights are computed."""

    MICROBATCH = "microbatch"
    OPTIMIZER_STEP = "optimizer_step"


@dataclass(frozen=True)
class MemoryWitnessBatch:
    """Sparse detached current-memory witness distributions."""

    indices: torch.Tensor
    distances: torch.Tensor
    probabilities: torch.Tensor
    sigma: torch.Tensor
    insertion_steps: torch.Tensor
    valid_memory_for_witnessing: bool
    retrieval_time_ms: float
    top_m: int
    k_sigma: int

    def diagnostics(self, *, eps: float = 1e-12) -> dict[str, float]:
        """Summarize witness distribution sharpness for logging."""
        logs = {
            "Memory/retrieval_time_ms": float(self.retrieval_time_ms),
            "MemoryWitness/valid_memory_for_witnessing": float(
                self.valid_memory_for_witnessing
            ),
        }
        if self.probabilities.numel() == 0 or self.sigma.numel() == 0:
            logs |= {
                "MemoryWitness/distribution_entropy_mean": 0.0,
                "MemoryWitness/distribution_entropy_p95": 0.0,
                "MemoryWitness/distribution_perplexity_mean": 0.0,
                "MemoryWitness/top1_mass_mean": 0.0,
                "MemoryWitness/top8_mass_mean": 0.0,
                "MemoryWitness/sigma_memory_mean": 0.0,
                "MemoryWitness/sigma_memory_p95": 0.0,
            }
            return logs

        probs = self.probabilities.detach().float()
        sigma = self.sigma.detach().float()
        entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=1)
        top1 = probs.max(dim=1).values
        top8 = probs[:, : min(8, probs.shape[1])].sum(dim=1)
        logs |= {
            "MemoryWitness/distribution_entropy_mean": _tensor_mean(entropy),
            "MemoryWitness/distribution_entropy_p95": _tensor_quantile(
                entropy, 0.95
            ),
            "MemoryWitness/distribution_perplexity_mean": _tensor_mean(
                entropy.exp()
            ),
            "MemoryWitness/top1_mass_mean": _tensor_mean(top1),
            "MemoryWitness/top8_mass_mean": _tensor_mean(top8),
            "MemoryWitness/sigma_memory_mean": _tensor_mean(sigma),
            "MemoryWitness/sigma_memory_p95": _tensor_quantile(sigma, 0.95),
        }
        return logs


@dataclass(frozen=True)
class MemoryWitnessOverlapScores:
    """Pairwise current-current witness overlap scores."""

    raw_overlap: torch.Tensor
    specificity_overlap: torch.Tensor
    selected_overlap: torch.Tensor
    selected_counts: torch.Tensor
    specificity_weights: torch.Tensor
    witness_score_mode: WitnessScoreMode
    valid_for_scoring: bool

    def diagnostics(self) -> dict[str, float]:
        """Summarize witness overlap and hub-specificity behavior."""
        logs = {
            f"MemoryWitness/witness_score_mode/{self.witness_score_mode.value}": 1.0,
        }
        if (
            not self.valid_for_scoring
            or self.raw_overlap.numel() == 0
            or self.selected_counts.numel() == 0
        ):
            logs |= _zero_witness_overlap_logs()
            return logs

        raw_pairs = _off_diagonal_values(self.raw_overlap.detach().float())
        spec_pairs = _off_diagonal_values(
            self.specificity_overlap.detach().float()
        )
        counts = self.selected_counts.detach().float()
        weights = self.specificity_weights.detach().float()
        batch_size = int(self.raw_overlap.shape[0])
        hub_threshold = max(2, _ceil_fraction(batch_size, 0.10))
        logs |= {
            "MemoryWitness/selected_count_mean": _tensor_mean(counts),
            "MemoryWitness/selected_count_p95": _tensor_quantile(counts, 0.95),
            "MemoryWitness/selected_count_max": _tensor_max(counts),
            "MemoryWitness/hub_fraction": _tensor_mean(
                (counts >= hub_threshold).float()
            ),
            "MemoryWitness/spec_weight_mean": _tensor_mean(weights),
            "MemoryWitness/spec_weight_min": _tensor_min(weights),
            "MemoryWitness/spec_weight_max": _tensor_max(weights),
            "MemoryWitness/raw_overlap_mean": _tensor_mean(raw_pairs),
            "MemoryWitness/spec_overlap_mean": _tensor_mean(spec_pairs),
        }
        return logs


@dataclass(frozen=True)
class PredictionTerms:
    """Per-view and per-sample LeJEPA prediction terms."""

    centers: torch.Tensor
    li_v: torch.Tensor
    li_local: torch.Tensor
    pred_erm: torch.Tensor


@dataclass(frozen=True)
class GraphData:
    """Undirected weighted graph stored as a single-pass edge list."""

    num_nodes: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    diagnostics: "GraphDiagnostics"


@dataclass(frozen=True)
class GraphDiagnostics:
    """Graph reliability diagnostics."""

    num_nodes: int
    num_edges: int
    num_components: int
    largest_component_size: int
    singleton_fraction: float
    degree_mean: float
    degree_std: float
    degree_max: float
    mutual_edge_fraction: float
    sigma_min: float
    sigma_mean: float
    sigma_max: float
    degenerate: bool = False


@dataclass(frozen=True)
class UtilityDiagnostics:
    """Utility and view-reliability diagnostics."""

    utility_mean: float
    utility_std: float
    utility_min: float
    utility_max: float
    view_disp_mean: float
    view_disp_max: float
    view_reliability_mean: float
    view_reliability_min: float
    graph_dirichlet_energy: float
    loss_standardized_positive_fraction: float
    view_reliability_by_loss_sign: float
    view_reliability_positive_loss_mean: float
    view_reliability_negative_loss_mean: float
    nan_or_inf_seen: bool = False


@dataclass(frozen=True)
class FlowDiagnostics:
    """Numerical diagnostics from the finite-time graph-flow solver."""

    clamp_activation_ratio: float
    nan_or_inf_seen: bool
    min_p_before_clamp: float
    max_p: float
    entropy: float
    ess_ratio: float
    flow_num_steps: int
    fell_back_to_uniform: bool = False
    accepted_substeps: int = 0
    rejected_substeps: int = 0
    minimum_accepted_dt: float = 0.0
    requested_horizon: float = 0.0
    completed_horizon: float = 0.0
    raw_utility_gain: float = 0.0
    regularized_objective_gain: float = 0.0


@dataclass(frozen=True)
class WeightDiagnostics:
    """Diagnostics for reliability-gated training weights."""

    alpha: float
    warmup_multiplier: float
    warmup_step: int | None
    warmup_total_steps: int | None
    graph_gate: float
    flow_gate: float
    fallback: bool
    fallback_reason: str | None
    entropy: float
    ess_ratio: float
    max_p: float
    min_p: float


@dataclass(frozen=True)
class GeoDROLeJEPALossOutput:
    """Structured output from the GeoDRO-LeJEPA loss."""

    total_loss: torch.Tensor
    pred_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    pred_erm: torch.Tensor
    li_local: torch.Tensor
    p_local: torch.Tensor
    p_global: torch.Tensor
    alpha: float
    graph_diagnostics: GraphDiagnostics
    utility_diagnostics: UtilityDiagnostics
    flow_diagnostics: FlowDiagnostics
    weight_diagnostics: WeightDiagnostics
    extra_logs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GeoDROAdversaryInputs:
    """Detached inputs needed to solve a GeoDRO adversary graph."""

    graph_features: torch.Tensor
    li_local: torch.Tensor
    li_v: torch.Tensor
    pred_erm: torch.Tensor


@dataclass(frozen=True)
class GeoDROAdversaryWeights:
    """Step-global adversary weights and diagnostics."""

    p_global: torch.Tensor
    graph_diagnostics: GraphDiagnostics
    utility_diagnostics: UtilityDiagnostics
    flow_diagnostics: FlowDiagnostics
    weight_diagnostics: WeightDiagnostics
    extra_logs: dict[str, float] = field(default_factory=dict)


def _tensor_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.mean().detach().cpu())


def _tensor_min(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.min().detach().cpu())


def _tensor_max(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.max().detach().cpu())


def _tensor_quantile(values: torch.Tensor, quantile: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), quantile).detach().cpu())


def _off_diagonal_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] <= 1 or matrix.shape[1] <= 1:
        return matrix.new_empty((0,))
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _ceil_fraction(value: int, fraction: float) -> int:
    scaled = float(value) * float(fraction)
    return int(torch.ceil(torch.tensor(scaled)).item())


def _zero_witness_overlap_logs() -> dict[str, float]:
    return {
        "MemoryWitness/selected_count_mean": 0.0,
        "MemoryWitness/selected_count_p95": 0.0,
        "MemoryWitness/selected_count_max": 0.0,
        "MemoryWitness/hub_fraction": 0.0,
        "MemoryWitness/spec_weight_mean": 0.0,
        "MemoryWitness/spec_weight_min": 0.0,
        "MemoryWitness/spec_weight_max": 0.0,
        "MemoryWitness/raw_overlap_mean": 0.0,
        "MemoryWitness/spec_overlap_mean": 0.0,
    }
