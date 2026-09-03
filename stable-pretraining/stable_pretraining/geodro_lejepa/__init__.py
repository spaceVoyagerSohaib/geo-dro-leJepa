"""GeoDRO-LeJEPA v1.1 training architecture."""

from .forward import geodro_lejepa_forward
from .loss import (
    CoherentHardnessGeoDROLeJEPALoss,
    GeoDROLeJEPALoss,
    GraphTransportGeoDROJEPALoss,
)
from .memory import GeoDROFeatureMemoryQueue
from .optimization_policy import GeoDROOptimizerStepPolicy
from .witness import compute_witness_overlap_scores
from .controlled import (
    CIFAR10ControlledCorruption,
    ControlledCorruption,
    ImageNet100ControlledCorruption,
)
from .types import (
    AdversaryScope,
    AggregationBehavior,
    FlowDiagnostics,
    GeoDROAdversaryInputs,
    GeoDROAdversaryWeights,
    GeoDROFamily,
    GeoDROLeJEPALossOutput,
    GeometrySupport,
    GraphDistanceMetric,
    GraphDiagnostics,
    GraphMode,
    GraphSpace,
    MemoryUpdateScope,
    MemoryUsageMode,
    MemoryWitnessAblationMode,
    MemoryWitnessOverlapScores,
    MemoryWitnessBatch,
    MemoryWitnessThresholdMode,
    SSLInstantiation,
    UtilityDiagnostics,
    UtilityMode,
    UtilitySmoothingGraph,
    WeightDiagnostics,
    WitnessScoreMode,
)

__all__ = [
    "AdversaryScope",
    "AggregationBehavior",
    "CIFAR10ControlledCorruption",
    "CoherentHardnessGeoDROLeJEPALoss",
    "ControlledCorruption",
    "FlowDiagnostics",
    "GeoDROAdversaryInputs",
    "GeoDROAdversaryWeights",
    "GeoDROFamily",
    "GeoDROLeJEPALoss",
    "GeoDROLeJEPALossOutput",
    "GeoDROOptimizerStepPolicy",
    "GeoDROFeatureMemoryQueue",
    "GeometrySupport",
    "GraphDistanceMetric",
    "GraphTransportGeoDROJEPALoss",
    "GraphDiagnostics",
    "GraphMode",
    "GraphSpace",
    "ImageNet100ControlledCorruption",
    "MemoryUpdateScope",
    "MemoryUsageMode",
    "MemoryWitnessAblationMode",
    "MemoryWitnessOverlapScores",
    "MemoryWitnessBatch",
    "MemoryWitnessThresholdMode",
    "SSLInstantiation",
    "UtilityDiagnostics",
    "UtilityMode",
    "UtilitySmoothingGraph",
    "WeightDiagnostics",
    "WitnessScoreMode",
    "compute_witness_overlap_scores",
    "geodro_lejepa_forward",
]
