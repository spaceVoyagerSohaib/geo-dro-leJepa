from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image

from stable_pretraining.geodro_lejepa import (
    AggregationBehavior,
    CIFAR10ControlledCorruption,
    CoherentHardnessGeoDROLeJEPALoss,
    GeoDROFamily,
    GeoDROFeatureMemoryQueue,
    GeoDROLeJEPALoss,
    GeometrySupport,
    GraphDistanceMetric,
    GraphTransportGeoDROJEPALoss,
    ImageNet100ControlledCorruption,
    MemoryUpdateScope,
    MemoryUsageMode,
    MemoryWitnessOverlapScores,
    MemoryWitnessAblationMode,
    MemoryWitnessBatch,
    MemoryWitnessThresholdMode,
    SSLInstantiation,
    UtilityMode,
    UtilitySmoothingGraph,
    WitnessScoreMode,
    compute_witness_overlap_scores,
    geodro_lejepa_forward,
)
import stable_pretraining.geodro_lejepa.distributed as distributed_module
import stable_pretraining.geodro_lejepa.loss as loss_module
import stable_pretraining.geodro_lejepa.utility as utility_module
from stable_pretraining.geodro_lejepa.distributed import (
    GatheredBatch,
    _offsets_from_sizes,
    detached_all_gather_batch_with_metadata,
    local_batch_slice,
)
from stable_pretraining.geodro_lejepa.flow import solve_graph_flow
from stable_pretraining.geodro_lejepa.gating import reliability_gated_weights
from stable_pretraining.geodro_lejepa.graph import (
    build_graph,
    build_graph_features,
    build_memory_witnessed_graph,
    prepare_graph_features,
)
from stable_pretraining.geodro_lejepa.optimizer_step import (
    _CollectedMicrobatch,
    _accumulation_corrected_total_steps,
    _solve_step_weights,
    optimizer_step_training_step,
)
from stable_pretraining.geodro_lejepa.prediction import compute_prediction_terms
from stable_pretraining.geodro_lejepa.types import (
    AdversaryScope,
    FlowDiagnostics,
    GraphMode,
    GraphSpace,
)
from stable_pretraining.geodro_lejepa.utility import build_utility
from stable_pretraining.losses import LeJEPALoss


EXAMPLES_CONFIG_DIR = Path(__file__).resolve().parents[3] / "examples"


@pytest.mark.unit
def test_prediction_decomposition_matches_baseline_lejepa():
    proj = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[3.0, 0.0], [3.0, 0.0]],
            [[10.0, 0.0], [10.0, 0.0]],
        ]
    )

    _, baseline_pred, _ = LeJEPALoss(lambda_=0.5)(
        proj, return_components=True, global_view_count=2
    )
    terms = compute_prediction_terms(proj, global_view_count=2)

    assert terms.li_v.shape == (3, 2)
    assert terms.li_local.shape == (2,)
    assert torch.isclose(terms.pred_erm, baseline_pred)
    assert torch.isclose(terms.pred_erm, torch.tensor(11.0))


@pytest.mark.unit
def test_alpha_zero_geodro_prediction_matches_erm():
    torch.manual_seed(0)
    proj = torch.randn(4, 6, 3, requires_grad=True)
    emb = torch.randn(4, 6, 5, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        k=2,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    output = loss_fn(proj, emb, return_output=True)

    assert torch.isclose(output.pred_loss, output.pred_erm)
    assert torch.isclose(output.total_loss, output.pred_erm)
    assert not output.p_local.requires_grad
    output.total_loss.backward()
    assert proj.grad is not None
    assert emb.grad is None


@pytest.mark.unit
def test_coherent_hardness_behavior_axes_default_to_current_v1_path():
    loss_fn = CoherentHardnessGeoDROLeJEPALoss()

    assert loss_fn.family == GeoDROFamily.GEODRO_JEPA
    assert loss_fn.aggregation == AggregationBehavior.COHERENT_HARDNESS
    assert loss_fn.geometry_support == GeometrySupport.BATCH
    assert loss_fn.memory_usage_mode == MemoryUsageMode.NONE
    assert loss_fn.ssl_instantiation == SSLInstantiation.LEJEPA


@pytest.mark.unit
def test_alias_equivalence_warns_and_matches_coherent_hardness_outputs():
    torch.manual_seed(1)
    proj = torch.randn(3, 5, 4, requires_grad=True)
    emb = torch.randn(3, 5, 6, requires_grad=True)
    common = dict(
        lambda_=0.0,
        alpha_max=0.0,
        k=2,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    new_loss = CoherentHardnessGeoDROLeJEPALoss(**common)
    with pytest.warns(FutureWarning, match="GeoDROLeJEPALoss is deprecated"):
        old_loss = GeoDROLeJEPALoss(**common)

    new_output = new_loss(proj, emb, return_output=True)
    old_output = old_loss(proj, emb, return_output=True)

    assert torch.allclose(new_output.total_loss, old_output.total_loss)
    assert torch.allclose(new_output.pred_loss, old_output.pred_loss)
    assert torch.allclose(new_output.p_global, old_output.p_global)
    assert new_loss.state_dict().keys() == old_loss.state_dict().keys()


@pytest.mark.unit
def test_old_config_still_loads_with_deprecation_warning():
    config = {
        "_target_": "stable_pretraining.geodro_lejepa.GeoDROLeJEPALoss",
        "lambda_": 0.0,
    }

    with pytest.warns(FutureWarning, match="GeoDROLeJEPALoss is deprecated"):
        loss_fn = instantiate(config)

    assert isinstance(loss_fn, CoherentHardnessGeoDROLeJEPALoss)


@pytest.mark.unit
def test_new_config_loads_coherent_hardness_class():
    config = {
        "_target_": (
            "stable_pretraining.geodro_lejepa.CoherentHardnessGeoDROLeJEPALoss"
        ),
        "family": "geodro_jepa",
        "aggregation": "coherent_hardness",
        "geometry_support": "batch",
        "memory_usage_mode": "none",
        "ssl_instantiation": "lejepa",
        "graph_feature_space": "pre_projector_global_center",
        "graph_feature_normalized": True,
        "graph_distance_metric": "cosine",
        "memory_stores_normalized_features": True,
        "lambda_": 0.0,
    }

    loss_fn = instantiate(config)

    assert isinstance(loss_fn, CoherentHardnessGeoDROLeJEPALoss)
    assert loss_fn.aggregation == AggregationBehavior.COHERENT_HARDNESS
    assert loss_fn.graph_feature_space == GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER
    assert loss_fn.graph_feature_normalized is True
    assert loss_fn.graph_distance_metric == GraphDistanceMetric.COSINE


@pytest.mark.unit
def test_state_dict_compatibility_between_alias_and_new_class():
    with pytest.warns(FutureWarning, match="GeoDROLeJEPALoss is deprecated"):
        old_loss = GeoDROLeJEPALoss(lambda_=0.0)
    new_loss = CoherentHardnessGeoDROLeJEPALoss(lambda_=0.0)

    new_loss.load_state_dict(old_loss.state_dict(), strict=True)

    for key, value in old_loss.state_dict().items():
        assert torch.equal(value, new_loss.state_dict()[key])


@pytest.mark.unit
def test_graph_transport_defaults_to_view_aware_batch():
    loss_fn = GraphTransportGeoDROJEPALoss()

    assert loss_fn.family == GeoDROFamily.GEODRO_JEPA
    assert loss_fn.aggregation == AggregationBehavior.GRAPH_TRANSPORT
    assert loss_fn.geometry_support == GeometrySupport.BATCH
    assert loss_fn.memory_usage_mode == MemoryUsageMode.NONE
    assert loss_fn.ssl_instantiation == SSLInstantiation.LEJEPA
    assert loss_fn.utility_mode == UtilityMode.VIEW_AWARE


@pytest.mark.unit
def test_graph_transport_rejects_neighbor_utility_smoothing():
    with pytest.raises(ValueError, match="utility_mode=view_aware"):
        GraphTransportGeoDROJEPALoss(utility_mode=UtilityMode.VIEW_GRAPH_COHERENT.value)


@pytest.mark.unit
def test_graph_transport_batch_runs_end_to_end_current_only_and_detached():
    torch.manual_seed(2)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 5, requires_grad=True)
    loss_fn = GraphTransportGeoDROJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        k=2,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    output = loss_fn(proj, emb, return_output=True)

    assert output.graph_diagnostics.num_nodes == 6
    assert output.p_global.numel() == 6
    assert output.p_local.shape == output.li_local.shape
    assert not output.p_global.requires_grad
    assert not output.p_local.requires_grad
    output.total_loss.backward()
    assert proj.grad is not None
    assert emb.grad is None


@pytest.mark.unit
def test_graph_transport_uses_no_neighbor_utility_smoothing(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("GraphTransport must not use graph-neighbor smoothing")

    monkeypatch.setattr(utility_module, "graph_neighbor_average", fail_if_called)
    torch.manual_seed(3)
    proj = torch.randn(3, 4, 4, requires_grad=True)
    emb = torch.randn(3, 4, 5, requires_grad=True)
    loss_fn = GraphTransportGeoDROJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        k=2,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    output = loss_fn(proj, emb, return_output=True)

    assert output.utility_diagnostics.nan_or_inf_seen is False


@pytest.mark.unit
def test_graph_transport_optimizer_step_replay_matches_microbatch_for_single_microbatch():
    torch.manual_seed(4)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 5, requires_grad=True)
    common = dict(
        lambda_=0.0,
        alpha_max=0.3,
        graph_mode=GraphMode.FULLY_CONNECTED.value,
        k=2,
        min_graph_nodes=4,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    microbatch_loss = GraphTransportGeoDROJEPALoss(**common)
    optimizer_step_loss = GraphTransportGeoDROJEPALoss(
        **common,
        adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
    )

    microbatch_output = microbatch_loss(
        proj,
        emb,
        step=5,
        total_steps=10,
        return_output=True,
    )
    adversary_inputs = optimizer_step_loss.compute_adversary_inputs(proj, emb)
    weights = optimizer_step_loss.solve_adversary_weights(
        adversary_inputs.graph_features,
        adversary_inputs.li_local,
        adversary_inputs.li_v,
        step=5,
        total_steps=10,
    )
    total_loss, pred_loss, _, pred_erm = optimizer_step_loss.weighted_replay_loss(
        proj,
        emb,
        weights.p_global,
    )

    assert torch.allclose(weights.p_global, microbatch_output.p_global)
    assert torch.allclose(pred_loss, microbatch_output.pred_loss)
    assert torch.allclose(total_loss, microbatch_output.total_loss)
    assert torch.allclose(pred_erm, microbatch_output.pred_erm)


@pytest.mark.unit
def test_graph_transport_config_loads():
    config = {
        "_target_": ("stable_pretraining.geodro_lejepa.GraphTransportGeoDROJEPALoss"),
        "family": "geodro_jepa",
        "aggregation": "graph_transport",
        "geometry_support": "batch",
        "memory_usage_mode": "none",
        "ssl_instantiation": "lejepa",
        "graph_feature_space": "pre_projector_global_center",
        "graph_feature_normalized": True,
        "graph_distance_metric": "cosine",
        "memory_stores_normalized_features": True,
        "lambda_": 0.0,
    }

    loss_fn = instantiate(config)

    assert isinstance(loss_fn, GraphTransportGeoDROJEPALoss)
    assert loss_fn.aggregation == AggregationBehavior.GRAPH_TRANSPORT
    assert loss_fn.utility_mode == UtilityMode.VIEW_AWARE
    assert loss_fn.graph_distance_metric == GraphDistanceMetric.COSINE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config_name", "loss_cls", "aggregation"),
    [
        (
            "geodro/geodro_jepa_v2_coherent_hardness_batch_memory_optstep_imagenet100ctrl",
            CoherentHardnessGeoDROLeJEPALoss,
            AggregationBehavior.COHERENT_HARDNESS,
        ),
        (
            "geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl",
            GraphTransportGeoDROJEPALoss,
            AggregationBehavior.GRAPH_TRANSPORT,
        ),
    ],
)
def test_v2_optstep_batch_memory_configs_compose(
    config_name,
    loss_cls,
    aggregation,
):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(EXAMPLES_CONFIG_DIR),
    ):
        cfg = compose(config_name=config_name)

    loss_fn = instantiate(cfg.module.geodro_lejepa_loss)

    assert isinstance(loss_fn, loss_cls)
    assert loss_fn.aggregation == aggregation
    assert loss_fn.geometry_support == GeometrySupport.BATCH_MEMORY
    assert loss_fn.memory_usage_mode == MemoryUsageMode.MEMORY_WITNESSED
    assert loss_fn.adversary_scope == AdversaryScope.OPTIMIZER_STEP
    assert loss_fn.memory_update_scope == MemoryUpdateScope.OPTIMIZER_STEP_DELAYED
    assert loss_fn.memory_queue_capacity == 16384
    assert int(cfg.trainer.num_nodes) == 1
    assert int(cfg.trainer.devices) == 4
    assert int(cfg.trainer.accumulate_grad_batches) == 4
    assert int(cfg.data.train.batch_size) == 128
    assert int(cfg.metadata.global_graph_batch_size) == 2048


@pytest.mark.unit
def test_graph_feature_shapes_and_mutual_knn_graph():
    emb = torch.randn(3, 5, 7)
    proj = torch.randn(3, 5, 4)
    global_mask = torch.tensor([True, True, False])

    features = build_graph_features(emb, proj, global_mask=global_mask)
    graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=2)

    assert features.shape == (5, 7)
    assert graph.num_nodes == 5
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_weight.ndim == 1
    assert graph.diagnostics.num_nodes == 5


@pytest.mark.unit
def test_prepare_graph_features_matches_legacy_default_wrapper():
    torch.manual_seed(5)
    emb = torch.randn(3, 5, 7)
    proj = torch.randn(3, 5, 4)
    global_mask = torch.tensor([True, True, False])

    prepared = prepare_graph_features(emb, proj, global_mask=global_mask)
    legacy = build_graph_features(emb, proj, global_mask=global_mask)

    assert torch.allclose(prepared, legacy)


@pytest.mark.unit
def test_prepare_graph_features_detaches_and_normalizes_by_default():
    torch.manual_seed(6)
    emb = torch.randn(3, 5, 7, requires_grad=True)
    proj = torch.randn(3, 5, 4, requires_grad=True)

    features = prepare_graph_features(emb, proj)

    assert not features.requires_grad
    assert torch.allclose(
        features.norm(dim=-1),
        torch.ones(features.shape[0]),
        atol=1e-6,
    )


@pytest.mark.unit
def test_prepare_graph_features_can_return_unnormalized_selected_features():
    emb = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )
    proj = torch.randn(2, 2, 3, requires_grad=True)

    features = prepare_graph_features(
        emb,
        proj,
        graph_space=GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER,
        normalized=False,
    )

    assert not features.requires_grad
    assert torch.equal(features, emb.detach().mean(dim=0))


@pytest.mark.unit
def test_build_graph_default_and_explicit_cosine_metric_match():
    torch.manual_seed(7)
    features = torch.randn(6, 4)

    default_graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=2)
    explicit_graph = build_graph(
        features,
        mode=GraphMode.MUTUAL_KNN,
        distance_metric=GraphDistanceMetric.COSINE,
        k=2,
    )

    assert torch.equal(default_graph.edge_index, explicit_graph.edge_index)
    assert torch.allclose(default_graph.edge_weight, explicit_graph.edge_weight)


@pytest.mark.unit
def test_build_graph_rejects_unknown_distance_metric():
    with pytest.raises(ValueError, match="euclidean"):
        build_graph(torch.eye(4), distance_metric="euclidean")


@pytest.mark.unit
def test_loss_logs_graph_feature_metric_contract():
    torch.manual_seed(8)
    proj = torch.randn(3, 4, 4, requires_grad=True)
    emb = torch.randn(3, 4, 5, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        k=2,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    output = loss_fn(proj, emb, return_output=True)

    assert output.extra_logs["GraphFeature/normalized"] == pytest.approx(1.0)
    assert output.extra_logs["GraphFeature/distance_metric/cosine"] == pytest.approx(
        1.0
    )
    assert output.extra_logs[
        "GraphFeature/feature_space/pre_projector_global_center"
    ] == pytest.approx(1.0)
    assert output.extra_logs[
        "GraphFeature/memory_stores_normalized_features"
    ] == pytest.approx(1.0)
    assert output.extra_logs["GraphFeature/norm_mean_current"] == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert output.extra_logs["GraphFeature/norm_std_current"] >= 0.0


@pytest.mark.unit
def test_loss_accepts_graph_feature_space_alias_and_checks_conflicts():
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        graph_feature_space=GraphSpace.PROJECTOR_GLOBAL_CENTER.value,
        graph_feature_normalized=False,
        memory_stores_normalized_features=False,
    )

    assert loss_fn.graph_space == GraphSpace.PROJECTOR_GLOBAL_CENTER
    assert loss_fn.graph_feature_space == GraphSpace.PROJECTOR_GLOBAL_CENTER
    assert loss_fn.graph_feature_normalized is False
    assert loss_fn.memory_stores_normalized_features is False

    with pytest.raises(ValueError, match="graph_space and graph_feature_space"):
        CoherentHardnessGeoDROLeJEPALoss(
            graph_space=GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER.value,
            graph_feature_space=GraphSpace.PROJECTOR_GLOBAL_CENTER.value,
        )


def _feature_memory_queue(capacity: int = 4) -> GeoDROFeatureMemoryQueue:
    return GeoDROFeatureMemoryQueue(
        capacity=capacity,
        graph_space=GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER,
        graph_distance_metric=GraphDistanceMetric.COSINE,
        graph_feature_normalized=True,
        update_scope=MemoryUpdateScope.MICROBATCH,
    )


@pytest.mark.unit
def test_memory_queue_empty_initial_state():
    queue = _feature_memory_queue(capacity=4)

    assert queue.valid_size == 0
    assert queue.cursor == 0
    assert queue.feature_dim == -1
    assert queue.valid_features().shape == (0, 0)
    assert queue.diagnostics()["Memory/fill_ratio"] == pytest.approx(0.0)


@pytest.mark.unit
def test_memory_queue_enqueue_detaches_and_tracks_fifo_order():
    queue = _feature_memory_queue(capacity=4)
    features = torch.arange(6.0).reshape(3, 2).requires_grad_()

    queue.enqueue(features, step=5)

    assert queue.valid_size == 3
    assert queue.cursor == 3
    assert queue.feature_dim == 2
    assert not queue.queue_features.requires_grad
    assert torch.equal(queue.valid_features(), features.detach())
    assert torch.equal(queue.valid_insertion_steps(), torch.tensor([5, 5, 5]))


@pytest.mark.unit
def test_memory_queue_wraparound_keeps_oldest_to_newest_valid_features():
    queue = _feature_memory_queue(capacity=4)
    first = torch.tensor([[0.0], [1.0], [2.0]])
    second = torch.tensor([[3.0], [4.0], [5.0]])

    queue.enqueue(first, step=0)
    queue.enqueue(second, step=1)

    assert queue.valid_size == 4
    assert queue.cursor == 2
    assert torch.equal(
        queue.valid_features().flatten(),
        torch.tensor([2.0, 3.0, 4.0, 5.0]),
    )
    assert torch.equal(
        queue.valid_insertion_steps(),
        torch.tensor([0, 1, 1, 1]),
    )


@pytest.mark.unit
def test_top_m_retrieval_correct_on_small_synthetic_features():
    queue = _feature_memory_queue(capacity=4)
    queue.enqueue(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        ),
        step=3,
    )
    current = torch.tensor([[0.98, 0.02], [0.05, 0.95]])

    witnesses = queue.retrieve_witnesses(
        current,
        top_m=2,
        k_sigma=2,
        min_fill_ratio=0.0,
    )

    assert witnesses.valid_memory_for_witnessing
    assert torch.equal(witnesses.indices.cpu(), torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(witnesses.insertion_steps.cpu(), torch.full((2, 2), 3))
    assert not witnesses.distances.requires_grad
    assert not witnesses.probabilities.requires_grad


@pytest.mark.unit
def test_chunked_retrieval_matches_full_retrieval():
    torch.manual_seed(13)
    queue = _feature_memory_queue(capacity=12)
    queue.enqueue(torch.randn(12, 5), step=1)
    current = torch.randn(4, 5)

    full = queue.retrieve_witnesses(
        current,
        top_m=4,
        k_sigma=5,
        min_fill_ratio=0.0,
        chunk_size=None,
    )
    chunked = queue.retrieve_witnesses(
        current,
        top_m=4,
        k_sigma=5,
        min_fill_ratio=0.0,
        chunk_size=3,
    )

    assert torch.equal(chunked.indices, full.indices)
    assert torch.allclose(chunked.distances, full.distances)
    assert torch.allclose(chunked.probabilities, full.probabilities)
    assert torch.allclose(chunked.sigma, full.sigma)


@pytest.mark.unit
def test_memory_sigma_uses_k_sigma_neighbor():
    queue = _feature_memory_queue(capacity=4)
    queue.enqueue(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.70710677, 0.70710677],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ),
        step=0,
    )

    witnesses = queue.retrieve_witnesses(
        torch.tensor([[1.0, 0.0]]),
        top_m=2,
        k_sigma=3,
        min_fill_ratio=0.0,
    )

    assert torch.equal(witnesses.indices.cpu(), torch.tensor([[0, 1]]))
    assert witnesses.sigma.item() == pytest.approx(1.0, abs=1e-6)
    expected_distances = torch.tensor([[0.0, 1.0 - 0.70710677]])
    expected_logits = -(expected_distances.square()) / (1.0 + 1e-12)
    assert torch.allclose(
        witnesses.probabilities.cpu(),
        torch.softmax(expected_logits, dim=1),
        atol=1e-6,
    )


@pytest.mark.unit
def test_witness_distribution_sums_to_one_and_logs_sharpness():
    queue = _feature_memory_queue(capacity=6)
    queue.enqueue(torch.randn(6, 3), step=2)

    witnesses = queue.retrieve_witnesses(
        torch.randn(3, 3),
        top_m=3,
        k_sigma=3,
        min_fill_ratio=0.0,
    )
    logs = witnesses.diagnostics()

    assert torch.allclose(
        witnesses.probabilities.sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )
    assert logs["MemoryWitness/distribution_entropy_mean"] >= 0.0
    assert logs["MemoryWitness/distribution_perplexity_mean"] >= 1.0
    assert 0.0 <= logs["MemoryWitness/top1_mass_mean"] <= 1.0
    assert 0.0 <= logs["MemoryWitness/top8_mass_mean"] <= 1.0


@pytest.mark.unit
def test_underfilled_memory_returns_invalid_witness_state():
    queue = _feature_memory_queue(capacity=8)
    queue.enqueue(torch.randn(3, 2), step=1)

    witnesses = queue.retrieve_witnesses(
        torch.randn(2, 2),
        top_m=4,
        k_sigma=4,
        min_fill_ratio=0.0,
    )
    logs = witnesses.diagnostics()

    assert not witnesses.valid_memory_for_witnessing
    assert witnesses.indices.shape == (2, 0)
    assert witnesses.probabilities.shape == (2, 0)
    assert witnesses.sigma.shape == (2,)
    assert logs["MemoryWitness/valid_memory_for_witnessing"] == pytest.approx(0.0)
    assert logs["MemoryWitness/distribution_entropy_mean"] == pytest.approx(0.0)


@pytest.mark.unit
def test_fill_ratio_gates_valid_witnessing_but_keeps_diagnostics():
    queue = _feature_memory_queue(capacity=10)
    queue.enqueue(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        ),
        step=1,
    )

    witnesses = queue.retrieve_witnesses(
        torch.tensor([[1.0, 0.0]]),
        top_m=2,
        k_sigma=2,
        min_fill_ratio=0.5,
    )
    logs = witnesses.diagnostics()

    assert witnesses.probabilities.shape == (1, 2)
    assert not witnesses.valid_memory_for_witnessing
    assert logs["MemoryWitness/valid_memory_for_witnessing"] == pytest.approx(0.0)
    assert logs["MemoryWitness/top1_mass_mean"] > 0.0


@pytest.mark.unit
def test_no_grad_through_memory_retrieval():
    queue = _feature_memory_queue(capacity=5)
    queue.enqueue(torch.randn(5, 3, requires_grad=True), step=0)
    current = torch.randn(2, 3, requires_grad=True)

    witnesses = queue.retrieve_witnesses(
        current,
        top_m=3,
        k_sigma=3,
        min_fill_ratio=0.0,
    )

    assert not witnesses.indices.requires_grad
    assert not witnesses.distances.requires_grad
    assert not witnesses.probabilities.requires_grad
    assert not witnesses.sigma.requires_grad
    assert not queue.queue_features.requires_grad


def _manual_witness_batch(
    indices: torch.Tensor,
    probabilities: torch.Tensor | None = None,
    *,
    valid: bool = True,
) -> MemoryWitnessBatch:
    if probabilities is None:
        probabilities = torch.ones_like(indices, dtype=torch.float32)
    return MemoryWitnessBatch(
        indices=indices.long(),
        distances=torch.zeros_like(probabilities, dtype=torch.float32),
        probabilities=probabilities.float(),
        sigma=torch.ones((indices.shape[0],), dtype=torch.float32),
        insertion_steps=torch.zeros_like(indices, dtype=torch.long),
        valid_memory_for_witnessing=valid,
        retrieval_time_ms=0.0,
        top_m=int(indices.shape[1]) if indices.ndim == 2 else 0,
        k_sigma=int(indices.shape[1]) if indices.ndim == 2 else 0,
    )


@pytest.mark.unit
def test_one_hot_popular_hub_score_lower_than_one_hot_rare_witness():
    witnesses = _manual_witness_batch(
        torch.tensor(
            [
                [0],
                [0],
                [0],
                [1],
                [1],
            ]
        )
    )

    scores = compute_witness_overlap_scores(witnesses)

    assert scores.raw_overlap[0, 1] == pytest.approx(1.0)
    assert scores.raw_overlap[3, 4] == pytest.approx(1.0)
    assert scores.specificity_overlap[0, 1] < scores.specificity_overlap[3, 4]


@pytest.mark.unit
def test_W_spec_in_0_1():
    witnesses = _manual_witness_batch(
        torch.tensor(
            [
                [0, 1],
                [0, 2],
                [1, 2],
            ]
        ),
        probabilities=torch.tensor(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.5, 0.5],
            ]
        ),
    )

    scores = compute_witness_overlap_scores(witnesses)

    assert torch.all(scores.specificity_overlap >= 0.0)
    assert torch.all(scores.specificity_overlap <= 1.0 + 1e-6)


@pytest.mark.unit
def test_W_spec_less_or_equal_raw_overlap():
    witnesses = _manual_witness_batch(
        torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 3, 4],
                [2, 4, 5],
            ]
        ),
        probabilities=torch.full((4, 3), 1.0 / 3.0),
    )

    scores = compute_witness_overlap_scores(witnesses)

    assert torch.all(scores.specificity_overlap <= scores.raw_overlap + 1e-6)


@pytest.mark.unit
def test_raw_and_specificity_differ_when_popularity_differs():
    witnesses = _manual_witness_batch(
        torch.tensor(
            [
                [0],
                [0],
                [0],
                [1],
                [1],
            ]
        )
    )

    scores = compute_witness_overlap_scores(witnesses)

    assert torch.allclose(scores.raw_overlap[0, 1], scores.raw_overlap[3, 4])
    assert not torch.allclose(
        scores.specificity_overlap[0, 1],
        scores.specificity_overlap[3, 4],
    )


@pytest.mark.unit
def test_witness_score_mode_selects_raw_or_specificity_overlap():
    witnesses = _manual_witness_batch(
        torch.tensor(
            [
                [0],
                [0],
                [0],
                [1],
                [1],
            ]
        )
    )

    raw_scores = compute_witness_overlap_scores(
        witnesses,
        mode=WitnessScoreMode.RAW_HELLINGER,
    )
    spec_scores = compute_witness_overlap_scores(
        witnesses,
        mode=WitnessScoreMode.SPECIFICITY_WEIGHTED_HELLINGER,
    )

    assert torch.equal(raw_scores.selected_overlap, raw_scores.raw_overlap)
    assert torch.equal(spec_scores.selected_overlap, spec_scores.specificity_overlap)
    assert raw_scores.diagnostics()[
        "MemoryWitness/witness_score_mode/raw_hellinger"
    ] == pytest.approx(1.0)
    assert spec_scores.diagnostics()[
        "MemoryWitness/witness_score_mode/specificity_weighted_hellinger"
    ] == pytest.approx(1.0)


@pytest.mark.unit
def test_overlap_scores_detached():
    witnesses = _manual_witness_batch(
        torch.tensor(
            [
                [0, 1],
                [0, 2],
            ]
        ),
        probabilities=torch.tensor(
            [
                [0.7, 0.3],
                [0.4, 0.6],
            ],
            requires_grad=True,
        ),
    )

    scores = compute_witness_overlap_scores(witnesses)

    assert not scores.raw_overlap.requires_grad
    assert not scores.specificity_overlap.requires_grad
    assert not scores.selected_overlap.requires_grad
    assert not scores.selected_counts.requires_grad
    assert not scores.specificity_weights.requires_grad


@pytest.mark.unit
def test_invalid_witness_overlap_returns_zero_diagnostics():
    witnesses = _manual_witness_batch(
        torch.empty((3, 0), dtype=torch.long),
        probabilities=torch.empty((3, 0)),
        valid=False,
    )

    scores = compute_witness_overlap_scores(witnesses)
    logs = scores.diagnostics()

    assert not scores.valid_for_scoring
    assert torch.equal(scores.raw_overlap, torch.zeros(3, 3))
    assert logs["MemoryWitness/selected_count_mean"] == pytest.approx(0.0)
    assert logs["MemoryWitness/raw_overlap_mean"] == pytest.approx(0.0)
    assert logs["MemoryWitness/spec_overlap_mean"] == pytest.approx(0.0)


def _raw_all_shared_overlap(batch_size: int):
    witnesses = _manual_witness_batch(torch.zeros((batch_size, 1), dtype=torch.long))
    return compute_witness_overlap_scores(
        witnesses,
        mode=WitnessScoreMode.RAW_HELLINGER,
    )


def _edge_set(graph):
    if graph.edge_index.numel() == 0:
        return set()
    return {tuple(edge) for edge in graph.edge_index.detach().cpu().T.tolist()}


def _added_edge_set(memory_graph, batch_graph):
    return _edge_set(memory_graph) - _edge_set(batch_graph)


def _reference_memory_added_edges(
    features,
    batch_graph,
    selected_overlap,
    *,
    k,
    memory_k_guard,
    memory_witness_score_min,
    memory_extra_edges_per_node_max,
    memory_added_edge_cap,
    eps=1e-12,
):
    normalized = torch.nn.functional.normalize(features, dim=-1)
    distances = (1.0 - normalized @ normalized.T).clamp_min(0.0)
    distances.fill_diagonal_(float("inf"))
    num_nodes = int(features.shape[0])
    k_eff = max(1, min(int(k), num_nodes - 1))
    knn_dist, _ = torch.topk(distances, k=k_eff, dim=1, largest=False)
    sigma = knn_dist[:, -1].clamp_min(eps)
    base_adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    if batch_graph.edge_index.numel() > 0:
        row, col = batch_graph.edge_index.cpu()
        base_adjacency[row, col] = True
        base_adjacency[col, row] = True

    _, guard_idx = torch.topk(
        distances,
        k=min(int(memory_k_guard), num_nodes - 1),
        dim=1,
        largest=False,
    )
    guard = torch.zeros_like(base_adjacency)
    guard.scatter_(1, guard_idx, True)
    guard = guard | guard.T
    guard.fill_diagonal_(False)
    row, col = torch.triu(guard & ~base_adjacency, diagonal=1).nonzero(as_tuple=True)
    witness_score = selected_overlap[row, col]
    keep = witness_score >= float(memory_witness_score_min)
    row = row[keep]
    col = col[keep]
    witness_score = witness_score[keep]
    current_weight = torch.exp(
        -(distances[row, col].square()) / (sigma[row] * sigma[col]).clamp_min(eps)
    )
    rank_score = witness_score.float() * current_weight.float()
    order = sorted(
        range(int(row.numel())),
        key=lambda idx: (
            -float(rank_score[idx]),
            int(row[idx]),
            int(col[idx]),
        ),
    )
    added_degree = [0 for _ in range(num_nodes)]
    selected = []
    for idx in order:
        left = int(row[idx])
        right = int(col[idx])
        if added_degree[left] >= memory_extra_edges_per_node_max:
            continue
        if added_degree[right] >= memory_extra_edges_per_node_max:
            continue
        selected.append((left, right))
        added_degree[left] += 1
        added_degree[right] += 1
        if len(selected) >= memory_added_edge_cap:
            break
    return set(selected)


@pytest.mark.unit
def test_existing_batch_edges_do_not_consume_memory_budget():
    features = torch.eye(4)
    batch_graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=1)
    scores = _raw_all_shared_overlap(batch_size=4)

    memory_graph, logs = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MUTUAL_KNN,
        k=1,
        memory_k_guard=3,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=8,
        memory_added_edge_ratio_max=8.0,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
    )

    all_possible_edges = 6
    expected_new_edges = all_possible_edges - batch_graph.diagnostics.num_edges
    assert logs["MemoryWitness/added_edges_before_budget"] == pytest.approx(
        expected_new_edges
    )
    assert logs["Graph/memory_added_edges"] == pytest.approx(expected_new_edges)
    assert len(_added_edge_set(memory_graph, batch_graph)) == expected_new_edges


@pytest.mark.unit
def test_incident_added_degree_never_exceeds_b_t():
    features = torch.eye(6)
    batch_graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=1)
    scores = _raw_all_shared_overlap(batch_size=6)

    memory_graph, logs = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MUTUAL_KNN,
        k=1,
        memory_k_guard=5,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=1,
        memory_added_edge_ratio_max=8.0,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
    )

    added_edges = _added_edge_set(memory_graph, batch_graph)
    added_degree = torch.zeros(6, dtype=torch.long)
    for left, right in added_edges:
        added_degree[left] += 1
        added_degree[right] += 1
    assert int(added_degree.max().item()) <= 1
    assert logs["MemoryWitness/extra_edges_per_node_budget_eff"] == pytest.approx(1.0)


@pytest.mark.unit
def test_global_cap_respected_after_deduplication():
    features = torch.eye(6)
    batch_graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=1)
    scores = _raw_all_shared_overlap(batch_size=6)

    memory_graph, logs = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MUTUAL_KNN,
        k=1,
        memory_k_guard=5,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=8,
        memory_added_edge_ratio_max=0.5,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
    )

    cap = int(0.5 * batch_graph.diagnostics.num_edges)
    assert len(_added_edge_set(memory_graph, batch_graph)) <= cap
    assert logs["MemoryWitness/global_added_edge_cap_eff"] == pytest.approx(cap)


@pytest.mark.unit
def test_duplicate_endpoint_proposals_create_one_edge():
    features = torch.eye(4)
    scores = _raw_all_shared_overlap(batch_size=4)

    memory_graph, _ = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MUTUAL_KNN,
        k=1,
        memory_k_guard=3,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=8,
        memory_added_edge_ratio_max=8.0,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
    )

    edges = list(_edge_set(memory_graph))
    assert len(edges) == len(set(edges))
    assert all(left < right for left, right in edges)


@pytest.mark.unit
def test_stable_tie_breaking_for_memory_added_edges():
    features = torch.eye(5)
    batch_graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=1)
    scores = _raw_all_shared_overlap(batch_size=5)
    candidate_edges = sorted(
        {(left, right) for left in range(5) for right in range(left + 1, 5)}
        - _edge_set(batch_graph)
    )

    memory_graph, _ = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MUTUAL_KNN,
        k=1,
        memory_k_guard=4,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=1,
        memory_added_edge_ratio_max=1.0,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
    )

    added_edges = sorted(_added_edge_set(memory_graph, batch_graph))
    assert added_edges[0] == candidate_edges[0]


@pytest.mark.unit
def test_memory_edge_selection_matches_reference_greedy_budget():
    torch.manual_seed(119)
    features = torch.randn(9, 5)
    k = 2
    memory_k_guard = 5
    memory_extra_edges_per_node_max = 2
    batch_graph = build_graph(features, mode=GraphMode.MAX_UNION_KNN, k=k)
    selected_overlap = torch.rand(9, 9)
    selected_overlap = (selected_overlap + selected_overlap.T) / 2.0
    selected_overlap.fill_diagonal_(0.0)
    raw_overlap = torch.rand(9, 9)
    raw_overlap = (raw_overlap + raw_overlap.T) / 2.0
    raw_overlap.fill_diagonal_(0.0)
    scores = MemoryWitnessOverlapScores(
        raw_overlap=raw_overlap,
        specificity_overlap=selected_overlap,
        selected_overlap=selected_overlap,
        selected_counts=torch.ones(3),
        specificity_weights=torch.ones(3),
        witness_score_mode=WitnessScoreMode.SPECIFICITY_WEIGHTED_HELLINGER,
        valid_for_scoring=True,
    )
    memory_added_edge_cap = int(batch_graph.diagnostics.num_edges)

    result = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MAX_UNION_KNN,
        k=k,
        memory_k_guard=memory_k_guard,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=memory_extra_edges_per_node_max,
        memory_added_edge_ratio_max=1.0,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
        return_result=True,
    )

    expected = _reference_memory_added_edges(
        features,
        batch_graph,
        selected_overlap,
        k=k,
        memory_k_guard=memory_k_guard,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=memory_extra_edges_per_node_max,
        memory_added_edge_cap=memory_added_edge_cap,
    )
    assert {tuple(edge) for edge in result.selected_edges.cpu().tolist()} == expected
    assert _added_edge_set(result.graph, batch_graph) == expected


@pytest.mark.unit
def test_memory_added_edges_use_current_only_edge_weights():
    features = torch.eye(4)
    batch_graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=1)
    scores = _raw_all_shared_overlap(batch_size=4)

    memory_graph, _ = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MUTUAL_KNN,
        k=1,
        memory_k_guard=3,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=8,
        memory_added_edge_ratio_max=8.0,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
    )

    added_edges = _added_edge_set(memory_graph, batch_graph)
    weights = {
        tuple(edge): weight
        for edge, weight in zip(
            memory_graph.edge_index.detach().cpu().T.tolist(),
            memory_graph.edge_weight.detach().cpu().tolist(),
            strict=True,
        )
    }
    for edge in added_edges:
        assert weights[edge] == pytest.approx(float(torch.exp(torch.tensor(-1.0))))


@pytest.mark.unit
def test_E_final_contains_E_batch_and_memory_cannot_remove_batch_edges():
    features = torch.eye(5)
    batch_graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=1)
    scores = _raw_all_shared_overlap(batch_size=5)

    memory_graph, _ = build_memory_witnessed_graph(
        features,
        scores,
        mode=GraphMode.MUTUAL_KNN,
        k=1,
        memory_k_guard=4,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=1.0,
        fill_ratio=1.0,
        memory_min_fill_ratio=0.0,
    )

    assert _edge_set(batch_graph).issubset(_edge_set(memory_graph))
    assert memory_graph.num_nodes == batch_graph.num_nodes


@pytest.mark.unit
def test_batch_memory_config_instantiates_inert_queue_only():
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=8,
    )

    assert loss_fn.feature_memory is not None
    assert loss_fn.feature_memory.valid_size == 0
    assert loss_fn.memory_update_scope == MemoryUpdateScope.MICROBATCH


@pytest.mark.unit
def test_batch_memory_requires_positive_capacity_and_matching_update_scope():
    with pytest.raises(ValueError, match="positive memory_queue_capacity"):
        CoherentHardnessGeoDROLeJEPALoss(
            geometry_support=GeometrySupport.BATCH_MEMORY.value,
            memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        )

    with pytest.raises(
        ValueError,
        match="optimizer_step requires memory_update_scope=optimizer_step_delayed",
    ):
        CoherentHardnessGeoDROLeJEPALoss(
            geometry_support=GeometrySupport.BATCH_MEMORY.value,
            memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
            memory_queue_capacity=8,
            adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
        )

    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=8,
        adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
        memory_update_scope=MemoryUpdateScope.OPTIMIZER_STEP_DELAYED.value,
    )
    assert loss_fn.feature_memory is not None
    assert loss_fn.memory_update_scope == MemoryUpdateScope.OPTIMIZER_STEP_DELAYED

    graph_transport_loss = GraphTransportGeoDROJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=8,
        adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
        memory_update_scope=MemoryUpdateScope.OPTIMIZER_STEP_DELAYED.value,
    )
    assert graph_transport_loss.feature_memory is not None
    assert graph_transport_loss.memory_update_scope == (
        MemoryUpdateScope.OPTIMIZER_STEP_DELAYED
    )

    with pytest.raises(
        ValueError,
        match="microbatch requires memory_update_scope=microbatch",
    ):
        CoherentHardnessGeoDROLeJEPALoss(
            geometry_support=GeometrySupport.BATCH_MEMORY.value,
            memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
            memory_queue_capacity=8,
            memory_update_scope=MemoryUpdateScope.OPTIMIZER_STEP_DELAYED.value,
        )

    with pytest.raises(ValueError, match="memory_usage_mode=memory_witnessed"):
        CoherentHardnessGeoDROLeJEPALoss(
            geometry_support=GeometrySupport.BATCH_MEMORY.value,
            memory_usage_mode=MemoryUsageMode.NONE.value,
            memory_queue_capacity=8,
        )

    with pytest.raises(ValueError, match="memory_k_guard"):
        CoherentHardnessGeoDROLeJEPALoss(memory_k_guard=0)

    with pytest.raises(ValueError, match="memory_witness_score_min"):
        CoherentHardnessGeoDROLeJEPALoss(memory_witness_score_min=1.5)

    with pytest.raises(ValueError, match="memory_extra_edges_per_node_max"):
        CoherentHardnessGeoDROLeJEPALoss(memory_extra_edges_per_node_max=-1)

    with pytest.raises(ValueError, match="memory_added_edge_ratio_max"):
        CoherentHardnessGeoDROLeJEPALoss(memory_added_edge_ratio_max=-0.1)


@pytest.mark.unit
def test_batch_memory_forward_updates_queue_after_current_graph_build(monkeypatch):
    torch.manual_seed(9)
    proj = torch.randn(3, 4, 4, requires_grad=True)
    emb = torch.randn(3, 4, 5, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=8,
        lambda_=0.0,
        alpha_max=0.0,
        k=2,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    original_build_graph = loss_module.build_graph
    observed_pre_graph_memory_size = []

    def assert_memory_empty_during_graph_build(*args, **kwargs):
        observed_pre_graph_memory_size.append(loss_fn.feature_memory.valid_size)
        return original_build_graph(*args, **kwargs)

    monkeypatch.setattr(
        loss_module, "build_graph", assert_memory_empty_during_graph_build
    )

    output = loss_fn(proj, emb, step=7, return_output=True)

    assert observed_pre_graph_memory_size == [0]
    assert output.extra_logs["Memory/size"] == pytest.approx(0.0)
    assert output.extra_logs["Memory/fill_ratio"] == pytest.approx(0.0)
    assert loss_fn.feature_memory.valid_size == 4
    assert torch.equal(
        loss_fn.feature_memory.valid_insertion_steps(),
        torch.tensor([7, 7, 7, 7]),
    )


@pytest.mark.unit
def test_batch_memory_phase5_matches_batch_only_outputs_and_keeps_graph_current_only():
    torch.manual_seed(10)
    proj = torch.randn(3, 5, 4, requires_grad=True)
    emb = torch.randn(3, 5, 6, requires_grad=True)
    common = dict(
        lambda_=0.0,
        alpha_max=0.0,
        graph_mode=GraphMode.FULLY_CONNECTED.value,
        k=2,
        min_graph_nodes=1,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    batch_loss = CoherentHardnessGeoDROLeJEPALoss(**common)
    memory_loss = CoherentHardnessGeoDROLeJEPALoss(
        **common,
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
    )

    batch_output = batch_loss(proj, emb, return_output=True)
    memory_output = memory_loss(proj, emb, return_output=True)

    assert torch.allclose(memory_output.total_loss, batch_output.total_loss)
    assert torch.allclose(memory_output.pred_loss, batch_output.pred_loss)
    assert torch.allclose(memory_output.p_global, batch_output.p_global)
    assert memory_output.graph_diagnostics.num_nodes == 5
    assert memory_output.p_global.numel() == 5
    assert memory_loss.feature_memory.valid_size == 5
    memory_output.total_loss.backward()
    assert proj.grad is not None
    assert emb.grad is None
    assert not memory_loss.feature_memory.queue_features.requires_grad


@pytest.mark.unit
def test_phase6_memory_retrieval_does_not_change_batch_only_loss_or_graph():
    torch.manual_seed(11)
    proj = torch.randn(3, 5, 4, requires_grad=True)
    emb = torch.randn(3, 5, 6, requires_grad=True)
    common = dict(
        lambda_=0.0,
        alpha_max=0.2,
        graph_mode=GraphMode.FULLY_CONNECTED.value,
        k=2,
        min_graph_nodes=1,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    batch_loss = CoherentHardnessGeoDROLeJEPALoss(**common)
    memory_loss = CoherentHardnessGeoDROLeJEPALoss(
        **common,
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
        memory_top_m=4,
        memory_k_sigma=4,
        memory_min_fill_ratio=0.25,
        memory_retrieval_chunk_size=3,
        witness_score_mode=WitnessScoreMode.SPECIFICITY_WEIGHTED_HELLINGER.value,
    )
    memory_loss.feature_memory.enqueue(torch.randn(8, 6), step=-1)

    batch_output = batch_loss(proj, emb, step=3, total_steps=10, return_output=True)
    memory_output = memory_loss(proj, emb, step=3, total_steps=10, return_output=True)

    assert torch.allclose(memory_output.total_loss, batch_output.total_loss)
    assert torch.allclose(memory_output.pred_loss, batch_output.pred_loss)
    assert torch.allclose(memory_output.p_global, batch_output.p_global)
    assert memory_output.graph_diagnostics.num_nodes == 5
    assert memory_output.p_global.numel() == 5
    assert memory_output.extra_logs[
        "MemoryWitness/valid_memory_for_witnessing"
    ] == pytest.approx(1.0)
    assert memory_output.extra_logs["MemoryWitness/top1_mass_mean"] > 0.0
    assert memory_output.extra_logs["Memory/retrieval_time_ms"] >= 0.0
    assert memory_output.extra_logs[
        "MemoryWitness/witness_score_mode/specificity_weighted_hellinger"
    ] == pytest.approx(1.0)
    assert memory_output.extra_logs["MemoryWitness/selected_count_mean"] > 0.0
    assert memory_output.extra_logs["MemoryWitness/raw_overlap_mean"] >= 0.0
    assert memory_output.extra_logs["MemoryWitness/spec_overlap_mean"] >= 0.0


@pytest.mark.unit
def test_batch_memory_requires_explicit_threshold_when_edges_can_be_added():
    torch.manual_seed(111)
    proj = torch.randn(3, 5, 4, requires_grad=True)
    emb = torch.randn(3, 5, 6, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
        memory_top_m=1,
        memory_k_sigma=1,
        memory_min_fill_ratio=0.0,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        memory_k_guard=4,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=2.0,
        lambda_=0.0,
        alpha_max=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.zeros(8, 6), step=-1)

    with pytest.raises(ValueError, match="memory_witness_score_min"):
        loss_fn(proj, emb, return_output=True)


@pytest.mark.unit
def test_batch_memory_can_add_current_only_edges_with_explicit_threshold():
    torch.manual_seed(112)
    proj = torch.randn(3, 5, 4, requires_grad=True)
    emb = torch.randn(3, 5, 6, requires_grad=True)
    batch_loss = CoherentHardnessGeoDROLeJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        min_graph_nodes=1,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    memory_loss = CoherentHardnessGeoDROLeJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        min_graph_nodes=1,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
        memory_top_m=1,
        memory_k_sigma=1,
        memory_min_fill_ratio=0.0,
        memory_k_guard=4,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=4.0,
        witness_score_mode=WitnessScoreMode.RAW_HELLINGER.value,
    )
    memory_loss.feature_memory.enqueue(torch.zeros(8, 6), step=-1)

    batch_output = batch_loss(proj, emb, step=3, total_steps=10, return_output=True)
    memory_output = memory_loss(proj, emb, step=3, total_steps=10, return_output=True)

    assert memory_output.graph_diagnostics.num_nodes == 5
    assert memory_output.p_global.numel() == 5
    assert (
        memory_output.graph_diagnostics.num_edges
        >= batch_output.graph_diagnostics.num_edges
    )
    assert memory_output.extra_logs["Graph/batch_edges"] == pytest.approx(
        batch_output.graph_diagnostics.num_edges
    )
    assert memory_output.extra_logs["Graph/final_edges"] == pytest.approx(
        memory_output.graph_diagnostics.num_edges
    )
    assert memory_output.extra_logs["Graph/memory_added_edges"] >= 0.0
    assert memory_output.extra_logs["MemoryWitness/threshold_value"] == pytest.approx(
        0.0
    )
    assert not memory_loss.feature_memory.queue_features.requires_grad


@pytest.mark.unit
def test_coherent_hardness_batch_memory_uses_final_smoothing_graph_by_default(
    monkeypatch,
):
    torch.manual_seed(113)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 6, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
        memory_top_m=1,
        memory_k_sigma=1,
        memory_min_fill_ratio=0.0,
        memory_k_guard=5,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=4.0,
        witness_score_mode=WitnessScoreMode.RAW_HELLINGER.value,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        lambda_=0.0,
        alpha_max=0.0,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.zeros(8, 6), step=-1)
    utility_graph_edges = []
    original_build_utility = loss_module.build_utility

    def capture_utility_graph(*args, **kwargs):
        utility_graph_edges.append(args[2].diagnostics.num_edges)
        return original_build_utility(*args, **kwargs)

    monkeypatch.setattr(loss_module, "build_utility", capture_utility_graph)

    output = loss_fn(proj, emb, return_output=True)

    assert output.extra_logs["Graph/memory_added_edges"] > 0.0
    assert utility_graph_edges == [output.graph_diagnostics.num_edges]
    assert output.extra_logs["Utility/smoothing_graph/final"] == pytest.approx(1.0)
    assert output.extra_logs["Utility/smoothing_graph_active"] == pytest.approx(1.0)


@pytest.mark.unit
def test_coherent_hardness_batch_memory_can_use_batch_smoothing_graph_ablation(
    monkeypatch,
):
    torch.manual_seed(114)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 6, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
        memory_top_m=1,
        memory_k_sigma=1,
        memory_min_fill_ratio=0.0,
        memory_k_guard=5,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=4.0,
        witness_score_mode=WitnessScoreMode.RAW_HELLINGER.value,
        utility_smoothing_graph=UtilitySmoothingGraph.BATCH.value,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        lambda_=0.0,
        alpha_max=0.0,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.zeros(8, 6), step=-1)
    utility_graph_edges = []
    original_build_utility = loss_module.build_utility

    def capture_utility_graph(*args, **kwargs):
        utility_graph_edges.append(args[2].diagnostics.num_edges)
        return original_build_utility(*args, **kwargs)

    monkeypatch.setattr(loss_module, "build_utility", capture_utility_graph)

    output = loss_fn(proj, emb, return_output=True)

    assert output.extra_logs["Graph/memory_added_edges"] > 0.0
    assert output.graph_diagnostics.num_edges > output.extra_logs["Graph/batch_edges"]
    assert utility_graph_edges == [int(output.extra_logs["Graph/batch_edges"])]
    assert output.extra_logs["Utility/smoothing_graph/batch"] == pytest.approx(1.0)
    assert output.extra_logs["Utility/smoothing_graph_active"] == pytest.approx(1.0)


@pytest.mark.unit
def test_graph_transport_batch_memory_ignores_smoothing_graph_option():
    torch.manual_seed(115)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 6, requires_grad=True)
    common = dict(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
        memory_top_m=1,
        memory_k_sigma=1,
        memory_min_fill_ratio=0.0,
        memory_k_guard=5,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=4.0,
        witness_score_mode=WitnessScoreMode.RAW_HELLINGER.value,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        lambda_=0.0,
        alpha_max=0.5,
        min_graph_nodes=1,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    final_loss = GraphTransportGeoDROJEPALoss(
        **common,
        utility_smoothing_graph=UtilitySmoothingGraph.FINAL.value,
    )
    batch_loss = GraphTransportGeoDROJEPALoss(
        **common,
        utility_smoothing_graph=UtilitySmoothingGraph.BATCH.value,
    )
    final_loss.feature_memory.enqueue(torch.zeros(8, 6), step=-1)
    batch_loss.feature_memory.enqueue(torch.zeros(8, 6), step=-1)

    final_output = final_loss(proj, emb, step=3, total_steps=10, return_output=True)
    batch_output = batch_loss(proj, emb, step=3, total_steps=10, return_output=True)

    assert torch.allclose(final_output.total_loss, batch_output.total_loss)
    assert torch.allclose(final_output.p_global, batch_output.p_global)
    assert final_output.extra_logs["Graph/memory_added_edges"] > 0.0
    assert batch_output.extra_logs["Graph/memory_added_edges"] > 0.0
    assert final_output.graph_diagnostics.num_nodes == 6
    assert batch_output.p_global.numel() == 6
    assert final_output.extra_logs["Utility/smoothing_graph_active"] == pytest.approx(
        0.0
    )
    assert batch_output.extra_logs["Utility/smoothing_graph_active"] == pytest.approx(
        0.0
    )
    assert not final_loss.feature_memory.queue_features.requires_grad
    assert not batch_loss.feature_memory.queue_features.requires_grad


@pytest.mark.unit
def test_batch_memory_required_diagnostics_present():
    torch.manual_seed(116)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 6, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=24,
        memory_top_m=2,
        memory_k_sigma=2,
        memory_min_fill_ratio=0.0,
        memory_k_guard=5,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=4.0,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        lambda_=0.0,
        alpha_max=0.0,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.randn(12, 6), step=-1)

    output = loss_fn(proj, emb, step=5, total_steps=10, return_output=True)

    required = {
        "Config/queue_capacity",
        "Config/memory_witness_score_min",
        "Memory/graph_build_time_ms",
        "Memory/peak_allocated_mb_optional",
        "MemoryWitness/threshold_mode/explicit",
        "MemoryWitness/ablation_mode/none",
        "MemoryWitness/null_score_mean",
        "MemoryWitness/added_edges",
        "MemoryWitness/added_edges_per_node_mean",
        "MemoryWitness/raw_vs_spec_added_edge_agreement",
        "MemoryWitness/added_edges_supported_by_top_hubs_fraction",
        "MemoryWitness/mean_witness_age_for_added_edges",
        "MemoryWitness/utility_coherence_for_kept_edges",
        "Utility/loss_standardized_positive_fraction",
        "Utility/view_reliability_by_loss_sign",
        "Utility/mean_weight_for_positive_utility",
        "Utility/mean_weight_for_negative_utility",
    }
    assert required.issubset(output.extra_logs)
    assert output.extra_logs["Memory/graph_build_time_ms"] >= 0.0
    assert output.extra_logs["MemoryWitness/threshold_value"] == pytest.approx(0.0)


@pytest.mark.unit
def test_missing_required_batch_memory_diagnostic_raises(monkeypatch):
    torch.manual_seed(117)
    proj = torch.randn(3, 5, 4, requires_grad=True)
    emb = torch.randn(3, 5, 6, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=16,
        memory_top_m=1,
        memory_k_sigma=1,
        memory_min_fill_ratio=0.0,
        memory_k_guard=4,
        memory_witness_score_min=0.0,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=4.0,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        lambda_=0.0,
        alpha_max=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.randn(8, 6), step=-1)
    monkeypatch.setattr(
        loss_module,
        "_REQUIRED_BATCH_MEMORY_LOGS",
        set(loss_module._REQUIRED_BATCH_MEMORY_LOGS) | {"Missing/diagnostic"},
    )

    with pytest.raises(RuntimeError, match="Missing required GeoDRO"):
        loss_fn(proj, emb, return_output=True)


@pytest.mark.unit
def test_shuffled_memory_deterministic_under_seed():
    torch.manual_seed(118)
    graph_features = torch.randn(6, 5)
    memory_features = torch.randn(16, 5)

    def overlaps(seed):
        loss_fn = CoherentHardnessGeoDROLeJEPALoss(
            geometry_support=GeometrySupport.BATCH_MEMORY.value,
            memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
            memory_queue_capacity=16,
            memory_top_m=3,
            memory_k_sigma=3,
            memory_min_fill_ratio=0.0,
            memory_witness_ablation_mode=MemoryWitnessAblationMode.SHUFFLED_MEMORY.value,
            memory_witness_null_seed=seed,
            memory_witness_score_min=0.0,
        )
        loss_fn.feature_memory.enqueue(memory_features, step=-1)
        witnesses, scores, _, logs = loss_fn._memory_witness_state(
            graph_features,
            step=4,
        )
        return witnesses, scores.selected_overlap, logs

    witnesses_a, overlap_a, logs_a = overlaps(7)
    witnesses_b, overlap_b, _ = overlaps(7)
    witnesses_c, overlap_c, _ = overlaps(8)

    assert torch.allclose(overlap_a, overlap_b)
    assert torch.allclose(witnesses_a.probabilities, witnesses_b.probabilities)
    assert torch.allclose(witnesses_a.probabilities, witnesses_c.probabilities)
    assert not torch.allclose(overlap_a, overlap_c)
    assert logs_a["MemoryWitness/ablation_mode/shuffled_memory"] == pytest.approx(1.0)
    assert logs_a["MemoryWitness/null_type/shuffled_memory"] == pytest.approx(1.0)


@pytest.mark.unit
def test_random_memory_deterministic_under_seed():
    torch.manual_seed(119)
    graph_features = torch.randn(6, 5)
    memory_features = torch.randn(16, 5)

    def overlaps(seed):
        loss_fn = CoherentHardnessGeoDROLeJEPALoss(
            geometry_support=GeometrySupport.BATCH_MEMORY.value,
            memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
            memory_queue_capacity=16,
            memory_top_m=3,
            memory_k_sigma=3,
            memory_min_fill_ratio=0.0,
            memory_witness_ablation_mode=MemoryWitnessAblationMode.RANDOM_MEMORY.value,
            memory_witness_null_seed=seed,
            memory_witness_score_min=0.0,
        )
        loss_fn.feature_memory.enqueue(memory_features, step=-1)
        _, scores, _, logs = loss_fn._memory_witness_state(graph_features, step=4)
        return scores.selected_overlap, logs

    overlap_a, logs_a = overlaps(11)
    overlap_b, _ = overlaps(11)
    overlap_c, _ = overlaps(12)

    assert torch.allclose(overlap_a, overlap_b)
    assert not torch.allclose(overlap_a, overlap_c)
    assert logs_a["MemoryWitness/ablation_mode/random_memory"] == pytest.approx(1.0)
    assert logs_a["MemoryWitness/null_type/random_memory"] == pytest.approx(1.0)


@pytest.mark.unit
def test_shuffled_null_quantile_threshold_is_logged_and_does_not_require_explicit_min():
    torch.manual_seed(120)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 6, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=24,
        memory_top_m=2,
        memory_k_sigma=2,
        memory_min_fill_ratio=0.0,
        memory_k_guard=5,
        memory_witness_score_min=None,
        memory_witness_threshold_mode=(
            MemoryWitnessThresholdMode.SHUFFLED_NULL_QUANTILE.value
        ),
        memory_witness_null_quantile=0.5,
        memory_extra_edges_per_node_max=2,
        memory_added_edge_ratio_max=4.0,
        graph_mode=GraphMode.MUTUAL_KNN.value,
        k=1,
        lambda_=0.0,
        alpha_max=0.0,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.randn(12, 6), step=-1)

    output = loss_fn(proj, emb, step=5, total_steps=10, return_output=True)

    assert output.extra_logs[
        "MemoryWitness/threshold_mode/shuffled_null_quantile"
    ] == pytest.approx(1.0)
    assert 0.0 <= output.extra_logs["MemoryWitness/threshold_value"] <= 1.0
    assert output.extra_logs["MemoryWitness/null_score_p95"] >= 0.0
    assert output.extra_logs["Config/memory_witness_score_min"] == pytest.approx(-1.0)


@pytest.mark.unit
def test_invalid_null_quantile_fails():
    with pytest.raises(ValueError, match="memory_witness_null_quantile"):
        CoherentHardnessGeoDROLeJEPALoss(memory_witness_null_quantile=1.5)


@pytest.mark.unit
def test_batch_memory_queue_receives_ddp_global_graph_features(monkeypatch):
    torch.manual_seed(12)
    proj = torch.randn(3, 2, 4, requires_grad=True)
    emb = torch.randn(3, 2, 5, requires_grad=True)

    def fake_gather(tensor, *, batch_dim=0):
        detached = tensor.detach()
        moved = detached.movedim(batch_dim, 0).contiguous()
        prefix = moved.new_zeros((1, *moved.shape[1:]))
        suffix = moved.new_full((2, *moved.shape[1:]), 2.0)
        gathered = torch.cat([prefix, moved, suffix], dim=0).movedim(0, batch_dim)
        local_size = int(moved.shape[0])
        sizes = (1, local_size, 2)
        return GatheredBatch(
            tensor=gathered.contiguous(),
            local_slice=slice(1, 1 + local_size),
            sizes=sizes,
            offsets=_offsets_from_sizes(sizes),
        )

    monkeypatch.setattr(
        loss_module,
        "detached_all_gather_batch_with_metadata",
        fake_gather,
    )
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=8,
        lambda_=0.0,
        alpha_max=0.0,
        graph_mode=GraphMode.FULLY_CONNECTED.value,
        k=1,
        min_graph_nodes=1,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    output = loss_fn(proj, emb, return_output=True)

    assert output.p_global.numel() == 5
    assert loss_fn.feature_memory.valid_size == 5


@pytest.mark.unit
def test_memory_checkpoint_restore_exact_queue_state():
    torch.manual_seed(13)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=6,
        lambda_=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.randn(4, 3), step=12)
    state = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in loss_fn.state_dict().items()
    }
    restored = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=6,
        lambda_=0.0,
    )

    restored.load_state_dict(state, strict=True)

    assert restored.feature_memory.valid_size == loss_fn.feature_memory.valid_size
    assert restored.feature_memory.cursor == loss_fn.feature_memory.cursor
    assert torch.equal(
        restored.feature_memory.valid_features(),
        loss_fn.feature_memory.valid_features(),
    )
    assert torch.equal(
        restored.feature_memory.valid_insertion_steps(),
        loss_fn.feature_memory.valid_insertion_steps(),
    )
    assert restored.feature_memory.diagnostics()["Memory/checkpoint_restored"] == 1.0


@pytest.mark.unit
def test_missing_memory_checkpoint_warns_and_starts_empty():
    batch_only_state = CoherentHardnessGeoDROLeJEPALoss(lambda_=0.0).state_dict()
    memory_loss = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=6,
        lambda_=0.0,
    )

    with pytest.warns(RuntimeWarning, match="memory queue state was missing"):
        memory_loss.load_state_dict(batch_only_state, strict=True)

    assert memory_loss.feature_memory.valid_size == 0
    assert (
        memory_loss.feature_memory.diagnostics()["Memory/checkpoint_missing_fallback"]
        == 1.0
    )


@pytest.mark.unit
def test_incompatible_memory_checkpoint_metadata_fails_loudly():
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=4,
        lambda_=0.0,
    )
    loss_fn.feature_memory.enqueue(torch.randn(3, 2), step=0)
    state = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in loss_fn.state_dict().items()
    }
    incompatible = CoherentHardnessGeoDROLeJEPALoss(
        geometry_support=GeometrySupport.BATCH_MEMORY.value,
        memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
        memory_queue_capacity=5,
        lambda_=0.0,
    )

    with pytest.raises(RuntimeError, match="queue_capacity mismatch"):
        incompatible.load_state_dict(state, strict=True)


@pytest.mark.unit
def test_no_graph_kl_mode_fails_loudly_until_kl_dro_is_implemented():
    with pytest.raises(NotImplementedError, match="no_graph_kl"):
        build_graph(torch.eye(4), mode=GraphMode.NO_GRAPH_KL, k=2)


@pytest.mark.unit
def test_flow_invariants_for_uniform_utility():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    graph = build_graph(features, mode=GraphMode.FULLY_CONNECTED, k=2)
    utility = torch.zeros(4)

    weights, diagnostics = solve_graph_flow(utility, graph, inner_steps=3)

    assert torch.all(weights >= 0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.allclose(weights, torch.ones(4) / 4, atol=1e-6)
    assert diagnostics.ess_ratio == pytest.approx(1.0)
    assert not diagnostics.nan_or_inf_seen


@pytest.mark.unit
def test_reliability_gate_falls_back_for_too_small_graph():
    graph = build_graph(torch.eye(8), mode=GraphMode.FULLY_CONNECTED, k=2)
    flow_diag = FlowDiagnostics(
        clamp_activation_ratio=0.0,
        nan_or_inf_seen=False,
        min_p_before_clamp=0.0,
        max_p=0.2,
        entropy=0.0,
        ess_ratio=1.0,
        flow_num_steps=1,
    )
    p_flow = torch.linspace(1.0, 8.0, 8)
    p_flow = p_flow / p_flow.sum()

    p_train, weight_diag = reliability_gated_weights(
        p_flow,
        graph.diagnostics,
        flow_diag,
        alpha_max=0.5,
        min_graph_nodes=64,
    )

    assert torch.allclose(p_train, torch.ones(8) / 8)
    assert weight_diag.fallback
    assert weight_diag.fallback_reason == "graph_too_small"
    assert weight_diag.alpha == pytest.approx(0.0)


@pytest.mark.unit
def test_relaxed_reliability_gate_can_activate_nonuniform_weights():
    graph = build_graph(torch.eye(8), mode=GraphMode.FULLY_CONNECTED, k=2)
    p_flow = torch.arange(1, 9, dtype=torch.float32)
    p_flow = p_flow / p_flow.sum()
    flow_diag = FlowDiagnostics(
        clamp_activation_ratio=0.45,
        nan_or_inf_seen=False,
        min_p_before_clamp=-1.0,
        max_p=float(p_flow.max()),
        entropy=0.0,
        ess_ratio=0.01,
        flow_num_steps=20,
    )

    p_train, weight_diag = reliability_gated_weights(
        p_flow,
        graph.diagnostics,
        flow_diag,
        step=10,
        total_steps=100,
        alpha_max=0.05,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
        ess_min_ratio=0.001,
        clamp_activation_fail=0.5,
        max_p_factor_fail=256,
        p_cap=0.5,
    )

    assert not weight_diag.fallback
    assert weight_diag.alpha == pytest.approx(0.05)
    assert weight_diag.warmup_step == 10
    assert weight_diag.warmup_total_steps == 100
    assert not torch.allclose(p_train, torch.ones_like(p_train) / p_train.numel())


@pytest.mark.unit
def test_connected_hard_cluster_gets_more_mass_than_isolated_spike():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.05],
            [0.99, -0.05],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.05, 0.99],
            [-0.05, 0.99],
        ]
    )
    graph = build_graph(features, mode=GraphMode.MUTUAL_KNN, k=2)
    li_global = torch.tensor([4.0, 4.0, 4.0, 10.0, 1.0, 1.0, 1.0])
    li_v_global = li_global.repeat(3, 1)

    utility, utility_diag = build_utility(li_global, li_v_global, graph)
    weights, flow_diag = solve_graph_flow(
        utility, graph, inner_steps=20, beta=0.2, tau_flow=0.1
    )

    assert utility_diag.utility_max > utility_diag.utility_min
    assert not flow_diag.nan_or_inf_seen
    assert weights[:3].sum() > weights[3]


@pytest.mark.unit
def test_utility_preserves_view_major_orientation_for_square_view_sample_shape():
    graph = build_graph(torch.eye(4), mode=GraphMode.FULLY_CONNECTED, k=2)
    li_global = torch.ones(4)
    li_v_global = torch.tensor(
        [
            [0.0, 0.0, 0.0, 100.0],
            [2.0, 0.0, 0.0, 100.0],
            [4.0, 0.0, 0.0, 100.0],
            [6.0, 0.0, 0.0, 100.0],
        ]
    )

    _, utility_diag = build_utility(
        li_global,
        li_v_global,
        graph,
        li_v_sample_dim=1,
    )

    sample_view = li_v_global.T
    expected_disp = (
        (sample_view - sample_view.median(dim=1, keepdim=True).values)
        .abs()
        .median(dim=1)
        .values
    )
    assert utility_diag.view_disp_mean == pytest.approx(float(expected_disp.mean()))


@pytest.mark.unit
def test_utility_rejects_ambiguous_square_view_sample_shape():
    graph = build_graph(torch.eye(3), mode=GraphMode.FULLY_CONNECTED, k=2)

    with pytest.raises(ValueError, match="Ambiguous li_v_global orientation"):
        build_utility(torch.ones(3), torch.ones(3, 3), graph)


@pytest.mark.unit
def test_cifar10_controlled_corruption_adds_diagnostic_tags_only():
    transform = CIFAR10ControlledCorruption(
        view_name="global_1",
        coherent_labels=(0,),
        isolated_period=2,
    )
    sample = {
        "image": Image.new("RGB", (32, 32), (128, 128, 128)),
        "label": 0,
        "sample_idx": 4,
    }

    output = transform(sample)

    assert output["image"].size == (32, 32)
    assert output["geodro_view_name"] == "global_1"
    assert output["geodro_group"] == "coherent_class_corruption"
    assert output["geodro_coherent_corruption"]
    assert output["geodro_isolated_view_corruption"]
    assert output["geodro_corruption_tag"] == "coherent_plus_isolated"


@pytest.mark.unit
def test_imagenet100_controlled_corruption_defaults_to_thirty_classes_and_five_percent():
    transform = ImageNet100ControlledCorruption(view_name="global_1")
    sample = {
        "image": Image.new("RGB", (224, 224), (128, 128, 128)),
        "label": 29,
        "sample_idx": 40,
    }

    output = transform(sample)

    assert output["image"].size == (224, 224)
    assert output["geodro_view_name"] == "global_1"
    assert output["geodro_group"] == "coherent_class_corruption"
    assert output["geodro_coherent_corruption"]
    assert output["geodro_isolated_view_corruption"]
    assert output["geodro_corruption_tag"] == "coherent_plus_isolated"


@pytest.mark.unit
def test_controlled_mass_diagnostics_are_logged_without_affecting_weights():
    torch.manual_seed(0)
    proj = torch.randn(3, 4, 3, requires_grad=True)
    emb = torch.randn(3, 4, 5, requires_grad=True)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        k=2,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    output = loss_fn(
        proj,
        emb,
        coherent_mask=torch.tensor([True, True, False, False]),
        isolated_mask=torch.tensor([False, True, False, False]),
        return_output=True,
    )

    assert output.extra_logs["Controlled/coherent_fraction"] == pytest.approx(0.5)
    assert output.extra_logs["Controlled/coherent_mass"] == pytest.approx(0.5)
    assert output.extra_logs["Controlled/coherent_mass_lift"] == pytest.approx(1.0)
    assert output.extra_logs["Controlled/isolated_fraction"] == pytest.approx(0.25)
    assert output.extra_logs["Controlled/isolated_mass"] == pytest.approx(0.25)
    assert output.extra_logs["Controlled/isolated_mass_lift"] == pytest.approx(1.0)


@pytest.mark.unit
def test_controlled_mass_logs_accept_step_global_masks_without_regather(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "step-global diagnostic masks should not be gathered again"
        )

    monkeypatch.setattr(loss_module, "detached_all_gather_batch", fail_if_called)
    p_global = torch.ones(4) / 4

    logs = loss_module._controlled_mass_logs(
        p_global,
        coherent_mask=torch.tensor([True, True, False, False]),
        isolated_mask=None,
    )

    assert logs["Controlled/coherent_fraction"] == pytest.approx(0.5)
    assert logs["Controlled/coherent_mass"] == pytest.approx(0.5)
    assert logs["Controlled/coherent_mass_lift"] == pytest.approx(1.0)


@pytest.mark.unit
def test_controlled_mass_logs_support_raw_flow_prefix():
    p_global = torch.tensor([0.4, 0.3, 0.2, 0.1])

    logs = loss_module._controlled_mass_logs(
        p_global,
        coherent_mask=torch.tensor([True, False, True, False]),
        isolated_mask=None,
        prefix="RawFlowControlled",
    )

    assert logs["RawFlowControlled/coherent_fraction"] == pytest.approx(0.5)
    assert logs["RawFlowControlled/coherent_mass"] == pytest.approx(0.6)
    assert logs["RawFlowControlled/coherent_mass_lift"] == pytest.approx(1.2)


@pytest.mark.unit
def test_optimizer_step_replay_matches_microbatch_for_single_microbatch():
    torch.manual_seed(0)
    proj = torch.randn(3, 6, 4, requires_grad=True)
    emb = torch.randn(3, 6, 5, requires_grad=True)
    common = dict(
        lambda_=0.0,
        alpha_max=0.3,
        graph_mode=GraphMode.FULLY_CONNECTED.value,
        k=2,
        min_graph_nodes=4,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    microbatch_loss = CoherentHardnessGeoDROLeJEPALoss(**common)
    optimizer_step_loss = CoherentHardnessGeoDROLeJEPALoss(
        **common,
        adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
    )

    microbatch_output = microbatch_loss(
        proj,
        emb,
        step=5,
        total_steps=10,
        return_output=True,
    )
    adversary_inputs = optimizer_step_loss.compute_adversary_inputs(proj, emb)
    weights = optimizer_step_loss.solve_adversary_weights(
        adversary_inputs.graph_features,
        adversary_inputs.li_local,
        adversary_inputs.li_v,
        step=5,
        total_steps=10,
    )
    total_loss, pred_loss, _, pred_erm = optimizer_step_loss.weighted_replay_loss(
        proj,
        emb,
        weights.p_global,
    )

    assert torch.allclose(weights.p_global, microbatch_output.p_global)
    assert torch.allclose(pred_loss, microbatch_output.pred_loss)
    assert torch.allclose(total_loss, microbatch_output.total_loss)
    assert torch.allclose(pred_erm, microbatch_output.pred_erm)


@pytest.mark.unit
def test_optimizer_step_weight_slices_cover_accumulated_step_graph():
    torch.manual_seed(1)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        lambda_=0.0,
        alpha_max=0.3,
        adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
        graph_mode=GraphMode.FULLY_CONNECTED.value,
        k=2,
        min_graph_nodes=4,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )
    proj_a = torch.randn(3, 3, 4, requires_grad=True)
    emb_a = torch.randn(3, 3, 5, requires_grad=True)
    proj_b = torch.randn(3, 2, 4, requires_grad=True)
    emb_b = torch.randn(3, 2, 5, requires_grad=True)

    inputs_a = loss_fn.compute_adversary_inputs(proj_a, emb_a)
    inputs_b = loss_fn.compute_adversary_inputs(proj_b, emb_b)
    weights = loss_fn.solve_adversary_weights(
        torch.cat([inputs_a.graph_features, inputs_b.graph_features], dim=0),
        torch.cat([inputs_a.li_local, inputs_b.li_local], dim=0),
        torch.cat([inputs_a.li_v, inputs_b.li_v], dim=1),
        step=5,
        total_steps=10,
    )
    p_a = weights.p_global[: inputs_a.li_local.numel()]
    p_b = weights.p_global[inputs_a.li_local.numel() :]

    total_a, pred_a, _, _ = loss_fn.weighted_replay_loss(proj_a, emb_a, p_a)
    total_b, pred_b, _, _ = loss_fn.weighted_replay_loss(proj_b, emb_b, p_b)
    expected = (p_a * inputs_a.li_local).sum() + (p_b * inputs_b.li_local).sum()

    assert p_a.shape == inputs_a.li_local.shape
    assert p_b.shape == inputs_b.li_local.shape
    assert torch.isclose(weights.p_global.sum(), torch.tensor(1.0))
    assert torch.allclose(pred_a + pred_b, expected)
    assert torch.allclose(total_a + total_b, expected)


@pytest.mark.unit
def test_optimizer_step_warmup_total_is_accumulation_corrected():
    class DummyTrainer:
        estimated_stepping_batches = 400

    class DummyModule:
        trainer = DummyTrainer()

    assert _accumulation_corrected_total_steps(DummyModule(), accum_steps=4) == 100


@pytest.mark.unit
def test_local_batch_slice_is_deterministic():
    assert local_batch_slice(8, rank=0) == slice(0, 8)
    assert local_batch_slice(8, rank=2) == slice(16, 24)


@pytest.mark.unit
def test_gathered_batch_metadata_non_distributed_detaches():
    x = torch.arange(6.0, requires_grad=True).reshape(2, 3)

    gathered = detached_all_gather_batch_with_metadata(x, batch_dim=0)

    assert torch.equal(gathered.tensor, x.detach())
    assert not gathered.tensor.requires_grad
    assert gathered.local_slice == slice(0, 2)
    assert gathered.sizes == (2,)
    assert gathered.offsets == (0,)


@pytest.mark.unit
def test_gathered_batch_offsets_support_uneven_and_empty_ranks():
    assert _offsets_from_sizes((3, 0, 5, 2)) == (0, 3, 3, 8)


@pytest.mark.unit
def test_size_aware_gather_trims_uneven_fake_ddp(monkeypatch):
    rank_chunks = (
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([[10.0], [11.0]]),
        torch.tensor([[20.0], [21.0], [22.0], [23.0]]),
    )
    sizes = tuple(chunk.shape[0] for chunk in rank_chunks)
    max_size = max(sizes)

    def fake_all_gather(outputs, input_tensor):
        if input_tensor.dtype == torch.long:
            for output, size in zip(outputs, sizes, strict=True):
                output.copy_(torch.tensor([size], device=output.device))
            return
        for output, chunk in zip(outputs, rank_chunks, strict=True):
            pad_shape = (max_size - chunk.shape[0], *chunk.shape[1:])
            padded = torch.cat([chunk, chunk.new_zeros(pad_shape)], dim=0)
            output.copy_(padded)

    monkeypatch.setattr(distributed_module, "is_distributed", lambda: True)
    monkeypatch.setattr(distributed_module, "get_world_size", lambda: 3)
    monkeypatch.setattr(distributed_module, "get_rank", lambda: 1)
    monkeypatch.setattr(distributed_module.dist, "all_gather", fake_all_gather)

    gathered = detached_all_gather_batch_with_metadata(rank_chunks[1], batch_dim=0)

    assert torch.equal(gathered.tensor, torch.cat(rank_chunks, dim=0))
    assert gathered.local_slice == slice(3, 5)
    assert gathered.sizes == sizes
    assert gathered.offsets == (0, 3, 5)


@pytest.mark.unit
def test_microbatch_p_local_uses_gather_metadata_slice(monkeypatch):
    torch.manual_seed(7)
    proj = torch.randn(3, 2, 4, requires_grad=True)
    emb = torch.randn(3, 2, 5, requires_grad=True)
    prefix_size = 3
    suffix_size = 2
    world_size = 3

    def fake_gather(tensor, *, batch_dim=0):
        detached = tensor.detach()
        moved = detached.movedim(batch_dim, 0).contiguous()
        prefix = moved.new_zeros((prefix_size, *moved.shape[1:]))
        suffix = moved.new_full((suffix_size, *moved.shape[1:]), 2.0)
        gathered = torch.cat([prefix, moved, suffix], dim=0).movedim(0, batch_dim)
        local_size = int(moved.shape[0])
        sizes = (prefix_size, local_size, suffix_size)
        return GatheredBatch(
            tensor=gathered.contiguous(),
            local_slice=slice(prefix_size, prefix_size + local_size),
            sizes=sizes,
            offsets=_offsets_from_sizes(sizes),
        )

    monkeypatch.setattr(
        loss_module,
        "detached_all_gather_batch_with_metadata",
        fake_gather,
    )
    monkeypatch.setattr(loss_module, "get_world_size", lambda: world_size)
    loss_fn = CoherentHardnessGeoDROLeJEPALoss(
        lambda_=0.0,
        alpha_max=0.0,
        graph_mode=GraphMode.FULLY_CONNECTED.value,
        k=2,
        min_graph_nodes=1,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    )

    output = loss_fn(proj, emb, return_output=True)
    expected_slice = slice(prefix_size, prefix_size + output.li_local.numel())
    expected_pred = world_size * (output.p_local * output.li_local).sum()

    assert torch.allclose(output.p_local, output.p_global[expected_slice])
    assert torch.allclose(output.pred_loss, expected_pred)


@pytest.mark.unit
def test_ddp_weighted_prediction_gradient_matches_global_objective():
    q_global = torch.tensor([0.10, 0.20, 0.05, 0.15, 0.50])
    coeff = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sizes = (2, 1, 2)
    world_size = len(sizes)
    lambda_ = 0.25

    theta_global = torch.tensor(1.7, requires_grad=True)
    global_pred = (q_global * coeff * theta_global.square()).sum()
    global_regularizer = 0.75 * theta_global.square()
    global_total = (1.0 - lambda_) * global_pred + lambda_ * global_regularizer
    global_total.backward()

    local_grads = []
    cursor = 0
    for size in sizes:
        theta_local = torch.tensor(1.7, requires_grad=True)
        local_slice = slice(cursor, cursor + size)
        local_pred = (
            world_size
            * (q_global[local_slice] * coeff[local_slice] * theta_local.square()).sum()
        )
        local_regularizer = 0.75 * theta_local.square()
        local_total = (1.0 - lambda_) * local_pred + lambda_ * local_regularizer
        local_total.backward()
        local_grads.append(theta_local.grad.detach())
        cursor += size

    ddp_averaged_grad = torch.stack(local_grads).mean()

    assert torch.allclose(ddp_averaged_grad, theta_global.grad)


@pytest.mark.unit
def test_optimizer_step_weight_slicing_uses_stored_gather_metadata():
    class DummyLoss:
        def solve_adversary_weights(self, graph_features, *args, **kwargs):
            return SimpleNamespace(
                p_global=torch.arange(graph_features.shape[0], dtype=torch.float32)
            )

    class DummyModule:
        geodro_lejepa_loss = DummyLoss()
        global_step = 0
        trainer = None

    first = _CollectedMicrobatch(
        batch={},
        rng_state={},
        output={},
        local_batch_size=2,
        local_slice=slice(3, 5),
        graph_features_global=torch.zeros(7, 1),
        li_global=torch.arange(7, dtype=torch.float32),
        li_v_global=torch.zeros(2, 7),
        coherent_mask_global=None,
        isolated_mask_global=None,
    )
    second = _CollectedMicrobatch(
        batch={},
        rng_state={},
        output={},
        local_batch_size=3,
        local_slice=slice(1, 4),
        graph_features_global=torch.zeros(4, 1),
        li_global=torch.arange(4, dtype=torch.float32),
        li_v_global=torch.zeros(2, 4),
        coherent_mask_global=None,
        isolated_mask_global=None,
    )

    _, local_weights, _ = _solve_step_weights(
        DummyModule(),
        [first, second],
        accum_steps=2,
    )

    assert torch.equal(local_weights[0], torch.tensor([3.0, 4.0]))
    assert torch.equal(local_weights[1], torch.tensor([8.0, 9.0, 10.0]))


@pytest.mark.unit
def test_geodro_forward_training_smoke_and_single_view_inference():
    class DummyBackbone(nn.Module):
        def __init__(self, out_dim=4):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(3, out_dim)

        def forward(self, x):
            return self.proj(self.pool(x).flatten(1))

    class DummyModule:
        def __init__(self):
            self.backbone = DummyBackbone()
            self.projector = nn.Linear(4, 3)
            self.geodro_lejepa_loss = CoherentHardnessGeoDROLeJEPALoss(
                lambda_=0.0,
                alpha_max=0.0,
                k=2,
                warmup_fraction=0.0,
                ramp_fraction=0.0,
            )
            self.training = True
            self.global_step = 0
            self.logged = {}

        def log(self, name, value, **kwargs):
            self.logged[name] = value

    module = DummyModule()
    views = [
        {"image": torch.randn(4, 3, 16, 16), "label": torch.arange(4)},
        {"image": torch.randn(4, 3, 12, 12), "label": torch.arange(4)},
    ]

    out = geodro_lejepa_forward(module, views, "fit")

    assert out["embedding"].shape == (8, 4)
    assert out["loss"].ndim == 0
    assert "train/geodro/Weight_alpha" in module.logged

    module.training = False
    val_out = geodro_lejepa_forward(
        module,
        {"image": torch.randn(4, 3, 16, 16), "label": torch.arange(4)},
        "validate",
    )
    assert val_out["embedding"].shape == (4, 4)
    assert torch.equal(val_out["label"], torch.arange(4))


@pytest.mark.unit
def test_geodro_forward_optimizer_step_requires_context_and_replays_fixed_weights():
    class DummyBackbone(nn.Module):
        def __init__(self, out_dim=4):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(3, out_dim)

        def forward(self, x):
            return self.proj(self.pool(x).flatten(1))

    class DummyModule:
        def __init__(self):
            self.backbone = DummyBackbone()
            self.projector = nn.Linear(4, 3)
            self.geodro_lejepa_loss = CoherentHardnessGeoDROLeJEPALoss(
                lambda_=0.0,
                alpha_max=0.0,
                adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
                k=2,
                warmup_fraction=0.0,
                ramp_fraction=0.0,
            )
            self.training = True
            self.global_step = 0
            self.logged = {}

        def log(self, name, value, **kwargs):
            self.logged[name] = value

    module = DummyModule()
    views = [
        {"image": torch.randn(4, 3, 16, 16), "label": torch.arange(4)},
        {"image": torch.randn(4, 3, 12, 12), "label": torch.arange(4)},
    ]

    with pytest.raises(RuntimeError, match="optimizer-step training loop context"):
        geodro_lejepa_forward(module, views, "fit")

    module._geodro_optimizer_step_context = {
        "p_local": torch.ones(4) / 4,
        "sigreg_scale": 0.5,
    }
    out = geodro_lejepa_forward(module, views, "fit")

    assert out["embedding"].shape == (8, 4)
    assert out["loss"].ndim == 0
    assert out["geodro_main_loss"].ndim == 0
    assert out["geodro_pred_loss"].ndim == 0
    assert out["geodro_sigreg_loss"].ndim == 0
    assert out["geodro_pred_erm_loss"].ndim == 0
    assert module.logged == {}


@pytest.mark.unit
def test_optimizer_step_training_step_buffers_then_replays_microbatches():
    class DummyTrainer:
        accumulate_grad_batches = 2
        estimated_stepping_batches = 4

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(3, 4)

        def forward(self, x):
            return self.proj(self.pool(x).flatten(1))

    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DummyBackbone()
            self.projector = nn.Linear(4, 3)
            self.geodro_lejepa_loss = CoherentHardnessGeoDROLeJEPALoss(
                lambda_=0.0,
                alpha_max=0.0,
                adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
                k=1,
                min_graph_nodes=2,
                warmup_fraction=0.0,
                ramp_fraction=0.0,
            )
            self.trainer = DummyTrainer()
            self.global_step = 0
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            self.backward_calls = 0
            self.step_calls = 0
            self.logged = {}

        def forward(self, batch, stage):
            return geodro_lejepa_forward(self, batch, stage)

        def _manual_optimization_handles(self):
            return [self.optimizer], False

        def _step_manual_optimizers(self, optimizers, batch_idx, accum_steps):
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            self.step_calls += 1

        def manual_backward(self, loss):
            self.backward_calls += 1
            loss.backward()

        def after_manual_backward(self):
            pass

        def log(self, name, value, **kwargs):
            self.logged[name] = value

    def batch(seed: int) -> dict[str, dict[str, torch.Tensor]]:
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(2)
        return {
            "global_0": {
                "image": torch.randn(2, 3, 8, 8, generator=generator),
                "label": labels,
            },
            "global_1": {
                "image": torch.randn(2, 3, 8, 8, generator=generator),
                "label": labels,
            },
        }

    module = DummyModule()
    before = module.projector.weight.detach().clone()

    first = optimizer_step_training_step(module, batch(0), 0)
    assert first["geodro_deferred_optimizer_step"]
    assert module.backward_calls == 0
    assert module.step_calls == 0
    assert len(module._geodro_optimizer_step_buffer) == 1

    second = optimizer_step_training_step(module, batch(1), 1)

    assert second["loss"].ndim == 0
    assert module.backward_calls == 2
    assert module.step_calls == 1
    assert module._geodro_optimizer_step_buffer == []
    assert not hasattr(module, "_geodro_optimizer_step_context")
    assert "train/geodro/Weight_alpha" in module.logged
    assert not torch.allclose(module.projector.weight.detach(), before)


@pytest.mark.unit
@pytest.mark.parametrize(
    "loss_cls",
    [
        CoherentHardnessGeoDROLeJEPALoss,
        GraphTransportGeoDROJEPALoss,
    ],
)
def test_optimizer_step_delayed_memory_updates_after_accumulated_replay(
    monkeypatch,
    loss_cls,
):
    class DummyTrainer:
        accumulate_grad_batches = 2
        estimated_stepping_batches = 4

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(3, 4)

        def forward(self, x):
            return self.proj(self.pool(x).flatten(1))

    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DummyBackbone()
            self.projector = nn.Linear(4, 3)
            self.geodro_lejepa_loss = loss_cls(
                lambda_=0.0,
                alpha_max=0.0,
                adversary_scope=AdversaryScope.OPTIMIZER_STEP.value,
                geometry_support=GeometrySupport.BATCH_MEMORY.value,
                memory_usage_mode=MemoryUsageMode.MEMORY_WITNESSED.value,
                memory_update_scope=MemoryUpdateScope.OPTIMIZER_STEP_DELAYED.value,
                memory_queue_capacity=16,
                memory_top_m=1,
                memory_k_sigma=1,
                memory_min_fill_ratio=0.0,
                memory_extra_edges_per_node_max=0,
                memory_added_edge_ratio_max=0.0,
                graph_mode=GraphMode.FULLY_CONNECTED.value,
                k=1,
                min_graph_nodes=2,
                warmup_fraction=0.0,
                ramp_fraction=0.0,
            )
            self.trainer = DummyTrainer()
            self.global_step = 11
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            self.logged = {}

        def forward(self, batch, stage):
            return geodro_lejepa_forward(self, batch, stage)

        def _manual_optimization_handles(self):
            return [self.optimizer], False

        def _step_manual_optimizers(self, optimizers, batch_idx, accum_steps):
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        def manual_backward(self, loss):
            loss.backward()

        def after_manual_backward(self):
            pass

        def log(self, name, value, **kwargs):
            self.logged[name] = value

    def batch(seed: int) -> dict[str, dict[str, torch.Tensor]]:
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(2)
        return {
            "global_0": {
                "image": torch.randn(2, 3, 8, 8, generator=generator),
                "label": labels,
            },
            "global_1": {
                "image": torch.randn(2, 3, 8, 8, generator=generator),
                "label": labels,
            },
        }

    module = DummyModule()
    memory = module.geodro_lejepa_loss.feature_memory
    preexisting = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    memory.enqueue(preexisting, step=-3)
    snapshots = []
    original_retrieve = memory.retrieve_witnesses

    def capture_retrieve(current_features, *args, **kwargs):
        snapshots.append(
            {
                "valid_size": memory.valid_size,
                "features": memory.valid_features().detach().clone(),
                "current_count": int(current_features.shape[0]),
            }
        )
        return original_retrieve(current_features, *args, **kwargs)

    monkeypatch.setattr(memory, "retrieve_witnesses", capture_retrieve)

    first = optimizer_step_training_step(module, batch(0), 0)

    assert first["geodro_deferred_optimizer_step"]
    assert snapshots == []
    assert memory.valid_size == 3

    second = optimizer_step_training_step(module, batch(1), 1)

    assert second["loss"].ndim == 0
    assert len(snapshots) == 1
    assert snapshots[0]["valid_size"] == 3
    assert snapshots[0]["current_count"] == 4
    assert torch.equal(snapshots[0]["features"], preexisting)
    assert memory.valid_size == 7
    assert torch.equal(
        memory.valid_insertion_steps(),
        torch.tensor([-3, -3, -3, 11, 11, 11, 11]),
    )
    assert "train/geodro/Memory_size" in module.logged
