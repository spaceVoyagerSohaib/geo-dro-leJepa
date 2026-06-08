"""Detached feature memory for optional GeoDRO geometry support."""

from __future__ import annotations

import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import (
    GraphDistanceMetric,
    GraphSpace,
    MemoryUpdateScope,
    MemoryWitnessBatch,
)


_GRAPH_SPACE_CODES = {
    GraphSpace.PRE_PROJECTOR_GLOBAL_CENTER: 1,
    GraphSpace.PROJECTOR_GLOBAL_CENTER: 2,
    GraphSpace.CONSENSUS_PREPROJ_PROJECTOR: 3,
    GraphSpace.RANDOM_FEATURES: 4,
}
_GRAPH_DISTANCE_METRIC_CODES = {
    GraphDistanceMetric.COSINE: 1,
}
_DTYPE_CODES = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
}
_TIE_BREAK_EPS = torch.finfo(torch.float64).eps * 16.0


class GeoDROFeatureMemoryQueue(nn.Module):
    """Replicated FIFO queue storing detached graph features only."""

    def __init__(
        self,
        *,
        capacity: int,
        graph_space: GraphSpace | str,
        graph_distance_metric: GraphDistanceMetric | str,
        graph_feature_normalized: bool,
        update_scope: MemoryUpdateScope | str = MemoryUpdateScope.MICROBATCH,
    ) -> None:
        super().__init__()
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("GeoDRO memory_queue_capacity must be positive.")

        self.capacity = capacity
        self.graph_space = GraphSpace(graph_space)
        self.graph_distance_metric = GraphDistanceMetric(graph_distance_metric)
        self.graph_feature_normalized = bool(graph_feature_normalized)
        self.update_scope = MemoryUpdateScope(update_scope)

        self.register_buffer("queue_features", torch.empty((capacity, 0)))
        self.register_buffer(
            "queue_insertion_steps",
            torch.full((capacity,), -1, dtype=torch.long),
        )
        self.register_buffer("queue_valid_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("queue_cursor", torch.tensor(0, dtype=torch.long))
        self.register_buffer("queue_feature_dim", torch.tensor(-1, dtype=torch.long))
        self.register_buffer(
            "queue_capacity_meta",
            torch.tensor(capacity, dtype=torch.long),
        )
        self.register_buffer(
            "graph_space_code",
            torch.tensor(_GRAPH_SPACE_CODES[self.graph_space], dtype=torch.long),
        )
        self.register_buffer(
            "graph_distance_metric_code",
            torch.tensor(
                _GRAPH_DISTANCE_METRIC_CODES[self.graph_distance_metric],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "graph_feature_normalized_meta",
            torch.tensor(int(self.graph_feature_normalized), dtype=torch.long),
        )
        self.register_buffer("queue_dtype_code", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("queue_update_count", torch.tensor(0, dtype=torch.long))

        self._checkpoint_restored = False
        self._checkpoint_missing_fallback = False

    @property
    def valid_size(self) -> int:
        return int(self.queue_valid_size.item())

    @property
    def cursor(self) -> int:
        return int(self.queue_cursor.item())

    @property
    def feature_dim(self) -> int:
        return int(self.queue_feature_dim.item())

    @property
    def fill_ratio(self) -> float:
        return self.valid_size / max(float(self.capacity), 1.0)

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, *, step: int | None = None) -> None:
        """Append a DDP-global graph feature batch after the current loss is built."""
        if features.ndim != 2:
            raise ValueError(
                f"Expected memory features with shape [B, D], got {tuple(features.shape)}."
            )
        if features.numel() == 0:
            self.queue_update_count.add_(1)
            return

        detached = features.detach().contiguous()
        self._ensure_storage(detached)
        detached = detached.to(
            device=self.queue_features.device,
            dtype=self.queue_features.dtype,
        )

        if detached.shape[0] >= self.capacity:
            detached = detached[-self.capacity :]

        write_count = int(detached.shape[0])
        cursor = self.cursor
        step_value = int(step) if step is not None else int(self.queue_update_count.item())
        insertion_steps = torch.full(
            (write_count,),
            step_value,
            device=self.queue_insertion_steps.device,
            dtype=self.queue_insertion_steps.dtype,
        )

        first = min(write_count, self.capacity - cursor)
        self.queue_features[cursor : cursor + first].copy_(detached[:first])
        self.queue_insertion_steps[cursor : cursor + first].copy_(
            insertion_steps[:first]
        )
        remaining = write_count - first
        if remaining > 0:
            self.queue_features[:remaining].copy_(detached[first:])
            self.queue_insertion_steps[:remaining].copy_(insertion_steps[first:])

        self.queue_cursor.fill_((cursor + write_count) % self.capacity)
        self.queue_valid_size.fill_(min(self.capacity, self.valid_size + write_count))
        self.queue_update_count.add_(1)

    def valid_features(self) -> torch.Tensor:
        """Return valid queued features in FIFO order."""
        indices = self._valid_indices()
        return self.queue_features.index_select(0, indices)

    def valid_insertion_steps(self) -> torch.Tensor:
        """Return valid insertion steps in FIFO order."""
        indices = self._valid_indices()
        return self.queue_insertion_steps.index_select(0, indices)

    @torch.no_grad()
    def retrieve_witnesses(
        self,
        current_features: torch.Tensor,
        *,
        top_m: int = 64,
        k_sigma: int | None = None,
        min_fill_ratio: float = 0.25,
        eps: float = 1e-12,
        chunk_size: int | None = None,
    ) -> MemoryWitnessBatch:
        """Retrieve detached sparse memory witnesses for current graph features."""
        start = time.perf_counter()
        top_m = int(top_m)
        k_sigma = top_m if k_sigma is None else int(k_sigma)
        _validate_retrieval_config(
            top_m=top_m,
            k_sigma=k_sigma,
            min_fill_ratio=min_fill_ratio,
            chunk_size=chunk_size,
        )
        if current_features.ndim != 2:
            raise ValueError(
                "Expected current graph features with shape [B, D], got "
                f"{tuple(current_features.shape)}."
            )
        current = current_features.detach()
        required_neighbors = max(top_m, k_sigma)
        current_count = int(current.shape[0])
        if (
            current_count == 0
            or self.valid_size < required_neighbors
            or self.feature_dim <= 0
        ):
            return _empty_witness_batch(
                current,
                top_m=top_m,
                k_sigma=k_sigma,
                retrieval_time_ms=_elapsed_ms(start),
            )
        if int(current.shape[1]) != self.feature_dim:
            raise ValueError(
                "GeoDRO memory retrieval feature_dim mismatch: "
                f"queue has {self.feature_dim}, current features have "
                f"{int(current.shape[1])}."
            )

        return retrieve_witnesses_from_memory_features(
            current,
            self.valid_features().detach(),
            self.valid_insertion_steps().detach(),
            metric=self.graph_distance_metric,
            top_m=top_m,
            k_sigma=k_sigma,
            min_fill_ratio=min_fill_ratio,
            fill_ratio=self.fill_ratio,
            eps=eps,
            chunk_size=chunk_size,
            retrieval_start=start,
        )

    def mark_checkpoint_missing_fallback(self) -> None:
        self._checkpoint_missing_fallback = True
        self._checkpoint_restored = False

    def diagnostics(self, *, step: int | None = None) -> dict[str, float]:
        valid_steps = self.valid_insertion_steps()
        if valid_steps.numel() == 0:
            age_min = age_median = age_max = 0.0
            effective_horizon = 0.0
        else:
            if step is None:
                current_step = int(self.queue_update_count.item())
            else:
                current_step = int(step)
            ages = (current_step - valid_steps).clamp_min(0).float()
            age_min = float(ages.min().detach().cpu())
            age_median = float(ages.median().detach().cpu())
            age_max = float(ages.max().detach().cpu())
            effective_horizon = float(
                (
                    valid_steps.max() - valid_steps.min() + 1
                )
                .clamp_min(0)
                .detach()
                .cpu()
            )

        memory_norm_mean, memory_norm_std = self._memory_norm_stats()
        queue_memory_mb = (
            self.capacity
            * max(self.feature_dim, 0)
            * max(self.queue_features.element_size(), 0)
            / 1e6
        )
        metadata_memory_mb = (
            self.queue_insertion_steps.numel()
            * self.queue_insertion_steps.element_size()
            / 1e6
        )
        peak_allocated_mb = 0.0
        if self.queue_features.device.type == "cuda" and torch.cuda.is_available():
            peak_allocated_mb = float(
                torch.cuda.max_memory_allocated(self.queue_features.device) / 1e6
            )
        return {
            "Memory/enabled": 1.0,
            "Memory/size": float(self.valid_size),
            "Memory/fill_ratio": float(self.fill_ratio),
            "Memory/effective_horizon_steps": effective_horizon,
            "Memory/age_min": age_min,
            "Memory/age_median": age_median,
            "Memory/age_max": age_max,
            "Memory/checkpoint_restored": float(self._checkpoint_restored),
            "Memory/checkpoint_missing_fallback": float(
                self._checkpoint_missing_fallback
            ),
            f"Memory/update_scope/{self.update_scope.value}": 1.0,
            f"Memory/update_clock/{self.update_scope.value}": 1.0,
            "Memory/updates_per_optimizer_step": 1.0,
            "Memory/update_count": float(int(self.queue_update_count.item())),
            "Memory/queue_memory_mb": float(queue_memory_mb),
            "Memory/metadata_memory_mb": float(metadata_memory_mb),
            "Memory/retrieval_time_ms": 0.0,
            "Memory/graph_build_time_ms": 0.0,
            "Memory/peak_allocated_mb_optional": peak_allocated_mb,
            "GraphFeature/norm_mean_memory": memory_norm_mean,
            "GraphFeature/norm_std_memory": memory_norm_std,
        }

    def _valid_indices(self) -> torch.Tensor:
        valid_size = self.valid_size
        if valid_size == 0:
            return torch.empty((0,), device=self.queue_features.device, dtype=torch.long)
        if valid_size < self.capacity:
            return torch.arange(valid_size, device=self.queue_features.device)
        return torch.cat(
            [
                torch.arange(self.cursor, self.capacity, device=self.queue_features.device),
                torch.arange(0, self.cursor, device=self.queue_features.device),
            ]
        )

    def _ensure_storage(self, features: torch.Tensor) -> None:
        feature_dim = int(features.shape[1])
        dtype_code = _dtype_code(features.dtype)
        if self.feature_dim < 0:
            self.queue_features = torch.empty(
                (self.capacity, feature_dim),
                device=features.device,
                dtype=features.dtype,
            )
            self.queue_feature_dim.fill_(feature_dim)
            self.queue_dtype_code.fill_(dtype_code)
            return
        if self.feature_dim != feature_dim:
            raise ValueError(
                "GeoDRO memory feature_dim mismatch: "
                f"queue has {self.feature_dim}, incoming features have {feature_dim}."
            )
        if int(self.queue_dtype_code.item()) != dtype_code:
            raise ValueError(
                "GeoDRO memory dtype mismatch: "
                f"queue has code {int(self.queue_dtype_code.item())}, "
                f"incoming features have code {dtype_code}."
            )

    def _memory_norm_stats(self) -> tuple[float, float]:
        if self.valid_size == 0 or self.feature_dim <= 0:
            return 0.0, 0.0
        norms = self.valid_features().detach().float().norm(dim=-1)
        return (
            float(norms.mean().detach().cpu()),
            float(norms.std(unbiased=False).detach().cpu()),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        capacity_key = prefix + "queue_capacity_meta"
        graph_space_key = prefix + "graph_space_code"
        metric_key = prefix + "graph_distance_metric_code"
        normalized_key = prefix + "graph_feature_normalized_meta"
        dtype_key = prefix + "queue_dtype_code"
        features_key = prefix + "queue_features"

        if capacity_key in state_dict:
            incoming_capacity = int(state_dict[capacity_key].item())
            if incoming_capacity != self.capacity:
                error_msgs.append(
                    "GeoDRO memory queue_capacity mismatch: "
                    f"checkpoint has {incoming_capacity}, current config has "
                    f"{self.capacity}."
                )
        if graph_space_key in state_dict:
            incoming_graph_space = int(state_dict[graph_space_key].item())
            expected_graph_space = _GRAPH_SPACE_CODES[self.graph_space]
            if incoming_graph_space != expected_graph_space:
                error_msgs.append("GeoDRO memory graph_space mismatch.")
        if metric_key in state_dict:
            incoming_metric = int(state_dict[metric_key].item())
            expected_metric = _GRAPH_DISTANCE_METRIC_CODES[self.graph_distance_metric]
            if incoming_metric != expected_metric:
                error_msgs.append("GeoDRO memory graph_distance_metric mismatch.")
        if normalized_key in state_dict:
            incoming_normalized = bool(int(state_dict[normalized_key].item()))
            if incoming_normalized != self.graph_feature_normalized:
                error_msgs.append("GeoDRO memory normalization_flag mismatch.")
        if dtype_key in state_dict and int(self.queue_dtype_code.item()) >= 0:
            incoming_dtype = int(state_dict[dtype_key].item())
            if incoming_dtype != int(self.queue_dtype_code.item()):
                error_msgs.append("GeoDRO memory dtype mismatch.")

        if features_key in state_dict:
            incoming_features = state_dict[features_key]
            if incoming_features.ndim != 2:
                error_msgs.append(
                    "GeoDRO memory queue_features checkpoint tensor must be rank 2."
                )
            elif int(incoming_features.shape[0]) == self.capacity:
                self.queue_features = torch.empty_like(incoming_features)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if features_key in state_dict and not error_msgs:
            self._checkpoint_restored = True
            self._checkpoint_missing_fallback = False


def _validate_retrieval_config(
    *,
    top_m: int,
    k_sigma: int,
    min_fill_ratio: float,
    chunk_size: int | None,
) -> None:
    if top_m <= 0:
        raise ValueError(f"memory_top_m must be positive, got {top_m}.")
    if k_sigma <= 0:
        raise ValueError(f"memory_k_sigma must be positive, got {k_sigma}.")
    if not 0.0 <= float(min_fill_ratio) <= 1.0:
        raise ValueError(
            "memory_min_fill_ratio must be in [0, 1], got "
            f"{float(min_fill_ratio)}."
        )
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError(
            "memory_retrieval_chunk_size must be positive when set, got "
            f"{chunk_size}."
        )


def _empty_witness_batch(
    current_features: torch.Tensor,
    *,
    top_m: int,
    k_sigma: int,
    retrieval_time_ms: float,
) -> MemoryWitnessBatch:
    current_count = int(current_features.shape[0])
    device = current_features.device
    return MemoryWitnessBatch(
        indices=torch.empty((current_count, 0), device=device, dtype=torch.long),
        distances=torch.empty((current_count, 0), device=device, dtype=torch.float32),
        probabilities=torch.empty(
            (current_count, 0), device=device, dtype=torch.float32
        ),
        sigma=torch.zeros((current_count,), device=device, dtype=torch.float32),
        insertion_steps=torch.empty(
            (current_count, 0), device=device, dtype=torch.long
        ),
        valid_memory_for_witnessing=False,
        retrieval_time_ms=float(retrieval_time_ms),
        top_m=top_m,
        k_sigma=k_sigma,
    )


@torch.no_grad()
def retrieve_witnesses_from_memory_features(
    current_features: torch.Tensor,
    memory_features: torch.Tensor,
    memory_insertion_steps: torch.Tensor,
    *,
    metric: GraphDistanceMetric,
    top_m: int,
    k_sigma: int,
    min_fill_ratio: float,
    fill_ratio: float,
    eps: float,
    chunk_size: int | None,
    retrieval_start: float | None = None,
) -> MemoryWitnessBatch:
    """Retrieve sparse witnesses from a detached memory feature tensor."""
    start = time.perf_counter() if retrieval_start is None else retrieval_start
    current = current_features.detach()
    memory = memory_features.detach()
    top_m = int(top_m)
    k_sigma = int(k_sigma)
    _validate_retrieval_config(
        top_m=top_m,
        k_sigma=k_sigma,
        min_fill_ratio=min_fill_ratio,
        chunk_size=chunk_size,
    )
    required_neighbors = max(top_m, k_sigma)
    if (
        current.ndim != 2
        or memory.ndim != 2
        or current.shape[0] == 0
        or memory.shape[0] < required_neighbors
        or memory.shape[1] != current.shape[1]
    ):
        return _empty_witness_batch(
            current,
            top_m=top_m,
            k_sigma=k_sigma,
            retrieval_time_ms=_elapsed_ms(start),
        )

    current = current.to(device=memory.device).float()
    memory = memory.float()
    all_distances, all_indices = _retrieve_topk_exact(
        current,
        memory,
        metric=metric,
        k=required_neighbors,
        chunk_size=chunk_size,
    )
    top_distances = all_distances[:, :top_m].contiguous()
    top_indices = all_indices[:, :top_m].contiguous()
    sigma = all_distances[:, k_sigma - 1].clamp_min(eps).contiguous()
    logits = -(top_distances.square()) / (sigma.square().unsqueeze(1) + eps)
    probabilities = torch.softmax(logits, dim=1).contiguous()
    memory_insertion_steps = memory_insertion_steps.to(
        device=top_indices.device,
        dtype=torch.long,
    )
    witness_steps = memory_insertion_steps.index_select(
        0, top_indices.reshape(-1)
    ).reshape_as(top_indices)
    valid_for_witnessing = bool(float(fill_ratio) >= float(min_fill_ratio))
    return MemoryWitnessBatch(
        indices=top_indices.detach(),
        distances=top_distances.detach(),
        probabilities=probabilities.detach(),
        sigma=sigma.detach(),
        insertion_steps=witness_steps.detach(),
        valid_memory_for_witnessing=valid_for_witnessing,
        retrieval_time_ms=_elapsed_ms(start),
        top_m=top_m,
        k_sigma=k_sigma,
    )


def _retrieve_topk_exact(
    current: torch.Tensor,
    memory: torch.Tensor,
    *,
    metric: GraphDistanceMetric,
    k: int,
    chunk_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory_size = int(memory.shape[0])
    if chunk_size is None or int(chunk_size) >= memory_size:
        distances = _current_memory_distances(current, memory, metric=metric)
        indices = torch.arange(memory_size, device=memory.device).expand_as(distances)
        return _select_smallest_by_distance(distances, indices, k=k)

    best_distances = torch.empty(
        (current.shape[0], 0), device=current.device, dtype=torch.float32
    )
    best_indices = torch.empty(
        (current.shape[0], 0), device=current.device, dtype=torch.long
    )
    chunk_size = int(chunk_size)
    for start in range(0, memory_size, chunk_size):
        stop = min(start + chunk_size, memory_size)
        chunk = memory[start:stop]
        chunk_distances = _current_memory_distances(current, chunk, metric=metric)
        chunk_indices = torch.arange(
            start, stop, device=current.device, dtype=torch.long
        ).expand_as(chunk_distances)
        combined_distances = torch.cat([best_distances, chunk_distances], dim=1)
        combined_indices = torch.cat([best_indices, chunk_indices], dim=1)
        keep = min(k, int(combined_distances.shape[1]))
        best_distances, best_indices = _select_smallest_by_distance(
            combined_distances,
            combined_indices,
            k=keep,
        )
    return best_distances, best_indices


def _current_memory_distances(
    current: torch.Tensor,
    memory: torch.Tensor,
    *,
    metric: GraphDistanceMetric,
) -> torch.Tensor:
    if metric == GraphDistanceMetric.COSINE:
        current_norm = F.normalize(current.float(), dim=-1)
        memory_norm = F.normalize(memory.float(), dim=-1)
        return (1.0 - current_norm @ memory_norm.T).clamp_min(0.0)
    raise ValueError(f"Unsupported graph distance metric: {metric.value}")


def _select_smallest_by_distance(
    distances: torch.Tensor,
    indices: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tie_break = indices.to(torch.float64) * _TIE_BREAK_EPS
    order = torch.argsort(distances.to(torch.float64) + tie_break, dim=1)[:, :k]
    return distances.gather(1, order).contiguous(), indices.gather(1, order).contiguous()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def warn_missing_memory_checkpoint() -> None:
    warnings.warn(
        "GeoDRO memory queue state was missing from the checkpoint; starting with "
        "an empty queue. This preserves compatibility but exact memory-enabled "
        "resume is not guaranteed.",
        RuntimeWarning,
        stacklevel=3,
    )


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype not in _DTYPE_CODES:
        raise ValueError(f"Unsupported GeoDRO memory dtype: {dtype}.")
    return _DTYPE_CODES[dtype]
