"""Detached distributed helpers for GeoDRO-LeJEPA adversary tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GatheredBatch:
    """Detached DDP-global batch tensor plus rank-local slice metadata."""

    tensor: torch.Tensor
    local_slice: slice
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


@torch.no_grad()
def detached_all_gather_batch_with_metadata(
    tensor: torch.Tensor,
    *,
    batch_dim: int = 0,
) -> GatheredBatch:
    detached = tensor.detach()
    moved = detached.movedim(batch_dim, 0).contiguous()
    local_size = int(moved.shape[0])
    if not is_distributed():
        return GatheredBatch(
            tensor=detached,
            local_slice=slice(0, local_size),
            sizes=(local_size,),
            offsets=(0,),
        )

    size_tensor = torch.tensor([local_size], device=moved.device, dtype=torch.long)
    gathered_size_tensors = [
        torch.empty_like(size_tensor) for _ in range(get_world_size())
    ]
    dist.all_gather(gathered_size_tensors, size_tensor)
    sizes = tuple(int(size.item()) for size in gathered_size_tensors)
    offsets = _offsets_from_sizes(sizes)
    rank = get_rank()
    max_size = max(sizes) if sizes else 0

    if max_size == 0:
        empty = moved.new_empty((0, *moved.shape[1:]))
        return GatheredBatch(
            tensor=empty.movedim(0, batch_dim).contiguous(),
            local_slice=slice(offsets[rank], offsets[rank]),
            sizes=sizes,
            offsets=offsets,
        )

    padded = _pad_batch_dim0(moved, target_size=max_size)
    gathered = [torch.empty_like(padded) for _ in range(get_world_size())]
    dist.all_gather(gathered, padded)
    stacked = torch.cat(
        [
            rank_tensor[:size]
            for rank_tensor, size in zip(gathered, sizes, strict=True)
        ],
        dim=0,
    )
    return GatheredBatch(
        tensor=stacked.movedim(0, batch_dim).contiguous(),
        local_slice=slice(offsets[rank], offsets[rank] + sizes[rank]),
        sizes=sizes,
        offsets=offsets,
    )


@torch.no_grad()
def detached_all_gather_batch(
    tensor: torch.Tensor,
    *,
    batch_dim: int = 0,
) -> torch.Tensor:
    return detached_all_gather_batch_with_metadata(tensor, batch_dim=batch_dim).tensor


def validate_gathered_batch_sizes(*gathered: GatheredBatch) -> None:
    if not gathered:
        return
    expected = gathered[0].sizes
    for index, item in enumerate(gathered[1:], start=1):
        if item.sizes != expected:
            raise ValueError(
                "Gathered DDP batch sizes must match across GeoDRO tensors; "
                f"tensor 0 has sizes {expected}, tensor {index} has sizes "
                f"{item.sizes}."
            )


def local_batch_slice(batch_size: int, *, rank: int | None = None) -> slice:
    rank = get_rank() if rank is None else rank
    start = rank * batch_size
    return slice(start, start + batch_size)


def _offsets_from_sizes(sizes: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for size in sizes:
        offsets.append(cursor)
        cursor += int(size)
    return tuple(offsets)


def _pad_batch_dim0(tensor: torch.Tensor, *, target_size: int) -> torch.Tensor:
    local_size = int(tensor.shape[0])
    if local_size == target_size:
        return tensor
    if local_size > target_size:
        raise ValueError(
            f"Cannot pad local batch with size {local_size} to smaller target "
            f"size {target_size}."
        )
    pad_shape = (target_size - local_size, *tensor.shape[1:])
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
