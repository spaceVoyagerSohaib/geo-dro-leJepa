import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_reduce as functional_all_reduce


def mean_std(
    x: Tensor,
    dim: int = 0,
    keepdim: bool = False,
    unbiased: bool = True,
) -> tuple[Tensor, Tensor, int]:
    """Compute mean and std across all DDP ranks with gradient support.

    Uses a single fused all_reduce for efficiency. Gradients flow back
    through the collective operation via torch.distributed.nn.functional.

    Note: Uses E[X²] - E[X]² formula which is efficient but may have
    numerical precision issues for very large values. For most deep
    learning applications this is fine.

    Args:
        x: Input tensor of shape (..., N, ...) where N is the size of
           dimension `dim` that will be reduced.
        dim: Dimension to reduce (typically 0 for batch dimension).
        keepdim: If True, retains the reduced dimension with size 1.
        unbiased: If True, use Bessel's correction (divide by N-1).
                  If False, divide by N.

    Returns:
        mean: Global mean across all ranks. Shape depends on `keepdim`.
        std: Global std across all ranks. Shape depends on `keepdim`.
        B_global: Total number of samples across all ranks.

    Example:
        >>> # On each DDP rank with local batch
        >>> x = model(inputs)  # (B_local, D)
        >>> mean, std, B_global = ddp_mean_std_fused(x, dim=0)
        >>> # mean, std have shape (D,), B_global = B_local * world_size
    """
    if not dist.is_initialized():
        return (
            x.mean(dim, keepdim=keepdim),
            x.std(dim, keepdim=keepdim, unbiased=unbiased),
            x.shape[dim],
        )

    world_size = dist.get_world_size()
    B_local = x.shape[dim]
    B_global = B_local * world_size

    # Compute local sums (keepdim=True for stacking)
    sum_x = x.sum(dim, keepdim=True)
    sum_x2 = x.square().sum(dim, keepdim=True)

    # Stack along reduced dim, single all_reduce, then unstack
    stacked = torch.cat([sum_x, sum_x2], dim=dim)
    stacked = functional_all_reduce(stacked, op=dist.ReduceOp.SUM)
    sum_x, sum_x2 = stacked.chunk(2, dim=dim)

    # Global mean and variance
    mean = sum_x / B_global
    var = sum_x2 / B_global - mean.square()

    # Bessel's correction
    if unbiased:
        var = var * B_global / (B_global - 1)

    std = var.clamp(min=1e-8).sqrt()

    if not keepdim:
        mean = mean.squeeze(dim)
        std = std.squeeze(dim)

    return mean, std, B_global
