"""Optimizer-step adversary scope for GeoDRO-LeJEPA."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from .distributed import (
    detached_all_gather_batch,
    detached_all_gather_batch_with_metadata,
    validate_gathered_batch_sizes,
)
from .forward import (
    _estimated_total_steps,
    _extract_embedding,
    _extract_sample_mask,
    _extract_views,
)


@dataclass
class _CollectedMicrobatch:
    batch: dict[str, Any]
    rng_state: dict[str, Any]
    output: dict[str, torch.Tensor]
    local_batch_size: int
    local_slice: slice
    graph_features_global: torch.Tensor
    li_global: torch.Tensor
    li_v_global: torch.Tensor
    coherent_mask_global: torch.Tensor | None
    isolated_mask_global: torch.Tensor | None


def optimizer_step_training_step(module, batch, batch_idx: int):
    """Run GeoDRO optimizer-step accumulation for one Lightning training batch."""
    if type(batch) is not dict:
        msg = f"batch is expected to be a dict! Not as {type(batch)}"
        raise ValueError(msg)

    batch["batch_idx"] = batch_idx
    accum_steps = _accum_steps(module)
    buffer = getattr(module, "_geodro_optimizer_step_buffer", [])

    rng_state = _capture_rng_state()
    with torch.no_grad(), _preserve_batchnorm_stats(module):
        collected = _collect_microbatch(module, batch)
    collected.rng_state = rng_state
    buffer.append(collected)
    module._geodro_optimizer_step_buffer = buffer

    if (batch_idx + 1) % accum_steps != 0:
        return _deferred_output(collected)

    try:
        optimizers, has_mock_optimizer = module._manual_optimization_handles()
        if has_mock_optimizer:
            return _deferred_output(collected)

        weights, local_weights, pred_erm = _solve_step_weights(
            module, buffer, accum_steps=accum_steps
        )
        totals = _replay_and_backward(
            module,
            buffer,
            local_weights,
            accum_steps=accum_steps,
        )
        memory_step = int(getattr(module, "global_step", 0))
        module._step_manual_optimizers(optimizers, batch_idx, accum_steps)
        _enqueue_optimizer_step_memory(module, buffer, step=memory_step)
        _log_step(module, totals, pred_erm, weights)

        state = totals["last_state"]
        state["loss"] = totals["loss"].detach()
        state["geodro_main_loss"] = totals["loss"].detach()
        state["geodro_pred_loss"] = totals["pred_loss"].detach()
        state["geodro_sigreg_loss"] = totals["sigreg_loss"].detach()
        state["geodro_pred_erm_loss"] = pred_erm.detach()
        return state
    finally:
        module._geodro_optimizer_step_buffer = []
        if hasattr(module, "_geodro_optimizer_step_context"):
            delattr(module, "_geodro_optimizer_step_context")


def _accum_steps(module) -> int:
    trainer = getattr(module, "trainer", None)
    return max(
        int(
            getattr(
                trainer,
                "accumulate_grad_batches_",
                getattr(trainer, "accumulate_grad_batches", 1),
            )
        ),
        1,
    )


def _collect_microbatch(module, batch: dict[str, Any]) -> _CollectedMicrobatch:
    views, global_mask = _extract_views(batch)
    if views is None:
        raise ValueError("GeoDRO optimizer-step scope requires multi-view batches.")

    embeddings = [_extract_embedding(module.backbone(view["image"])) for view in views]
    output: dict[str, torch.Tensor] = {
        "embedding": torch.cat([embedding.detach() for embedding in embeddings], dim=0)
    }
    if "label" in views[0]:
        output["label"] = torch.cat([view["label"].detach() for view in views], dim=0)

    emb = torch.stack(embeddings, dim=0)
    proj = torch.stack([module.projector(embedding) for embedding in embeddings], dim=0)
    global_mask_tensor = _global_mask_tensor(global_mask, proj.device)
    adversary_inputs = module.geodro_lejepa_loss.compute_adversary_inputs(
        proj,
        emb,
        global_mask=global_mask_tensor,
    )

    coherent_mask = _extract_sample_mask(
        views, "geodro_coherent_corruption", proj.device
    )
    isolated_mask = _extract_sample_mask(
        views, "geodro_isolated_view_corruption", proj.device
    )

    graph_gather = detached_all_gather_batch_with_metadata(
        adversary_inputs.graph_features.float(),
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

    return _CollectedMicrobatch(
        batch=batch,
        rng_state={},
        output=output,
        local_batch_size=int(adversary_inputs.li_local.shape[0]),
        local_slice=graph_gather.local_slice,
        graph_features_global=graph_gather.tensor,
        li_global=li_gather.tensor,
        li_v_global=li_v_gather.tensor,
        coherent_mask_global=_gather_optional_mask(coherent_mask),
        isolated_mask_global=_gather_optional_mask(isolated_mask),
    )


def _solve_step_weights(
    module,
    buffer: list[_CollectedMicrobatch],
    *,
    accum_steps: int,
):
    graph_features = torch.cat([entry.graph_features_global for entry in buffer], dim=0)
    li_global = torch.cat([entry.li_global for entry in buffer], dim=0)
    li_v_global = torch.cat([entry.li_v_global for entry in buffer], dim=1)
    coherent_mask = _cat_optional_masks(
        buffer, "coherent_mask_global", reference=li_global
    )
    isolated_mask = _cat_optional_masks(
        buffer, "isolated_mask_global", reference=li_global
    )

    weights = module.geodro_lejepa_loss.solve_adversary_weights(
        graph_features,
        li_global,
        li_v_global,
        step=int(getattr(module, "global_step", 0)),
        total_steps=_accumulation_corrected_total_steps(
            module, accum_steps=accum_steps
        ),
        coherent_mask=coherent_mask,
        isolated_mask=isolated_mask,
    )

    local_weights = []
    cursor = 0
    for entry in buffer:
        global_batch = int(entry.graph_features_global.shape[0])
        start = cursor + entry.local_slice.start
        stop = cursor + entry.local_slice.stop
        local_weights.append(weights.p_global[start:stop].detach())
        cursor += global_batch

    return weights, local_weights, li_global.mean().detach()


def _enqueue_optimizer_step_memory(
    module,
    buffer: list[_CollectedMicrobatch],
    *,
    step: int,
) -> None:
    loss_fn = getattr(module, "geodro_lejepa_loss", None)
    enqueue = getattr(loss_fn, "enqueue_memory_after_optimizer_step", None)
    if enqueue is None:
        return
    graph_features = torch.cat(
        [entry.graph_features_global for entry in buffer],
        dim=0,
    )
    enqueue(
        graph_features,
        step=step,
    )


def _replay_and_backward(
    module,
    buffer: list[_CollectedMicrobatch],
    local_weights: list[torch.Tensor],
    *,
    accum_steps: int,
):
    loss_total = None
    pred_total = None
    sigreg_total = None
    last_state = None

    for entry, p_local in zip(buffer, local_weights, strict=True):
        _restore_rng_state(entry.rng_state)
        module._geodro_optimizer_step_context = {
            "p_local": p_local,
            "sigreg_scale": 1.0 / max(len(buffer), 1),
        }
        state = module(entry.batch, stage="fit")
        main_loss = state["geodro_main_loss"]
        raw_loss = state["loss"]
        callback_loss = raw_loss - main_loss
        backward_loss = main_loss + callback_loss / accum_steps

        module.manual_backward(backward_loss)
        module.after_manual_backward()

        loss_total = _add_or_init(loss_total, main_loss.detach())
        pred_total = _add_or_init(pred_total, state["geodro_pred_loss"].detach())
        sigreg_total = _add_or_init(
            sigreg_total,
            state["geodro_sigreg_loss"].detach() / max(len(buffer), 1),
        )
        last_state = state

    if last_state is None:
        raise RuntimeError("GeoDRO optimizer-step replay had no buffered batches.")
    return {
        "loss": loss_total,
        "pred_loss": pred_total,
        "sigreg_loss": sigreg_total,
        "last_state": last_state,
    }


def _log_step(module, totals, pred_erm: torch.Tensor, weights) -> None:
    base_logs = {
        "train/loss": totals["loss"],
        "train/pred_loss": totals["pred_loss"],
        "train/pred_erm_loss": pred_erm,
        "train/pred_loss_minus_erm": totals["pred_loss"] - pred_erm,
        "train/sigreg_loss": totals["sigreg_loss"],
    }
    for key, value in base_logs.items():
        module.log(key, value, on_step=True, on_epoch=True, sync_dist=True)

    for key, value in weights.extra_logs.items():
        module.log(
            f"train/geodro/{key.replace('/', '_')}",
            value,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )


def _deferred_output(collected: _CollectedMicrobatch) -> dict[str, torch.Tensor]:
    output = dict(collected.output)
    device = output["embedding"].device
    output["loss"] = torch.zeros((), device=device)
    output["geodro_deferred_optimizer_step"] = torch.tensor(True, device=device)
    return output


def _accumulation_corrected_total_steps(module, *, accum_steps: int) -> int | None:
    """Return a warmup total that matches optimizer-step graph accumulation.

    Lightning reports the long-horizon estimate before this custom replay path
    collapses accumulated microbatches into one GeoDRO graph solve. Dividing by
    the configured accumulation factor keeps warmup fractions epoch-aligned for
    the optimizer-step adversary scope.
    """
    total_steps = _estimated_total_steps(module)
    if total_steps is None:
        return None
    return max(1, math.ceil(int(total_steps) / max(int(accum_steps), 1)))


def _global_mask_tensor(
    global_mask: list[bool],
    device: torch.device,
) -> torch.Tensor | None:
    if any(global_mask) and not all(global_mask):
        return torch.tensor(global_mask, device=device, dtype=torch.bool)
    return None


def _gather_optional_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return detached_all_gather_batch(mask.bool(), batch_dim=0)


def _cat_optional_masks(
    buffer: list[_CollectedMicrobatch],
    field_name: str,
    *,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    tensors = [getattr(entry, field_name) for entry in buffer]
    if all(tensor is None for tensor in tensors):
        return None
    filled = []
    for entry, tensor in zip(buffer, tensors, strict=True):
        if tensor is None:
            filled.append(
                torch.zeros(
                    entry.li_global.shape[0],
                    device=reference.device,
                    dtype=torch.bool,
                )
            )
        else:
            filled.append(tensor.to(device=reference.device, dtype=torch.bool))
    return torch.cat(filled, dim=0)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def _preserve_batchnorm_stats(module):
    snapshots = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            if not child.track_running_stats:
                continue
            snapshots.append(
                (
                    child,
                    None if child.running_mean is None else child.running_mean.clone(),
                    None if child.running_var is None else child.running_var.clone(),
                    None
                    if child.num_batches_tracked is None
                    else child.num_batches_tracked.clone(),
                )
            )
    try:
        yield
    finally:
        for child, running_mean, running_var, num_batches_tracked in snapshots:
            if running_mean is not None:
                child.running_mean.copy_(running_mean)
            if running_var is not None:
                child.running_var.copy_(running_var)
            if num_batches_tracked is not None:
                child.num_batches_tracked.copy_(num_batches_tracked)


def _add_or_init(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    if current is None:
        return value
    return current + value
