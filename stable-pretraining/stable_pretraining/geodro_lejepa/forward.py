"""Opt-in Lightning forward path for GeoDRO-LeJEPA."""

from __future__ import annotations

import math
from typing import Any

import torch

from .types import AdversaryScope


def geodro_lejepa_forward(self, batch, stage):
    """Forward function for training-only GeoDRO-LeJEPA aggregation."""
    out: dict[str, Any] = {}
    views, global_mask = _extract_views(batch)

    if views is not None:
        embeddings = [
            _extract_embedding(self.backbone(view["image"])) for view in views
        ]
        out["embedding"] = torch.cat(embeddings, dim=0)
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            if not hasattr(self, "geodro_lejepa_loss"):
                raise ValueError(
                    "geodro_lejepa_forward requires 'geodro_lejepa_loss' to be "
                    "provided, e.g. stable_pretraining.geodro_lejepa."
                    "CoherentHardnessGeoDROLeJEPALoss."
                )
            loss_fn = self.geodro_lejepa_loss
            _validate_accumulation(self, loss_fn)

            emb = torch.stack(embeddings, dim=0)
            proj = torch.stack(
                [self.projector(embedding) for embedding in embeddings], dim=0
            )
            global_mask_tensor = None
            if any(global_mask) and not all(global_mask):
                global_mask_tensor = torch.tensor(
                    global_mask, device=proj.device, dtype=torch.bool
                )

            if _adversary_scope(loss_fn) == AdversaryScope.OPTIMIZER_STEP:
                context = getattr(self, "_geodro_optimizer_step_context", None)
                if context is None:
                    raise RuntimeError(
                        "adversary_scope=optimizer_step requires the GeoDRO "
                        "optimizer-step training loop context."
                    )
                total_loss, pred_loss, sigreg_loss, pred_erm = (
                    loss_fn.weighted_replay_loss(
                        proj,
                        emb,
                        context["p_local"],
                        sigreg_scale=context["sigreg_scale"],
                        global_mask=global_mask_tensor,
                    )
                )
                out["loss"] = total_loss
                out["geodro_main_loss"] = total_loss
                out["geodro_pred_loss"] = pred_loss
                out["geodro_sigreg_loss"] = sigreg_loss
                out["geodro_pred_erm_loss"] = pred_erm
                return out

            loss_output = loss_fn(
                proj,
                emb,
                return_output=True,
                global_mask=global_mask_tensor,
                step=int(getattr(self, "global_step", 0)),
                total_steps=_estimated_total_steps(self),
                coherent_mask=_extract_sample_mask(
                    views, "geodro_coherent_corruption", proj.device
                ),
                isolated_mask=_extract_sample_mask(
                    views, "geodro_isolated_view_corruption", proj.device
                ),
            )
            out["loss"] = loss_output.total_loss
            out["geodro_main_loss"] = loss_output.total_loss
            out["geodro_pred_loss"] = loss_output.pred_loss
            out["geodro_sigreg_loss"] = loss_output.sigreg_loss
            out["geodro_pred_erm_loss"] = loss_output.pred_erm
            _log_loss_output(self, loss_output, stage)
    else:
        out["embedding"] = _extract_embedding(self.backbone(batch["image"]))
        if "label" in batch:
            out["label"] = batch["label"]
        _copy_eval_metadata(batch, out)

    return out


def _extract_views(batch) -> tuple[list[dict[str, Any]] | None, list[bool]]:
    if isinstance(batch, dict) and "image" not in batch:
        views = []
        global_mask = []
        for key, view in batch.items():
            if not isinstance(view, dict) or "image" not in view:
                continue
            views.append(view)
            if "global" in key:
                global_mask.append(True)
            elif "local" in key:
                global_mask.append(False)
            else:
                global_mask.append(True)
        if not views:
            raise ValueError(
                "Multi-view batch did not contain any view dicts with 'image'"
            )
        return views, global_mask
    if isinstance(batch, list):
        views = batch
        n_global = min(2, len(views))
        return views, [idx < n_global for idx in range(len(views))]
    return None, []


def _extract_embedding(output: torch.Tensor) -> torch.Tensor:
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim == 3:
        output = output[:, 0, :]
    return output


def _adversary_scope(loss_fn) -> AdversaryScope:
    return AdversaryScope(
        getattr(loss_fn, "adversary_scope", AdversaryScope.MICROBATCH.value)
    )


def _validate_accumulation(module, loss_fn) -> None:
    trainer = getattr(module, "trainer", None)
    if trainer is None:
        return
    accum_steps = int(
        getattr(
            trainer,
            "accumulate_grad_batches_",
            getattr(trainer, "accumulate_grad_batches", 1),
        )
    )
    if accum_steps != 1 and _adversary_scope(loss_fn) != AdversaryScope.OPTIMIZER_STEP:
        raise ValueError(
            "GeoDRO-LeJEPA iteration 1 supports only adversary_scope=microbatch "
            "with accumulate_grad_batches=1. Set adversary_scope=optimizer_step "
            "when using gradient accumulation."
        )


def _estimated_total_steps(module) -> int | None:
    trainer = getattr(module, "trainer", None)
    total = getattr(trainer, "estimated_stepping_batches", None) if trainer else None
    if total is None:
        return None
    try:
        total_float = float(total)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(total_float) or total_float <= 0:
        return None
    return int(total_float)


def _extract_sample_mask(
    views: list[dict[str, Any]],
    key: str,
    device: torch.device,
) -> torch.Tensor | None:
    masks = []
    for view in views:
        if key not in view:
            continue
        mask = torch.as_tensor(view[key], device=device).bool().flatten()
        masks.append(mask)
    if not masks:
        return None
    return torch.stack(masks, dim=0).any(dim=0)


def _log_loss_output(module, loss_output, stage: str) -> None:
    log_stage = "train" if stage == "fit" else stage
    base_logs = {
        f"{log_stage}/loss": loss_output.total_loss,
        f"{log_stage}/pred_loss": loss_output.pred_loss,
        f"{log_stage}/pred_erm_loss": loss_output.pred_erm,
        f"{log_stage}/pred_loss_minus_erm": (
            loss_output.pred_loss - loss_output.pred_erm
        ),
        f"{log_stage}/sigreg_loss": loss_output.sigreg_loss,
    }
    for key, value in base_logs.items():
        module.log(key, value, on_step=True, on_epoch=True, sync_dist=True)

    for key, value in loss_output.extra_logs.items():
        module.log(
            f"{log_stage}/geodro/{key.replace('/', '_')}",
            value,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )


def _copy_eval_metadata(batch, out: dict[str, Any]) -> None:
    for key in (
        "sample_idx",
        "geodro_view_name",
        "geodro_group",
        "geodro_coherent_corruption",
        "geodro_isolated_view_corruption",
        "geodro_corruption_tag",
    ):
        if key in batch:
            out[key] = batch[key]
