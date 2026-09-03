"""GeoDRO optimizer-step policy adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stable_pretraining.module import Module


class GeoDROOptimizerStepPolicy:
    """Delegate optimizer-step accumulation to the validated GeoDRO loop."""

    name = "geodro_optimizer_step"

    def training_step(
        self,
        module: Module,
        batch: Any,
        batch_idx: int,
    ) -> dict[str, Any]:
        """Execute the existing GeoDRO optimizer-step implementation."""
        from .optimizer_step import optimizer_step_training_step

        return optimizer_step_training_step(module, batch, batch_idx)
