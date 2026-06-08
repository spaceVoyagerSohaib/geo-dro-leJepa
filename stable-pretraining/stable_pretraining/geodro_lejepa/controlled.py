"""Controlled corruption utilities for GeoDRO debug and mechanism runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


class ControlledCorruption:
    """Apply deterministic diagnostic corruptions without exposing them to GeoDRO.

    The transform writes subgroup/corruption metadata into the sample dictionary so
    downstream evaluation can audit groups. The GeoDRO forward path only consumes
    ``image`` and ``label``, so these tags are diagnostics-only.
    """

    def __init__(
        self,
        *,
        view_name: str,
        coherent_labels: Sequence[int] = (0, 1, 8, 9),
        coherent_contrast: float = 0.55,
        coherent_color: float = 0.65,
        isolated_view: str = "global_1",
        isolated_period: int = 5,
        isolated_blur_radius: float = 1.25,
        source: str = "image",
        target: str = "image",
    ) -> None:
        self.view_name = view_name
        self.coherent_labels = {int(label) for label in coherent_labels}
        self.coherent_contrast = float(coherent_contrast)
        self.coherent_color = float(coherent_color)
        self.isolated_view = isolated_view
        self.isolated_period = int(isolated_period)
        self.isolated_blur_radius = float(isolated_blur_radius)
        self.source = source
        self.target = target

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        label = _as_int(sample.get("label", -1))
        sample_idx = _as_int(sample.get("sample_idx", -1))

        image = sample[self.source]
        if not isinstance(image, Image.Image):
            raise TypeError(
                "ControlledCorruption expects a PIL image before ToImage; "
                f"got {type(image)!r}."
            )

        coherent = label in self.coherent_labels
        isolated = (
            self.view_name == self.isolated_view
            and self.isolated_period > 0
            and sample_idx >= 0
            and sample_idx % self.isolated_period == 0
        )

        if coherent:
            image = _apply_coherent_style(
                image,
                contrast=self.coherent_contrast,
                color=self.coherent_color,
            )
        if isolated:
            image = image.filter(ImageFilter.GaussianBlur(self.isolated_blur_radius))

        group = "coherent_class_corruption" if coherent else "background"
        sample[self.target] = image
        sample["geodro_view_name"] = self.view_name
        sample["geodro_group"] = group
        sample["geodro_coherent_corruption"] = bool(coherent)
        sample["geodro_isolated_view_corruption"] = bool(isolated)
        sample["geodro_corruption_tag"] = _corruption_tag(coherent, isolated)
        return sample


class CIFAR10ControlledCorruption(ControlledCorruption):
    """Backward-compatible CIFAR-10 controlled corruption transform."""

    def __init__(
        self,
        *,
        view_name: str,
        coherent_labels: Sequence[int] = (0, 1, 8, 9),
        coherent_contrast: float = 0.55,
        coherent_color: float = 0.65,
        isolated_view: str = "global_1",
        isolated_period: int = 5,
        isolated_blur_radius: float = 1.25,
        source: str = "image",
        target: str = "image",
    ) -> None:
        super().__init__(
            view_name=view_name,
            coherent_labels=coherent_labels,
            coherent_contrast=coherent_contrast,
            coherent_color=coherent_color,
            isolated_view=isolated_view,
            isolated_period=isolated_period,
            isolated_blur_radius=isolated_blur_radius,
            source=source,
            target=target,
        )


class ImageNet100ControlledCorruption(ControlledCorruption):
    """ImageNet-100 controlled corruption transform for v1 mechanism pilots."""

    def __init__(
        self,
        *,
        view_name: str,
        coherent_labels: Sequence[int] = tuple(range(30)),
        coherent_contrast: float = 0.58,
        coherent_color: float = 0.70,
        isolated_view: str = "global_1",
        isolated_period: int = 20,
        isolated_blur_radius: float = 1.50,
        source: str = "image",
        target: str = "image",
    ) -> None:
        super().__init__(
            view_name=view_name,
            coherent_labels=coherent_labels,
            coherent_contrast=coherent_contrast,
            coherent_color=coherent_color,
            isolated_view=isolated_view,
            isolated_period=isolated_period,
            isolated_blur_radius=isolated_blur_radius,
            source=source,
            target=target,
        )


def _apply_coherent_style(
    image: Image.Image,
    *,
    contrast: float,
    color: float,
) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    overlay = Image.new("RGB", image.size, (36, 64, 128))
    return Image.blend(image.convert("RGB"), overlay, alpha=0.12)


def _corruption_tag(coherent: bool, isolated: bool) -> str:
    if coherent and isolated:
        return "coherent_plus_isolated"
    if coherent:
        return "coherent"
    if isolated:
        return "isolated_view"
    return "clean"


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
