#!/usr/bin/env python3
"""Post-pretraining evaluation harness for GeoDRO-LeJEPA v1 runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.linear_eval import FeatureExtractor, LinearProbe, load_checkpoint
from scripts._eval_metrics import (
    auroc_max_softmax,
    clean_vs_shifted_gap,
    expected_calibration_error,
    knn_probe,
    mean_corruption_error,
    negative_log_likelihood,
    selective_prediction_auc,
    top_k_accuracy,
)

# Schema version emitted in metrics.json.
# v1 = legacy: top-1 only on the three existing dispatchers.
# v2 = current: + top-5, kNN, ECE, NLL on dispatchers and mCE/clean-gap on IN-C,
#               plus six new dispatchers (sketch/r/a/o, celeba, camelyon17).
EVAL_SCHEMA_VERSION = 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IMAGENET100_DEFAULT = "ilee0022/ImageNet100"
IMAGENETC_DEFAULT = "WNJXYK/TTA-ImageNet-C"
WATERBIRDS_DEFAULT = "grodino/waterbirds"
PARQUET_REVISION = "refs/convert/parquet"
IMAGENETC_REVISION = "v1.0"
IMAGENETC_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
WATERBIRDS_GROUP_NAMES = (
    "landbird_land",
    "landbird_water",
    "waterbird_land",
    "waterbird_water",
)
IMAGENET1K_LABEL_ALIASES = {
    # ImageNet-100 uses "rooster"; torchvision's ImageNet-1K category is "cock".
    "rooster": ("cock",),
}


class EvalDataset(torch.utils.data.Dataset):
    """Dataset wrapper returning image, remapped label, and numeric group id."""

    def __init__(
        self,
        dataset,
        transform: Callable[[Any], torch.Tensor],
        *,
        label_map: Mapping[int, int] | None = None,
        group_fn: Callable[[Mapping[str, Any], int, int], int] | None = None,
        skip_unmapped: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.label_map = dict(label_map or {})
        self.group_fn = group_fn
        self.skip_unmapped = skip_unmapped
        self.indices = _select_indices(dataset, self.label_map, skip_unmapped, max_samples)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.dataset[self.indices[idx]]
        raw_label = int(_to_scalar(item["label"]))
        if self.label_map:
            label = self.label_map.get(raw_label)
            if label is None:
                raise KeyError(f"Label {raw_label} is not in label_map")
        else:
            label = raw_label
        group = self.group_fn(item, raw_label, label) if self.group_fn else -1
        image = item["image"].convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(label, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }


def _select_indices(
    dataset,
    label_map: Mapping[int, int],
    skip_unmapped: bool,
    max_samples: int | None,
) -> list[int]:
    if not label_map or not skip_unmapped:
        n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        return list(range(n))

    allowed = set(label_map)
    try:
        labels = dataset["label"]
    except (KeyError, TypeError):
        labels = [dataset[idx]["label"] for idx in range(len(dataset))]

    indices = [idx for idx, label in enumerate(labels) if int(_to_scalar(label)) in allowed]
    if max_samples is not None:
        indices = indices[:max_samples]
    return indices


def _to_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def parse_int_set(spec: str) -> set[int]:
    """Parse comma-separated integers and inclusive ranges."""
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))
    return values


def parse_str_list(spec: str | None, default: Iterable[str]) -> list[str]:
    if spec is None or not spec.strip():
        return list(default)
    return [part.strip() for part in spec.split(",") if part.strip()]


def parse_int_list(spec: str | None, default: Iterable[int]) -> list[int]:
    if spec is None or not spec.strip():
        return list(default)
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def accuracy(correct: int | float, total: int | float) -> float | None:
    return None if total == 0 else float(correct) / float(total)


def summarize_group_counts(
    group_correct: Mapping[int, int],
    group_total: Mapping[int, int],
    group_names: Mapping[int, str],
    *,
    prefix: str,
) -> dict[str, float | int | None]:
    """Build stable group metric keys from integer group counts."""
    metrics: dict[str, float | int | None] = {}
    group_accs = []
    for group_id, name in group_names.items():
        total = int(group_total.get(group_id, 0))
        correct = int(group_correct.get(group_id, 0))
        acc = accuracy(correct, total)
        metrics[f"{prefix}/{name}_acc"] = acc
        metrics[f"{prefix}/{name}_correct"] = correct
        metrics[f"{prefix}/{name}_total"] = total
        if acc is not None:
            group_accs.append(acc)
    metrics[f"{prefix}/worst_group_acc"] = min(group_accs) if group_accs else None
    return metrics


def summarize_controlled_counts(
    group_correct: Mapping[int, int],
    group_total: Mapping[int, int],
    *,
    prefix: str,
) -> dict[str, float | int | None]:
    group_names = {0: "background", 1: "coherent"}
    metrics = summarize_group_counts(group_correct, group_total, group_names, prefix=prefix)
    metrics[f"{prefix}/background_acc"] = metrics[f"{prefix}/background_acc"]
    metrics[f"{prefix}/coherent_acc"] = metrics[f"{prefix}/coherent_acc"]
    accs = [
        value
        for key, value in metrics.items()
        if key in {f"{prefix}/background_acc", f"{prefix}/coherent_acc"}
        and value is not None
    ]
    metrics[f"{prefix}/worst_subset_acc"] = min(accs) if accs else None
    return metrics


def aggregate_imagenetc_metrics(per_split: Mapping[str, Mapping[str, float | int]]) -> dict[str, float]:
    """Aggregate ImageNet-C accuracy by corruption and severity."""
    by_corruption: dict[str, list[float]] = {}
    by_severity: dict[int, list[float]] = {}
    all_accs = []
    for split_name, metrics in per_split.items():
        acc = metrics.get("acc")
        if acc is None:
            continue
        acc = float(acc)
        all_accs.append(acc)
        corruption, severity = split_name.rsplit("/severity_", maxsplit=1)
        by_corruption.setdefault(corruption, []).append(acc)
        by_severity.setdefault(int(severity), []).append(acc)

    out = {
        "imagenetc/mean_acc": _mean(all_accs),
        "imagenetc/mean_error": 1.0 - _mean(all_accs),
    }
    for corruption, values in by_corruption.items():
        out[f"imagenetc/{corruption}/mean_acc"] = _mean(values)
    for severity, values in sorted(by_severity.items()):
        out[f"imagenetc/severity_{severity}/mean_acc"] = _mean(values)
    return out


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def build_imagenet100_to_imagenet1k_map(
    label_texts: Mapping[int, str],
    imagenet1k_categories: list[str],
) -> dict[int, int]:
    """Map ImageNet-100 target labels to ImageNet-1K source labels by class text."""
    category_index = {_normalize_label(name): idx for idx, name in enumerate(imagenet1k_categories)}
    out: dict[int, int] = {}
    missing = {}
    for label, text in label_texts.items():
        candidates = []
        for candidate in [_normalize_label(text), *(_normalize_label(part) for part in str(text).split(","))]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
            for alias in IMAGENET1K_LABEL_ALIASES.get(candidate, ()):
                alias = _normalize_label(alias)
                if alias and alias not in candidates:
                    candidates.append(alias)
        match = next((category_index[candidate] for candidate in candidates if candidate in category_index), None)
        if match is None:
            for idx, category in enumerate(imagenet1k_categories):
                category_norm = _normalize_label(category)
                if any(candidate and candidate in category_norm for candidate in candidates[1:]):
                    match = idx
                    break
        if match is None:
            missing[label] = text
        else:
            out[int(label)] = int(match)
    if missing:
        raise ValueError(f"Could not map ImageNet-100 labels to ImageNet-1K: {missing}")
    return out


def invert_label_map(label_map: Mapping[int, int]) -> dict[int, int]:
    """Invert target->source label map into source->target label map."""
    return {source: target for target, source in label_map.items()}


def _normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def get_imagenet1k_categories() -> list[str]:
    from torchvision.models import ResNet50_Weights

    return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])


def infer_label_texts(dataset) -> dict[int, str]:
    """Infer label text for ImageNet-100 labels from text column or ClassLabel names."""
    if "text" in getattr(dataset, "column_names", []):
        texts: dict[int, str] = {}
        for row in dataset:
            label = int(_to_scalar(row["label"]))
            texts.setdefault(label, str(row["text"]))
            if len(texts) >= 100:
                break
        return texts

    feature = getattr(dataset, "features", {}).get("label")
    names = getattr(feature, "names", None)
    if names:
        return {idx: name for idx, name in enumerate(names)}

    raise ValueError("Cannot infer label texts; provide a dataset with text or ClassLabel names.")


def load_dataset_split(
    dataset_path: str,
    *,
    split: str,
    name: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
):
    """Load one Hugging Face or save_to_disk split."""
    from datasets import DatasetDict, load_dataset, load_from_disk

    path = Path(dataset_path)
    if path.exists():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            return dataset[split]
        return dataset

    kwargs: dict[str, Any] = {"split": split}
    if name:
        kwargs["name"] = name
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        return load_dataset(dataset_path, **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "dataset scripts are no longer supported" in message and revision != PARQUET_REVISION:
            kwargs["revision"] = PARQUET_REVISION
            return load_dataset(dataset_path, **kwargs)
        raise


def default_cache_dir() -> str | None:
    if os.environ.get("HF_DATASETS_CACHE"):
        return os.environ["HF_DATASETS_CACHE"]
    if os.environ.get("HF_HOME"):
        return str(Path(os.environ["HF_HOME"]) / "datasets")
    if os.environ.get("MCMLSCRATCH"):
        return str(Path(os.environ["MCMLSCRATCH"]) / ".cache" / "huggingface" / "datasets")
    return None


def build_transforms(img_size: int):
    from torchvision.transforms import v2

    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize(int(img_size * 256 / 224)),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_loader(
    dataset,
    transform,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    label_map: Mapping[int, int] | None = None,
    group_fn: Callable[[Mapping[str, Any], int, int], int] | None = None,
    skip_unmapped: bool = False,
    max_samples: int | None = None,
):
    wrapped = EvalDataset(
        dataset,
        transform,
        label_map=label_map,
        group_fn=group_fn,
        skip_unmapped=skip_unmapped,
        max_samples=max_samples,
    )
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def build_feature_extractor(args) -> tuple[nn.Module, int]:
    import timm

    backbone = timm.create_model(
        args.backbone,
        pretrained=False,
        num_classes=0,
        img_size=args.img_size,
    )
    state_dict = load_checkpoint(args.checkpoint, weights_only=args.weights_only)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading backbone: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading backbone: %s", unexpected)

    model_type = "vit" if "vit" in args.backbone.lower() else "cnn"
    feature_extractor = FeatureExtractor(backbone, layers=args.feature_layers, model_type=model_type)
    feature_extractor.eval()
    with torch.no_grad():
        dim = feature_extractor(torch.randn(1, 3, args.img_size, args.img_size)).shape[-1]
    logger.info("Feature dimension: %s", dim)
    return feature_extractor, int(dim)


def train_probe(
    feature_extractor: nn.Module,
    probe: nn.Module,
    train_loader: DataLoader,
    monitor_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    device: str,
    wandb_run=None,
) -> dict[str, float]:
    feature_extractor = feature_extractor.to(device)
    probe = probe.to(device)
    feature_extractor.eval()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"
    scaler = GradScaler() if use_amp else None

    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, len(train_loader)))
    warmup_steps = min(total_steps - 1, max(0, warmup_epochs * max(1, len(train_loader))))
    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps)),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_monitor_acc = 0.0
    final_monitor_acc = 0.0
    for epoch in range(epochs):
        started = time.time()
        probe.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Probe epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.no_grad():
                with (autocast(device_type) if use_amp else nullcontext()):
                    features = feature_extractor(images)
            with (autocast(device_type) if use_amp else nullcontext()):
                logits = probe(features)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            if use_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            batch_size = labels.size(0)
            train_loss += float(loss.item()) * batch_size
            train_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            train_total += batch_size
            pbar.set_postfix({"loss": loss.item(), "acc": accuracy(train_correct, train_total)})

        monitor = evaluate_loader(feature_extractor, probe, monitor_loader, device=device, prefix="monitor")
        final_monitor_acc = float(monitor["monitor/acc"] or 0.0)
        best_monitor_acc = max(best_monitor_acc, final_monitor_acc)
        train_metrics = {
            "epoch": epoch + 1,
            "train/loss": float(train_loss / max(train_total, 1)),
            "train/acc": float(accuracy(train_correct, train_total) or 0.0),
            "monitor/acc": final_monitor_acc,
            "monitor/best_acc": best_monitor_acc,
            "time/epoch_seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.info("Epoch %s metrics: %s", epoch + 1, train_metrics)
        if wandb_run is not None:
            wandb_run.log(train_metrics, step=epoch + 1)

    return {
        "monitor/final_acc": final_monitor_acc,
        "monitor/best_acc": best_monitor_acc,
    }


@torch.no_grad()
def evaluate_loader(
    feature_extractor: nn.Module,
    probe: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    prefix: str,
    group_names: Mapping[int, str] | None = None,
) -> dict[str, float | int | None]:
    feature_extractor.eval()
    probe.eval()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"
    total = 0
    correct = 0
    loss_sum = 0.0
    group_correct: dict[int, int] = {}
    group_total: dict[int, int] = {}
    for batch in tqdm(loader, desc=f"Eval {prefix}"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        groups = batch["group"].to(device, non_blocking=True)
        with autocast(device_type) if use_amp else nullcontext():
            logits = probe(feature_extractor(images))
            loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        matches = preds == labels
        batch_size = labels.size(0)
        total += batch_size
        correct += int(matches.sum().item())
        loss_sum += float(loss.item()) * batch_size
        for group_id in torch.unique(groups).tolist():
            group_id = int(group_id)
            if group_id < 0:
                continue
            mask = groups == group_id
            group_total[group_id] = group_total.get(group_id, 0) + int(mask.sum().item())
            group_correct[group_id] = group_correct.get(group_id, 0) + int(matches[mask].sum().item())

    metrics: dict[str, float | int | None] = {
        f"{prefix}/loss": loss_sum / max(total, 1),
        f"{prefix}/acc": accuracy(correct, total),
        f"{prefix}/correct": correct,
        f"{prefix}/total": total,
    }
    if group_names:
        metrics.update(summarize_group_counts(group_correct, group_total, group_names, prefix=prefix))
    return metrics


@torch.no_grad()
def collect_predictions(
    feature_extractor: nn.Module,
    probe: nn.Module | None,
    loader: DataLoader,
    *,
    device: str,
    keep_features: bool = False,
) -> dict[str, torch.Tensor]:
    """Run a single forward pass and stash logits, labels, (optional) features.

    Used by Phase C / Phase D dispatchers to compute kNN, ECE, NLL, top-5,
    AUROC, and selective-prediction AUC without re-running the loader.
    """
    feature_extractor.eval()
    if probe is not None:
        probe.eval()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"

    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    group_chunks: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Collect predictions"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        groups = batch["group"].to(device, non_blocking=True)
        with autocast(device_type) if use_amp else nullcontext():
            features = feature_extractor(images)
            logits = probe(features) if probe is not None else None
        if logits is not None:
            logits_chunks.append(logits.detach().float().cpu())
        label_chunks.append(labels.detach().cpu())
        group_chunks.append(groups.detach().cpu())
        if keep_features:
            feature_chunks.append(features.detach().float().cpu())
    out: dict[str, torch.Tensor] = {
        "labels": torch.cat(label_chunks) if label_chunks else torch.empty(0, dtype=torch.long),
        "groups": torch.cat(group_chunks) if group_chunks else torch.empty(0, dtype=torch.long),
    }
    if logits_chunks:
        out["logits"] = torch.cat(logits_chunks)
    if feature_chunks:
        out["features"] = torch.cat(feature_chunks)
    return out


def compute_extra_metrics(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix: str,
    train_features: torch.Tensor | None = None,
    train_labels: torch.Tensor | None = None,
    eval_features: torch.Tensor | None = None,
    knn_k: int = 20,
) -> dict[str, float]:
    """Compute the v2 metric bundle on already-materialized predictions.

    All inputs are CPU tensors; the helpers in `scripts._eval_metrics` are pure.
    Empty inputs are tolerated and yield zero-filled stats.
    """
    out: dict[str, float] = {}
    if logits.numel() == 0 or labels.numel() == 0:
        return out
    probs = logits.softmax(dim=-1)
    for k, v in top_k_accuracy(logits, labels, ks=(1, 5)).items():
        out[f"{prefix}/{k}"] = v
    out[f"{prefix}/ece"] = expected_calibration_error(probs, labels)
    out[f"{prefix}/nll"] = negative_log_likelihood(logits, labels)
    out[f"{prefix}/selective_prediction_aurc"] = selective_prediction_auc(probs, labels)
    if train_features is not None and train_labels is not None and eval_features is not None:
        knn = knn_probe(
            train_features,
            train_labels,
            eval_features,
            labels,
            k=knn_k,
        )
        for k, v in knn.items():
            out[f"{prefix}/{k}"] = v
    return out


@torch.no_grad()
def collect_train_features(
    feature_extractor: nn.Module,
    train_dataset,
    transform,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features over the full clean-train set for kNN reference.

    Returns (features, labels) both on CPU, matching the order yielded by
    `create_loader`. Mirrors the build_clean_probe transform conventions.
    """
    train_loader = create_loader(
        train_dataset,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        max_samples=max_samples,
    )
    pred = collect_predictions(
        feature_extractor,
        probe=None,
        loader=train_loader,
        device=device,
        keep_features=True,
    )
    return pred.get("features", torch.empty(0)), pred["labels"]


def imagenet100_group_fn(coherent_labels: set[int]) -> Callable[[Mapping[str, Any], int, int], int]:
    def group_fn(item: Mapping[str, Any], raw_label: int, mapped_label: int) -> int:
        return 1 if mapped_label in coherent_labels else 0

    return group_fn


def waterbirds_group_fn(item: Mapping[str, Any], raw_label: int, mapped_label: int) -> int:
    place = int(_to_scalar(item["place"]))
    return raw_label * 2 + place


def load_imagenet100_splits(args):
    cache_dir = args.imagenet100_cache_dir or args.dataset_cache_dir or default_cache_dir()
    train = load_dataset_split(
        args.imagenet100_dataset,
        name=args.imagenet100_dataset_name or None,
        split=args.imagenet100_train_split,
        revision=args.imagenet100_revision,
        cache_dir=cache_dir,
    )
    val = load_dataset_split(
        args.imagenet100_dataset,
        name=args.imagenet100_dataset_name or None,
        split=args.imagenet100_val_split,
        revision=args.imagenet100_revision,
        cache_dir=cache_dir,
    )
    return train, val


def build_clean_probe(args, feature_extractor, feature_dim, wandb_run=None):
    train_transform, eval_transform = build_transforms(args.img_size)
    train_ds, val_ds = load_imagenet100_splits(args)
    train_loader = create_loader(
        train_ds,
        train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_samples=args.max_train_samples,
    )
    monitor_loader = create_loader(
        val_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_eval_samples,
    )
    probe = LinearProbe(feature_dim, args.num_classes, normalize=not args.no_normalize)
    train_metrics = train_probe(
        feature_extractor,
        probe,
        train_loader,
        monitor_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        wandb_run=wandb_run,
    )
    return probe, train_metrics, train_ds, val_ds, eval_transform


def run_imagenet100ctrl(args, feature_extractor, feature_dim, wandb_run=None):
    probe, train_metrics, train_ds, val_ds, eval_transform = build_clean_probe(
        args, feature_extractor, feature_dim, wandb_run
    )
    coherent_labels = parse_int_set(args.coherent_labels)
    val_loader = create_loader(
        val_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        group_fn=imagenet100_group_fn(coherent_labels),
        max_samples=args.max_eval_samples,
    )
    metrics = train_metrics | evaluate_loader(
        feature_extractor,
        probe,
        val_loader,
        device=args.device,
        prefix="imagenet100ctrl/val",
        group_names={0: "background", 1: "coherent"},
    )
    metrics.update(
        summarize_controlled_counts(
            {
                0: int(metrics.get("imagenet100ctrl/val/background_correct") or 0),
                1: int(metrics.get("imagenet100ctrl/val/coherent_correct") or 0),
            },
            {
                0: int(metrics.get("imagenet100ctrl/val/background_total") or 0),
                1: int(metrics.get("imagenet100ctrl/val/coherent_total") or 0),
            },
            prefix="imagenet100ctrl/val",
        )
    )
    # v2 schema additions: top-5, ECE, NLL, selective-prediction AURC, kNN.
    val_pred = collect_predictions(
        feature_extractor,
        probe,
        val_loader,
        device=args.device,
        keep_features=True,
    )
    train_features, train_labels = collect_train_features(
        feature_extractor,
        train_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_samples=args.max_train_samples,
    )
    metrics.update(
        compute_extra_metrics(
            logits=val_pred["logits"],
            labels=val_pred["labels"],
            prefix="imagenet100ctrl/val",
            train_features=train_features,
            train_labels=train_labels,
            eval_features=val_pred.get("features"),
            knn_k=args.knn_k,
        )
    )
    return probe, metrics


def run_imagenetc(args, feature_extractor, feature_dim, wandb_run=None):
    probe, train_metrics, train_ds, _val_ds, eval_transform = build_clean_probe(
        args, feature_extractor, feature_dim, wandb_run
    )
    label_texts = infer_label_texts(train_ds)
    imagenet100_to_1k = build_imagenet100_to_imagenet1k_map(
        label_texts, get_imagenet1k_categories()
    )
    imagenetc_to_100 = invert_label_map(imagenet100_to_1k)
    cache_dir = args.imagenetc_cache_dir or args.dataset_cache_dir or default_cache_dir()
    corruptions = parse_str_list(args.imagenetc_corruptions, IMAGENETC_CORRUPTIONS)
    severities = parse_int_list(args.imagenetc_severities, range(1, 6))
    per_split: dict[str, dict[str, float | int | None]] = {}
    # v2 additions: pool logits across severities per corruption to compute
    # top-5/ECE/NLL once per corruption.
    per_corruption_logits: dict[str, list[torch.Tensor]] = {}
    per_corruption_labels: dict[str, list[torch.Tensor]] = {}
    for corruption in corruptions:
        for severity in severities:
            split = f"severity_{severity}"
            dataset = load_dataset_split(
                args.imagenetc_dataset,
                name=corruption,
                split=split,
                revision=args.imagenetc_revision,
                cache_dir=cache_dir,
            )
            loader = create_loader(
                dataset,
                eval_transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                label_map=imagenetc_to_100,
                skip_unmapped=True,
                max_samples=args.max_imagenetc_samples,
            )
            key = f"{corruption}/severity_{severity}"
            split_metrics = evaluate_loader(
                feature_extractor,
                probe,
                loader,
                device=args.device,
                prefix=f"imagenetc/{key}",
            )
            per_split[key] = {
                "acc": split_metrics[f"imagenetc/{key}/acc"],
                "correct": split_metrics[f"imagenetc/{key}/correct"],
                "total": split_metrics[f"imagenetc/{key}/total"],
            }
            # Second pass to collect logits/labels for top-5/ECE/NLL.
            preds = collect_predictions(
                feature_extractor,
                probe,
                loader,
                device=args.device,
                keep_features=False,
            )
            per_corruption_logits.setdefault(corruption, []).append(preds["logits"])
            per_corruption_labels.setdefault(corruption, []).append(preds["labels"])
    flat = train_metrics | {f"imagenetc/{k}/{m}": v for k, vals in per_split.items() for m, v in vals.items()}
    flat.update(aggregate_imagenetc_metrics(per_split))
    flat["imagenetc/remapped_num_classes"] = len(imagenetc_to_100)

    # Per-corruption top-5/ECE/NLL averaged across severities.
    per_corruption_acc: dict[str, float] = {}
    for corruption in corruptions:
        if not per_corruption_logits.get(corruption):
            continue
        logits = torch.cat(per_corruption_logits[corruption], dim=0)
        labels = torch.cat(per_corruption_labels[corruption], dim=0)
        extras = compute_extra_metrics(
            logits=logits,
            labels=labels,
            prefix=f"imagenetc/{corruption}",
        )
        flat.update(extras)
        per_corruption_acc[corruption] = float(extras.get(f"imagenetc/{corruption}/top1_acc", 0.0))

    # mCE (15 standard corruptions, AlexNet-IN1K denominator).
    if per_corruption_acc:
        for k, v in mean_corruption_error(per_corruption_acc).items():
            flat[f"imagenetc/{k}"] = v

    # Clean-vs-corrupted gap, using the in-distribution monitor accuracy
    # captured during probe training.
    clean_acc = float(train_metrics.get("monitor/best_acc") or 0.0)
    mean_corrupted_acc = float(flat.get("imagenetc/mean_acc") or 0.0)
    if clean_acc > 0 and mean_corrupted_acc > 0:
        flat["imagenetc/clean_vs_corrupted_gap"] = clean_vs_shifted_gap(
            clean_acc, mean_corrupted_acc
        )
    return probe, flat


def run_waterbirds(args, feature_extractor, feature_dim, wandb_run=None):
    train_transform, eval_transform = build_transforms(args.img_size)
    cache_dir = args.waterbirds_cache_dir or args.dataset_cache_dir or default_cache_dir()
    train_ds = load_dataset_split(
        args.waterbirds_dataset,
        split=args.waterbirds_train_split,
        revision=args.waterbirds_revision,
        cache_dir=cache_dir,
    )
    val_ds = load_dataset_split(
        args.waterbirds_dataset,
        split=args.waterbirds_val_split,
        revision=args.waterbirds_revision,
        cache_dir=cache_dir,
    )
    test_ds = load_dataset_split(
        args.waterbirds_dataset,
        split=args.waterbirds_test_split,
        revision=args.waterbirds_revision,
        cache_dir=cache_dir,
    )
    train_loader = create_loader(
        train_ds,
        train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_samples=args.max_train_samples,
    )
    val_loader = create_loader(
        val_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        group_fn=waterbirds_group_fn,
        max_samples=args.max_eval_samples,
    )
    test_loader = create_loader(
        test_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        group_fn=waterbirds_group_fn,
        max_samples=args.max_eval_samples,
    )
    probe = LinearProbe(feature_dim, 2, normalize=not args.no_normalize)
    train_metrics = train_probe(
        feature_extractor,
        probe,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        wandb_run=wandb_run,
    )
    group_names = {idx: name for idx, name in enumerate(WATERBIRDS_GROUP_NAMES)}
    metrics = train_metrics
    metrics.update(
        evaluate_loader(
            feature_extractor,
            probe,
            val_loader,
            device=args.device,
            prefix="waterbirds/val",
            group_names=group_names,
        )
    )
    metrics.update(
        evaluate_loader(
            feature_extractor,
            probe,
            test_loader,
            device=args.device,
            prefix="waterbirds/test",
            group_names=group_names,
        )
    )
    # v2 schema additions: top-5 (degenerate at 2 classes but kept for parity),
    # ECE, NLL, AURC, plus a kNN probe on the val/test loaders.
    train_features, train_labels = collect_train_features(
        feature_extractor,
        train_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_samples=args.max_train_samples,
    )
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        preds = collect_predictions(
            feature_extractor,
            probe,
            loader,
            device=args.device,
            keep_features=True,
        )
        metrics.update(
            compute_extra_metrics(
                logits=preds["logits"],
                labels=preds["labels"],
                prefix=f"waterbirds/{split_name}",
                train_features=train_features,
                train_labels=train_labels,
                eval_features=preds.get("features"),
                knn_k=args.knn_k,
            )
        )
    return probe, metrics


# =============================================================================
# Phase D — new dispatchers for IN-Sketch / IN-R / IN-A / IN-O / CelebA / Camelyon17
# =============================================================================


def _imagenet1k_to_imagenet100(args, train_ds) -> dict[int, int]:
    """Build the IN-1k -> IN-100 reverse map (shared by sketch/R/A dispatchers)."""
    label_texts = infer_label_texts(train_ds)
    in100_to_1k = build_imagenet100_to_imagenet1k_map(label_texts, get_imagenet1k_categories())
    return invert_label_map(in100_to_1k)


def _load_imagenet_tarball_dataset(extracted_dir: Path) -> Any:
    """Load an extracted IN-{R,A} tarball as an HF Dataset with IN-1k labels.

    The tarball layout is `<extracted_dir>/n*/.JPEG`. Each WNID-named
    subdirectory is mapped to its IN-1k integer index via the canonical
    1000-WNID list. Subdirectories whose WNID is not in IN-1k are dropped —
    correct for IN-R/A (every WNID is in IN-1k by construction). For IN-O,
    use `_load_imagenet_o_tarball_dataset` instead, which keeps all images
    with a dummy label.

    Source order for the WNID list:
      1. `<extracted_dir>/imagenet1k_wnids.json` written by the prewarm sbatch.
      2. Fall back to reading `timm.data/imagenet_synsets.txt` (timm is a
         project dep). Identical ordering to the prewarm-side source.
    """
    from datasets import Dataset, Features, Image, Value

    wnid_to_idx = _load_imagenet1k_wnid_index(extracted_dir)

    rows: list[dict[str, Any]] = []
    skipped_wnids: list[str] = []
    for wnid_dir in sorted(extracted_dir.iterdir()):
        if not wnid_dir.is_dir():
            continue
        idx = wnid_to_idx.get(wnid_dir.name)
        if idx is None:
            # IN-R/A WNIDs are all in IN-1k; non-WNID subdirs (e.g. README) are
            # benign. Track for diagnostics so the user can spot real misses.
            if wnid_dir.name.startswith("n") and wnid_dir.name[1:].isdigit():
                skipped_wnids.append(wnid_dir.name)
            continue
        for image_path in sorted(wnid_dir.iterdir()):
            if not image_path.is_file():
                continue
            rows.append({"image": str(image_path), "label": int(idx)})

    if skipped_wnids:
        logger.warning(
            "Dropped %d WNID subdirs not present in IN-1k while loading %s: %s%s",
            len(skipped_wnids),
            extracted_dir,
            skipped_wnids[:5],
            " ..." if len(skipped_wnids) > 5 else "",
        )

    features = Features({"image": Image(decode=True), "label": Value("int64")})
    return Dataset.from_list(rows, features=features)


def _load_imagenet_o_tarball_dataset(extracted_dir: Path) -> Any:
    """Load an extracted IN-O tarball as an HF Dataset with dummy labels.

    IN-O classes are by construction *not* in IN-1k, so the WNID -> IN-1k
    index map is meaningless. The OOD-detection metric (`auroc_max_softmax`)
    only needs the model's output probabilities, never the labels. We
    therefore walk every image file under the extraction directory and
    emit rows with `label: 0`. This keeps the existing `EvalDataset` /
    `create_loader` shape working unchanged.
    """
    from datasets import Dataset, Features, Image, Value

    image_extensions = {".jpeg", ".jpg", ".png", ".webp", ".JPEG", ".JPG", ".PNG"}
    rows: list[dict[str, Any]] = []
    for path in sorted(extracted_dir.rglob("*")):
        if path.is_file() and path.suffix in image_extensions:
            rows.append({"image": str(path), "label": 0})

    features = Features({"image": Image(decode=True), "label": Value("int64")})
    return Dataset.from_list(rows, features=features)


def _load_imagenet1k_wnid_index(extracted_dir: Path) -> dict[str, int]:
    """Resolve `WNID -> IN-1k integer index` from sidecar or timm package.

    Sidecar (`<extracted_dir>/imagenet1k_wnids.json`) is the canonical source;
    it is written by `scripts/slurm/download_datasets.sbatch` at prewarm time.
    The fallback reads timm's `imagenet_synsets.txt` resource directly (timm
    is a project dep) so manual eval invocations without the prewarm sentinel
    still work.
    """
    sidecar = extracted_dir / "imagenet1k_wnids.json"
    if sidecar.is_file():
        wnids = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        try:
            from importlib import resources

            # timm packages the canonical IN-1k synset list under
            # `timm.data._info/imagenet_synsets.txt` (verified against
            # timm 1.0.x). Older / forked builds may expose `timm.data`
            # directly, so we fall back to that location too before
            # giving up.
            text: str | None = None
            for resource_pkg in ("timm.data._info", "timm.data"):
                try:
                    text = (
                        resources.files(resource_pkg)
                        .joinpath("imagenet_synsets.txt")
                        .read_text()
                    )
                    break
                except (FileNotFoundError, ModuleNotFoundError):
                    continue
            if text is None:
                raise FileNotFoundError(
                    "imagenet_synsets.txt not found under timm.data._info or timm.data"
                )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Cannot resolve WNID -> IN-1k index mapping. Re-run the prewarm "
                f"sbatch so it writes `imagenet1k_wnids.json` next to {extracted_dir}, "
                "or ensure timm is installed in the eval env."
            ) from exc
        wnids = [line.strip() for line in text.splitlines() if line.strip()]

    if len(wnids) != 1000:
        raise RuntimeError(
            f"WNID list has length {len(wnids)} but IN-1k requires 1000. "
            "Source file is corrupt; re-run the prewarm sbatch or replace "
            f"{sidecar} with a valid 1000-line list."
        )
    return {wnid: idx for idx, wnid in enumerate(wnids)}


def _resolve_tarball_extract_dir(cache_dir: str | None) -> str | None:
    """Return the `extract_dir` recorded in the prewarm sentinel, or None.

    The sentinel `<cache_dir>/.prewarm_complete.json` (schema_version >= 2)
    contains the extracted directory path under each tarball entry. This
    helper lets eval invocations omit `--imagenet_<v>_tarball_dir` when the
    prewarm has already run for that dataset.
    """
    if not cache_dir:
        return None
    sentinel = Path(cache_dir).parent / ".prewarm_complete.json"
    if not sentinel.is_file():
        return None
    try:
        manifest = json.loads(sentinel.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for entry in manifest.get("entries", []):
        if isinstance(entry, dict) and entry.get("kind") == "tarball":
            extract_dir = entry.get("extract_dir")
            if extract_dir and Path(extract_dir).is_dir():
                return str(extract_dir)
    return None


def _load_imagenet_variant(
    args,
    *,
    cache_dir: str | None,
    hf_dataset_id: str | None,
    hf_split: str | None,
    tarball_extract_dir: str | None,
    hf_revision: str | None = None,
    is_imagenet_o: bool = False,
) -> Any:
    """Locate a sketch/R/A/O dataset, preferring HF mirror, falling back to tarball.

    For IN-O (`is_imagenet_o=True`) the tarball path uses
    `_load_imagenet_o_tarball_dataset`, which keeps every image with a dummy
    label (the OOD-detection metric ignores labels). For IN-R/A the standard
    WNID-mapped loader is used.
    """
    if hf_dataset_id and cache_dir:
        try:
            return load_dataset_split(
                hf_dataset_id,
                split=hf_split or "train",
                revision=hf_revision,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            logger.warning("Falling back to tarball after HF load failed: %s", exc)
    # If the explicit tarball dir wasn't supplied, fall back to the prewarm
    # sentinel — `download_datasets.sbatch` records `extract_dir` for every
    # tarball entry it prepares.
    if not tarball_extract_dir:
        tarball_extract_dir = _resolve_tarball_extract_dir(cache_dir)
    if tarball_extract_dir:
        loader = (
            _load_imagenet_o_tarball_dataset
            if is_imagenet_o
            else _load_imagenet_tarball_dataset
        )
        return loader(Path(tarball_extract_dir))
    raise RuntimeError(
        "No dataset source provided. Either run "
        "`sbatch scripts/slurm/download_datasets.sbatch` so the prewarm "
        "sentinel records `extract_dir`, or pass --<dataset>_tarball_dir "
        "directly, or pass --<dataset>_dataset and --<dataset>_cache_dir for "
        "an HF mirror."
    )


def _run_imagenet_shifted_variant(
    args,
    feature_extractor,
    feature_dim,
    *,
    variant: str,
    dataset,
    wandb_run=None,
) -> tuple[nn.Module, dict[str, float | int | None]]:
    """Shared logic for IN-Sketch / IN-R / IN-A: probe trained on IN-100, eval on shifted."""
    probe, train_metrics, train_ds, _val_ds, eval_transform = build_clean_probe(
        args, feature_extractor, feature_dim, wandb_run
    )
    in1k_to_100 = _imagenet1k_to_imagenet100(args, train_ds)
    eval_loader = create_loader(
        dataset,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_map=in1k_to_100,
        skip_unmapped=True,
        max_samples=args.max_eval_samples,
    )
    metrics = train_metrics | evaluate_loader(
        feature_extractor,
        probe,
        eval_loader,
        device=args.device,
        prefix=f"{variant}/val",
    )
    train_features, train_labels = collect_train_features(
        feature_extractor,
        train_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_samples=args.max_train_samples,
    )
    preds = collect_predictions(
        feature_extractor,
        probe,
        eval_loader,
        device=args.device,
        keep_features=True,
    )
    metrics.update(
        compute_extra_metrics(
            logits=preds["logits"],
            labels=preds["labels"],
            prefix=f"{variant}/val",
            train_features=train_features,
            train_labels=train_labels,
            eval_features=preds.get("features"),
            knn_k=args.knn_k,
        )
    )
    clean_acc = float(train_metrics.get("monitor/best_acc") or 0.0)
    shifted_acc = float(metrics.get(f"{variant}/val/acc") or 0.0)
    if clean_acc > 0 and shifted_acc > 0:
        metrics[f"{variant}/clean_vs_shifted_gap"] = clean_vs_shifted_gap(clean_acc, shifted_acc)
    metrics[f"{variant}/remapped_num_classes"] = len(in1k_to_100)
    return probe, metrics


def run_imagenet_sketch(args, feature_extractor, feature_dim, wandb_run=None):
    cache_dir = args.imagenet_sketch_cache_dir or args.dataset_cache_dir or default_cache_dir()
    dataset = _load_imagenet_variant(
        args,
        cache_dir=cache_dir,
        hf_dataset_id=args.imagenet_sketch_dataset,
        hf_split=args.imagenet_sketch_split,
        hf_revision=args.imagenet_sketch_revision,
        tarball_extract_dir=args.imagenet_sketch_tarball_dir,
    )
    return _run_imagenet_shifted_variant(
        args, feature_extractor, feature_dim,
        variant="imagenet_sketch", dataset=dataset, wandb_run=wandb_run,
    )


def run_imagenet_r(args, feature_extractor, feature_dim, wandb_run=None):
    cache_dir = args.imagenet_r_cache_dir or args.dataset_cache_dir or default_cache_dir()
    dataset = _load_imagenet_variant(
        args,
        cache_dir=cache_dir,
        hf_dataset_id=args.imagenet_r_dataset,
        hf_split=args.imagenet_r_split,
        tarball_extract_dir=args.imagenet_r_tarball_dir,
    )
    return _run_imagenet_shifted_variant(
        args, feature_extractor, feature_dim,
        variant="imagenet_r", dataset=dataset, wandb_run=wandb_run,
    )


def run_imagenet_a(args, feature_extractor, feature_dim, wandb_run=None):
    cache_dir = args.imagenet_a_cache_dir or args.dataset_cache_dir or default_cache_dir()
    dataset = _load_imagenet_variant(
        args,
        cache_dir=cache_dir,
        hf_dataset_id=args.imagenet_a_dataset,
        hf_split=args.imagenet_a_split,
        tarball_extract_dir=args.imagenet_a_tarball_dir,
    )
    return _run_imagenet_shifted_variant(
        args, feature_extractor, feature_dim,
        variant="imagenet_a", dataset=dataset, wandb_run=wandb_run,
    )


def run_imagenet_o(args, feature_extractor, feature_dim, wandb_run=None):
    """OOD detection on ImageNet-O.

    The probe is trained on IN-100 clean. ID = IN-100 val (already used for
    monitoring). OOD = IN-O. AUROC uses max-softmax as the score function.
    No per-sample accuracy is reported (IN-O classes are by definition not
    in IN-1k or IN-100).
    """
    probe, train_metrics, train_ds, val_ds, eval_transform = build_clean_probe(
        args, feature_extractor, feature_dim, wandb_run
    )
    cache_dir = args.imagenet_o_cache_dir or args.dataset_cache_dir or default_cache_dir()
    ood_dataset = _load_imagenet_variant(
        args,
        cache_dir=cache_dir,
        hf_dataset_id=args.imagenet_o_dataset,
        hf_split=args.imagenet_o_split,
        tarball_extract_dir=args.imagenet_o_tarball_dir,
        is_imagenet_o=True,
    )
    # ID samples: IN-100 val.
    id_loader = create_loader(
        val_ds,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_eval_samples,
    )
    # OOD samples: IN-O. Labels are arbitrary (ignored); we only need probs.
    # We use a permissive label_map so EvalDataset doesn't drop rows.
    ood_loader = create_loader(
        ood_dataset,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # label_map=None so labels pass through; we won't use them
        max_samples=args.max_eval_samples,
    )
    id_preds = collect_predictions(feature_extractor, probe, id_loader, device=args.device)
    ood_preds = collect_predictions(feature_extractor, probe, ood_loader, device=args.device)
    auroc = auroc_max_softmax(
        id_preds["logits"].softmax(dim=-1),
        ood_preds["logits"].softmax(dim=-1),
    )
    metrics = train_metrics | {
        "imagenet_o/auroc_max_softmax": auroc,
        "imagenet_o/n_id": int(id_preds["labels"].numel()),
        "imagenet_o/n_ood": int(ood_preds["labels"].numel()),
    }
    return probe, metrics


# ---------------------------------------------------------------------------
# CelebA (Blond_Hair × Male) groups
# ---------------------------------------------------------------------------
def _binary_attr(value: Any, attr_name: str = "attr") -> int:
    """Normalize common binary attribute codings to {0, 1}.

    CelebA mirrors typically encode attributes as {-1, 1}, while some loaders
    expose {0, 1}. We accept both and reject anything else to avoid silently
    training probes with invalid class indices.
    """
    raw = int(_to_scalar(value))
    if raw in (-1, 0):
        return 0
    if raw == 1:
        return 1
    raise ValueError(
        f"Expected binary coding for {attr_name!r} in {{-1,0,1}}, got {raw}."
    )


def _celeba_attr_index(dataset, attr_name: str) -> int:
    """Locate an attribute column / one-hot index in the HF CelebA-attrs schema."""
    columns = list(getattr(dataset, "column_names", []) or [])
    if attr_name in columns:
        return -1  # signal: dedicated column
    if "attributes" in columns and isinstance(dataset[0]["attributes"], (list, tuple)):
        # Some mirrors serialize as a list aligned with `attribute_names`.
        names = dataset.info.features["attributes"].feature.names if hasattr(dataset.info.features["attributes"], "feature") else None
        if names and attr_name in names:
            return list(names).index(attr_name)
    raise KeyError(
        f"CelebA dataset is missing attribute {attr_name!r}. Available columns: {columns}."
    )


def _celeba_group_fn(target_attr: str, spurious_attr: str):
    """Return a group function: (raw_label, spurious_attr) -> 4-way group id.

    Group encoding: 2 * target_attr + spurious_attr, so groups 0..3 correspond
    to (target=0, spurious=0), (target=0, spurious=1), (target=1, spurious=0),
    (target=1, spurious=1).
    """
    def fn(item: Mapping[str, Any], raw_label: int, mapped_label: int) -> int:
        target = _binary_attr(item.get(target_attr, 0), target_attr)
        spurious = _binary_attr(item.get(spurious_attr, 0), spurious_attr)
        return 2 * target + spurious
    return fn


def run_celeba_groups(args, feature_extractor, feature_dim, wandb_run=None):
    """Train a 2-class probe on CelebA Blond_Hair, evaluate worst-group on (Blond × Male)."""
    train_transform, eval_transform = build_transforms(args.img_size)
    cache_dir = args.celeba_cache_dir or args.dataset_cache_dir or default_cache_dir()
    train_ds = load_dataset_split(
        args.celeba_dataset, split=args.celeba_train_split, cache_dir=cache_dir,
    )
    val_ds = load_dataset_split(
        args.celeba_dataset, split=args.celeba_val_split, cache_dir=cache_dir,
    )
    test_ds = load_dataset_split(
        args.celeba_dataset, split=args.celeba_test_split, cache_dir=cache_dir,
    )

    target_attr = args.celeba_target_attr
    spurious_attr = args.celeba_spurious_attr
    group_fn = _celeba_group_fn(target_attr, spurious_attr)

    # Re-label: target attribute becomes the classification target.
    # Build EvalDataset directly with a wrapper that overrides label.
    class _CelebATargetDataset:
        """Adapter that maps `label` to the target attribute on the fly."""
        def __init__(self, base):
            self.base = base
            self.column_names = list(getattr(base, "column_names", []) or [])
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            return {
                "image": item["image"],
                "label": _binary_attr(item.get(target_attr, 0), target_attr),
                **{k: v for k, v in item.items() if k != "label"},
            }

    train_loader = create_loader(
        _CelebATargetDataset(train_ds), train_transform,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, max_samples=args.max_train_samples,
    )
    val_loader = create_loader(
        _CelebATargetDataset(val_ds), eval_transform,
        batch_size=args.batch_size, num_workers=args.num_workers,
        group_fn=group_fn, max_samples=args.max_eval_samples,
    )
    test_loader = create_loader(
        _CelebATargetDataset(test_ds), eval_transform,
        batch_size=args.batch_size, num_workers=args.num_workers,
        group_fn=group_fn, max_samples=args.max_eval_samples,
    )

    probe = LinearProbe(feature_dim, 2, normalize=not args.no_normalize)
    train_metrics = train_probe(
        feature_extractor, probe, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, device=args.device, wandb_run=wandb_run,
    )

    group_names = {
        0: f"{target_attr}=0,{spurious_attr}=0",
        1: f"{target_attr}=0,{spurious_attr}=1",
        2: f"{target_attr}=1,{spurious_attr}=0",
        3: f"{target_attr}=1,{spurious_attr}=1",
    }
    metrics = train_metrics
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        metrics.update(
            evaluate_loader(
                feature_extractor, probe, loader,
                device=args.device, prefix=f"celeba/{split_name}",
                group_names=group_names,
            )
        )
        preds = collect_predictions(feature_extractor, probe, loader, device=args.device)
        metrics.update(
            compute_extra_metrics(
                logits=preds["logits"], labels=preds["labels"],
                prefix=f"celeba/{split_name}",
            )
        )
    return probe, metrics


# ---------------------------------------------------------------------------
# WILDS Camelyon17
# ---------------------------------------------------------------------------
class _WildsLazyDataset:
    """Duck-typed adapter exposing a WILDS subset to `EvalDataset`/`create_loader`.

    `EvalDataset` only needs `len`, integer indexing returning a dict with
    `"image"` (PIL.Image-or-similar) and `"label"` (int), and optionally a
    `column_names` attribute. We forward straight through to `wilds_subset[i]`
    so each call returns the underlying PIL.Image lazily — no HF
    `Dataset.from_list` round-trip, no `Image(decode=...)` mismatch.
    """

    column_names = ["image", "label"]

    def __init__(self, wilds_subset):
        self._subset = wilds_subset

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x, y, _meta = self._subset[idx]
        return {"image": x, "label": int(y)}


def run_camelyon17(args, feature_extractor, feature_dim, wandb_run=None):
    """WILDS Camelyon17: train probe on labeled train hospital, eval on OOD val + OOD test."""
    try:
        from wilds import get_dataset as wilds_get_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The `wilds` package is required for Camelyon17. Install with "
            "`pip install wilds` in the eval conda env."
        ) from exc

    cache_dir = args.camelyon17_cache_dir or args.dataset_cache_dir or default_cache_dir()
    full = wilds_get_dataset(dataset="camelyon17", root_dir=cache_dir, download=False)
    train_subset = full.get_subset("train")
    val_subset = full.get_subset("val")  # OOD validation hospital (Koh et al. 2021)
    test_subset = full.get_subset("test")  # OOD test hospital

    train_transform, eval_transform = build_transforms(args.img_size)

    train_loader = create_loader(
        _WildsLazyDataset(train_subset), train_transform,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, max_samples=args.max_train_samples,
    )
    val_loader = create_loader(
        _WildsLazyDataset(val_subset), eval_transform,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_samples=args.max_eval_samples,
    )
    test_loader = create_loader(
        _WildsLazyDataset(test_subset), eval_transform,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_samples=args.max_eval_samples,
    )
    probe = LinearProbe(feature_dim, 2, normalize=not args.no_normalize)
    train_metrics = train_probe(
        feature_extractor, probe, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, device=args.device, wandb_run=wandb_run,
    )
    metrics = train_metrics
    for split_name, loader in (("ood_val", val_loader), ("ood_test", test_loader)):
        metrics.update(
            evaluate_loader(
                feature_extractor, probe, loader,
                device=args.device, prefix=f"camelyon17/{split_name}",
            )
        )
        preds = collect_predictions(feature_extractor, probe, loader, device=args.device)
        metrics.update(
            compute_extra_metrics(
                logits=preds["logits"], labels=preds["labels"],
                prefix=f"camelyon17/{split_name}",
            )
        )
    return probe, metrics


def has_wandb_credentials() -> bool:
    if os.getenv("WANDB_API_KEY"):
        return True
    key_file = os.getenv("WANDB_API_KEY_FILE")
    if key_file and Path(key_file).is_file():
        return True
    if (Path.home() / ".config" / "wandb" / "settings").is_file():
        return True
    if (Path.home() / ".netrc").is_file():
        return True
    return False


def init_wandb(args, job_type: str):
    if args.no_wandb or args.wandb_mode == "disabled":
        return None
    if args.wandb_mode == "online" and not has_wandb_credentials():
        raise RuntimeError(
            "W&B credentials not found. Export WANDB_API_KEY, run 'wandb login', "
            "or set --wandb_mode offline."
        )
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=job_type,
        mode=args.wandb_mode,
        config={k: _jsonable(v) for k, v in vars(args).items()},
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "imagenet100ctrl",
            "imagenet100c",
            "waterbirds",
            "imagenet_sketch",
            "imagenet_r",
            "imagenet_a",
            "imagenet_o",
            "celeba",
            "camelyon17",
        ],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrain_run_id", default=os.environ.get("PRETRAIN_RUN_ID", "unknown"))
    parser.add_argument("--method", default=os.environ.get("METHOD", "unknown"))
    parser.add_argument("--weights_only", action="store_true")
    parser.add_argument("--backbone", default="vit_small_patch8_224")
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[-1, -2])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--no_save_probe", action="store_true")
    parser.add_argument(
        "--knn_k",
        type=int,
        default=20,
        help="k for the kNN probe added in v2; 20 matches DINO/i-BOT.",
    )
    parser.add_argument("--dataset_cache_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_imagenetc_samples", type=int, default=None)
    parser.add_argument("--coherent_labels", default="0-29")
    parser.add_argument("--imagenet100_dataset", default=IMAGENET100_DEFAULT)
    parser.add_argument("--imagenet100_cache_dir", default=None)
    parser.add_argument("--imagenet100_dataset_name", default="")
    parser.add_argument("--imagenet100_revision", default=PARQUET_REVISION)
    parser.add_argument("--imagenet100_train_split", default="train")
    parser.add_argument("--imagenet100_val_split", default="validation")
    parser.add_argument("--imagenetc_dataset", default=IMAGENETC_DEFAULT)
    parser.add_argument("--imagenetc_cache_dir", default=None)
    parser.add_argument("--imagenetc_revision", default=IMAGENETC_REVISION)
    parser.add_argument("--imagenetc_corruptions", default=",".join(IMAGENETC_CORRUPTIONS))
    parser.add_argument("--imagenetc_severities", default="1,2,3,4,5")
    parser.add_argument("--waterbirds_dataset", default=WATERBIRDS_DEFAULT)
    parser.add_argument("--waterbirds_cache_dir", default=None)
    parser.add_argument("--waterbirds_revision", default=None)
    parser.add_argument("--waterbirds_train_split", default="train")
    parser.add_argument("--waterbirds_val_split", default="validation")
    parser.add_argument("--waterbirds_test_split", default="test")
    # ----- Phase D dataset flags -----
    parser.add_argument(
        "--imagenet_sketch_dataset",
        default="Huanyiiiii/Imagenet_Sketch",
    )
    parser.add_argument("--imagenet_sketch_cache_dir", default=None)
    parser.add_argument("--imagenet_sketch_split", default="train")
    parser.add_argument("--imagenet_sketch_revision", default=PARQUET_REVISION)
    parser.add_argument("--imagenet_sketch_tarball_dir", default=None)
    parser.add_argument("--imagenet_r_dataset", default=None)
    parser.add_argument("--imagenet_r_cache_dir", default=None)
    parser.add_argument("--imagenet_r_split", default=None)
    parser.add_argument("--imagenet_r_tarball_dir", default=None)
    parser.add_argument("--imagenet_a_dataset", default=None)
    parser.add_argument("--imagenet_a_cache_dir", default=None)
    parser.add_argument("--imagenet_a_split", default=None)
    parser.add_argument("--imagenet_a_tarball_dir", default=None)
    parser.add_argument("--imagenet_o_dataset", default=None)
    parser.add_argument("--imagenet_o_cache_dir", default=None)
    parser.add_argument("--imagenet_o_split", default=None)
    parser.add_argument("--imagenet_o_tarball_dir", default=None)
    parser.add_argument("--celeba_dataset", default="tpremoli/CelebA-attrs")
    parser.add_argument("--celeba_cache_dir", default=None)
    parser.add_argument("--celeba_train_split", default="train")
    parser.add_argument("--celeba_val_split", default="validation")
    parser.add_argument("--celeba_test_split", default="test")
    parser.add_argument(
        "--celeba_target_attr",
        default="Blond_Hair",
        help="Target attribute for the 2-class probe (default Sagawa et al. 2020).",
    )
    parser.add_argument(
        "--celeba_spurious_attr",
        default="Male",
        help="Spurious attribute partitioning the 4 groups (Blond_Hair × Male).",
    )
    parser.add_argument("--camelyon17_cache_dir", default=None)
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT_NAME", "ssl-geo-dro"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    parser.add_argument("--no_wandb", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_group is None:
        args.wandb_group = args.pretrain_run_id
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.pretrain_run_id}-{args.mode}"

    write_json(output_dir / "args.json", {k: _jsonable(v) for k, v in vars(args).items()})
    job_type = {
        "imagenet100ctrl": "linear_eval",
        "imagenet100c": "corruption_eval",
        "waterbirds": "downstream_worst_group_eval",
        "imagenet_sketch": "covariate_shift_eval",
        "imagenet_r": "covariate_shift_eval",
        "imagenet_a": "covariate_shift_eval",
        "imagenet_o": "ood_detection_eval",
        "celeba": "downstream_worst_group_eval",
        "camelyon17": "domain_shift_eval",
    }[args.mode]
    wandb_run = init_wandb(args, job_type=job_type)
    feature_extractor, feature_dim = build_feature_extractor(args)

    dispatch = {
        "imagenet100ctrl": run_imagenet100ctrl,
        "imagenet100c": run_imagenetc,
        "waterbirds": run_waterbirds,
        "imagenet_sketch": run_imagenet_sketch,
        "imagenet_r": run_imagenet_r,
        "imagenet_a": run_imagenet_a,
        "imagenet_o": run_imagenet_o,
        "celeba": run_celeba_groups,
        "camelyon17": run_camelyon17,
    }
    probe, metrics = dispatch[args.mode](args, feature_extractor, feature_dim, wandb_run)

    payload = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "mode": args.mode,
        "method": args.method,
        "pretrain_run_id": args.pretrain_run_id,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
    }
    write_json(output_dir / "metrics.json", payload)
    if not args.no_save_probe:
        torch.save(probe.state_dict(), output_dir / "probe.pt")
    if wandb_run is not None:
        wandb_run.log({f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        wandb_run.save(str(output_dir / "metrics.json"))
        wandb_run.finish()
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
