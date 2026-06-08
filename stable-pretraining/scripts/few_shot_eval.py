#!/usr/bin/env python3
"""Few-shot evaluation script for pretrained SSL models.

This script implements the few-shot evaluation protocol from the LeJEPA paper:
- Sample K examples per class (K=1 or K=10)
- Train linear probe on sampled data
- Evaluate on full test set
- Report mean and std over multiple runs

Usage:
    # 1-shot evaluation
    python scripts/few_shot_eval.py \
        --checkpoint path/to/checkpoint.ckpt \
        --dataset frgfm/imagenette \
        --num_classes 10 \
        --shots 1 \
        --num_runs 5

    # 10-shot evaluation
    python scripts/few_shot_eval.py \
        --checkpoint path/to/checkpoint.ckpt \
        --dataset frgfm/imagenette \
        --num_classes 10 \
        --shots 10 \
        --num_runs 5

    # Evaluate on multiple downstream datasets
    python scripts/few_shot_eval.py \
        --checkpoint path/to/checkpoint.ckpt \
        --datasets cifar10 cifar100 food101 \
        --shots 1 10 \
        --num_runs 5

Reference:
    Balestriero & LeCun (2025). LeJEPA: Provable and Scalable Self-Supervised
    Learning Without the Heuristics. arXiv:2511.08544
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dataset configurations for downstream evaluation
DATASET_CONFIGS = {
    "imagenette": {
        "path": "frgfm/imagenette",
        "name": "160px",
        "num_classes": 10,
        "train_split": "train",
        "val_split": "validation",
    },
    "cifar10": {
        "path": "cifar10",
        "name": None,
        "num_classes": 10,
        "train_split": "train",
        "val_split": "test",
    },
    "cifar100": {
        "path": "cifar100",
        "name": None,
        "num_classes": 100,
        "train_split": "train",
        "val_split": "test",
    },
    "food101": {
        "path": "food101",
        "name": None,
        "num_classes": 101,
        "train_split": "train",
        "val_split": "validation",
    },
    "flowers102": {
        "path": "nelorth/oxford-flowers",
        "name": None,
        "num_classes": 102,
        "train_split": "train",
        "val_split": "test",
    },
    "pets": {
        "path": "timm/oxford-iiit-pet",
        "name": None,
        "num_classes": 37,
        "train_split": "train",
        "val_split": "test",
    },
    "dtd": {
        "path": "tanganke/dtd",
        "name": None,
        "num_classes": 47,
        "train_split": "train",
        "val_split": "test",
    },
    "aircraft": {
        "path": "Multimodal-Fatima/FGVC_Aircraft_train",
        "name": None,
        "num_classes": 100,
        "train_split": "train",
        "val_split": "test",
    },
}


class FeatureExtractor(nn.Module):
    """Extract features from specific layers of a backbone."""

    def __init__(self, backbone: nn.Module, layers: list[int] = [-1, -2], model_type: str = "vit"):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.model_type = model_type
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        if self.model_type == "vit":
            if hasattr(self.backbone, "blocks"):
                blocks = self.backbone.blocks
            elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                blocks = self.backbone.encoder.layer
            else:
                raise ValueError("Cannot find transformer blocks in backbone")

            for idx in self.layers:
                actual_idx = idx if idx >= 0 else len(blocks) + idx
                blocks[actual_idx].register_forward_hook(
                    lambda m, inp, out, idx=idx: self._save_feature(idx, out)
                )
        else:
            self.backbone.register_forward_hook(self._save_cnn_feature)

    def _save_feature(self, idx: int, output):
        if isinstance(output, tuple):
            output = output[0]
        self._features[idx] = output

    def _save_cnn_feature(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        for idx in self.layers:
            self._features[idx] = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features.clear()
        _ = self.backbone(x)

        features = []
        for idx in self.layers:
            feat = self._features[idx]
            if self.model_type == "vit":
                if feat.ndim == 3:
                    feat = feat[:, 0, :]
                elif hasattr(feat, "last_hidden_state"):
                    feat = feat.last_hidden_state[:, 0, :]
            else:
                if feat.ndim == 4:
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            features.append(feat)

        return torch.cat(features, dim=-1)


class LinearProbe(nn.Module):
    """Linear probe classifier with LayerNorm."""

    def __init__(self, in_features: int, num_classes: int, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        if normalize:
            self.norm = nn.LayerNorm(in_features)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = self.norm(x)
        return self.classifier(x)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load backbone weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_state_dict[k[9:]] = v
        elif not k.startswith("projector.") and not k.startswith("classifier."):
            backbone_state_dict[k] = v

    return backbone_state_dict


def sample_few_shot_indices(
    dataset,
    num_classes: int,
    shots: int,
    seed: int = 0,
) -> list[int]:
    """Sample K examples per class for few-shot learning.

    Args:
        dataset: Dataset to sample from.
        num_classes: Number of classes.
        shots: Number of examples per class.
        seed: Random seed for reproducibility.

    Returns:
        List of indices for sampled examples.
    """
    rng = np.random.RandomState(seed)

    # Group indices by class
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        label = dataset[idx]["label"]
        class_indices[label].append(idx)

    # Sample K examples per class
    sampled_indices = []
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        if len(indices) < shots:
            logger.warning(f"Class {class_id} has only {len(indices)} examples, using all")
            sampled_indices.extend(indices)
        else:
            sampled = rng.choice(indices, shots, replace=False)
            sampled_indices.extend(sampled.tolist())

    return sampled_indices


def create_few_shot_dataloaders(
    dataset_path: str,
    dataset_name: str = None,
    train_split: str = "train",
    val_split: str = "validation",
    num_classes: int = 10,
    shots: int = 1,
    seed: int = 0,
    batch_size: int = 256,
    num_workers: int = 8,
    img_size: int = 224,
):
    """Create few-shot train loader and full validation loader."""
    from datasets import load_dataset
    from torchvision.transforms import v2

    # Load dataset
    if dataset_name:
        dataset = load_dataset(dataset_path, dataset_name)
    else:
        dataset = load_dataset(dataset_path)

    # Define transforms
    train_transform = v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.Resize(int(img_size * 256 / 224)),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            img = item["image"].convert("RGB")
            label = item["label"]
            return self.transform(img), label

    # Sample few-shot indices
    train_hf = dataset[train_split]
    few_shot_indices = sample_few_shot_indices(train_hf, num_classes, shots, seed)

    # Create datasets
    train_ds = TransformedDataset(train_hf, train_transform)
    train_subset = Subset(train_ds, few_shot_indices)
    val_ds = TransformedDataset(dataset[val_split], val_transform)

    # Create dataloaders
    # For few-shot, we may have very few samples, so adjust batch size
    actual_batch_size = min(batch_size, len(few_shot_indices))
    train_loader = DataLoader(
        train_subset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(few_shot_indices) > actual_batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_few_shot_probe(
    feature_extractor: nn.Module,
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: str = "cuda",
):
    """Train a linear probe on few-shot data."""
    feature_extractor = feature_extractor.to(device)
    probe = probe.to(device)
    feature_extractor.eval()

    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    val_acc_metric = MulticlassAccuracy(num_classes=probe.classifier.out_features).to(device)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        probe.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                with autocast(device):
                    features = feature_extractor(images)

            with autocast(device):
                logits = probe(features)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation (every 10 epochs for speed)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            probe.eval()
            val_acc_metric.reset()

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with autocast(device):
                        features = feature_extractor(images)
                        logits = probe(features)

                    val_acc_metric.update(logits.argmax(dim=-1), labels)

            current_val_acc = val_acc_metric.compute().item()
            best_val_acc = max(best_val_acc, current_val_acc)

    return best_val_acc


def evaluate_few_shot(
    feature_extractor: nn.Module,
    feature_dim: int,
    dataset_config: dict,
    shots: int,
    num_runs: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    img_size: int = 224,
    device: str = "cuda",
):
    """Run few-shot evaluation with multiple random seeds.

    Returns:
        Tuple of (mean_accuracy, std_accuracy).
    """
    accuracies = []

    for run in range(num_runs):
        logger.info(f"  Run {run + 1}/{num_runs}")

        # Create dataloaders with different seed
        train_loader, val_loader = create_few_shot_dataloaders(
            dataset_path=dataset_config["path"],
            dataset_name=dataset_config["name"],
            train_split=dataset_config["train_split"],
            val_split=dataset_config["val_split"],
            num_classes=dataset_config["num_classes"],
            shots=shots,
            seed=run,
            batch_size=batch_size,
            img_size=img_size,
        )

        # Create fresh probe for each run
        probe = LinearProbe(
            in_features=feature_dim,
            num_classes=dataset_config["num_classes"],
            normalize=True,
        )

        # Train and evaluate
        acc = train_few_shot_probe(
            feature_extractor=feature_extractor,
            probe=probe,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )
        accuracies.append(acc)
        logger.info(f"    Accuracy: {acc:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description="Few-shot evaluation for SSL models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--datasets", type=str, nargs="+", default=["imagenette"],
                        choices=list(DATASET_CONFIGS.keys()), help="Datasets to evaluate on")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 10], help="Number of shots (examples per class)")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs per evaluation")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for probe training")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=128, help="Image size")
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[-1, -2], help="Layers to extract features from")
    parser.add_argument("--backbone", type=str, default="vit_small_patch8_224", help="Backbone architecture")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Create backbone
    import timm
    backbone = timm.create_model(
        args.backbone,
        pretrained=False,
        num_classes=0,
        img_size=args.img_size,
    )

    # Load pretrained weights
    state_dict = load_checkpoint(args.checkpoint)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Create feature extractor
    model_type = "vit" if "vit" in args.backbone.lower() else "cnn"
    feature_extractor = FeatureExtractor(
        backbone,
        layers=args.feature_layers,
        model_type=model_type,
    )
    feature_extractor.eval()

    # Determine feature dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
        feature_dim = feature_extractor(dummy_input).shape[-1]
    logger.info(f"Feature dimension: {feature_dim}")

    # Results table
    results = {}

    # Evaluate on each dataset and shot count
    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]
        results[dataset_name] = {}

        for shots in args.shots:
            logger.info(f"Evaluating {dataset_name} with {shots}-shot...")

            mean_acc, std_acc = evaluate_few_shot(
                feature_extractor=feature_extractor,
                feature_dim=feature_dim,
                dataset_config=config,
                shots=shots,
                num_runs=args.num_runs,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                img_size=args.img_size,
                device=args.device,
            )

            results[dataset_name][f"{shots}-shot"] = {
                "mean": mean_acc,
                "std": std_acc,
            }

            logger.info(f"  {shots}-shot: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Few-Shot Evaluation Results")
    print("=" * 60)
    for dataset_name, shot_results in results.items():
        print(f"\n{dataset_name}:")
        for shot_name, metrics in shot_results.items():
            print(f"  {shot_name}: {metrics['mean']*100:.2f}% +/- {metrics['std']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
