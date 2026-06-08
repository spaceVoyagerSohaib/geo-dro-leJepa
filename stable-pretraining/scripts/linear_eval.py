#!/usr/bin/env python3
"""Linear probe evaluation script for pretrained SSL models.

This script implements the linear probe evaluation protocol from the LeJEPA paper:
- Feature extraction: CLS token from last 2 layers (concatenated)
- Normalization: LayerNorm on concatenated features
- Optimizer: AdamW with weight_decay=1e-6
- LR schedule: Linear warmup + cosine annealing
- Frozen backbone evaluation

Usage:
    python scripts/linear_eval.py \
        --checkpoint path/to/checkpoint.ckpt \
        --dataset frgfm/imagenette \
        --num_classes 10 \
        --epochs 100 \
        --batch_size 256

    # With custom feature extraction
    python scripts/linear_eval.py \
        --checkpoint path/to/checkpoint.ckpt \
        --dataset ILSVRC/imagenet-1k \
        --num_classes 1000 \
        --feature_layers -1 -2 \
        --epochs 90

Reference:
    Balestriero & LeCun (2025). LeJEPA: Provable and Scalable Self-Supervised
    Learning Without the Heuristics. arXiv:2511.08544
"""

import argparse
import logging
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def has_wandb_credentials() -> bool:
    """Check for W&B credentials from env or config files."""
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


def init_wandb(args, is_rank_zero: bool = True):
    """Initialize W&B run for linear evaluation (rank 0 only)."""
    if not is_rank_zero:
        return None
    if args.no_wandb or args.wandb_mode == "disabled":
        logger.info("W&B logging disabled.")
        return None

    if args.wandb_mode == "online" and not has_wandb_credentials():
        raise RuntimeError(
            "W&B credentials not found. Export WANDB_API_KEY, run 'wandb login', "
            "or set --wandb_mode offline."
        )

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config=vars(args),
    )
    return run


def get_dist_info():
    """Return (rank, world_size, local_rank) for torchrun or Slurm."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        return rank, world_size, local_rank
    return 0, 1, 0


def init_distributed(backend: str, rank: int, world_size: int, timeout_minutes: int = 30):
    """Initialize torch.distributed if not already initialized."""
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=timeout_minutes),
        )


class FeatureExtractor(nn.Module):
    """Extract features from specific layers of a backbone.

    For ViT models, extracts CLS token from specified layers.
    For CNN models, uses global average pooling.

    Args:
        backbone: The pretrained backbone model.
        layers: List of layer indices to extract features from (e.g., [-1, -2]).
        model_type: Type of model ('vit' or 'cnn').
    """

    def __init__(self, backbone: nn.Module, layers: list[int] = [-1, -2], model_type: str = "vit"):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.model_type = model_type

        # Register hooks to capture intermediate features
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate layer outputs."""
        if self.model_type == "vit":
            # For ViT, hook into transformer blocks
            if hasattr(self.backbone, "blocks"):
                # timm ViT
                blocks = self.backbone.blocks
            elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                # HuggingFace ViT
                blocks = self.backbone.encoder.layer
            else:
                raise ValueError("Cannot find transformer blocks in backbone")

            for idx in self.layers:
                actual_idx = idx if idx >= 0 else len(blocks) + idx
                blocks[actual_idx].register_forward_hook(
                    lambda m, inp, out, idx=idx: self._save_feature(idx, out)
                )
        else:
            # For CNNs, fall back to capturing backbone output
            self.backbone.register_forward_hook(self._save_cnn_feature)

    def _save_feature(self, idx: int, output):
        """Hook callback to save layer output."""
        if isinstance(output, tuple):
            output = output[0]
        self._features[idx] = output

    def _save_cnn_feature(self, module, inputs, output):
        """Hook callback to save CNN backbone output for all requested layers."""
        if isinstance(output, tuple):
            output = output[0]
        for idx in self.layers:
            self._features[idx] = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate features from specified layers.

        Args:
            x: Input images [N, C, H, W].

        Returns:
            Concatenated features [N, D] where D = sum of feature dims.
        """
        self._features.clear()

        # Forward pass through backbone
        _ = self.backbone(x)

        # Collect features from hooked layers
        features = []
        for idx in self.layers:
            feat = self._features[idx]
            if self.model_type == "vit":
                # Extract CLS token (first token)
                if feat.ndim == 3:
                    feat = feat[:, 0, :]  # [N, D]
                elif hasattr(feat, "last_hidden_state"):
                    feat = feat.last_hidden_state[:, 0, :]
            else:
                # CNN: global average pool
                if feat.ndim == 4:
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            features.append(feat)

        # Concatenate features from all layers
        return torch.cat(features, dim=-1)


class LinearProbe(nn.Module):
    """Linear probe classifier with optional normalization.

    Args:
        in_features: Input feature dimension.
        num_classes: Number of output classes.
        normalize: Whether to apply LayerNorm before classification.
    """

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


def load_checkpoint(checkpoint_path: str, *, weights_only: bool = False) -> dict:
    """Load a pretrained backbone from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        State dict containing pretrained weights.
    """
    if not weights_only:
        logger.info(
            "Loading checkpoint with weights_only=False (trusted source assumed)."
        )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=weights_only
        )
    except TypeError:
        # Older PyTorch versions do not support weights_only
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Extract backbone weights (remove "backbone." prefix if present)
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_state_dict[k[9:]] = v
        elif not k.startswith("projector.") and not k.startswith("classifier."):
            backbone_state_dict[k] = v

    return backbone_state_dict


# Auto-converted parquet branch for datasets with loading scripts
PARQUET_REVISION = "refs/convert/parquet"


def _load_hf_dataset_with_fallback(
    dataset_path: str,
    dataset_name: str = None,
    revision: str = None,
):
    """Load HuggingFace dataset with automatic fallback to parquet branch.
    
    For datasets with loading scripts (deprecated in datasets>=4.0), this function
    automatically falls back to the auto-converted parquet branch.
    
    Args:
        dataset_path: HuggingFace dataset path (e.g., 'ilee0022/ImageNet100').
        dataset_name: Dataset configuration name (e.g., '160px' for imagenette).
        revision: Git revision to load from. If None and loading fails due to
            deprecated scripts, will automatically fall back to 'refs/convert/parquet'.
    
    Returns:
        The loaded HuggingFace dataset.
    """
    from datasets import load_dataset
    
    load_kwargs = {}
    if dataset_name:
        load_kwargs["name"] = dataset_name
    if revision:
        load_kwargs["revision"] = revision
    
    try:
        return load_dataset(dataset_path, **load_kwargs)
    except RuntimeError as e:
        error_msg = str(e).lower()
        # Check if this is the "dataset scripts no longer supported" error
        if "dataset scripts are no longer supported" in error_msg or "no longer supported" in error_msg:
            if revision == PARQUET_REVISION:
                raise  # Already using parquet branch, nothing more to try
            
            logger.warning(
                f"Dataset '{dataset_path}' uses loading scripts which are no longer supported "
                f"in datasets>=4.0. Falling back to auto-converted Parquet files at "
                f"revision='{PARQUET_REVISION}'."
            )
            load_kwargs["revision"] = PARQUET_REVISION
            return load_dataset(dataset_path, **load_kwargs)
        raise  # Re-raise other RuntimeErrors


def create_dataloaders(
    dataset_path: str,
    dataset_name: str = None,
    batch_size: int = 256,
    num_workers: int = 8,
    img_size: int = 224,
    revision: str = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
):
    """Create train and validation dataloaders.

    Args:
        dataset_path: HuggingFace dataset path (e.g., 'ilee0022/ImageNet100').
        dataset_name: Dataset configuration name (e.g., '160px' for imagenette).
        batch_size: Batch size for dataloaders.
        num_workers: Number of workers for data loading.
        img_size: Image size for transforms.
        revision: Git revision to load from. Use 'refs/convert/parquet' to load from
            auto-converted Parquet files (required for datasets>=4.0 with loading scripts).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from torchvision.transforms import v2

    # Load dataset from HuggingFace
    # For datasets with loading scripts (deprecated in datasets>=4.0), use the
    # auto-converted parquet branch: revision='refs/convert/parquet'
    dataset = _load_hf_dataset_with_fallback(
        dataset_path, dataset_name, revision
    )

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

    train_ds = TransformedDataset(dataset["train"], train_transform)
    val_ds = TransformedDataset(
        dataset.get("validation", dataset.get("test")),
        val_transform
    )

    train_sampler = None
    val_sampler = None
    if distributed and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def train_linear_probe(
    feature_extractor: nn.Module,
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    warmup_epochs: int = 10,
    device: str = "cuda",
    wandb_run=None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    train_sampler=None,
    val_sampler=None,
):
    """Train a linear probe on frozen features.

    Args:
        feature_extractor: Feature extraction model (frozen).
        probe: Linear probe to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_epochs: Number of warmup epochs.
        device: Device to train on.

    Returns:
        Dictionary with final metrics.
    """
    is_rank_zero = rank == 0
    feature_extractor = feature_extractor.to(device)
    probe = probe.to(device)
    feature_extractor.eval()

    # Mixed precision (CUDA only)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"
    scaler = GradScaler() if use_amp else None

    # Wrap probe with DDP if needed
    if distributed and world_size > 1:
        if device_type == "cuda":
            probe = torch.nn.parallel.DistributedDataParallel(
                probe,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            probe = torch.nn.parallel.DistributedDataParallel(probe)

    # Optimizer and scheduler
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = epochs * len(train_loader)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    best_val_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        # Training
        probe.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not is_rank_zero)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                with (autocast(device_type) if use_amp else nullcontext()):
                    features = feature_extractor(images)

            with (autocast(device_type) if use_amp else nullcontext()):
                logits = probe(features)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += batch_size

            if is_rank_zero:
                local_acc = train_correct / max(train_total, 1)
                pbar.set_postfix({"loss": loss.item(), "acc": local_acc})

        # Validation
        probe.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with (autocast(device_type) if use_amp else nullcontext()):
                    features = feature_extractor(images)
                    logits = probe(features)
                    loss = F.cross_entropy(logits, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += batch_size

        # Sync metrics across ranks
        train_correct_t = torch.tensor(train_correct, device=device, dtype=torch.float64)
        train_total_t = torch.tensor(train_total, device=device, dtype=torch.float64)
        train_loss_t = torch.tensor(train_loss_sum, device=device, dtype=torch.float64)
        val_correct_t = torch.tensor(val_correct, device=device, dtype=torch.float64)
        val_total_t = torch.tensor(val_total, device=device, dtype=torch.float64)
        val_loss_t = torch.tensor(val_loss_sum, device=device, dtype=torch.float64)

        if distributed and world_size > 1:
            dist.all_reduce(train_correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_total_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_total_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)

        train_loss_mean = (train_loss_t / train_total_t).item()
        train_acc = (train_correct_t / train_total_t).item()
        val_loss_mean = (val_loss_t / val_total_t).item()
        current_val_acc = (val_correct_t / val_total_t).item()
        best_val_acc = max(best_val_acc, current_val_acc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        if is_rank_zero:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss_mean:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss_mean:.4f}, "
                f"Val Acc: {current_val_acc:.4f}, "
                f"Best Val Acc: {best_val_acc:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Epoch Time: {epoch_time:.1f}s"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss_mean,
                        "train/acc": train_acc,
                        "val/loss": val_loss_mean,
                        "val/acc": current_val_acc,
                        "val/best_acc": best_val_acc,
                        "lr": current_lr,
                        "time/epoch_seconds": epoch_time,
                    },
                    step=epoch + 1,
                )

    return {
        "final_val_acc": current_val_acc,
        "best_val_acc": best_val_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Linear probe evaluation for SSL models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument(
        "--weights_only",
        action="store_true",
        help=(
            "Load checkpoint with weights_only=True (safe mode). "
            "Default loads full checkpoint to support Lightning .ckpt files."
        ),
    )
    parser.add_argument("--dataset", type=str, default="frgfm/imagenette", help="HuggingFace dataset path")
    parser.add_argument("--dataset_name", type=str, default="160px", help="Dataset configuration name")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--img_size", type=int, default=128, help="Image size")
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[-1, -2], help="Layers to extract features from")
    parser.add_argument("--backbone", type=str, default="vit_small_patch8_224", help="Backbone architecture (timm model name)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., cuda, cuda:0, cpu, auto)")
    parser.add_argument("--distributed", action="store_true", help="Enable DDP (auto if WORLD_SIZE>1)")
    parser.add_argument(
        "--dist_backend",
        type=str,
        default=None,
        help="Distributed backend (defaults to nccl for CUDA, gloo for CPU)",
    )
    parser.add_argument(
        "--dist_timeout",
        type=int,
        default=30,
        help="DDP init timeout in minutes",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_normalize", action="store_true", help="Disable LayerNorm in probe")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "lejepa-linear-eval"),
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (optional)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=os.environ.get("WANDB_RUN_NAME"),
        help="W&B run name (optional)",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=os.environ.get("WANDB_GROUP"),
        help="W&B run group (optional)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
        help="W&B mode",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="refs/convert/parquet",
        help="Git revision to load from. Use 'refs/convert/parquet' to load from "
             "auto-converted Parquet files (required for datasets>=4.0 with loading scripts)",
    )

    args = parser.parse_args()

    # Random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Distributed setup
    rank, world_size, local_rank = get_dist_info()
    distributed = args.distributed or world_size > 1

    # Resolve device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if distributed and device.startswith("cuda"):
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

    if distributed:
        backend = args.dist_backend or ("nccl" if device.startswith("cuda") else "gloo")
        init_distributed(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout_minutes=args.dist_timeout,
        )

    is_rank_zero = rank == 0

    # Initialize W&B (rank 0 only)
    wandb_run = init_wandb(args, is_rank_zero=is_rank_zero)

    # Create backbone
    import timm
    backbone = timm.create_model(
        args.backbone,
        pretrained=False,
        num_classes=0,  # Remove classifier
        img_size=args.img_size,
    )

    # Load pretrained weights
    state_dict = load_checkpoint(args.checkpoint, weights_only=args.weights_only)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if is_rank_zero:
        if missing:
            logger.warning(f"Missing keys when loading backbone: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading backbone: {unexpected}")

    # Create feature extractor
    model_type = "vit" if "vit" in args.backbone.lower() else "cnn"
    feature_extractor = FeatureExtractor(
        backbone,
        layers=args.feature_layers,
        model_type=model_type,
    )
    feature_extractor.eval()

    # Determine feature dimension by forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
        feature_dim = feature_extractor(dummy_input).shape[-1]
    if is_rank_zero:
        logger.info(f"Feature dimension: {feature_dim}")

    # Create linear probe
    probe = LinearProbe(
        in_features=feature_dim,
        num_classes=args.num_classes,
        normalize=not args.no_normalize,
    )

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        revision=args.revision,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )

    # Train linear probe
    results = train_linear_probe(
        feature_extractor=feature_extractor,
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        device=device,
        wandb_run=wandb_run,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
    )

    if is_rank_zero:
        logger.info(f"Final results: {results}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "final/val_acc": results["final_val_acc"],
                    "final/best_val_acc": results["best_val_acc"],
                }
            )
            wandb_run.finish()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
