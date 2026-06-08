"""Dataset classes for real data sources.

This module provides dataset wrappers and utilities for working with real data sources
including PyTorch datasets, HuggingFace datasets, and dataset subsets.
"""

from pathlib import Path
import os
import time
from collections.abc import Sequence

import lightning as pl
import torch
from loguru import logger as logging
from datasets import config as hf_config
from ..utils import with_hf_retry_ratelimit


class Dataset(torch.utils.data.Dataset):
    """Base dataset class with transform support and PyTorch Lightning integration."""

    def __init__(self, transform=None):
        self.transform = transform
        self._trainer = None

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer

    def process_sample(self, sample, **kwargs):
        for k, v in kwargs.items():
            sample[k] = v
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Subset(Dataset):
    r"""Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)

    @property
    def column_names(self):
        return self.dataset.column_names


class FromTorchDataset(Dataset):
    """Wrapper for PyTorch datasets with custom column naming and transforms.

    Args:
        dataset: PyTorch dataset to wrap
        names: List of names for each element returned by the dataset
        transform: Optional transform to apply to samples
        add_sample_idx: If True, automatically adds 'sample_idx' field to each sample
    """

    def __init__(self, dataset, names, transform=None, add_sample_idx=True):
        super().__init__(transform)
        self.dataset = dataset
        self.names = names
        self.add_sample_idx = add_sample_idx

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = {k: v for k, v in zip(self.names, sample)}
        if self.add_sample_idx:
            sample["sample_idx"] = idx
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)

    @property
    def column_names(self):
        columns = list(self.names)
        if self.add_sample_idx and "sample_idx" not in columns:
            columns.append("sample_idx")
        return columns


class HFDataset(Dataset):
    """Hugging Face dataset wrapper with transform and column manipulation support.
    
    Supports loading from HuggingFace Hub datasets. For datasets with loading scripts
    (which are no longer supported in datasets>=4.0), use revision='refs/convert/parquet'
    to load from the auto-converted Parquet files.
    
    Args:
        path: HuggingFace dataset path (e.g., 'ilee0022/ImageNet100')
        split: Dataset split (e.g., 'train', 'validation')
        revision: Git revision to load from. Use 'refs/convert/parquet' to load
            from auto-converted Parquet files (recommended for datasets with scripts).
        transform: Optional transform to apply to samples
        rename_columns: Optional dict mapping old column names to new names
        remove_columns: Optional list of columns to remove
        **kwargs: Additional arguments passed to datasets.load_dataset()
    """

    # Auto-converted parquet branch for datasets with loading scripts
    PARQUET_REVISION = "refs/convert/parquet"

    def __init__(
        self, *args, transform=None, rename_columns=None, remove_columns=None, **kwargs
    ):
        super().__init__(transform)
        import datasets

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            s = int(torch.distributed.get_rank()) * 2
            logging.info(
                f"Sleeping for {s}s to avoid race condition of dataset cache"
                " see https://github.com/huggingface/transformers/issues/15976)"
            )
            time.sleep(s)
        if "storage_options" not in kwargs:
            logging.warning(
                "You didn't pass a storage option, adding one to avoid timeout"
            )
            from aiohttp import ClientTimeout

            kwargs["storage_options"] = {
                "client_kwargs": {"timeout": ClientTimeout(total=3600)}
            }

        hf_path = kwargs.get("path", args[0] if len(args) > 0 else None)

        if not isinstance(hf_path, str):
            raise ValueError("Only string dataset path/name is supported")

        self._set_default_cache_dir(hf_path, kwargs)

        load_dataset_fn = datasets.load_dataset
        if self.is_saved_with_save_to_disk(hf_path):
            logging.info(f"Loading dataset with load_from_disk {hf_path}")
            load_dataset_fn = datasets.load_from_disk

        # Try loading the dataset, with automatic fallback to parquet branch
        # if the dataset has deprecated loading scripts
        dataset = self._load_with_fallback(load_dataset_fn, *args, **kwargs)
        dataset = dataset.add_column("sample_idx", list(range(dataset.num_rows)))

        if rename_columns is not None:
            for k, v in rename_columns.items():
                dataset = dataset.rename_column(k, v)
        if remove_columns is not None:
            dataset = dataset.remove_columns(remove_columns)
        self.dataset = dataset

    def _load_with_fallback(self, load_fn, *args, **kwargs):
        """Load dataset with automatic fallback to parquet branch if scripts are not supported."""
        try:
            return with_hf_retry_ratelimit(load_fn, *args, **kwargs)
        except RuntimeError as e:
            error_msg = str(e).lower()
            # Check if this is the "dataset scripts no longer supported" error
            if "dataset scripts are no longer supported" in error_msg or "no longer supported" in error_msg:
                # Check if revision is already set to parquet
                if kwargs.get("revision") == self.PARQUET_REVISION:
                    raise  # Already using parquet branch, nothing more to try
                
                hf_path = kwargs.get("path", args[0] if len(args) > 0 else None)
                logging.warning(
                    f"Dataset '{hf_path}' uses loading scripts which are no longer supported "
                    f"in datasets>=4.0. Falling back to auto-converted Parquet files at "
                    f"revision='{self.PARQUET_REVISION}'."
                )
                kwargs["revision"] = self.PARQUET_REVISION
                return with_hf_retry_ratelimit(load_fn, *args, **kwargs)
            raise  # Re-raise other RuntimeErrors

    def _set_default_cache_dir(self, hf_path, kwargs):
        """Inject cache_dir for HF Hub datasets when caller did not specify one."""
        if "cache_dir" in kwargs:
            return
        if self.is_saved_with_save_to_disk(hf_path):
            return

        cache_dir = self._default_hf_datasets_cache()
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
            logging.info(f"Using HuggingFace datasets cache_dir={cache_dir}")

    def _default_hf_datasets_cache(self):
        """Resolve a cache dir with priority on MCMLSCRATCH-oriented env vars."""
        if os.environ.get("HF_DATASETS_CACHE"):
            return os.environ["HF_DATASETS_CACHE"]

        if os.environ.get("HF_HOME"):
            return str(Path(os.environ["HF_HOME"]) / "datasets")

        if os.environ.get("MCMLSCRATCH"):
            return str(Path(os.environ["MCMLSCRATCH"]) / ".cache" / "huggingface" / "datasets")

        if os.environ.get("XDG_CACHE_HOME"):
            return str(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "datasets")

        return None

    def is_saved_with_save_to_disk(self, path):
        return Path(path, hf_config.DATASET_STATE_JSON_FILENAME).exists()

    def __getitem__(self, idx):
        extra = {}
        if type(idx) is tuple:
            extra["view_idx"] = idx[1]
            idx = idx[0]
        sample = self.dataset[idx]
        return self.process_sample(sample, **extra)

    def __len__(self):
        return self.dataset.num_rows

    @property
    def column_names(self):
        return self.dataset.column_names
