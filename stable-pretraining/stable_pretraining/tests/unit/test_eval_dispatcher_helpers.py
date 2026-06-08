"""Unit tests for the Codex-review fix round (2026-06-02).

Covers:
- `_resolve_tarball_extract_dir` (sentinel-driven tarball-dir resolver).
- `_load_imagenet1k_wnid_index` (sidecar path + length / sort invariants).
- `_load_imagenet_o_tarball_dataset` (keeps every image with dummy label).
- `_WildsLazyDataset` (PIL round-trip through `EvalDataset` shape).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# These tests touch dispatcher helpers that import optional heavy deps
# (`datasets` for the IN-O walker, `torch`/`torchvision` indirectly via
# linear_eval). Skip on minimal envs.
torch = pytest.importorskip("torch")  # noqa: F841

from scripts.geodro_lejepa_eval import (  # noqa: E402
    _WildsLazyDataset,
    _binary_attr,
    _celeba_group_fn,
    _load_imagenet1k_wnid_index,
    _resolve_tarball_extract_dir,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _resolve_tarball_extract_dir
# ---------------------------------------------------------------------------
def test_resolve_tarball_extract_dir_returns_none_when_cache_dir_unset():
    assert _resolve_tarball_extract_dir(None) is None
    assert _resolve_tarball_extract_dir("") is None


def test_resolve_tarball_extract_dir_returns_none_when_sentinel_missing(tmp_path: Path):
    cache = tmp_path / "imagenet_r" / "data"
    cache.mkdir(parents=True)
    # No .prewarm_complete.json next to cache parent.
    assert _resolve_tarball_extract_dir(str(cache)) is None


def test_resolve_tarball_extract_dir_finds_extract_dir_from_sentinel(tmp_path: Path):
    cache = tmp_path / "imagenet_r" / "data"
    cache.mkdir(parents=True)
    extract_dir = cache / "imagenet-r"
    extract_dir.mkdir()
    sentinel = cache.parent / ".prewarm_complete.json"
    sentinel.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_key": "imagenet_r",
                "entries": [
                    {
                        "kind": "tarball",
                        "key": "imagenet_r",
                        "extract_dir": str(extract_dir),
                        "archive_path": str(cache / "imagenet-r.tar"),
                    }
                ],
            }
        )
    )
    assert _resolve_tarball_extract_dir(str(cache)) == str(extract_dir)


def test_resolve_tarball_extract_dir_skips_non_tarball_entries(tmp_path: Path):
    cache = tmp_path / "imagenet_sketch" / "hf_cache"
    cache.mkdir(parents=True)
    sentinel = cache.parent / ".prewarm_complete.json"
    sentinel.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_key": "imagenet_sketch",
                "entries": [
                    {"kind": "hf", "split": "train", "num_rows": 50000},
                ],
            }
        )
    )
    assert _resolve_tarball_extract_dir(str(cache)) is None


def test_resolve_tarball_extract_dir_handles_corrupt_sentinel(tmp_path: Path):
    cache = tmp_path / "imagenet_r" / "data"
    cache.mkdir(parents=True)
    sentinel = cache.parent / ".prewarm_complete.json"
    sentinel.write_text("not valid json {{")
    assert _resolve_tarball_extract_dir(str(cache)) is None


# ---------------------------------------------------------------------------
# _load_imagenet1k_wnid_index
# ---------------------------------------------------------------------------
def test_load_imagenet1k_wnid_index_from_sidecar(tmp_path: Path):
    """Sidecar path: 1000-line WNID list -> map of length 1000."""
    extracted = tmp_path / "imagenet-r"
    extracted.mkdir()
    wnids = [f"n{idx:08d}" for idx in range(1000)]
    (extracted / "imagenet1k_wnids.json").write_text(json.dumps(wnids))
    mapping = _load_imagenet1k_wnid_index(extracted)
    assert len(mapping) == 1000
    assert mapping["n00000000"] == 0
    assert mapping["n00000999"] == 999


def test_load_imagenet1k_wnid_index_rejects_short_sidecar(tmp_path: Path):
    extracted = tmp_path / "imagenet-r"
    extracted.mkdir()
    (extracted / "imagenet1k_wnids.json").write_text(json.dumps(["n00001234"] * 5))
    with pytest.raises(RuntimeError, match="length 5"):
        _load_imagenet1k_wnid_index(extracted)


def test_load_imagenet1k_wnid_index_falls_back_to_timm_when_no_sidecar(tmp_path: Path):
    """The timm fallback must resolve via `timm.data._info`, not `timm.data`.

    timm 1.0.x packages `imagenet_synsets.txt` under `timm.data._info`.
    Codex flagged a prior implementation that pointed at `timm.data` directly,
    which silently fell through and made the eval fallback unusable. This test
    pins the resource path so the regression cannot reappear.
    """
    pytest.importorskip("timm")
    extracted = tmp_path / "imagenet-r"
    extracted.mkdir()
    # No sidecar present -> exercise the timm fallback.
    mapping = _load_imagenet1k_wnid_index(extracted)
    assert len(mapping) == 1000
    # n01440764 is the canonical IN-1k index 0 (goldfish synset).
    assert mapping["n01440764"] == 0
    # Spot-check ordering invariant: keys are alphabetically sorted.
    keys = list(mapping.keys())
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# _load_imagenet_o_tarball_dataset
# ---------------------------------------------------------------------------
def test_load_imagenet_o_tarball_dataset_keeps_all_images(tmp_path: Path):
    """IN-O loader walks every image and assigns label 0, regardless of WNID."""
    pytest.importorskip("datasets")
    pil = pytest.importorskip("PIL.Image")
    from scripts.geodro_lejepa_eval import _load_imagenet_o_tarball_dataset

    extracted = tmp_path / "imagenet-o"
    extracted.mkdir()

    def _write_jpeg(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pil.new("RGB", (8, 8), color=(20, 40, 80)).save(str(path), format="JPEG")

    # Place real images under WNIDs that are NOT in IN-1k.
    for wnid in ["n99999991", "n99999992"]:
        for i in range(3):
            _write_jpeg(extracted / wnid / f"img_{i}.JPEG")
    # A stray PNG at the top level — should also be picked up.
    pil.new("RGB", (4, 4), color=(0, 0, 0)).save(
        str(extracted / "stray.png"), format="PNG"
    )

    dataset = _load_imagenet_o_tarball_dataset(extracted)
    assert len(dataset) == 7  # 3 + 3 + 1
    assert all(int(row["label"]) == 0 for row in dataset)


def test_load_imagenet_o_tarball_dataset_ignores_non_image_files(tmp_path: Path):
    pytest.importorskip("datasets")
    pil = pytest.importorskip("PIL.Image")
    from scripts.geodro_lejepa_eval import _load_imagenet_o_tarball_dataset

    extracted = tmp_path / "imagenet-o"
    extracted.mkdir()
    pil.new("RGB", (8, 8), color=(255, 0, 0)).save(
        str(extracted / "img.JPEG"), format="JPEG"
    )
    (extracted / "README").write_text("not an image")
    (extracted / "manifest.json").write_text("{}")

    dataset = _load_imagenet_o_tarball_dataset(extracted)
    assert len(dataset) == 1


# ---------------------------------------------------------------------------
# _WildsLazyDataset
# ---------------------------------------------------------------------------
class _StubWildsSubset:
    """Minimal stub matching the WILDS subset interface used by the wrapper."""

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        x, y = self._items[idx]
        # WILDS returns (x, y, metadata); metadata content is unused here.
        return x, y, {}


def test_wilds_lazy_dataset_yields_pil_compatible_objects():
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (4, 4), color=(123, 222, 64))
    subset = _StubWildsSubset([(img, 0), (img, 1), (img, 0)])
    ds = _WildsLazyDataset(subset)
    assert len(ds) == 3
    item = ds[1]
    assert "image" in item and "label" in item
    assert int(item["label"]) == 1
    # The crucial assertion: EvalDataset.__getitem__ calls .convert("RGB")
    # on item["image"], which must succeed (i.e., not be an encoded dict).
    converted = item["image"].convert("RGB")
    assert converted.size == (4, 4)


def test_wilds_lazy_dataset_exposes_column_names_for_select_indices():
    """`_select_indices` checks `dataset["label"]` only when `label_map` is set;
    Camelyon17 doesn't pass one, so the index path through `len(dataset)` must
    work. Confirm the wrapper supplies `len` and `column_names`.
    """
    subset = _StubWildsSubset([(object(), 0)] * 5)
    ds = _WildsLazyDataset(subset)
    assert len(ds) == 5
    assert ds.column_names == ["image", "label"]


# ---------------------------------------------------------------------------
# CelebA attribute normalization helpers
# ---------------------------------------------------------------------------
def test_binary_attr_accepts_common_binary_codings():
    assert _binary_attr(-1, "Blond_Hair") == 0
    assert _binary_attr(0, "Blond_Hair") == 0
    assert _binary_attr(1, "Blond_Hair") == 1
    assert _binary_attr(torch.tensor(-1), "Male") == 0
    assert _binary_attr(torch.tensor(1), "Male") == 1


def test_binary_attr_rejects_non_binary_values():
    with pytest.raises(ValueError, match="Expected binary coding"):
        _binary_attr(2, "Male")


def test_celeba_group_fn_handles_neg1_pos1_encoding():
    fn = _celeba_group_fn("Blond_Hair", "Male")
    assert fn({"Blond_Hair": -1, "Male": -1}, 0, 0) == 0
    assert fn({"Blond_Hair": -1, "Male": 1}, 0, 0) == 1
    assert fn({"Blond_Hair": 1, "Male": -1}, 0, 0) == 2
    assert fn({"Blond_Hair": 1, "Male": 1}, 0, 0) == 3
