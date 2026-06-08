#!/bin/bash
#===============================================================================
# Prewarm GeoDRO-LeJEPA evaluation-suite datasets on MCML scratch
#
# Usage:
#   sbatch scripts/slurm/download_datasets.sbatch
#
# The dataset scope covers the canonical 9-mode evaluation suite wrapped by
# scripts/slurm/run_full_eval_suite.sh:
#   - ilee0022/ImageNet100, train + validation, refs/convert/parquet  (HF)
#   - WNJXYK/TTA-ImageNet-C, v1.0, configured corruptions/severities  (HF)
#   - grodino/waterbirds, train + validation + test                   (HF)
#   - uoft-cs/cifar10, train + test                                   (HF)
#   - Huanyiiiii/Imagenet_Sketch (parquet revision)                   (HF mirror)
#   - tpremoli/CelebA-attrs, train + validation + test                (HF mirror)
#   - imagenet_r / imagenet_a / imagenet_o                            (tarballs)
#   - camelyon17 via wilds Python package                             (custom)
#
# Set DATASETS_TO_PREWARM to a comma-separated subset to skip datasets, e.g.:
#   DATASETS_TO_PREWARM=imagenet_sketch,imagenet_r sbatch …
#
# This job uses one process so Hugging Face cache preparation cannot race across
# DDP ranks. Each dataset writes a .prewarm_complete.json sentinel only after all
# required splits/configs/tarballs land successfully.
#===============================================================================

#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
#SBATCH --job-name=download-geodro-v1-data
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

ENV_FILE="${PROJECT_ROOT}/.env.local"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MCMLSCRATCH="${MCMLSCRATCH:-/dss/mcmlscratch/0D/ra59dut2}"
GEODRO_DATASET_ROOT="${GEODRO_DATASET_ROOT:-${MCMLSCRATCH}/datasets/geodro_v1}"
export MCMLSCRATCH GEODRO_DATASET_ROOT
export HF_HOME="${GEODRO_HF_HOME:-${GEODRO_DATASET_ROOT}/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

case "${GEODRO_DATASET_ROOT}" in
    "${MCMLSCRATCH}"/*) ;;
    *)
        echo "ERROR: GEODRO_DATASET_ROOT must be on MCML scratch: ${MCMLSCRATCH}"
        echo "Current GEODRO_DATASET_ROOT=${GEODRO_DATASET_ROOT}"
        exit 1
        ;;
esac
case "${HF_HOME}" in
    "${MCMLSCRATCH}"/*) ;;
    *)
        echo "ERROR: HF_HOME must be on MCML scratch: ${MCMLSCRATCH}"
        echo "Current HF_HOME=${HF_HOME}"
        exit 1
        ;;
esac

mkdir -p "${GEODRO_DATASET_ROOT}" "${HF_HOME}" "${PROJECT_ROOT}/logs"

LOG_DIR="${MCMLSCRATCH}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/download-geodro-v1-datasets-${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "${HUGGINGFACE_HUB_TOKEN}" > "${HF_HOME}/token"
    chmod 600 "${HF_HOME}/token"
elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
    chmod 600 "${HF_HOME}/token"
fi

echo "============================================================"
echo "GeoDRO-LeJEPA v1 Dataset Prewarm"
echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Dataset root:        ${GEODRO_DATASET_ROOT}"
echo "HF_HOME:             ${HF_HOME}"
echo "Token file:          ${HF_HOME}/token"
echo "Log file:            ${LOG_FILE}"
echo "Datasets:            ${DATASETS_TO_PREWARM:-imagenet100,imagenetc,waterbirds,cifar10,imagenet_sketch,imagenet_r,imagenet_a,imagenet_o,celeba_groups,camelyon17}"
echo "============================================================"

source ~/.bashrc
mamba activate py312

echo ""
echo "Testing internet connectivity..."
if curl -s --max-time 10 https://huggingface.co > /dev/null; then
    echo "Internet connectivity OK"
else
    echo "ERROR: Cannot reach huggingface.co"
    echo "This node may not have internet access. Try a different partition."
    exit 1
fi

echo ""
echo "Checking Hugging Face auth..."
python << 'PYTHON_SCRIPT'
from huggingface_hub import HfApi

try:
    who = HfApi().whoami()
    print(f"Logged in as: {who.get('name', who)}")
except Exception as exc:
    print(f"WARNING: Hugging Face auth check failed: {exc}")
PYTHON_SCRIPT

python << 'PYTHON_SCRIPT'
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("geodro_v1_dataset_prewarm")

DATASET_ROOT = Path(os.environ["GEODRO_DATASET_ROOT"]).expanduser()
PARQUET_REVISION = "refs/convert/parquet"
IMAGENETC_REVISION = "v1.0"
SCHEMA_VERSION = 2  # bumped: tarball + wilds branches added

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

# Tarball URLs for the Hendrycks ImageNet-{R,A,O} datasets.
# Source: https://github.com/hendrycks/imagenet-r and https://github.com/hendrycks/natural-adv-examples
IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
IMAGENET_A_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
IMAGENET_O_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"

DATASET_ORDER = (
    "imagenet100",
    "imagenetc",
    "waterbirds",
    "cifar10",
    "imagenet_sketch",
    "imagenet_r",
    "imagenet_a",
    "imagenet_o",
    "celeba_groups",
    "camelyon17",
)
selected = os.environ.get("DATASETS_TO_PREWARM", ",".join(DATASET_ORDER))
SELECTED_DATASETS = {part.strip() for part in selected.split(",") if part.strip()}
unknown = SELECTED_DATASETS.difference(DATASET_ORDER)
if unknown:
    raise SystemExit(f"Unknown DATASETS_TO_PREWARM values: {sorted(unknown)}")


def dataset_dir(key: str) -> Path:
    return DATASET_ROOT / key


def cache_dir(key: str) -> Path:
    env_name = f"{key.upper()}_CACHE_DIR"
    default = dataset_dir(key) / ("hf_cache" if _is_hf_kind(key) else "data")
    return Path(os.environ.get(env_name, str(default))).expanduser()


def _is_hf_kind(key: str) -> bool:
    return key in {
        "imagenet100",
        "imagenetc",
        "waterbirds",
        "cifar10",
        "imagenet_sketch",
        "celeba_groups",
    }


def sentinel_path(key: str) -> Path:
    return dataset_dir(key) / ".prewarm_complete.json"


def hf_entry(
    key: str,
    dataset_id: str,
    split: str,
    *,
    name: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "hf",
        "key": key,
        "dataset_id": dataset_id,
        "name": name,
        "split": split,
        "revision": revision,
        "cache_dir": str(cache_dir(key)),
    }


def tarball_entry(key: str, url: str, archive_name: str) -> dict[str, Any]:
    return {
        "kind": "tarball",
        "key": key,
        "url": url,
        "archive_name": archive_name,
        "cache_dir": str(cache_dir(key)),
    }


def wilds_entry(key: str, dataset_name: str, version: str | None = None) -> dict[str, Any]:
    return {
        "kind": "wilds",
        "key": key,
        "wilds_name": dataset_name,
        "version": version,
        "cache_dir": str(cache_dir(key)),
    }


# Backwards-compatible alias used by older sentinels (schema_version=1).
def entry(*args, **kwargs):  # noqa: D401 - shim for schema-v1 manifest readers
    return hf_entry(*args, **kwargs)


def expected_entries(key: str) -> list[dict[str, Any]]:
    if key == "imagenet100":
        return [
            hf_entry(key, "ilee0022/ImageNet100", "train", revision=PARQUET_REVISION),
            hf_entry(key, "ilee0022/ImageNet100", "validation", revision=PARQUET_REVISION),
        ]
    if key == "imagenetc":
        return [
            hf_entry(
                key,
                "WNJXYK/TTA-ImageNet-C",
                f"severity_{severity}",
                name=corruption,
                revision=IMAGENETC_REVISION,
            )
            for corruption in IMAGENETC_CORRUPTIONS
            for severity in range(1, 6)
        ]
    if key == "waterbirds":
        return [
            hf_entry(key, "grodino/waterbirds", "train"),
            hf_entry(key, "grodino/waterbirds", "validation"),
            hf_entry(key, "grodino/waterbirds", "test"),
        ]
    if key == "cifar10":
        return [
            hf_entry(key, "uoft-cs/cifar10", "train"),
            hf_entry(key, "uoft-cs/cifar10", "test"),
        ]
    if key == "imagenet_sketch":
        # ImageNet-Sketch (Wang et al. 2019). Parquet mirror avoids legacy dataset scripts
        # removed in recent `datasets` releases (songweig/imagenet_sketch breaks).
        return [
            hf_entry(key, "Huanyiiiii/Imagenet_Sketch", "train", revision=PARQUET_REVISION),
        ]
    if key == "imagenet_r":
        return [tarball_entry(key, IMAGENET_R_URL, "imagenet-r.tar")]
    if key == "imagenet_a":
        return [tarball_entry(key, IMAGENET_A_URL, "imagenet-a.tar")]
    if key == "imagenet_o":
        return [tarball_entry(key, IMAGENET_O_URL, "imagenet-o.tar")]
    if key == "celeba_groups":
        # CelebA with attribute labels; (Blond_Hair × Male) is the canonical
        # Sagawa-et-al subpopulation eval (4 groups).
        return [
            hf_entry(key, "tpremoli/CelebA-attrs", "train"),
            hf_entry(key, "tpremoli/CelebA-attrs", "validation"),
            hf_entry(key, "tpremoli/CelebA-attrs", "test"),
        ]
    if key == "camelyon17":
        return [wilds_entry(key, "camelyon17")]
    raise ValueError(f"Unknown dataset key: {key}")


def entry_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    kind = item.get("kind", "hf")
    if kind == "tarball":
        return ("tarball", item["url"], item["archive_name"])
    if kind == "wilds":
        return ("wilds", item["wilds_name"], item.get("version"))
    return ("hf", item["dataset_id"], item.get("name"), item["split"], item.get("revision"))


def manifest_is_valid(key: str, manifest: dict[str, Any]) -> bool:
    # Accept schema_version 1 (older HF-only sentinels) and 2 (current).
    if manifest.get("schema_version") not in (1, SCHEMA_VERSION):
        return False
    if manifest.get("dataset_key") != key:
        return False
    prepared = manifest.get("entries", [])
    if not isinstance(prepared, list):
        return False

    prepared_by_identity = {entry_identity(item): item for item in prepared if isinstance(item, dict)}
    for item in expected_entries(key):
        current = prepared_by_identity.get(entry_identity(item))
        if current is None:
            return False
        kind = item.get("kind", "hf")
        if kind == "hf":
            if int(current.get("num_rows", 0)) <= 0:
                return False
            if current.get("cache_dir") != str(cache_dir(key)):
                return False
        elif kind == "tarball":
            archive = Path(current.get("archive_path", ""))
            extract = Path(current.get("extract_dir", ""))
            if not archive.is_file() or not extract.is_dir():
                return False
        elif kind == "wilds":
            root = Path(current.get("wilds_root_dir", ""))
            if not root.is_dir():
                return False
        else:
            return False
    return True


def load_one(item: dict[str, Any]) -> dict[str, Any]:
    """Dispatch on item['kind'] and prepare the dataset locally."""
    kind = item.get("kind", "hf")
    if kind == "hf":
        return load_one_hf(item)
    if kind == "tarball":
        return load_one_tarball(item)
    if kind == "wilds":
        return load_one_wilds(item)
    raise ValueError(f"Unknown entry kind: {kind!r}")


def load_one_hf(item: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "split": item["split"],
        "cache_dir": item["cache_dir"],
    }
    if item.get("name"):
        kwargs["name"] = item["name"]
    if item.get("revision"):
        kwargs["revision"] = item["revision"]

    label = item["dataset_id"]
    if item.get("name"):
        label += f"/{item['name']}"
    label += f"::{item['split']}"
    logger.info("PREPARE hf %s", label)

    try:
        dataset = load_dataset(item["dataset_id"], **kwargs)
    except RuntimeError as exc:
        if "dataset scripts are no longer supported" in str(exc).lower():
            kwargs["revision"] = PARQUET_REVISION
            dataset = load_dataset(item["dataset_id"], **kwargs)
        else:
            raise
    num_rows = len(dataset)
    if num_rows <= 0:
        raise RuntimeError(f"Loaded empty dataset for {label}")

    out = dict(item)
    out["num_rows"] = int(num_rows)
    out["column_names"] = list(dataset.column_names)
    logger.info("READY hf %s rows=%s", label, num_rows)
    return out


def load_one_tarball(item: dict[str, Any]) -> dict[str, Any]:
    """Download a tarball to cache_dir/archive and extract once.

    Idempotent: skips download if the archive exists and a top-level extract
    directory exists. The caller's manifest validator re-checks paths each run.
    """
    cache = Path(item["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / item["archive_name"]
    extract_dir = cache / item["archive_name"].rsplit(".", 1)[0]

    if archive_path.is_file():
        logger.info("SKIP tarball download (already present): %s", archive_path)
    else:
        logger.info("DOWNLOAD tarball %s -> %s", item["url"], archive_path)
        tmp = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            with urllib.request.urlopen(item["url"], timeout=3600) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            tmp.replace(archive_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    if extract_dir.is_dir() and any(extract_dir.iterdir()):
        logger.info("SKIP tarball extract (already present): %s", extract_dir)
    else:
        logger.info("EXTRACT tarball %s -> %s", archive_path, cache)
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(path=cache)

    if not extract_dir.is_dir():
        # Some archives extract under a slightly different folder name; pick the
        # only directory in cache that was modified by this extraction.
        candidates = sorted(p for p in cache.iterdir() if p.is_dir())
        if not candidates:
            raise RuntimeError(f"No extracted directory found under {cache}")
        extract_dir = candidates[0]
        logger.info("Tarball extracted to %s", extract_dir)

    # IN-R/A use the WNID sidecar at eval time to map WNIDs -> IN-1k indices.
    # IN-O ignores labels (it's an OOD-detection-only dataset), so writing the
    # sidecar is harmless but unused. Best effort: log on miss, never fail
    # prewarm because of it.
    if item["key"] in ("imagenet_r", "imagenet_a", "imagenet_o"):
        try:
            _write_imagenet1k_wnids_sidecar(extract_dir)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "WNID sidecar not written for %s (%s). Eval will still work for "
                "IN-O; IN-R/A will fail at load until the sidecar is provided "
                "manually.",
                extract_dir,
                exc,
            )

    out = dict(item)
    out["archive_path"] = str(archive_path)
    out["extract_dir"] = str(extract_dir)
    out["archive_size_bytes"] = int(archive_path.stat().st_size)
    return out


def _write_imagenet1k_wnids_sidecar(extract_dir: Path) -> None:
    """Write `<extract_dir>/imagenet1k_wnids.json` from a trusted package source.

    The eval script (`scripts.geodro_lejepa_eval._load_imagenet_tarball_dataset`)
    reads this sidecar to map WNID-named subdirectories to IN-1k integer indices.
    The list must be the canonical alphabetical order. We source it from `timm`
    (already a project dep) which ships `imagenet_synsets` as a packaged
    string constant. Skips if the sidecar already exists.
    """
    sidecar = extract_dir / "imagenet1k_wnids.json"
    if sidecar.is_file():
        logger.info("SKIP imagenet1k_wnids sidecar (already present): %s", sidecar)
        return

    wnids = _read_imagenet1k_wnids_from_timm()
    if len(wnids) != 1000:
        raise RuntimeError(
            f"Refusing to write WNID sidecar with non-canonical length {len(wnids)} "
            "(expected 1000)."
        )
    if list(wnids) != sorted(wnids):
        raise RuntimeError(
            "Refusing to write WNID sidecar that is not alphabetically sorted "
            "(IN-1k integer indices are derived from sorted-WNID order)."
        )
    sidecar.write_text(json.dumps(list(wnids), indent=0) + "\n", encoding="utf-8")
    logger.info("WROTE imagenet1k_wnids sidecar: %s (n=%d)", sidecar, len(wnids))


def _read_imagenet1k_wnids_from_timm() -> list[str]:
    """Read the IN-1k WNID list from timm's packaged synset resource.

    timm packages the canonical IN-1k synset list under
    `timm.data._info/imagenet_synsets.txt` (verified against timm 1.0.x).
    Older forks may expose `timm.data` directly; we try both before failing.
    The file contains 1000 alphabetically-sorted WNIDs, matching the IN-1k
    integer-index ordering used by torchvision.
    """
    try:
        from importlib import resources
    except Exception as exc:
        raise RuntimeError(
            "importlib.resources is unavailable; cannot read timm synsets."
        ) from exc

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
        raise RuntimeError(
            "Could not locate `imagenet_synsets.txt` under timm.data._info or "
            "timm.data. timm is a project dependency; verify the conda env. "
            "To bypass, manually drop imagenet1k_wnids.json (1000 alphabetically "
            "sorted WNIDs) at the extracted-tarball root."
        )

    wnids = [line.strip() for line in text.splitlines() if line.strip()]
    return wnids


def load_one_wilds(item: dict[str, Any]) -> dict[str, Any]:
    """Use the official `wilds` package to download the dataset.

    Adds a soft import: if `wilds` is missing the user gets a clear pip-install
    message instead of an obscure ImportError mid-job.
    """
    try:
        from wilds import get_dataset as wilds_get_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The `wilds` package is required for the camelyon17 prewarm branch. "
            "Install via `pip install wilds` in the prewarm conda env."
        ) from exc

    cache = Path(item["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    name = item["wilds_name"]
    version = item.get("version")

    logger.info("PREPARE wilds %s root=%s version=%s", name, cache, version)
    kwargs: dict[str, Any] = {"dataset": name, "root_dir": str(cache), "download": True}
    if version:
        kwargs["version"] = version
    dataset = wilds_get_dataset(**kwargs)
    logger.info("READY wilds %s n=%d", name, len(dataset))

    out = dict(item)
    out["wilds_root_dir"] = str(cache)
    out["wilds_n"] = int(len(dataset))
    return out


def prepare_dataset(key: str) -> dict[str, Any]:
    dataset_dir(key).mkdir(parents=True, exist_ok=True)
    cache_dir(key).mkdir(parents=True, exist_ok=True)
    sentinel = sentinel_path(key)

    if sentinel.is_file():
        try:
            manifest = json.loads(sentinel.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest_is_valid(key, manifest):
            print(f"SKIP {key}: already prepared at {cache_dir(key)}")
            return {"dataset_key": key, "status": "skipped", "sentinel": str(sentinel)}
        print(f"REBUILD {key}: existing sentinel failed validation")

    prepared_entries = [load_one(item) for item in expected_entries(key)]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_key": key,
        "status": "ready",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir(key)),
        "entries": prepared_entries,
    }
    sentinel.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"READY {key}: wrote {sentinel}")
    return {"dataset_key": key, "status": "prepared", "sentinel": str(sentinel)}


def _check_optional_dependencies() -> None:
    """Fail fast with a pip-install pointer if a selected dataset needs an extra.

    Currently only `camelyon17` requires an optional package (`wilds`). The
    project ships `wilds` under the `[eval]` extra (see pyproject.toml).
    Detecting it missing here avoids burning prewarm time on the HF datasets
    only to error out when reaching camelyon17.
    """
    if "camelyon17" not in SELECTED_DATASETS:
        return
    try:
        import wilds  # noqa: F401
    except ImportError:
        raise SystemExit(
            "ERROR: the `wilds` package is required for the camelyon17 prewarm "
            "branch but is not installed in the active env.\n"
            "Install via:\n"
            "    pip install -e \"stable-pretraining[eval]\"\n"
            "or remove camelyon17 from DATASETS_TO_PREWARM if you do not plan "
            "to evaluate on Camelyon17."
        )


def main() -> int:
    _check_optional_dependencies()
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Selected datasets: {', '.join(key for key in DATASET_ORDER if key in SELECTED_DATASETS)}")
    results: list[dict[str, Any]] = []
    for key in DATASET_ORDER:
        if key not in SELECTED_DATASETS:
            print(f"SKIP {key}: not selected")
            continue
        results.append(prepare_dataset(key))

    summary_path = DATASET_ROOT / "prewarm_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Dataset prewarm failed")
        raise
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Dataset prewarm completed"
echo "============================================================"
echo "Dataset root: ${GEODRO_DATASET_ROOT}"
echo "Checking dataset root size..."
du -sh "${GEODRO_DATASET_ROOT}" 2>/dev/null || echo "Could not determine dataset root size"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done"
