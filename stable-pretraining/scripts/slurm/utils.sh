#!/bin/bash
#===============================================================================
# Shared utility functions for Slurm job scripts
#
# Source this file in your sbatch scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
#===============================================================================

#-------------------------------------------------------------------------------
# resolve_wandb_key - Resolve WANDB_API_KEY from env or config files
#
# Priority:
#   1) WANDB_API_KEY env var
#   2) WANDB_API_KEY_FILE (if set and readable)
#   3) ~/.config/wandb/settings (api_key=...)
#   4) ~/.netrc (machine api.wandb.ai)
#
# Returns:
#   0 if WANDB_API_KEY is set, 1 otherwise
#-------------------------------------------------------------------------------
resolve_wandb_key() {
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        return 0
    fi

    if [[ -n "${WANDB_API_KEY_FILE:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
        export WANDB_API_KEY="$(< "${WANDB_API_KEY_FILE}")"
        [[ -n "${WANDB_API_KEY}" ]] && return 0
    fi

    local settings_file="${HOME}/.config/wandb/settings"
    if [[ -f "${settings_file}" ]]; then
        local key
        key="$(grep -E '^api_key[[:space:]]*=' "${settings_file}" | tail -n 1 | cut -d'=' -f2- | tr -d '[:space:]')"
        if [[ -n "${key}" ]]; then
            export WANDB_API_KEY="${key}"
            return 0
        fi
    fi

    local netrc_file="${HOME}/.netrc"
    if [[ -f "${netrc_file}" ]]; then
        local key
        key="$(awk 'BEGIN{found=0} /machine[[:space:]]+api\\.wandb\\.ai/{found=1} found && /password/{print $2; exit}' "${netrc_file}")"
        if [[ -n "${key}" ]]; then
            export WANDB_API_KEY="${key}"
            return 0
        fi
    fi

    return 1
}

#-------------------------------------------------------------------------------
# require_wandb_key - Ensure W&B credentials exist for online logging
#
# Returns:
#   0 if credentials found or WANDB_MODE != online, 1 otherwise
#-------------------------------------------------------------------------------
require_wandb_key() {
    if [[ "${WANDB_MODE:-online}" != "online" ]]; then
        return 0
    fi
    if resolve_wandb_key; then
        return 0
    fi
    echo "ERROR: WANDB_API_KEY is not set and no W&B credentials found"
    echo "Set WANDB_API_KEY, or run 'wandb login' once, or set WANDB_MODE=offline"
    return 1
}

#-------------------------------------------------------------------------------
# find_checkpoint - Locate the best or last checkpoint in a directory
#
# Usage:
#   CKPT=$(find_checkpoint "/path/to/output/dir")
#
# Returns:
#   Path to checkpoint file, or empty string if not found
#-------------------------------------------------------------------------------
find_checkpoint() {
    local output_dir="$1"
    local checkpoint_dir="${output_dir}/checkpoints"
    local checkpoint=""

    # Priority order: best -> last -> any .ckpt file
    if [[ -f "${checkpoint_dir}/best.ckpt" ]]; then
        checkpoint="${checkpoint_dir}/best.ckpt"
    else
        # Handle versioned best checkpoints (best-v1.ckpt, best-v2.ckpt, ...)
        local best_versioned
        best_versioned="$(ls -t "${checkpoint_dir}"/best*.ckpt 2>/dev/null | head -n 1)"
        if [[ -n "${best_versioned}" ]]; then
            checkpoint="${best_versioned}"
        elif [[ -f "${checkpoint_dir}/last.ckpt" ]]; then
            # Try to resolve best_model_path stored inside last.ckpt (Lightning)
            checkpoint="$(python - <<'PY' "${checkpoint_dir}/last.ckpt"
import os
import sys

last_ckpt = sys.argv[1]
best = None

try:
    import torch
    ckpt = torch.load(last_ckpt, map_location="cpu")
    callbacks = ckpt.get("callbacks", {})
    for state in callbacks.values():
        if isinstance(state, dict):
            best = state.get("best_model_path")
            if best:
                break
except Exception:
    best = None

if best:
    best = os.path.expanduser(best)
    if not os.path.isabs(best):
        best = os.path.join(os.path.dirname(last_ckpt), best)
    if os.path.isfile(best):
        print(best)
        sys.exit(0)

print(last_ckpt)
PY
)"
        else
            # Search for any checkpoint file
            checkpoint=$(find "${output_dir}" -name "*.ckpt" -type f 2>/dev/null | sort -V | tail -n 1)
        fi
    fi

    echo "${checkpoint}"
}

#-------------------------------------------------------------------------------
# setup_logging - Create logging directory structure
#
# Usage:
#   setup_logging "/path/to/output/dir"
#
# Creates:
#   - output_dir/
#   - output_dir/checkpoints/
#   - output_dir/logs/
#-------------------------------------------------------------------------------
setup_logging() {
    local output_dir="$1"
    
    mkdir -p "${output_dir}"
    mkdir -p "${output_dir}/checkpoints"
    mkdir -p "${output_dir}/logs"
    
    echo "Created logging directories in: ${output_dir}"
}

#-------------------------------------------------------------------------------
# validate_env - Check that the conda environment has required packages
#
# Usage:
#   validate_env "py312"
#
# Returns:
#   0 if all packages are present, 1 otherwise
#-------------------------------------------------------------------------------
validate_env() {
    local env_name="${1:-py312}"
    local missing_packages=()
    
    # List of required packages
    local required_packages=(
        "torch"
        "lightning"
        "hydra-core"
        "omegaconf"
        "timm"
        "wandb"
        "torchmetrics"
        "datasets"
        "psutil"
    )
    
    echo "Validating conda environment: ${env_name}"
    
    for pkg in "${required_packages[@]}"; do
        import_name="${pkg//-/_}"
        case "${pkg}" in
            "hydra-core") import_name="hydra" ;;
        esac
        if ! python -c "import ${import_name}" 2>/dev/null; then
            missing_packages+=("${pkg}")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        echo "ERROR: Missing packages: ${missing_packages[*]}"
        echo "Install them with: mamba install ${missing_packages[*]}"
        return 1
    fi

    echo "All required packages are installed"
    return 0
}

#-------------------------------------------------------------------------------
# require_eval_extra - Verify an eval-suite optional dependency is installed
#
# Some eval-suite dispatchers depend on optional packages declared under the
# `eval` extra in pyproject.toml (currently only `wilds` for Camelyon17).
# Sbatch launchers for those modes should call this helper so the user gets a
# pip-install pointer instead of a half-finished job that errors mid-run.
#
# Usage:
#   require_eval_extra wilds   # exits 1 if `import wilds` fails
#-------------------------------------------------------------------------------
require_eval_extra() {
    local pkg="$1"
    if [[ -z "${pkg}" ]]; then
        echo "ERROR: require_eval_extra called without a package argument" >&2
        return 1
    fi
    local py
    if command -v python >/dev/null 2>&1; then
        py=python
    elif command -v python3 >/dev/null 2>&1; then
        py=python3
    else
        echo "ERROR: No python interpreter on PATH; cannot verify ${pkg}." >&2
        return 1
    fi
    if "${py}" -c "import ${pkg}" 2>/dev/null; then
        return 0
    fi
    echo "ERROR: Eval extra '${pkg}' is not installed in the active env." >&2
    echo "Install via:" >&2
    echo "    pip install -e \"stable-pretraining[eval]\"" >&2
    return 1
}

#-------------------------------------------------------------------------------
# GeoDRO v1 dataset cache helpers
#
# The dataset prewarm job writes one sentinel per dataset under:
#   ${GEODRO_DATASET_ROOT:-${MCMLSCRATCH}/datasets/geodro_v1}/<dataset>/
#
# Training and evaluation jobs should require the sentinel before starting any
# DDP or eval work. This prevents multiple ranks/jobs from racing Hugging Face's
# download_and_prepare path on shared scratch storage.
#-------------------------------------------------------------------------------
geodro_dataset_root() {
    local scratch="${MCMLSCRATCH:-/dss/mcmlscratch/0D/ra59dut2}"
    echo "${GEODRO_DATASET_ROOT:-${scratch}/datasets/geodro_v1}"
}

geodro_dataset_cache_dir() {
    local dataset_key="$1"
    local root
    root="$(geodro_dataset_root)"
    case "${dataset_key}" in
        imagenet100)
            echo "${IMAGENET100_CACHE_DIR:-${root}/imagenet100/hf_cache}"
            ;;
        imagenetc)
            echo "${IMAGENETC_CACHE_DIR:-${root}/imagenetc/hf_cache}"
            ;;
        waterbirds)
            echo "${WATERBIRDS_CACHE_DIR:-${root}/waterbirds/hf_cache}"
            ;;
        cifar10)
            echo "${CIFAR10_CACHE_DIR:-${root}/cifar10/hf_cache}"
            ;;
        imagenet_sketch)
            echo "${IMAGENET_SKETCH_CACHE_DIR:-${root}/imagenet_sketch/hf_cache}"
            ;;
        imagenet_r)
            echo "${IMAGENET_R_CACHE_DIR:-${root}/imagenet_r/data}"
            ;;
        imagenet_a)
            echo "${IMAGENET_A_CACHE_DIR:-${root}/imagenet_a/data}"
            ;;
        imagenet_o)
            echo "${IMAGENET_O_CACHE_DIR:-${root}/imagenet_o/data}"
            ;;
        celeba_groups)
            echo "${CELEBA_GROUPS_CACHE_DIR:-${root}/celeba_groups/hf_cache}"
            ;;
        camelyon17)
            echo "${CAMELYON17_CACHE_DIR:-${root}/camelyon17/data}"
            ;;
        *)
            echo "ERROR: unknown GeoDRO dataset key '${dataset_key}'" >&2
            return 1
            ;;
    esac
}

geodro_dataset_sentinel() {
    local dataset_key="$1"
    local root
    root="$(geodro_dataset_root)"
    case "${dataset_key}" in
        imagenet100|imagenetc|waterbirds|cifar10|imagenet_sketch|imagenet_r|imagenet_a|imagenet_o|celeba_groups|camelyon17)
            echo "${root}/${dataset_key}/.prewarm_complete.json"
            ;;
        *)
            echo "ERROR: unknown GeoDRO dataset key '${dataset_key}'" >&2
            return 1
            ;;
    esac
}

require_geodro_dataset_prewarm() {
    local dataset_key="$1"
    local sentinel
    sentinel="$(geodro_dataset_sentinel "${dataset_key}")"

    if [[ -f "${sentinel}" ]]; then
        return 0
    fi

    echo "ERROR: GeoDRO v1 dataset '${dataset_key}' is not prewarmed."
    echo "Missing sentinel: ${sentinel}"
    echo "Run: cd stable-pretraining && sbatch scripts/slurm/download_datasets.sbatch"
    return 1
}

setup_geodro_dataset_cache() {
    local dataset_key="$1"
    local require_prewarm="${2:-${REQUIRE_GEODRO_DATASET_PREWARM:-1}}"
    local scratch="${MCMLSCRATCH:-/dss/mcmlscratch/0D/ra59dut2}"
    local root
    local cache_dir

    export MCMLSCRATCH="${scratch}"
    root="$(geodro_dataset_root)"
    cache_dir="$(geodro_dataset_cache_dir "${dataset_key}")"

    case "${root}" in
        "${MCMLSCRATCH}"/*) ;;
        *)
            echo "ERROR: GEODRO_DATASET_ROOT must be on MCML scratch: ${MCMLSCRATCH}"
            echo "Current GEODRO_DATASET_ROOT=${root}"
            return 1
            ;;
    esac

    export GEODRO_DATASET_ROOT="${root}"
    export HF_HOME="${GEODRO_HF_HOME:-${GEODRO_DATASET_ROOT}/hf_home}"
    export HF_DATASETS_CACHE="${cache_dir}"

    case "${HF_HOME}" in
        "${MCMLSCRATCH}"/*) ;;
        *)
            echo "ERROR: HF_HOME must be on MCML scratch shared storage: ${MCMLSCRATCH}"
            echo "Current HF_HOME=${HF_HOME}"
            return 1
            ;;
    esac
    case "${HF_DATASETS_CACHE}" in
        "${MCMLSCRATCH}"/*) ;;
        *)
            echo "ERROR: HF_DATASETS_CACHE must be on MCML scratch shared storage: ${MCMLSCRATCH}"
            echo "Current HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
            return 1
            ;;
    esac

    if [[ "${require_prewarm}" == "1" ]]; then
        require_geodro_dataset_prewarm "${dataset_key}" || return 1
    fi

    mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
    echo "GeoDRO dataset root: ${GEODRO_DATASET_ROOT}"
    echo "GeoDRO dataset key:  ${dataset_key}"
    echo "HF_HOME:            ${HF_HOME}"
    echo "HF datasets:        ${HF_DATASETS_CACHE}"
}

#-------------------------------------------------------------------------------
# log_job_info - Save job information for reproducibility
#
# Usage:
#   log_job_info "/path/to/output/dir"
#-------------------------------------------------------------------------------
log_job_info() {
    local output_dir="$1"
    local info_file="${output_dir}/job_info.txt"
    
    cat > "${info_file}" << EOF
================================================================================
Job Information
================================================================================
Job ID:            ${SLURM_JOB_ID:-local}
Job Name:          ${SLURM_JOB_NAME:-N/A}
Submit Time:       $(date -Iseconds)
Node List:         ${SLURM_JOB_NODELIST:-N/A}
Num Nodes:         ${SLURM_NNODES:-1}
Tasks per Node:    ${SLURM_NTASKS_PER_NODE:-1}
CPUs per Task:     ${SLURM_CPUS_PER_TASK:-1}
Partition:         ${SLURM_JOB_PARTITION:-N/A}
Working Directory: $(pwd)

================================================================================
Environment
================================================================================
CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-N/A}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}
OMP_NUM_THREADS: ${OMP_NUM_THREADS:-N/A}
MASTER_ADDR: ${MASTER_ADDR:-N/A}
MASTER_PORT: ${MASTER_PORT:-N/A}
WANDB_PROJECT: ${WANDB_PROJECT:-N/A}

================================================================================
Git Information
================================================================================
$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
$(git status --porcelain 2>/dev/null | head -5 || echo "N/A")
================================================================================
EOF
    
    echo "Job info saved to: ${info_file}"
}

#-------------------------------------------------------------------------------
# get_master_addr - Get the master node address for distributed training
#
# Usage:
#   MASTER_ADDR=$(get_master_addr)
#-------------------------------------------------------------------------------
get_master_addr() {
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1
    else
        echo "localhost"
    fi
}

#-------------------------------------------------------------------------------
# wait_for_checkpoint - Wait for a checkpoint file to appear
#
# Usage:
#   wait_for_checkpoint "/path/to/checkpoint.ckpt" 60
#
# Arguments:
#   $1 - Path to checkpoint file
#   $2 - Timeout in seconds (default: 300)
#-------------------------------------------------------------------------------
wait_for_checkpoint() {
    local checkpoint_path="$1"
    local timeout="${2:-300}"
    local elapsed=0
    local interval=10
    
    echo "Waiting for checkpoint: ${checkpoint_path}"
    
    while [[ ! -f "${checkpoint_path}" && ${elapsed} -lt ${timeout} ]]; do
        sleep ${interval}
        elapsed=$((elapsed + interval))
        echo "  ... waiting (${elapsed}s / ${timeout}s)"
    done
    
    if [[ -f "${checkpoint_path}" ]]; then
        echo "Checkpoint found: ${checkpoint_path}"
        return 0
    else
        echo "ERROR: Timeout waiting for checkpoint"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# cleanup_old_checkpoints - Remove old checkpoints, keeping only the N most recent
#
# Usage:
#   cleanup_old_checkpoints "/path/to/checkpoints" 3
#
# Arguments:
#   $1 - Checkpoint directory
#   $2 - Number of checkpoints to keep (default: 2)
#-------------------------------------------------------------------------------
cleanup_old_checkpoints() {
    local checkpoint_dir="$1"
    local keep_count="${2:-2}"
    
    if [[ ! -d "${checkpoint_dir}" ]]; then
        echo "Checkpoint directory does not exist: ${checkpoint_dir}"
        return 1
    fi
    
    # Get list of checkpoints sorted by modification time (oldest first)
    local checkpoints
    checkpoints=$(find "${checkpoint_dir}" -name "epoch_*.ckpt" -type f | sort -V)
    local total_count
    total_count=$(echo "${checkpoints}" | grep -c . || echo 0)
    
    if [[ ${total_count} -le ${keep_count} ]]; then
        echo "Found ${total_count} checkpoints, keeping all (threshold: ${keep_count})"
        return 0
    fi
    
    local remove_count=$((total_count - keep_count))
    echo "Removing ${remove_count} old checkpoints, keeping ${keep_count}"
    
    echo "${checkpoints}" | head -n "${remove_count}" | while read -r ckpt; do
        echo "  Removing: ${ckpt}"
        rm -f "${ckpt}"
    done
}

#-------------------------------------------------------------------------------
# print_gpu_info - Print GPU information for debugging
#-------------------------------------------------------------------------------
print_gpu_info() {
    echo "================================================================================
GPU Information
================================================================================"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi query failed"
    else
        echo "nvidia-smi not available"
    fi
    echo "================================================================================"
}
