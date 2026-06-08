#!/bin/bash
#===============================================================================
# GeoDRO-LeJEPA CIFAR-10 Controlled Sanity Check
#
# Usage:
#   sbatch scripts/slurm/geodro_lejepa_cifar10ctrl_sanity.sbatch
#
# Runs a short single-GPU GeoDRO v1.1 debug job on the controlled CIFAR-10
# surface. This is an implementation sanity check, not a research run.
#===============================================================================

#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
#SBATCH --job-name=geodro-cifar10ctrl-sanity
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCHES="${VAL_BATCHES:-2}"
CONFIG_NAME="${CONFIG_NAME:-geodro_lejepa_cifar10ctrl_v1_debug}"
RUN_ID="geodro-v1.1-cifar10ctrl-conv32-sanity-${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_ID}"

ENV_FILE="${PROJECT_ROOT}/.env.local"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MCMLSCRATCH="${MCMLSCRATCH:-/dss/mcmlscratch/0D/ra59dut2}"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
fi
if declare -F setup_geodro_dataset_cache >/dev/null; then
    setup_geodro_dataset_cache cifar10 || exit 1
else
    export HF_HOME="${GEODRO_HF_HOME:-${MCMLSCRATCH}/datasets/geodro_v1/hf_home}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MCMLSCRATCH}/datasets/geodro_v1/cifar10/hf_cache}"
    mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
fi

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ssl-geo-dro}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}}"
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_PROJECT

echo "============================================================"
echo "GeoDRO-LeJEPA CIFAR-10 Controlled Sanity"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Config:        ${CONFIG_NAME}"
echo "Max Steps:     ${MAX_STEPS}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Output Dir:    ${OUTPUT_DIR}"
echo "W&B Project:   ${WANDB_PROJECT_NAME}"
echo "HF_HOME:       ${HF_HOME}"
echo "HF datasets:   ${HF_DATASETS_CACHE}"
echo "============================================================"

source ~/.bashrc
mamba activate py312

if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
    validate_env py312 || exit 1
    require_wandb_key || exit 1
    print_gpu_info
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs"

cd "${PROJECT_ROOT}"

if declare -F log_job_info >/dev/null; then
    log_job_info "${OUTPUT_DIR}"
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running GeoDRO CIFAR-10 controlled sanity..."
echo ""

python -m stable_pretraining.run \
    --config-path "${PROJECT_ROOT}/examples/geodro" \
    --config-name "${CONFIG_NAME}" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.max_epochs=1 \
    trainer.limit_val_batches="${VAL_BATCHES}" \
    trainer.logger.project="${WANDB_PROJECT_NAME}" \
    trainer.logger.name="${WANDB_RUN_NAME}" \
    trainer.logger.group="geodro-v1.1/cifar10ctrl/conv32/sanity" \
    data.train.batch_size="${BATCH_SIZE}" \
    data.val.batch_size="${BATCH_SIZE}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/pretrain.log"

RUN_EXIT_CODE=${PIPESTATUS[0]}
if [[ ${RUN_EXIT_CODE} -ne 0 ]]; then
    echo "ERROR: GeoDRO CIFAR-10 controlled sanity failed with exit code ${RUN_EXIT_CODE}"
    exit ${RUN_EXIT_CODE}
fi

CHECKPOINT=""
if declare -F find_checkpoint >/dev/null; then
    CHECKPOINT="$(find_checkpoint "${OUTPUT_DIR}")"
else
    CHECKPOINT="$(find "${OUTPUT_DIR}" -name "*.ckpt" -type f 2>/dev/null | head -n 1)"
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: No checkpoint found in ${OUTPUT_DIR}"
    exit 1
fi

echo ""
echo "============================================================"
echo "GeoDRO CIFAR-10 controlled sanity PASSED"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT_DIR}"
echo "============================================================"
