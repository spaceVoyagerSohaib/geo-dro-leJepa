#!/bin/bash
#===============================================================================
# GeoDRO-LeJEPA ImageNet-Sketch Linear Evaluation
#
# Evaluates a pretrained checkpoint on ImageNet-Sketch (texture-bias OOD).
# Reuses the IN-100 train split for probe training and IN-1k -> IN-100 label
# remapping; only samples whose IN-1k class falls within IN-100 are scored.
#
# Run after pretraining:
#   sbatch -p mcml-hgx-a100-80x4 --qos=mcml \
#     --export=ALL,PRETRAIN_RUN_ID=<run-id>,PRETRAIN_OUTPUT_DIR=<output-dir>,METHOD=<erm-or-v1> \
#     scripts/slurm/geodro_lejepa_imagenet_sketch_eval.sbatch
#===============================================================================

#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
#SBATCH --job-name=geodro-imsketch-eval
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

MODE="imagenet_sketch"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-}"
METHOD="${METHOD:-${PRETRAIN_RUN_ID:-unknown}}"

EVAL_EPOCHS="${EVAL_EPOCHS:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_LR="${EVAL_LR:-1e-3}"
EVAL_WEIGHT_DECAY="${EVAL_WEIGHT_DECAY:-1e-6}"
EVAL_WARMUP_EPOCHS="${EVAL_WARMUP_EPOCHS:-10}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
EVAL_IMG_SIZE="${EVAL_IMG_SIZE:-224}"
EVAL_FEATURE_LAYERS="${EVAL_FEATURE_LAYERS:--1 -2}"
KNN_K="${KNN_K:-20}"

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
    setup_geodro_dataset_cache imagenet100 || exit 1
    setup_geodro_dataset_cache imagenet_sketch || exit 1
    IMAGENET100_CACHE_DIR="$(geodro_dataset_cache_dir imagenet100)"
    IMAGENET_SKETCH_CACHE_DIR="$(geodro_dataset_cache_dir imagenet_sketch)"
fi

source ~/.bashrc
mamba activate py312

if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
    validate_env py312 || exit 1
    require_wandb_key || exit 1
fi

if [[ -z "${PRETRAIN_OUTPUT_DIR}" && -n "${PRETRAIN_RUN_ID}" ]]; then
    PRETRAIN_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${PRETRAIN_RUN_ID}"
fi
if [[ -z "${CHECKPOINT_PATH}" && -n "${PRETRAIN_OUTPUT_DIR}" ]]; then
    CHECKPOINT_PATH="$(find_checkpoint "${PRETRAIN_OUTPUT_DIR}")"
fi
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: Checkpoint not found. Set CHECKPOINT_PATH or PRETRAIN_OUTPUT_DIR/PRETRAIN_RUN_ID."
    exit 1
fi
if [[ -z "${PRETRAIN_OUTPUT_DIR}" ]]; then
    PRETRAIN_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${PRETRAIN_RUN_ID:-manual-eval}"
fi

RUN_NAME="${PRETRAIN_RUN_ID:-manual}-${MODE}-${SLURM_JOB_ID:-local}"
EVAL_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR}/eval/${MODE}-${SLURM_JOB_ID:-local}"
LOG_DIR="${PRETRAIN_OUTPUT_DIR}/logs"
mkdir -p "${EVAL_OUTPUT_DIR}" "${LOG_DIR}"

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ssl-geo-dro}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-${PRETRAIN_RUN_ID:-${RUN_NAME}}}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "============================================================"
echo "GeoDRO-LeJEPA ImageNet-Sketch Eval"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output:     ${EVAL_OUTPUT_DIR}"
echo "Run ID:     ${PRETRAIN_RUN_ID:-manual}"
echo "Method:     ${METHOD}"
echo "Epochs:     ${EVAL_EPOCHS}"
echo "============================================================"

cd "${PROJECT_ROOT}"
python -m scripts.geodro_lejepa_eval \
    --mode "${MODE}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --pretrain_run_id "${PRETRAIN_RUN_ID:-manual}" \
    --method "${METHOD}" \
    --epochs "${EVAL_EPOCHS}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --lr "${EVAL_LR}" \
    --weight_decay "${EVAL_WEIGHT_DECAY}" \
    --warmup_epochs "${EVAL_WARMUP_EPOCHS}" \
    --img_size "${EVAL_IMG_SIZE}" \
    --feature_layers ${EVAL_FEATURE_LAYERS} \
    --num_workers "${EVAL_NUM_WORKERS}" \
    --knn_k "${KNN_K}" \
    --imagenet100_cache_dir "${IMAGENET100_CACHE_DIR}" \
    --imagenet_sketch_cache_dir "${IMAGENET_SKETCH_CACHE_DIR}" \
    --wandb_project "${WANDB_PROJECT_NAME}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_mode "${WANDB_MODE}" \
    2>&1 | tee "${LOG_DIR}/${MODE}_eval-${SLURM_JOB_ID:-local}.log"
