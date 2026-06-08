#!/bin/bash
#===============================================================================
# GeoDRO-LeJEPA ImageNet-100 controlled — V100 / small-GPU smoke launcher
#
# Adds Hydra overrides the stock main sbatch cannot pass (e.g. fp16, lighter
# GeoDRO head for smoke). Defaults target lrz-v100x2 (16GB, 2 GPUs/node).
#
# Usage:
#   cd stable-pretraining
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # optional
#   sbatch scripts/slurm/geodro_lejepa_imagenet100ctrl_v100_hydra_smoke.sbatch
#
# Override resources (example: MCML A100 full box — then raise BATCH_SIZE):
#   sbatch -p mcml-hgx-a100-80x4 --qos=mcml --nodes=2 --ntasks-per-node=4 \
#     --gres=gpu:4 --cpus-per-task=8 -t 4:00:00 \
#     --export=ALL,BATCH_SIZE=128,NUM_SLICES=1024,INNER_STEPS=10,TRAINER_PRECISION=bf16-mixed \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_v100_hydra_smoke.sbatch
#
# Env knobs (optional, with defaults):
#   CONFIG_NAME, NUM_NODES, GPUS_PER_NODE, TASKS_PER_NODE, TRAINER_DEVICES,
#   BATCH_SIZE, VAL_BATCH_SIZE, MAX_EPOCHS, ACCUM_GRAD_BATCHES,
#   TRAINER_PRECISION (16-mixed | bf16-mixed | 32-true),
#   NUM_SLICES, INNER_STEPS, NUM_WORKERS, RUN_PREFLIGHT, PREFLIGHT_STEPS,
#   WANDB_MODE, WANDB_PROJECT_NAME, WANDB_RUN_NAME
#===============================================================================

#SBATCH -p lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH -o logs/slurm-bootstrap-%j.out
#SBATCH -e logs/slurm-bootstrap-%j.err
#SBATCH --job-name=geodro-im100ctrl-v100-smoke
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-2}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-2}}}}"
TASKS_PER_NODE="${TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-${GPUS_PER_NODE}}}"
TRAINER_DEVICES="${TRAINER_DEVICES:-${GPUS_PER_NODE}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-1}"
CONFIG_NAME="${CONFIG_NAME:-geodro/geodro_lejepa_imagenet100ctrl_v1}"
ADVERSARY_SCOPE="${ADVERSARY_SCOPE:-}"
if [[ -z "${ADVERSARY_SCOPE}" ]]; then
  if [[ "${CONFIG_NAME}" == *"optstep"* ]]; then
    ADVERSARY_SCOPE="optimizer_step"
  else
    ADVERSARY_SCOPE="microbatch"
  fi
fi
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-2}"

TRAINER_PRECISION="${TRAINER_PRECISION:-16-mixed}"
NUM_SLICES="${NUM_SLICES:-256}"
INNER_STEPS="${INNER_STEPS:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_BASE="${CONFIG_NAME##*/}"
RUN_ID="${CONFIG_BASE}-v100-smoke-${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
SLURM_LOG_DIR="${PROJECT_ROOT}/logs"
SLURM_STDOUT_LOG="${SLURM_LOG_DIR}/slurm-${RUN_ID}.out"
SLURM_STDERR_LOG="${SLURM_LOG_DIR}/slurm-${RUN_ID}.err"
if [[ "${ADVERSARY_SCOPE}" == "optimizer_step" ]]; then
  GRAPH_BATCH=$((BATCH_SIZE * NUM_NODES * GPUS_PER_NODE * ACCUM_GRAD_BATCHES))
else
  GRAPH_BATCH=$((BATCH_SIZE * NUM_NODES * GPUS_PER_NODE))
fi

mkdir -p "${SLURM_LOG_DIR}"
echo "Redirecting Slurm stdout to ${SLURM_STDOUT_LOG}"
echo "Redirecting Slurm stderr to ${SLURM_STDERR_LOG}" >&2
exec >> "${SLURM_STDOUT_LOG}" 2>> "${SLURM_STDERR_LOG}"
echo "Slurm stdout log: ${SLURM_STDOUT_LOG}"
echo "Slurm stderr log: ${SLURM_STDERR_LOG}" >&2

ENV_FILE="${PROJECT_ROOT}/.env.local"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MCMLSCRATCH="${MCMLSCRATCH:-/dss/mcmlscratch/0D/ra59dut2}"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/utils.sh"
fi
if declare -F setup_geodro_dataset_cache >/dev/null; then
  setup_geodro_dataset_cache imagenet100 || exit 1
else
  export HF_HOME="${GEODRO_HF_HOME:-${MCMLSCRATCH}/datasets/geodro_v1/hf_home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MCMLSCRATCH}/datasets/geodro_v1/imagenet100/hf_cache}"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MCMLSCRATCH}/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${ADVERSARY_SCOPE}" != "microbatch" && "${ADVERSARY_SCOPE}" != "optimizer_step" ]]; then
  echo "ERROR: ADVERSARY_SCOPE must be microbatch or optimizer_step, got ${ADVERSARY_SCOPE}."
  exit 1
fi
if [[ "${ACCUM_GRAD_BATCHES}" != "1" && "${ADVERSARY_SCOPE}" != "optimizer_step" ]]; then
  echo "ERROR: ACCUM_GRAD_BATCHES>1 requires ADVERSARY_SCOPE=optimizer_step."
  echo "Naive gradient accumulation changes the microbatch adversary objective."
  exit 1
fi
if [[ "${TASKS_PER_NODE}" -ne "${GPUS_PER_NODE}" ]]; then
  echo "ERROR: TASKS_PER_NODE (${TASKS_PER_NODE}) must equal GPUS_PER_NODE (${GPUS_PER_NODE}) for DDP."
  exit 1
fi
if [[ "${TRAINER_DEVICES}" -ne "${GPUS_PER_NODE}" ]]; then
  echo "ERROR: TRAINER_DEVICES (${TRAINER_DEVICES}) must equal GPUS_PER_NODE (${GPUS_PER_NODE}) for DDP."
  exit 1
fi

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ssl-geo-dro}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset WANDB_PROJECT

export GEODRO_CONFIG_NAME="${CONFIG_NAME}"
export GEODRO_RUN_ID="${RUN_ID}"
export GEODRO_BATCH_SIZE="${BATCH_SIZE}"
export GEODRO_VAL_BATCH_SIZE="${VAL_BATCH_SIZE}"
export GEODRO_MAX_EPOCHS="${MAX_EPOCHS}"
export GEODRO_ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES}"
export GEODRO_GRAPH_BATCH="${GRAPH_BATCH}"
export GEODRO_ADVERSARY_SCOPE="${ADVERSARY_SCOPE}"
export GEODRO_PREFLIGHT_ENABLED="${RUN_PREFLIGHT}"
export GEODRO_PREFLIGHT_STEPS="${PREFLIGHT_STEPS}"

echo "============================================================"
echo "GeoDRO-LeJEPA ImageNet-100 — V100 Hydra smoke"
echo "============================================================"
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Config:           ${CONFIG_NAME}"
echo "Nodes:            ${NUM_NODES}"
echo "GPUs/Node:        ${GPUS_PER_NODE}"
echo "Total GPUs:       $((NUM_NODES * GPUS_PER_NODE))"
echo "Batch/GPU:        ${BATCH_SIZE}"
echo "Graph Batch:      ${GRAPH_BATCH}"
echo "Grad Accum:       ${ACCUM_GRAD_BATCHES}"
echo "Adv Scope:        ${ADVERSARY_SCOPE}"
echo "Trainer precision:${TRAINER_PRECISION}"
echo "num_slices:       ${NUM_SLICES}"
echo "inner_steps:      ${INNER_STEPS}"
echo "Data num_workers: ${NUM_WORKERS}"
echo "Max Epochs:       ${MAX_EPOCHS}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "PYTORCH_CUDA:     ${PYTORCH_CUDA_ALLOC_CONF}"
echo "W&B mode:         ${WANDB_MODE}"
echo "============================================================"

source ~/.bashrc
mamba activate py312

if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/utils.sh"
  validate_env py312 || exit 1
  require_wandb_key || exit 1
  print_gpu_info
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  export MASTER_ADDR="127.0.0.1"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  DEFAULT_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
else
  DEFAULT_MASTER_PORT=$((20000 + RANDOM % 20000))
fi
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if declare -F log_job_info >/dev/null; then
  log_job_info "${OUTPUT_DIR}"
fi

HYDRA_OVERRIDES=(
  trainer.num_nodes="${NUM_NODES}"
  trainer.devices="${TRAINER_DEVICES}"
  trainer.strategy=ddp
  trainer.accumulate_grad_batches="${ACCUM_GRAD_BATCHES}"
  trainer.precision="${TRAINER_PRECISION}"
  data.train.batch_size="${BATCH_SIZE}"
  data.val.batch_size="${VAL_BATCH_SIZE}"
  data.train.num_workers="${NUM_WORKERS}"
  data.val.num_workers="${NUM_WORKERS}"
  metadata.adversary_scope="${ADVERSARY_SCOPE}"
  metadata.local_batch_size="${BATCH_SIZE}"
  metadata.global_graph_batch_size="${GRAPH_BATCH}"
  module.geodro_lejepa_loss.num_slices="${NUM_SLICES}"
  module.geodro_lejepa_loss.inner_steps="${INNER_STEPS}"
)

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${PREFLIGHT_STEPS}-step memory/config check..."
  echo ""
  srun --kill-on-bad-exit=1 python -m stable_pretraining.run \
    --config-path "${PROJECT_ROOT}/examples" \
    --config-name "${CONFIG_NAME}" \
    "${HYDRA_OVERRIDES[@]}" \
    +trainer.max_steps="${PREFLIGHT_STEPS}" \
    trainer.max_epochs=1 \
    trainer.logger=false \
    trainer.callbacks=[] \
    trainer.enable_checkpointing=false \
    hydra.run.dir="${OUTPUT_DIR}/preflight" \
    2>&1 | tee "${LOG_DIR}/preflight.log"
  PREFLIGHT_EXIT_CODE=${PIPESTATUS[0]}
  if [[ ${PREFLIGHT_EXIT_CODE} -ne 0 ]]; then
    echo "ERROR: Preflight failed with exit code ${PREFLIGHT_EXIT_CODE}."
    exit "${PREFLIGHT_EXIT_CODE}"
  fi
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting smoke training (${MAX_EPOCHS} epoch(s))..."
echo ""
srun --kill-on-bad-exit=1 python -m stable_pretraining.run \
  --config-path "${PROJECT_ROOT}/examples" \
  --config-name "${CONFIG_NAME}" \
  "${HYDRA_OVERRIDES[@]}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.logger.project="${WANDB_PROJECT_NAME}" \
  trainer.logger.name="${WANDB_RUN_NAME}" \
  hydra.run.dir="${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/pretrain.log"

RUN_EXIT_CODE=${PIPESTATUS[0]}
if [[ ${RUN_EXIT_CODE} -ne 0 ]]; then
  echo "ERROR: Smoke training failed with exit code ${RUN_EXIT_CODE}"
  exit "${RUN_EXIT_CODE}"
fi

echo ""
echo "============================================================"
echo "GeoDRO V100 hydra smoke PASSED"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
