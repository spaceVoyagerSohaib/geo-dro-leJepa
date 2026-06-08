#!/bin/bash
#===============================================================================
# GeoDRO-LeJEPA ImageNet-100 Controlled Shape Check
#
# This is a short Slurm job for testing one candidate compute shape before
# committing to a full 400-epoch run. It runs a few training steps and exits.
#
# Example:
#   sbatch -p mcml-hgx-a100-80x4 --qos=mcml --nodes=1 \
#     --export=ALL,BATCH_SIZE=512,SHAPE_CHECK_STEPS=20 \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_shape_check.sbatch
#
# Optimizer-step accumulation example:
#   sbatch -p mcml-hgx-a100-80x4 --qos=mcml --nodes=1 \
#     --export=ALL,CONFIG_NAME=geodro/geodro_lejepa_imagenet100ctrl_v1_optstep_accum,BATCH_SIZE=256,ACCUM_GRAD_BATCHES=2,SHAPE_CHECK_STEPS=20 \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_shape_check.sbatch
#===============================================================================

#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
#SBATCH --job-name=geodro-im100ctrl-shape
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-4}}}}"
TASKS_PER_NODE="${TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-${GPUS_PER_NODE}}}"
TRAINER_DEVICES="${TRAINER_DEVICES:-${GPUS_PER_NODE}}"
SHAPE_CHECK_STEPS="${SHAPE_CHECK_STEPS:-20}"
CONFIG_NAME="${CONFIG_NAME:-geodro/geodro_lejepa_imagenet100ctrl_v1}"
ADVERSARY_SCOPE="${ADVERSARY_SCOPE:-}"
if [[ -z "${ADVERSARY_SCOPE}" ]]; then
    if [[ "${CONFIG_NAME}" == *"optstep"* ]]; then
        ADVERSARY_SCOPE="optimizer_step"
    else
        ADVERSARY_SCOPE="microbatch"
    fi
fi
if [[ "${ADVERSARY_SCOPE}" == "optimizer_step" ]]; then
    DEFAULT_BATCH_SIZE=128
    DEFAULT_ACCUM_GRAD_BATCHES=4
else
    DEFAULT_BATCH_SIZE=512
    DEFAULT_ACCUM_GRAD_BATCHES=1
fi
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-${DEFAULT_ACCUM_GRAD_BATCHES}}"

RUN_ID_BASE="${CONFIG_NAME##*/}"
RUN_ID="shapecheck-${RUN_ID_BASE}-${NUM_NODES}n-b${BATCH_SIZE}-${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
if [[ "${ADVERSARY_SCOPE}" == "optimizer_step" ]]; then
    GRAPH_BATCH=$((BATCH_SIZE * NUM_NODES * GPUS_PER_NODE * ACCUM_GRAD_BATCHES))
else
    GRAPH_BATCH=$((BATCH_SIZE * NUM_NODES * GPUS_PER_NODE))
fi

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
if [[ -n "${SLURM_NNODES:-}" && "${NUM_NODES}" -ne "${SLURM_NNODES}" ]]; then
    echo "ERROR: NUM_NODES (${NUM_NODES}) does not match Slurm nodes (${SLURM_NNODES})."
    echo "Set --nodes and NUM_NODES to the same value, or omit NUM_NODES."
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

export GEODRO_CONFIG_NAME="${CONFIG_NAME}"
export GEODRO_RUN_ID="${RUN_ID}"
export GEODRO_BATCH_SIZE="${BATCH_SIZE}"
export GEODRO_VAL_BATCH_SIZE="${VAL_BATCH_SIZE}"
export GEODRO_MAX_STEPS="${SHAPE_CHECK_STEPS}"
export GEODRO_ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES}"
export GEODRO_GRAPH_BATCH="${GRAPH_BATCH}"
export GEODRO_ADVERSARY_SCOPE="${ADVERSARY_SCOPE}"

echo "============================================================"
echo "GeoDRO-LeJEPA ImageNet-100 Controlled Shape Check"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Config:        ${CONFIG_NAME}"
echo "Nodes:         ${NUM_NODES}"
echo "GPUs/Node:     ${GPUS_PER_NODE}"
echo "Total GPUs:    $((NUM_NODES * GPUS_PER_NODE))"
echo "Batch/GPU:     ${BATCH_SIZE}"
echo "Graph Batch:   ${GRAPH_BATCH}"
echo "Grad Accum:    ${ACCUM_GRAD_BATCHES}"
echo "Adv Scope:     ${ADVERSARY_SCOPE}"
echo "Max Steps:     ${SHAPE_CHECK_STEPS}"
echo "Output Dir:    ${OUTPUT_DIR}"
echo "HF_HOME:       ${HF_HOME}"
echo "HF datasets:   ${HF_DATASETS_CACHE}"
echo "MPL config:    ${MPLCONFIGDIR}"
echo "============================================================"

source ~/.bashrc
mamba activate py312

if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
    validate_env py312 || exit 1
    print_gpu_info
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
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

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
        | tee "${LOG_DIR}/nvidia_smi_before.csv" || true
fi

COMMON_OVERRIDES=(
    trainer.num_nodes="${NUM_NODES}"
    trainer.devices="${TRAINER_DEVICES}"
    trainer.strategy=ddp
    trainer.accumulate_grad_batches="${ACCUM_GRAD_BATCHES}"
    data.train.batch_size="${BATCH_SIZE}"
    data.val.batch_size="${VAL_BATCH_SIZE}"
    metadata.adversary_scope="${ADVERSARY_SCOPE}"
    metadata.local_batch_size="${BATCH_SIZE}"
    metadata.global_graph_batch_size="${GRAPH_BATCH}"
)

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running short shape check..."
echo ""
srun --kill-on-bad-exit=1 python -m stable_pretraining.run \
    --config-path "${PROJECT_ROOT}/examples" \
    --config-name "${CONFIG_NAME}" \
    "${COMMON_OVERRIDES[@]}" \
    +trainer.max_steps="${SHAPE_CHECK_STEPS}" \
    trainer.max_epochs=1 \
    trainer.logger=false \
    trainer.callbacks=[] \
    trainer.enable_checkpointing=false \
    hydra.run.dir="${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/shape_check.log"

RUN_EXIT_CODE=${PIPESTATUS[0]}
if [[ ${RUN_EXIT_CODE} -ne 0 ]]; then
    echo "ERROR: Shape check failed with exit code ${RUN_EXIT_CODE}"
    exit ${RUN_EXIT_CODE}
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
        | tee "${LOG_DIR}/nvidia_smi_after.csv" || true
fi

cat > "${OUTPUT_DIR}/shape_check_summary.txt" << EOF
GeoDRO-LeJEPA ImageNet-100 controlled shape check passed.
Config: ${CONFIG_NAME}
Nodes: ${NUM_NODES}
GPUs per node: ${GPUS_PER_NODE}
Batch per GPU: ${BATCH_SIZE}
Graph batch: ${GRAPH_BATCH}
Gradient accumulation: ${ACCUM_GRAD_BATCHES}
Adversary scope: ${ADVERSARY_SCOPE}
Steps: ${SHAPE_CHECK_STEPS}
Output: ${OUTPUT_DIR}
EOF

echo ""
echo "============================================================"
echo "GeoDRO ImageNet-100 controlled shape check PASSED"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}"
echo "Log:    ${LOG_DIR}/shape_check.log"
echo "============================================================"
