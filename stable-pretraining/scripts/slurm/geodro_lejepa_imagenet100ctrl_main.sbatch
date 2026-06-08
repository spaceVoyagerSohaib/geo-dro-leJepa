#!/bin/bash
#===============================================================================
# GeoDRO-LeJEPA ImageNet-100 Controlled Main Run
#
# Usage:
#   sbatch scripts/slurm/geodro_lejepa_imagenet100ctrl_main.sbatch
#   sbatch --export=ALL,CONFIG_NAME=geodro/geodro_lejepa_imagenet100ctrl_erm \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_main.sbatch
#
# For 400-epoch FULL pretraining (48h, LRZ H100), use instead:
#   scripts/slurm/geodro_lejepa_imagenet100ctrl_full.sbatch
#
# Continuation (resume from a prior checkpoint):
#   sbatch --export=ALL,CONFIG_NAME=...,RUN_LABEL=...,RUN_PREFLIGHT=0,\
#     RESUME_CKPT_PATH=/abs/path/to/last.ckpt,... \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_main.sbatch
#
# Resume env vars:
#   RESUME_CKPT_PATH      - checkpoint path (.ckpt optional; relative to project root)
#   RESUME_WEIGHTS_ONLY   - false (default) for full optimizer/LR/probe resume
#   AUTO_RESUME_FALLBACK  - true (default): retry with weights-only on state_dict mismatch
#
# Runs one full ImageNet-100 controlled pretraining variant. Before the full
# training starts, the same Slurm job runs a tiny 2-step memory/config check at
# the target batch size. If that check fails, the job stops instead of silently
# switching to naive gradient accumulation.
#
# Optional experiment-label and GeoDRO tuning overrides:
#   RUN_LABEL, GEODRO_K, GRAPH_MODE, INNER_STEPS, TAU_FLOW, BETA, ALPHA_MAX,
#   WARMUP_FRACTION, RAMP_FRACTION, CLAMP_ACTIVATION_FAIL,
#   ESS_MIN_RATIO, MAX_P_FACTOR_FAIL, P_CAP
#   MEMORY_UPDATE_SCOPE, MEMORY_QUEUE_CAPACITY, MEMORY_TOP_M,
#   MEMORY_K_SIGMA, MEMORY_K_GUARD, MEMORY_RETRIEVAL_CHUNK_SIZE,
#   MEMORY_WITNESS_THRESHOLD_MODE, MEMORY_WITNESS_NULL_QUANTILE,
#   MEMORY_WITNESS_SCORE_MIN, MEMORY_EXTRA_EDGES_PER_NODE_MAX,
#   MEMORY_ADDED_EDGE_RATIO_MAX, UTILITY_SMOOTHING_GRAPH,
#   WITNESS_SCORE_MODE, MEMORY_WITNESS_ABLATION_MODE,
#   MEMORY_WITNESS_NULL_SEED
#===============================================================================

#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm-bootstrap-%j.out
#SBATCH -e logs/slurm-bootstrap-%j.err
#SBATCH --job-name=geodro-im100ctrl-main
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-2}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-4}}}}"
TASKS_PER_NODE="${TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-${GPUS_PER_NODE}}}"
TRAINER_DEVICES="${TRAINER_DEVICES:-${GPUS_PER_NODE}}"
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
    DEFAULT_BATCH_SIZE=256
    DEFAULT_ACCUM_GRAD_BATCHES=1
fi
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-${DEFAULT_ACCUM_GRAD_BATCHES}}"
RUN_LABEL="${RUN_LABEL:-}"
GEODRO_K="${GEODRO_K:-}"
GRAPH_MODE="${GRAPH_MODE:-}"
INNER_STEPS="${INNER_STEPS:-}"
TAU_FLOW="${TAU_FLOW:-}"
BETA="${BETA:-}"
ALPHA_MAX="${ALPHA_MAX:-}"
WARMUP_FRACTION="${WARMUP_FRACTION:-}"
RAMP_FRACTION="${RAMP_FRACTION:-}"
CLAMP_ACTIVATION_FAIL="${CLAMP_ACTIVATION_FAIL:-}"
ESS_MIN_RATIO="${ESS_MIN_RATIO:-}"
MAX_P_FACTOR_FAIL="${MAX_P_FACTOR_FAIL:-}"
P_CAP="${P_CAP:-}"
MEMORY_UPDATE_SCOPE="${MEMORY_UPDATE_SCOPE:-}"
MEMORY_QUEUE_CAPACITY="${MEMORY_QUEUE_CAPACITY:-}"
MEMORY_TOP_M="${MEMORY_TOP_M:-}"
MEMORY_K_SIGMA="${MEMORY_K_SIGMA:-}"
MEMORY_K_GUARD="${MEMORY_K_GUARD:-}"
MEMORY_RETRIEVAL_CHUNK_SIZE="${MEMORY_RETRIEVAL_CHUNK_SIZE:-}"
MEMORY_WITNESS_THRESHOLD_MODE="${MEMORY_WITNESS_THRESHOLD_MODE:-}"
MEMORY_WITNESS_NULL_QUANTILE="${MEMORY_WITNESS_NULL_QUANTILE:-}"
MEMORY_WITNESS_SCORE_MIN="${MEMORY_WITNESS_SCORE_MIN:-}"
MEMORY_EXTRA_EDGES_PER_NODE_MAX="${MEMORY_EXTRA_EDGES_PER_NODE_MAX:-}"
MEMORY_ADDED_EDGE_RATIO_MAX="${MEMORY_ADDED_EDGE_RATIO_MAX:-}"
UTILITY_SMOOTHING_GRAPH="${UTILITY_SMOOTHING_GRAPH:-}"
WITNESS_SCORE_MODE="${WITNESS_SCORE_MODE:-}"
MEMORY_WITNESS_ABLATION_MODE="${MEMORY_WITNESS_ABLATION_MODE:-}"
MEMORY_WITNESS_NULL_SEED="${MEMORY_WITNESS_NULL_SEED:-}"
TRAINER_PRECISION="${TRAINER_PRECISION:-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-2}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
RESUME_WEIGHTS_ONLY="${RESUME_WEIGHTS_ONLY:-false}"
AUTO_RESUME_FALLBACK="${AUTO_RESUME_FALLBACK:-true}"
RESUME_EPOCH=""

normalize_bool() {
    local raw="${1:-false}"
    case "${raw,,}" in
        1|true|yes|y|on) echo "true" ;;
        0|false|no|n|off|"") echo "false" ;;
        *)
            echo "ERROR: Invalid boolean value '${raw}'. Use one of: true/false/1/0/yes/no"
            exit 1
            ;;
    esac
}

is_resume_compat_error() {
    local log_file="$1"
    grep -Eq "Error\\(s\\) in loading state_dict|size mismatch for|Missing key\\(s\\) in state_dict|Unexpected key\\(s\\) in state_dict" "${log_file}"
}

RUN_ID_BASE="${CONFIG_NAME##*/}"
if [[ -n "${RUN_LABEL}" ]]; then
    RUN_ID="${RUN_LABEL}-${SLURM_JOB_ID:-local}"
else
    RUN_ID="${RUN_ID_BASE}-${SLURM_JOB_ID:-local}"
fi
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
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_PROJECT

export GEODRO_CONFIG_NAME="${CONFIG_NAME}"
export GEODRO_RUN_ID="${RUN_ID}"
export GEODRO_RUN_LABEL="${RUN_LABEL}"
export GEODRO_BATCH_SIZE="${BATCH_SIZE}"
export GEODRO_VAL_BATCH_SIZE="${VAL_BATCH_SIZE}"
export GEODRO_MAX_EPOCHS="${MAX_EPOCHS}"
export GEODRO_ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES}"
export GEODRO_GRAPH_BATCH="${GRAPH_BATCH}"
export GEODRO_ADVERSARY_SCOPE="${ADVERSARY_SCOPE}"
export GEODRO_MEMORY_UPDATE_SCOPE="${MEMORY_UPDATE_SCOPE}"
export GEODRO_PREFLIGHT_ENABLED="${RUN_PREFLIGHT}"
export GEODRO_PREFLIGHT_STEPS="${PREFLIGHT_STEPS}"
export GEODRO_K_OVERRIDE="${GEODRO_K}"
export GEODRO_GRAPH_MODE="${GRAPH_MODE}"
export GEODRO_INNER_STEPS="${INNER_STEPS}"
export GEODRO_TAU_FLOW="${TAU_FLOW}"
export GEODRO_BETA="${BETA}"
export GEODRO_ALPHA_MAX="${ALPHA_MAX}"
export GEODRO_WARMUP_FRACTION="${WARMUP_FRACTION}"
export GEODRO_RAMP_FRACTION="${RAMP_FRACTION}"
export GEODRO_CLAMP_ACTIVATION_FAIL="${CLAMP_ACTIVATION_FAIL}"
export GEODRO_ESS_MIN_RATIO="${ESS_MIN_RATIO}"
export GEODRO_MAX_P_FACTOR_FAIL="${MAX_P_FACTOR_FAIL}"
export GEODRO_P_CAP="${P_CAP}"
export GEODRO_MEMORY_QUEUE_CAPACITY="${MEMORY_QUEUE_CAPACITY}"
export GEODRO_MEMORY_TOP_M="${MEMORY_TOP_M}"
export GEODRO_MEMORY_K_SIGMA="${MEMORY_K_SIGMA}"
export GEODRO_MEMORY_K_GUARD="${MEMORY_K_GUARD}"
export GEODRO_MEMORY_RETRIEVAL_CHUNK_SIZE="${MEMORY_RETRIEVAL_CHUNK_SIZE}"
export GEODRO_MEMORY_WITNESS_THRESHOLD_MODE="${MEMORY_WITNESS_THRESHOLD_MODE}"
export GEODRO_MEMORY_WITNESS_NULL_QUANTILE="${MEMORY_WITNESS_NULL_QUANTILE}"
export GEODRO_MEMORY_WITNESS_SCORE_MIN="${MEMORY_WITNESS_SCORE_MIN}"
export GEODRO_MEMORY_EXTRA_EDGES_PER_NODE_MAX="${MEMORY_EXTRA_EDGES_PER_NODE_MAX}"
export GEODRO_MEMORY_ADDED_EDGE_RATIO_MAX="${MEMORY_ADDED_EDGE_RATIO_MAX}"
export GEODRO_UTILITY_SMOOTHING_GRAPH="${UTILITY_SMOOTHING_GRAPH}"
export GEODRO_WITNESS_SCORE_MODE="${WITNESS_SCORE_MODE}"
export GEODRO_MEMORY_WITNESS_ABLATION_MODE="${MEMORY_WITNESS_ABLATION_MODE}"
export GEODRO_MEMORY_WITNESS_NULL_SEED="${MEMORY_WITNESS_NULL_SEED}"

echo "============================================================"
echo "GeoDRO-LeJEPA ImageNet-100 Controlled Main"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Config:        ${CONFIG_NAME}"
echo "Run Label:     ${RUN_LABEL:-<auto>}"
echo "Nodes:         ${NUM_NODES}"
echo "GPUs/Node:     ${GPUS_PER_NODE}"
echo "Total GPUs:    $((NUM_NODES * GPUS_PER_NODE))"
echo "Batch/GPU:     ${BATCH_SIZE}"
echo "Graph Batch:   ${GRAPH_BATCH}"
echo "Grad Accum:    ${ACCUM_GRAD_BATCHES}"
echo "Adv Scope:     ${ADVERSARY_SCOPE}"
echo "Memory Clock:  ${MEMORY_UPDATE_SCOPE:-<config>}"
echo "GEODRO_K:      ${GEODRO_K:-<config>}"
echo "Graph Mode:    ${GRAPH_MODE:-<config>}"
echo "Inner Steps:   ${INNER_STEPS:-<config>}"
echo "Tau Flow:      ${TAU_FLOW:-<config>}"
echo "Beta:          ${BETA:-<config>}"
echo "Alpha Max:     ${ALPHA_MAX:-<config>}"
echo "Warmup Frac:   ${WARMUP_FRACTION:-<config>}"
echo "Ramp Frac:     ${RAMP_FRACTION:-<config>}"
echo "Clamp Fail:    ${CLAMP_ACTIVATION_FAIL:-<config>}"
echo "ESS Min:       ${ESS_MIN_RATIO:-<config>}"
echo "MaxP Factor:   ${MAX_P_FACTOR_FAIL:-<config>}"
echo "P Cap:         ${P_CAP:-<config>}"
echo "Memory Queue:  ${MEMORY_QUEUE_CAPACITY:-<config>}"
echo "Memory top_m:  ${MEMORY_TOP_M:-<config>}"
echo "Memory k_sig:  ${MEMORY_K_SIGMA:-<config>}"
echo "Memory Kguard: ${MEMORY_K_GUARD:-<config>}"
echo "Witness Thres: ${MEMORY_WITNESS_THRESHOLD_MODE:-<config>}"
echo "Witness q:     ${MEMORY_WITNESS_NULL_QUANTILE:-<config>}"
echo "Witness min:   ${MEMORY_WITNESS_SCORE_MIN:-<config>}"
echo "Memory Edges:  ${MEMORY_EXTRA_EDGES_PER_NODE_MAX:-<config>} / ${MEMORY_ADDED_EDGE_RATIO_MAX:-<config>}"
echo "Witness Mode:  ${WITNESS_SCORE_MODE:-<config>}"
echo "Memory Ablate: ${MEMORY_WITNESS_ABLATION_MODE:-<config>}"
echo "Precision:     ${TRAINER_PRECISION:-<config>}"
echo "Max Epochs:    ${MAX_EPOCHS}"
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
    echo "Resume Ckpt:   ${RESUME_CKPT_PATH}"
    echo "Resume Epoch:  ${RESUME_EPOCH}"
    echo "Resume Mode:   weights_only=${RESUME_WEIGHTS_ONLY}"
    echo "Auto Fallback: ${AUTO_RESUME_FALLBACK}"
fi
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

RESUME_WEIGHTS_ONLY="$(normalize_bool "${RESUME_WEIGHTS_ONLY}")"
AUTO_RESUME_FALLBACK="$(normalize_bool "${AUTO_RESUME_FALLBACK}")"
RESUME_EPOCH=""
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
    if [[ "${RESUME_CKPT_PATH}" != *.ckpt ]]; then
        RESUME_CKPT_PATH="${RESUME_CKPT_PATH}.ckpt"
    fi
    if [[ "${RESUME_CKPT_PATH}" != /* ]]; then
        RESUME_CKPT_PATH="${PROJECT_ROOT}/${RESUME_CKPT_PATH}"
    fi
    RESUME_CKPT_PATH="$(python - "${RESUME_CKPT_PATH}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
    if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
        echo "ERROR: RESUME_CKPT_PATH specified but not found: ${RESUME_CKPT_PATH}"
        exit 1
    fi
    RESUME_EPOCH="$(python - "${RESUME_CKPT_PATH}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(path, map_location="cpu")
print(int(ckpt.get("epoch", 0)))
PY
)"
    if [[ "${RUN_PREFLIGHT}" == "1" && -z "${FORCE_PREFLIGHT_ON_RESUME:-}" ]]; then
        RUN_PREFLIGHT=0
        echo "Auto-disabled preflight for continuation (set FORCE_PREFLIGHT_ON_RESUME=1 to override)."
    fi
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
if [[ -n "${GEODRO_K}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.k="${GEODRO_K}")
fi
if [[ -n "${GRAPH_MODE}" ]]; then
    COMMON_OVERRIDES+=(metadata.graph_mode="${GRAPH_MODE}")
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.graph_mode="${GRAPH_MODE}")
fi
if [[ -n "${INNER_STEPS}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.inner_steps="${INNER_STEPS}")
fi
if [[ -n "${TAU_FLOW}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.tau_flow="${TAU_FLOW}")
fi
if [[ -n "${BETA}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.beta="${BETA}")
fi
if [[ -n "${ALPHA_MAX}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.alpha_max="${ALPHA_MAX}")
fi
if [[ -n "${WARMUP_FRACTION}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.warmup_fraction="${WARMUP_FRACTION}")
fi
if [[ -n "${RAMP_FRACTION}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.ramp_fraction="${RAMP_FRACTION}")
fi
if [[ -n "${CLAMP_ACTIVATION_FAIL}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.clamp_activation_fail="${CLAMP_ACTIVATION_FAIL}")
fi
if [[ -n "${ESS_MIN_RATIO}" ]]; then
    COMMON_OVERRIDES+=(module.geodro_lejepa_loss.ess_min_ratio="${ESS_MIN_RATIO}")
fi
if [[ -n "${MAX_P_FACTOR_FAIL}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.max_p_factor_fail="${MAX_P_FACTOR_FAIL}")
fi
if [[ -n "${P_CAP}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.p_cap="${P_CAP}")
fi
if [[ -n "${MEMORY_UPDATE_SCOPE}" ]]; then
    COMMON_OVERRIDES+=(++metadata.memory_update_scope="${MEMORY_UPDATE_SCOPE}")
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_update_scope="${MEMORY_UPDATE_SCOPE}")
fi
if [[ -n "${MEMORY_QUEUE_CAPACITY}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_queue_capacity="${MEMORY_QUEUE_CAPACITY}")
fi
if [[ -n "${MEMORY_TOP_M}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_top_m="${MEMORY_TOP_M}")
fi
if [[ -n "${MEMORY_K_SIGMA}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_k_sigma="${MEMORY_K_SIGMA}")
fi
if [[ -n "${MEMORY_K_GUARD}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_k_guard="${MEMORY_K_GUARD}")
fi
if [[ -n "${MEMORY_RETRIEVAL_CHUNK_SIZE}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_retrieval_chunk_size="${MEMORY_RETRIEVAL_CHUNK_SIZE}")
fi
if [[ -n "${MEMORY_WITNESS_THRESHOLD_MODE}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_threshold_mode="${MEMORY_WITNESS_THRESHOLD_MODE}")
fi
if [[ -n "${MEMORY_WITNESS_NULL_QUANTILE}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_null_quantile="${MEMORY_WITNESS_NULL_QUANTILE}")
fi
if [[ -n "${MEMORY_WITNESS_SCORE_MIN}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_score_min="${MEMORY_WITNESS_SCORE_MIN}")
fi
if [[ -n "${MEMORY_EXTRA_EDGES_PER_NODE_MAX}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_extra_edges_per_node_max="${MEMORY_EXTRA_EDGES_PER_NODE_MAX}")
fi
if [[ -n "${MEMORY_ADDED_EDGE_RATIO_MAX}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_added_edge_ratio_max="${MEMORY_ADDED_EDGE_RATIO_MAX}")
fi
if [[ -n "${UTILITY_SMOOTHING_GRAPH}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.utility_smoothing_graph="${UTILITY_SMOOTHING_GRAPH}")
fi
if [[ -n "${WITNESS_SCORE_MODE}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.witness_score_mode="${WITNESS_SCORE_MODE}")
fi
if [[ -n "${MEMORY_WITNESS_ABLATION_MODE}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_ablation_mode="${MEMORY_WITNESS_ABLATION_MODE}")
fi
if [[ -n "${MEMORY_WITNESS_NULL_SEED}" ]]; then
    COMMON_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_null_seed="${MEMORY_WITNESS_NULL_SEED}")
fi
if [[ -n "${TRAINER_PRECISION}" ]]; then
    COMMON_OVERRIDES+=(trainer.precision="${TRAINER_PRECISION}")
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running 2-step memory/config check..."
    echo ""
    srun --kill-on-bad-exit=1 python -m stable_pretraining.run \
        --config-path "${PROJECT_ROOT}/examples" \
        --config-name "${CONFIG_NAME}" \
        "${COMMON_OVERRIDES[@]}" \
        +trainer.max_steps="${PREFLIGHT_STEPS}" \
        trainer.max_epochs=1 \
        trainer.logger=false \
        trainer.callbacks=[] \
        trainer.enable_checkpointing=false \
        hydra.run.dir="${OUTPUT_DIR}/preflight" \
        2>&1 | tee "${LOG_DIR}/preflight.log"
    PREFLIGHT_EXIT_CODE=${PIPESTATUS[0]}
    if [[ ${PREFLIGHT_EXIT_CODE} -ne 0 ]]; then
        echo "ERROR: The 2-step memory/config check failed with exit code ${PREFLIGHT_EXIT_CODE}."
        exit ${PREFLIGHT_EXIT_CODE}
    fi
fi

PRETRAIN_LOG_FILE="${LOG_DIR}/pretrain.log"

run_pretraining() {
    local weights_only="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running pretraining with resume_weights_only=${weights_only}"
    local -a resume_args=()
    if [[ -n "${RESUME_CKPT_PATH}" ]]; then
        resume_args=(
            ckpt_path="${RESUME_CKPT_PATH}"
            "+resume_weights_only=${weights_only}"
        )
    fi
    srun --kill-on-bad-exit=1 python -m stable_pretraining.run \
        --config-path "${PROJECT_ROOT}/examples" \
        --config-name "${CONFIG_NAME}" \
        "${COMMON_OVERRIDES[@]}" \
        trainer.max_epochs="${MAX_EPOCHS}" \
        trainer.logger.project="${WANDB_PROJECT_NAME}" \
        trainer.logger.name="${WANDB_RUN_NAME}" \
        "${resume_args[@]}" \
        hydra.run.dir="${OUTPUT_DIR}" \
        2>&1 | tee -a "${PRETRAIN_LOG_FILE}"
    return "${PIPESTATUS[0]}"
}

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting full pretraining run..."
echo ""
: > "${PRETRAIN_LOG_FILE}"
RUN_EXIT_CODE=0
set +e
run_pretraining "${RESUME_WEIGHTS_ONLY}"
RUN_EXIT_CODE=$?
set -e
if [[ ${RUN_EXIT_CODE} -ne 0 && "${AUTO_RESUME_FALLBACK}" == "true" && "${RESUME_WEIGHTS_ONLY}" == "false" && -n "${RESUME_CKPT_PATH}" ]] && \
   is_resume_compat_error "${PRETRAIN_LOG_FILE}"; then
    echo "WARNING: Detected checkpoint compatibility issue. Retrying with resume_weights_only=true."
    echo "WARNING: Fallback will resume model weights but may skip optimizer/callback/trainer runtime state."
    set +e
    run_pretraining "true"
    RUN_EXIT_CODE=$?
    set -e
fi

if [[ ${RUN_EXIT_CODE} -ne 0 ]]; then
    echo "ERROR: ImageNet-100 controlled pretraining failed with exit code ${RUN_EXIT_CODE}"
    exit ${RUN_EXIT_CODE}
fi

if [[ -n "${RESUME_CKPT_PATH}" ]]; then
    if grep -q "specified, but does not exist, using None for now" "${PRETRAIN_LOG_FILE}" || \
       grep -q "CALLING trainer.fit with ckpt_path=None" "${PRETRAIN_LOG_FILE}"; then
        echo "ERROR: Resume was requested but training fell back to scratch."
        echo "Inspect: ${PRETRAIN_LOG_FILE}"
        exit 1
    fi
    if ! grep -q "CALLING trainer.fit with ckpt_path='" "${PRETRAIN_LOG_FILE}"; then
        echo "WARNING: Could not find explicit ckpt_path confirmation in log."
        echo "Inspect: ${PRETRAIN_LOG_FILE}"
    fi
fi

echo ""
echo "============================================================"
echo "GeoDRO ImageNet-100 controlled run PASSED"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
