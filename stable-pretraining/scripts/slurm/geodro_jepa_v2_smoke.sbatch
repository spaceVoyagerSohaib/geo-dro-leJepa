#!/bin/bash
#===============================================================================
# GeoDRO-JEPA V2.2 — V100 ImageNet-100ctrl smoke launcher (Phase 13)
#
# Single parameterized launcher for the four V2.2 pre-pilot smoke cells:
#   - coherent_hardness + batch
#   - graph_transport   + batch
#   - coherent_hardness + batch_memory
#   - graph_transport   + batch_memory
#
# All variation is env-var driven; cell selection is via CONFIG_NAME.
# Defaults target lrz-v100x2 (16GB, 2 GPUs/node x 2 nodes). Override resources
# with normal sbatch flags or via env vars.
#
# Quickstart — submit all four cells in parallel from stable-pretraining/:
#
#   for cell in \
#     geodro_jepa_v2_coherent_hardness_imagenet100ctrl_smoke \
#     geodro_jepa_v2_graph_transport_imagenet100ctrl_smoke \
#     geodro_jepa_v2_coherent_hardness_batch_memory_imagenet100ctrl_smoke \
#     geodro_jepa_v2_graph_transport_batch_memory_imagenet100ctrl_smoke; do
#       sbatch --export=ALL,CONFIG_NAME=geodro/${cell},RUN_LABEL=${cell##*_v2_}-v100-smoke \
#         scripts/slurm/geodro_jepa_v2_smoke.sbatch
#   done
#
# Override examples:
#   # Sweep memory queue capacity
#   sbatch --export=ALL,CONFIG_NAME=geodro/geodro_jepa_v2_coherent_hardness_batch_memory_imagenet100ctrl_smoke,MEMORY_QUEUE_CAPACITY=1024,RUN_LABEL=ch-mem-cap1024 \
#     scripts/slurm/geodro_jepa_v2_smoke.sbatch
#
#   # Try explicit threshold instead of null-quantile
#   sbatch --export=ALL,CONFIG_NAME=geodro/geodro_jepa_v2_graph_transport_batch_memory_imagenet100ctrl_smoke,MEMORY_WITNESS_THRESHOLD_MODE=explicit,MEMORY_WITNESS_SCORE_MIN=0.05,RUN_LABEL=gt-mem-explicit \
#     scripts/slurm/geodro_jepa_v2_smoke.sbatch
#
#   # Move to MCML A100 with bigger batch
#   sbatch -p mcml-hgx-a100-80x4 --qos=mcml --nodes=1 --ntasks-per-node=4 \
#     --gres=gpu:4 --cpus-per-task=8 -t 2:00:00 \
#     --export=ALL,CONFIG_NAME=geodro/geodro_jepa_v2_coherent_hardness_batch_memory_imagenet100ctrl_smoke,BATCH_SIZE=128,TRAINER_PRECISION=bf16-mixed,RUN_LABEL=ch-mem-a100 \
#     scripts/slurm/geodro_jepa_v2_smoke.sbatch
#
# Env knobs (defaults in parens):
#   Selection:
#     CONFIG_NAME             — required; geodro/<one of the four smoke cells>
#                               or one of the V2 optstep memory configs.
#     RUN_LABEL               — descriptive label folded into output dir + W&B
#                               run name; defaults to CONFIG_NAME basename.
#   Shape:
#     NUM_NODES (2), GPUS_PER_NODE (2), TRAINER_DEVICES (=GPUS_PER_NODE),
#     TASKS_PER_NODE (=GPUS_PER_NODE),
#     BATCH_SIZE (64), VAL_BATCH_SIZE (=BATCH_SIZE),
#     MAX_EPOCHS (1), ACCUM_GRAD_BATCHES (1),
#     ADVERSARY_SCOPE (microbatch),
#     MEMORY_UPDATE_SCOPE (microbatch for microbatch scope,
#       optimizer_step_delayed for optimizer_step scope),
#     TRAINER_PRECISION (16-mixed),
#     NUM_SLICES (256), INNER_STEPS (5), NUM_WORKERS (2),
#     RUN_PREFLIGHT (1), PREFLIGHT_STEPS (2).
#   GeoDRO core (carry-over from V1 launcher; empty = config default):
#     GEODRO_K, GRAPH_MODE, BETA, TAU_FLOW, ALPHA_MAX, WARMUP_FRACTION, RAMP_FRACTION,
#     ESS_MIN_RATIO, MAX_P_FACTOR_FAIL, CLAMP_ACTIVATION_FAIL, P_CAP.
#   V2 memory (empty = config default; only meaningful for *batch_memory* cells):
#     MEMORY_QUEUE_CAPACITY, MEMORY_TOP_M, MEMORY_K_SIGMA, MEMORY_K_GUARD,
#     MEMORY_RETRIEVAL_CHUNK_SIZE,
#     MEMORY_WITNESS_THRESHOLD_MODE (explicit | shuffled_null_quantile),
#     MEMORY_WITNESS_NULL_QUANTILE, MEMORY_WITNESS_SCORE_MIN,
#     MEMORY_EXTRA_EDGES_PER_NODE_MAX, MEMORY_ADDED_EDGE_RATIO_MAX,
#     UTILITY_SMOOTHING_GRAPH (final | batch),
#     WITNESS_SCORE_MODE (specificity_weighted_hellinger | raw_hellinger),
#     MEMORY_WITNESS_ABLATION_MODE (none | shuffled_memory | random_memory),
#     MEMORY_WITNESS_NULL_SEED.
#   Bookkeeping:
#     WANDB_MODE (offline), WANDB_PROJECT_NAME (ssl-geo-dro), WANDB_RUN_NAME (=RUN_LABEL).
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
#SBATCH --job-name=geodro-jepa-v2-smoke
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

CONFIG_NAME="${CONFIG_NAME:-}"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: CONFIG_NAME must be set. Pick one of:"
  echo "  geodro/geodro_jepa_v2_coherent_hardness_imagenet100ctrl_smoke"
  echo "  geodro/geodro_jepa_v2_graph_transport_imagenet100ctrl_smoke"
  echo "  geodro/geodro_jepa_v2_coherent_hardness_batch_memory_imagenet100ctrl_smoke"
  echo "  geodro/geodro_jepa_v2_graph_transport_batch_memory_imagenet100ctrl_smoke"
  echo "  geodro/geodro_jepa_v2_coherent_hardness_batch_memory_optstep_imagenet100ctrl"
  echo "  geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl"
  exit 1
fi

NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-2}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-2}}}}"
TASKS_PER_NODE="${TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-${GPUS_PER_NODE}}}"
TRAINER_DEVICES="${TRAINER_DEVICES:-${GPUS_PER_NODE}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-1}"
ADVERSARY_SCOPE="${ADVERSARY_SCOPE:-microbatch}"
if [[ -z "${MEMORY_UPDATE_SCOPE:-}" ]]; then
  if [[ "${ADVERSARY_SCOPE}" == "optimizer_step" ]]; then
    MEMORY_UPDATE_SCOPE="optimizer_step_delayed"
  else
    MEMORY_UPDATE_SCOPE="microbatch"
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
RUN_LABEL="${RUN_LABEL:-${CONFIG_BASE}}"
RUN_ID="${RUN_LABEL}-${SLURM_JOB_ID:-local}"
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

case "${ADVERSARY_SCOPE}" in
  microbatch)
    if [[ "${ACCUM_GRAD_BATCHES}" != "1" ]]; then
      echo "ERROR: ADVERSARY_SCOPE=microbatch requires ACCUM_GRAD_BATCHES=1 (got ${ACCUM_GRAD_BATCHES})."
      exit 1
    fi
    if [[ "${MEMORY_UPDATE_SCOPE}" != "microbatch" ]]; then
      echo "ERROR: ADVERSARY_SCOPE=microbatch requires MEMORY_UPDATE_SCOPE=microbatch (got ${MEMORY_UPDATE_SCOPE})."
      exit 1
    fi
    ;;
  optimizer_step)
    if [[ "${MEMORY_UPDATE_SCOPE}" != "optimizer_step_delayed" ]]; then
      echo "ERROR: ADVERSARY_SCOPE=optimizer_step requires MEMORY_UPDATE_SCOPE=optimizer_step_delayed (got ${MEMORY_UPDATE_SCOPE})."
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unsupported ADVERSARY_SCOPE=${ADVERSARY_SCOPE}; expected microbatch or optimizer_step."
    exit 1
    ;;
esac
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

# Echo every variable that affects the run for reproducibility.
echo "============================================================"
echo "GeoDRO-JEPA V2.2 — Phase 13 V100 smoke"
echo "============================================================"
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Run label:        ${RUN_LABEL}"
echo "Run id:           ${RUN_ID}"
echo "Config:           ${CONFIG_NAME}"
echo "Nodes:            ${NUM_NODES}"
echo "GPUs/Node:        ${GPUS_PER_NODE}"
echo "Total GPUs:       $((NUM_NODES * GPUS_PER_NODE))"
echo "Batch/GPU:        ${BATCH_SIZE}"
echo "Graph Batch:      ${GRAPH_BATCH}"
echo "Grad Accum:       ${ACCUM_GRAD_BATCHES}"
echo "Adv Scope:        ${ADVERSARY_SCOPE}"
echo "Memory Clock:     ${MEMORY_UPDATE_SCOPE}"
echo "Trainer prec:     ${TRAINER_PRECISION}"
echo "num_slices:       ${NUM_SLICES}"
echo "inner_steps:      ${INNER_STEPS}"
echo "Data num_workers: ${NUM_WORKERS}"
echo "Max Epochs:       ${MAX_EPOCHS}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "PYTORCH_CUDA:     ${PYTORCH_CUDA_ALLOC_CONF}"
echo "W&B mode:         ${WANDB_MODE}"
echo "W&B project:      ${WANDB_PROJECT_NAME}"
echo "W&B run name:     ${WANDB_RUN_NAME}"
echo "------------------------------------------------------------"
echo "GeoDRO core knobs (empty = config default):"
echo "  GEODRO_K=${GEODRO_K:-}, GRAPH_MODE=${GRAPH_MODE:-}, BETA=${BETA:-}, TAU_FLOW=${TAU_FLOW:-}"
echo "  ALPHA_MAX=${ALPHA_MAX:-}, WARMUP_FRACTION=${WARMUP_FRACTION:-}, RAMP_FRACTION=${RAMP_FRACTION:-}"
echo "  ESS_MIN_RATIO=${ESS_MIN_RATIO:-}, MAX_P_FACTOR_FAIL=${MAX_P_FACTOR_FAIL:-}"
echo "  CLAMP_ACTIVATION_FAIL=${CLAMP_ACTIVATION_FAIL:-}, P_CAP=${P_CAP:-}"
echo "V2 memory knobs (empty = config default):"
echo "  MEMORY_QUEUE_CAPACITY=${MEMORY_QUEUE_CAPACITY:-}, MEMORY_TOP_M=${MEMORY_TOP_M:-}, MEMORY_K_SIGMA=${MEMORY_K_SIGMA:-}"
echo "  MEMORY_K_GUARD=${MEMORY_K_GUARD:-}, MEMORY_RETRIEVAL_CHUNK_SIZE=${MEMORY_RETRIEVAL_CHUNK_SIZE:-}"
echo "  MEMORY_WITNESS_THRESHOLD_MODE=${MEMORY_WITNESS_THRESHOLD_MODE:-}, MEMORY_WITNESS_NULL_QUANTILE=${MEMORY_WITNESS_NULL_QUANTILE:-}"
echo "  MEMORY_WITNESS_SCORE_MIN=${MEMORY_WITNESS_SCORE_MIN:-}"
echo "  MEMORY_EXTRA_EDGES_PER_NODE_MAX=${MEMORY_EXTRA_EDGES_PER_NODE_MAX:-}, MEMORY_ADDED_EDGE_RATIO_MAX=${MEMORY_ADDED_EDGE_RATIO_MAX:-}"
echo "  UTILITY_SMOOTHING_GRAPH=${UTILITY_SMOOTHING_GRAPH:-}, WITNESS_SCORE_MODE=${WITNESS_SCORE_MODE:-}"
echo "  MEMORY_WITNESS_ABLATION_MODE=${MEMORY_WITNESS_ABLATION_MODE:-}, MEMORY_WITNESS_NULL_SEED=${MEMORY_WITNESS_NULL_SEED:-}"
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

# Build Hydra overrides. Shape + GeoDRO core overrides are unconditional.
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
  ++metadata.memory_update_scope="${MEMORY_UPDATE_SCOPE}"
  metadata.local_batch_size="${BATCH_SIZE}"
  metadata.global_graph_batch_size="${GRAPH_BATCH}"
  ++module.geodro_lejepa_loss.memory_update_scope="${MEMORY_UPDATE_SCOPE}"
  module.geodro_lejepa_loss.num_slices="${NUM_SLICES}"
  module.geodro_lejepa_loss.inner_steps="${INNER_STEPS}"
)

# Optional GeoDRO core knobs — append only if env var is non-empty.
[[ -n "${GEODRO_K:-}" ]]                && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.k="${GEODRO_K}")
[[ -n "${GRAPH_MODE:-}" ]]              && HYDRA_OVERRIDES+=(metadata.graph_mode="${GRAPH_MODE}" module.geodro_lejepa_loss.graph_mode="${GRAPH_MODE}")
[[ -n "${BETA:-}" ]]                    && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.beta="${BETA}")
[[ -n "${TAU_FLOW:-}" ]]                && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.tau_flow="${TAU_FLOW}")
[[ -n "${ALPHA_MAX:-}" ]]               && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.alpha_max="${ALPHA_MAX}")
[[ -n "${WARMUP_FRACTION:-}" ]]         && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.warmup_fraction="${WARMUP_FRACTION}")
[[ -n "${RAMP_FRACTION:-}" ]]           && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.ramp_fraction="${RAMP_FRACTION}")
[[ -n "${ESS_MIN_RATIO:-}" ]]           && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.ess_min_ratio="${ESS_MIN_RATIO}")
[[ -n "${MAX_P_FACTOR_FAIL:-}" ]]       && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.max_p_factor_fail="${MAX_P_FACTOR_FAIL}")
[[ -n "${CLAMP_ACTIVATION_FAIL:-}" ]]   && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.clamp_activation_fail="${CLAMP_ACTIVATION_FAIL}")
[[ -n "${P_CAP:-}" ]]                   && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.p_cap="${P_CAP}")

# Optional V2 memory knobs — append only if env var is non-empty.
[[ -n "${MEMORY_QUEUE_CAPACITY:-}" ]]            && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.memory_queue_capacity="${MEMORY_QUEUE_CAPACITY}")
[[ -n "${MEMORY_TOP_M:-}" ]]                     && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.memory_top_m="${MEMORY_TOP_M}")
[[ -n "${MEMORY_K_SIGMA:-}" ]]                   && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.memory_k_sigma="${MEMORY_K_SIGMA}")
[[ -n "${MEMORY_K_GUARD:-}" ]]                   && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.memory_k_guard="${MEMORY_K_GUARD}")
[[ -n "${MEMORY_RETRIEVAL_CHUNK_SIZE:-}" ]]      && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.memory_retrieval_chunk_size="${MEMORY_RETRIEVAL_CHUNK_SIZE}")
[[ -n "${MEMORY_WITNESS_THRESHOLD_MODE:-}" ]]    && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.memory_witness_threshold_mode="${MEMORY_WITNESS_THRESHOLD_MODE}")
[[ -n "${MEMORY_WITNESS_NULL_QUANTILE:-}" ]]     && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.memory_witness_null_quantile="${MEMORY_WITNESS_NULL_QUANTILE}")
[[ -n "${MEMORY_WITNESS_SCORE_MIN:-}" ]]         && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_score_min="${MEMORY_WITNESS_SCORE_MIN}")
[[ -n "${MEMORY_EXTRA_EDGES_PER_NODE_MAX:-}" ]]  && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.memory_extra_edges_per_node_max="${MEMORY_EXTRA_EDGES_PER_NODE_MAX}")
[[ -n "${MEMORY_ADDED_EDGE_RATIO_MAX:-}" ]]      && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.memory_added_edge_ratio_max="${MEMORY_ADDED_EDGE_RATIO_MAX}")
[[ -n "${UTILITY_SMOOTHING_GRAPH:-}" ]]          && HYDRA_OVERRIDES+=(module.geodro_lejepa_loss.utility_smoothing_graph="${UTILITY_SMOOTHING_GRAPH}")
[[ -n "${WITNESS_SCORE_MODE:-}" ]]               && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.witness_score_mode="${WITNESS_SCORE_MODE}")
[[ -n "${MEMORY_WITNESS_ABLATION_MODE:-}" ]]     && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_ablation_mode="${MEMORY_WITNESS_ABLATION_MODE}")
[[ -n "${MEMORY_WITNESS_NULL_SEED:-}" ]]         && HYDRA_OVERRIDES+=(++module.geodro_lejepa_loss.memory_witness_null_seed="${MEMORY_WITNESS_NULL_SEED}")

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
echo "GeoDRO-JEPA V2.2 V100 smoke (${RUN_LABEL}) PASSED"
echo "============================================================"
echo "Output:    ${OUTPUT_DIR}"
echo "Run check: python scripts/check_smoke_diagnostics.py ${OUTPUT_DIR}"
echo "============================================================"
