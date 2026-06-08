#!/bin/bash
#===============================================================================
# GeoDRO-LeJEPA Resume Sanity Check (ImageNet-100 controlled, ERM optstep config)
#
# Goal:
#   Validate that a GeoDRO main-run checkpoint is actually loaded before a
#   full continuation job. Runs on 1 node / 1 GPU for a few extra epochs.
#
# Usage:
#   sbatch --export=ALL,RESUME_CKPT_PATH=/abs/path/to/last.ckpt \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_resume_sanity.sbatch
#
# Example (50-epoch ERM pilot checkpoint):
#   sbatch --export=ALL,\
#     RESUME_CKPT_PATH=outputs/erm-d0-b128-a4-5649125/checkpoints/last.ckpt,\
#     EXTRA_EPOCHS=2 \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_resume_sanity.sbatch
#
# Notes:
#   - RESUME_CKPT_PATH may be given with or without ".ckpt".
#   - Relative paths resolve from PROJECT_ROOT.
#===============================================================================

#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
#SBATCH --job-name=geodro-im100ctrl-resume-sanity
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

CONFIG_NAME="${CONFIG_NAME:-geodro/geodro_lejepa_imagenet100ctrl_erm_optstep_accum}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
if [[ -z "${RESUME_CKPT_PATH}" ]]; then
  echo "ERROR: RESUME_CKPT_PATH is required."
  echo "Example:"
  echo "  sbatch --export=ALL,RESUME_CKPT_PATH=outputs/erm-d0-b128-a4-5649125/checkpoints/last.ckpt \\"
  echo "    scripts/slurm/geodro_lejepa_imagenet100ctrl_resume_sanity.sbatch"
  exit 1
fi

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
  echo "ERROR: Checkpoint not found: ${RESUME_CKPT_PATH}"
  exit 1
fi

EXTRA_EPOCHS="${EXTRA_EPOCHS:-2}"
if ! [[ "${EXTRA_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${EXTRA_EPOCHS}" -lt 1 ]]; then
  echo "ERROR: EXTRA_EPOCHS must be a positive integer (got: ${EXTRA_EPOCHS})"
  exit 1
fi

source ~/.bashrc
mamba activate py312

if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
  source "${SCRIPT_DIR}/utils.sh"
  validate_env py312 || exit 1
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
MAX_EPOCHS="$((RESUME_EPOCH + EXTRA_EPOCHS))"

BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-4}"
RESUME_WEIGHTS_ONLY="${RESUME_WEIGHTS_ONLY:-false}"
AUTO_RESUME_FALLBACK="${AUTO_RESUME_FALLBACK:-true}"

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

RESUME_WEIGHTS_ONLY="$(normalize_bool "${RESUME_WEIGHTS_ONLY}")"
AUTO_RESUME_FALLBACK="$(normalize_bool "${AUTO_RESUME_FALLBACK}")"

RUN_ID="geodro-im100ctrl-resume-sanity-${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/resume_sanity.log"
GRAPH_BATCH=$((BATCH_SIZE * ACCUM_GRAD_BATCHES))

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
  setup_geodro_dataset_cache imagenet100 || exit 1
fi

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-geodro-im100ctrl-resume-sanity}"
WANDB_RUN_NAME="${RUN_ID}"
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_PROJECT

if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
  require_wandb_key || exit 1
fi

echo "============================================================"
echo "GeoDRO-LeJEPA Resume Sanity (ImageNet-100 controlled ERM)"
echo "============================================================"
echo "Job ID:               ${SLURM_JOB_ID:-local}"
echo "Config:               ${CONFIG_NAME}"
echo "Checkpoint:           ${RESUME_CKPT_PATH}"
echo "Checkpoint epoch:     ${RESUME_EPOCH}"
echo "Additional epochs:    ${EXTRA_EPOCHS}"
echo "Target trainer epoch: ${MAX_EPOCHS}"
echo "Batch/GPU:            ${BATCH_SIZE}"
echo "Grad accum:           ${ACCUM_GRAD_BATCHES}"
echo "Graph batch:          ${GRAPH_BATCH}"
echo "Resume weights only: ${RESUME_WEIGHTS_ONLY}"
echo "Auto fallback:        ${AUTO_RESUME_FALLBACK}"
echo "Output dir:           ${OUTPUT_DIR}"
echo "============================================================"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

run_resume_sanity() {
  local weights_only="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running resume sanity with resume_weights_only=${weights_only}"
  python -m stable_pretraining.run \
    --config-path "${PROJECT_ROOT}/examples" \
    --config-name "${CONFIG_NAME}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.accumulate_grad_batches="${ACCUM_GRAD_BATCHES}" \
    trainer.logger.name="${WANDB_RUN_NAME}" \
    trainer.logger.project="${WANDB_PROJECT_NAME}" \
    data.train.batch_size="${BATCH_SIZE}" \
    data.val.batch_size="${VAL_BATCH_SIZE}" \
    metadata.adversary_scope=optimizer_step \
    metadata.local_batch_size="${BATCH_SIZE}" \
    metadata.global_graph_batch_size="${GRAPH_BATCH}" \
    ckpt_path="${RESUME_CKPT_PATH}" \
    +resume_weights_only="${weights_only}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"
  return "${PIPESTATUS[0]}"
}

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting resume sanity run..."
echo ""

: > "${LOG_FILE}"
RUN_EXIT_CODE=0
set +e
run_resume_sanity "${RESUME_WEIGHTS_ONLY}"
RUN_EXIT_CODE=$?
set -e
if [[ ${RUN_EXIT_CODE} -ne 0 && "${AUTO_RESUME_FALLBACK}" == "true" && "${RESUME_WEIGHTS_ONLY}" == "false" ]] && \
   is_resume_compat_error "${LOG_FILE}"; then
  echo "WARNING: Detected checkpoint compatibility issue. Retrying with resume_weights_only=true."
  echo "WARNING: Fallback will resume model weights but may skip optimizer/callback/trainer runtime state."
  set +e
  run_resume_sanity "true"
  RUN_EXIT_CODE=$?
  set -e
fi

if [[ ${RUN_EXIT_CODE} -ne 0 ]]; then
  echo "ERROR: Resume sanity run failed with exit code ${RUN_EXIT_CODE}"
  exit ${RUN_EXIT_CODE}
fi

if grep -q "specified, but does not exist, using None for now" "${LOG_FILE}" || \
   grep -q "CALLING trainer.fit with ckpt_path=None" "${LOG_FILE}"; then
  echo "ERROR: Detected fallback to training from scratch."
  echo "Inspect: ${LOG_FILE}"
  exit 1
fi

if ! grep -q "CALLING trainer.fit with ckpt_path='" "${LOG_FILE}"; then
  echo "ERROR: Could not find explicit ckpt_path confirmation in log."
  echo "Inspect: ${LOG_FILE}"
  exit 1
fi

EXPECTED_NEXT_EPOCH="$((RESUME_EPOCH + 1))"
if ! grep -q "Epoch ${EXPECTED_NEXT_EPOCH}," "${LOG_FILE}"; then
  echo "WARNING: Expected to see training progress at epoch ${EXPECTED_NEXT_EPOCH} in log."
  echo "Inspect: ${LOG_FILE}"
fi

echo ""
echo "============================================================"
echo "GeoDRO resume sanity check PASSED"
echo "============================================================"
echo "Checkpoint load was validated without fallback to scratch."
echo "Log file: ${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done"
