#!/usr/bin/env bash
#===============================================================================
# Submit the full GeoDRO-LeJEPA evaluation suite for one pretrained checkpoint.
#
# Usage:
#   bash scripts/slurm/run_full_eval_suite.sh <CHECKPOINT_PATH> [PRETRAIN_RUN_ID] [METHOD]
#
# Example:
#   bash scripts/slurm/run_full_eval_suite.sh \
#     /path/to/best.ckpt geo-full-r6-k24-b128-a4-e400 geodro_v1_1
#
# Submits one sbatch job per evaluation mode (9 jobs by default). Honors
# environment-variable overrides forwarded via sbatch --export=ALL, so e.g.
# WANDB_MODE=offline carries through to all 9 child jobs.
#
# Set EVAL_MODES to a comma-separated list to skip modes:
#   EVAL_MODES=imagenet100ctrl,waterbirds bash run_full_eval_suite.sh ...
#===============================================================================

set -euo pipefail

CHECKPOINT_PATH="${1:-}"
PRETRAIN_RUN_ID="${2:-${PRETRAIN_RUN_ID:-}}"
METHOD="${3:-${METHOD:-${PRETRAIN_RUN_ID:-unknown}}}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Usage: $0 <CHECKPOINT_PATH> [PRETRAIN_RUN_ID] [METHOD]" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ALL_MODES=(
  imagenet100ctrl
  imagenet100c
  waterbirds
  imagenet_sketch
  imagenet_r
  imagenet_a
  imagenet_o
  celeba
  camelyon17
)

# Filter modes if EVAL_MODES is set.
declare -a SELECTED_MODES
if [[ -n "${EVAL_MODES:-}" ]]; then
  IFS=',' read -r -a SELECTED_MODES <<< "${EVAL_MODES}"
else
  SELECTED_MODES=("${ALL_MODES[@]}")
fi

# Map mode -> sbatch file name.
sbatch_for_mode() {
  case "$1" in
    imagenet100ctrl) echo "geodro_lejepa_imagenet100ctrl_eval.sbatch" ;;
    imagenet100c)    echo "geodro_lejepa_imagenet100c_eval.sbatch" ;;
    waterbirds)      echo "geodro_lejepa_waterbirds_eval.sbatch" ;;
    imagenet_sketch) echo "geodro_lejepa_imagenet_sketch_eval.sbatch" ;;
    imagenet_r)      echo "geodro_lejepa_imagenet_r_eval.sbatch" ;;
    imagenet_a)      echo "geodro_lejepa_imagenet_a_eval.sbatch" ;;
    imagenet_o)      echo "geodro_lejepa_imagenet_o_eval.sbatch" ;;
    celeba)          echo "geodro_lejepa_celeba_eval.sbatch" ;;
    camelyon17)      echo "geodro_lejepa_camelyon17_eval.sbatch" ;;
    *)               echo "" ;;
  esac
}

PARTITION="${PARTITION:-mcml-hgx-a100-80x4}"
QOS="${QOS:-mcml}"

EXPORT_VARS="ALL,CHECKPOINT_PATH=${CHECKPOINT_PATH},PRETRAIN_RUN_ID=${PRETRAIN_RUN_ID},METHOD=${METHOD}"

echo "============================================================"
echo "Submitting full GeoDRO-LeJEPA eval suite"
echo "============================================================"
echo "Checkpoint:       ${CHECKPOINT_PATH}"
echo "PRETRAIN_RUN_ID:  ${PRETRAIN_RUN_ID:-(unset)}"
echo "METHOD:           ${METHOD}"
echo "Modes:            ${SELECTED_MODES[*]}"
echo "Partition:        ${PARTITION}"
echo "QoS:              ${QOS}"
echo "============================================================"

for mode in "${SELECTED_MODES[@]}"; do
  sbatch_file="$(sbatch_for_mode "${mode}")"
  if [[ -z "${sbatch_file}" ]]; then
    echo "WARNING: unknown mode '${mode}', skipping" >&2
    continue
  fi
  if [[ ! -f "${SCRIPT_DIR}/${sbatch_file}" ]]; then
    echo "WARNING: sbatch file not found, skipping: ${SCRIPT_DIR}/${sbatch_file}" >&2
    continue
  fi
  echo ""
  echo "[submit] mode=${mode}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  DRY_RUN: sbatch -p ${PARTITION} --qos=${QOS} --export=${EXPORT_VARS} ${sbatch_file}"
  else
    sbatch -p "${PARTITION}" --qos="${QOS}" \
      --export="${EXPORT_VARS}" \
      "${SCRIPT_DIR}/${sbatch_file}"
  fi
done
