#!/bin/bash
#===============================================================================
# GeoDRO-LeJEPA ImageNet-100 — FULL pretraining (400 epochs, 48h walltime)
#
# Use this launcher for erm-full, geo-full-*, geo-full-*-mem, and continuations
# that must finish within ~48h. Targets LRZ H100 (historically ~7 min/epoch vs
# ~11 min/epoch on MCML A100 for the same graph-batch-2048 shape).
#
# Short diagnostics, probes, and exploration grids should keep using
# geodro_lejepa_imagenet100ctrl_main.sbatch (MCML A100 by default) or the
# v2 smoke launcher — do not submit those through this script.
#
# Usage (fresh full run):
#   cd stable-pretraining
#   sbatch --export=ALL,RUN_LABEL=erm-full-b128-a4-e400,\
#     CONFIG_NAME=geodro/geodro_lejepa_imagenet100ctrl_erm_optstep_accum,\
#     NUM_NODES=1,GPUS_PER_NODE=4,TASKS_PER_NODE=4,TRAINER_DEVICES=4,\
#     BATCH_SIZE=128,VAL_BATCH_SIZE=128,ACCUM_GRAD_BATCHES=4,\
#     MAX_EPOCHS=400,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2 \
#     scripts/slurm/geodro_lejepa_imagenet100ctrl_full.sbatch
#
# Continuation: same launcher, RUN_PREFLIGHT=0, RESUME_CKPT_PATH=...
# See examples/geodro/README.md.
#===============================================================================

#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=750000M
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm-bootstrap-%j.out
#SBATCH -e logs/slurm-bootstrap-%j.err
#SBATCH --job-name=geodro-im100ctrl-full
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm"
fi

MAIN_LAUNCHER="${SCRIPT_DIR}/geodro_lejepa_imagenet100ctrl_main.sbatch"
if [[ ! -f "${MAIN_LAUNCHER}" ]]; then
  echo "ERROR: missing ${MAIN_LAUNCHER}."
  echo "Submit from: ${PROJECT_ROOT}"
  echo "Expected submit dir with examples/: /dss/dsshome1/0D/ra59dut2/workspace/thesis/ssl-geo-dro/stable-pretraining"
  exit 1
fi

# shellcheck disable=SC1091
source "${MAIN_LAUNCHER}"
