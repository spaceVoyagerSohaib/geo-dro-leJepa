#!/bin/bash
# Compare Slurm start estimates for GeoDRO-LeJEPA ImageNet-100 controlled runs.
#
# This script uses sbatch --test-only and does not submit jobs.
# Override CONFIG_NAME to probe a specific variant, for example:
#   CONFIG_NAME=geodro/geodro_lejepa_imagenet100ctrl_alpha0 \
#     bash scripts/slurm/geodro_lejepa_queue_probe.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/geodro_lejepa_imagenet100ctrl_main.sbatch"

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
    DEFAULT_ACCUM_GRAD_BATCHES=4
    DEFAULT_BATCH_SIZE_2NODE=64
    DEFAULT_BATCH_SIZE_1NODE=128
else
    DEFAULT_ACCUM_GRAD_BATCHES=1
    DEFAULT_BATCH_SIZE_2NODE=256
    DEFAULT_BATCH_SIZE_1NODE=512
fi
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-${DEFAULT_ACCUM_GRAD_BATCHES}}"
BATCH_SIZE_2NODE="${BATCH_SIZE_2NODE:-${DEFAULT_BATCH_SIZE_2NODE}}"
BATCH_SIZE_1NODE="${BATCH_SIZE_1NODE:-${DEFAULT_BATCH_SIZE_1NODE}}"
VAL_BATCH_SIZE_2NODE="${VAL_BATCH_SIZE_2NODE:-${BATCH_SIZE_2NODE}}"
VAL_BATCH_SIZE_1NODE="${VAL_BATCH_SIZE_1NODE:-${BATCH_SIZE_1NODE}}"

cd "${PROJECT_ROOT}"

probe() {
    local label="$1"
    shift

    echo ""
    echo "============================================================"
    echo "${label}"
    echo "============================================================"

    set +e
    sbatch --test-only "$@" "${MAIN_SCRIPT}"
    local status=$?
    set -e

    if [[ ${status} -ne 0 ]]; then
        echo "Probe failed with exit code ${status}: ${label}"
    fi
}

COMMON_EXPORTS="ALL,CONFIG_NAME=${CONFIG_NAME},ADVERSARY_SCOPE=${ADVERSARY_SCOPE},ACCUM_GRAD_BATCHES=${ACCUM_GRAD_BATCHES}"

echo "GeoDRO-LeJEPA queue probe"
echo "Config: ${CONFIG_NAME}"
echo "Adversary scope: ${ADVERSARY_SCOPE}"
echo "Gradient accumulation: ${ACCUM_GRAD_BATCHES}"
echo "Main script: ${MAIN_SCRIPT}"
echo "No jobs will be submitted."

probe "MCML A100 2 nodes, 48h, batch ${BATCH_SIZE_2NODE}/GPU" \
    -p mcml-hgx-a100-80x4 \
    --qos=mcml \
    --nodes=2 \
    --time=48:00:00 \
    --job-name=probe-geodro-mcml-2n \
    --export="${COMMON_EXPORTS},NUM_NODES=2,BATCH_SIZE=${BATCH_SIZE_2NODE},VAL_BATCH_SIZE=${VAL_BATCH_SIZE_2NODE}"

probe "MCML A100 1 node, 48h, batch ${BATCH_SIZE_1NODE}/GPU" \
    -p mcml-hgx-a100-80x4 \
    --qos=mcml \
    --nodes=1 \
    --time=48:00:00 \
    --job-name=probe-geodro-mcml-1n-48h \
    --export="${COMMON_EXPORTS},NUM_NODES=1,BATCH_SIZE=${BATCH_SIZE_1NODE},VAL_BATCH_SIZE=${VAL_BATCH_SIZE_1NODE}"

probe "MCML A100 1 node, 72h, batch ${BATCH_SIZE_1NODE}/GPU" \
    -p mcml-hgx-a100-80x4 \
    --qos=mcml \
    --nodes=1 \
    --time=72:00:00 \
    --job-name=probe-geodro-mcml-1n-72h \
    --export="${COMMON_EXPORTS},NUM_NODES=1,BATCH_SIZE=${BATCH_SIZE_1NODE},VAL_BATCH_SIZE=${VAL_BATCH_SIZE_1NODE}"

probe "MCML A100 1 node, 96h, batch ${BATCH_SIZE_1NODE}/GPU" \
    -p mcml-hgx-a100-80x4 \
    --qos=mcml \
    --nodes=1 \
    --time=96:00:00 \
    --job-name=probe-geodro-mcml-1n-96h \
    --export="${COMMON_EXPORTS},NUM_NODES=1,BATCH_SIZE=${BATCH_SIZE_1NODE},VAL_BATCH_SIZE=${VAL_BATCH_SIZE_1NODE}"

probe "LRZ H100 1 node, 48h, batch ${BATCH_SIZE_1NODE}/GPU" \
    -p lrz-hgx-h100-94x4 \
    --qos=gpu \
    --nodes=1 \
    --ntasks-per-node=4 \
    --gres=gpu:4 \
    --cpus-per-task=20 \
    --mem=750000M \
    --time=48:00:00 \
    --job-name=probe-geodro-lrz-h100-1n \
    --export="${COMMON_EXPORTS},NUM_NODES=1,GPUS_PER_NODE=4,TASKS_PER_NODE=4,TRAINER_DEVICES=4,BATCH_SIZE=${BATCH_SIZE_1NODE},VAL_BATCH_SIZE=${VAL_BATCH_SIZE_1NODE}"
