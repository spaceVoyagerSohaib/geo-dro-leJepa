#!/bin/bash
#===============================================================================
# GeoDRO-JEPA V2.2 broader exploration helper.
#
# This helper is intentionally separate from geodro_jepa_v2_pilot_wave.sh:
# - pilot_wave restores the baseline 33-job recovery grid;
# - this file defines targeted follow-up blocks for graph, flow/gate, and memory
#   witness exploration after the baseline mechanism diagnostics are readable.
#
# Usage from stable-pretraining/:
#
#   DRY_RUN=1 bash scripts/slurm/geodro_jepa_v2_exploration_grid.sh graph_block
#   DRY_RUN=1 bash scripts/slurm/geodro_jepa_v2_exploration_grid.sh flow_gate_block
#   DRY_RUN=1 bash scripts/slurm/geodro_jepa_v2_exploration_grid.sh memory_witness_block
#
# Remove DRY_RUN=1 only after checking queue pressure and baseline V2 diagnostics.
#===============================================================================

set -euo pipefail

ACTION="${1:-help}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

LAUNCHER="scripts/slurm/geodro_jepa_v2_smoke.sbatch"
if [[ ! -f "${LAUNCHER}" ]]; then
  echo "ERROR: expected launcher at ${LAUNCHER}" >&2
  exit 1
fi

PARTITION="${PARTITION:-mcml-hgx-a100-80x4}"
QOS="${QOS:-mcml}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TASKS_PER_NODE="${TASKS_PER_NODE:-${GPUS_PER_NODE}}"
TRAINER_DEVICES="${TRAINER_DEVICES:-${GPUS_PER_NODE}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_SLICES="${NUM_SLICES:-256}"
ADVERSARY_SCOPE="${ADVERSARY_SCOPE:-optimizer_step}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-4}"
MEMORY_UPDATE_SCOPE="${MEMORY_UPDATE_SCOPE:-optimizer_step_delayed}"
GRID_WALLTIME="${GRID_WALLTIME:-06:00:00}"
SMOKE_WALLTIME="${SMOKE_WALLTIME:-02:00:00}"

case "${ADVERSARY_SCOPE}" in
  optimizer_step)
    if [[ "${MEMORY_UPDATE_SCOPE}" != "optimizer_step_delayed" ]]; then
      echo "ERROR: optimizer_step requires MEMORY_UPDATE_SCOPE=optimizer_step_delayed." >&2
      exit 1
    fi
    GRAPH_BATCH=$((NUM_NODES * GPUS_PER_NODE * BATCH_SIZE * ACCUM_GRAD_BATCHES))
    ;;
  microbatch)
    if [[ "${ACCUM_GRAD_BATCHES}" != "1" || "${MEMORY_UPDATE_SCOPE}" != "microbatch" ]]; then
      echo "ERROR: microbatch requires ACCUM_GRAD_BATCHES=1 and MEMORY_UPDATE_SCOPE=microbatch." >&2
      exit 1
    fi
    GRAPH_BATCH=$((NUM_NODES * GPUS_PER_NODE * BATCH_SIZE))
    ;;
  *)
    echo "ERROR: unsupported ADVERSARY_SCOPE=${ADVERSARY_SCOPE}." >&2
    exit 1
    ;;
esac

if [[ "${GRAPH_BATCH}" != "2048" ]]; then
  echo "WARNING: graph batch is ${GRAPH_BATCH}, not the canonical 2048." >&2
fi

submit_job() {
  local label="$1"
  local config="$2"
  local walltime="$3"
  local exports="$4"

  local cmd=(
    sbatch
    -p "${PARTITION}"
    --qos="${QOS}"
    --nodes="${NUM_NODES}"
    --ntasks-per-node="${TASKS_PER_NODE}"
    --gres="gpu:${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    -t "${walltime}"
    --job-name="${label}"
    --export="ALL,CONFIG_NAME=${config},RUN_LABEL=${label},NUM_NODES=${NUM_NODES},GPUS_PER_NODE=${GPUS_PER_NODE},TASKS_PER_NODE=${TASKS_PER_NODE},TRAINER_DEVICES=${TRAINER_DEVICES},BATCH_SIZE=${BATCH_SIZE},VAL_BATCH_SIZE=${VAL_BATCH_SIZE},TRAINER_PRECISION=${TRAINER_PRECISION},NUM_WORKERS=${NUM_WORKERS},NUM_SLICES=${NUM_SLICES},ADVERSARY_SCOPE=${ADVERSARY_SCOPE},ACCUM_GRAD_BATCHES=${ACCUM_GRAD_BATCHES},MEMORY_UPDATE_SCOPE=${MEMORY_UPDATE_SCOPE},${exports}"
    "${LAUNCHER}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

batch_configs() {
  printf '%s\n' \
    "ch:geodro/geodro_jepa_v2_coherent_hardness_imagenet100ctrl_smoke" \
    "gt:geodro/geodro_jepa_v2_graph_transport_imagenet100ctrl_smoke"
}

memory_configs() {
  printf '%s\n' \
    "ch:geodro/geodro_jepa_v2_coherent_hardness_batch_memory_optstep_imagenet100ctrl" \
    "gt:geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl"
}

base_run_exports() {
  local epochs="$1"
  printf 'MAX_EPOCHS=%s,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2' "${epochs}"
}

r6_flow() {
  local p_cap="${1:-0.0035}"
  printf 'GEODRO_K=24,INNER_STEPS=20,TAU_FLOW=0.025,BETA=0.2,ALPHA_MAX=0.03,WARMUP_FRACTION=0.0,RAMP_FRACTION=0.0,CLAMP_ACTIVATION_FAIL=0.50,ESS_MIN_RATIO=0.005,MAX_P_FACTOR_FAIL=256,P_CAP=%s' "${p_cap}"
}

r4_flow() {
  local p_cap="${1:-0.0035}"
  printf 'GEODRO_K=16,INNER_STEPS=20,TAU_FLOW=0.025,BETA=0.3,ALPHA_MAX=0.03,WARMUP_FRACTION=0.0,RAMP_FRACTION=0.0,CLAMP_ACTIVATION_FAIL=0.50,ESS_MIN_RATIO=0.005,MAX_P_FACTOR_FAIL=256,P_CAP=%s' "${p_cap}"
}

n4_flow() {
  local p_cap="${1:-0.0035}"
  printf 'GEODRO_K=24,INNER_STEPS=40,TAU_FLOW=0.0125,BETA=0.2,ALPHA_MAX=0.03,WARMUP_FRACTION=0.0,RAMP_FRACTION=0.0,CLAMP_ACTIVATION_FAIL=0.55,ESS_MIN_RATIO=0.003,MAX_P_FACTOR_FAIL=256,P_CAP=%s' "${p_cap}"
}

alpha04_flow() {
  printf 'GEODRO_K=24,INNER_STEPS=20,TAU_FLOW=0.025,BETA=0.2,ALPHA_MAX=0.04,WARMUP_FRACTION=0.0,RAMP_FRACTION=0.0,CLAMP_ACTIVATION_FAIL=0.50,ESS_MIN_RATIO=0.005,MAX_P_FACTOR_FAIL=256,P_CAP=0.005'
}

memory_common() {
  printf 'MEMORY_QUEUE_CAPACITY=16384,MEMORY_RETRIEVAL_CHUNK_SIZE=512,WITNESS_SCORE_MODE=specificity_weighted_hellinger'
}

memory_default_axes() {
  printf 'MEMORY_TOP_M=64,MEMORY_K_SIGMA=64,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=shuffled_null_quantile,MEMORY_WITNESS_NULL_QUANTILE=0.95,MEMORY_EXTRA_EDGES_PER_NODE_MAX=2,MEMORY_ADDED_EDGE_RATIO_MAX=0.25'
}

submit_graph_block() {
  local cell_config cell config k topology flow exports

  for cell_config in $(batch_configs); do
    cell="${cell_config%%:*}"
    config="${cell_config#*:}"
    for k in 12 16 20 24 32; do
      flow="$(r6_flow 0.0035)"
      flow="${flow/GEODRO_K=24/GEODRO_K=${k}}"
      submit_job "v2x-${cell}-graph-k${k}-b${BATCH_SIZE}" "${config}" "${GRID_WALLTIME}" \
        "$(base_run_exports 10),${flow}"
    done
    for topology in max_union_knn random_regular; do
      submit_job "v2x-${cell}-graph-${topology}-b${BATCH_SIZE}" "${config}" "${GRID_WALLTIME}" \
        "$(base_run_exports 10),$(r6_flow 0.0035),GRAPH_MODE=${topology}"
    done
    submit_job "v2x-${cell}-graph-fully-connected-smoke-b${BATCH_SIZE}" "${config}" "${SMOKE_WALLTIME}" \
      "$(base_run_exports 1),$(r6_flow 0.0035),GRAPH_MODE=fully_connected"
  done
}

submit_flow_gate_block() {
  local cell_config cell config preset exports
  local presets=(
    "r6-pcap0035:$(r6_flow 0.0035)"
    "r6-pcap005:$(r6_flow 0.005)"
    "r4-pcap0035:$(r4_flow 0.0035)"
    "n4-pcap0035:$(n4_flow 0.0035)"
    "alpha04-pcap005:$(alpha04_flow)"
  )

  for cell_config in $(batch_configs); do
    cell="${cell_config%%:*}"
    config="${cell_config#*:}"
    for preset in "${presets[@]}"; do
      submit_job "v2x-${cell}-flow-${preset%%:*}-b${BATCH_SIZE}" "${config}" "${GRID_WALLTIME}" \
        "$(base_run_exports 10),${preset#*:}"
    done
  done
}

submit_memory_witness_block() {
  local cell_config cell config variant
  local variants=(
    "topm32:MEMORY_TOP_M=32,MEMORY_K_SIGMA=32,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=shuffled_null_quantile,MEMORY_WITNESS_NULL_QUANTILE=0.95,MEMORY_EXTRA_EDGES_PER_NODE_MAX=2,MEMORY_ADDED_EDGE_RATIO_MAX=0.25"
    "topm128:MEMORY_TOP_M=128,MEMORY_K_SIGMA=128,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=shuffled_null_quantile,MEMORY_WITNESS_NULL_QUANTILE=0.95,MEMORY_EXTRA_EDGES_PER_NODE_MAX=2,MEMORY_ADDED_EDGE_RATIO_MAX=0.25"
    "topm64-ksigma32:MEMORY_TOP_M=64,MEMORY_K_SIGMA=32,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=shuffled_null_quantile,MEMORY_WITNESS_NULL_QUANTILE=0.95,MEMORY_EXTRA_EDGES_PER_NODE_MAX=2,MEMORY_ADDED_EDGE_RATIO_MAX=0.25"
    "topm64-ksigma128:MEMORY_TOP_M=64,MEMORY_K_SIGMA=128,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=shuffled_null_quantile,MEMORY_WITNESS_NULL_QUANTILE=0.95,MEMORY_EXTRA_EDGES_PER_NODE_MAX=2,MEMORY_ADDED_EDGE_RATIO_MAX=0.25"
    "explicit002:MEMORY_TOP_M=64,MEMORY_K_SIGMA=64,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=explicit,MEMORY_WITNESS_SCORE_MIN=0.02,MEMORY_EXTRA_EDGES_PER_NODE_MAX=2,MEMORY_ADDED_EDGE_RATIO_MAX=0.25"
    "explicit005:MEMORY_TOP_M=64,MEMORY_K_SIGMA=64,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=explicit,MEMORY_WITNESS_SCORE_MIN=0.05,MEMORY_EXTRA_EDGES_PER_NODE_MAX=2,MEMORY_ADDED_EDGE_RATIO_MAX=0.25"
    "edge4:MEMORY_TOP_M=64,MEMORY_K_SIGMA=64,MEMORY_K_GUARD=64,MEMORY_WITNESS_THRESHOLD_MODE=shuffled_null_quantile,MEMORY_WITNESS_NULL_QUANTILE=0.95,MEMORY_EXTRA_EDGES_PER_NODE_MAX=4,MEMORY_ADDED_EDGE_RATIO_MAX=0.50"
    "shuffled-null:$(memory_default_axes),MEMORY_WITNESS_ABLATION_MODE=shuffled_memory,MEMORY_WITNESS_NULL_SEED=17"
    "random-null:$(memory_default_axes),MEMORY_WITNESS_ABLATION_MODE=random_memory,MEMORY_WITNESS_NULL_SEED=17"
  )

  for cell_config in $(memory_configs); do
    cell="${cell_config%%:*}"
    config="${cell_config#*:}"
    for variant in "${variants[@]}"; do
      submit_job "v2x-${cell}-mem-${variant%%:*}-b${BATCH_SIZE}" "${config}" "${GRID_WALLTIME}" \
        "$(base_run_exports 10),$(r6_flow 0.0035),$(memory_common),${variant#*:}"
    done
  done
}

case "${ACTION}" in
  graph_block)
    submit_graph_block
    ;;
  flow_gate_block)
    submit_flow_gate_block
    ;;
  memory_witness_block)
    submit_memory_witness_block
    ;;
  all_extra)
    submit_graph_block
    submit_flow_gate_block
    submit_memory_witness_block
    ;;
  help|-h|--help)
    echo "Usage: DRY_RUN=1 bash $0 <graph_block|flow_gate_block|memory_witness_block|all_extra>"
    ;;
  *)
    echo "ERROR: unknown action '${ACTION}'." >&2
    echo "Usage: DRY_RUN=1 bash $0 <graph_block|flow_gate_block|memory_witness_block|all_extra>" >&2
    exit 1
    ;;
esac
