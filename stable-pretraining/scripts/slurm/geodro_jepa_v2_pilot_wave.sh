#!/bin/bash
#===============================================================================
# GeoDRO-JEPA V2.2 pilot-wave submit helper.
#
# This helper wraps geodro_jepa_v2_smoke.sbatch for the next post-smoke V2.2
# phase. It intentionally submits the B128 memory shape gate by default. The
# broader 10-epoch grids require an explicit action after the shape gate passes.
#
# Usage from stable-pretraining/:
#
#   # Submit only the B128 optimizer-step memory shape gate.
#   bash scripts/slurm/geodro_jepa_v2_pilot_wave.sh shape_gate
#
#   # After the shape gate passes, submit the activation grid.
#   bash scripts/slurm/geodro_jepa_v2_pilot_wave.sh activation_grid
#
#   # After the shape gate passes, submit the memory topology grid.
#   bash scripts/slurm/geodro_jepa_v2_pilot_wave.sh memory_grid
#
#   # Optional: re-submit TIMEOUT recovery jobs (not used for current pilot analysis).
#   bash scripts/slurm/geodro_jepa_v2_pilot_wave.sh resubmit_timeouts
#   bash scripts/slurm/geodro_jepa_v2_pilot_wave.sh resubmit_priority
#
#   # Print commands without submitting.
#   DRY_RUN=1 bash scripts/slurm/geodro_jepa_v2_pilot_wave.sh all
#===============================================================================

set -euo pipefail

ACTION="${1:-shape_gate}"

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
if [[ -z "${ACCUM_GRAD_BATCHES:-}" ]]; then
  if [[ "${ADVERSARY_SCOPE}" == "optimizer_step" ]]; then
    ACCUM_GRAD_BATCHES=4
  else
    ACCUM_GRAD_BATCHES=1
  fi
fi
if [[ -z "${MEMORY_UPDATE_SCOPE:-}" ]]; then
  if [[ "${ADVERSARY_SCOPE}" == "optimizer_step" ]]; then
    MEMORY_UPDATE_SCOPE="optimizer_step_delayed"
  else
    MEMORY_UPDATE_SCOPE="microbatch"
  fi
fi

case "${ADVERSARY_SCOPE}" in
  microbatch)
    if [[ "${ACCUM_GRAD_BATCHES}" != "1" ]]; then
      echo "ERROR: ADVERSARY_SCOPE=microbatch requires ACCUM_GRAD_BATCHES=1." >&2
      exit 1
    fi
    if [[ "${MEMORY_UPDATE_SCOPE}" != "microbatch" ]]; then
      echo "ERROR: ADVERSARY_SCOPE=microbatch requires MEMORY_UPDATE_SCOPE=microbatch." >&2
      exit 1
    fi
    GRAPH_BATCH=$((NUM_NODES * GPUS_PER_NODE * BATCH_SIZE))
    ;;
  optimizer_step)
    if [[ "${MEMORY_UPDATE_SCOPE}" != "optimizer_step_delayed" ]]; then
      echo "ERROR: ADVERSARY_SCOPE=optimizer_step requires MEMORY_UPDATE_SCOPE=optimizer_step_delayed." >&2
      exit 1
    fi
    GRAPH_BATCH=$((NUM_NODES * GPUS_PER_NODE * BATCH_SIZE * ACCUM_GRAD_BATCHES))
    ;;
  *)
    echo "ERROR: unsupported ADVERSARY_SCOPE=${ADVERSARY_SCOPE}; expected microbatch or optimizer_step." >&2
    exit 1
    ;;
esac

if [[ -n "${MEMORY_QUEUE_BASE:-}" ]]; then
  QUEUE_BASE="${MEMORY_QUEUE_BASE}"
else
  QUEUE_BASE=16384
fi
QUEUE_SMALL=$((QUEUE_BASE / 2))
QUEUE_LARGE=$((QUEUE_BASE * 4))
if [[ "${QUEUE_SMALL}" -lt 512 ]]; then
  QUEUE_SMALL=512
fi

submit_job() {
  local label="$1"
  local config="$2"
  local walltime="$3"
  local exports="$4"
  shift 4 || true

  if [[ -n "${RESUBMIT_LABELS:-}" ]]; then
    local allowed=0
    local token
    IFS=',' read -ra _resubmit_tokens <<< "${RESUBMIT_LABELS}"
    for token in "${_resubmit_tokens[@]}"; do
      if [[ "${label}" == "${token}" ]]; then
        allowed=1
        break
      fi
    done
    if [[ "${allowed}" -eq 0 ]]; then
      return 0
    fi
  fi

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

relaxed_flow_exports() {
  local k="$1"
  local inner_steps="$2"
  local tau_flow="$3"
  local beta="$4"
  local alpha="$5"
  printf 'GEODRO_K=%s,INNER_STEPS=%s,TAU_FLOW=%s,BETA=%s,ALPHA_MAX=%s,WARMUP_FRACTION=0.0,RAMP_FRACTION=0.0,CLAMP_ACTIVATION_FAIL=0.50,ESS_MIN_RATIO=0.005,MAX_P_FACTOR_FAIL=256,P_CAP=0.01' \
    "${k}" "${inner_steps}" "${tau_flow}" "${beta}" "${alpha}"
}

memory_base_exports() {
  local queue_capacity="$1"
  local k_guard="$2"
  local top_m="$3"
  local threshold_mode="$4"
  local null_quantile="$5"
  local extra_edges="$6"
  local edge_ratio="$7"
  local witness_mode="$8"
  local extra="${9:-}"

  printf 'MEMORY_QUEUE_CAPACITY=%s,MEMORY_TOP_M=%s,MEMORY_K_SIGMA=%s,MEMORY_K_GUARD=%s,MEMORY_RETRIEVAL_CHUNK_SIZE=512,MEMORY_WITNESS_THRESHOLD_MODE=%s,MEMORY_WITNESS_NULL_QUANTILE=%s,MEMORY_EXTRA_EDGES_PER_NODE_MAX=%s,MEMORY_ADDED_EDGE_RATIO_MAX=%s,WITNESS_SCORE_MODE=%s%s' \
    "${queue_capacity}" "${top_m}" "${top_m}" "${k_guard}" "${threshold_mode}" \
    "${null_quantile}" "${extra_edges}" "${edge_ratio}" "${witness_mode}" "${extra}"
}

submit_shape_gate() {
  local flow
  flow="$(relaxed_flow_exports 24 20 0.025 0.2 0.03)"
  local memory
  memory="$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"

  submit_job \
    "v2-shape-ch-mem-b${BATCH_SIZE}" \
    "geodro/geodro_jepa_v2_coherent_hardness_batch_memory_optstep_imagenet100ctrl" \
    "${SHAPE_WALLTIME:-02:00:00}" \
    "MAX_EPOCHS=1,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2,${flow},${memory}"

  submit_job \
    "v2-shape-gt-mem-b${BATCH_SIZE}" \
    "geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl" \
    "${SHAPE_WALLTIME:-02:00:00}" \
    "MAX_EPOCHS=1,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2,${flow},${memory}"
}

submit_activation_grid() {
  local configs=(
    "ch:geodro/geodro_jepa_v2_coherent_hardness_imagenet100ctrl_smoke"
    "gt:geodro/geodro_jepa_v2_graph_transport_imagenet100ctrl_smoke"
  )
  local presets=(
    "default-strict:"
    "act-k16-a01:$(relaxed_flow_exports 16 20 0.025 0.2 0.01)"
    "act-k24-a03:$(relaxed_flow_exports 24 20 0.025 0.2 0.03)"
    "act-k16-beta03:$(relaxed_flow_exports 16 20 0.025 0.3 0.03)"
    "act-smooth:$(relaxed_flow_exports 24 40 0.0125 0.2 0.03)"
    "act-a05:$(relaxed_flow_exports 24 20 0.025 0.2 0.05)"
  )

  local cell config preset_name preset_exports label exports
  for cell_config in "${configs[@]}"; do
    cell="${cell_config%%:*}"
    config="${cell_config#*:}"
    for preset in "${presets[@]}"; do
      preset_name="${preset%%:*}"
      preset_exports="${preset#*:}"
      label="v2-${cell}-${preset_name}-b${BATCH_SIZE}"
      exports="MAX_EPOCHS=10,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2"
      if [[ -n "${preset_exports}" ]]; then
        exports="${exports},${preset_exports}"
      fi
      submit_job "${label}" "${config}" "${GRID_WALLTIME:-06:00:00}" "${exports}"
    done
  done
}

submit_memory_grid() {
  local flow
  flow="$(relaxed_flow_exports 24 20 0.025 0.2 0.03)"
  local configs=(
    "ch:geodro/geodro_jepa_v2_coherent_hardness_batch_memory_optstep_imagenet100ctrl"
    "gt:geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl"
  )
  local variants=(
    "mem-base:$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"
    "mem-cap-small:$(memory_base_exports "${QUEUE_SMALL}" 64 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"
    "mem-cap-large:$(memory_base_exports "${QUEUE_LARGE}" 64 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"
    "mem-kguard32:$(memory_base_exports "${QUEUE_BASE}" 32 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"
    "mem-kguard128:$(memory_base_exports "${QUEUE_BASE}" 128 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"
    "mem-q99:$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.99 2 0.25 specificity_weighted_hellinger)"
    "mem-budget1:$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.95 1 0.125 specificity_weighted_hellinger)"
    "mem-raw:$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.95 2 0.25 raw_hellinger)"
  )

  local cell_config cell config variant variant_name variant_exports label
  for cell_config in "${configs[@]}"; do
    cell="${cell_config%%:*}"
    config="${cell_config#*:}"
    for variant in "${variants[@]}"; do
      variant_name="${variant%%:*}"
      variant_exports="${variant#*:}"
      label="v2-${cell}-${variant_name}-b${BATCH_SIZE}"
      submit_job "${label}" "${config}" "${GRID_WALLTIME:-06:00:00}" \
        "MAX_EPOCHS=10,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2,${flow},${variant_exports}"
    done
  done

  submit_job \
    "v2-gt-mem-flow-k16-a01-b${BATCH_SIZE}" \
    "geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl" \
    "${GRID_WALLTIME:-06:00:00}" \
    "MAX_EPOCHS=10,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2,$(relaxed_flow_exports 16 20 0.025 0.2 0.01),$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"

  submit_job \
    "v2-gt-mem-flow-beta03-b${BATCH_SIZE}" \
    "geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl" \
    "${GRID_WALLTIME:-06:00:00}" \
    "MAX_EPOCHS=10,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2,$(relaxed_flow_exports 16 20 0.025 0.3 0.03),$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"

  submit_job \
    "v2-gt-mem-flow-smooth-b${BATCH_SIZE}" \
    "geodro/geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl" \
    "${GRID_WALLTIME:-06:00:00}" \
    "MAX_EPOCHS=10,RUN_PREFLIGHT=1,PREFLIGHT_STEPS=2,$(relaxed_flow_exports 24 40 0.0125 0.2 0.03),$(memory_base_exports "${QUEUE_BASE}" 64 64 shuffled_null_quantile 0.95 2 0.25 specificity_weighted_hellinger)"
}

# Labels that reached full 10 epochs in the 2026-05-28 recovery wave (skip on resubmit).
RECOVERY_COMPLETED_LABELS=(
  "v2-ch-default-strict-b${BATCH_SIZE}"
  "v2-ch-act-k16-beta03-b${BATCH_SIZE}"
  "v2-gt-default-strict-b${BATCH_SIZE}"
  "v2-ch-mem-base-b${BATCH_SIZE}"
  "v2-ch-mem-budget1-b${BATCH_SIZE}"
  "v2-gt-mem-base-b${BATCH_SIZE}"
  "v2-gt-mem-budget1-b${BATCH_SIZE}"
)

# Priority timeout re-runs: healthy alpha/fallback @ ~ep8 with best partial linear.
RECOVERY_PRIORITY_TIMEOUT_LABELS=(
  "v2-ch-act-k24-a03-b${BATCH_SIZE}"
  "v2-ch-act-smooth-b${BATCH_SIZE}"
  "v2-gt-act-a05-b${BATCH_SIZE}"
  "v2-ch-mem-cap-small-b${BATCH_SIZE}"
  "v2-ch-mem-q99-b${BATCH_SIZE}"
  "v2-gt-mem-q99-b${BATCH_SIZE}"
  "v2-gt-mem-cap-small-b${BATCH_SIZE}"
)

resubmit_timeout_labels() {
  local mode="${1:-all}"
  local -a labels=()
  local label completed skip=0

  if [[ "${mode}" == "priority" ]]; then
    labels=("${RECOVERY_PRIORITY_TIMEOUT_LABELS[@]}")
  else
    local configs presets variants cell preset preset_name variant_name
    configs=(
      "ch:geodro/geodro_jepa_v2_coherent_hardness_imagenet100ctrl_smoke"
      "gt:geodro/geodro_jepa_v2_graph_transport_imagenet100ctrl_smoke"
    )
    presets=(
      "default-strict:"
      "act-k16-a01:$(relaxed_flow_exports 16 20 0.025 0.2 0.01)"
      "act-k24-a03:$(relaxed_flow_exports 24 20 0.025 0.2 0.03)"
      "act-k16-beta03:$(relaxed_flow_exports 16 20 0.025 0.3 0.03)"
      "act-smooth:$(relaxed_flow_exports 24 40 0.0125 0.2 0.03)"
      "act-a05:$(relaxed_flow_exports 24 20 0.025 0.2 0.05)"
    )
    for cell_config in "${configs[@]}"; do
      cell="${cell_config%%:*}"
      for preset in "${presets[@]}"; do
        preset_name="${preset%%:*}"
        labels+=("v2-${cell}-${preset_name}-b${BATCH_SIZE}")
      done
    done
    variants=(
      "mem-base"
      "mem-cap-small"
      "mem-cap-large"
      "mem-kguard32"
      "mem-kguard128"
      "mem-q99"
      "mem-budget1"
      "mem-raw"
    )
    for cell in ch gt; do
      for variant_name in "${variants[@]}"; do
        labels+=("v2-${cell}-${variant_name}-b${BATCH_SIZE}")
      done
    done
    labels+=(
      "v2-gt-mem-flow-k16-a01-b${BATCH_SIZE}"
      "v2-gt-mem-flow-beta03-b${BATCH_SIZE}"
      "v2-gt-mem-flow-smooth-b${BATCH_SIZE}"
    )
  fi

  local -a filtered=()
  for label in "${labels[@]}"; do
    skip=0
    for completed in "${RECOVERY_COMPLETED_LABELS[@]}"; do
      if [[ "${label}" == "${completed}" ]]; then
        skip=1
        break
      fi
    done
    if [[ "${skip}" -eq 0 ]]; then
      filtered+=("${label}")
    fi
  done
  RESUBMIT_LABELS="$(IFS=','; echo "${filtered[*]}")"
  export RESUBMIT_LABELS
  GRID_WALLTIME="${GRID_WALLTIME:-06:00:00}"
  export GRID_WALLTIME
  echo "Resubmitting ${#filtered[@]} recovery labels with walltime=${GRID_WALLTIME}"
  submit_activation_grid
  submit_memory_grid
}

case "${ACTION}" in
  shape_gate)
    submit_shape_gate
    ;;
  activation_grid)
    submit_activation_grid
    ;;
  memory_grid)
    submit_memory_grid
    ;;
  all)
    submit_shape_gate
    submit_activation_grid
    submit_memory_grid
    ;;
  resubmit_timeouts)
    resubmit_timeout_labels all
    ;;
  resubmit_priority)
    resubmit_timeout_labels priority
    ;;
  *)
    echo "ERROR: unknown action '${ACTION}'." >&2
    echo "Valid actions: shape_gate, activation_grid, memory_grid, all, resubmit_timeouts, resubmit_priority" >&2
    exit 1
    ;;
esac
