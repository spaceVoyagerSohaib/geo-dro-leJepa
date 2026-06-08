# GeoDRO configs

This directory contains the Hydra configs for **GeoDRO-LeJEPA** (v1.1 Reliability-Gated Coherent-Hardness, and v2.2 Graph-Transport / memory-witnessed). Every config consumed by the SLURM launchers under `../../scripts/slurm/` lives here.

For the implementation surface itself, see `stable_pretraining.geodro_lejepa` — `loss.py` (`CoherentHardnessGeoDROLeJEPALoss` for v1.1, `GraphTransportGeoDROJEPALoss` for v2.2; `GeoDROLeJEPALoss` is a deprecated alias for coherent-hardness only), `graph.py`, `flow.py`, `gating.py`, `utility.py`, `witness.py`, `memory.py`, `controlled.py`.

## Smoke

`geodro_lejepa_fake_data_smoke.yaml` is the first smoke config: it runs on `torchvision.datasets.FakeData`, uses `accumulate_grad_batches: 1`, and exercises the training-only GeoDRO weight path without changing inference. `geodro_jepa_graph_transport_fake_data_smoke.yaml` is the v2.2 GraphTransport batch-only fake-data smoke.

## CIFAR-10 controlled debug

The `geodro_lejepa_cifar10ctrl_*_debug.yaml` configs are the CIFAR-10 readiness surface:

- `geodro_lejepa_cifar10ctrl_erm_debug.yaml` — baseline LeJEPA ERM anchor.
- `geodro_lejepa_cifar10ctrl_alpha0_debug.yaml` — GeoDRO plumbing with `alpha_max: 0.0` for ERM parity.
- `geodro_lejepa_cifar10ctrl_v1_debug.yaml` — canonical v1.1 reliability-gated debug run.

They use a class-conditioned coherent corruption subgroup plus an isolated single-view corruption tag. Those tags are diagnostics-only and are not consumed by graph construction, utility construction, flow, or weighting.

The CIFAR-10 controlled configs load `uoft-cs/cifar10` through `stable_pretraining.data.datasets.HFDataset`. Cluster launchers expect the v1 dataset prewarm sentinel under `${MCMLSCRATCH}/datasets/geodro_v1/cifar10/`. Run `../../scripts/slurm/download_datasets.sbatch` first, or set `REQUIRE_GEODRO_DATASET_PREWARM=0` only for disposable local debugging.

Run the single-GPU sanity launcher with:

```bash
cd stable-pretraining
sbatch scripts/slurm/geodro_lejepa_cifar10ctrl_sanity.sbatch
```

## ImageNet-100 controlled main matrix

The `geodro_lejepa_imagenet100ctrl_*` configs are the v1 main experiment surface. They inherit a ViT-S/8 LeJEPA baseline and add controlled training diagnostics:

- 30% deterministic class-index coherent hard region.
- 5% deterministic isolated bad-view artifacts on `global_1`.
- 2 global + 6 local LeJEPA views.
- Explicit adversary-scope metadata:
  - `microbatch` — graph batch is `batch_per_gpu * nodes * gpus_per_node`; `ACCUM_GRAD_BATCHES` must stay at 1.
  - `optimizer_step` — graph batch is `batch_per_gpu * nodes * gpus_per_node * ACCUM_GRAD_BATCHES`; this uses a two-pass step that solves one graph over the accumulated optimizer-step batch and replays microbatches with fixed weight slices.

Before submitting ImageNet-100 controlled pretraining, prewarm the v1 datasets with a single-process CPU job:

```bash
cd stable-pretraining
sbatch scripts/slurm/download_datasets.sbatch
```

The prewarm job writes per-dataset caches under `${MCMLSCRATCH}/datasets/geodro_v1/` and records `.prewarm_complete.json` sentinels. The ImageNet-100 launchers require the `imagenet100` sentinel before starting DDP; this prevents multiple ranks or jobs from concurrently building a shared Hugging Face Arrow cache. Re-run the prewarm only after cache deletion, dataset-scope changes, or sentinel invalidation.

### Available variants

v1 (Reliability-Gated Coherent-Hardness):

- `geodro_lejepa_imagenet100ctrl_erm.yaml` — ERM anchor.
- `geodro_lejepa_imagenet100ctrl_erm_optstep_accum.yaml` — ERM with optimizer-step accumulation.
- `geodro_lejepa_imagenet100ctrl_alpha0.yaml` — GeoDRO plumbing with `alpha_max: 0.0`.
- `geodro_lejepa_imagenet100ctrl_v1.yaml` — canonical v1.1.
- `geodro_lejepa_imagenet100ctrl_base.yaml` — v1.1 base.
- `geodro_lejepa_imagenet100ctrl_raw_loss.yaml`, `…_random_graph.yaml`, `…_fully_connected.yaml`, `…_max_union.yaml` — ablations.
- `geodro_lejepa_imagenet100ctrl_v1_optstep_accum.yaml` — v1.1 with optimizer-step accumulation.
- `geodro_lejepa_imagenet100ctrl_v1_optstep_accum_activation.yaml` — H100 activation pilot (k16, immediate warmup, relaxed reliability gates).
- `geodro_lejepa_imagenet100ctrl_v1_optstep_accum_activation_memory.yaml` — activation config with `batch_memory`, `memory_witnessed`, and delayed optimizer-step memory updates.

v2.2 (Graph-Transport / memory-witnessed):

- `geodro_jepa_v2_coherent_hardness_imagenet100ctrl_smoke.yaml`, `…_batch_memory_imagenet100ctrl_smoke.yaml` — coherent-hardness anchors with optional batch memory.
- `geodro_jepa_v2_graph_transport_imagenet100ctrl_smoke.yaml`, `…_batch_memory_imagenet100ctrl_smoke.yaml` — GraphTransport with optional batch memory.
- `geodro_jepa_v2_coherent_hardness_batch_memory_optstep_imagenet100ctrl.yaml`, `geodro_jepa_v2_graph_transport_batch_memory_optstep_imagenet100ctrl.yaml` — batch memory with optimizer-step delayed memory updates.
- `geodro_jepa_coherent_hardness_batch_memory_fake_data_smoke.yaml`, `geodro_jepa_graph_transport_batch_memory_fake_data_smoke.yaml` — fake-data smoke variants.

Full main jobs use `geodro_lejepa_imagenet100ctrl_main.sbatch`. Each submitted main job first runs a small preflight at the target batch shape; if that check passes, the same job continues into the requested training run.

## Optimizer-step accumulation

Use `ACCUM_GRAD_BATCHES=1` for all `microbatch` configs. If the direct graph batch does not fit in memory, switch to an explicit `optimizer_step` config instead of turning on naive accumulation. For `optimizer_step` configs, preserve the intended graph size with:

```text
graph_batch = batch_per_gpu * nodes * gpus_per_node * ACCUM_GRAD_BATCHES
```

Examples that preserve a 2048-node optimizer-step graph:

```text
1 node x 4 GPUs x 128 samples/GPU x 4 accumulation steps = 2048
1 node x 2 GPUs x 128 samples/GPU x 8 accumulation steps = 2048
```

Each main run writes `run_manifest.json` in its Hydra output directory and adds the same trace summary to W&B under `run_trace`.

## Continuation / resume

When a 48h job times out before `max_epochs`, submit a **continuation** with the same config and hyperparameter overrides as segment 1. Checkpoints are written to `${hydra:run.dir}/checkpoints/` (`last.ckpt` plus best `epoch_epoch=XXXX.ckpt` from `ModelCheckpoint`).

Rules:

- `MAX_EPOCHS` is the **final** target (e.g. `400`), not "400 more". Lightning continues from the stored epoch in the checkpoint.
- Match the original run exactly: `CONFIG_NAME`, `NUM_NODES`, `BATCH_SIZE`, `ACCUM_GRAD_BATCHES`, `ADVERSARY_SCOPE`, and every `GEODRO_*` override you used on the first segment.
- Pass `RESUME_CKPT_PATH` to the absolute or project-relative path of `last.ckpt` (or the best epoch checkpoint).
- Use `RESUME_WEIGHTS_ONLY=false` (default) for full resume (optimizer, LR schedule, online linear probe / kNN callback state).
- Set `RUN_PREFLIGHT=0` on continuation (or omit it: preflight is auto-disabled when a valid `RESUME_CKPT_PATH` is provided).
- ERM and no-memory Geo runs do not need `feature_memory` keys in the checkpoint. Memory-enabled runs require a checkpoint that includes `feature_memory.*` state.

**Full pretraining partition:** use `geodro_lejepa_imagenet100ctrl_full.sbatch` (`lrz-hgx-h100-94x4`, 1×4 H100, 48h). Keep `geodro_lejepa_imagenet100ctrl_main.sbatch` for short diagnostics.

**Launcher env vars** (`geodro_lejepa_imagenet100ctrl_main.sbatch` / `_full.sbatch`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `RESUME_CKPT_PATH` | (empty) | Checkpoint to load; omit for fresh runs. |
| `RESUME_WEIGHTS_ONLY` | `false` | Full resume vs weights-only. |
| `AUTO_RESUME_FALLBACK` | `true` | Retry with weights-only on `state_dict` mismatch. |

**Verify after submit** (in the new SLURM log):

```bash
grep -E "CALLING trainer.fit|Epoch [0-9]+, global step" logs/slurm-<run-label>-<jobid>.out | head -20
```

Expect `CALLING trainer.fit with ckpt_path='...'` and the first completed epoch to be `checkpoint_epoch + 1`, not `0`.

**Optional 1-GPU sanity check** (~1h) before a full continuation:

```bash
cd stable-pretraining
sbatch --export=ALL,RESUME_CKPT_PATH=outputs/<run-label>/checkpoints/last.ckpt,EXTRA_EPOCHS=2 \
    scripts/slurm/geodro_lejepa_imagenet100ctrl_resume_sanity.sbatch
```

**Example: ERM full continuation:**

```bash
cd stable-pretraining
sbatch --job-name=erm-full-cont \
    --export=ALL,\
CONFIG_NAME=geodro/geodro_lejepa_imagenet100ctrl_erm_optstep_accum,\
RUN_LABEL=erm-full-b128-a4-e400-cont,\
NUM_NODES=1,GPUS_PER_NODE=4,TASKS_PER_NODE=4,TRAINER_DEVICES=4,\
BATCH_SIZE=128,VAL_BATCH_SIZE=128,ACCUM_GRAD_BATCHES=4,\
MAX_EPOCHS=400,RUN_PREFLIGHT=0,\
RESUME_CKPT_PATH=outputs/<run-label>/checkpoints/last.ckpt,\
RESUME_WEIGHTS_ONLY=false,AUTO_RESUME_FALLBACK=true \
    scripts/slurm/geodro_lejepa_imagenet100ctrl_full.sbatch
```

**Example: Geo v1 full continuation** (repeat all Geo overrides):

```bash
cd stable-pretraining
sbatch --job-name=geo-full-n4-cont \
    --export=ALL,\
CONFIG_NAME=geodro/geodro_lejepa_imagenet100ctrl_v1_optstep_accum_activation,\
RUN_LABEL=geo-full-n4-k24-inner40-b128-a4-e400-cont,\
NUM_NODES=1,GPUS_PER_NODE=4,TASKS_PER_NODE=4,TRAINER_DEVICES=4,\
BATCH_SIZE=128,VAL_BATCH_SIZE=128,ACCUM_GRAD_BATCHES=4,\
MAX_EPOCHS=400,RUN_PREFLIGHT=0,\
GEODRO_K=24,INNER_STEPS=40,TAU_FLOW=0.0125,BETA=0.2,ALPHA_MAX=0.03,\
WARMUP_FRACTION=0.0,RAMP_FRACTION=0.0,CLAMP_ACTIVATION_FAIL=0.55,\
ESS_MIN_RATIO=0.003,MAX_P_FACTOR_FAIL=256,P_CAP=0.0035,\
RESUME_CKPT_PATH=outputs/<run-label>/checkpoints/last.ckpt,\
RESUME_WEIGHTS_ONLY=false,AUTO_RESUME_FALLBACK=true \
    scripts/slurm/geodro_lejepa_imagenet100ctrl_full.sbatch
```

Chain additional 48h segments if needed: each time, point `RESUME_CKPT_PATH` at the latest `last.ckpt` from the previous segment's output directory.

## Activation diagnostics

The optimizer-step activation pilots are mechanism diagnostics, not canonical full results. Common success checks:

- `train/geodro/Weight_warmup_multiplier_epoch = 1`
- `train/geodro/Weight_alpha_epoch > 0`
- `train/geodro/Weight_fallback_epoch < 1`
- `train/geodro/Weight_max_p_epoch > 1 / 2048`
- `train/pred_loss_minus_erm_epoch` is nonzero
- no NaN/OOM and no sharp probe collapse versus the ERM pilot

Promote only a run that is active and stable to a longer pilot.

## Post-pretraining evaluation

After a full ERM or canonical v1 checkpoint exists, run downstream evals. The canonical 9-mode suite (IN-100ctrl, IN-100C, Waterbirds, IN-Sketch/R/A/O, CelebA, Camelyon17) is wrapped in `../../scripts/slurm/run_full_eval_suite.sh`. Individual sidecars use one GPU, train frozen linear probes with the 100-epoch protocol, and link W&B runs back to `PRETRAIN_RUN_ID`. Single-mode reruns:

```bash
cd stable-pretraining
export PRETRAIN_RUN_ID=<run-id>
export PRETRAIN_OUTPUT_DIR=<output-dir>
export METHOD=<erm-or-v1>

sbatch -p mcml-hgx-a100-80x4 --qos=mcml \
    --export=ALL,PRETRAIN_RUN_ID,PRETRAIN_OUTPUT_DIR,METHOD \
    scripts/slurm/geodro_lejepa_imagenet100ctrl_eval.sbatch

sbatch -p mcml-hgx-a100-80x4 --qos=mcml \
    --export=ALL,PRETRAIN_RUN_ID,PRETRAIN_OUTPUT_DIR,METHOD \
    scripts/slurm/geodro_lejepa_imagenet100c_eval.sbatch

sbatch -p mcml-hgx-a100-80x4 --qos=mcml \
    --export=ALL,PRETRAIN_RUN_ID,PRETRAIN_OUTPUT_DIR,METHOD \
    scripts/slurm/geodro_lejepa_waterbirds_eval.sbatch
```

Outputs are written under `${PRETRAIN_OUTPUT_DIR}/eval/<mode>-<job-id>/` with `metrics.json`, `args.json`, and `probe.pt`. The ImageNet-C and Waterbirds defaults use Hugging Face datasets cached on MCML scratch through `.env.local`.
