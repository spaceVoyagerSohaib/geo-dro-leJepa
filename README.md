# geo-dro-leJepa

**Geometry-Aware Robust Aggregation for Self-Supervised Representation Learning**  This repository the code, Hydra configs, and launcher scripts needed to reproduce the GeoDRO-LeJEPA experiments — 

The original contribution of this work is the `stable_pretraining/geodro_lejepa/` subpackage and its associated configs, tests, and SLURM launchers. Two upstream packages — `lejepa` and `stable-pretraining` — are re-distributed here so the project is buildable from a single clone.

## Attribution

This repository **adapts and re-distributes** two third-party packages. Full credit for that code goes to the upstream authors. See [`NOTICE.md`](NOTICE.md) for the complete attribution statement (authors, licenses, list of modifications).

| Package | Upstream | License |
|---|---|---|
| `lejepa/` | https://github.com/galilai-group/lejepa | CC-BY-NC 4.0 (non-commercial) |
| `stable-pretraining/` | https://github.com/galilai-group/stable-pretraining | MIT |

The original GeoDRO-LeJEPA contribution is licensed under MIT — see [`LICENSE`](LICENSE).

## Repository layout

```text
geo-dro-jepa/
├── lejepa/                          # vendored statistical primitives (SIGReg, normality tests)
│   ├── lejepa/                      # source: univariate/, multivariate/
│   ├── tests/                       # deterministic CPU pytest suite
│   └── pyproject.toml
└── stable-pretraining/              # vendored framework + the GeoDRO contribution
    ├── stable_pretraining/
    │   ├── geodro_lejepa/           # GeoDRO-LeJEPA v1.1 + v2.2 (original contribution)
    │   ├── module.py, manager.py, forward.py, ...
    │   └── tests/                   # unit + integration tests
    ├── examples/geodro/             # Hydra configs for every reported run
    ├── scripts/                     # eval harness + SLURM launchers
    └── pyproject.toml
```

## Install

`stable-pretraining` depends on `lejepa`, so install order matters:

```bash
pip install -e ./lejepa
pip install -e "./stable-pretraining[dev]"
pip install -e "./stable-pretraining[eval]"   # optional: adds wilds for Camelyon17
```

## Quick start

All commands below are run from `stable-pretraining/`.

```bash
# Fast deterministic CPU tests (units only)
python -m pytest -m unit

# Local fake-data smoke run for GeoDRO-LeJEPA
python -m stable_pretraining.run \
    --config-path examples/geodro \
    --config-name geodro_lejepa_fake_data_smoke \
    trainer.max_epochs=1 trainer.strategy=auto trainer.devices=1 trainer.num_nodes=1

# SLURM launchers (cluster)
sbatch scripts/slurm/geodro_lejepa_imagenet100ctrl_main.sbatch       # v1.1 main
sbatch scripts/slurm/geodro_jepa_v2_smoke.sbatch                     # v2.2 smoke
sbatch scripts/slurm/download_datasets.sbatch                        # prewarm dataset cache
```

Override Hydra parameters from the CLI as usual: append `module.optimizer.lr=0.01 trainer.max_epochs=10`, etc.

## Method overview

The repository implements two variants of GeoDRO-LeJEPA, both exported from `stable_pretraining.geodro_lejepa.loss`:

- **v1.1 — Reliability-Gated Coherent-Hardness Aggregation** (`CoherentHardnessGeoDROLeJEPALoss`). Trains on raw LeJEPA prediction losses; adversarial weights are detached, derived from a robust, view-consistent, graph-coherent utility. Graph defaults: pre-projector global-view centers, mutual-kNN, self-tuned RBF. A reliability gate mixes finite-time GDRO flow weights with uniform ERM weights and fails closed to uniform on instability.
- **v2.2 — Graph-Transport / memory-witnessed geometry** (`GraphTransportGeoDROJEPALoss`). Current-sample-only graph in its first implementation; the `GeoDROFeatureMemoryQueue` in `memory.py` provides optional memory-witnessed geometry support and is checkpointed for deterministic resume.

The legacy alias `GeoDROLeJEPALoss` refers to the coherent-hardness behavior only.

## Environment variables

Local overrides live in `stable-pretraining/.env.local` (gitignored), based on the template at `stable-pretraining/.env.example`. Common fields:

- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_GROUP`, `WANDB_MODE` (`online`/`offline`)
- `MCMLSCRATCH` — cluster scratch root
- `GEODRO_DATASET_ROOT` — typically `${MCMLSCRATCH}/datasets/geodro_v1`
- `GEODRO_HF_HOME` — Hugging Face cache for the GeoDRO scope
- `HF_HOME`, `HF_DATASETS_CACHE` — non-GeoDRO cache overrides

For cluster runs, the dataset root must live on shared scratch (not `~/data`) so login and compute nodes see the same files.

## Datasets

GeoDRO v1 controlled-pretraining uses these public Hugging Face datasets:

- `ilee0022/ImageNet100` — ImageNet-100 controlled pretraining
- `WNJXYK/TTA-ImageNet-C`, `grodino/waterbirds` — v1 sidecars
- `uoft-cs/cifar10` — CIFAR-10 controlled sanity
- `ILSVRC/imagenet-1k` — gated; only used by ImageNet-1K configs

Multi-rank GeoDRO v1 SLURM launchers refuse to start until per-dataset `.prewarm_complete.json` sentinels are present under `${GEODRO_DATASET_ROOT}` to avoid HF cache races. Run `sbatch scripts/slurm/download_datasets.sbatch` once per cluster scratch to populate them.

## Outputs

A run produces:

- `stable-pretraining/outputs/<run-id>/{checkpoints,logs,.hydra}/` — Hydra outputs
- `stable-pretraining/logs/slurm-<job-name>-<job-id>.{out,err}` — SLURM streams
- W&B remote project (or `wandb/` local cache when `WANDB_MODE=offline`)

All four directories are gitignored.

## License

- The original GeoDRO-LeJEPA contribution: MIT — see [`LICENSE`](LICENSE).
- `lejepa/` (vendored): CC-BY-NC 4.0 — see [`lejepa/LICENSE`](lejepa/LICENSE). Non-commercial use only.
- `stable-pretraining/` (vendored): MIT — see [`stable-pretraining/LICENSE`](stable-pretraining/LICENSE).

The bundled CC-BY-NC 4.0 license on `lejepa/` is the strictest; downstream redistribution must respect its non-commercial term.
