# NOTICE

This repository — **geo-dro-jepa** — is the reference implementation accompanying my Master's thesis *Geometry-Aware Robust Aggregation for Self-Supervised Representation Learning* (with LMU Munich, supervised by Dr. Mina Rezaei).

It bundles two third-party Python packages that have been **adapted and re-distributed** here so the project builds from a single clone. This file records the upstream credit, license terms, and the modifications introduced for the thesis.

---

## 1. `lejepa/`

> *LeJEPA — Lean Joint-Embedding Predictive Architecture*

- **Upstream repository:** https://github.com/galilai-group/lejepa
- **Original author:** Randall Balestriero (`rbalestr@brown.edu`)
- **Paper:** Balestriero, *LeJEPA: Lean Joint-Embedding Predictive Architecture*, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0). The full license text is preserved at [`lejepa/LICENSE`](lejepa/LICENSE). **This license restricts commercial use**; any downstream redistribution of the bundled `lejepa/` directory — or of derivative works that include it — must comply with the CC-BY-NC 4.0 terms.

### Modifications in this repository

- `lejepa/pyproject.toml`: placeholder URL fields (`https://example.com`, `https://github.com/me/spam.git`, …) and the `description = "ToDo"` field were replaced with the real upstream URL and a one-line description. No functional change.
- `lejepa/README.md`: replaced with a short attribution stub. The full upstream README (with benchmark tables and figures) is available at the upstream repository linked above.
- A subset of upstream auxiliaries was omitted to keep this thesis distribution code-only: `eval/` (demo gifs/PNGs), `figures/` (paper plotting scripts), `paper/LeJEPA-paper.pdf`, `scripts/` (ablation launchers), `setup.py` (legacy duplicate of `pyproject.toml`), `MINIMAL.md`. Source code under `lejepa/lejepa/` and the `tests/` suite are unchanged.

### Suggested citation

```bibtex
@article{balestriero2025lejepa,
  title  = {LeJEPA: Lean Joint-Embedding Predictive Architecture},
  author = {Balestriero, Randall},
  journal = {arXiv preprint arXiv:2511.08544},
  year   = {2025},
}
```

---

## 2. `stable-pretraining/`

> *Stable Pretraining — a PyTorch Lightning / Hydra framework for self-supervised learning*

- **Upstream repository:** https://github.com/galilai-group/stable-pretraining
- **Original authors:** the `rbalestr-lab` organization
- **License:** MIT. The full license text is preserved at [`stable-pretraining/LICENSE`](stable-pretraining/LICENSE).

### Modifications in this repository

The thesis contribution sits **inside** this package. Specifically:

1. **Added** `stable_pretraining/geodro_lejepa/` (~5,000 LOC) — the GeoDRO-LeJEPA v1.1 (Reliability-Gated Coherent-Hardness) and v2.2 (GraphTransport / memory-witnessed) implementation. Modules: `loss.py`, `forward.py`, `graph.py`, `flow.py`, `gating.py`, `utility.py`, `witness.py`, `memory.py`, `controlled.py`, `optimizer_step.py`, `prediction.py`, `distributed.py`, `types.py`.
2. **Added** controlled-corruption datasets `CIFAR10ControlledCorruption` and `ImageNet100ControlledCorruption` (in `geodro_lejepa/controlled.py`).
3. **Added** unit tests for the contribution: `stable_pretraining/tests/unit/test_geodro_lejepa.py`, `test_geodro_lejepa_eval.py`.
4. **Added** Hydra configs under `examples/geodro/` covering every reported run (debug, smoke, sanity, main, optimizer-step accumulation, ERM/alpha-0 reference points, v2.2 GraphTransport variants).
5. **Added** SLURM launchers under `scripts/slurm/` (one per config), the `download_datasets.sbatch` prewarm job, the `run_full_eval_suite.sh` driver, and `geodro_lejepa_*_eval.sbatch` downstream-eval launchers.
6. **Added** evaluation scripts under `scripts/`: `geodro_lejepa_eval.py`, `linear_eval.py`, `few_shot_eval.py`, `collect_full_eval_metrics.py`, `validate_full_eval_matrix.py`, `_eval_metrics.py`.

The following upstream auxiliaries were omitted to keep this thesis distribution code-only and focused: `benchmarks/` (comparative SimCLR/BYOL/NNCLR/Barlow impls), `docs/` (Sphinx tree), `.github/` (CI workflows + PR template), `examples/sanity/`, top-level upstream demos under `examples/` (multi-probe, supervised, SimCLR, W&B utilities), upstream community docs (`CODE_OF_CONDUCT.md`, `TESTING.md`, `RELEASES.rst`, `.pre-commit-config.yaml`, `codecov.yml`), and `assets/` font/CLI helpers (only the runtime-loaded `static_*.json` model registries are retained). From `examples/baselines/`, only `lejepa_vits8_imagenet100.yaml` is retained because the GeoDRO ImageNet-100 configs compose it via Hydra defaults; the other baseline YAMLs and their SLURM launchers are dropped. Generated artifacts (`outputs/`, `logs/`, `wandb/`, `multirun/`, `snapshots/`, `.pytest_cache/`) are not redistributed.

The upstream framework code under `stable_pretraining/{module,forward,manager,config,cli,backbone,callbacks,data,losses,optim,utils}.py` and its `tests/{unit,integration,scripts}/` suites are unchanged from upstream.

`stable-pretraining/README.md` was replaced with a short attribution stub. The full upstream README, including framework tutorials, is available at the upstream repository linked above.

---

## 3. Original contribution (root `LICENSE`)

The original work added by this thesis — the `geodro_lejepa/` subpackage, its configs, tests, scripts, and the project-level scaffolding (`README.md`, `NOTICE.md`, `LICENSE`, `.gitignore`, per-package README stubs) — is licensed under MIT. See the root [`LICENSE`](LICENSE). That license **does not** modify the terms of the bundled `lejepa/LICENSE` (CC-BY-NC 4.0) or `stable-pretraining/LICENSE` (MIT), which continue to govern the code in their respective directories.
