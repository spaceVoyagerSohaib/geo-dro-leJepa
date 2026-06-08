# stable-pretraining

PyTorch + Lightning + Hydra framework for self-supervised pretraining. This package provides the modular `Module`/`Manager`/`Callbacks` scaffolding, dataset wiring, optimizer/scheduler factories, and the `stable_pretraining.run` Hydra entry point used by all GeoDRO-LeJEPA experiments.

The `stable_pretraining/geodro_lejepa/` subpackage and the `examples/geodro/` configs are the **original contribution** of the GeoDRO-LeJEPA thesis (LMU Munich); everything else under `stable_pretraining/` is upstream code redistributed for reproducibility.

## Attribution

- **Upstream:** https://github.com/galilai-group/stable-pretraining
- **Original authors:** the `rbalestr-lab` organization
- **License:** MIT — see [`LICENSE`](LICENSE).

See the project root [`README.md`](../README.md) and [`NOTICE.md`](../NOTICE.md) for the full attribution statement, install order, and the list of modifications introduced for this thesis.
