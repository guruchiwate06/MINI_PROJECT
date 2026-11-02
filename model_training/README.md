# Rail-VIVID model training

All new model-pipeline files are organized here on the `model-training` branch. The historical experiments at the repository root remain intact.

- [Training and evaluation guide](TRAINING_PIPELINE.md)
- [Research audit and evidence](research_audit/README.md)
- [Implementation status](research_audit/PIPELINE_BUILD_STATUS.md)
- [Fitting-free validation record](reports/pipeline_validation.json)
- [Default training config](configs/pipeline.json)
- [117-task benchmark definition](configs/benchmark_suite.json)

```text
model_training/
    railvivid/          Pipeline package and CLI
    configs/            Data, model, alignment and benchmark settings
    tests_pipeline/     Fitting-free tests
    scripts/            Validation and explicit GPU-training launcher
    src/                Snapshot of the legacy modules imported by the package
    research_audit/     Audit reports, evidence and reproducibility utilities
    reports/            Pipeline validation results
    pyproject.toml      Editable installation and CLI entry point
    artifacts/          Local-only caches, plans and bundles; excluded from Git
```

From the repository root:

```powershell
cd model_training
python -m pip install -e ".[vision,dev]"
python -m railvivid doctor --config configs/pipeline.json
python scripts/validate_pipeline.py
```

Install an appropriate CUDA-enabled PyTorch build on the RTX 5070 computer before training. The model-training commands are explicit, default to requiring CUDA and do not silently fall back to CPU. No models were trained during local implementation or this folder reorganization.

Data and model artifacts are not in Git: raw recordings/images, memory maps, feature caches, pretrained weights, checkpoints, generated plans and zip bundles stay local. Run `plan` on the training computer to resolve its local paths. The pinned model configuration and full benchmark generator are versioned here.

The `src/` snapshot preserves existing model/feature imports so this folder can be installed independently. Original root-level modules/results are unchanged. Historical audit utilities that reproduce old experiments use the repository-root implementation; they are separate from the fitting-free pipeline validation command above.
