# Rail-VIVID model training

GPU pilot recovery: [numerical fixes, profiling and a five-epoch FP32 handoff](RECOVERY.md). The first GPU run failed after one epoch. Preserve its evidence; diagnose and use a new experiment rather than resuming across changed code.

[Recovery verification](reports/track_ot_recovery_validation.json): 100 passing tests, full-model FP32/BF16 synthetic gradient checks without parameter updates, and a passing frontend production build. Actual RTX recovery remains unverified.

New: [TRACK-OT++ implementation and training-device handoff](TRACK_OT_PLUS_PLUS.md) adds unbalanced sequence matching, bounded drift correction, weak-label learning, cross-run cycles and withheld-marker evaluation on the same Rail-VIVID dataset. No TRACK-OT++ training result is claimed yet.

Verification: [TRACK-OT++ fitting-free checks](reports/track_ot_validation.json): 89 passing checks, including 18 TRACK-OT tests; zero local optimizer steps.

All new model-pipeline files are organized here on the `model-training` branch. The historical experiments at the repository root remain intact.

Follow [Full scale run](FULL_SCALE_RUN.md) for the full fusion configuration, strict preflight, timing logs, clean comparisons, reliability tests and repeated-pass studies. Keep any active experiment on its original checkout and commit.

- [Training and evaluation guide](TRAINING_PIPELINE.md)
- [Research audit and evidence](research_audit/README.md)
- [Implementation status](research_audit/PIPELINE_BUILD_STATUS.md)
- [Fitting-free validation record](reports/pipeline_validation.json)
- [Default training config](configs/pipeline.json)
- [117-task benchmark definition](configs/benchmark_suite.json)
- [Full fusion configuration](configs/full_scale.json)
- [Full scale validation](reports/full_scale_validation.json)

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

The branch retains the previously published benchmark results. Keep new raw recordings/images, memory maps, feature caches, weights, checkpoints and generated plans outside source commits. Run `plan` on the training computer to resolve its local paths. The pinned configuration and benchmark generator are versioned here.

The `src/` snapshot preserves existing model/feature imports so this folder can be installed independently. Original root-level modules/results are unchanged. Historical audit utilities that reproduce old experiments use the repository-root implementation; they are separate from the fitting-free pipeline validation command above.
