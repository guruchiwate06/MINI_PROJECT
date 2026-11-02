# Pipeline implementation status — 2 November 2026

**Implemented; no models trained during this implementation.** This laptop performed code validation, data conversion and forward-only tests. GPU optimization and real pretrained-encoder execution remain unrun. No pretrained weights were downloaded.

Start with [the full-scale run guide](../FULL_SCALE_RUN.md). The latest fitting-free record is [full_scale_validation.json](../reports/full_scale_validation.json); the actual paired metadata workload is [full_scale_workload.json](../reports/full_scale_workload.json). These supersede the initial implementation's test count and provide no trained-model scores. Current remote GPU testing remains pending review.

## Delivered

- Strict configurable preparation, memory-mapped raw datasets and SHA-256 manifests.
- Explicit nominal/landmark alignment, image quarantine, source-support guarding and independent site/run partitions.
- CNN, dilated TCN, frozen DINO feature adapters, patch-memory baseline, independent late fusion, candidate pooling/attention/quality gating.
- Time/frequency self-supervision, training-only cross-run consistency, and a full combined variant with controls.
- CUDA-gated training, AMP, accumulation, validation-only selection/calibration, atomic checkpoints, RNG/epoch-boundary recovery and failure diagnostics.
- Final event/classification/calibration reports, reliability stresses, repeated-pass evaluation, paired model comparisons and complete matched-cohort aggregation.
- GPU handoff script, DINOv2/DINOv3 configurations, landmark schema, source-only bundle and a fitting-free test suite.

The default suite definition covers 12 methods, three site folds and three seeds, sharing nine encoder-pretraining tasks: **117 planned tasks**. Planning does not execute fitting or final evaluation. Source configs are in `configs/`; the generated local plan is `artifacts/plans/full/plan.json`. Regenerate this plan on the training machine to resolve paths for that machine.

## Verified locally

The [validation record](../reports/pipeline_validation.json) reports **61 passed tests**, zero optimizer steps, no model fitting and no pretrained-weight downloads. Tests cover invalid configurations, raw timestamp corruption, interval overlap, label budgets, training-only normalization/partners, all neural forward paths including missing vision, checkpoint serialization/RNG, fixture inference/reporting, repeated-pass budgets, paired comparisons and feature-cache integrity using a synthetic encoder stub. Neither fixture scores nor random-weight outputs are research benchmark results.

All 20 real vibration CSVs were converted to a versioned memory-map cache under `artifacts/railvivid_v1`, retaining 9,401,255 rows and 18,775 window definitions. The earlier development cache under `artifacts/railvivid` is preserved; the current config points to `railvivid_v1`. Source-support bounds in the current preparation code use every sample in a window. No raw dataset files were changed.

Original historical benchmark modules/results were preserved. The original build validation took place before Git delivery; the pipeline and audit are now organized under `model_training/` for the `model-training` branch. Git delivery does not run training. The dataset checkout at `D:/Rail-VIVID` remains untouched.

## Conditions before scientifically meaningful training/evaluation

1. Set the data/cache paths on the RTX 5070 machine and verify its CUDA-enabled PyTorch installation.
2. Inspect/build the desired guarded split and prepare feature caches. No real vision cache is claimed complete here.
3. Pin the vision revision and obtain authorized pretrained weights on the training machine. The DINOv3 template deliberately requires a revision before extraction.
4. Independently validate alignment and cone/ROI controls before making physically aligned or shortcut-controlled claims. Nominal candidate bags and assumed support envelopes do not replace this evidence.
5. Freeze selection/evaluation roles. Keep the dataset's nine-site limitation and previous test use explicit. The pipeline does not guarantee accuracy, novelty or publication.

The code includes a nearest-patch baseline and a VICReg-style time/frequency objective. It does not claim exact reproductions of AnomalyDINO, FoundAD or the cited IEEE time/frequency method. GNN, DANN and evidential-model research remain deferred, consistent with the audit recommendation.
