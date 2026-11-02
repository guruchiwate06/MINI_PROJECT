# Research audit — 2 November 2026

This is the organized copy of the initial research audit. For current, fitting-free code verification run `python scripts/validate_pipeline.py` from `model_training/`. Historical audit reproduction commands below include model fitting and are separate from that validation.

Start with [FINAL_RESEARCH_VERDICT.md](FINAL_RESEARCH_VERDICT.md). Overall verdict: **CONDITIONAL GO**. RTX 5070 training availability is accepted; dataset validity/evaluation are the primary constraints.

Follow-up: [Decision on the proposed research upgrade](RESEARCH_UPGRADE_DECISION.md) assesses newer encoders, few-label learning and repeated-pass fusion, with a focused pilot and corrected evaluation requirements.

Implementation: [Training pipeline and GPU handoff](../TRAINING_PIPELINE.md). The pipeline is now coded; no local model training has been performed during its implementation.

See [pipeline build status](PIPELINE_BUILD_STATUS.md) for implemented features, the fitting-free validation record and remaining training-machine checks.

## Reports

1. [Project current state](PROJECT_CURRENT_STATE.md)
2. [Dataset feasibility](DATASET_FEASIBILITY.md)
3. [Labels and splits](LABEL_AND_SPLIT_AUDIT.md)
4. [Multimodal alignment](MULTIMODAL_ALIGNMENT_AUDIT.md)
5. [Benchmark verification](EXISTING_BENCHMARK_VERIFICATION.md)
6. [Model/compute feasibility](GPU_MODEL_FEASIBILITY.md)
7. [Novelty matrix](RESEARCH_NOVELTY_MATRIX.md)
8. [Recommended architecture](RECOMMENDED_ARCHITECTURE.md)
9. [Implementation roadmap](IMPLEMENTATION_ROADMAP.md)
10. [Final decision and all 30 answers](FINAL_RESEARCH_VERDICT.md)

## Reproduce audit outputs

Run from the workspace root, with the inspected Python environment:

```powershell
python research_audit/scripts/audit.py --data D:/Rail-VIVID
python research_audit/scripts/verify_benchmarks.py
python research_audit/scripts/supplement.py
python research_audit/scripts/deep_checks.py
python research_audit/scripts/final_checks.py
python research_audit/scripts/additional_checks.py
python research_audit/scripts/write_reports.py
python research_audit/scripts/validate_delivery.py
```

These utilities write only under `research_audit/evidence` and the ten report files. They do not modify original dataset files or existing benchmark results. Rerunning the audit updates this new evidence directory; archive it first if comparing audit versions. Several supplemental scripts intentionally use the actual inspected D: paths. The benchmark verifier reruns ten RF configurations, combined Isolation Forest and feature autoencoder plus the existing tiny DR-Train comparison; it does not launch all proposed GPU models. `supplement.py` requests upstream metadata only and `final_checks.py` checks four upstream file metadata records, never dataset image payload downloads. Full verification takes several minutes and reads ~1.9 GB of CSV plus selected frames/MAT records.

## Evidence index

| Evidence | Purpose |
| --- | --- |
| `run_inventory.csv` | All 20 raw CSV/file/time/channel counts, frame rates and nominal pairs |
| `channel_statistics.csv` | All-run channel moments, extrema and sampled spectral diagnostics |
| `raw_csv_sha256.csv`, `provenance.json` | Input CSV, code and cache fingerprints |
| `anomaly_sites.csv`, `site_run_counts.csv`, `windows_by_run_label.csv` | Site definitions and actual label counts |
| `split_indices_*.json`, `split_counts.csv`, `split_overlap.csv` | Exact reconstructed original indices and source overlap |
| `purged_*_indices.json`, `purged_split_counts.csv` | Additive raw-sample-only correction, no replacement model metrics |
| `candidate_fold_counts.csv`, `candidate_fold_indices.json` | Three alternate spatial/combined feasibility folds, not pristine tests |
| `pair_counts.csv`, `nominal_pair_manifest.csv` | All-frame nominal association counts and 190 concrete paired intervals |
| `window_pair_manifest.csv`, `window_pair_split_counts.csv` | 100 ms cached-window pairing cohort and split counts |
| `cross_run_image_duplicates.json`, `upstream_duplicate_spotcheck.json` | Wrong-run duplicate names/content and upstream hash evidence |
| `image_decode_sample.csv`, `frame_contact_sheet.jpg` | 271 decoded sample records and visually inspected contact sheet |
| `vibration_diagnostics.png` | Time, distance and frequency plots from actual records |
| `canonical_segments_diagnostic.csv`, `cross_run_route_repeatability.csv` | Frozen training-route projection; not surveyed accuracy |
| `benchmark_reproduction.csv/json`, `*_predictions_*.csv`, `*_combined_predictions.csv` | Fresh metrics and per-observation predictions |
| `raw_cache_spotchecks.csv`, `cache_validation.json` | Raw feature and label consistency |
| `dr_train_actual_inventory.csv`, `dr_train_decoded_records.json`, `dr_train_route_unit_check.csv`, `dr_train_reproduction.txt` | Existing maintenance project data/code checks |
| `environment.json`, `supplement_environment.json`, `torch_smoke.json` | Local dependencies, compute and real CNN inference |
| `environment_by_run.csv`, `legacy_eda_checks.json` | Actual weather extrema, notebook execution state, and historical Figure 6 segmentation check |
| `upstream_metadata.json`, `upstream_presence.csv` | Pinned official revision; 100,000-file response is incomplete/capped |
| `final_decision_matrix.csv` | Requested machine-readable feasibility matrix |

## Scope of verification

All sensor rows and image names were inventoried, but only 271 images fully decoded, 11 duplicate pairs content-hashed, four image LFS metadata records checked, eight feature windows recomputed, and four DR-Train records decoded. No physical image correction was independently fitted. Published metadata inconsistencies remain documented. Metrics are exact reproductions only where explicitly stated. Prior test reuse prevents treating the old folds as confirmatory. No original research artifacts were overwritten and no code was pushed/published.
