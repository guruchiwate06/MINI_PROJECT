Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# Current verified project state

## Scope and preservation

Inspected `mini_project_rajguru`, workspace research briefs, `D:/Rail-VIVID`, and `D:/DR_Train/audit_scripts` plus actual records. Both Git checkouts were clean on `main`, commit `c3dd22d`, with the same three-commit history. Key Python, configuration and result files are equal after newline normalization; byte hashes differ because of checkout line endings. No datasets, checkpoints, historical outputs, branches or research briefs were changed. No push, publication, reset or full GPU training was performed. All new outputs are in this directory.

## Implemented versus proposed

| Component | Actual implementation and evidence |
| --- | --- |
| Input loading | `src/data/loader.py`: chunked CSV, lazy OpenCV frames; no synchronization or paired dataset class |
| Features | `src/features/extractor.py`: 11 statistics/spectral features per channel, 66 total; 1,000 samples, stride 500; midpoint GPS labels |
| Labels/splits | `src/evaluation/splitters.py`: hardcoded published site table, radii 6/18 m, five deterministic protocols |
| Statistical models | Scaled RF, logistic regression, normal-only Isolation Forest; validation threshold functions |
| Autoencoder | MLP reconstruction on extracted feature vectors, not a raw-waveform autoencoder |
| Raw deep model | 18,770-parameter `Vibration1DCNN`; CPU training wrapper exists; no saved trained CNN or standalone raw-window training pipeline found |
| Benchmarks | `run_benchmarks.py` reads an external cache; saves aggregate JSON, overwrites that file if rerun unchanged; no historical per-window predictions/checkpoints |
| Config | `configs/baseline_config.json` exists, but benchmark driver does not read it; actual hyperparameters/splits are hardcoded in Python |
| Tests | Seven existing tests pass. They do not establish raw-sample nonoverlap, CNN performance or multimodal validity |
| Vision/fusion | No trained vision encoder, TCN, cross-attention, GNN, DANN or evidential model implementation/experiment found |
| DR-Train | Maintenance crosswalk, route reconciliation, whole-region feature extraction and two-region classifiers exist outside the Git repository |

The workspace checkout omits `data/cached_window_features.csv`; the real 26,906,514-byte cache is under `D:/Rail-VIVID/data`. The documented one-command reproduction from the workspace would fail without pointing at that file. The driver imports the extractor but does not generate a missing cache. Dependencies use lower bounds rather than a lockfile. Old README metrics and prose are inconsistent with the final saved JSON.

## Verification completed

All 20 full vibration CSVs were parsed (9,401,255 rows), all 105,901 frame filenames inventoried, 271 selected frames fully decoded, and real pairs built with explicit uncertainty flags. All 18,775 cached labels were recomputed with zero disagreement. Eight raw feature windows across four runs match the cache to floating-point roundoff. Ten RF configurations reproduced the final JSON at its four-decimal precision. Combined Isolation Forest also reproduced; the autoencoder is only approximately reproduced, with small metric and threshold discrepancies. Historical DR-Train script was executed and does not support the requested accuracy range. See the benchmark report.

Raw overlap was measured from `(run_id, start_idx, end_idx)` rather than row-index disjointness. Block and spatial boundary defects were confirmed. Additive purged manifests and four interval-regression tests were created; the original splitters and benchmark results were preserved.

## Incomplete evidence

No physically corrected multimodal alignment, human frame-level joint labels, full 105,901-image corruption scan, raw acquisition calibration replay, remote RTX 5070 execution, complete hyperparameter/tuning history, or prospective repair-event experiment was established. These limitations are explicit throughout the reports. A numerically reproducible score is not automatically a valid generalization result.

The wider workspace also contains team research notes, draft proposals, a paper-summary library, presentation-generation scripts, and a small DR-Train snapshot. These are background/artifacts, not additional trained anomaly models. `DR_Train_Snapshot/DATASET_OVERVIEW.md` explicitly describes sample files and incorrectly calls an Alisjahbana 2024 study the original dataset paper; use Liu et al. 2019. The active HTML/Markdown research brief retains older location-memorization and literature-comparability language that is not established by this audit. It was preserved rather than silently rewritten. The EDA notebook has five code cells, all with null execution counts and no outputs; it is not execution evidence. The existing figure script reads real data, but its claimed 7.62 m segmentation is actually latitude binning. See the dataset report for additional historical EDA discrepancies. Local paper summaries are not independently verified citations.

Machine-readable provenance: `evidence/provenance.json`, `raw_csv_sha256.csv`, `checkout_comparison.json`, `environment.json`, and `upstream_metadata.json`. Current upstream revision is `5fbbc49ca795296c7fb3fd6ee55a3cae2bb01410`. The metadata response contains exactly 100,000 sibling paths and is truncated relative to the local image inventory; all returned paths exist locally, which does not prove full upstream completeness.
