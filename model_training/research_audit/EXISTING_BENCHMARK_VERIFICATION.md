Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# Existing benchmark verification

## Exact reproduction

Read the existing source and saved JSON; trained the exact Random Forest configurations on the unchanged external cache and deterministic reconstructed split rows. RF uses 100 trees, maximum depth 10, balanced class weights, seed 42 and `n_jobs=-1`. StandardScaler is fitted only on training features. Feature dimensions: 66 vibration, 68 with GPS. Validation threshold maximizes F1 over 33 cutoffs from 0.1 to 0.9 inclusive (strict `>` for best-score updates selects the first tied threshold). Prediction uses score ≥ cutoff. RF scaling is redundant in principle but retained for exact reproducibility. No test threshold selection was performed by this audit.

All ten RF rows reproduce precision, recall, F1, ROC-AUC, average precision and selected threshold to the four decimals in the saved JSON. Maximum unrounded comparison error after loading JSON is ≤1.2e-16. This is a fresh deterministic rerun, not evidence of multi-seed uncertainty. Predictions with original cache IDs are saved separately in `evidence/rf_predictions_*.csv`; prior outputs remain untouched.

| model | split | modality | precision | recall | f1 | roc_auc | pr_auc | calibrated_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | Random_Window (Leaky Control) | Vibration_Only | 0.4420 | 0.7531 | 0.5571 | 0.9431 | 0.4730 | 0.5500 |
| Random Forest | Random_Window (Leaky Control) | Vibration_Plus_GPS | 0.4765 | 0.8354 | 0.6069 | 0.9579 | 0.6010 | 0.5000 |
| Random Forest | Block_Grouped_Window (Overlap-Free) | Vibration_Only | 0.3996 | 0.8242 | 0.5383 | 0.9293 | 0.4170 | 0.3500 |
| Random Forest | Block_Grouped_Window (Overlap-Free) | Vibration_Plus_GPS | 0.4539 | 0.8086 | 0.5815 | 0.9447 | 0.4885 | 0.4000 |
| Random Forest | Run_Held_Out | Vibration_Only | 0.3685 | 0.6597 | 0.4729 | 0.9036 | 0.3736 | 0.4500 |
| Random Forest | Run_Held_Out | Vibration_Plus_GPS | 0.3547 | 0.7437 | 0.4803 | 0.9244 | 0.5009 | 0.4000 |
| Random Forest | Spatial_Held_Out | Vibration_Only | 0.5954 | 0.4988 | 0.5429 | 0.9191 | 0.5229 | 0.6750 |
| Random Forest | Spatial_Held_Out | Vibration_Plus_GPS | 0.5046 | 0.5251 | 0.5146 | 0.9071 | 0.4793 | 0.2500 |
| Random Forest | Combined_Run_Spatial | Vibration_Only | 0.3519 | 0.2346 | 0.2815 | 0.9012 | 0.3360 | 0.5250 |
| Random Forest | Combined_Run_Spatial | Vibration_Plus_GPS | 0.2368 | 0.2222 | 0.2293 | 0.8452 | 0.2254 | 0.2000 |
| Isolation Forest | Combined_Run_Spatial | Vibration_Only | 0.0858 | 0.4938 | 0.1463 | 0.5888 | 0.1227 | 0.0166 |
| Signal Autoencoder | Combined_Run_Spatial | Vibration_Only | 0.0941 | 0.6914 | 0.1657 | 0.5986 | 0.0900 | 1.7605 |

“PR-AUC” in the existing code is sklearn **average precision**, not trapezoidal integration of the precision–recall curve. Keep this distinction in the paper and report prevalence. Combined test prevalence is 81/1,178 ≈6.88%; AP 0.3360 is above that baseline but does not imply high operational precision or a deployable safety monitor.

## Discrepancies in the supplied claims

The seven requested RF entries align with current saved artifacts after rounding; the old README contains different numbers. Current spatial vibration RF is F1 0.5429/AP 0.5229; GPS is F1 0.5146/AP 0.4793. The README's 0% spatial GPS recall is outdated for the current validation-selected cutoff. Do not retain that as evidence of a proven causal mechanism.

The supplied Isolation Forest “combined” tuple (precision ~0.071, recall ~0.714, ROC-AUC ~0.478, AP ~0.076) is actually the saved **run-held-out** row, not combined. The fresh combined run reproduces the saved combined row: precision 0.0858, recall 0.4938, F1 0.1463, ROC-AUC 0.5888, AP 0.1227. The misattributed claim is rejected. Isolation Forest fits normal training rows only, 100 trees and seed 42; threshold candidates are validation score percentiles 70–99.

The combined feature-autoencoder is **approximately reproduced, not exact**. Fresh precision/recall/F1/ROC-AUC/AP are 0.0941/0.6914/0.1657/0.5986/0.0900; saved JSON has 0.0943/0.6914/0.1659/0.5988/0.0901. The fresh threshold is 1.7605 versus saved 1.7563 (difference 0.0042). Both runs round to the supplied three-decimal summary, but this does not establish agreement at saved precision. No old weights/predictions or exact dependency lock exist to explain the discrepancy conclusively. Architecture is 66→32→12→32→66 with BatchNorm/LeakyReLU, trained on scaled normal features, 12 epochs, batch 64, Adam lr 0.001, weight decay 1e-5, seed 42. It is not evidence about raw-waveform autoencoder quality.

Logistic regression and environmental-feature configurations were inspected in source and saved artifacts but not rerun by this audit. All metrics not explicitly reproduced remain historical-artifact evidence.

## Label and feature provenance checks

Eight first/offset windows from four runs were recomputed from actual raw data. Maximum absolute feature discrepancy is 5.69e-14; midpoint timestamps and GPS coordinates match. All cache labels/nearest-site IDs recompute with zero disagreement, all cache windows are length 1,000, no repeated `(run_id,start_idx)` keys or nonfinite features. This is strong cache consistency evidence, but only eight raw feature vectors were recalculated. Full cache extraction provenance is not proven. A new raw CSV hash manifest and cache/code hashes now preserve current inputs.

The existing baseline config file is not read by the executable driver. Reproduction is source/config-value equivalent, not evidence of an existing fully configuration-driven pipeline. No historical split manifest, model checkpoints, prediction files or complete tuning ledger was found; this audit now supplies reconstructed indices and fresh predictions.

## Leakage interpretation and historical tuning

The original commit's driver directly took score percentiles on test data for Isolation Forest and autoencoder. Git inspection confirmed those expressions; later code moved them to validation. Calling that historical operation “test-label peeking” is too strong: it is test-distribution-dependent thresholding without test labels, a transductive protocol choice incompatible with a claimed fixed inductive threshold unless explicitly declared. Current calibration avoids that particular dependence.

Block grouping does not fully eliminate raw overlap (196 train/test pairs), so the difference from random-window AP is **not a causal estimate of leakage inflation**. The splits also change composition and validation operating points. A controlled study must hold test identities fixed and compare training inclusion/exclusion of overlapping and adjacent contexts, matching remaining sample counts and site/run distributions, plus distance/time embargo sensitivity. GPS decreases under one spatial split are also descriptive: use GPS-only, GPS-plus-signal, shuffled GPS, same-cohort fixed-capacity controls and multiple outer folds before claiming coordinate memorization as the cause. GPS-derived target construction makes a random-split GPS control especially optimistic.

All current test results have been repeatedly inspected in reports; complete earlier tuning history is unavailable. Treat them as development/exploratory benchmarks. Outer grouped evaluation with frozen selection rules is the minimum next design; a different shuffle does not create a pristine test set. Generalization to other railways requires external sites or new data.

## DR-Train reproduction

Executed the existing `D:/DR_Train/audit_scripts/13_model_comparison.py` without modifying its table or parameters. It uses 263 complete feature rows: Denise 128 (56 positive), Mesta_KSR 135 (67 positive). Eight whole-region features are scaled on training only. Models: kNN k=5, RF 200 trees seed 42, RBF SVM C=1. Exact stdout is preserved in `dr_train_reproduction.txt`.

| Train → test | kNN accuracy | RF accuracy | SVM accuracy |
| --- | --- | --- | --- |
| Stratified random 70/30 | 0.646 | 0.835 | 0.582 |
| Denise → Mesta_KSR | 0.459 | 0.593 | 0.356 |
| Mesta_KSR → Denise | 0.461 | 0.391 | 0.367 |

The claimed 52.4–58.1% location-held-out range is **not reproduced** by this current script/table. Features average available sensor channels over whole-region passes. Positive means within 60 days before a joined maintenance event; negatives exclude an extended near-event buffer. Mesta and Kings School Road share region 8 and are not independent locations at this feature resolution. Only two physical region groups, proxy event labels, serially correlated runs, unequal channels, unnormalized total signal energy, and differing maintenance histories prevent comparison with a published single-location study as equivalent experiments. Published papers' random split and leakage claims in script comments were not independently established here and must not be repeated as findings.
