Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# Concrete implementation roadmap

## Completed audit milestone

This directory now contains executable read-only inventory, cache checks, timestamp pairing, benchmark reproduction, SHA-256 provenance, canonical-segment diagnostics, interval guards, source-overlap tests and additive purged manifests. This is intentionally separate from the validated historical code. No speculative full architecture tree was generated. Exact commands and evidence index are in `README.md`.

## Milestone 1 — freeze source provenance, labels and support-aware splits

First justified coding work is to promote the audit manifests/guards into the existing repository, after preserving commit `c3dd22d` and the original cache/results. Reuse `src/data/loader.py`, `src/features/extractor.py` and published-site constants. Add `src/data/manifest.py` (raw CSV hash, run/date/direction/target speed, sensor start/end, image hash/path, unit provenance, label source) and `src/evaluation/leakage_audit.py` using the tested interval logic. Upgrade `src/evaluation/splitters.py` additively with a purged alternative; do not rewrite old protocol identities. Put actual settings in a config read by every driver.

Acceptance: immutable original results preserved; cache IDs and raw intervals recoverable; no duplicate-image content across assigned groups (exact or perceptual hashes with review); zero intersecting raw support in train/validation/test; explicit discarded counts; both classes and independent site counts in every fold; candidate image footprints/position envelopes stay within permitted sectors. Keep AtoB_22.5_4 vibration but mark its image modality unavailable until upstream correction. The audit's raw-only purges are a starting point, not completion of the multimodal gate.

Required regression tests: the demonstrated 20-window block-boundary counterexample; irregular/removed buffers; same run vs different run; canonical reverse-run IDs; no train/validation/test overlap after purging; candidate-frame boundary overlap; duplicate `_3`/`_4` imagery rejected; wrong-date pair rejection. Existing row-index-only tests are insufficient. Save fold configs, generated IDs, geometry version and a schema version together.

## Milestone 2 — independently measure alignment and visual-label quality

Add `src/data/alignment.py` and `src/data/segmentation.py`, retaining nominal/corrected position and uncertainty separately. Build a manual landmark table with frame path, landmark identity, independently established chainage, confidence and reviewer. Use training-only/non-label landmarks for calibration; reserve landmark encounters for residual validation. Estimate per-run/direction monotone time-to-position mappings; compare nominal, marker interpolation, soft assignment and candidate pooling. Define rail ROI and cone-mask protocol, have two reviewers adjudicate a small balanced site/normal sample blinded to coordinate labels, and retain uncertain cases.

Acceptance: measured held-out landmark residuals in metres; pairing coverage per run/site/condition; no invented pairs for `_4`; no test-anomaly location used to tune image correction; clear camera field-of-view and sensor-offset assumptions; segment width/soft labels justified from measured errors. If accurate calibration is unavailable, document the failure and restrict the contribution to coarse uncertain site detection or vibration-only evaluation. Do not silently claim 5 m visual localization.

## Milestone 3 — smallest valid deep baseline on RTX 5070

Add a lazy raw-window dataset in `src/data/loader.py` or a small adjacent module and `src/training/train_vibration.py`. Reuse `Vibration1DCNN`, train-only channel statistics, class weights from training only, deterministic seed, optimizer configuration, validation-based checkpoint/threshold selection, and device option. The current wrapper allocates all data and runs CPU; adjust device/batch handling without altering historical experiment outputs. Record torch/CUDA versions and peak VRAM on the actual 5070.

Acceptance: a real forward/backward finite-gradient smoke test; one short training run on frozen support-purged inner/development groups; train/validation learning curves and prediction manifest; no test access during model selection. Compare the newly rerun RF and CNN on identical outer groups only after settings are frozen. Start with seed 42 for debugging, then a prespecified small seed set (e.g. 42/43/44) for the final comparison; report site/run variability as well as seed variability. There is no minimum F1 gate. This is the smallest scientifically justified GPU baseline.

## Milestone 4 — frozen vision and simple fusion

Add `src/features/vision_features.py`, `src/models/multimodal_fusion.py`, and one configuration-driven `src/training/train_multimodal.py`; avoid duplicating several training engines. Freeze a pinned pretrained encoder and save preprocessing/crop and image hashes. Train vision head, late-fusion and concatenation/candidate-pooling models with matched supervision/cohorts. Keep missing-modality evaluation separate from accepted-pair evaluation. Do not compare old 1,178-window vibration test metrics against an 846-window multimodal test without rerunning vibration on that same cohort.

Acceptance: all modalities share exact accepted IDs; no visual duplicates or support crossing; cone and alignment controls completed; per-site/run predictions, metrics and calibration artifacts saved; predetermined hyperparameter budget. If fusion does not improve consistent held-out-site event metrics, retain the simpler result and examine whether vision adds information at all.

## Milestone 5 — one alignment-aware method, only if justified

Add `src/models/spatial_attention.py` for bounded candidate cross-attention only after pooling/concatenation establish meaningful paired information. Use shared encoders/candidate budgets and relative-distance masks, keep anomaly coordinates out of features. Candidate support itself must respect the split guard; uncertainty envelopes may require removing more windows than raw-only purging.

Acceptance: compare equal-capacity pooling, concatenation and attention with a frozen evaluation plan; nominal/corrected/synthetic offsets; report per-site improvements and failures, event false alarms/localization, uncertainty from small site/run clusters, and compute cost. A single favourable F1 or ROC-AUC is not proof of alignment recovery. If calibration coverage or independent sites are inadequate, do not pursue the architecture novelty claim.

## Optional work and stopping rules

Add evaluation modules for event metrics/calibration/robustness when those experiments run. Sensor subsets are the first extension. Speed normalization precedes DANN; ensembles/temperature scaling precede evidential heads; smoothing/line convolution precede any GNN. Defer DR-Train pretraining, large raw Transformers, end-to-end vision fine-tuning and simultaneous six-model research. No future repair-event forecasting claim is supported by Rail-VIVID. Do not build an elaborate UI to compensate for unverified labels.

Before submission, disclose legacy test reuse, preregister or otherwise freeze new outer folds and inner selection, update literature, release artifact/version metadata where authorized, and use benchmark-focused language. External railway data or independently collected unseen sites would be needed for a strong deployment-generalization claim. No publish/push step is authorized or performed in this audit.
