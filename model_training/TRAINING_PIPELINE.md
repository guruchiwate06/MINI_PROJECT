# Rail-VIVID training and benchmark pipeline

For the next full campaign use [FULL_SCALE_RUN.md](FULL_SCALE_RUN.md) and `configs/full_scale.json`. That guide includes strict preflight, shared-encoding/data-loader changes, measured timing fields and explicit additional benchmark stages. Existing experiments must remain on their original source identity.

This is executable pipeline code for the RTX 5070 training machine. **No model training is performed by preparation, doctor, smoke, plan, bundle, or the fitting-free test suite.** Training requires an explicit `train`, `pretrain`, or `suite-run` command and defaults to CUDA with no CPU fallback.

Historical `src/`, `run_benchmarks.py`, cached results and checkpoints at the repository root are preserved. This separate `model_training/` folder includes the new `railvivid/` package and a snapshot of the legacy `src/` dependencies it imports. Do not run the historical benchmark driver to launch this pipeline: it uses different protocols and overwrites its historical JSON output.

## What is implemented

| Stage | Code and behaviour |
| --- | --- |
| Configuration | Strict JSON settings, unknown-key rejection, typed limits, CLI overrides, portable root overrides |
| Data preparation | Streaming 20-run CSV conversion to memory maps; finite/rate/shape checks; SHA-256 input/cache provenance; consistent canonical direction |
| Labels | Published nine-site proximity labels: positive ≤6 m; ambiguous through 18 m; other observations are nominal negatives |
| Alignment | Declared nominal candidate bags; optional independent non-anomaly landmark interpolation with held-out checks; wrong-run image quarantine |
| Splits | Purged random/block controls, run holdout, spatial holdout, combined holdout, matched-direction speed holdout; raw/image support guards |
| Raw models | Reused CNN and a residual dilated TCN; channel-subset ablations |
| Vision | Pinned frozen DINOv2/DINOv3 ViT feature extraction, configurable ROI/masks, patch tokens, image quality and content hashes |
| Vision baselines | Supervised frozen-feature head and normal-reference patch nearest-neighbour detector |
| Fusion | Nearest-frame concatenation, mean/distance candidate pooling, bounded cross-attention, optional learned quality gate; missing-vision fallback |
| Independent late fusion | Separately fitted vibration and vision models; mixture selected on validation only |
| Self-supervision | Time/frequency VICReg-style encoder pretraining on permitted train windows; optional training-only same-position cross-run consistency |
| Training | Train-only unique-sample normalization, weighted BCE, AdamW, AMP, accumulation, clipping, deterministic seeds, AP-based checkpoint selection and early stopping |
| Recovery | Atomic checkpoints, epoch-boundary RNG restoration, run locks, failure diagnostics, OOM message, immutable completed outputs |
| Evaluation | AP, ROC-AUC, precision/recall/F1, Brier/NLL/ECE, risk–coverage summary, PR/ROC/calibration plots; per-run/site/condition tables |
| Event evaluation | Deduplicated alerts, one-to-one nominal-site matching, false alarms per evaluated kilometre, localization errors, transition rows retained |
| Comparison | Two-way clustered AP intervals and paired model AP differences on identical window IDs |
| Reliability | Explicit missing vision/channel cases, normalized vibration noise, candidate-prior shifts, re-encoded blur/darkness stresses |
| Repeated passes | Fixed common track segments; equal independent-pass budgets; validation-only false-alarm threshold per pass count |
| Benchmark suite | Three site folds × three seeds, configuration files, job dependencies, separate final test phase, integrity-checked complete result aggregation |

The patch-memory and self-supervised models are transparent baselines, **not claimed reproductions of AnomalyDINO, FoundAD or TFSSCL**. No GNN, DANN or evidential-method result is fabricated. Those remain optional research extensions.

## Installation on the training machine

Use Python 3.11+ in a dedicated environment. Install an appropriate CUDA-enabled PyTorch build for the RTX 5070 using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing project dependencies. The laptop's CPU build is not a CUDA environment template.

From `mini_project_rajguru/model_training`:

```powershell
python -m pip install -e ".[vision,dev]"
python -m railvivid doctor --config configs/pipeline.json
```

The implementation was checked against locally installed torch 2.10 APIs. Every run records actual package/CUDA/device versions and a pipeline source fingerprint; exact determinism across different hardware/library versions is not promised.

Root overrides on either Windows or Linux:

```powershell
$env:RAILVIVID_DATA_ROOT = "D:\Rail-VIVID"
$env:RAILVIVID_ARTIFACT_ROOT = "D:\RailVIVID_Prepared"
```

Keep prepared artifacts outside the raw dataset. Alternatively use the relative artifact directory already in the config. Copying `artifacts/railvivid_v1` to the equivalent location on the training machine avoids rereading the raw CSVs. Its memory maps and manifest paths are portable. Image extraction still needs the original image files. Generate suite plans on the training machine so their resolved experiment paths match that machine.

## Commands that do not train

```powershell
python -m railvivid prepare --config configs/pipeline.json
python -m railvivid split --config configs/pipeline.json
python -m railvivid doctor --config configs/pipeline.json --verify-cache
python -m railvivid smoke --config configs/pipeline.json --device cpu
python scripts/validate_pipeline.py
```

`prepare` and `split` are immutable: existing destinations are never overwritten. If a valid prepared directory already exists, skip preparation. A partially written directory has no `READY.json` and cannot be consumed. Choose a new artifact root after changing preparation settings. `smoke` uses a few real windows, random untrained weights, `inference_mode`, and **zero optimizer steps**. It provides no accuracy result.

The prepared maps retain float32 acceleration, int64 timestamp nanoseconds, float64 GPS and diagnostic projected chainage. Split manifests retain ambiguous windows for event evaluation while marking them ineligible for supervised loss. Training normalization merges source intervals so overlapping windows do not count the same raw sample twice.

## Build feature caches on the training machine

```powershell
python -m railvivid features --config configs/pipeline.json
python -m railvivid extract-vision --config configs/pipeline.json --device cuda --allow-download
```

`--allow-download` is explicit authorization to fetch the configured pretrained weights when not already cached; it is never enabled implicitly. The default DINOv2-small revision is pinned. `configs/dinov3.json` is a separate encoder template: set its approved exact revision before extraction. Gated checkpoints may require the user's existing authorized access. No weight downloads occurred during local development.

All accepted image frames are encoded once with a frozen backbone. Default patch storage can require tens of GB; global embeddings are much smaller. Set `vision.store_patch_tokens=false` before extraction if only global-head/fusion experiments are planned; the patch-memory baseline then correctly refuses to run. Image-content hashes are checked across partitions before fitting. Decode or feature failures are reported, not silently omitted.

ROI/mask boxes are normalized original-image coordinates. An empty mask list **does not mean cones were removed**. Create and independently inspect appropriate masks before making shortcut-controlled vision claims. A global image embedding is used for fusion; patch tokens are used by the separate patch-memory baseline. This implementation does not claim pixel-level defect segmentation ground truth.

## First training command — only on the RTX 5070 machine

After preparing its corresponding split:

```powershell
python -m railvivid train --config configs/pipeline.json
```

This is the standalone vibration TCN, with validation checkpoint/threshold selection. It does not inspect the test partition for selection. `scripts/train_on_gpu.ps1` is a thin wrapper around the same explicit entry point.

After freezing the experiment settings:

```powershell
python -m railvivid evaluate --config configs/pipeline.json
```

Artifacts include `config.json`, `environment.json`, `run.json`, `normalizer.json`, `epochs/`, `best.pt`, `last.pt`, validation predictions/metrics and calibration. Final evaluation adds predictions with original window IDs, events, subgroup metrics, plots, a report and checksums. RF uses a checked joblib artifact; patch memory uses `.npy`. Load artifacts created by this project, not arbitrary external pickles.

Resume only an interrupted matching experiment:

```powershell
python -m railvivid train --config configs/pipeline.json --set "training.resume=../artifacts/experiments/tcn_combined_seed42/last.pt"
```

Override paths are resolved relative to the config file. Resume restores model/optimizer/scheduler/scaler and Python/NumPy/torch RNG states from the last completed epoch. A mid-epoch interruption restarts that epoch; it does not pretend to resume a half-consumed batch. Changing model settings, split, seed or pipeline source rejects resume. Completed experiments are preserved; choose a new name for another fit. A stale `ACTIVE.lock` after a hard process kill must be checked against its recorded PID before manual recovery.

## Full matched-cohort benchmark

```powershell
python -m railvivid plan --config configs/pipeline.json --suite configs/benchmark_suite.json --output artifacts/plans/full
```

This generates configs and a job manifest only. The default plan contains 117 tasks: 12 methods × 3 folds × 3 seeds plus 9 encoder pretraining tasks. Nine tasks are score-level late-fusion calibration, not neural training. The `full` variant combines self-supervised initialization, training-only cross-run consistency, candidate attention and quality gating; the simpler variants are its controls. Every method in a fold/seed uses the same guarded paired observation population. The three test-site groups are 1/2/3, 4/5/6 and 7/8/9. The middle group includes the two unique turnout sites; report type composition and per-site outcomes.

On the training machine, after feature caches exist:

```powershell
python -m railvivid suite-run --plan artifacts/plans/full/plan.json
```

The suite builds missing split manifests, follows encoder/late-fusion dependencies, skips completed results only after identity checks, and resumes matching unfinished neural runs. It does not replace failed models or select better test scores. Default is stop on failure. `--continue-on-error` records failures and returns a failing status; incomplete runs never produce a completed leaderboard. An interrupted statistical fit must be restarted under a new run name or its incomplete directory explicitly archived; there is no fictitious partial-forest resume.

After development and selection are frozen, run the separate test phase and aggregate:

```powershell
python -m railvivid suite-evaluate --plan artifacts/plans/full/plan.json
python -m railvivid summarize --plan artifacts/plans/full/plan.json --output artifacts/reports/full
```

Do not repeatedly use this table for architecture selection. The dataset's old test folds were already viewed, so results must disclose their exploratory history. The pipeline enforces data access boundaries; it cannot erase earlier human decisions.

Small development plans can be generated using `--methods tcn,vision,late,pooling,attention --seeds 42 --folds 3` instead of `--suite`. This changes the planned experiment budget explicitly.

## Self-supervision, sensor subsets and label budgets

The `ssl_tcn` suite method pretrains a time/frequency encoder on training windows only, then transfers it to the same TCN classifier. Fixed pretraining epochs are used; no test or validation samples enter that loss. `training.cross_run_weight` enables an additional same-position consistency loss, with partners selected exclusively from the permitted training dataset and another run. Its location-identity shortcut risk requires the ablation without that loss.

Channel ablations use `--set 'model.channels=[2,4]'` (axle), `[1,3]` (bogie), `[5,6]` (body) or all six. Label budgets use `--set 'split.label_sites=[1,2]'` on a fold where those are training sites; held-out site IDs are rejected. Positives from other training sites remain available for unlabelled pretraining but are excluded from supervised loss. Known-normal annotation availability and validation labels must be reported separately. No window-level percentage is misrepresented as an independent-defect budget.

The `condition` protocol additionally requires `split.condition_direction` and `split.test_speeds`, with explicit validation runs from the remaining speeds. It restricts the entire cohort to that direction before splitting, so a change in direction is not passed off as a pure speed shift. Date/actual-speed confounding still requires scientific interpretation.

## Alignment and reliability experiments

Nominal timestamp matching remains physically unvalidated. `alignment.uncertainty_m` and `camera_footprint_m` define conservative **assumed support envelopes**; they are not measured error quantiles. The distance-pooling bandwidth is a separate tunable design parameter, not an assertion that 15 m is a Gaussian standard deviation.

For landmark mode, populate `configs/landmarks.example.csv` with real independent annotations. Each used run needs at least two monotone fit landmarks and separate check landmarks within their range. Strict mode rejects anomaly markers, fit/check reuse, nonmonotonic corrections, extrapolation and observed check error beyond the declared envelope. This checks the annotations supplied; it does not independently certify surveying quality or unseen-frame residuals.

Camera candidate envelopes are checked against spatial boundaries. Exact images cannot cross partitions. Raw source windows are purged at temporal boundaries, and missing-modality state is explicit. No test anomaly coordinates enter the model features or its correction map.

Example future stress command for a fitted fusion config:

```powershell
python -m railvivid robustness --config YOUR_FUSION_CONFIG.json --cases missing_vision,missing_axles,noise_0p1,shift_minus15,shift_plus15
```

Noise is explicitly in train-normalized channel units; its severity is synthetic, not a measured physical failure rate. Alignment shifts change the candidate positional prior within already guarded support, not a claimed known true image displacement. Blur/darkness require separate feature caches via `extract-vision --corruption '{"blur_radius_px":2.0}'` or `'{"brightness_factor":0.5}'`. Test labels are not used to refit weights, mixture, temperature or thresholds. RF corruption is rejected unless features are recomputed from the corrupted waveform; clean cached statistics are never mislabeled as a noise experiment.

## Repeated passes and paired comparisons

```powershell
python -m railvivid repeated --config YOUR_FITTED_CONFIG.json --predictions TEST_PREDICTIONS.csv --validation-predictions VALIDATION_PREDICTIONS.csv --counts 1,2,3 --output artifacts/repeated/study1
python -m railvivid compare --first BASELINE_PREDICTIONS.csv --second METHOD_PREDICTIONS.csv --output artifacts/comparisons/method_vs_baseline.json
```

Repeated-pass evaluation fixes the segment population before sampling, uses equal pass budgets and calibrates the requested false-alarm operating point using validation pools at each count. It rejects an eight-pass request if either validation or test has fewer than eight independent runs. The original combined paired test has at most three valid image runs. Mean/median/max are inference-only controls. Variation across overlapping pass draws is not reported as independent replication.

Paired model comparisons reject mismatched IDs/labels and use the same site/run bootstrap weights for both models. Nine physical sites still limit statistical power. AP means sklearn average precision, not trapezoidal PR integration. False-alarm distance uses evaluated support coverage, and event matching includes transition rows. Published-image AUC and this spatial proxy-label task are not automatically comparable.

## Fit-free verification and handoff

```powershell
python scripts/validate_pipeline.py
python -m railvivid bundle --output artifacts/railvivid_pipeline.zip
```

The tests prohibit optimizer steps and Random Forest fitting, while exercising source overlap, data integrity, forward-only models, missing-modality paths, inference/report generation with clearly untrained synthetic fixtures, checkpoint round trips/RNG, atomic-write failures, invalid inputs and benchmark planning. They do **not** certify GPU optimization, pretrained vision execution or achievable accuracy. The original legacy tests contain fits; they are deliberately not run by the new fitting-free test entry point.

The bundle includes package code, historical dependencies used by the package, configs, tests, scripts and this guide. It excludes raw datasets, prepared maps, models and experiment results. Transfer/copy caches separately or prepare them on the training machine. No Git push or model publication is performed.
