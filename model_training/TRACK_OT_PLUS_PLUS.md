# TRACK-OT++: complete experimental pipeline on Rail-VIVID

This implementation keeps **Rail-VIVID as the only dataset**. It does not modify raw files, add another railway dataset, invent annotations, or replace the original evaluation labels. Training is explicit, CUDA-only, and has not been run on the development laptop. Publication quality and benchmark improvements remain experimental questions.

Start from `model_training`, using Python 3.11 or newer and a CUDA-enabled PyTorch build suitable for the training GPU. Install the project and its vision/alignment extras in that environment. Do not install a CPU-only Torch wheel over the GPU environment.

```powershell
python -m pip install -e ".[vision,alignment,dev]"
python -m track_ot doctor --config configs/track_ot_plus_plus.json
python -m track_ot smoke --synthetic --device cpu
```

`doctor` and `smoke` do not fit anything. The synthetic smoke uses random tensors and reports numerical/model-shape validity, not measured detection accuracy, physical alignment or GPU throughput.

## Implemented model

The default model has 3,371,052 trainable parameters with the pinned 384-dimensional DINOv2 cache. The pretrained vision encoder remains frozen and is not included in that count.

1. Three dilated temporal branches process bogie channels 1/3, axle-box channels 2/4, and car-body channels 5/6. A gated frequency branch uses the 0–500 Hz spectrum.
2. Separate Transformers contextualize ordered vibration windows and image sequences using relative positions. Learned attention pools a fixed subset of DINO patch tokens and combines it with global frame features. Source absolute GPS, site IDs and anomaly coordinates are not model inputs.
3. A bounded monotone position warp predicts a base drift and positive local stretch. Its correction remains within the configured 15 m envelope.
4. KL-unbalanced entropic OT matches modality sequences with a physical feasibility band, feature/position costs and explicit unmatched observations. A reject-augmented balanced alternative is available as an ablation.
5. Soft crossing regularization discourages order inversions. **The position warp is monotone; the transport plan has a soft ordering penalty, not a hard monotonic guarantee.** These two properties are distinct.
6. Spatial top-k MIL learns from weak bag labels. Separate unimodal heads support missing sensors. Original anchor-window labels remain the evaluation targets.
7. Cross-run InfoNCE uses permitted training partners and spatially separated negatives. A separate correspondence cycle maps A-vibration → A-image → B-image → B-vibration and compares reconstructed physical support on training data.
8. Training Brier regularization, validation-only temperature scaling, a fixed false-alarm/km operating point, and validation-selected selective classification support confidence evaluation. Abstention never removes errors from the primary event/false-alarm metrics.

The joint loss combines MIL/radius supervision, auxiliary heads, the regularized transport objective, soft crossing, bounded-drift regularization, synthetic-shift equivariance, cross-run contrastive consistency, correspondence cycles and Brier loss. Each weight is explicit in `track_ot/config.py`. This is a proposed research formulation; no first-of-its-kind claim is made.

## Data and leakage contracts

The sequence builder reuses each row of the existing guarded combined/paired split as an anchor. Context contains only contiguous, already permitted same-run/same-partition windows. Images are drawn only from the union of those windows' existing candidate lists; no wider source search or new frame is silently introduced. Image selection is deterministic under a fixed maximum budget. Context support, selected frames, weak labels and a fingerprint are saved to `bags.json`.

Bag targets are positive when at least one permitted context label is positive, negative when all are negative, and unknown otherwise. Labels remain documented-site proximity, not adjudicated defect visibility. Original window labels, partition assignment, anchor counts and event coverage remain unchanged. The radius-label ablation uses the same sequences with original instance supervision.

Existing image-content hash checks reject cross-partition duplicates. Training normalization counts unique permitted raw training samples. Partner bags never use validation/test runs. Unknown windows may provide context but do not become invented negative targets. No few-label/site-budget experiment is silently approximated; unsupported budgets are rejected.

The old RF AP of 0.3957 is a useful reference. A fair attribution study must also compare models with the same sequence context, anchors, image cache and calibration policy. Original published accuracy, window AP, segment AP and event recall are different measures.

## Prepare once on the training computer

The default config references existing `configs/full_scale.json` and selects fold 1. It uses the same prepared-data settings and pinned vision revision. Existing valid artifacts can be reused. Commands refuse to overwrite an existing cache/split.

```powershell
# Only if the matching prepared artifacts do not already exist:
python -m track_ot prepare --config configs/track_ot_plus_plus.json
python -m track_ot split --config configs/track_ot_plus_plus.json

# Only if this exact frozen vision cache does not already exist:
python -m track_ot extract-vision --config configs/track_ot_plus_plus.json --allow-download

# Real-data forward-only verification, after caches exist:
python -m track_ot smoke --config configs/track_ot_plus_plus.json --device cuda
```

Patch pooling requires the verified `patches.npy` cache. Extraction/download is never triggered by training, doctor or smoke. The original cache's ROI and masks are recorded; review rail visibility and cone exclusion before a definitive campaign. Changing ROI/masks requires an explicitly separate vision cache, while raw Rail-VIVID remains unchanged. Do not represent an unreviewed full-frame cache as marker-free.

## Pilot and full benchmark

```powershell
# One explicit pilot, on the RTX training computer:
python -m track_ot train --config configs/track_ot_plus_plus.json --on-training-device

# Only after model/settings are frozen, evaluate the test partition:
python -m track_ot evaluate --config configs/track_ot_plus_plus.json --partition test --device cuda

# Resume an interrupted run with identical code/config/data (inspect any stale lock first):
python -m track_ot train --config configs/track_ot_plus_plus.json --on-training-device --resume
```

Training records actual epoch time, validation time within total epoch duration, peak CUDA allocation, component losses and measured remaining-time estimates. No RTX 5070 duration is assumed from CPU results. Checkpoints, RNG, optimizer, scheduler and scaler are saved atomically. Resume checks source/config/data/cache identity; completed runs are not refitted. Numerical failures and OOM stop with an actionable `FAILURE.json`; no examples or model components are silently dropped.

Generate campaign plans **on the training device**, so artifact/output paths match that checkout:

```powershell
python -m track_ot plan --config configs/track_ot_plus_plus.json --output artifacts/track_ot_plan
python -m track_ot campaign-run --plan artifacts/track_ot_plan/plan.json --on-training-device
python -m track_ot campaign-evaluate --plan artifacts/track_ot_plan/plan.json
python -m track_ot summarize --plan artifacts/track_ot_plan/plan.json --output reports/track_ot_results
```

The default plan contains 11 variants × 3 folds × 3 seeds = **99 training jobs**. Planning does not execute them. Run the single pilot first to measure feasibility and throughput. Campaign test evaluation is separate from training and validation selection. All current outer folds have already been inspected historically; disclose their exploratory status.

Default variants: full, no cycle, no cross-run losses, no learned drift, no soft order penalty, radius labels, OT control, generic OT, attention, distance pooling, and sequence vibration-only. `ot_control`, `attention` and `distance` disable learned drift/cycle/extra transport/order objectives together, isolating the matching rule. `generic_ot` additionally removes position costs/features and the physical mask, while retaining the same permitted source candidates. It is a generic uncertainty-aware matching control, not an ACML reproduction. Full-vs-control comparisons alone cannot assign gains to OT. Additional options include no spectral branch, global images, no position cost, no InfoNCE, and reject-augmented OT. `no_position_cost` removes that cost term; the physical band and other geometry remain.

This implementation does not claim faithful reproductions of TAIL-MIL, ACML or VSAA. Their external implementations/formulations require separate documented adaptations; the existing RF/CNN/TCN and current fusion controls remain available through `railvivid`.

## Alignment study using the same dataset

Create annotation files and contact sheets from existing Rail-VIVID frames:

```powershell
python -m track_ot annotation-template --config configs/track_ot_plus_plus.json --output annotations/railvivid
python -m track_ot contact-sheet --config configs/track_ot_plus_plus.json --run AtoB_20_1 --every 100 --output annotations/sheets_AtoB_20_1
```

The templates are intentionally empty. An evaluator must identify actual start/end encounters and withheld cone encounters. A visible cone is not automatically the camera's physical crossing instant; account for viewpoint, camera/sensor offset, rounded coordinates and annotation intervals. No measured alignment accuracy can be manufactured from an empty template.

`endpoint_anchors.csv`: `run_id,frame_id,chainage_m,role,source,is_anomaly_marker`. Supply exactly two non-anomaly `fit` endpoints per evaluated run. The estimator rejects `check` annotations and anomaly-marker anchors.

`withheld_markers.csv`: `marker_id,run_id,frame_id,chainage_m,annotation_uncertainty_m,role,source`. Keep its `check` coordinates separate until predictions are frozen. The published anomaly cones can serve as evaluator references; their coordinates must not correct their own held-out model inputs.

```powershell
# Endpoint-scaled baselines. Each output is immutable and hashed.
python -m track_ot align --anchors annotations/railvivid/endpoint_anchors.csv --method nominal --output artifacts/alignment_nominal
python -m track_ot align --anchors annotations/railvivid/endpoint_anchors.csv --method uniform --output artifacts/alignment_uniform
python -m track_ot align --anchors annotations/railvivid/endpoint_anchors.csv --method flow --output artifacts/alignment_flow

# Withheld references enter only this evaluator:
python -m track_ot align-evaluate --predictions artifacts/alignment_flow/frame_alignment.csv --references annotations/railvivid/withheld_markers.csv --output reports/alignment_flow
```

The `nominal` baseline projects nominal positions monotonically and anchors endpoints. `uniform` interpolates frame order. `flow` uses forward/backward-checked local optical-flow magnitude to estimate progress between endpoints. It is **not calibrated metric visual odometry** and does not independently establish absolute accuracy. Optical-flow failures abort by default; `--low-feature-policy timestamp` explicitly enables and records a fallback. This alignment study is an independent baseline/evaluation workflow; it does not silently replace the guarded model candidate coordinates.

Export the frozen trained model's predicted image positions for a matching comparison:

```powershell
python -m track_ot alignment-predict --partition train --output artifacts/model_alignment_train
python -m track_ot alignment-predict --partition test --output artifacts/model_alignment_test
```

Overlapping context predictions for the same selected frame are averaged and their variation is reported. That variation is not a calibrated confidence interval. Some frames may not be selected under the fixed candidate budget; the evaluator reports missing-marker coverage rather than silently interpolating reference encounters.

For an empirical interval study, create a separate `calibration` reference CSV whose frames belong to permitted TRAIN candidates, then freeze its residual quantile before evaluating `check` encounters:

```powershell
python -m track_ot align-calibrate --predictions artifacts/model_alignment_train/frame_alignment.csv --references annotations/train_calibration.csv --output artifacts/alignment_interval_calibration
python -m track_ot align-evaluate --predictions artifacts/model_alignment_test/frame_alignment.csv --references annotations/railvivid/withheld_markers.csv --calibration artifacts/alignment_interval_calibration/calibration.json --output reports/model_alignment_test
```

The interval evaluator checks estimator identity and rejects reused calibration/fit encounters. It reports median/MAE/p95/worst error, reference-centre coverage, coverage of entire annotated reference intervals, missing predictions, and run/marker groups. Repeated sites are dependent; the finite quantile does not imply a population coverage guarantee.

## Stress, repeated passes and comparisons

```powershell
python -m track_ot evaluate --partition test --stress missing_vision
python -m track_ot evaluate --partition test --stress shift_plus15
python -m track_ot robustness --partition test --device cuda
python -m track_ot evaluate --partition validation
python -m track_ot repeated --predictions artifacts/track_ot/track_ot_pp_fold1_full_s42/evaluations/test_clean/predictions.csv --validation-predictions artifacts/track_ot/track_ot_pp_fold1_full_s42/evaluations/validation_clean/predictions.csv --output reports/track_ot_repeated
```

The default experiment directory is `model_training/artifacts/track_ot`; commands above assume the default config and current directory. For campaign jobs, use each planned config and its job-specific run directory. Stress reuses the original validation calibration. Synthetic prior shifts test controlled sensitivity, not independently verified physical drift. Repeated-pass studies reuse the existing common-coverage 1/2/3-pass protocol and validation-only operating thresholds. Do not call repeated draws independent defects.

Use `python -m railvivid compare --first <predictions.csv> --second <predictions.csv> --output <new_directory>` for strict paired anchor comparisons, and retain all site/run failure tables. Report detection and alignment outcomes separately.

## Verification and remaining empirical work

```powershell
python -m pytest tests_pipeline -q
```

Tests prohibit optimizer steps and Random Forest fitting. They check gradients without updating weights, numerical transport behavior, invalid inputs, modality fallbacks, cross-run cycles, source isolation, immutable plans and withheld-marker evaluation. They do not establish CUDA compatibility, completed training, physical calibration or benchmark superiority. Those measurements must come from the training host and actual evaluator annotations. Rail-VIVID's nine physical sites and one corridor still limit the scope of generalization claims.
