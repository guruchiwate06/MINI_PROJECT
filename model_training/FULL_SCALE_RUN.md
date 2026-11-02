# Full scale training and benchmarks

Use `model-training` on the RTX 5070 training computer. **Keep an active experiment on its current commit.** Pull this update after it completes, or use a separate checkout and artifact directory. Source identity changed: old checkpoints cannot resume under this release. New GPU results remain pending; this release contains no trained weights or deep-model scores.

## Model and comparison scope

`configs/full_scale.json` selects TCN plus frozen DINOv2-small, bounded candidate attention, a quality gate and cross-run consistency. The suite's `full` variant also transfers its train-only time-frequency SSL checkpoint. Standalone `train` does not automatically pretrain. DINOv3 needs a separate pinned revision, cache and campaign. Patch memory and SSL are custom baselines, not faithful AnomalyDINO, FoundAD or TFSSCL reproductions.

The primary suite contains 12 methods × three folds × three seeds plus nine shared SSL jobs: **117 tasks**. Controls include RF, CNN, TCN, vision, patch memory, concatenation, spatial pooling, attention, cross-run attention, SSL TCN and late fusion. Only `cross_run` and `full` enable consistency; all methods use the same guarded paired population per fold. Validation selects the operating threshold at a requested false-alarm budget of 1/km. The test rate is measured, not guaranteed.

| Fold | Labelled train | Positive train | Labelled validation | Labelled test | Batches/epoch |
|---|---:|---:|---:|---:|---:|
| 1 | 6,972 | 454 | 85 | 567 | 109 |
| 2 | 4,486 | 454 | 425 | 687 | 71 |
| 3 | 3,993 | 449 | 452 | 828 | 63 |

These are inspected metadata counts, recounted by `workload` on the training host. Each fold has five train anomaly sites, one validation site and three test sites. Prior test use is disclosed. Nominal alignment still needs independent landmarks; visual shortcuts and the small number of independent sites remain research limitations.

## Performance changes

Dataset metadata, normalization and vision lookup indices are prepared once; the window hot path avoids Pandas rows and unnecessary clean-inference RNG creation. Cross-run neighbour queries are batched. Configurable memory-map retention can keep all 20 runs open. Fusion shares the current encoding between prediction, auxiliary loss and consistency, reducing up to four temporal encoder passes to two (current plus partner). Sharing also shares that dropout realization, so use fresh checkpoints and rerun controls.

CUDA mixed precision and deterministic per-item augmentation remain enabled. Default `workers=0` avoids expensive Windows process spawning for small epochs; explicit positive worker counts support configurable prefetch. Tune on the training host. No measured GPU speedup is claimed. Epoch logs now separate training/validation/checkpoint seconds, updates, windows/s and peak CUDA allocation; suite status includes job wall time.

## Setup and preparation

Install the appropriate CUDA PyTorch environment on the training machine, then run from the repository root:

```powershell
cd model_training
python -m pip install -e ".[vision,dev]"
python scripts/full_scale_run.py prepare
python scripts/full_scale_run.py plan
python scripts/full_scale_run.py workload
```

Set `RAILVIVID_DATA_ROOT` and `RAILVIVID_ARTIFACT_ROOT` when needed. Preparation reuses only complete verified caches and refuses partial outputs. Planning freezes source/config hashes in `artifacts/plans/full_scale/campaign.json`. Workload builds metadata splits and counts updates without forwarding or fitting models. Every stage is explicit; none automatically launches the next one. Repeated planning never overwrites an existing plan.

To change settings, supply a fresh plan and experiment output at planning time, for example:

```powershell
python scripts/full_scale_run.py plan --plan artifacts/plans/workers2/plan.json --set training.workers=2 --set experiment.output=../artifacts/experiments/workers2
```

Use the same `--plan` for subsequent stages. Data/vision/alignment overrides must also be supplied to `prepare` and `extract`. Later stages consume immutable job configs and reject overrides. Old local plans should be archived or replaced by a new path, never silently rewritten.

## Extract and preflight

```powershell
python scripts/full_scale_run.py extract --allow-download
python scripts/full_scale_run.py preflight
```

Frozen extraction is inference-only; downloads require the flag. Preflight checks prepared hashes, split provenance, camera coverage, duplicate image contents across partitions, statistical features, patch-cache integrity and CUDA availability. It records missing assets and exits nonzero before training. It does not validate physical correspondence or GPU optimization.

Clean extraction currently enumerates 100,408 accepted images (106.1 GB JPEG source), shared across folds/seeds. Global 384-D embeddings require about 154 MB; at 256 float16 patch tokens/image, patch storage requires about 19.7 GB. Actual token count depends on the processor. At an **assumed** 25 images/s end-to-end, extraction would take about 67 minutes plus final checks. Decode, hashing and writes must be included in any measured throughput claim.

## Train and evaluate

```powershell
python scripts/full_scale_run.py train
```

Training performs preflight, follows dependencies and stops on errors. It preserves diagnostics and last completed-epoch checkpoints, resumes matching interrupted jobs and skips verified complete ones. No silent CPU fallback, data substitution or batch-size reduction is performed. Do not change source/configs during the campaign.

At configured epoch limits, the inspected cohort has **524,880 supervised + 42,450 SSL = 567,330 updates**. Assumed mean loaded-update times of 20/50/100 ms imply 3.15/7.88/15.76 hours for updates alone. Early stopping can reduce supervised time. Extraction, validation, checkpoints, statistical fits, startup and final evaluation add time. The earlier 45–75-minute full-suite estimate is unverified; use actual epoch/job logs.

After freezing selection decisions:

```powershell
python scripts/full_scale_run.py evaluate
python scripts/full_scale_run.py summarize
```

The complete leaderboard requires all 108 detector evaluations. Reports include AP, F1, event recall, false alarms/km, localization, calibration, subgroup outcomes and clustered uncertainty. A failed or missing model cannot disappear from the ranking. Summary output is immutable.

## Major additional benchmarks

```powershell
python scripts/full_scale_run.py stress
python scripts/full_scale_run.py repeated
python scripts/full_scale_run.py compare
```

- Stress: 81 neural runs × 11 cases = 891 additional evaluations. Cases cover missing vision, missing vibration, missing axle channels, two noise levels and positional-prior shifts of ±5/10/15 m. RF, patch memory and score-level late fusion are explicitly excluded because equivalent corruption adapters are absent; their clean results remain required.
- Repeated passes: all 108 detectors, 1/2/3 held-out passes, mean/median/max pooling, fixed common segment populations and validation-only operating points per count. Insufficient support/passes fails. Draw variability does not represent independent defects.
- Paired comparison: `full` against 11 alternatives for each fold/seed, totaling 99 matched AP comparisons with shared site/run bootstrap weights. Few sites and multiple comparisons limit interpretation.

Optional image stresses need separate feature passes:

```powershell
python scripts/full_scale_run.py extract --allow-download --image-stress
python scripts/full_scale_run.py stress --image-stress
```

Blur/darkness add 108 evaluations over 54 vision/fusion runs. Stresses are synthetic; positional shifts perturb the prior inside guarded support, not measured true displacement. These stages never refit weights or thresholds using test labels. Completed extra outputs are checked before skipping; partial outputs are preserved for inspection.

Few-label and pure site/run studies require separate configurations, plans and experiment output directories. Faithful modern published comparators and independent alignment checks remain additional work before claiming state-of-the-art performance. No score target or superiority is guaranteed.

## Local verification

```powershell
python scripts/validate_pipeline.py --output reports/full_scale_validation.json
```

Tests prohibit optimizer steps and RF fitting. They cover inference-only models, splits, campaign immutability, missing assets, CPU rejection before dispatch, worker determinism and shared encoding. This validates implementation, not GPU training or final accuracy. No model was trained on the development laptop.
