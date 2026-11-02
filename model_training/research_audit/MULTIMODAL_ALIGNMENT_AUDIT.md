Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# Image–vibration alignment

## Clocks and physical correspondence

`Acquisition.py` timestamps images using `datetime.now()` after `cap.read()` and display work, immediately before `cv2.imwrite`. DAQ acquisition uses a hardware sample clock with system-clock file timestamps; final processed CSVs contain uniform reconstructed 2 kHz date/time values. A common laptop clock supplies an absolute reference, but there is no inspected per-exposure hardware synchronization signal. Timestamp precision is not exposure accuracy. The upstream README explicitly warns of variable image-write timing and nonconstant positional drift up to ~15 m, despite more optimistic synchronization language in the paper. No verified per-run correction function or residual uncertainty distribution is supplied locally. Camera look-ahead distance, axle/bogie/body offsets and direction-dependent footprint further separate a frame timestamp from the track patch that generated a response.

Keep four distinct fields: nominal recorded timestamp/position; surveyed or independently verified true landmark position; estimated corrected image position; uncertainty of that estimate. This audit constructs nominal pairs only. It does not invent true positions from anomaly proximity.

## Executed pairing tests

Every image filename was parsed and matched to its own run's nearest sensor row. A nominal frame pair requires time inside the sensor record, residual ≤1 ms, and 500 preceding/following rows for a complete 1,000-row waveform. Attempted: **105,901**; accepted nominal pairs: **100,222**; rejected: **5,679**. Of these, 5,493 frames are outside their run's sensor interval; the remaining 186 lack full waveform context near boundaries. Accepted coverage is 19 runs and all nine proximity-labelled sites. **Physically validated/corrected pairs: zero established by this audit.** The submillisecond nearest-row residual quantifies numerical association, not the physical drift.

| Site | Nominal positive frame pairs | Runs |
| --- | --- | --- |
| 1 | 687 | 19 |
| 2 | 755 | 19 |
| 3 | 758 | 19 |
| 4 | 738 | 19 |
| 5 | 756 | 19 |
| 6 | 791 | 19 |
| 7 | 745 | 19 |
| 8 | 742 | 19 |
| 9 | 809 | 19 |

`nominal_pair_manifest.csv` provides 190 concrete sensor intervals with image paths (one nominal normal plus nine nominal site-positive samples per accepted run). Actual imagery was decoded; successful pairing is not inferred from an intended folder layout. `pair_counts.csv` and `run_inventory.csv` contain all per-run counts.

An alternative view pairs **cached vibration-window midpoints** to the nearest frame in the same run with a declared 100 ms numerical tolerance: 17,742 of 18,775 windows have candidates. This is a different pairing rule and count, explicitly recorded in `window_pair_manifest.csv`. It does not repair the 15 m problem. When buffer windows are excluded, counts under the existing exact splits are:

| protocol | partition | windows | nominal_pairs | positive_nominal_pairs | unique_images | physically_validated_pairs |
| --- | --- | --- | --- | --- | --- | --- |
| random | train | 10579 | 9981 | 760 | 9981 | 0 |
| random | validation | 2441 | 2316 | 184 | 2316 | 0 |
| random | test | 3254 | 3080 | 225 | 3080 | 0 |
| block | train | 10615 | 10143 | 739 | 10143 | 0 |
| block | validation | 2392 | 2230 | 195 | 2230 | 0 |
| block | test | 3267 | 3004 | 235 | 3004 | 0 |
| run | train | 10701 | 10692 | 812 | 10692 | 0 |
| run | validation | 2446 | 2439 | 187 | 2439 | 0 |
| run | test | 3127 | 2246 | 170 | 2246 | 0 |
| spatial | train | 6751 | 6368 | 643 | 6368 | 0 |
| spatial | validation | 3295 | 3122 | 131 | 3122 | 0 |
| spatial | test | 6228 | 5887 | 395 | 5887 | 0 |
| combined | train | 4423 | 4419 | 449 | 4419 | 0 |
| combined | validation | 504 | 504 | 22 | 504 | 0 |
| combined | test | 1178 | 846 | 59 | 846 | 0 |

For combined evaluation, nominal multimodal counts fall from train/validation/test 4,423/504/1,178 to **4,419/504/846**, with **449/22/59 positive windows**. The accepted test windows cover only three image runs, because AtoB_22.5_4 has no valid same-run frame timestamps. Do not compare a vibration-only score on all 1,178 test windows with fusion on 846 as if modality were the only change. Re-evaluate all modalities on the identical accepted cohort; separately report missing-modality performance on the full cohort.

## Duplicate-image blocker

All 5,119 names in AtoB_22.5_4 also occur in AtoB_22.5_3; 11 sampled contents are identical. Two corresponding images are identical in the pinned upstream LFS metadata as well. Their sensor recordings occur on different dates. Blind sequence rescaling could therefore turn a training run's actual images into a test run's nominal images. Correct matching must reject them. Preserve evidence and request corrected files later; no external message was sent. Full equality of all 5,119 image contents was not exhaustively hashed.

## Alignment alternatives and acceptance gates

| Method | Supported now? | Main assumption and required test |
| --- | --- | --- |
| Nearest timestamp | Numerically implemented | Treat as uncertain baseline; one run rejected entirely |
| Nearest nominal track position | Conditional | Relabels the same imperfect clock-derived positions; cannot independently fix drift |
| Marker-corrected interpolation | Conditional, preferred calibration study | Independently annotate visible surveyed landmarks per run; estimate monotone map, validate with withheld landmarks |
| Distance-weighted soft assignment | Conditional | Needs defensible position intervals; report width sensitivity, never treat 15 m as known sigma |
| Candidate-window pooling/fusion | Conditional, preferred first fusion | Mask candidates at split boundaries; verify true event is inside candidate set; account for camera look-ahead |
| Spatially constrained cross-attention | Conditional extension | Multiple candidate keys/positions, measured alignment targets or robust objective, same control baselines |

Annotate several landmark encounters throughout each run and direction rather than only endpoints. Evaluate correction by withholding landmarks and reporting residual median, 95th percentile and worst case in metres; do not evaluate on the same marker points used to fit the map. Use non-anomaly landmarks for strict unseen-anomaly claims. Using test anomaly coordinates to calibrate their own inputs is label-derived localization assistance and must be excluded or explicitly reported as an oracle-assisted setting.

Cones are visible in the contact sheet. Generate separate standard and cone-suppressed ROI cohorts, keeping removal masks away from rail defects; report both. A simple bottom-centre crop alone may retain cones at the rail edge and is not a verified shortcut control. Manual rail/joint annotations should be blinded to proximity labels, with second-reviewer agreement and an uncertain category. Avoid treating turnout presence as damage severity.

Proceed with fine-grained fusion only once residual errors are measured at a scale justified relative to the chosen segments. Otherwise retain coarse candidate bags and label the task as uncertainty-aware site-proximity detection. Synthetic offsets (for example 0, ±5, ±10, ±15 m) are controlled stress tests; their zero setting is the nominal baseline, not verified perfect alignment. Cross-attention does not automatically learn a correction: one global image token and one global vibration token offer no alternative spatial correspondence at all.

Source: [official timestamp limitation](https://huggingface.co/datasets/saluslab/Rail-VIVID/blob/main/README.md). Visual observations and counts are from the local decoded files.
