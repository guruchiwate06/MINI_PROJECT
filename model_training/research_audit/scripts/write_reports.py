"""Generate narrative audit reports from preserved local evidence, no model training."""
from pathlib import Path
import json,textwrap
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'research_audit'; E=OUT/'evidence'

def read(name): return pd.read_csv(E/name)
def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None))
def write(name,text):
    (OUT/name).write_text(textwrap.dedent(text).strip()+'\n',encoding='utf-8')

def main():
    inv=read('run_inventory.csv'); splits=read('split_counts.csv'); overlap=read('split_overlap.csv'); metrics=read('benchmark_reproduction.csv')
    sites=read('site_run_counts.csv').groupby('nearest_anomaly_id').agg(positive_windows=('positive_windows','sum'),runs=('run_id','nunique')).reset_index()
    pairs=read('pair_counts.csv'); paired=read('window_pair_split_counts.csv'); env=json.loads((E/'torch_smoke.json').read_text())
    conditions=inv.groupby(['direction','target_speed_kmh']).agg(runs=('run_id','nunique'),rows=('rows','sum'),images=('images','sum'),nominal_pairs=('nominal_valid_pairs','sum')).reset_index()
    weather=read('environment_by_run.csv').groupby('date').agg(temperature_min=('Temperature_min','min'),temperature_max=('Temperature_max','max'),humidity_min=('Humidity_min','min'),humidity_max=('Humidity_max','max')).reset_index()
    header='Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.\n\n'
    write('PROJECT_CURRENT_STATE.md',header+'''# Current verified project state

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
''')
    write('DATASET_FEASIBILITY.md',header+f'''# Dataset feasibility

## Rail-VIVID: direct measurements

The local dataset contains **{int(inv.rows.sum()):,} vibration time rows**, six channels per row, and **{int(inv.images.sum()):,} JPG files** across 20 run folders. Thus there are 56,407,530 scalar acceleration values; “9.4 million readings” is a row count, not a six-channel scalar count. CSVs total {int(inv.csv_bytes.sum()):,} bytes; frame files total {int(inv.image_bytes.sum()):,} bytes. Anomaly scans add further storage.

Every CSV has `Timestamp, Channel1, Channel2, Channel3, Channel4, Channel5, Channel6, Temperature, Humidity, Latitude, Longitude`: 11 actual columns, despite the README saying 12 and listing underscored channel names. There is no measured-speed column. All raw rows were checked: zero missing values and zero nonfinite acceleration values. Timestamps increase at 0.0005 s with no off-rate intervals under a 2-microsecond numerical tolerance. The exact grid demonstrates processed-file continuity, not independently lossless original hardware capture.

Acceleration units should be treated as **g**, as the paper's calibration table and processing description specify. The upstream README contradicts itself by listing m/s² elsewhere. Retain a units field and obtain author clarification before amplitude-sensitive physical conclusions; do not multiply the already calibrated values by 9.8 twice. All channels measure vertical vehicle response and share the time grid, but their placement and dynamic responses differ. They are suitable for joint encoding with train-only per-channel normalization, not interchangeable replicas.

| Channels | Documented placement | Audit implication |
| --- | --- | --- |
| 1, 3 | Front/back bogie | Preserve separate channels; strong response with suspension filtering |
| 2, 4 | Front/back axle | Wheel/rail excitation plus local vehicle noise; test as a sensor subset |
| 5, 6 | Front/middle body | Much smaller amplitudes; normalization must not erase placement information |

Observed global extrema span approximately -2.035 to +1.993 in the published units. No processed value approaches the paper's stated ±5 g sensor range. This provides no evidence of processed full-scale clipping, but cannot exclude upstream ADC clipping or events lost before filtering. Channel standard deviations and exact extrema by run are in `channel_statistics.csv`. Welch spectra from the first 60 s of each run/channel place very little power above 500 Hz (consistent with the stated low-pass processing). Spectra are sampled diagnostic checks, not an exhaustive spectral calibration. No universal speed-squared law was fitted or assumed.

`evidence/vibration_diagnostics.png` shows all six channels against time, Channel1 against an approximate canonical distance, and a representative PSD. The distance plot's straight-chord coordinate is explicitly diagnostic. `canonical_segments_diagnostic.csv` additionally projects cached window midpoints onto a frozen, label-independent training-run polyline; its approximate length is 1,494.049 m, rather than an exact surveyed 1.4 km. End-of-run extent differences and processing mean neither number is a surveyed replacement for the authors' nominal segment length.

## Images and label visibility

All {int(inv.images.sum()):,} names parse as local-looking date/time strings with fractional seconds. All **271 selected images decoded successfully at 1920 × 1080**; the rest have filename/size inventory only. Zero corruption in the entire image dataset is therefore not established. Recorded average frame rates are **{inv.mean_recorded_fps.min():.2f}–{inv.mean_recorded_fps.max():.2f} Hz**, not the paper's stated effective 30–40 FPS. These are rates computed from the released timestamps, not independent capture-clock measurements. Maximum recorded interframe gap is {inv.frame_dt_max.max():.3f} s. No source frame counter exists in the inspected CSVs, so exact dropped-frame counts are unknown.

The contact sheet and two full-resolution views show rails, ties, joints and turnout geometry. Numbered orange cones are visible in several frames; this is a concrete shortcut risk. A nominal joint-positive frame can show ordinary-looking rail with no confidently identifiable gap at this scale; position proximity is not a visual annotation. Turnout geometry is conspicuous but not itself proof of a fault requiring repair. Images are rendered in color in these files despite the publication describing a monochrome camera; preserve actual decoded channels and do not infer sensor hardware from file encoding alone.

**Critical run issue:** `AtoB_22.5_4` sensor timestamps are on 18 April; all 5,119 image names are on 11 April and match the image-name set in `_3`. Eleven sampled corresponding SHA-256 comparisons all match. Two files from each directory were checked against official upstream LFS object hashes at the pinned revision and match locally. This supports an upstream duplicated-image issue in the checked files, rather than a download-only error. Exclude `_4` imagery from multimodal training/evaluation until corrected; do not retime or rename it into apparent validity.

## Operating-condition matrix

{table(conditions)}

There are 15 sensor runs on 11 April and five on 18 April; date is another possible confound. Target speed is encoded in folder names. The chord-based GPS speed estimate is diagnostic and differs from targets (approximately 18.05–31.21 km/h), so do not present folder labels as measured constant velocity. Different run durations, smoothed positions and route extents also affect that estimate. Same-direction speed comparisons exist: AtoB 20 vs 22.5, BtoA 22.5 vs 25 vs 27.5, four nominal runs per cell. Both directions exist only at 22.5. There is no fully crossed speed-by-direction design. After excluding `_4` images, AtoB 22.5 has only three usable image runs.

Actual environmental extrema from all raw rows (temperature in the documented °F, humidity fraction):

{table(weather)}

Temperature/humidity are interpolated distant-weather context, not high-rate onboard physical measurements. The historical `data_quality_report.md` temperature/humidity ranges do not match these files; do not copy its numbers. Its ±50 g saturation specification also disagrees with the publication's ±5 g description; its approximately 2.8 g peaks disagree with the current inspected maxima. Its asserted speed-duration correlation -0.96 is not reproduced: current 20-run target-speed versus duration Pearson r is -0.8946, descriptive only, with direction/date confounding. `generate_eda_figures.py` Figure 6 constructs 159 equal latitude bins between 40.2319 and 40.2360 and titles them “7.62 m segments”; these span about 2.87 m northing each and have nonconstant along-route length. That figure is not evidence of a correctly implemented 7.62 m chainage segmentation. Its sharpness sample is only the first 100 images of one run, not exhaustive camera validation. The unexecuted notebook and static narrative do not repair these discrepancies. New measured evidence is in `environment_by_run.csv` and `legacy_eda_checks.json`.

## DR-Train: actual availability and reuse

{table(read('dr_train_actual_inventory.csv'))}

Actual MATLAB records were decoded with both scipy and h5py: LRV4306 waveform shape 1 × 4,667,459; LRV4313 waveform length 331,695; GPS arrays 209 × 5 and 195 × 5 in the chosen samples, all finite. Variables differ: `save_var` for acceleration and `save_var_gps` for GPS. The first simple scipy-only read failed on a v7.3 record; the follow-up h5py read succeeded. See `dr_train_decoded_records.json` for exact filenames. Sample shapes are not whole-dataset lengths or a proven sampling rate. Sampling-rate recovery must use the pass metadata/record lengths and original documentation before pretraining across datasets.

LRV4306 has five documented sensor channels; LRV4313 eight, with different orientation/placement from Rail-VIVID. The local counts indicate a partial selection, especially LRV4313 GPS. Existing station crosswalk/reconciliation work must be reused rather than rebuilt. Its reports say 298 maintenance rows, 180 deduplicated rows and 57 tamping/surfacing rows; these remain historical artifact counts, not newly validated physical repair episodes. Actual route-layout step checks across all 14 CSVs give geodesic/raw-distance ratios near one, supporting metres rather than feet. The two-region feature table is available, but 60-day pre-event labels are maintenance-proximity proxies, not Rail-VIVID defect labels. Optional self-supervised waveform pretraining would need channel adaptation, verified rates/units, training-only data selection and a no-pretraining control. It is unnecessary for the first milestone.

Sources: [Rail-VIVID official metadata](https://huggingface.co/datasets/saluslab/Rail-VIVID), [data descriptor](https://doi.org/10.1038/s41597-026-07555-y), [DR-Train official record](https://zenodo.org/records/1432702). Local processed files, not metadata prose, supply the numerical inventory above.
''')
    write('LABEL_AND_SPLIT_AUDIT.md',header+f'''# Labels, independence and split integrity

## What the label means

The code transcribes nine sites from the publication: seven joint gaps (2.0–3.0 cm) and two turnout sites, divergence and convergence. For a window midpoint, geodesic distance ≤6 m yields label 1; distance >6 m and ≤18 m yields -1 (excluded); otherwise label 0. Nearest-site ID is assigned even to distant normal windows. All cached labels and nearest IDs reproduce exactly under that code. This verifies implementation consistency, **not independently adjudicated defect truth**, sensor impulse timing, or the adequacy of the chosen radii. “Normal” means distant from the documented sites, not inspected defect-free.

| Site | Type | Published latitude | Published longitude | Joint gap cm | Local scan directory |
| --- | --- | --- | --- | --- | --- |
| 1 | Joint | 40.23553 | -77.88898 | 2.0 | Present |
| 2 | Joint | 40.23512 | -77.88770 | 2.0 | Present |
| 3 | Joint | 40.23493 | -77.88711 | 2.4 | Not found |
| 4 | Divergence | 40.23461 | -77.88612 | N/A | Present |
| 5 | Convergence | 40.23425 | -77.88485 | N/A | Present |
| 6 | Joint | 40.23388 | -77.88320 | 2.0 | Not found |
| 7 | Joint | 40.23242 | -77.87795 | 2.5 | Not found |
| 8 | Joint | 40.23205 | -77.87651 | 2.3 | Not found |
| 9 | Joint | 40.23194 | -77.87508 | 3.0 | Not found |

The actual anomaly folder contains FARO scan structures for sites 1, 2, 4 and 5, with preview images and binary scan data. No standalone nine-site machine-readable ground-truth table was found locally; the nine-site coordinates come from the paper and source constants. The upstream listing is capped, so absence of other local scan directories is not proof that the authors never collected them. All sites have vibration-proximity windows in all 20 runs; nominal image-positive pairs in 19 runs. Physical image evidence for each joint remains unadjudicated. Site-level binary presence/event detection is supportable with limitations; accurate visual localization is conditional; multiclass learning has only one divergence and one convergence site and cannot establish unseen-site generalization of those classes.

{table(sites)}

Total cached windows: 18,775 = 15,035 negatives + 1,239 positives + 2,501 excluded. Independent physical anomaly units: **nine sites**, with 180 repeated site/run traversals at most; these are not 1,239 independent defects. Run and site dependencies both matter. Use per-site tables and two-way site/run clustered uncertainty or hierarchical resampling with explicit small-cluster caveats. A window bootstrap or treating all image frames as independent produces misleadingly narrow confidence intervals.

## Exact saved split reconstruction

{table(splits)}

Split indices are saved as original cache-row IDs in `split_indices_*.json`. Random/block protocols share all nine sites and all 20 runs. Run-held-out is a valid different-run question, but does not test unseen sites. Spatial-held-out separates positive site IDs, yet shares runs and lacks guards at region boundaries. Combined excludes both test runs and test sites from training; its 1,178 test windows contain 81 positives from only sites 7/8/9 and four runs. Off-diagonal run/sector combinations are discarded: combined retained total is 6,105 of the 16,274 nonbuffer windows. Report that composition rather than calling it the same test population as other splits.

## Executable overlap findings

{table(overlap[['protocol','partition_a','partition_b','overlapping_window_pairs','shared_runs','shared_positive_sites']])}

Overlap means intersecting half-open raw intervals from the same run. Every interval is 1,000 samples and stride is 500. The block splitter groups by `cumcount()` **after removing buffer windows**, so blocks need not even cover a contiguous five-second span. It never purges cross-block neighbours. Thus its claim of “overlap-free” is false. Spatial Voronoi boundaries also cut normal overlapping windows: 20 train/validation and 20 validation/test pairs. Spatial train/test happens to have zero direct raw overlap here, but validation leakage remains. Combined has zero pairwise raw overlap and no shared positive sites/runs; this does not verify image footprints, geographic uncertainty guards, earlier tuning independence, or source preprocessing provenance.

The seven original tests pass while these violations exist, because row-index disjointness is weaker than source-sample disjointness. Four new tests exercise actual source overlap, half-open endpoints, nested intervals, and purging. `interval_guards.py` preserves test identities and purges validation then training. Additive corrected manifests produce:

{table(read('purged_split_counts.csv'))}

These purged manifests remove **raw-sample overlap only**. No replacement metrics were trained on them, and they are not certified multimodal splits. Multimodal splitting must guard the union of waveform support, camera ground footprint, all candidate frames and the alignment uncertainty envelope, including adjacent physical sectors in different runs.

## Canonical location and candidate folds

`canonical_segments_diagnostic.csv` assigns the same increasing A-to-B distance convention to forward/reverse windows using a polyline from training run AtoB_20_1; this avoids direction-specific segment IDs. Cross-run median distance to that processed reference is submillimetric for many runs, while end/extent differences reach approximately 49 m. This implausibly precise agreement is **processed-coordinate consistency**, not measured GNSS absolute accuracy. Closest projection to a finite reference also inflates endpoint deviations. Obtain original positioning/correction metadata before claiming submetre surveying or “perfect spatial repeatability.” No branch topology was inferred from latitude/longitude or turnout labels.

| Segment size | Defensible use under unresolved image drift |
| --- | --- |
| 5 m | Vibration-only diagnostic possible; hard image labels unjustified without much smaller measured residuals |
| 7.62 m | Historic engineering unit, not a dataset-supported localization resolution; same prerequisite |
| 10 m | Useful sensitivity scale; still smaller than the stated possible camera displacement |
| 15 m | Sensitivity scale only; a 15 m error can cross a whole segment |
| 25 m | Coarse aggregation may help, but still has boundary errors; requires soft assignment/guards and localization reporting |

Choose resolution from measured alignment uncertainty and sensor support, not an arbitrary fixed length. The official ~15 m drift is a possible bound, not an estimated Gaussian standard deviation. No fixed section size makes it disappear.

`candidate_fold_counts.csv` demonstrates three alternative three-site test folds (1/2/3, 4/5/6, 7/8/9), each with both classes and five training sites plus one validation site. Combined test positives are respectively 78, 79 and 81 before image exclusions. These folds are **feasibility counts, not pristine confirmatory tests**: they reuse this small dataset, their unpurged spatial forms have boundary risks, and fold 2 uniquely holds out turnout types. Report joint-only and turnout strata. For future work preregister outer site folds and inner site/run selection, expand validation beyond a single site where feasible, freeze all hyperparameters before outer evaluation, rotate held-out runs and disclose all previously viewed scores. New random seeds cannot undo earlier test inspection.
''')
    write('MULTIMODAL_ALIGNMENT_AUDIT.md',header+f'''# Image–vibration alignment

## Clocks and physical correspondence

`Acquisition.py` timestamps images using `datetime.now()` after `cap.read()` and display work, immediately before `cv2.imwrite`. DAQ acquisition uses a hardware sample clock with system-clock file timestamps; final processed CSVs contain uniform reconstructed 2 kHz date/time values. A common laptop clock supplies an absolute reference, but there is no inspected per-exposure hardware synchronization signal. Timestamp precision is not exposure accuracy. The upstream README explicitly warns of variable image-write timing and nonconstant positional drift up to ~15 m, despite more optimistic synchronization language in the paper. No verified per-run correction function or residual uncertainty distribution is supplied locally. Camera look-ahead distance, axle/bogie/body offsets and direction-dependent footprint further separate a frame timestamp from the track patch that generated a response.

Keep four distinct fields: nominal recorded timestamp/position; surveyed or independently verified true landmark position; estimated corrected image position; uncertainty of that estimate. This audit constructs nominal pairs only. It does not invent true positions from anomaly proximity.

## Executed pairing tests

Every image filename was parsed and matched to its own run's nearest sensor row. A nominal frame pair requires time inside the sensor record, residual ≤1 ms, and 500 preceding/following rows for a complete 1,000-row waveform. Attempted: **105,901**; accepted nominal pairs: **100,222**; rejected: **5,679**. Of these, 5,493 frames are outside their run's sensor interval; the remaining 186 lack full waveform context near boundaries. Accepted coverage is 19 runs and all nine proximity-labelled sites. **Physically validated/corrected pairs: zero established by this audit.** The submillisecond nearest-row residual quantifies numerical association, not the physical drift.

| Site | Nominal positive frame pairs | Runs |
| --- | --- | --- |
'''+'\n'.join(f'| {int(i)} | {int(g.nominal_pairs.sum())} | {int(g.nominal_pairs.gt(0).sum())} |' for i,g in pairs[pairs.site_id.gt(0)].groupby('site_id'))+f'''

`nominal_pair_manifest.csv` provides 190 concrete sensor intervals with image paths (one nominal normal plus nine nominal site-positive samples per accepted run). Actual imagery was decoded; successful pairing is not inferred from an intended folder layout. `pair_counts.csv` and `run_inventory.csv` contain all per-run counts.

An alternative view pairs **cached vibration-window midpoints** to the nearest frame in the same run with a declared 100 ms numerical tolerance: 17,742 of 18,775 windows have candidates. This is a different pairing rule and count, explicitly recorded in `window_pair_manifest.csv`. It does not repair the 15 m problem. When buffer windows are excluded, counts under the existing exact splits are:

{table(paired)}

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
''')
    display=metrics[['model','split','modality','precision','recall','f1','roc_auc','pr_auc','calibrated_threshold']]
    write('EXISTING_BENCHMARK_VERIFICATION.md',header+f'''# Existing benchmark verification

## Exact reproduction

Read the existing source and saved JSON; trained the exact Random Forest configurations on the unchanged external cache and deterministic reconstructed split rows. RF uses 100 trees, maximum depth 10, balanced class weights, seed 42 and `n_jobs=-1`. StandardScaler is fitted only on training features. Feature dimensions: 66 vibration, 68 with GPS. Validation threshold maximizes F1 over 33 cutoffs from 0.1 to 0.9 inclusive (strict `>` for best-score updates selects the first tied threshold). Prediction uses score ≥ cutoff. RF scaling is redundant in principle but retained for exact reproducibility. No test threshold selection was performed by this audit.

All ten RF rows reproduce precision, recall, F1, ROC-AUC, average precision and selected threshold to the four decimals in the saved JSON. Maximum unrounded comparison error after loading JSON is ≤1.2e-16. This is a fresh deterministic rerun, not evidence of multi-seed uncertainty. Predictions with original cache IDs are saved separately in `evidence/rf_predictions_*.csv`; prior outputs remain untouched.

{table(display)}

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
''')
    write('GPU_MODEL_FEASIBILITY.md',header+'''# Model and compute feasibility

## Available compute and executed test

The user confirms an **RTX 5070 is available for training**; this laptop is the local audit host. Remote 5070 VRAM, drivers and environment were not accessed, so no exact device-memory claim is made. The local machine has an RTX 3050 Laptop 6 GB, approximately 16.87 GB physical RAM, roughly 50 GB free on C: and 1.18 TB on D: at inspection. Its torch is `2.10.0+cpu`, torchvision `0.25.0+cpu`, and CUDA is unavailable to that installation. This is a local environment detail, **not a research blocker**. No package replacement was performed.

A real two-window batch from AtoB_20_1 passed the existing raw CNN on CPU: input `[2,6,1000]`, finite output `[2,2]`, 18,770 parameters. The model's fit wrapper currently keeps data/model on CPU; add explicit device selection and stream batches for remote training. No accuracy was inferred from this forward pass. No pretrained DINOv2/ConvNeXt weights or project model checkpoints were found in the inspected cache/output paths; two unrelated text-model caches exist. Transformers/torchvision are installed, timm is not. No DINO forward pass or complete fusion architecture was executed, so end-to-end DINO+CNN feasibility remains conditional on weight acquisition and a 5070 smoke test.

## Transparent memory estimates, not measurements

| Representation/model | Estimate and assumption |
| --- | --- |
| Raw 0.5 s window | 6 × 1,000 × float32 = 24,000 bytes; batch 64 input = 1.536 MB |
| All cached waveform windows materialized | 18,775 × 24,000 = 450.6 MB, with duplicate overlapping content; prefer lazy reads/memmaps |
| Existing feature matrix | 18,775 × 66 × float32 ≈4.96 MB |
| Frozen 384-D image embeddings | 105,901 × 384 × float32 ≈162.66 MB, or ≈81.33 MB fp16; candidate subsampling reduces this |
| Decoded resized images cached as float32 | 105,901 × 224 × 224 × 3 × 4 ≈63.77 GB; avoid full RAM materialization |
| CNN input batch | Input memory is tiny; activations/optimizer exceed it. Existing model likely needs well under 1 GB for modest batches, but this is an estimate to profile |
| Candidate-token attention | With B=32, 4 heads, 15 vibration queries and 9 image keys, attention-score tensor has 17,280 values (~69 KB float32), excluding projections/gradients |
| Full raw temporal attention | For B=32, 4 heads, length 1,000, naive scores alone are 128 million values (~512 MB); multiple layers/backprop add substantial memory. Patch/downsample first |
| Spectrogram example | Six channels ×129 bins ×12 frames ×float32 ≈37 KB/window for n_fft=256, hop=64, no boundary padding on 1,000 samples |

For any actual model, include weights, gradient, optimizer state, activations and decoder batches in profiling. A rough Adam parameter budget in fp32 training is 16 bytes per trainable parameter before activations (weight, gradient, first and second moments); mixed-precision implementation/master weights can change it. Frozen feature extraction removes optimizer and gradient state for that backbone. Do not equate input tensor size with peak VRAM or claim these calculations are measured 5070 capacity.

On the training machine, run `nvidia-smi` and capture torch/CUDA versions, verify one real raw batch, then a frozen pretrained image batch at 224 px. Record peak allocated and reserved memory, throughput, finite gradients and logits. Start with batches 8–16 for image extraction and 32–64 for tiny raw/embedding fusion; adjust from actual measurements. Cache features and pin the encoder revision. These are starting configurations, not tested hardware limits.

## Architecture assessment

| Method | Feasibility | Evidence and caveat |
| --- | --- | --- |
| Raw CNN / small TCN | VERIFIED FEASIBLE for raw CNN execution; TCN conditional | Six synchronized finite channels and CPU smoke test; no validated deep generalization score yet |
| Frozen DINOv2 ViT-S/14 / ConvNeXt | CONDITIONALLY FEASIBLE | Actual images decode; pretrained weights and 5070 forward pass still required; fine-tuning is not justified initially |
| Cross-attention | CONDITIONALLY FEASIBLE | Computationally small with candidate tokens; requires real pair calibration and strict candidate-support boundaries |
| GCN/GATv2/ST-GNN | CONDITIONALLY FEASIBLE computationally | Short corridor supports a line graph only after canonical mapping; branch connectivity and added scientific value not established |
| DANN/GRL | CONDITIONALLY FEASIBLE | Same-direction multiple target speeds exist; very few runs per cell, site/type/date confounds, target-data access rules must be explicit |
| Evidential/Dirichlet fusion | CONDITIONALLY FEASIBLE | Easy head implementation does not establish calibrated epistemic uncertainty; nine sites limit calibration/OOD claims |
| Raw unpatched Transformer | Not recommended | Long-token cost and low site diversity; no evidence of advantage over tiny CNN/TCN |

Use normal training statistics only, no normalization over test runs or all runs. Preserve per-channel scale metadata and compare waveform normalization choices. Spectrograms offer an alternate representation but are not a second physical modality. Do not claim new sensors from reshaping one signal into an image. DR-Train pretraining is optional and currently less urgent than split/alignment work.
''')
    write('RESEARCH_NOVELTY_MATRIX.md',header+'''# Novelty and prior art

Search performed on 2 November 2026 using exact dataset name, dataset DOI, Rail-VIVID plus fusion/cross-attention/benchmark, Wang image–vibration graph fusion, and Radosavljevic visual/inertial misalignment. Primary publication/author-institution/repository evidence was preferred. This is a focused literature check, not proof of absence across all publications, indexes or unpublished work. Some publisher/PMC pages were blocked; local extracted Rail-VIVID paper text and indexed primary records supplied context. Do not claim a systematic review was completed.

| Study / primary source | Established contribution relevant here | Difference from this project / remaining evidence |
| --- | --- | --- |
| [Ma et al., 2026, Rail-VIVID](https://doi.org/10.1038/s41597-026-07555-y) | Public repeated-run image/vibration data; four normal-image reconstruction baselines | Existing local RF uses GPS-proximity window labels; protocols/supervision differ. No direct performance superiority established |
| [Wang et al., 2025, PJR-GAFN](https://www.mdpi.com/2504-2289/9/5/127) | Image–vibration rail-surface diagnosis with progressive joint graph-attention fusion | Broad graph fusion/heterogeneous representation novelty is already occupied. Not verified as a Rail-VIVID spatial-holdout study |
| [Radosavljevic et al., 2026, institutional publication record](https://satie.ens-paris-saclay.fr/fr/publications/multimodal-railway-defect-detection-visual-and-inertial-data-early-fusion-strategies) | Visual/inertial early fusion and spatial desynchronization analysis; compares simple concatenation and multiplicative fusion | Directly overlaps the generic alignment-robustness question. Published institutional record dated 5 October 2026; linked HAL full text inaccessible here, so detailed split/dataset claims remain unverified |
| [Zhukov et al., 2025, FusWay preprint](https://arxiv.org/abs/2509.06987) | Railway vision/audio fusion using detector and transformer representations | Different modalities/dataset; evidence against claiming all railway multimodal fusion is new. This is not the Radosavljevic paper |
| [Xu et al., online 2025 / issue 2026](https://doi.org/10.1177/03611981251378485) | Rail-surface state assessment with images and vibration | Further direct application prior art; public publisher metadata verified, detailed experimental comparability not established |
| [Zhang et al., 2026, variable-scale pattern alignment](https://link.springer.com/article/10.1007/s10044-026-01620-3) | Rail image/axle-box alignment using feature-point sequences and uncertainty-related calibration; author code/data linked as VSAA | Direct alignment prior art, published 5 February 2026. Public abstract reviewed, detailed transfer to Rail-VIVID untested; consider as an alignment comparator with explicit label-access restrictions |
| [Ghiasi et al., 2025, railway UDA](https://doi.org/10.1016/j.engappai.2024.109516) | Cross-line drive-by domain adaptation and sensor-layout sensitivity | Adaptation and placement analysis are established railway research; Rail-VIVID same-direction speed shifts are a narrower evaluation, not general domain-adaptation novelty |
| [Liu et al., 2019, DR-Train record](https://zenodo.org/records/1432702) | Long-term vehicle response, location and maintenance schedules | Different target construction and sensor setup; labels cannot be transferred to Rail-VIVID |
| Current project (completed) | RF/feature-autoencoder/Isolation Forest CPU comparisons, spatial/run proxy-label protocols | Reproducible RF; residual boundary leakage; no completed deep multimodal benchmark |
| Candidate contribution (uncompleted) | Reproducible site/run benchmark with source-support guards, duplicate-image rejection, alignment uncertainty, matched-cohort fusion controls | Narrow, plausible contribution; must demonstrate calibrated alignment and robust held-out-site improvement or a useful negative result |

## Five claims adjudicated

1. **“No published end-to-end multimodal Rail-VIVID benchmark exists.” NOT VERIFIED as a universal novelty claim.** Focused searches found no clearly verified independent paper with that exact dataset/protocol, but absence of a retrieved match is not evidence of absence. Phrase narrowly: “we evaluate [explicit protocol] on Rail-VIVID,” and update citations before submission. Image–vibration fusion and misalignment robustness already have prior art.
2. **“0.96 ROC-AUC beats the paper's 0.73.” INVALID direct comparison.** The paper's number is image-level normal-reconstruction detection. Local RF/deep targets are spatial window proximity labels, with supervised positives, different observation units and held-out groups. Rerun comparable vision-only and multimodal baselines on one frozen cohort before claiming improvement.
3. **“Cross-attention natively fixes drift.” UNSUPPORTED.** Fusion may exploit correlations without recovering correspondence. Requires positional candidate support, measured/withheld alignment targets, and comparisons with pooling/interpolation.
4. **“High-end GPUs make F1≥0.80 achievable.” UNSUPPORTED.** Compute enables execution, not new sites or better labels. Targets F1≥0.75, AP≥0.70 and ROC-AUC≥0.96 are aspirations without evidence under the hardest strict split; no likelihood or numerical feasibility score is justified.
5. **“The benchmarks are leakage-free.” FALSE as a blanket statement.** Random and block share raw samples; spatial has validation-boundary overlap; combined passes raw/run/site intersection checks but remains exploratory and lacks multimodal footprint verification.

## Contribution that could survive review

Preregister outer physical-site/run folds; publish machine-readable source supports, nominal/corrected pairs and uncertainty; compare RF, raw CNN, frozen vision, late/concatenation/candidate pooling and one spatial-attention variant on identical accepted observations. Validate marker corrections out of sample and remove cone shortcuts. Show whether a measured advantage persists across sites, run repeats and known offset perturbations, with event recall/false alarms and small-cluster uncertainty. Report failure and site heterogeneity openly. The contribution is protocol and alignment evidence, with model improvements conditional on those results.

Do not claim future maintenance prediction, universal rail topology learning, physically proved sensor transmissibility, state-of-the-art across incompatible datasets, first use of attention/GNN/Dirichlet fusion, cross-railway generalization, or guaranteed conference acceptance. A result based only on thousands of correlated windows and one favourable test split will be weak evidence.

The local `papers` companion summaries include bibliographic/access inconsistencies: the Wang summary calls the paper paywalled even though the publisher search record describes open access, and an evidential-method summary uses DOI `10.1088/1361-6501/ad1234`, which was not verified through the attempted primary lookup. Treat that evidential citation as NOT VERIFIED rather than established related work. No exhaustive validation of all 20 local paper summaries was performed. The stronger Zhang alignment prior art was found in the workspace brief and checked at its publisher; this further narrows any defensible novelty claim.
''')
    write('RECOMMENDED_ARCHITECTURE.md',header+'''# What to build

**Primary contribution:** a reproducible benchmark of railway anomaly-site detection under physical-location and run shifts, with explicitly measured image-alignment uncertainty. **Primary first deep model:** the existing lightweight six-channel raw CNN. **First multimodal model:** frozen DINOv2 ViT-S/14 features plus that CNN, with a small concatenation classifier and late-fusion control. A true residual/dilated TCN is optional if the existing CNN proves inadequate on validation; do not rename the existing sequential CNN a TCN implementation.

## Six proposed research experiments, ranked

| Rank | Experiment | Recommendation | Scientific value / data fit / main risk / reproducibility / mini-project scope |
| --- | --- | --- | --- |
| 1 | F: strict generalization benchmark | Primary, mandatory | Directly exposes verified evaluation problems; low compute, reusable artifacts, strong reproducibility opportunity; nine sites constrain external claims |
| 2 | E: sensor placement analysis | Essential compact ablation | Six real placements, same labels/splits, low compute; publication already studies channel responses, so classification subset novelty alone is modest |
| 3 | A: deep multimodal fusion | Conditional main model study | Real data in 19 paired runs; first simple fusion, then attention only if alignment controls justify it; highest pair/shortcut risk |
| 4 | D: operating-condition robustness | Secondary controlled experiment | Matched-direction cells exist; evaluate held-out speed with run/site controls before DANN; only four nominal repeats/cell and date confounds |
| 5 | B: uncertainty/reliability | Optional evaluation extension | Missing channels/blur/noise can be stress-tested; temperature scaling/ensemble first; small site count makes calibration claims fragile |
| 6 | C: spatial-temporal GNN | Defer | Short single corridor, no validated branch topology, likely competing with smoothing; more complexity and leakage exposure without proven benefit |

This ranking is qualitative, not a fabricated novelty score. F/E are strongest in data suitability, baseline availability, compute cost and reproducibility. A has the best conditional multimodal contribution but requires more work. D/B/C add selection burden and reduce mini-project feasibility if pursued simultaneously. A scientifically useful system can display relative track distance, anomaly score and uncertainty/missing-modality status without pretending to forecast repairs.

## Model sequence and baselines

1. **RF reference:** preserve existing 66-feature implementation, rerun only on repaired/frozen cohorts. Include GPS-only and GPS+vibration as diagnostic controls, not default production features.
2. **Raw CNN:** `[B,6,1000]`, train-only per-channel normalization, existing 18,770-parameter network, weighted cross-entropy, deterministic data order/seed, validation AP for checkpoint selection and validation-selected operating threshold. Store complete logits and supports. Evaluate 0.5 s versus a small prespecified longer-context choice only in inner validation.
3. **Vision:** freeze pretrained small encoder; rail ROI at documented 224 px transform, normalized to its official preprocessing; 384-D DINO embeddings if using that model. A linear/small MLP head and a normal-only anomaly detector provide supervision-matched controls. Do not train a large vision model from scratch on nine sites.
4. **Simple multimodal:** project raw and frozen image embeddings to modest widths, concatenate, MLP classifier; compare calibrated score-level late fusion and mean/distance-weighted pooling of nearby image candidates. Test identical accepted rows for every modality.
5. **Conditional spatial attention:** retain CNN temporal tokens rather than only its final global average, project to 128 dimensions, attend to a bounded set of corrected/uncertain image candidates with relative-distance encodings and candidate masks. Freeze vision initially. The image candidate list and allowed distance prior must be constructed without held-out anomaly labels. Keep dimensions and image-candidate budget fixed across pooling/attention comparisons. Add an alignment supervision/loss only where independently labelled correspondences exist.

No target F1 is used as an acceptance gate. The current hardest RF F1 0.2815/AP 0.3360 is a reference on its exact cohort, not a lower bound or performance promise. Compare against newly rerun RF on each new cohort. Negative fusion results remain scientifically informative if uncertainty, shortcuts and evaluation composition are controlled.

## Required ablations and reporting

Use vibration-only, vision-only, GPS-only diagnostic, concatenation, late fusion, distance pooling and optional attention; nominal versus independently corrected alignment; cone-present versus suppressed; synthetic offset curves; identical source-support guards; axle 2/4, bogie 1/3, body 5/6, all six; measured/estimated speed conditioning versus none; multiple frozen seeds and site/run folds. The sensor-subset study must keep the observation cohort, hyperparameter budget and split identities fixed. Report inference cost and spectral descriptive evidence without claiming classification proves a physical law.

Primary metrics: average precision, per-site event recall, false alarms per physical kilometre/run and localization error with explicit tolerance. Also report ROC-AUC, precision/recall/F1 at one validation-fixed threshold. Merge repeated adjacent alerts into events with validation-fixed distance/time parameters; match predicted and true events one-to-one. Define “localization” separately for proximity labels and visually adjudicated defect locations. Include counts, prevalence and cluster uncertainty; do not silently remove difficult sites.

For missing-modality reliability, compare signal-only fallback, modality dropout, temperature scaling, a small deep ensemble and optional evidential head. Use NLL/Brier score, reliability diagrams, risk–coverage/AURC and ECE with stated bins, stratified by site/run and corruption type. Synthetic blur/darkening/noise/channel dropout must be labelled synthetic; severity should come from observed brightness/noise distributions or explicit sensitivity ranges. No real failure prevalence or calibrated epistemic interpretation is established by synthetic tests alone.

For any later graph experiment use a verified line graph, no imagined turnout branches. Compare local scores, neighbour concatenation, Gaussian smoothing, distance-axis 1D convolution, and GNN with normal/shuffled/removed edges. Prevent test-label propagation and cross-run summaries from using test traversals in training. For DANN, distinguish domain generalization with no target data from adaptation that uses unlabeled target-train data; never use the final target test set for representation fitting.
''')
    write('IMPLEMENTATION_ROADMAP.md',header+'''# Concrete implementation roadmap

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
''')
    decision=pd.DataFrame([
        ['Deep vibration CNN/TCN','Yes: six-channel files','Site-proximity proxies','CNN yes; TCN no','Method alone low','Nine sites; proxy timing','VERIFIED FEASIBLE for CNN; first baseline'],
        ['Vision encoder','Yes: 19 nominal paired runs','Visual labels unresolved','Loader only','Method alone low','Cones, wrong-run images','CONDITIONALLY FEASIBLE'],
        ['Basic multimodal fusion','Nominal candidates yes','Spatial proxy only','Not implemented','Benchmark opportunity','Physical drift; cohort changes','CONDITIONALLY FEASIBLE'],
        ['Spatially constrained cross-attention','Candidate construction possible','Correspondences unvalidated','No','Candidate, unproved','No measured alignment recovery','CONDITIONALLY FEASIBLE; defer'],
        ['Evidential fusion','Corruption stress inputs possible','Calibration small-cluster','No','Weak alone','Evidence need not be calibrated','CONDITIONALLY FEASIBLE; optional'],
        ['Spatial-temporal GNN','Line geometry diagnostic only','Same nine sites','No','Weak without topology/value','Smoothing; transductive leakage','CONDITIONALLY FEASIBLE for line graph; defer'],
        ['Cross-speed adaptation','Matched-direction cells yes','Proxies; sparse runs','No','Secondary','Direction/date/target-data use','CONDITIONALLY FEASIBLE'],
        ['Sensor placement analysis','Yes: mapped six channels','Same proxy labels','Feature columns reusable','Modest incremental','Selection and site dependence','VERIFIED FEASIBLE as controlled subset study'],
        ['Combined location/run evaluation','Yes: intervals and groups','Three test sites in old fold','Yes; audited','Protocol contribution possible','Prior tuning; image envelope','VERIFIED FEASIBLE for vibration grouping; multimodal conditional']
    ],columns=['Experiment','Required data exists?','Valid labels?','Existing code?','Scientific novelty','Main risk','Recommendation'])
    decision.to_csv(E/'final_decision_matrix.csv',index=False)
    write('FINAL_RESEARCH_VERDICT.md',header+f'''# Final research verdict: CONDITIONAL GO

**GO** for an auditable vibration/generalization baseline and controlled sensor study. **CONDITIONAL GO** for a benchmark-focused multimodal paper after pairing/visual-label and split-support gates. **NO-GO in current form** for claims of guaranteed high F1, automatically drift-correcting attention, universal graph superiority, fine visual localization from raw timestamps, or future maintenance forecasting.

The dataset is substantial in repeated signals, but small in independent anomalies. All 20 sensor records are readable and the RF baseline is reproducible. The main blockers are evidential: 5,119 wrong-run/duplicated image names, unmeasured physical alignment residuals, coordinate-proximity rather than visual fault labels, residual block/spatial boundary overlap, prior test reuse, and nine sites from one railway. Available RTX 5070 compute is not the limiting factor.

Recommended title: **“Beyond Random Windows: Auditing Spatial and Run Generalization for Railway Anomaly Detection on Rail-VIVID.”** After genuine multimodal and alignment experiments exist, use **“Railway Anomaly Detection under Spatial and Run Shifts: A Multimodal Benchmark with Alignment Uncertainty.”** Avoid committing the title to cross-attention until it earns an advantage over simple fusion. The most credible conference contribution is a reproducible benchmark/protocol plus a carefully controlled alignment/fusion result, including a defensible negative finding if complex fusion fails.

## Decision matrix

{table(decision)}

“VERIFIED FEASIBLE” refers to the stated scope, never verified future accuracy. “CONDITIONALLY FEASIBLE” names an unmet prerequisite. A branched physical-topology GNN is **NOT CURRENTLY FEASIBLE** from the inspected evidence; first-in-literature status and remote 5070 execution are **NOT VERIFIED**. No fabricated feasibility percentages were assigned.

## Explicit answers to all 30 questions

1. **Required images/vibration?** Yes: 9,401,255 six-channel rows and 105,901 image files. One image run is temporally misassigned; 19 runs have nominal timestamp pairs.
2. **Reliable alignment?** Numerical same-run matching works; reliable physical alignment is not established. The audit has 100,222 nominal frame/window pairs and zero independently corrected pairs. Measure landmark residuals first.
3. **Nine sites sufficient for deep evaluation?** For a small, explicitly exploratory grouped study, yes. For broad fault-family/railway generalization or precise confidence claims, no. Repeated windows do not add independent defects.
4. **Unseen anomaly locations testable?** Yes within this corridor: preserve disjoint site IDs and source/position support; old combined test holds out sites 7/8/9. It is not a new-railway test.
5. **Enough independent observations for proposed architecture?** Enough waveform/image observations for a small pretrained/frozen-encoder model; insufficient evidence for a high-capacity end-to-end model to generalize across defect populations. A large sample count does not resolve nine-site diversity.
6. **Six channels usable/comparable?** All finite, synchronized at 2 kHz and mapped to real placements. Comparable through explicit units and train-only channel handling, not interchangeable amplitudes. Original units metadata is inconsistent and must be documented.
7. **Fair speed/direction tests?** Same-direction speed shifts are possible; only 22.5 km/h exists both ways. Full independent factorial speed/direction conclusions are unsupported, and actual speed/date require controls.
8. **CPU benchmarks reproducible?** Ten RF rows exactly at saved precision; combined Isolation Forest matches its true saved row. Autoencoder matches the supplied three-decimal summary only, with small four-decimal metric differences and a 0.0042 cutoff difference. Claimed Isolation Forest combined tuple is misattributed from run-held-out; claimed DR-Train range is not reproduced.
9. **Splits leakage-free?** No blanket claim. Block has 196 train/test shared-sample pairs; spatial has 20 train/validation and 20 validation/test pairs. Combined passes raw/run/site-disjoint checks, with remaining image/support/tuning qualifications.
10. **GPS experiments interpretable?** As descriptive same-protocol controls, yes. They do not prove memorization causally; labels themselves depend on GPS, and confounded split/composition/threshold changes need controlled ablations.
11. **Labels correct?** The published-coordinate proximity rule is reproduced exactly. Correct visual defect labels, actual impulse support and defect-free negatives are not independently established. Turnouts are not automatically damage events.
12. **Earlier tuning contaminated tests?** Scores were repeatedly viewed and historical test-distribution thresholding existed. Complete architecture/hyperparameter history is unavailable. Treat old tests as exploratory; freeze a new nested grouped evaluation, disclose prior reuse, and seek external/new sites for confirmation.
13. **DINOv2 plus raw encoder implementable?** Yes in principle with readable images/raw signals and available training compute. Only the raw CNN forward pass was executed here; weights, paired dataset and full vision/fusion smoke test remain prerequisites.
14. **Cross-attention justified now?** Not yet. First demonstrate usable paired information and improvement over frozen-encoder vision, late fusion, concatenation and candidate pooling on matched cohorts.
15. **Can it handle uncertain alignment?** Conditionally, with candidate positions, uncertainty/support masks and measured correspondence controls. Plain attention does not inherently recover clock drift.
16. **GNN beyond smoothing?** No supporting experiment yet. Short corridor and no verified sampled branch topology make line convolution/smoothing essential cheaper comparators; defer the GNN.
17. **Evidential fusion reliable evaluation?** Synthetic sensor-failure/corruption testing is possible; robust calibration/epistemic claims remain weak with nine sites. Compare temperature scaling/ensembles and risk–coverage first.
18. **Cross-speed without direction confounding?** Yes for within-AtoB or within-BtoA comparisons, conditionally on run/date/site controls. Cross-direction changes at different speeds are a combined condition shift. DANN must disclose target data access.
19. **Strongest novelty opportunity?** An explicit, reproducible source-support and alignment-aware site/run benchmark with fair fusion controls, not six architectural additions.
20. **Beyond published baseline?** The existing vibration/grouped evaluation studies a different question; a completed, comparable multimodal benchmark would add evidence. Current code has no completed multimodal deep result.
21. **Already-published problem?** Yes: image–vibration rail fusion (Wang 2025), visual/inertial misalignment analysis (Radosavljevic 2026), explicit image/axle-box alignment (Zhang 2026), and railway domain adaptation (Ghiasi 2025) exist. Exact Rail-VIVID protocol priority remains unverified, not a “first” claim.
22. **Experiment establishing new contribution?** Independently validate alignment; exclude wrong-run imagery and cone shortcuts; compare simple and spatial-candidate fusion on frozen outer site/run folds with identical observations and offset curves. Show repeatable site-level/event improvement or rigorous limits that prior benchmarks conceal.
23. **F1≥0.75/ROC-AUC≥0.96 plausible?** Speculative aspirations, unsupported under the hardest strict protocol. RF combined F1 is 0.2815 on its exact cohort; do not promise a target or extrapolate from GPU capacity.
24. **Claims a reviewer would reject?** Unqualified leakage-free splits; causal overlap/GPS conclusions from changed splits; 0.96 vs 0.73 as direct improvement; thousands of independent defects; perfect timestamp alignment; first multimodal/misalignment method; future repairs; guaranteed high performance or acceptance.
25. **Final title?** Use the benchmark-focused title above now. Adopt the multimodal title only after paired experiments are validated; omit cross-attention superiority until demonstrated.
26. **Most credible conference contribution?** A transparent Rail-VIVID benchmark with reproducible RF/CNN baselines, source-support guards, multiple site/run folds, measured alignment limitations and controlled multimodal/sensor-ablation evidence. Publication strength depends on completed evidence and venue fit; acceptance is not guaranteed.
27. **First code?** A versioned data/provenance manifest and support-aware split audit, integrating the additive interval guards and rejecting the wrong-date image run. Complete alignment calibration next. First GPU model is the existing raw CNN.
28. **Reusable code?** CSV/image loader, 66-feature extractor, site constants, statistical models, raw CNN, validation calibration functions, old experiment cache and DR-Train station crosswalk. Preserve old protocol code/results for reproduction; don't duplicate the whole tree.
29. **Avoid?** Six-model parallel research, invented graph branches, raw unpatched Transformers, vision training from scratch, premature end-to-end fine-tuning, DR labels transferred to Rail-VIVID, test-based selection, uncalibrated fine localization, and architectural claims supported only by a random split.
30. **Smallest valid next GPU baseline?** On RTX 5070, run a real finite forward/backward smoke test and short six-channel 0.5 s CNN training using frozen purged inner groups, train-only statistics and validation-only checkpoint/threshold selection. Save provenance/predictions and compare with same-cohort RF only after outer evaluation settings are frozen. No advanced fusion is needed for this first baseline.

Read `IMPLEMENTATION_ROADMAP.md` for acceptance gates and `EXISTING_BENCHMARK_VERIFICATION.md` for reproduced metrics. Literature basis and qualified novelty claims are linked in `RESEARCH_NOVELTY_MATRIX.md`.
''')
    write('README.md','''# Research audit — 2 November 2026

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
''')
    print('Wrote 10 reports + README',flush=True)

if __name__=='__main__': main()
