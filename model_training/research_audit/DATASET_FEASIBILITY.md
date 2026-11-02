Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# Dataset feasibility

## Rail-VIVID: direct measurements

The local dataset contains **9,401,255 vibration time rows**, six channels per row, and **105,901 JPG files** across 20 run folders. Thus there are 56,407,530 scalar acceleration values; “9.4 million readings” is a row count, not a six-channel scalar count. CSVs total 1,889,194,775 bytes; frame files total 112,078,209,305 bytes. Anomaly scans add further storage.

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

All 105,901 names parse as local-looking date/time strings with fractional seconds. All **271 selected images decoded successfully at 1920 × 1080**; the rest have filename/size inventory only. Zero corruption in the entire image dataset is therefore not established. Recorded average frame rates are **20.08–25.54 Hz**, not the paper's stated effective 30–40 FPS. These are rates computed from the released timestamps, not independent capture-clock measurements. Maximum recorded interframe gap is 0.899 s. No source frame counter exists in the inspected CSVs, so exact dropped-frame counts are unknown.

The contact sheet and two full-resolution views show rails, ties, joints and turnout geometry. Numbered orange cones are visible in several frames; this is a concrete shortcut risk. A nominal joint-positive frame can show ordinary-looking rail with no confidently identifiable gap at this scale; position proximity is not a visual annotation. Turnout geometry is conspicuous but not itself proof of a fault requiring repair. Images are rendered in color in these files despite the publication describing a monochrome camera; preserve actual decoded channels and do not infer sensor hardware from file encoding alone.

**Critical run issue:** `AtoB_22.5_4` sensor timestamps are on 18 April; all 5,119 image names are on 11 April and match the image-name set in `_3`. Eleven sampled corresponding SHA-256 comparisons all match. Two files from each directory were checked against official upstream LFS object hashes at the pinned revision and match locally. This supports an upstream duplicated-image issue in the checked files, rather than a download-only error. Exclude `_4` imagery from multimodal training/evaluation until corrected; do not retime or rename it into apparent validity.

## Operating-condition matrix

| direction | target_speed_kmh | runs | rows | images | nominal_pairs |
| --- | --- | --- | --- | --- | --- |
| AtoB | 20.0000 | 4 | 2207963 | 24213 | 24095 |
| AtoB | 22.5000 | 4 | 1954513 | 22090 | 16894 |
| BtoA | 22.5000 | 4 | 2079336 | 23840 | 23722 |
| BtoA | 25.0000 | 4 | 1631041 | 18196 | 18069 |
| BtoA | 27.5000 | 4 | 1528402 | 17562 | 17442 |

There are 15 sensor runs on 11 April and five on 18 April; date is another possible confound. Target speed is encoded in folder names. The chord-based GPS speed estimate is diagnostic and differs from targets (approximately 18.05–31.21 km/h), so do not present folder labels as measured constant velocity. Different run durations, smoothed positions and route extents also affect that estimate. Same-direction speed comparisons exist: AtoB 20 vs 22.5, BtoA 22.5 vs 25 vs 27.5, four nominal runs per cell. Both directions exist only at 22.5. There is no fully crossed speed-by-direction design. After excluding `_4` images, AtoB 22.5 has only three usable image runs.

Actual environmental extrema from all raw rows (temperature in the documented °F, humidity fraction):

| date | temperature_min | temperature_max | humidity_min | humidity_max |
| --- | --- | --- | --- | --- |
| 2025-04-11 | 40.0000 | 41.0000 | 0.6786 | 0.7300 |
| 2025-04-18 | 66.0136 | 76.0000 | 0.2824 | 0.3233 |

Temperature/humidity are interpolated distant-weather context, not high-rate onboard physical measurements. The historical `data_quality_report.md` temperature/humidity ranges do not match these files; do not copy its numbers. Its ±50 g saturation specification also disagrees with the publication's ±5 g description; its approximately 2.8 g peaks disagree with the current inspected maxima. Its asserted speed-duration correlation -0.96 is not reproduced: current 20-run target-speed versus duration Pearson r is -0.8946, descriptive only, with direction/date confounding. `generate_eda_figures.py` Figure 6 constructs 159 equal latitude bins between 40.2319 and 40.2360 and titles them “7.62 m segments”; these span about 2.87 m northing each and have nonconstant along-route length. That figure is not evidence of a correctly implemented 7.62 m chainage segmentation. Its sharpness sample is only the first 100 images of one run, not exhaustive camera validation. The unexecuted notebook and static narrative do not repair these discrepancies. New measured evidence is in `environment_by_run.csv` and `legacy_eda_checks.json`.

## DR-Train: actual availability and reuse

| vehicle | kind | files | bytes |
| --- | --- | --- | --- |
| LRV4306 | accelerometer_data | 5500 | 24752889884 |
| LRV4306 | gps_data | 5500 | 33874910 |
| LRV4313 | accelerometer_data | 5500 | 26783288507 |
| LRV4313 | gps_data | 1459 | 32869740 |

Actual MATLAB records were decoded with both scipy and h5py: LRV4306 waveform shape 1 × 4,667,459; LRV4313 waveform length 331,695; GPS arrays 209 × 5 and 195 × 5 in the chosen samples, all finite. Variables differ: `save_var` for acceleration and `save_var_gps` for GPS. The first simple scipy-only read failed on a v7.3 record; the follow-up h5py read succeeded. See `dr_train_decoded_records.json` for exact filenames. Sample shapes are not whole-dataset lengths or a proven sampling rate. Sampling-rate recovery must use the pass metadata/record lengths and original documentation before pretraining across datasets.

LRV4306 has five documented sensor channels; LRV4313 eight, with different orientation/placement from Rail-VIVID. The local counts indicate a partial selection, especially LRV4313 GPS. Existing station crosswalk/reconciliation work must be reused rather than rebuilt. Its reports say 298 maintenance rows, 180 deduplicated rows and 57 tamping/surfacing rows; these remain historical artifact counts, not newly validated physical repair episodes. Actual route-layout step checks across all 14 CSVs give geodesic/raw-distance ratios near one, supporting metres rather than feet. The two-region feature table is available, but 60-day pre-event labels are maintenance-proximity proxies, not Rail-VIVID defect labels. Optional self-supervised waveform pretraining would need channel adaptation, verified rates/units, training-only data selection and a no-pretraining control. It is unnecessary for the first milestone.

Sources: [Rail-VIVID official metadata](https://huggingface.co/datasets/saluslab/Rail-VIVID), [data descriptor](https://doi.org/10.1038/s41597-026-07555-y), [DR-Train official record](https://zenodo.org/records/1432702). Local processed files, not metadata prose, supply the numerical inventory above.
