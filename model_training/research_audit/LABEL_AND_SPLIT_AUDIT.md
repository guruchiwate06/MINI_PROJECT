Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# Labels, independence and split integrity

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

| nearest_anomaly_id | positive_windows | runs |
| --- | --- | --- |
| 1 | 135 | 20 |
| 2 | 137 | 20 |
| 3 | 135 | 20 |
| 4 | 137 | 20 |
| 5 | 138 | 20 |
| 6 | 138 | 20 |
| 7 | 135 | 20 |
| 8 | 133 | 20 |
| 9 | 151 | 20 |

Total cached windows: 18,775 = 15,035 negatives + 1,239 positives + 2,501 excluded. Independent physical anomaly units: **nine sites**, with 180 repeated site/run traversals at most; these are not 1,239 independent defects. Run and site dependencies both matter. Use per-site tables and two-way site/run clustered uncertainty or hierarchical resampling with explicit small-cluster caveats. A window bootstrap or treating all image frames as independent produces misleadingly narrow confidence intervals.

## Exact saved split reconstruction

| protocol | partition | windows | positive | negative | runs | sites |
| --- | --- | --- | --- | --- | --- | --- |
| random | train | 10579 | 806 | 9773 | 20 | 1,2,3,4,5,6,7,8,9 |
| random | validation | 2441 | 190 | 2251 | 20 | 1,2,3,4,5,6,7,8,9 |
| random | test | 3254 | 243 | 3011 | 20 | 1,2,3,4,5,6,7,8,9 |
| block | train | 10615 | 772 | 9843 | 20 | 1,2,3,4,5,6,7,8,9 |
| block | validation | 2392 | 211 | 2181 | 20 | 1,2,3,4,5,6,7,8,9 |
| block | test | 3267 | 256 | 3011 | 20 | 1,2,3,4,5,6,7,8,9 |
| run | train | 10701 | 813 | 9888 | 13 | 1,2,3,4,5,6,7,8,9 |
| run | validation | 2446 | 188 | 2258 | 3 | 1,2,3,4,5,6,7,8,9 |
| run | test | 3127 | 238 | 2889 | 4 | 1,2,3,4,5,6,7,8,9 |
| spatial | train | 6751 | 682 | 6069 | 20 | 1,2,3,4,5 |
| spatial | validation | 3295 | 138 | 3157 | 20 | 6 |
| spatial | test | 6228 | 419 | 5809 | 20 | 7,8,9 |
| combined | train | 4423 | 449 | 3974 | 13 | 1,2,3,4,5 |
| combined | validation | 504 | 22 | 482 | 3 | 6 |
| combined | test | 1178 | 81 | 1097 | 4 | 7,8,9 |

Split indices are saved as original cache-row IDs in `split_indices_*.json`. Random/block protocols share all nine sites and all 20 runs. Run-held-out is a valid different-run question, but does not test unseen sites. Spatial-held-out separates positive site IDs, yet shares runs and lacks guards at region boundaries. Combined excludes both test runs and test sites from training; its 1,178 test windows contain 81 positives from only sites 7/8/9 and four runs. Off-diagonal run/sector combinations are discarded: combined retained total is 6,105 of the 16,274 nonbuffer windows. Report that composition rather than calling it the same test population as other splits.

## Executable overlap findings

| protocol | partition_a | partition_b | overlapping_window_pairs | shared_runs | shared_positive_sites |
| --- | --- | --- | --- | --- | --- |
| random | train | validation | 3109 | 20 | 9 |
| random | train | test | 4161 | 20 | 9 |
| random | validation | test | 941 | 20 | 9 |
| block | train | validation | 150 | 20 | 9 |
| block | train | test | 196 | 20 | 9 |
| block | validation | test | 47 | 20 | 9 |
| run | train | validation | 0 | 0 | 9 |
| run | train | test | 0 | 0 | 9 |
| run | validation | test | 0 | 0 | 9 |
| spatial | train | validation | 20 | 20 | 0 |
| spatial | train | test | 0 | 20 | 0 |
| spatial | validation | test | 20 | 20 | 0 |
| combined | train | validation | 0 | 0 | 0 |
| combined | train | test | 0 | 0 | 0 |
| combined | validation | test | 0 | 0 | 0 |

Overlap means intersecting half-open raw intervals from the same run. Every interval is 1,000 samples and stride is 500. The block splitter groups by `cumcount()` **after removing buffer windows**, so blocks need not even cover a contiguous five-second span. It never purges cross-block neighbours. Thus its claim of “overlap-free” is false. Spatial Voronoi boundaries also cut normal overlapping windows: 20 train/validation and 20 validation/test pairs. Spatial train/test happens to have zero direct raw overlap here, but validation leakage remains. Combined has zero pairwise raw overlap and no shared positive sites/runs; this does not verify image footprints, geographic uncertainty guards, earlier tuning independence, or source preprocessing provenance.

The seven original tests pass while these violations exist, because row-index disjointness is weaker than source-sample disjointness. Four new tests exercise actual source overlap, half-open endpoints, nested intervals, and purging. `interval_guards.py` preserves test identities and purges validation then training. Additive corrected manifests produce:

| protocol | partition | before | after | dropped | positive_after |
| --- | --- | --- | --- | --- | --- |
| block | train | 10615 | 10269 | 346 | 747 |
| block | validation | 2392 | 2345 | 47 | 207 |
| block | test | 3267 | 3267 | 0 | 256 |
| spatial | train | 6751 | 6731 | 20 | 682 |
| spatial | validation | 3295 | 3275 | 20 | 138 |
| spatial | test | 6228 | 6228 | 0 | 419 |

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
