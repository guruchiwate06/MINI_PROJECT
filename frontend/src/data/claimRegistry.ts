import { ScientificClaim } from '../types';

/**
 * SCIENTIFIC CLAIM REGISTRY
 * Fully cross-referenced against the Rail-VIVID repository files,
 * dataset metadata, model outputs, and companion publication.
 */
export const SCIENTIFIC_CLAIM_REGISTRY: ScientificClaim[] = [
  {
    id: 'CR-01',
    title: 'Anomaly 7 Mechanical Joint Gap',
    statement: 'Physical track anomaly 7 is an expansion joint gap measuring exactly 2.5 cm located at 40.23242° N, -77.87795° W.',
    verified_value: 2.5,
    unit: 'cm',
    source_file: 'data/README.md, src/evaluation/splitters.py, reports/ground_truth_audit.md',
    line_range: 'data/README.md L17; splitters.py L25; ground_truth_audit.md L35',
    code_excerpt: "{'id': 7, 'type': 'Joints', 'size_cm': 2.5, 'lat': 40.23242, 'lon': -77.87795}",
    verification_method: 'Ground-truth cross-referencing between dataset README Table 2 and splitters.py constant definitions.',
    status: 'VERIFIED',
    notes: 'Documented in Table 2 of Ma et al. (Nature Scientific Data, 2026). Sits 474.3 meters past Anomaly 6, representing the largest continuous section of preceding normal track.'
  },
  {
    id: 'CR-02',
    title: 'Definite Anomaly Zone Spatial Radius',
    statement: 'A spatial radius of 6.0 meters around documented defect coordinates defines the Definite Anomaly boundary (y = 1).',
    verified_value: 6.0,
    unit: 'meters',
    source_file: 'configs/baseline_config.json, src/evaluation/splitters.py, reports/baseline_results.md',
    line_range: 'baseline_config.json L18; splitters.py L43, L70; baseline_results.md L13',
    code_excerpt: '"anomaly_radius_m": 6.0 | labels[min_dists <= anomaly_radius_m] = 1',
    verification_method: 'Verified via configuration JSON and spatial labeling algorithm in splitters.py.',
    status: 'VERIFIED',
    notes: 'Produces 1,239 clean anomaly windows (6.6% of total clean track data) across the 20 transits.'
  },
  {
    id: 'CR-03',
    title: 'Post-Shock Suspension Ringing Buffer',
    statement: 'A spatial zone extending from 6.0 to 18.0 meters around defects is quarantined (y = -1) and excluded from all training and testing.',
    verified_value: 18.0,
    unit: 'meters',
    source_file: 'configs/baseline_config.json, src/evaluation/splitters.py, reports/evaluation_protocol.md',
    line_range: 'baseline_config.json L19; splitters.py L44, L71; evaluation_protocol.md L59-62',
    code_excerpt: 'labels[(min_dists > anomaly_radius_m) & (min_dists <= buffer_radius_m)] = -1',
    verification_method: 'Checked in baseline config and splitting logic in splitters.py.',
    status: 'VERIFIED',
    notes: 'Quarantines 2,501 transition windows (13.3% of total data) to eliminate label contamination caused by vehicle secondary suspension ringing.'
  },
  {
    id: 'CR-04',
    title: 'Sliding Window Overlap Autocorrelation Artifact',
    statement: 'Splitting sliding windows randomly with 50% overlap artificially inflates Random Forest PR-AUC by +13.43% relative inflation over leak-free block splitting.',
    verified_value: '+13.43% (relative)',
    unit: '% relative inflation',
    source_file: 'reports/baseline_results.md, reports/evaluation_audit.md, reports/baseline_results.json',
    line_range: 'baseline_results.md L60-62; evaluation_audit.md L95-97',
    code_excerpt: 'Random_Window PR-AUC: 0.4730 vs Block_Grouped PR-AUC: 0.4170. Relative inflation: (0.4730 - 0.4170) / 0.4170 = +13.43%',
    verification_method: 'Calculated mathematical relative difference between verified PR-AUC scores in baseline_results.json.',
    status: 'VERIFIED',
    notes: 'Crucial distinction: this represents a +13.43% relative inflation (+0.0560 points of absolute PR-AUC), confirming that overlapping windows leak sensor autocorrelation.'
  },
  {
    id: 'CR-05',
    title: 'Camera Shutter Timestamp Disk I/O Drift',
    statement: 'Monochrome track inspection frames experience variable disk write latency displacing images up to ±15 meters (±2.0 s) from true GPS coordinates.',
    verified_value: '±15',
    unit: 'meters',
    source_file: 'reports/ground_truth_audit.md, reports/feasibility_assessment.md, README.md',
    line_range: 'ground_truth_audit.md L59-63; feasibility_assessment.md L42-45; README.md L18',
    code_excerpt: 'Under variable-FPS conditions this introduces a non-constant drift that can displace an image by up to ~15 m from its true location.',
    verification_method: 'Cross-checked with the author release notes quoted in ground_truth_audit.md and Acquisition.py threading logic.',
    status: 'VERIFIED',
    notes: 'Precludes raw microsecond video-vibration late fusion without spatial landmark realignment (e.g. surveyed orange marker cones).'
  },
  {
    id: 'CR-06',
    title: 'Unsprung Axle Vertical Kurtosis Peak',
    statement: 'Silicon Designs accelerometers on the front and back wheel axles (Channels 2 and 4) record impulse kurtosis between 8.4 and 14.2.',
    verified_value: 14.2,
    unit: 'kurtosis (dimensionless)',
    source_file: 'reports/data_quality_report.md, figures/03_acceleration_distributions.png',
    line_range: 'data_quality_report.md L46, L52',
    code_excerpt: 'Wheel Axle (Ch2, Ch4) | RMS: 0.28-0.42g | Kurtosis: 8.4 - 14.2 | Physical Role: Unsprung mass, direct shock impacts',
    verification_method: 'Verified against statistical tables in data_quality_report.md and kernel density distributions in Figure 03.',
    status: 'VERIFIED',
    notes: 'Contrast with bogie frame sensors (kurtosis 4.6-6.8) and car body sensors (kurtosis 3.1-3.8), demonstrating mechanical suspension filtering.'
  },
  {
    id: 'CR-07',
    title: 'Spatial Holdout: Uncalibrated Collapse vs Calibrated Recovery',
    statement: 'On unseen spatial sectors with GPS coordinates, static 0.50 thresholding collapses recall to 0.0000; calibrating decision threshold to 0.25 recovers recall to 0.5251.',
    verified_value: 'Threshold = 0.25 (Recall = 0.5251)',
    unit: 'probability threshold & recall',
    source_file: 'README.md, reports/feasibility_assessment.md, reports/baseline_results.json',
    line_range: 'README.md L110; feasibility_assessment.md L49; baseline_results.json L916-924',
    code_excerpt: '"calibrated_threshold": 0.25, "precision": 0.5046, "recall": 0.5251, "f1": 0.5146, "roc_auc": 0.9071',
    verification_method: 'Cross-checked README benchmark summary and exact entry in baseline_results.json for Random Forest on Spatial_Held_Out with GPS.',
    status: 'VERIFIED',
    notes: 'CRITICAL RESOLUTION: 0.25 is the decision probability threshold calibrated on the validation set, while 0.5251 is the resulting test recall. When uncalibrated at default 0.50, recall is 0.0000 due to coordinate memorization.'
  }
];

