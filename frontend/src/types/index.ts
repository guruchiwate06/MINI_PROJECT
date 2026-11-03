/**
 * Rail-VIVID TypeScript Core Data Schemas & Integrity Taxonomy
 */

export type DataIntegrityClassification =
  | 'MEASURED'       // Real 2 kHz sensor readings from Orbisonia testbed
  | 'GROUND_TRUTH'   // Table 2 coordinates and surveyed cone markers
  | 'ML_PREDICTION'  // Audited benchmark models trained via run_benchmarks.py
  | 'SIMULATED'      // Virtual train kinematics and trajectory playback
  | 'PROCEDURAL'     // Procedurally synthesized audio via Web Audio API
  | 'ILLUSTRATIVE';   // Visual HUD overlays, 3D buffer rings, and bounding guides

export interface ScientificClaim {
  id: string;                      // e.g. CR-01
  title: string;                   // Short descriptive title
  statement: string;               // Formal scientific statement
  verified_value: string | number; // Exact quantified value
  unit: string;                    // Physical or statistical unit
  source_file: string;             // Repository path
  line_range: string;              // e.g. "lines 17-20"
  code_excerpt: string;            // Direct quote or code snippet
  verification_method: string;     // How it was verified
  status: 'VERIFIED' | 'UNVERIFIED' | 'PARTIALLY_VERIFIED';
  notes: string;                   // Scientific and forensic context
}

export interface GroundTruthAnomaly {
  id: number;
  type: 'Rail Joint Gap' | 'Track Divergence (Switch)' | 'Track Convergence (Switch)';
  size_cm: number | null;
  lat: number;
  lon: number;
  dist_from_prev_m: number;
  cum_track_offset_m: number;
  chainage_m?: number;
  cone_color: 'Orange' | 'Black';
  point_cloud_available: boolean;
  scan_file?: string;
}

export interface ExperimentalRun {
  run_id: string;
  direction: 'AtoB' | 'BtoA';
  target_speed_kmh: number;
  run_index: number;
  csv_size_mb: number;
  row_count: number;
  duration_s: number;
  image_count: number;
  effective_fps: number;
  start_timestamp: string;
}

export interface BenchmarkRecord {
  model: 'Isolation Forest' | 'Random Forest' | 'Logistic Regression' | 'Signal Autoencoder' | 'Vibration 1D CNN' | 'ViT VAE (Published)';
  modalities: 'Vibration_Only' | 'Vibration_Plus_Env' | 'Vibration_Plus_GPS' | 'Vision Only';
  split_strategy:
  | 'Random_Window (Leaky Control)'
  | 'Block_Grouped_Window (Overlap-Free)'
  | 'Run_Held_Out'
  | 'Spatial_Held_Out'
  | 'Combined_Run_Spatial';
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  pr_auc: number | null;
  calibrated_threshold: number;
  threshold_source: string;
  test_set_peeking: boolean;
  confusion_matrix: [[number, number], [number, number]];
  limitations: string;
}

export interface TelemetryState {
  current_distance_m: number;
  speed_kmh: number;
  active_run_id: string;
  direction: 'AtoB' | 'BtoA';
  latitude: number;
  longitude: number;
  bogie_z_g: number;
  axle_z_g: number;
  body_z_g: number;
  nearest_anomaly_id: number;
  dist_to_anomaly_m: number;
  spatial_zone: 'ANOMALY' | 'BUFFER' | 'NORMAL';
}



// ==========================================
// Phase 2 Signal Intelligence Types
// ==========================================

export interface ChannelDefinition {
  id: number;
  name: string;
  label: string;
  location: string;
  color: string;
  role: string;
  filtering: string;
}

export interface SignalOverviewPoint {
  t: number;          // Time in seconds from start of run
  d: number;          // Chainage / distance along track in meters
  a: number;          // Anomaly classification flag: 0=normal, 1=anomaly (<=6m), -1=buffer zone
  ch: Array<[number, number]>; // 6 channels, each [min_g, max_g]
}

export interface SignalRunOverview {
  run_id: string;
  sampling_rate_hz: number;
  decimation_step: number;
  effective_rate_hz: number;
  total_points: number;
  duration_s: number;
  channels: string[];
  points: SignalOverviewPoint[];
}

export interface MicroWindowData {
  id: string;
  title: string;
  type: 'ANOMALOUS' | 'NORMAL';
  provenance: DataIntegrityClassification;
  site_id: number | null;
  chainage_m: number;
  time_start_s: number;
  time_end_s: number;
  sample_count: number;
  sampling_rate_hz: number;
  channels_data: number[][]; // [6 channels, 2000 samples each]
}

export interface MicroWindowsFile {
  run_id: string;
  windows: MicroWindowData[];
}

export interface RunCatalogItem {
  run_id: string;
  direction: 'AtoB' | 'BtoA';
  target_speed_kmh: number;
  run_number: number;
  total_samples: number;
  duration_s: number;
  sampling_rate_hz: number;
  track_length_m: number;
  anomalies_count: number;
  anomalies: Array<{
    site_id: number;
    type: string;
    size_cm: number | null;
    sample_idx: number;
    time_s: number;
    chainage_m: number;
    lat: number;
    lon: number;
  }>;
}

export interface SignalsIndexData {
  dataset_name: string;
  description: string;
  provenance: DataIntegrityClassification;
  channels: ChannelDefinition[];
  runs: RunCatalogItem[];
  ground_truth_sites: Array<{
    id: number;
    type: string;
    size_cm?: number;
    lat: number;
    lon: number;
    chainage_m: number;
    reference_distance_m: number;
  }>;
}

export interface FFTSpectrumData {
  frequencies: number[]; // In Hz (0 to 1000 Hz)
  magnitudes: number[];  // In dB or amplitude g
  dominant_freq: number; // Primary peak frequency in Hz
  top_peaks: Array<{ freq: number; mag: number }>;
  nyquist_hz: number;
}

// ==========================================
// Phase 3 Photorealistic 3D Environment Types
// ==========================================

export type CameraViewMode =
  | 'side-follow'
  | 'front-track'
  | 'rear-track'
  | 'rail-macro'
  | 'bogie-inspect'
  | 'free-orbit';

export interface Scene3DOverlays {
  showSensors: boolean;
  showAnomalies: boolean;
  showZones: boolean;
  showBogieWireframe: boolean;
  showCameraFrustum: boolean;
  lightingMode: 'night' | 'day';
}

// ==========================================
// Phase 4 AI Model Evaluation & Benchmarking Types
// ==========================================

export interface EvaluatedExperiment {
  id: string;
  model_name: string;
  method: string;
  architecture: string;
  fold: string;
  seed: number;
  split_strategy: string;
  split_fingerprint?: string;
  split_cohort?: string;
  evaluation_partition?: string;
  is_test_evaluated?: boolean;
  record_type?: string;
  checkpoint_path?: string | null;
  interrupted_epoch?: number;
  interrupted_batch?: string;
  failure_reason?: string;
  modality?: string;
  test_sites: number[] | string;
  val_sites: number[] | string;
  train_sites: number[] | string;
  dataset_version: string;
  observations: number;
  positives: number;
  tp: number | string;
  fp: number | string;
  tn: number | string;
  fn: number | string;
  confusion_matrix?: number[][];
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  specificity: number | null;
  f1: number | null;
  roc_auc: number | null;
  average_precision: number | null;
  validation_ap?: number | null;
  validation_epoch?: number;
  test_ap?: string;
  balanced_accuracy: number | null;
  mcc: number | null;
  calibrated_threshold: number | null;
  test_threshold_used: number | null;
  threshold_source: string;
  brier_score: number | null;
  ece: number | null;
  event_recall: number | null;
  false_alarms_per_km: number | null;
  localization_mae_m: number | null;
  checkpoint_size_kb: number | null;
  throughput_windows_sec: number | null;
  latency_ms: number | null;
  device: string;
  status: string;
  provenance: DataIntegrityClassification;
  training_notes?: string;
}

export interface SiteHoldoutMetric {
  site_id: number;
  fold: string;
  model: string;
  defect_type: string;
  cone_color: string;
  chainage_m: number;
  observations: number;
  positives: number;
  negatives: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  average_precision: number;
  calibrated_threshold: number;
  brier: number;
  ece: number;
}

export interface ThresholdStepMetrics {
  threshold: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  balanced_accuracy: number;
  mcc: number;
}

export interface ThresholdExplorerModel {
  model_id: string;
  model_name: string;
  val_calibrated_threshold: number;
  default_threshold: number;
  observations: number;
  positives: number;
  negatives: number;
  threshold_sweep: ThresholdStepMetrics[];
  roc_curve: Array<{ fpr: number; tpr: number; threshold: number }>;
  pr_curve: Array<{ recall: number; precision: number; threshold: number }>;
  predictions_sample: Array<{
    window_id: string;
    chainage_m: number;
    label: number;
    probability: number;
    site_id: number | null;
    direction: string;
    speed_kmh: number;
  }>;
}
