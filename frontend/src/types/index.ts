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

