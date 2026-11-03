import { GroundTruthAnomaly } from '../types';

/**
 * 9 GROUND-TRUTH ANOMALIES
 * Sourced directly from Table 2 of Ma et al. (Nature Scientific Data, 2026)
 * and verified in reports/ground_truth_audit.md.
 */
export const GROUND_TRUTH_ANOMALIES: GroundTruthAnomaly[] = [
  {
    id: 1,
    type: 'Rail Joint Gap',
    size_cm: 2.0,
    lat: 40.23553,
    lon: -77.88898,
    dist_from_prev_m: 0.0,
    cum_track_offset_m: 0.0,
    chainage_m: 177.5,
    cone_color: 'Orange',
    point_cloud_available: true,
    scan_file: 'Anomaly1/FARO_LS03353_017.fls'
  },
  {
    id: 2,
    type: 'Rail Joint Gap',
    size_cm: 2.0,
    lat: 40.23512,
    lon: -77.88770,
    dist_from_prev_m: 117.8,
    cum_track_offset_m: 117.8,
    chainage_m: 295.3,
    cone_color: 'Orange',
    point_cloud_available: true,
    scan_file: 'Anomaly2/FARO_LS03353_018.fls'
  },
  {
    id: 3,
    type: 'Rail Joint Gap',
    size_cm: 2.4,
    lat: 40.23493,
    lon: -77.88711,
    dist_from_prev_m: 54.4,
    cum_track_offset_m: 172.2,
    chainage_m: 349.5,
    cone_color: 'Orange',
    point_cloud_available: false
  },
  {
    id: 4,
    type: 'Track Divergence (Switch)',
    size_cm: null,
    lat: 40.23461,
    lon: -77.88612,
    dist_from_prev_m: 91.3,
    cum_track_offset_m: 263.4,
    chainage_m: 441.1,
    cone_color: 'Black',
    point_cloud_available: true,
    scan_file: 'Anomaly4/FARO_LS03353_019.fls'
  },
  {
    id: 5,
    type: 'Track Convergence (Switch)',
    size_cm: null,
    lat: 40.23425,
    lon: -77.88485,
    dist_from_prev_m: 115.0,
    cum_track_offset_m: 378.4,
    chainage_m: 556.2,
    cone_color: 'Black',
    point_cloud_available: true,
    scan_file: 'Anomaly5/FARO_LS03353_020.fls'
  },
  {
    id: 6,
    type: 'Rail Joint Gap',
    size_cm: 2.0,
    lat: 40.23388,
    lon: -77.88320,
    dist_from_prev_m: 146.0,
    cum_track_offset_m: 523.8,
    chainage_m: 702.3,
    cone_color: 'Orange',
    point_cloud_available: false
  },
  {
    id: 7,
    type: 'Rail Joint Gap',
    size_cm: 2.5,
    lat: 40.23242,
    lon: -77.87795,
    dist_from_prev_m: 474.3,
    cum_track_offset_m: 998.1,
    chainage_m: 1177.7,
    cone_color: 'Orange',
    point_cloud_available: false
  },
  {
    id: 8,
    type: 'Rail Joint Gap',
    size_cm: 2.3,
    lat: 40.23205,
    lon: -77.87651,
    dist_from_prev_m: 129.0,
    cum_track_offset_m: 1127.1,
    chainage_m: 1307.4,
    cone_color: 'Orange',
    point_cloud_available: false
  },
  {
    id: 9,
    type: 'Rail Joint Gap',
    size_cm: 3.0,
    lat: 40.23194,
    lon: -77.87508,
    dist_from_prev_m: 122.0,
    cum_track_offset_m: 1245.6,
    chainage_m: 1429.5,
    cone_color: 'Orange',
    point_cloud_available: false
  }
];

