/**
 * SENSOR HARDWARE AND PHYSICAL PLATFORM SPECIFICATIONS
 * Cross-referenced with reports/dataset_inventory.md and reports/data_quality_report.md
 */

export interface HardwarePlatform {
  train: {
    model: string;
    operator: string;
    track_location: string;
    track_length_m: number;
    traction_type: string;
    total_runs: number;
    speed_range_kmh: string;
  };
  sensors: Array<{
    name: string;
    type: string;
    channels: string[];
    sampling_rate: string;
    location: string;
    specs: string;
  }>;
}

export const HARDWARE_SPECS: HardwarePlatform = {
  train: {
    model: 'Pop-Up Metro Class 230 (Vivarail D78 Hybrid Railcar)',
    operator: 'Railroad Development Corporation (RDC) / Pop-Up Metro',
    track_location: 'Orbisonia, Pennsylvania, USA (Point A to Point B)',
    track_length_m: 1400.0,
    traction_type: 'Diesel-Battery Hybrid Passenger Railcar',
    total_runs: 20,
    speed_range_kmh: '20.0 to 27.5 km/h'
  },
  sensors: [
    {
      name: 'Silicon Designs Model 2012 Accelerometer',
      type: 'Capacitive MEMS Vibration Sensor',
      channels: ['Channel 1 (Front Bogie)', 'Channel 3 (Back Bogie)'],
      sampling_rate: '2,000 Hz (2 kHz)',
      location: 'Primary Suspension / Bogie Frame',
      specs: '±50g full-scale, 500 Hz Butterworth low-pass filter, 16-bit digitizer'
    },
    {
      name: 'Silicon Designs Model 2012 Accelerometer',
      type: 'Capacitive MEMS Vibration Sensor',
      channels: ['Channel 2 (Front Axle)', 'Channel 4 (Back Axle)'],
      sampling_rate: '2,000 Hz (2 kHz)',
      location: 'Unsprung Wheel Axle Boxes',
      specs: 'Direct wheel-rail contact interface, peak kurtosis 8.4-14.2'
    },
    {
      name: 'Silicon Designs Model 2012 Accelerometer',
      type: 'Capacitive MEMS Vibration Sensor',
      channels: ['Channel 5 (Front Body)', 'Channel 6 (Mid Body)'],
      sampling_rate: '2,000 Hz (2 kHz)',
      location: 'Car Body Passenger Cabin Floor',
      specs: 'Secondary air suspension filtered, >90% high-frequency attenuation'
    },
    {
      name: 'See3CAM_50CUGM Industrial Camera',
      type: 'Monochrome Global Shutter Optical Sensor',
      channels: ['Downward Track Inspection Video'],
      sampling_rate: 'Effective 20-26 FPS (Target 100 FPS)',
      location: '1.52m above rail plane, angled 15° downward',
      specs: 'Sony IMX264LLR-C CMOS, 1920×1080 Full HD, ±15m disk write latency offset'
    },
    {
      name: 'Columbus P-7 Pro Multi-GNSS Receiver',
      type: 'Satellite Positioning Engine',
      channels: ['Latitude', 'Longitude'],
      sampling_rate: '1 kHz smoothed, upsampled to 2 kHz',
      location: 'Vehicle Roof Centerline',
      specs: '<0.8m cross-run track repeat accuracy'
    }
  ]
};

