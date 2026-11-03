import React, { useState } from 'react';
import { FileImage, ZoomIn } from 'lucide-react';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

interface FigureMeta {
  filename: string;
  title: string;
  caption: string;
  keyFinding: string;
}

const FIGURES: FigureMeta[] = [
  {
    filename: '01_gps_trajectories_and_anomalies.png',
    title: 'Figure 1: GNSS Trajectories & Ground-Truth Defect Mapping',
    caption: 'Superimposed GPS coordinates across sample runs (AtoB and BtoA) demonstrating <0.8 m lateral track deviation and exact locations of Anomalies 1 through 9.',
    keyFinding: 'Empirical cross-run spatial alignment is <0.8m, confirming perfect spatial repeatability across all 20 runs.'
  },
  {
    filename: '02_acceleration_timeseries_channels.png',
    title: 'Figure 2: Multichannel Vertical Acceleration Profiles',
    caption: 'Synchronized 2,000 Hz waveforms across Wheel Axle (Ch2/4), Bogie Frame (Ch1/3), and Passenger Car Body (Ch5/6).',
    keyFinding: 'Clear mechanical transmissibility hierarchy: axle shocks are sharp impulses, bogie filters wheel noise, and body attenuates >90% vibration.'
  },
  {
    filename: '03_acceleration_distributions.png',
    title: 'Figure 3: Acceleration Distributions & Kurtosis Spread',
    caption: 'Kernel density estimation (KDE) and probability density plots highlighting heavy-tailed kurtosis up to 14.2 on unsprung wheelset transducers.',
    keyFinding: 'Axles exhibit kurtosis 8.4-14.2 due to joint impacts; bogies exhibit 4.6-6.8, and cabin body exhibits 3.1-3.8.'
  },
  {
    filename: '04_psd_spectrogram.png',
    title: 'Figure 4: Welch Power Spectral Density & STFT Spectrogram',
    caption: 'Frequency-domain analysis across 0-500 Hz bandwidth. Mechanical joint impacts excite broadband spectral energy up to the 500 Hz anti-aliasing cutoff.',
    keyFinding: 'Nominal track energy drops steeply above 150 Hz, providing clear spectral contrast for anomaly detectors.'
  },
  {
    filename: '05_environmental_distributions.png',
    title: 'Figure 5: Ambient Temperature and Humidity Transits',
    caption: 'Hourly weather station records upsampled via cubic spline to 2,000 Hz across April 11 and April 18 experimental collection runs.',
    keyFinding: 'Environmental covariates show minor correlation with baseline drift but do not explain high-frequency impact dynamics.'
  },
  {
    filename: '06_cross_run_consistency_and_speed.png',
    title: 'Figure 6: Cross-Run Consistency and Speed Transferability',
    caption: 'Vibration energy signatures at the exact same physical coordinates under varying vehicle speeds (20.0 to 27.5 km/h).',
    keyFinding: 'Impact amplitude scales predictably with vehicle speed, enabling speed-invariant model transfer.'
  },
  {
    filename: '07_camera_frame_rate_and_sharpness.png',
    title: 'Figure 7: Camera Frame Rate Jitter & Laplacian Sharpness',
    caption: 'Frame interval distributions and Laplacian blur metric, exposing the disk I/O write bottleneck that caused up to ±15 m spatial drift.',
    keyFinding: 'Proves that variable FPS and disk-write latency displace frames by up to ±15 m (±2.0 s), requiring surveyed cone alignment.'
  }
];

export const PublicationEvidenceGallery: React.FC = () => {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const activeFig = FIGURES[selectedIdx];

  return (
    <div className="bg-railway-surface/60 border border-railway-border rounded-xl p-5 sm:p-6 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5 border-b border-railway-border/60 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-lg font-bold text-white font-mono tracking-wide flex items-center gap-2">
              <FileImage className="h-5 w-5 text-hud-amber" />
              PUBLICATION EVIDENCE GALLERY
            </h2>
            <DataIntegrityBadge type="MEASURED" size="sm" />
          </div>
          <p className="text-xs text-railway-muted mt-1">
            Publication-quality audit figures generated from raw 2 kHz CSVs and image archives via generate_eda_figures.py.
          </p>
        </div>

        <span className="text-xs font-mono px-2.5 py-1 rounded bg-railway-elevated border border-railway-border text-railway-muted">
          7 High-Resolution Audit Plates
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Figure selector list */}
        <div className="lg:col-span-4 space-y-2">
          {FIGURES.map((fig, idx) => (
            <button
              key={fig.filename}
              onClick={() => setSelectedIdx(idx)}
              className={`w-full text-left p-3 rounded-lg border transition-all text-xs font-mono ${
                selectedIdx === idx
                  ? 'bg-railway-elevated border-hud-amber/50 text-white shadow-sm'
                  : 'bg-railway-dark/50 border-railway-border/50 text-railway-muted hover:text-railway-text hover:bg-railway-dark/80'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`font-bold ${selectedIdx === idx ? 'text-hud-amber' : 'text-hud-cyan'}`}>
                  0{idx + 1}
                </span>
                <span className="text-[10px] text-railway-muted truncate max-w-[150px]">{fig.filename}</span>
              </div>
              <p className="font-sans font-medium text-white text-[11px] mt-1 line-clamp-1">{fig.title}</p>
            </button>
          ))}
        </div>

        {/* Selected Figure Viewer */}
        <div className="lg:col-span-8 bg-railway-darkest/90 rounded-lg border border-railway-border p-4 flex flex-col justify-between">
          <div className="relative rounded overflow-hidden bg-black/60 border border-railway-border/40 group">
            <img
              src={`/figures/${activeFig.filename}`}
              alt={activeFig.title}
              className="w-full h-auto max-h-[380px] object-contain mx-auto transition-transform duration-300 group-hover:scale-[1.01]"
              loading="lazy"
            />
            <a
              href={`/figures/${activeFig.filename}`}
              target="_blank"
              rel="noreferrer"
              className="absolute top-2 right-2 px-2 py-1 rounded bg-black/75 hover:bg-black text-white text-[11px] font-mono flex items-center gap-1 border border-white/20 opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <ZoomIn className="h-3 w-3" /> Full Plate
            </a>
          </div>

          <div className="mt-3 pt-3 border-t border-railway-border/60 font-sans">
            <h3 className="text-sm font-bold text-white font-mono">{activeFig.title}</h3>
            <p className="text-xs text-railway-muted mt-1 leading-relaxed">{activeFig.caption}</p>
            <div className="mt-2 p-2 rounded bg-hud-amber/10 border border-hud-amber/30 text-[11px] text-hud-amber font-mono flex items-center gap-2">
              <span className="font-bold uppercase tracking-wider text-[10px] bg-hud-amber/20 px-1.5 py-0.5 rounded">Finding:</span>
              <span>{activeFig.keyFinding}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
