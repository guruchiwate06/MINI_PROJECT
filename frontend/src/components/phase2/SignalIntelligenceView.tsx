import React, { useEffect } from 'react';
import { useSignalStore } from '../../store/signalStore';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import { SixChannelStreamView } from './SixChannelStreamView';
import { SignalOscilloscope } from './SignalOscilloscope';
import { FrequencyAnalysisView } from './FrequencyAnalysisView';
import { AudioTelemetryControl } from './AudioTelemetryControl';

export const SignalIntelligenceView: React.FC = () => {
  const { initialize, isLoading, error } = useSignalStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-16 rounded-xl border border-railway-border/80 bg-railway-surface/40 backdrop-blur font-mono text-sm space-y-3">
        <div className="h-8 w-8 rounded-full border-2 border-hud-cyan border-t-transparent animate-spin" />
        <span className="text-hud-cyan tracking-wider font-bold">
          LOADING REAL 2,000 Hz RAIL-VIVID SIGNAL INTELLIGENCE TELEMETRY...
        </span>
        <span className="text-xs text-railway-muted">
          Decimating 593,028 measured points & loading calibrated micro-window transients
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 rounded-xl border border-rose-500/40 bg-rose-500/10 font-mono text-xs text-rose-300 space-y-2">
        <span className="font-bold text-sm block">Failed to load signal data:</span>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Cinematic Digital Twin Banner */}
      <section className="relative overflow-hidden rounded-2xl border border-railway-border/80 bg-gradient-to-b from-railway-surface/90 via-railway-dark/95 to-railway-darkest p-6 sm:p-8 shadow-2xl backdrop-blur">
        <div className="absolute top-0 right-0 w-96 h-96 bg-hud-cyan/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-hud-amber/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-4xl space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2.5 py-1 rounded-md text-[11px] font-mono font-bold bg-hud-cyan/15 text-hud-cyan border border-hud-cyan/40 tracking-wider uppercase">
              Phase 2 • Signal Intelligence & Forensic Dynamics
            </span>
            <DataIntegrityBadge type="MEASURED" size="sm" />
            <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
            <DataIntegrityBadge type="PROCEDURAL" size="sm" />
          </div>

          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-black text-white tracking-tight font-mono">
            RAILWAY SIGNAL INTELLIGENCE TWIN
          </h2>

          <p className="text-sm sm:text-base text-railway-muted leading-relaxed font-sans">
            Interactive multi-channel vibration investigation workstation. Digitize, scrub, zoom, and forensically analyze 
            <strong> 2,000 Hz Class 230 passenger railcar accelerometer recordings</strong> along the 1,400-meter Orbisonia testbed. 
            All waveforms, time indices, and GPS coordinates are derived from verified empirical data.
          </p>

          {/* Core Technical Capabilities Pill Grid */}
          <div className="pt-2 grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
            <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
              <span className="text-railway-muted text-[10px] uppercase block tracking-wider">DAQ Sampling Rate</span>
              <span className="text-hud-cyan font-bold text-sm">2,000 Hz</span>
              <span className="text-[10px] text-railway-muted block">0.5 ms Resolution</span>
            </div>
            <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
              <span className="text-railway-muted text-[10px] uppercase block tracking-wider">Synchronized Channels</span>
              <span className="text-hud-amber font-bold text-sm">6 Accelerometers</span>
              <span className="text-[10px] text-railway-muted block">Bogie, Axle & Cabin</span>
            </div>
            <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
              <span className="text-railway-muted text-[10px] uppercase block tracking-wider">Spectral Bandwidth</span>
              <span className="text-white font-bold text-sm">0 – 1,000 Hz</span>
              <span className="text-[10px] text-railway-muted block">Radix-2 Cooley-Tukey</span>
            </div>
            <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
              <span className="text-railway-muted text-[10px] uppercase block tracking-wider">Scientific Honesty</span>
              <span className="text-hud-emerald font-bold text-sm">Zero Fabrication</span>
              <span className="text-[10px] text-hud-emerald block">Verified Provenance</span>
            </div>
          </div>
        </div>
      </section>

      {/* Module 1: Synchronized 6-Channel Stream & Real-time Transport */}
      <SixChannelStreamView />

      {/* Module 2: Interactive Time-Domain Oscilloscope */}
      <SignalOscilloscope />

      {/* Module 3: Frequency-Domain Intelligence (FFT) */}
      <FrequencyAnalysisView />

      {/* Module 4: Procedural Audio Telemetry Control */}
      <AudioTelemetryControl />
    </div>
  );
};
