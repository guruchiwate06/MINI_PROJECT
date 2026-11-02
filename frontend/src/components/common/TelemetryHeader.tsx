import React from 'react';
import { ShieldCheck, Activity, Train, Terminal, Radio } from 'lucide-react';
import { DataIntegrityBadge } from './DataIntegrityBadge';
import { useSignalStore } from '../../store/signalStore';

export const TelemetryHeader: React.FC = () => {
  const { selectedRunId, currentTime, currentDistance, currentSpeed, spatialZone, playbackState } = useSignalStore();

  return (
    <header className="border-b border-railway-border/80 bg-railway-dark/95 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex flex-wrap items-center justify-between gap-4">
        {/* Branding & Status */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-hud-cyan/10 border border-hud-cyan/40 flex items-center justify-center text-hud-cyan glow-cyan">
            <Train className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold tracking-wider text-base uppercase text-white font-mono">
                RAIL-VIVID
              </span>
              <span className="text-[11px] px-2 py-0.5 rounded bg-hud-cyan/15 text-hud-cyan font-mono border border-hud-cyan/30 font-bold">
                PHASE 2 • SIGNAL INTELLIGENCE
              </span>
              <DataIntegrityBadge type="MEASURED" size="sm" />
            </div>
            <p className="text-[11px] text-railway-muted tracking-wide font-sans">
              Orbisonia Testbed • Class 230 Hybrid Railcar • 2,000 Hz Multi-Sensor Rig
            </p>
          </div>
        </div>

        {/* Live Synchronized Telemetry Quick Stats */}
        <div className="flex items-center gap-4 text-xs font-mono">
          {/* Active Run & Telemetry */}
          <div className="hidden sm:flex flex-col items-end">
            <span className="text-railway-muted text-[10px] uppercase tracking-wider flex items-center gap-1">
              <Radio className={`h-3 w-3 ${playbackState === 'playing' ? 'text-emerald-400 animate-pulse' : 'text-hud-cyan'}`} />
              Run: {selectedRunId}
            </span>
            <span className="text-white font-bold">
              t={currentTime.toFixed(1)}s • d={currentDistance.toFixed(1)}m
            </span>
          </div>

          {/* Speed & Spatial Status */}
          <div className="hidden md:flex flex-col items-end">
            <span className="text-railway-muted text-[10px] uppercase tracking-wider flex items-center gap-1">
              <Activity className="h-3 w-3 text-hud-amber" /> Zone Status
            </span>
            <span
              className={`font-bold ${
                spatialZone === 'ANOMALY'
                  ? 'text-rose-400 animate-pulse'
                  : spatialZone === 'BUFFER'
                  ? 'text-amber-400'
                  : 'text-hud-cyan'
              }`}
            >
              {spatialZone} ({currentSpeed} km/h)
            </span>
          </div>

          {/* Audit Protocol */}
          <div className="hidden lg:flex flex-col items-end">
            <span className="text-railway-muted text-[10px] uppercase tracking-wider flex items-center gap-1">
              <ShieldCheck className="h-3 w-3 text-hud-emerald" /> DAQ Protocol
            </span>
            <span className="text-hud-emerald font-bold">2 kHz Radix-2 FFT</span>
          </div>

          <a
            href="https://doi.org/10.1038/s41597-026-07555-y"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-railway-border bg-railway-elevated hover:bg-railway-border text-railway-text hover:text-white transition-all text-xs"
          >
            <Terminal className="h-3.5 w-3.5 text-hud-cyan" />
            <span className="hidden sm:inline">Nature DOI:</span>
            <span>13:1343</span>
          </a>
        </div>
      </div>
    </header>
  );
};
