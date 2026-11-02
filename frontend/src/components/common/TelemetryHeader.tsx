import React from 'react';
import { ShieldCheck, Activity, Train, Database, Terminal } from 'lucide-react';
import { DataIntegrityBadge } from './DataIntegrityBadge';

export const TelemetryHeader: React.FC = () => {
  return (
    <header className="border-b border-railway-border/80 bg-railway-dark/95 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-4">
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
              <span className="text-xs px-1.5 py-0.5 rounded bg-railway-elevated text-hud-cyan font-mono border border-hud-cyan/30">
                PHASE 1 FOUNDATION
              </span>
              <DataIntegrityBadge type="MEASURED" size="sm" />
            </div>
            <p className="text-[11px] text-railway-muted tracking-wide font-sans">
              Orbisonia Testbed • Class 230 Hybrid Railcar • 2,000 Hz Multi-Sensor Rig
            </p>
          </div>
        </div>

        {/* Telemetry quick stats */}
        <div className="flex items-center gap-5 text-xs font-mono">
          <div className="hidden md:flex flex-col items-end">
            <span className="text-railway-muted text-[10px] uppercase tracking-wider flex items-center gap-1">
              <Activity className="h-3 w-3 text-hud-cyan" /> DAQ Sample Rate
            </span>
            <span className="text-hud-cyan font-bold">2,000 Hz Continuous</span>
          </div>

          <div className="hidden lg:flex flex-col items-end">
            <span className="text-railway-muted text-[10px] uppercase tracking-wider flex items-center gap-1">
              <Database className="h-3 w-3 text-hud-amber" /> Physical Anomalies
            </span>
            <span className="text-hud-amber font-bold">N = 9 (7 Joints, 2 Turnouts)</span>
          </div>

          <div className="flex flex-col items-end">
            <span className="text-railway-muted text-[10px] uppercase tracking-wider flex items-center gap-1">
              <ShieldCheck className="h-3 w-3 text-hud-emerald" /> Audit Protocol
            </span>
            <span className="text-hud-emerald font-bold">Leak-Free Calibrated</span>
          </div>

          <a
            href="https://doi.org/10.1038/s41597-026-07555-y"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-railway-border bg-railway-elevated hover:bg-railway-border text-railway-text hover:text-white transition-all text-xs"
          >
            <Terminal className="h-3.5 w-3.5 text-hud-cyan" />
            <span>Nature DOI: 13:1343</span>
          </a>
        </div>
      </div>
    </header>
  );
};

