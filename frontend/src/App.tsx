import React, { useState } from 'react';
import { TelemetryHeader } from './components/common/TelemetryHeader';
import { SignalIntelligenceView } from './components/phase2/SignalIntelligenceView';
import { ScientificClaimRegistryView } from './components/phase1/ScientificClaimRegistryView';
import { DatasetArchitectureView } from './components/phase1/DatasetArchitectureView';
import { PublicationEvidenceGallery } from './components/phase1/PublicationEvidenceGallery';
import {
  Activity,
  FileText,
  Database,
  Award,
  Sparkles
} from 'lucide-react';

type NavigationTab = 'signal-intelligence' | 'claim-registry' | 'dataset-architecture' | 'publication-evidence';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavigationTab>('signal-intelligence');

  return (
    <div className="min-h-screen bg-railway-darkest text-railway-text flex flex-col font-sans hud-grid">
      {/* Sticky Telemetry HUD Header */}
      <TelemetryHeader />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        
        {/* Navigation Tab Bar */}
        <nav className="flex items-center gap-1.5 p-1.5 rounded-xl bg-railway-dark/90 border border-railway-border/80 backdrop-blur font-mono text-xs overflow-x-auto shadow-lg">
          <button
            onClick={() => setActiveTab('signal-intelligence')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-bold transition-all whitespace-nowrap ${
              activeTab === 'signal-intelligence'
                ? 'bg-hud-cyan text-railway-darkest shadow-md shadow-hud-cyan/20'
                : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
            }`}
          >
            <Activity className="h-4 w-4" />
            <span>SIGNAL INTELLIGENCE</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded font-extrabold ${
                activeTab === 'signal-intelligence'
                  ? 'bg-railway-darkest text-hud-cyan'
                  : 'bg-hud-cyan/15 text-hud-cyan'
              }`}
            >
              PHASE 2
            </span>
          </button>

          <button
            onClick={() => setActiveTab('claim-registry')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-bold transition-all whitespace-nowrap ${
              activeTab === 'claim-registry'
                ? 'bg-hud-cyan text-railway-darkest shadow-md shadow-hud-cyan/20'
                : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
            }`}
          >
            <FileText className="h-4 w-4" />
            <span>SCIENTIFIC CLAIM REGISTRY</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded font-extrabold ${
                activeTab === 'claim-registry'
                  ? 'bg-railway-darkest text-emerald-400'
                  : 'bg-emerald-500/15 text-emerald-400'
              }`}
            >
              7/7 VERIFIED
            </span>
          </button>

          <button
            onClick={() => setActiveTab('dataset-architecture')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-bold transition-all whitespace-nowrap ${
              activeTab === 'dataset-architecture'
                ? 'bg-hud-cyan text-railway-darkest shadow-md shadow-hud-cyan/20'
                : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
            }`}
          >
            <Database className="h-4 w-4" />
            <span>TESTBED & CLASS 230</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded font-extrabold ${
                activeTab === 'dataset-architecture'
                  ? 'bg-railway-darkest text-hud-amber'
                  : 'bg-hud-amber/15 text-hud-amber'
              }`}
            >
              N=9 SITES
            </span>
          </button>

          <button
            onClick={() => setActiveTab('publication-evidence')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-bold transition-all whitespace-nowrap ${
              activeTab === 'publication-evidence'
                ? 'bg-hud-cyan text-railway-darkest shadow-md shadow-hud-cyan/20'
                : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
            }`}
          >
            <Award className="h-4 w-4" />
            <span>AUDIT & PUBLICATION</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded font-extrabold ${
                activeTab === 'publication-evidence'
                  ? 'bg-railway-darkest text-purple-400'
                  : 'bg-purple-500/15 text-purple-400'
              }`}
            >
              NATURE 2026
            </span>
          </button>
        </nav>

        {/* Dynamic Tab Views */}
        {activeTab === 'signal-intelligence' && <SignalIntelligenceView />}
        {activeTab === 'claim-registry' && <ScientificClaimRegistryView />}
        {activeTab === 'dataset-architecture' && <DatasetArchitectureView />}
        {activeTab === 'publication-evidence' && <PublicationEvidenceGallery />}

        {/* Phase 2 Delivery & Verification Footer Card */}
        <section className="bg-railway-surface/40 border border-railway-border/80 rounded-xl p-5 font-mono text-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-sm font-bold text-white flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-hud-cyan" />
              PHASE 2 SCIENTIFIC VERIFICATION & AUDIT CHECKLIST
            </span>
            <span className="text-hud-cyan text-[11px] font-bold">
              PHASE 2 COMPLETE • AWAITING USER APPROVAL (PHASE 3 LOCKED)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-1">
            <div className="p-3 bg-hud-cyan/10 rounded-lg border border-hud-cyan/40">
              <span className="text-hud-cyan font-bold block mb-1">✓ 6-CHANNEL INTELLIGENCE</span>
              <p className="text-[11px] text-railway-text font-sans">
                Ch 1–6 real-time synchronized playback, play/pause scrubber, speed control (0.25x–4x), position tracking.
              </p>
            </div>
            <div className="p-3 bg-hud-amber/10 rounded-lg border border-hud-amber/40">
              <span className="text-hud-amber font-bold block mb-1">✓ MULTIRESOLUTION OSCILLOSCOPE</span>
              <p className="text-[11px] text-railway-text font-sans">
                Decimated min-max transit overview + 2,000 Hz micro-window zoom across verified ground-truth defect sites.
              </p>
            </div>
            <div className="p-3 bg-hud-emerald/10 rounded-lg border border-hud-emerald/40">
              <span className="text-hud-emerald font-bold block mb-1">✓ FREQUENCY INTELLIGENCE</span>
              <p className="text-[11px] text-railway-text font-sans">
                Cooley-Tukey Radix-2 FFT (0–1,000 Hz Nyquist), Hanning/Hamming windowing, normal vs. anomalous comparative overlay.
              </p>
            </div>
            <div className="p-3 bg-purple-500/10 rounded-lg border border-purple-500/40">
              <span className="text-purple-300 font-bold block mb-1">✓ PROCEDURAL AUDIO & INTEGRITY</span>
              <p className="text-[11px] text-railway-text font-sans">
                Web Audio sonification labeled [PROCEDURAL], stops on pause/reset, zero fabricated sensor measurements.
              </p>
            </div>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="border-t border-railway-border/80 bg-railway-darkest/90 mt-12 py-6 text-xs text-railway-muted font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            <span className="text-white font-bold">RAIL-VIVID SIGNAL INTELLIGENCE PLATFORM</span>
            <p className="text-[11px] text-railway-muted font-sans mt-0.5">
              Ma, Flanigan, Bergés, & Brooks (2026). Nature Scientific Data 13:1343.
            </p>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <a
              href="https://doi.org/10.1038/s41597-026-07555-y"
              target="_blank"
              rel="noreferrer"
              className="text-hud-cyan hover:underline"
            >
              DOI: 10.1038/s41597-026-07555-y
            </a>
            <span>•</span>
            <a
              href="https://huggingface.co/datasets/atharva1305/Rail-VIVID"
              target="_blank"
              rel="noreferrer"
              className="text-hud-amber hover:underline"
            >
              Hugging Face Dataset
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
