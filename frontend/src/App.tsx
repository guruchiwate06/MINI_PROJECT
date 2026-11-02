import React from 'react';
import { TelemetryHeader } from './components/common/TelemetryHeader';
import { DataIntegrityBadge } from './components/common/DataIntegrityBadge';
import { ScientificClaimRegistryView } from './components/phase1/ScientificClaimRegistryView';
import { DatasetArchitectureView } from './components/phase1/DatasetArchitectureView';
import { PublicationEvidenceGallery } from './components/phase1/PublicationEvidenceGallery';
import { ShieldCheck, Sparkles } from 'lucide-react';

export const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-railway-darkest text-railway-text flex flex-col font-sans hud-grid">
      {/* Sticky Telemetry HUD Header */}
      <TelemetryHeader />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 space-y-8">
        
        {/* Hero Documentary Foundation Card */}
        <section className="relative overflow-hidden rounded-2xl border border-railway-border/80 bg-gradient-to-b from-railway-surface/90 via-railway-dark/95 to-railway-darkest p-6 sm:p-10 shadow-2xl backdrop-blur">
          {/* Subtle ambient glow elements */}
          <div className="absolute top-0 right-0 w-96 h-96 bg-hud-cyan/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-hud-amber/5 rounded-full blur-3xl pointer-events-none" />

          <div className="relative z-10 max-w-4xl space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2.5 py-1 rounded-md text-[11px] font-mono font-bold bg-hud-cyan/15 text-hud-cyan border border-hud-cyan/40 tracking-wider uppercase">
                Phase 1 • Scientific Scaffolding & Verification
              </span>
              <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
              <DataIntegrityBadge type="MEASURED" size="sm" />
            </div>

            <h1 className="text-2xl sm:text-4xl lg:text-5xl font-black text-white tracking-tight font-mono leading-tight">
              RAIL-VIVID
              <span className="block text-lg sm:text-2xl text-hud-cyan font-normal font-sans mt-1">
                Cinematic Multimodal Railway Infrastructure Monitoring Platform
              </span>
            </h1>

            <p className="text-sm sm:text-base text-railway-muted leading-relaxed font-sans">
              An evidence-based structural health monitoring audit and time-series benchmark of the 
              <strong> Pop-Up Metro Class 230 passenger railcar</strong> along the 1.4-kilometer Orbisonia testbed. 
              Investigating 2,000 Hz vibration dynamics, spatial generalization across held-out track sectors, and forensic 
              remediation of data leakage in railway AI.
            </p>

            {/* Quick stats pills */}
            <div className="pt-2 grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
              <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
                <span className="text-railway-muted text-[10px] uppercase block tracking-wider">Digitization</span>
                <span className="text-hud-cyan font-bold text-sm">2,000 Hz Continuous</span>
                <span className="text-[10px] text-railway-muted block">6 Accelerometers</span>
              </div>
              <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
                <span className="text-railway-muted text-[10px] uppercase block tracking-wider">Defect Sample Size</span>
                <span className="text-hud-amber font-bold text-sm">N = 9 Physical Sites</span>
                <span className="text-[10px] text-railway-muted block">7 Joints, 2 Switches</span>
              </div>
              <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
                <span className="text-railway-muted text-[10px] uppercase block tracking-wider">Experimental Runs</span>
                <span className="text-white font-bold text-sm">20 Full Transits</span>
                <span className="text-[10px] text-railway-muted block">8 Uphill / 12 Downhill</span>
              </div>
              <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/80">
                <span className="text-railway-muted text-[10px] uppercase block tracking-wider">Claim Registry</span>
                <span className="text-hud-emerald font-bold text-sm">7 / 7 Claims Verified</span>
                <span className="text-[10px] text-hud-emerald block">0 Unverified</span>
              </div>
            </div>
          </div>
        </section>

        {/* Data Integrity Taxonomy Banner */}
        <section className="bg-railway-surface/40 border border-railway-border/70 rounded-xl p-4 sm:p-5 backdrop-blur">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <span className="text-xs font-mono font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                <ShieldCheck className="h-4 w-4 text-hud-cyan" />
                Data Integrity Taxonomy & Separation
              </span>
              <p className="text-[11px] text-railway-muted mt-0.5 font-sans">
                Every metric, signal, and visual asset is strictly classified to maintain complete scientific honesty.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <DataIntegrityBadge type="MEASURED" size="sm" />
              <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
              <DataIntegrityBadge type="ML_PREDICTION" size="sm" />
              <DataIntegrityBadge type="SIMULATED" size="sm" />
              <DataIntegrityBadge type="PROCEDURAL" size="sm" />
              <DataIntegrityBadge type="ILLUSTRATIVE" size="sm" />
            </div>
          </div>
        </section>

        {/* Section 1: Scientific Claim Registry */}
        <ScientificClaimRegistryView />

        {/* Section 2: Physical Testbed & Dataset Architecture */}
        <DatasetArchitectureView />

        {/* Section 3: Publication Evidence Gallery */}
        <PublicationEvidenceGallery />

        {/* Next Phases Roadmap Preview */}
        <section className="bg-railway-surface/50 border border-railway-border/80 rounded-xl p-5 sm:p-6 font-mono text-xs space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-bold text-white flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-hud-cyan" />
              PRODUCTION DEVELOPMENT ROADMAP
            </span>
            <span className="text-hud-cyan text-[11px]">Phase 1 Complete • Awaiting User Review</span>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
            <div className="p-3 bg-hud-cyan/10 rounded-lg border border-hud-cyan/40">
              <span className="text-hud-cyan font-bold block mb-1">✓ PHASE 1 (CURRENT)</span>
              <p className="text-[11px] text-railway-text font-sans">
                Scaffolding, TypeScript schemas, Claim Registry (CR-01–CR-07), verified artifact ingestion, and visual foundation.
              </p>
            </div>
            <div className="p-3 bg-railway-dark/60 rounded-lg border border-railway-border/60 opacity-80">
              <span className="text-hud-amber font-bold block mb-1">⏳ PHASE 2 (NEXT)</span>
              <p className="text-[11px] text-railway-muted font-sans">
                Web Audio procedural train soundscape, Zustand simulation store, and 2 kHz multiresolution ECharts oscilloscope.
              </p>
            </div>
            <div className="p-3 bg-railway-dark/60 rounded-lg border border-railway-border/60 opacity-60">
              <span className="text-railway-muted font-bold block mb-1">⏳ PHASE 3 & 4</span>
              <p className="text-[11px] text-railway-muted font-sans">
                Photorealistic Three.js Class 230 3D train, procedural tracks, and full 8-act cinematic documentary story.
              </p>
            </div>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="border-t border-railway-border/80 bg-railway-darkest/90 mt-12 py-6 text-xs text-railway-muted font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            <span className="text-white font-bold">RAIL-VIVID BENCHMARK AUDIT</span>
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
