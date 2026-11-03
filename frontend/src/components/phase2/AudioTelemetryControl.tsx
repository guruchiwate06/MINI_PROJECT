import React from 'react';
import { Volume2, VolumeX, Radio, AlertTriangle } from 'lucide-react';
import { useSignalStore } from '../../store/signalStore';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

export const AudioTelemetryControl: React.FC = () => {
  const {
    audioEnabled,
    audioVolume,
    audioMuted,
    playbackState,
    setAudioEnabled,
    setAudioVolume,
    toggleAudioMute
  } = useSignalStore();

  return (
    <div className="rounded-xl border border-railway-border/80 bg-railway-surface/60 p-4 backdrop-blur shadow-lg space-y-3">
      {/* Title & Integrity Badge */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-railway-border/60 pb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            <Radio className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono font-bold text-xs uppercase tracking-wider text-white">
                Procedural Audio Synthesis Engine
              </span>
              <DataIntegrityBadge type="PROCEDURAL" size="sm" />
            </div>
            <p className="text-[10px] text-railway-muted font-sans">
              Real-time Web Audio API frequency-modulated telemetry sonification
            </p>
          </div>
        </div>

        {/* Audio Enable / Disable Master Toggle */}
        <button
          onClick={() => setAudioEnabled(!audioEnabled)}
          className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-mono font-bold transition-all border ${
            audioEnabled
              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50 shadow-sm shadow-emerald-500/20'
              : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
          }`}
          title="Opt-in Web Audio API synthesizer (no autoplay)"
        >
          <span className={`h-2 w-2 rounded-full ${audioEnabled ? 'bg-emerald-400 animate-pulse' : 'bg-gray-600'}`} />
          {audioEnabled ? 'SYNTHESIZER ACTIVE' : 'ENABLE PROCEDURAL AUDIO'}
        </button>
      </div>

      {/* Synthesis Controls & Node Readout */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-mono">
        {/* Engine Status */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/60 space-y-1">
          <span className="text-[10px] text-railway-muted uppercase tracking-wider block">Synthesizer State</span>
          <div className="flex items-center justify-between">
            <span
              className={`font-bold ${
                audioEnabled && playbackState === 'playing' && !audioMuted
                  ? 'text-emerald-400'
                  : audioEnabled && playbackState === 'paused'
                  ? 'text-hud-amber'
                  : 'text-railway-muted'
              }`}
            >
              {!audioEnabled
                ? 'STANDBY (MUTED)'
                : playbackState === 'playing' && !audioMuted
                ? 'GENERATING SOUNDSCAPE'
                : playbackState === 'paused'
                ? 'PAUSED (MUTED)'
                : 'IDLE (STOPPED)'}
            </span>
          </div>
        </div>

        {/* Dynamic Nodes */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/60 space-y-1">
          <span className="text-[10px] text-railway-muted uppercase tracking-wider block">Oscillator Nodes</span>
          <div className="text-[11px] text-white">
            <span className="text-hud-cyan">32 Hz</span> Triangle Sub-rumble • <span className="text-hud-amber">380 Hz</span> Friction
          </div>
        </div>

        {/* Impact Filter */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/60 space-y-1">
          <span className="text-[10px] text-railway-muted uppercase tracking-wider block">Transient Resonator</span>
          <div className="text-[11px] text-hud-emerald">
            <span>Joint Gap Ping (Ch 2 Axle &gt; 0.8g)</span>
          </div>
        </div>

        {/* Volume & Mute Controls */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/60 flex items-center justify-between gap-3">
          <button
            onClick={toggleAudioMute}
            disabled={!audioEnabled}
            className={`p-1.5 rounded border transition-colors ${
              audioMuted || !audioEnabled
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                : 'bg-railway-elevated border-railway-border text-hud-cyan hover:bg-railway-surface'
            }`}
            title={audioMuted ? 'Unmute procedural sound' : 'Mute procedural sound'}
          >
            {audioMuted || !audioEnabled ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </button>

          <div className="flex-1 space-y-0.5">
            <div className="flex justify-between text-[10px] text-railway-muted">
              <span>GAIN</span>
              <span>{Math.round(audioVolume * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={audioVolume}
              disabled={!audioEnabled}
              onChange={(e) => setAudioVolume(parseFloat(e.target.value))}
              className="w-full accent-hud-cyan h-1 bg-railway-border rounded cursor-pointer disabled:opacity-40"
            />
          </div>
        </div>
      </div>

      {/* Mandatory Scientific Provenance Disclaimer */}
      <div className="flex items-start gap-2 p-2 rounded bg-railway-darkest/90 border border-hud-amber/20 text-[11px] font-sans text-railway-muted">
        <AlertTriangle className="h-3.5 w-3.5 text-hud-amber shrink-0 mt-0.5" />
        <span>
          <strong className="text-hud-amber font-mono font-normal uppercase">[SCIENTIFIC INTEGRITY NOTICE]</strong>: This audio is{' '}
          <strong className="text-white">procedurally synthesized in real time via the Web Audio API</strong> based on accelerometer amplitudes
          and vehicle velocity. It is <strong className="text-rose-400">NEVER</strong> presented as recorded acoustic rail noise. Audio automatically
          silences upon pause or reset. Autoplay is disabled by default.
        </span>
      </div>
    </div>
  );
};

