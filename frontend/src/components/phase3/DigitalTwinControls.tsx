import React from 'react';
import {
  Camera,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  MapPin,
  Sun,
  Moon,
  AlertTriangle
} from 'lucide-react';
import { useSignalStore } from '../../store/signalStore';
import { CameraViewMode, Scene3DOverlays } from '../../types';
import { GROUND_TRUTH_ANOMALIES } from '../../data/groundTruthAnomalies';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

interface DigitalTwinControlsProps {
  viewMode: CameraViewMode;
  setViewMode: (mode: CameraViewMode) => void;
  overlays: Scene3DOverlays;
  setOverlays: React.Dispatch<React.SetStateAction<Scene3DOverlays>>;
  onJumpToAnomaly: (siteId: number, chainage: number) => void;
}

export const DigitalTwinControls: React.FC<DigitalTwinControlsProps> = ({
  viewMode,
  setViewMode,
  overlays,
  setOverlays,
  onJumpToAnomaly
}) => {
  const {
    playbackState,
    currentTime,
    currentDistance,
    currentSpeed,
    spatialZone,
    nearestAnomaly,
    playbackSpeed,
    overview,
    audioEnabled,
    audioMuted,
    play,
    pause,
    reset,
    seekTime,
    setPlaybackSpeed,
    setAudioEnabled
  } = useSignalStore();

  const maxDuration = overview?.duration_s || 300;

  const cameraModes: Array<{ id: CameraViewMode; label: string; icon: string }> = [
    { id: 'side-follow', label: 'SIDE FOLLOW', icon: '🎥' },
    { id: 'front-track', label: 'FRONT CAB', icon: '🚆' },
    { id: 'rear-track', label: 'REAR CHASE', icon: '🔭' },
    { id: 'rail-macro', label: 'RAIL MACRO', icon: '⚡' },
    { id: 'bogie-inspect', label: 'BOGIE INSPECT', icon: '🔍' },
    { id: 'free-orbit', label: 'FREE ORBIT', icon: '🌐' }
  ];

  return (
    <div className="space-y-3 font-mono text-xs select-none">
      {/* ========================================================================= */}
      {/* 1. TOP TELEMETRY HEADS-UP DISPLAY (HUD) BAR                               */}
      {/* ========================================================================= */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-railway-dark/95 border border-railway-border/90 backdrop-blur shadow-2xl">
        {/* Physical Chainage & Speed Readout */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-hud-cyan animate-pulse" />
            <span className="font-extrabold text-sm uppercase text-white tracking-wider">
              CLASS 230 DIGITAL TWIN
            </span>
            <DataIntegrityBadge type="SIMULATED" size="sm" />
          </div>

          <div className="hidden sm:flex items-center gap-3 border-l border-railway-border/80 pl-3">
            <div>
              <span className="text-[10px] text-railway-muted block uppercase">Chainage</span>
              <span className="text-hud-cyan font-bold text-sm">
                {currentDistance.toFixed(1)} <span className="text-[10px] text-railway-muted">/ 1,400 m</span>
              </span>
            </div>

            <div>
              <span className="text-[10px] text-railway-muted block uppercase">Speed</span>
              <span className="text-white font-bold text-sm">
                {currentSpeed.toFixed(1)} <span className="text-[10px] text-railway-muted">km/h</span>
              </span>
            </div>

            <div>
              <span className="text-[10px] text-railway-muted block uppercase">Playback Time</span>
              <span className="text-white font-bold text-sm">
                t={currentTime.toFixed(1)}s
              </span>
            </div>
          </div>
        </div>

        {/* Spatial Zone Status & Anomaly Warning */}
        <div className="flex items-center gap-2">
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border font-bold text-xs ${
              spatialZone === 'ANOMALY'
                ? 'bg-rose-500/20 text-rose-300 border-rose-500/50 shadow-md shadow-rose-500/20 animate-pulse'
                : spatialZone === 'BUFFER'
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
            }`}
          >
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
            <span>ZONE: {spatialZone}</span>
          </div>

          {nearestAnomaly && (
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-railway-surface/70 border border-railway-border text-[11px] text-railway-muted">
              <span className="text-white font-bold">Site {nearestAnomaly.site_id}</span>: {nearestAnomaly.type}{' '}
              <span className="text-hud-cyan font-bold">({nearestAnomaly.distance_m.toFixed(1)}m away)</span>
            </div>
          )}
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 2. CAMERA ANGLE SELECTOR BAR                                              */}
      {/* ========================================================================= */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2 rounded-xl bg-railway-dark/90 border border-railway-border/80 backdrop-blur">
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-railway-muted px-2 uppercase flex items-center gap-1">
            <Camera className="h-3.5 w-3.5 text-hud-cyan" /> Camera:
          </span>
          <div className="flex flex-wrap gap-1">
            {cameraModes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setViewMode(mode.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-bold text-xs transition-all ${
                  viewMode === mode.id
                    ? 'bg-hud-cyan text-railway-darkest shadow-md shadow-hud-cyan/20'
                    : 'bg-railway-elevated text-railway-muted border border-railway-border hover:text-white hover:bg-railway-surface'
                }`}
              >
                <span>{mode.icon}</span>
                <span>{mode.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Scientific Overlay Quick Toggles */}
        <div className="flex items-center gap-1.5">
          {/* Sensor Nodes Overlay Toggle */}
          <button
            onClick={() => setOverlays((prev) => ({ ...prev, showSensors: !prev.showSensors }))}
            className={`px-2 py-1 rounded border text-[11px] font-bold transition-all ${
              overlays.showSensors
                ? 'bg-hud-cyan/20 text-hud-cyan border-hud-cyan/50'
                : 'bg-railway-elevated text-railway-muted border-railway-border'
            }`}
            title="Toggle Ch 1-6 physical sensor mounts"
          >
            SENSORS (CH 1-6)
          </button>

          {/* Anomaly Beacons Overlay Toggle */}
          <button
            onClick={() => setOverlays((prev) => ({ ...prev, showAnomalies: !prev.showAnomalies }))}
            className={`px-2 py-1 rounded border text-[11px] font-bold transition-all ${
              overlays.showAnomalies
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50'
                : 'bg-railway-elevated text-railway-muted border-railway-border'
            }`}
            title="Toggle ground-truth surveyed anomaly beacons and cones"
          >
            DEFECT SITES
          </button>

          {/* Spatial Zones Overlay Toggle */}
          <button
            onClick={() => setOverlays((prev) => ({ ...prev, showZones: !prev.showZones }))}
            className={`px-2 py-1 rounded border text-[11px] font-bold transition-all ${
              overlays.showZones
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                : 'bg-railway-elevated text-railway-muted border-railway-border'
            }`}
            title="Toggle 6m Definite Anomaly and 18m Buffer zones"
          >
            ZONES (6m/18m)
          </button>

          {/* Day / Night Environment Toggle */}
          <button
            onClick={() =>
              setOverlays((prev) => ({
                ...prev,
                lightingMode: prev.lightingMode === 'night' ? 'day' : 'night'
              }))
            }
            className="p-1 rounded bg-railway-elevated border border-railway-border text-railway-muted hover:text-white"
            title="Toggle between cinematic night lighting and high-contrast day lighting"
          >
            {overlays.lightingMode === 'night' ? <Moon className="h-4 w-4 text-hud-cyan" /> : <Sun className="h-4 w-4 text-hud-amber" />}
          </button>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 3. SYNCHRONIZED PLAYBACK TIMELINE & CONTROLS                              */}
      {/* ========================================================================= */}
      <div className="p-3 rounded-xl bg-railway-dark/95 border border-railway-border/80 backdrop-blur space-y-2">
        {/* Scrubber Bar with Anomaly Markers */}
        <div className="space-y-1">
          <div className="relative w-full h-3 bg-railway-darkest rounded-full border border-railway-border flex items-center overflow-hidden">
            {/* Surveyed Anomaly Tick Marks along timeline */}
            {GROUND_TRUTH_ANOMALIES.map((site) => {
              // Estimate timeline position from chainage proportion
              const pct = (site.cum_track_offset_m / 1400.0) * 100;
              return (
                <div
                  key={`scrub-site-${site.id}`}
                  style={{ left: `${pct}%` }}
                  className="absolute top-0 bottom-0 w-1 bg-hud-amber z-10 hover:w-1.5 cursor-pointer"
                  title={`Site ${site.id}: ${site.type} (${site.cum_track_offset_m.toFixed(1)}m)`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onJumpToAnomaly(site.id, site.cum_track_offset_m);
                  }}
                />
              );
            })}

            {/* Active Playback Progress Bar */}
            <div
              className="h-full bg-gradient-to-r from-hud-cyan/40 to-hud-cyan transition-all"
              style={{ width: `${(currentTime / maxDuration) * 100}%` }}
            />
          </div>

          <input
            type="range"
            min={0}
            max={maxDuration}
            step={0.1}
            value={currentTime}
            onChange={(e) => seekTime(parseFloat(e.target.value))}
            className="w-full accent-hud-cyan h-1 bg-transparent cursor-pointer"
          />
        </div>

        {/* Playback Controls & Speed Selector */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <div className="flex items-center gap-2">
            {/* Play / Pause Toggle */}
            <button
              onClick={() => (playbackState === 'playing' ? pause() : play())}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-bold text-xs transition-all ${
                playbackState === 'playing'
                  ? 'bg-amber-500 text-railway-darkest shadow-md shadow-amber-500/20'
                  : 'bg-hud-cyan text-railway-darkest shadow-md shadow-hud-cyan/20'
              }`}
            >
              {playbackState === 'playing' ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              <span>{playbackState === 'playing' ? 'PAUSE PLAYBACK' : 'PLAY 3D TRACK'}</span>
            </button>

            {/* Reset to Start */}
            <button
              onClick={reset}
              className="p-1.5 rounded-lg bg-railway-elevated border border-railway-border text-railway-muted hover:text-white"
              title="Reset to 0m (Point A)"
            >
              <RotateCcw className="h-4 w-4" />
            </button>

            {/* Playback Speed Multipliers */}
            <div className="flex items-center gap-1 bg-railway-darkest border border-railway-border rounded-lg p-0.5">
              {[0.5, 1.0, 2.0, 4.0].map((spd) => (
                <button
                  key={`spd-${spd}`}
                  onClick={() => setPlaybackSpeed(spd)}
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    playbackSpeed === spd
                      ? 'bg-hud-cyan text-railway-darkest font-extrabold'
                      : 'text-railway-muted hover:text-white'
                  }`}
                >
                  {spd}x
                </button>
              ))}
            </div>
          </div>

          {/* Quick Anomaly Jump Pills */}
          <div className="flex items-center gap-1 overflow-x-auto py-0.5">
            <span className="text-[10px] text-railway-muted uppercase pr-1 flex items-center gap-1">
              <MapPin className="h-3 w-3 text-hud-amber" /> Jump:
            </span>
            {GROUND_TRUTH_ANOMALIES.map((site) => (
              <button
                key={`jump-${site.id}`}
                onClick={() => onJumpToAnomaly(site.id, site.cum_track_offset_m)}
                className={`px-2 py-0.5 rounded border text-[10px] font-bold transition-all whitespace-nowrap ${
                  nearestAnomaly?.site_id === site.id
                    ? 'bg-hud-amber text-railway-darkest border-hud-amber'
                    : 'bg-railway-darkest text-railway-muted border-railway-border hover:text-white hover:border-hud-cyan'
                }`}
                title={`Site ${site.id}: ${site.type} (${site.cum_track_offset_m.toFixed(1)}m)`}
              >
                S{site.id} ({site.cum_track_offset_m.toFixed(0)}m)
              </button>
            ))}
          </div>

          {/* Procedural Audio Toggle */}
          <button
            onClick={() => setAudioEnabled(!audioEnabled)}
            className={`flex items-center gap-1 px-2.5 py-1 rounded border text-[11px] font-bold transition-all ${
              audioEnabled
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50'
                : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
            }`}
            title="Real-time Web Audio API procedural sonification"
          >
            {audioEnabled && !audioMuted ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
            <span>{audioEnabled ? 'AUDIO ON' : 'AUDIO OFF'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
