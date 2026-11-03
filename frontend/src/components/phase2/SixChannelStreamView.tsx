import React, { useRef, useEffect } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Gauge,
  MapPin,
  Clock,
  Eye,
  EyeOff,
  Sliders
} from 'lucide-react';
import { useSignalStore } from '../../store/signalStore';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

export const SixChannelStreamView: React.FC = () => {
  const {
    runs,
    selectedRunId,
    currentRunMeta,
    channels,
    overview,
    playbackState,
    playbackSpeed,
    currentTime,
    currentDistance,
    currentSpeed,
    spatialZone,
    nearestAnomaly,
    activeChannel,
    visibleChannels,
    selectRun,
    play,
    pause,
    reset,
    seekTime,
    setPlaybackSpeed,
    setActiveChannel,
    toggleChannelVisibility
  } = useSignalStore();

  const trackBarRef = useRef<HTMLDivElement>(null);
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null, null, null]);

  const duration = overview?.duration_s || 296.5;
  const trackLength = currentRunMeta?.track_length_m || 1400.0;
  const trackRatio = trackLength > 0 ? Math.min(1, Math.max(0, currentDistance / trackLength)) : 0;

  // Format time mm:ss.s
  const formatTime = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    return `${mins.toString().padStart(2, '0')}:${parseFloat(s) < 10 ? '0' : ''}${s}`;
  };

  // Click on track bar to scrub
  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!trackBarRef.current || !overview || !overview.points) return;
    const rect = trackBarRef.current.getBoundingClientRect();
    const clickX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const ratio = clickX / rect.width;
    const targetChainage = ratio * trackLength;

    // Find point nearest to this chainage
    let bestPt = overview.points[0];
    let minDiff = Infinity;
    for (const pt of overview.points) {
      const diff = Math.abs(pt.d - targetChainage);
      if (diff < minDiff) {
        minDiff = diff;
        bestPt = pt;
      }
    }
    seekTime(bestPt.t);
  };

  // Render synchronized waveforms on 2D canvases for 60fps performance without React re-renders
  useEffect(() => {
    if (!overview || !overview.points || overview.points.length === 0) return;

    const points = overview.points;
    const rate = overview.effective_rate_hz || 20;
    const currentIdx = Math.floor(currentTime * rate);
    const windowSeconds = 14; // Display a 14-second window
    const halfWindowPts = Math.floor((windowSeconds * rate) / 2);

    const startIdx = Math.max(0, currentIdx - halfWindowPts);
    const endIdx = Math.min(points.length - 1, currentIdx + halfWindowPts);

    canvasRefs.current.forEach((canvas, chIdx) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      const midY = height / 2;

      ctx.clearRect(0, 0, width, height);

      // Draw subtle grid lines
      ctx.strokeStyle = '#1e2430';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(width, midY);
      ctx.stroke();

      const chDef = channels[chIdx];
      const strokeColor = chDef ? chDef.color : '#06b6d4';

      // Amplitude range scaling per channel
      // Axle box (Ch 2, 4) has much larger g scale (±2.5g) than cabin floor (Ch 5, 6: ±0.1g)
      let scaleG = 1.5;
      if (chIdx === 1 || chIdx === 3) scaleG = 2.5; // Axle boxes
      if (chIdx === 4 || chIdx === 5) scaleG = 0.15; // Floor

      const totalSlice = endIdx - startIdx;
      if (totalSlice <= 1) return;

      // Draw waveform envelope
      ctx.beginPath();
      for (let i = startIdx; i <= endIdx; i++) {
        const pt = points[i];
        const x = ((i - startIdx) / totalSlice) * width;
        const [minVal, maxVal] = pt.ch[chIdx] || [-0.1, 0.1];

        // Anomaly highlight background
        if (pt.a === 1) {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
          ctx.fillRect(x - 1, 0, 3, height);
        }

        const yMin = midY - (minVal / scaleG) * (height * 0.45);
        const yMax = midY - (maxVal / scaleG) * (height * 0.45);

        ctx.strokeStyle = pt.a === 1 ? '#ef4444' : strokeColor;
        ctx.beginPath();
        ctx.moveTo(x, Math.min(height - 1, Math.max(1, yMin)));
        ctx.lineTo(x, Math.min(height - 1, Math.max(1, yMax)));
        ctx.stroke();
      }

      // Draw center playhead marker (Current Playhead)
      const playheadX = ((currentIdx - startIdx) / totalSlice) * width;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  }, [currentTime, overview, channels]);

  // Current instantaneous sample values from current point
  const currentPt = overview?.points
    ? overview.points[Math.min(overview.points.length - 1, Math.floor(currentTime * (overview.effective_rate_hz || 20)))]
    : null;

  return (
    <div className="space-y-4 rounded-xl border border-railway-border/80 bg-railway-surface/60 p-5 backdrop-blur shadow-xl">
      {/* Top Header & Recording Selector */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-railway-border/70 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono font-extrabold text-sm sm:text-base uppercase tracking-wider text-white">
              6-Channel Synchronized Vibration Stream
            </span>
            <DataIntegrityBadge type="MEASURED" size="sm" />
            <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
          </div>
          <p className="text-xs text-railway-muted font-sans mt-0.5">
            Recorded 2,000 Hz Class 230 telemetry playback synchronized with physical chainage and surveyed defect sites
          </p>
        </div>

        {/* Recording Selector & Direction Indicator */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-railway-muted uppercase">Run:</span>
          <select
            value={selectedRunId}
            onChange={(e) => selectRun(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-railway-darkest border border-railway-border text-xs font-mono text-white focus:outline-none focus:border-hud-cyan cursor-pointer"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.direction} • {r.target_speed_kmh} km/h • {r.duration_s}s)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Primary Telemetry Readout Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5 font-mono text-xs">
        {/* Playback Time */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/80 border border-railway-border/70">
          <span className="text-[10px] text-railway-muted uppercase flex items-center gap-1">
            <Clock className="h-3 w-3 text-hud-cyan" /> Timestamp
          </span>
          <div className="text-white font-bold text-sm">
            {formatTime(currentTime)}{' '}
            <span className="text-railway-muted text-[11px] font-normal">/ {formatTime(duration)}</span>
          </div>
        </div>

        {/* Track Chainage */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/80 border border-railway-border/70">
          <span className="text-[10px] text-railway-muted uppercase flex items-center gap-1">
            <MapPin className="h-3 w-3 text-hud-amber" /> Chainage
          </span>
          <div className="text-hud-amber font-bold text-sm">
            {currentDistance.toFixed(1)} m{' '}
            <span className="text-railway-muted text-[11px] font-normal">/ {trackLength.toFixed(0)}m</span>
          </div>
        </div>

        {/* Vehicle Speed */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/80 border border-railway-border/70">
          <span className="text-[10px] text-railway-muted uppercase flex items-center gap-1">
            <Gauge className="h-3 w-3 text-hud-cyan" /> Target Speed
          </span>
          <div className="text-white font-bold text-sm">
            {currentSpeed.toFixed(1)} km/h{' '}
            <span className="text-railway-muted text-[10px] font-normal uppercase">
              ({currentRunMeta?.direction === 'AtoB' ? 'Uphill' : 'Downhill'})
            </span>
          </div>
        </div>

        {/* Spatial Zone Classification */}
        <div className="p-2.5 rounded-lg bg-railway-darkest/80 border border-railway-border/70">
          <span className="text-[10px] text-railway-muted uppercase block">Spatial Zone</span>
          <span
            className={`font-bold text-sm px-1.5 py-0.5 rounded text-[11px] inline-block ${
              spatialZone === 'ANOMALY'
                ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse'
                : spatialZone === 'BUFFER'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                : 'bg-hud-cyan/15 text-hud-cyan border border-hud-cyan/30'
            }`}
          >
            {spatialZone === 'ANOMALY' ? '● DEFECT SITE' : spatialZone === 'BUFFER' ? '◐ BUFFER ZONE' : '○ TANGENT TRACK'}
          </span>
        </div>

        {/* Nearest Ground Truth Anomaly */}
        <div className="col-span-2 p-2.5 rounded-lg bg-railway-darkest/80 border border-railway-border/70">
          <span className="text-[10px] text-railway-muted uppercase block">Nearest Physical Anomaly</span>
          {nearestAnomaly ? (
            <div className="flex items-center justify-between text-xs mt-0.5">
              <span className="font-bold text-white">
                Site {nearestAnomaly.site_id}: {nearestAnomaly.type}
              </span>
              <span
                className={`font-mono text-xs ${
                  nearestAnomaly.distance_m <= 6 ? 'text-rose-400 font-bold' : 'text-railway-muted'
                }`}
              >
                Δ = {nearestAnomaly.distance_m.toFixed(1)} m ({nearestAnomaly.chainage_m.toFixed(1)}m)
              </span>
            </div>
          ) : (
            <span className="text-railway-muted text-xs">Scanning track...</span>
          )}
        </div>
      </div>

      {/* Visual Track Schematic & Ground Truth Marker Bar */}
      <div className="space-y-1.5 pt-1">
        <div className="flex items-center justify-between text-[11px] font-mono text-railway-muted">
          <span>TRACK SCHEMATIC (Orbisonia Station 0.0m → Robertsdale 1,400.0m)</span>
          <span className="text-hud-cyan">Click anywhere on track to scrub</span>
        </div>

        <div
          ref={trackBarRef}
          onClick={handleTrackClick}
          className="relative h-9 w-full bg-railway-darkest rounded-lg border border-railway-border/90 overflow-hidden cursor-pointer select-none group"
        >
          {/* Track Rails Representation */}
          <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1.5 bg-railway-border/40" />
          <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-0.5 bg-hud-cyan/30" />

          {/* Surveyed Ground Truth Defect Markers on Track */}
          {currentRunMeta?.anomalies.map((anom) => {
            const anomRatio = trackLength > 0 ? anom.chainage_m / trackLength : 0;
            const leftPct = `${(anomRatio * 100).toFixed(2)}%`;
            const isNear = nearestAnomaly?.site_id === anom.site_id && nearestAnomaly.distance_m <= 6;

            return (
              <div
                key={anom.site_id}
                style={{ left: leftPct }}
                title={`Site ${anom.site_id}: ${anom.type} at ${anom.chainage_m}m (Cone surveyed)`}
                className="absolute top-0 bottom-0 -translate-x-1/2 flex flex-col items-center justify-between py-0.5 pointer-events-none"
              >
                <div
                  className={`h-2.5 w-1.5 rounded-sm ${
                    isNear
                      ? 'bg-rose-500 ring-2 ring-rose-400 animate-ping'
                      : anom.type.includes('Switch') || anom.type.includes('Divergence') || anom.type.includes('Convergence')
                      ? 'bg-hud-amber'
                      : 'bg-rose-500'
                  }`}
                />
                <span className="text-[9px] font-mono text-white/70 bg-railway-darkest/90 px-0.5 rounded scale-75">
                  S{anom.site_id}
                </span>
              </div>
            );
          })}

          {/* Real-time Moving Train Cursor */}
          <div
            style={{ left: `${(trackRatio * 100).toFixed(2)}%` }}
            className="absolute top-0 bottom-0 -translate-x-1/2 w-4 flex flex-col items-center justify-center pointer-events-none transition-all duration-75"
          >
            <div className="w-2.5 h-full bg-hud-cyan/30 border-x border-hud-cyan" />
            <div className="absolute w-3.5 h-3.5 rounded-full bg-hud-cyan shadow-lg shadow-hud-cyan/80 border border-white" />
          </div>
        </div>
      </div>

      {/* Transport Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-lg bg-railway-darkest/90 border border-railway-border/80">
        {/* Play / Pause / Reset Buttons */}
        <div className="flex items-center gap-2">
          {playbackState === 'playing' ? (
            <button
              onClick={pause}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-hud-amber/20 hover:bg-hud-amber/30 text-hud-amber border border-hud-amber/50 font-mono text-xs font-bold transition-all shadow-md shadow-hud-amber/10"
            >
              <Pause className="h-4 w-4" />
              PAUSE
            </button>
          ) : (
            <button
              onClick={play}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-hud-cyan/20 hover:bg-hud-cyan/30 text-hud-cyan border border-hud-cyan/50 font-mono text-xs font-bold transition-all shadow-md shadow-hud-cyan/10"
            >
              <Play className="h-4 w-4" />
              PLAY
            </button>
          )}

          <button
            onClick={reset}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-railway-elevated hover:bg-railway-border text-railway-text border border-railway-border font-mono text-xs transition-colors"
            title="Reset playback to 0.0s"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            RESET
          </button>
        </div>

        {/* Speed Controls */}
        <div className="flex items-center gap-1.5 font-mono text-xs">
          <span className="text-railway-muted uppercase text-[10px] mr-1 flex items-center gap-1">
            <Sliders className="h-3 w-3 text-hud-cyan" /> Speed:
          </span>
          {[0.25, 0.5, 1.0, 2.0, 4.0].map((spd) => (
            <button
              key={spd}
              onClick={() => setPlaybackSpeed(spd)}
              className={`px-2 py-1 rounded border text-[11px] transition-colors ${
                playbackSpeed === spd
                  ? 'bg-hud-cyan text-railway-darkest font-bold border-hud-cyan'
                  : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
              }`}
            >
              {spd}x
            </button>
          ))}
        </div>

        {/* Scrubber slider */}
        <div className="flex items-center gap-2 flex-1 min-w-[200px]">
          <input
            type="range"
            min="0"
            max={duration}
            step="0.05"
            value={currentTime}
            onChange={(e) => seekTime(parseFloat(e.target.value))}
            className="w-full accent-hud-cyan h-1.5 bg-railway-border rounded cursor-pointer"
          />
        </div>
      </div>

      {/* 6 Synchronized Waveform Strips */}
      <div className="space-y-2.5 pt-2">
        <div className="flex items-center justify-between text-xs font-mono text-railway-muted">
          <span>6-CHANNEL WAVEFORM STRIPS (Real-Time 14-Second Moving Window)</span>
          <span className="text-[11px] text-railway-muted">Red regions indicate verified ground-truth anomaly zones</span>
        </div>

        <div className="space-y-2">
          {channels.map((ch, idx) => {
            const isVisible = visibleChannels[idx];
            const isSolo = activeChannel === ch.id;
            const currentAmp = currentPt && currentPt.ch[idx]
              ? ((currentPt.ch[idx][0] + currentPt.ch[idx][1]) / 2).toFixed(3)
              : '0.000';

            return (
              <div
                key={ch.id}
                className={`rounded-lg border transition-all ${
                  isSolo
                    ? 'border-hud-cyan/60 bg-railway-darkest/95 shadow-md shadow-hud-cyan/10'
                    : 'border-railway-border/60 bg-railway-darkest/70'
                }`}
              >
                {/* Channel Header Strip */}
                <div className="px-3 py-1.5 flex flex-wrap items-center justify-between gap-2 border-b border-railway-border/40 text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <span
                      style={{ backgroundColor: ch.color }}
                      className="h-2.5 w-2.5 rounded-full inline-block shrink-0"
                    />
                    <span className="font-bold text-white tracking-wide">
                      CH {ch.id}: {ch.label}
                    </span>
                    <span className="text-[10px] text-railway-muted hidden sm:inline">
                      • {ch.location}
                    </span>
                  </div>

                  <div className="flex items-center gap-3">
                    {/* Instantaneous Amplitude Readout */}
                    <div className="text-[11px]">
                      <span className="text-railway-muted text-[10px] mr-1">AMP:</span>
                      <span
                        style={{ color: ch.color }}
                        className="font-bold font-mono"
                      >
                        {parseFloat(currentAmp) > 0 ? `+${currentAmp}` : currentAmp} g
                      </span>
                    </div>

                    {/* Solo Button */}
                    <button
                      onClick={() => setActiveChannel(ch.id)}
                      className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-colors ${
                        isSolo
                          ? 'bg-hud-cyan text-railway-darkest border-hud-cyan'
                          : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
                      }`}
                    >
                      {isSolo ? 'SOLO ACTIVE' : 'SOLO'}
                    </button>

                    {/* Visibility Toggle */}
                    <button
                      onClick={() => toggleChannelVisibility(ch.id)}
                      className="p-1 rounded text-railway-muted hover:text-white"
                      title={isVisible ? 'Hide channel strip' : 'Show channel strip'}
                    >
                      {isVisible ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5 opacity-50" />}
                    </button>
                  </div>
                </div>

                {/* Oscilloscope Mini Canvas */}
                {isVisible && (
                  <div className="relative h-14 w-full bg-[#07080a]">
                    <canvas
                      ref={(el) => (canvasRefs.current[idx] = el)}
                      width={900}
                      height={56}
                      className="w-full h-full block"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
