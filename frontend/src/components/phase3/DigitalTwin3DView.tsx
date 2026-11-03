import React, { useState, Suspense, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { Class230Train } from './Class230Train';
import { TrackEnvironment } from './TrackEnvironment';
import { CinematicCameraController } from './CinematicCameraController';
import { DigitalTwinControls } from './DigitalTwinControls';
import { CameraViewMode, Scene3DOverlays } from '../../types';
import { useSignalStore } from '../../store/signalStore';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

export const DigitalTwin3DView: React.FC = () => {
  const [viewMode, setViewMode] = useState<CameraViewMode>('side-follow');
  const [targetAnomalyZ, setTargetAnomalyZ] = useState<number | null>(null);

  const [overlays, setOverlays] = useState<Scene3DOverlays>({
    showSensors: true,
    showAnomalies: true,
    showZones: true,
    showBogieWireframe: false,
    showCameraFrustum: true,
    lightingMode: 'night'
  });

  const {
    overview,
    runs,
    selectedRunId,
    selectRun,
    seekTime,
    setActiveChannel,
    selectMicroWindow,
    initialize
  } = useSignalStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  // Jump camera and playback chainage to a specific surveyed defect site
  const handleJumpToAnomaly = (siteId: number, chainage: number) => {
    if (!overview || !overview.points || overview.points.length === 0) return;

    // Find closest playback time for the chainage
    let closestTime = 0;
    let minDiff = Infinity;
    overview.points.forEach((pt) => {
      const diff = Math.abs(pt.d - chainage);
      if (diff < minDiff) {
        minDiff = diff;
        closestTime = pt.t;
      }
    });

    seekTime(closestTime);
    setTargetAnomalyZ(chainage);

    // Also select corresponding micro-window if available
    selectMicroWindow(`anomaly_site_${siteId}`);

    // Automatically clear focus target after 4 seconds to return to dynamic tracking
    setTimeout(() => {
      setTargetAnomalyZ(null);
    }, 4500);
  };

  const handleSelectSensor = (channelId: number) => {
    setActiveChannel(channelId);
  };

  const isNight = overlays.lightingMode === 'night';

  return (
    <div className="space-y-4">
      {/* ========================================================================= */}
      {/* 1. TOP INTERACTIVE CONTROLS & TELEMETRY HUD BAR                           */}
      {/* ========================================================================= */}
      <DigitalTwinControls
        viewMode={viewMode}
        setViewMode={setViewMode}
        overlays={overlays}
        setOverlays={setOverlays}
        onJumpToAnomaly={handleJumpToAnomaly}
      />

      {/* ========================================================================= */}
      {/* 2. 3D WEBGL PHOTOREALISTIC CANVAS VIEWPORT                                */}
      {/* ========================================================================= */}
      <div className="relative w-full h-[620px] rounded-xl overflow-hidden border border-railway-border/90 bg-[#08090c] shadow-2xl">
        {/* Real-time Cinematic Watermark & Provenance Tag */}
        <div className="absolute top-4 left-4 z-10 flex flex-col gap-1 pointer-events-none">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-extrabold text-white uppercase tracking-wider bg-railway-darkest/80 px-2.5 py-1 rounded border border-railway-border/70 backdrop-blur">
              CLASS 230 DIGITAL TWIN • ORBISONIA TESTBED
            </span>
            <DataIntegrityBadge type="SIMULATED" size="sm" />
          </div>
          <span className="text-[10px] font-mono text-railway-muted bg-railway-darkest/70 px-2 py-0.5 rounded w-fit backdrop-blur">
            Physically Grounded Kinematic Playback (1,400m Track Span)
          </span>
        </div>

        {/* Active Camera View Watermark (Bottom Right) */}
        <div className="absolute bottom-4 right-4 z-10 pointer-events-none">
          <span className="font-mono text-[11px] text-hud-cyan bg-railway-darkest/80 px-3 py-1 rounded border border-hud-cyan/30 backdrop-blur font-bold uppercase flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-hud-cyan animate-pulse" />
            CAMERA: {viewMode.replace('-', ' ')}
          </span>
        </div>

        {/* Active Run Selector Overlay (Top Right) */}
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
          <select
            value={selectedRunId}
            onChange={(e) => selectRun(e.target.value)}
            className="px-3 py-1 rounded-lg bg-railway-darkest/90 border border-railway-border text-xs font-mono text-white focus:outline-none focus:border-hud-cyan cursor-pointer shadow-lg backdrop-blur"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.direction} • {r.target_speed_kmh} km/h)
              </option>
            ))}
          </select>
        </div>

        {/* 3D Canvas Context */}
        <Canvas
          shadows
          gl={{
            antialias: true,
            powerPreference: 'high-performance',
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: isNight ? 1.15 : 1.05
          }}
          camera={{
            position: [9.5, 3.2, -2.5],
            fov: 45,
            near: 0.1,
            far: 2000
          }}
          style={{ width: '100%', height: '100%' }}
        >
          {/* Atmospheric Environmental Fog blending into Carbon-Black (#08090c) */}
          <color attach="background" args={[isNight ? '#08090c' : '#1e2430']} />
          <fog attach="fog" args={[isNight ? '#08090c' : '#1e2430', 40, 420]} />

          {/* Cinematic Lighting Setup */}
          <Suspense fallback={null}>
            {/* Ambient Lighting Fill */}
            <ambientLight
              color={isNight ? '#94a3b8' : '#cbd5e1'}
              intensity={isNight ? 0.45 : 0.85}
            />

            {/* Main Directional Moonlight / Sunlight */}
            <directionalLight
              position={isNight ? [25, 45, 35] : [45, 75, 55]}
              color={isNight ? '#93c5fd' : '#fffbeb'}
              intensity={isNight ? 1.6 : 3.0}
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
              shadow-camera-near={0.5}
              shadow-camera-far={250}
              shadow-camera-left={-25}
              shadow-camera-right={25}
              shadow-camera-top={25}
              shadow-camera-bottom={-25}
              shadow-bias={-0.0005}
            />

            {/* Subtle Horizon Blue Rim Light */}
            <directionalLight
              position={[-30, 15, -20]}
              color="#0284c7"
              intensity={isNight ? 0.75 : 0.3}
            />

            {/* Dynamic Train Model */}
            <Class230Train
              overlays={overlays}
              onSelectSensor={handleSelectSensor}
            />

            {/* Railway Trackbed & Anomaly Environment */}
            <TrackEnvironment
              overlays={overlays}
              onSelectAnomaly={handleJumpToAnomaly}
            />

            {/* Cinematic Camera Controller with smooth lerp interpolation */}
            <CinematicCameraController
              viewMode={viewMode}
              targetAnomalyZ={targetAnomalyZ}
            />
          </Suspense>
        </Canvas>
      </div>

      {/* ========================================================================= */}
      {/* 3. SCIENTIFIC VISUALIZATION & DATA PROVENANCE NOTICE                      */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
        <div className="p-3 rounded-lg bg-railway-surface/50 border border-railway-border/80 space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-hud-cyan font-bold uppercase">Train Kinematics</span>
            <DataIntegrityBadge type="SIMULATED" size="sm" />
            <DataIntegrityBadge type="PROCEDURAL" size="sm" />
          </div>
          <p className="text-[11px] text-railway-muted font-sans">
            Train longitudinal displacement is driven by recorded GPS chainage and measured velocity from the selected run. Chassis vertical oscillations and bogie pitch are illustrative procedural animations synchronized with anomaly spatial zones, not empirical double-integration displacements.
          </p>
        </div>

        <div className="p-3 rounded-lg bg-railway-surface/50 border border-railway-border/80 space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-hud-amber font-bold uppercase">Ground-Truth Anomaly Sites</span>
            <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
          </div>
          <p className="text-[11px] text-railway-muted font-sans">
            All 9 defect sites and surveyed orange/black cone positions are positioned at their physical surveyed coordinates from Table 2 of Ma et al. (Nature Scientific Data, 2026).
          </p>
        </div>

        <div className="p-3 rounded-lg bg-railway-surface/50 border border-railway-border/80 space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-purple-400 font-bold uppercase">Sensor Mounting Nodes</span>
            <DataIntegrityBadge type="MEASURED" size="sm" />
          </div>
          <p className="text-[11px] text-railway-muted font-sans">
            Cyan glowing nodes represent the 6 physical Silicon Designs Model 2012 accelerometers mounted on the bogie frames (Ch 1/3), axle boxes (Ch 2/4), and car body floor (Ch 5/6).
          </p>
        </div>
      </div>
    </div>
  );
};
