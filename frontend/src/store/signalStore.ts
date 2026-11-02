import { create } from 'zustand';
import {
  SignalsIndexData,
  SignalRunOverview,
  MicroWindowsFile,
  MicroWindowData,
  ChannelDefinition,
  RunCatalogItem
} from '../types';
import { computeSpectrum, WindowFunctionType, SpectralAnalysisResult } from '../utils/fft';
import { proceduralAudio } from '../utils/proceduralAudio';

export interface NearestAnomalyInfo {
  site_id: number;
  type: string;
  chainage_m: number;
  distance_m: number;
}

export interface SignalStoreState {
  // Index & Metadata
  signalsIndex: SignalsIndexData | null;
  channels: ChannelDefinition[];
  runs: RunCatalogItem[];
  selectedRunId: string;
  currentRunMeta: RunCatalogItem | null;

  // Active Datasets for selected run
  overview: SignalRunOverview | null;
  microWindowsFile: MicroWindowsFile | null;
  isLoading: boolean;
  error: string | null;

  // Playback & Time State
  playbackState: 'idle' | 'playing' | 'paused';
  playbackSpeed: number; // 0.25, 0.5, 1, 2, 4
  currentTime: number; // in seconds
  currentDistance: number; // in meters
  currentSpeed: number; // in km/h
  spatialZone: 'NORMAL' | 'ANOMALY' | 'BUFFER';
  nearestAnomaly: NearestAnomalyInfo | null;

  // Channel & Selection State
  activeChannel: number; // 1 to 6 (default 2: Axle Box)
  visibleChannels: boolean[]; // [ch1, ch2, ch3, ch4, ch5, ch6]
  activeMicroWindowId: string | null;
  selectedMicroWindow: MicroWindowData | null;

  // FFT Configuration & Spectral Analysis
  fftWindowSize: number; // 512, 1024, 2048
  fftWindowType: WindowFunctionType; // 'hanning' | 'hamming' | 'rectangular'
  fftChannel: number; // 1 to 6
  fftCompareMode: boolean;
  fftResult: SpectralAnalysisResult | null;
  baselineFftResult: SpectralAnalysisResult | null;

  // Procedural Audio State
  audioEnabled: boolean;
  audioVolume: number;
  audioMuted: boolean;

  // Actions
  initialize: () => Promise<void>;
  selectRun: (runId: string) => Promise<void>;
  play: () => void;
  pause: () => void;
  reset: () => void;
  seekTime: (timeSec: number) => void;
  setPlaybackSpeed: (speed: number) => void;
  setActiveChannel: (channelId: number) => void;
  toggleChannelVisibility: (channelId: number) => void;
  selectMicroWindow: (windowId: string | null) => void;
  setFFTConfig: (config: {
    windowSize?: number;
    windowType?: WindowFunctionType;
    channel?: number;
    compareMode?: boolean;
  }) => void;
  setAudioEnabled: (enabled: boolean) => void;
  setAudioVolume: (volume: number) => void;
  toggleAudioMute: () => void;
}

let animationFrameId: number | null = null;
let lastTimestamp: number | null = null;

export const useSignalStore = create<SignalStoreState>((set, get) => {
  // Helper to recompute FFTs
  const updateFFT = (
    microWindows: MicroWindowsFile | null,
    selectedWinId: string | null,
    fftCh: number,
    winSize: number,
    winType: WindowFunctionType
  ) => {
    if (!microWindows || !microWindows.windows || microWindows.windows.length === 0) {
      return { fftResult: null, baselineFftResult: null };
    }

    // Selected target window
    const targetWin = selectedWinId
      ? microWindows.windows.find((w) => w.id === selectedWinId) || microWindows.windows[0]
      : microWindows.windows[0];

    // Baseline normal tangent window
    const baselineWin = microWindows.windows.find((w) => w.type === 'NORMAL') || microWindows.windows[0];

    const chIdx = Math.max(0, Math.min(5, fftCh - 1));

    let fftRes: SpectralAnalysisResult | null = null;
    let baseRes: SpectralAnalysisResult | null = null;

    if (targetWin && targetWin.channels_data && targetWin.channels_data[chIdx]) {
      fftRes = computeSpectrum(targetWin.channels_data[chIdx], 2000, winSize, winType);
    }

    if (baselineWin && baselineWin.channels_data && baselineWin.channels_data[chIdx]) {
      baseRes = computeSpectrum(baselineWin.channels_data[chIdx], 2000, winSize, winType);
    }

    return { fftResult: fftRes, baselineFftResult: baseRes };
  };

  // Helper to calculate telemetry at a given time
  const getTelemetryAtTime = (t: number, overview: SignalRunOverview | null, runMeta: RunCatalogItem | null) => {
    if (!overview || !overview.points || overview.points.length === 0) {
      return {
        distance: 0,
        zone: 'NORMAL' as const,
        nearest: null,
        ch1_g: 0,
        ch2_g: 0
      };
    }

    const duration = overview.duration_s || 296.0;
    const clampedT = Math.max(0, Math.min(duration, t));
    const estIdx = Math.floor(clampedT * overview.effective_rate_hz);
    const idx = Math.max(0, Math.min(overview.points.length - 1, estIdx));
    const pt = overview.points[idx];

    const dist = pt.d;
    let zone: 'NORMAL' | 'ANOMALY' | 'BUFFER' = 'NORMAL';
    if (pt.a === 1) zone = 'ANOMALY';
    else if (pt.a === -1) zone = 'BUFFER';

    let nearest: NearestAnomalyInfo | null = null;
    if (runMeta && runMeta.anomalies && runMeta.anomalies.length > 0) {
      let minDelta = Infinity;
      for (const a of runMeta.anomalies) {
        const delta = Math.abs(a.chainage_m - dist);
        if (delta < minDelta) {
          minDelta = delta;
          nearest = {
            site_id: a.site_id,
            type: a.type,
            chainage_m: a.chainage_m,
            distance_m: Math.round(delta * 10) / 10
          };
        }
      }
    }

    const ch1_g = (pt.ch[0][0] + pt.ch[0][1]) / 2;
    const ch2_g = Math.abs(pt.ch[1][1]) > Math.abs(pt.ch[1][0]) ? pt.ch[1][1] : pt.ch[1][0];

    return {
      distance: dist,
      zone,
      nearest,
      ch1_g,
      ch2_g
    };
  };

  // Main playback tick loop
  const tick = (now: number) => {
    const state = get();
    if (state.playbackState !== 'playing') {
      return;
    }

    if (lastTimestamp === null) {
      lastTimestamp = now;
      animationFrameId = requestAnimationFrame(tick);
      return;
    }

    const dt = (now - lastTimestamp) / 1000;
    lastTimestamp = now;

    const maxDuration = state.overview?.duration_s || 300;
    let nextTime = state.currentTime + dt * state.playbackSpeed;

    if (nextTime >= maxDuration) {
      nextTime = maxDuration;
      get().pause();
      return;
    }

    const telemetry = getTelemetryAtTime(nextTime, state.overview, state.currentRunMeta);

    // Update procedural audio if active
    if (state.audioEnabled && !state.audioMuted) {
      const speed = state.currentRunMeta?.target_speed_kmh || 20;
      proceduralAudio.updateTelemetry(speed, telemetry.ch1_g, telemetry.ch2_g, telemetry.zone === 'ANOMALY');
    }

    set({
      currentTime: nextTime,
      currentDistance: telemetry.distance,
      spatialZone: telemetry.zone,
      nearestAnomaly: telemetry.nearest
    });

    animationFrameId = requestAnimationFrame(tick);
  };

  return {
    signalsIndex: null,
    channels: [],
    runs: [],
    selectedRunId: 'AtoB_20_1',
    currentRunMeta: null,

    overview: null,
    microWindowsFile: null,
    isLoading: false,
    error: null,

    playbackState: 'idle',
    playbackSpeed: 1.0,
    currentTime: 0.0,
    currentDistance: 0.0,
    currentSpeed: 20.0,
    spatialZone: 'NORMAL',
    nearestAnomaly: null,

    activeChannel: 2, // Front Wheel Axle Box
    visibleChannels: [true, true, true, true, true, true],
    activeMicroWindowId: 'anomaly_site_1',
    selectedMicroWindow: null,

    fftWindowSize: 1024,
    fftWindowType: 'hanning',
    fftChannel: 2,
    fftCompareMode: true,
    fftResult: null,
    baselineFftResult: null,

    audioEnabled: false,
    audioVolume: 0.3,
    audioMuted: false,

    initialize: async () => {
      set({ isLoading: true, error: null });
      try {
        const indexRes = await fetch('/data/signals_index.json');
        if (!indexRes.ok) throw new Error(`HTTP ${indexRes.status} loading signals_index.json`);
        const indexData: SignalsIndexData = await indexRes.json();

        const defaultRun = indexData.runs.find((r) => r.run_id === 'AtoB_20_1') || indexData.runs[0];

        set({
          signalsIndex: indexData,
          channels: indexData.channels,
          runs: indexData.runs,
          selectedRunId: defaultRun.run_id,
          currentRunMeta: defaultRun,
          currentSpeed: defaultRun.target_speed_kmh
        });

        // Load default run data
        await get().selectRun(defaultRun.run_id);
      } catch (err) {
        console.error('Failed to initialize SignalStore:', err);
        set({ error: (err as Error).message, isLoading: false });
      }
    },

    selectRun: async (runId: string) => {
      set({ isLoading: true, error: null, selectedRunId: runId });
      get().pause();

      try {
        const [overviewRes, microRes] = await Promise.all([
          fetch(`/data/run_${runId}_overview.json`),
          fetch(`/data/run_${runId}_microwindows.json`)
        ]);

        if (!overviewRes.ok) throw new Error(`Overview HTTP ${overviewRes.status}`);
        if (!microRes.ok) throw new Error(`MicroWindows HTTP ${microRes.status}`);

        const overviewData: SignalRunOverview = await overviewRes.json();
        const microData: MicroWindowsFile = await microRes.json();

        const runMeta = get().runs.find((r) => r.run_id === runId) || null;
        const initialMicro = microData.windows.find((w) => w.id === 'anomaly_site_1') || microData.windows[0] || null;

        const { fftResult, baselineFftResult } = updateFFT(
          microData,
          initialMicro ? initialMicro.id : null,
          get().fftChannel,
          get().fftWindowSize,
          get().fftWindowType
        );

        const initialTelem = getTelemetryAtTime(0, overviewData, runMeta);

        set({
          overview: overviewData,
          microWindowsFile: microData,
          currentRunMeta: runMeta,
          currentSpeed: runMeta ? runMeta.target_speed_kmh : 20.0,
          currentTime: 0,
          currentDistance: initialTelem.distance,
          spatialZone: initialTelem.zone,
          nearestAnomaly: initialTelem.nearest,
          playbackState: 'idle',
          activeMicroWindowId: initialMicro ? initialMicro.id : null,
          selectedMicroWindow: initialMicro,
          fftResult,
          baselineFftResult,
          isLoading: false
        });
      } catch (err) {
        console.error(`Failed to load run data for ${runId}:`, err);
        set({ error: (err as Error).message, isLoading: false });
      }
    },

    play: () => {
      if (get().playbackState === 'playing') return;

      if (get().audioEnabled && !get().audioMuted) {
        proceduralAudio.init();
        proceduralAudio.resume();
      }

      lastTimestamp = performance.now();
      set({ playbackState: 'playing' });
      animationFrameId = requestAnimationFrame(tick);
    },

    pause: () => {
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
      lastTimestamp = null;
      proceduralAudio.pause();
      set({ playbackState: 'paused' });
    },

    reset: () => {
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
      lastTimestamp = null;
      proceduralAudio.stop();

      const initialTelem = getTelemetryAtTime(0, get().overview, get().currentRunMeta);

      set({
        playbackState: 'idle',
        currentTime: 0,
        currentDistance: initialTelem.distance,
        spatialZone: initialTelem.zone,
        nearestAnomaly: initialTelem.nearest
      });
    },

    seekTime: (timeSec: number) => {
      const maxDur = get().overview?.duration_s || 300;
      const clampedT = Math.max(0, Math.min(maxDur, timeSec));
      const telem = getTelemetryAtTime(clampedT, get().overview, get().currentRunMeta);

      if (get().audioEnabled && !get().audioMuted) {
        const speed = get().currentRunMeta?.target_speed_kmh || 20;
        proceduralAudio.updateTelemetry(speed, telem.ch1_g, telem.ch2_g, telem.zone === 'ANOMALY');
      }

      set({
        currentTime: clampedT,
        currentDistance: telem.distance,
        spatialZone: telem.zone,
        nearestAnomaly: telem.nearest
      });
    },

    setPlaybackSpeed: (speed: number) => {
      set({ playbackSpeed: speed });
    },

    setActiveChannel: (channelId: number) => {
      const clamped = Math.max(1, Math.min(6, channelId));
      const { fftResult, baselineFftResult } = updateFFT(
        get().microWindowsFile,
        get().activeMicroWindowId,
        clamped,
        get().fftWindowSize,
        get().fftWindowType
      );

      set({
        activeChannel: clamped,
        fftChannel: clamped,
        fftResult,
        baselineFftResult
      });
    },

    toggleChannelVisibility: (channelId: number) => {
      const idx = channelId - 1;
      if (idx < 0 || idx > 5) return;
      const newVis = [...get().visibleChannels];
      newVis[idx] = !newVis[idx];
      set({ visibleChannels: newVis });
    },

    selectMicroWindow: (windowId: string | null) => {
      const microFile = get().microWindowsFile;
      let targetWin: MicroWindowData | null = null;

      if (windowId && microFile) {
        targetWin = microFile.windows.find((w) => w.id === windowId) || null;
      }

      const { fftResult, baselineFftResult } = updateFFT(
        microFile,
        windowId,
        get().fftChannel,
        get().fftWindowSize,
        get().fftWindowType
      );

      // Seek playback scrubber to window start time for synchronization
      if (targetWin) {
        get().seekTime(targetWin.time_start_s);
      }

      set({
        activeMicroWindowId: windowId,
        selectedMicroWindow: targetWin,
        fftResult,
        baselineFftResult
      });
    },

    setFFTConfig: (config) => {
      const winSize = config.windowSize !== undefined ? config.windowSize : get().fftWindowSize;
      const winType = config.windowType !== undefined ? config.windowType : get().fftWindowType;
      const ch = config.channel !== undefined ? config.channel : get().fftChannel;
      const compare = config.compareMode !== undefined ? config.compareMode : get().fftCompareMode;

      const { fftResult, baselineFftResult } = updateFFT(
        get().microWindowsFile,
        get().activeMicroWindowId,
        ch,
        winSize,
        winType
      );

      set({
        fftWindowSize: winSize,
        fftWindowType: winType,
        fftChannel: ch,
        fftCompareMode: compare,
        fftResult,
        baselineFftResult
      });
    },

    setAudioEnabled: (enabled: boolean) => {
      if (enabled) {
        proceduralAudio.init();
        if (get().playbackState === 'playing') {
          proceduralAudio.resume();
        }
      } else {
        proceduralAudio.stop();
      }
      set({ audioEnabled: enabled });
    },

    setAudioVolume: (volume: number) => {
      proceduralAudio.setVolume(volume);
      set({ audioVolume: volume });
    },

    toggleAudioMute: () => {
      const newMuted = !get().audioMuted;
      proceduralAudio.setMuted(newMuted);
      set({ audioMuted: newMuted });
    }
  };
});
