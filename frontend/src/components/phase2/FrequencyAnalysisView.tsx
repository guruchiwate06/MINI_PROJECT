import React from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Activity,
  CheckCircle2,
  TrendingUp,
  BarChart2,
  Layers,
  Info
} from 'lucide-react';
import { useSignalStore } from '../../store/signalStore';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import { WindowFunctionType } from '../../utils/fft';

export const FrequencyAnalysisView: React.FC = () => {
  const {
    channels,
    selectedMicroWindow,
    fftWindowSize,
    fftWindowType,
    fftChannel,
    fftCompareMode,
    fftResult,
    baselineFftResult,
    setFFTConfig
  } = useSignalStore();

  const currentCh = channels.find((c) => c.id === fftChannel) || channels[1];

  // ECharts Frequency Spectrum Plot
  const chartOption = React.useMemo(() => {
    if (!fftResult || !fftResult.frequencies) {
      return {};
    }

    const freqs = fftResult.frequencies;
    const mags = fftResult.magnitudes; // In dB
    const baseMags = baselineFftResult ? baselineFftResult.magnitudes : [];

    const series: any[] = [
      {
        name: `${selectedMicroWindow?.title || 'Selected Site'} (CH ${currentCh.id})`,
        type: 'line',
        showSymbol: false,
        sampling: 'lttb',
        lineStyle: {
          width: 2,
          color: currentCh.color
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: `${currentCh.color}44` },
              { offset: 1, color: `${currentCh.color}00` }
            ]
          }
        },
        markPoint: {
          data: [
            {
              coord: [fftResult.dominantFrequency, mags[freqs.indexOf(fftResult.dominantFrequency)] || -20],
              name: 'Dominant Peak',
              value: `${fftResult.dominantFrequency} Hz`
            }
          ],
          label: {
            color: '#ffffff',
            fontSize: 10,
            formatter: '{c}'
          },
          itemStyle: {
            color: '#ef4444'
          }
        },
        data: mags
      }
    ];

    if (fftCompareMode && baselineFftResult && baseMags.length > 0) {
      series.push({
        name: 'Nominal Tangent Track (Baseline)',
        type: 'line',
        showSymbol: false,
        sampling: 'lttb',
        lineStyle: {
          width: 1.5,
          color: '#10b981',
          type: 'dashed'
        },
        data: baseMags
      });
    }

    return {
      backgroundColor: '#07080a',
      animation: false,
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0e1117',
        borderColor: '#1e293b',
        textStyle: { color: '#f1f5f9', fontSize: 11, fontFamily: 'monospace' },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return '';
          const idx = params[0].dataIndex;
          const f = freqs[idx];
          let html = `<div style="font-weight:bold; margin-bottom:4px; color:#38bdf8;">Frequency: ${f} Hz (Nyquist: 1,000 Hz)</div>`;
          params.forEach((p) => {
            html += `<div style="display:flex; justify-content:space-between; gap:12px;">
              <span style="color:${p.color}">● ${p.seriesName}:</span>
              <span style="font-weight:bold">${Number(p.value).toFixed(1)} dB</span>
            </div>`;
          });
          return html;
        }
      },
      legend: {
        show: true,
        top: 8,
        right: 20,
        textStyle: { color: '#94a3b8', fontSize: 11, fontFamily: 'monospace' },
        icon: 'rect'
      },
      grid: {
        top: 45,
        left: 55,
        right: 30,
        bottom: 45
      },
      xAxis: {
        type: 'category',
        data: freqs,
        name: 'Frequency (Hz)',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { color: '#64748b', fontSize: 11 },
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: {
          color: '#94a3b8',
          fontSize: 10,
          interval: Math.floor(freqs.length / 10)
        }
      },
      yAxis: {
        type: 'value',
        name: 'Magnitude (dB re 1.0g)',
        nameTextStyle: { color: '#64748b', fontSize: 11 },
        splitLine: { lineStyle: { color: '#141a24' } },
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, formatter: '{value} dB' }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        }
      ],
      series
    };
  }, [fftResult, baselineFftResult, fftCompareMode, currentCh, selectedMicroWindow]);

  return (
    <div className="space-y-4 rounded-xl border border-railway-border/80 bg-railway-surface/60 p-5 backdrop-blur shadow-xl">
      {/* Title & Provenance */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-railway-border/70 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono font-extrabold text-sm sm:text-base uppercase tracking-wider text-white">
              Frequency-Domain Intelligence (Cooley-Tukey Radix-2 FFT)
            </span>
            <DataIntegrityBadge type="MEASURED" size="sm" />
          </div>
          <p className="text-xs text-railway-muted font-sans mt-0.5">
            Single-sided discrete Fourier amplitude spectrum (0 to 1,000 Hz Nyquist, dB re 1.0g) with windowed peak normalization
          </p>
        </div>

        {/* Normal vs Anomaly Comparative Mode Toggle */}
        <button
          onClick={() => setFFTConfig({ compareMode: !fftCompareMode })}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border font-mono text-xs font-bold transition-all ${
            fftCompareMode
              ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/50 shadow-md shadow-emerald-500/10'
              : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
          }`}
        >
          <Layers className="h-3.5 w-3.5" />
          {fftCompareMode ? 'OVERLAY BASELINE ACTIVE' : 'COMPARE WITH NOMINAL TRACK'}
        </button>
      </div>

      {/* Spectral Metrics Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
        {/* Dominant Frequency */}
        <div className="p-3 rounded-lg bg-railway-darkest/80 border border-railway-border/70 space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block flex items-center gap-1">
            <TrendingUp className="h-3 w-3 text-hud-cyan" /> Dominant Frequency
          </span>
          <div className="text-hud-cyan font-bold text-lg">
            {fftResult ? `${fftResult.dominantFrequency.toFixed(1)} Hz` : '---'}
          </div>
          <span className="text-[10px] text-railway-muted block">Peak Energy Concentration</span>
        </div>

        {/* Spectral Centroid */}
        <div className="p-3 rounded-lg bg-railway-darkest/80 border border-railway-border/70 space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block flex items-center gap-1">
            <Activity className="h-3 w-3 text-hud-amber" /> Spectral Centroid
          </span>
          <div className="text-hud-amber font-bold text-lg">
            {fftResult ? `${fftResult.spectralCentroid.toFixed(1)} Hz` : '---'}
          </div>
          <span className="text-[10px] text-railway-muted block">Center of Spectral Mass</span>
        </div>

        {/* Peak Amplitude */}
        <div className="p-3 rounded-lg bg-railway-darkest/80 border border-railway-border/70 space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block flex items-center gap-1">
            <BarChart2 className="h-3 w-3 text-white" /> Peak Amplitude
          </span>
          <div className="text-white font-bold text-lg">
            {fftResult ? `${fftResult.peakValue.toFixed(3)} g` : '---'}
          </div>
          <span className="text-[10px] text-railway-muted block">FFT Max Bin Amplitude</span>
        </div>

        {/* Nyquist Frequency */}
        <div className="p-3 rounded-lg bg-railway-darkest/80 border border-railway-border/70 space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block flex items-center gap-1">
            <CheckCircle2 className="h-3 w-3 text-hud-emerald" /> Nyquist Bandwidth
          </span>
          <div className="text-hud-emerald font-bold text-lg">
            1,000.0 Hz
          </div>
          <span className="text-[10px] text-railway-muted block">fs / 2 = 2,000 Hz / 2</span>
        </div>
      </div>

      {/* FFT Configuration Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-lg bg-railway-darkest/80 border border-railway-border/70 text-xs font-mono">
        {/* FFT Channel Selector */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-railway-muted text-[10px] uppercase mr-1">Channel:</span>
          {channels.map((ch) => (
            <button
              key={ch.id}
              onClick={() => setFFTConfig({ channel: ch.id })}
              className={`px-2 py-1 rounded text-[11px] font-bold border transition-colors ${
                fftChannel === ch.id
                  ? 'border-hud-cyan text-railway-darkest font-extrabold'
                  : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
              }`}
              style={{
                backgroundColor: fftChannel === ch.id ? ch.color : undefined
              }}
            >
              CH {ch.id}
            </button>
          ))}
        </div>

        {/* Window Size Selector */}
        <div className="flex items-center gap-1.5">
          <span className="text-railway-muted text-[10px] uppercase">N (Bins):</span>
          {[512, 1024, 2048].map((size) => (
            <button
              key={size}
              onClick={() => setFFTConfig({ windowSize: size })}
              className={`px-2 py-0.5 rounded border text-[10px] transition-colors ${
                fftWindowSize === size
                  ? 'bg-hud-cyan text-railway-darkest font-bold border-hud-cyan'
                  : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
              }`}
            >
              {size}
            </button>
          ))}
        </div>

        {/* Window Function Type */}
        <div className="flex items-center gap-1.5">
          <span className="text-railway-muted text-[10px] uppercase">Window:</span>
          {(['hanning', 'hamming', 'rectangular'] as WindowFunctionType[]).map((wType) => (
            <button
              key={wType}
              onClick={() => setFFTConfig({ windowType: wType })}
              className={`px-2 py-0.5 rounded border text-[10px] uppercase transition-colors ${
                fftWindowType === wType
                  ? 'bg-hud-cyan text-railway-darkest font-bold border-hud-cyan'
                  : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
              }`}
            >
              {wType}
            </button>
          ))}
        </div>
      </div>

      {/* Energy Band Distribution Bars */}
      {fftResult && (
        <div className="p-3 rounded-lg bg-railway-darkest/90 border border-railway-border/80 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-railway-muted">
            <span className="uppercase text-white font-bold flex items-center gap-1.5">
              <BarChart2 className="h-3.5 w-3.5 text-hud-cyan" />
              Spectral Band Energy Allocation
            </span>
            <span className="text-[11px] text-railway-muted">Physics Interpretation</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs font-mono">
            {/* 0 - 50 Hz */}
            <div className="p-2 rounded bg-railway-surface/70 border border-railway-border/60 space-y-1">
              <div className="flex justify-between text-[11px]">
                <span className="text-hud-cyan font-bold">0 - 50 Hz</span>
                <span className="text-white font-bold">{fftResult.bandEnergies.low}%</span>
              </div>
              <div className="w-full bg-railway-border h-1.5 rounded-full overflow-hidden">
                <div
                  style={{ width: `${fftResult.bandEnergies.low}%` }}
                  className="h-full bg-hud-cyan rounded-full transition-all"
                />
              </div>
              <span className="text-[10px] text-railway-muted block">Bogie Pitch & Hunting</span>
            </div>

            {/* 50 - 200 Hz */}
            <div className="p-2 rounded bg-railway-surface/70 border border-railway-border/60 space-y-1">
              <div className="flex justify-between text-[11px]">
                <span className="text-hud-amber font-bold">50 - 200 Hz</span>
                <span className="text-white font-bold">{fftResult.bandEnergies.mid}%</span>
              </div>
              <div className="w-full bg-railway-border h-1.5 rounded-full overflow-hidden">
                <div
                  style={{ width: `${fftResult.bandEnergies.mid}%` }}
                  className="h-full bg-hud-amber rounded-full transition-all"
                />
              </div>
              <span className="text-[10px] text-railway-muted block">Track Corrugation / Sleepers</span>
            </div>

            {/* 200 - 500 Hz */}
            <div className="p-2 rounded bg-railway-surface/70 border border-railway-border/60 space-y-1">
              <div className="flex justify-between text-[11px]">
                <span className="text-rose-400 font-bold">200 - 500 Hz</span>
                <span className="text-white font-bold">{fftResult.bandEnergies.high}%</span>
              </div>
              <div className="w-full bg-railway-border h-1.5 rounded-full overflow-hidden">
                <div
                  style={{ width: `${fftResult.bandEnergies.high}%` }}
                  className="h-full bg-rose-400 rounded-full transition-all"
                />
              </div>
              <span className="text-[10px] text-railway-muted block">Axle Box Wheel Impacts</span>
            </div>

            {/* 500 - 1000 Hz */}
            <div className="p-2 rounded bg-railway-surface/70 border border-railway-border/60 space-y-1">
              <div className="flex justify-between text-[11px]">
                <span className="text-purple-400 font-bold">500 - 1000 Hz</span>
                <span className="text-white font-bold">{fftResult.bandEnergies.ultra}%</span>
              </div>
              <div className="w-full bg-railway-border h-1.5 rounded-full overflow-hidden">
                <div
                  style={{ width: `${fftResult.bandEnergies.ultra}%` }}
                  className="h-full bg-purple-400 rounded-full transition-all"
                />
              </div>
              <span className="text-[10px] text-railway-muted block">Structural Shock Ring</span>
            </div>
          </div>
        </div>
      )}

      {/* Main FFT Chart Display */}
      <div className="rounded-lg border border-railway-border bg-[#07080a] overflow-hidden">
        <ReactECharts
          option={chartOption}
          style={{ height: '320px', width: '100%' }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>

      {/* Comparative Forensic Findings */}
      {fftCompareMode && baselineFftResult && fftResult && (
        <div className="p-3 rounded-lg bg-railway-darkest/80 border border-emerald-500/30 flex items-start gap-2.5 text-xs font-mono">
          <Info className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold text-emerald-400 uppercase">
              Forensic Spectral Comparison: Defect Impact vs. Nominal Tangent Track
            </span>
            <p className="text-[11px] text-railway-muted font-sans leading-relaxed">
              When traversing defect site <strong>{selectedMicroWindow?.title || 'anomalous zone'}</strong>, high-frequency energy in the{' '}
              <strong className="text-rose-400">200–500 Hz axle impact band</strong> expands from{' '}
              <span className="text-emerald-400">{baselineFftResult.bandEnergies.high}%</span> on smooth rail to{' '}
              <span className="text-rose-400 font-bold">{fftResult.bandEnergies.high}%</span> at the joint gap, shifting the spectral centroid from{' '}
              <span className="text-emerald-400">{baselineFftResult.spectralCentroid.toFixed(1)} Hz</span> to{' '}
              <span className="text-hud-cyan font-bold">{fftResult.spectralCentroid.toFixed(1)} Hz</span>.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
