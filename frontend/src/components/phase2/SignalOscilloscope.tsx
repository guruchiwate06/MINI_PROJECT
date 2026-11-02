import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Maximize2,
  ZoomIn,
  Layers,
  Sliders,
  CheckCircle2,
  AlertOctagon
} from 'lucide-react';
import { useSignalStore } from '../../store/signalStore';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

export const SignalOscilloscope: React.FC = () => {
  const {
    overview,
    microWindowsFile,
    channels,
    activeChannel,
    activeMicroWindowId,
    selectedMicroWindow,
    currentTime,
    currentRunMeta,
    setActiveChannel,
    selectMicroWindow
  } = useSignalStore();

  // Oscilloscope display mode: 'overview' (decimated full run) vs 'microwindow' (raw 2,000 Hz)
  const [viewMode, setViewMode] = useState<'overview' | 'microwindow'>('microwindow');
  const [amplitudeScale, setAmplitudeScale] = useState<'auto' | '0.5' | '1.0' | '2.5' | '5.0'>('auto');
  const [displayAllChannels, setDisplayAllChannels] = useState<boolean>(false);

  // Active channel definition
  const chDef = channels[activeChannel - 1] || channels[1] || {
    id: 2,
    name: 'Channel 2',
    label: 'Front Wheel Axle Box',
    location: 'Unsprung Axle Box 1',
    color: '#f59e0b',
    role: 'Direct Rail Impact Interface',
    filtering: 'Unsprung wheel-rail contact'
  };

  // ECharts Option Construction
  const chartOption = useMemo(() => {
    // 1. Full-Resolution 2,000 Hz Micro-Window Mode
    if (viewMode === 'microwindow') {
      const win = selectedMicroWindow || microWindowsFile?.windows[0];
      if (!win || !win.channels_data) {
        return {};
      }

      const sampleCount = win.sample_count || 2000;
      const dt = 1 / (win.sampling_rate_hz || 2000); // 0.0005s = 0.5ms

      // Time axis points (0 to 1.000s)
      const times = new Array(sampleCount);
      for (let i = 0; i < sampleCount; i++) {
        times[i] = (i * dt).toFixed(4);
      }

      const series: any[] = [];

      if (displayAllChannels) {
        // Multi-channel overlay
        channels.forEach((ch, chIdx) => {
          if (!win.channels_data[chIdx]) return;
          series.push({
            name: `CH ${ch.id}: ${ch.label}`,
            type: 'line',
            showSymbol: false,
            sampling: 'lttb',
            lineStyle: {
              width: ch.id === activeChannel ? 2 : 1,
              color: ch.color,
              opacity: ch.id === activeChannel ? 1.0 : 0.4
            },
            data: win.channels_data[chIdx]
          });
        });
      } else {
        // Single active channel solo
        const chIdx = activeChannel - 1;
        const data = win.channels_data[chIdx] || [];
        series.push({
          name: `CH ${chDef.id}: ${chDef.label}`,
          type: 'line',
          showSymbol: false,
          lineStyle: {
            width: 1.5,
            color: chDef.color
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: `${chDef.color}33` },
                { offset: 1, color: `${chDef.color}05` }
              ]
            }
          },
          markPoint: {
            data: [
              { type: 'max', name: 'Peak Acceleration (g)' },
              { type: 'min', name: 'Min Acceleration (g)' }
            ],
            label: {
              color: '#ffffff',
              fontSize: 10,
              formatter: '{c} g'
            }
          },
          markLine: {
            silent: true,
            lineStyle: { color: 'rgba(255, 255, 255, 0.2)', type: 'dashed' },
            data: [{ yAxis: 0 }]
          },
          data
        });
      }

      let yMin: number | undefined = undefined;
      let yMax: number | undefined = undefined;
      if (amplitudeScale !== 'auto') {
        const val = parseFloat(amplitudeScale);
        yMin = -val;
        yMax = val;
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
            const tRel = (idx * dt).toFixed(4);
            const tAbs = (win.time_start_s + idx * dt).toFixed(4);
            let html = `<div style="font-weight:bold; margin-bottom:4px; color:#38bdf8;">Relative t = ${tRel}s (Run t = ${tAbs}s)</div>`;
            params.forEach((p) => {
              html += `<div style="display:flex; justify-content:space-between; gap:12px;">
                <span style="color:${p.color}">● ${p.seriesName}:</span>
                <span style="font-weight:bold">${Number(p.value).toFixed(3)} g</span>
              </div>`;
            });
            return html;
          }
        },
        grid: {
          top: 30,
          left: 55,
          right: 30,
          bottom: 50
        },
        xAxis: {
          type: 'category',
          data: times,
          name: 'Window Time (seconds @ 2,000 Hz)',
          nameLocation: 'middle',
          nameGap: 30,
          nameTextStyle: { color: '#64748b', fontSize: 11 },
          axisLine: { lineStyle: { color: '#334155' } },
          axisLabel: { color: '#94a3b8', fontSize: 10 }
        },
        yAxis: {
          type: 'value',
          min: yMin,
          max: yMax,
          name: 'Amplitude (g)',
          nameTextStyle: { color: '#64748b', fontSize: 11 },
          splitLine: { lineStyle: { color: '#141a24' } },
          axisLine: { lineStyle: { color: '#334155' } },
          axisLabel: { color: '#94a3b8', fontSize: 10, formatter: '{value} g' }
        },
        dataZoom: [
          {
            type: 'inside',
            start: 0,
            end: 100
          },
          {
            type: 'slider',
            show: true,
            bottom: 6,
            height: 18,
            borderColor: '#1e293b',
            fillerColor: 'rgba(6, 182, 212, 0.15)',
            handleStyle: { color: '#06b6d4' },
            textStyle: { color: '#94a3b8', fontSize: 9 }
          }
        ],
        series
      };
    }

    // 2. Full Run Min-Max Envelope Overview Mode
    if (!overview || !overview.points) return {};

    const points = overview.points;
    const times = points.map((p) => p.t.toFixed(1));
    const chIdx = activeChannel - 1;

    // Build min and max series
    const minData = points.map((p) => p.ch[chIdx][0]);
    const maxData = points.map((p) => p.ch[chIdx][1]);

    // Mark areas for ground-truth surveyed anomaly regions
    const markAreas: any[] = [];
    if (currentRunMeta && currentRunMeta.anomalies) {
      currentRunMeta.anomalies.forEach((anom) => {
        markAreas.push([
          {
            name: `S${anom.site_id}: ${anom.type}`,
            xAxis: Math.max(0, anom.time_s - 0.8).toFixed(1),
            itemStyle: {
              color: 'rgba(239, 68, 68, 0.18)'
            },
            label: {
              show: true,
              color: '#f87171',
              fontSize: 9,
              position: 'insideTop'
            }
          },
          {
            xAxis: (anom.time_s + 0.8).toFixed(1)
          }
        ]);
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
          const pt = points[idx];
          return `<div style="font-weight:bold; margin-bottom:4px; color:#38bdf8;">Time: ${pt.t.toFixed(2)}s • Dist: ${pt.d.toFixed(1)}m</div>
            <div style="color:${chDef.color}">Peak Max: <b>${pt.ch[chIdx][1].toFixed(3)} g</b></div>
            <div style="color:${chDef.color}">Peak Min: <b>${pt.ch[chIdx][0].toFixed(3)} g</b></div>
            ${pt.a === 1 ? '<div style="color:#ef4444; font-weight:bold; margin-top:2px;">[VERIFIED GROUND-TRUTH ANOMALY ZONE]</div>' : ''}`;
        }
      },
      grid: {
        top: 30,
        left: 55,
        right: 30,
        bottom: 50
      },
      xAxis: {
        type: 'category',
        data: times,
        name: 'Transit Time (seconds)',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: { color: '#64748b', fontSize: 11 },
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 }
      },
      yAxis: {
        type: 'value',
        name: 'Amplitude Envelope (g)',
        nameTextStyle: { color: '#64748b', fontSize: 11 },
        splitLine: { lineStyle: { color: '#141a24' } },
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, formatter: '{value} g' }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          type: 'slider',
          show: true,
          bottom: 6,
          height: 18,
          borderColor: '#1e293b',
          fillerColor: 'rgba(6, 182, 212, 0.15)',
          handleStyle: { color: '#06b6d4' },
          textStyle: { color: '#94a3b8', fontSize: 9 }
        }
      ],
      series: [
        {
          name: 'Peak Positive Envelope',
          type: 'line',
          showSymbol: false,
          sampling: 'lttb',
          lineStyle: { width: 1.2, color: chDef.color },
          markArea: {
            data: markAreas
          },
          markLine: {
            silent: true,
            lineStyle: { color: '#ffffff', type: 'solid', width: 1.5 },
            data: [{ xAxis: currentTime.toFixed(1), name: 'Playhead' }]
          },
          data: maxData
        },
        {
          name: 'Peak Negative Envelope',
          type: 'line',
          showSymbol: false,
          sampling: 'lttb',
          lineStyle: { width: 1.2, color: chDef.color },
          data: minData
        }
      ]
    };
  }, [
    viewMode,
    selectedMicroWindow,
    microWindowsFile,
    overview,
    activeChannel,
    displayAllChannels,
    amplitudeScale,
    currentTime,
    chDef,
    channels,
    currentRunMeta
  ]);

  return (
    <div className="space-y-4 rounded-xl border border-railway-border/80 bg-railway-surface/60 p-5 backdrop-blur shadow-xl">
      {/* Header and Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-railway-border/70 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono font-extrabold text-sm sm:text-base uppercase tracking-wider text-white">
              Interactive Time-Domain Oscilloscope
            </span>
            <DataIntegrityBadge type="MEASURED" size="sm" />
            <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
          </div>
          <p className="text-xs text-railway-muted font-sans mt-0.5">
            Multiresolution zoom: Decimated peak-preserving min-max overview vs. full-resolution 2,000 Hz micro-windows
          </p>
        </div>

        {/* Multiresolution Mode Switcher */}
        <div className="flex items-center p-1 rounded-lg bg-railway-darkest border border-railway-border text-xs font-mono">
          <button
            onClick={() => setViewMode('microwindow')}
            className={`px-3 py-1.5 rounded-md transition-all font-bold ${
              viewMode === 'microwindow'
                ? 'bg-hud-cyan text-railway-darkest shadow-md'
                : 'text-railway-muted hover:text-white'
            }`}
          >
            2,000 Hz MICRO-WINDOW (RAW SAMPLES)
          </button>
          <button
            onClick={() => setViewMode('overview')}
            className={`px-3 py-1.5 rounded-md transition-all font-bold ${
              viewMode === 'overview'
                ? 'bg-hud-cyan text-railway-darkest shadow-md'
                : 'text-railway-muted hover:text-white'
            }`}
          >
            FULL TRANSIT ENVELOPE (DECIMATED)
          </button>
        </div>
      </div>

      {/* Control Bar: Channel Selection, Scale, and Micro-Window Selector */}
      <div className="space-y-3">
        {/* Micro-Window Selector (Shown in microwindow mode) */}
        {viewMode === 'microwindow' && microWindowsFile && (
          <div className="p-3 rounded-lg bg-railway-darkest/90 border border-railway-border/80 space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-railway-muted">
              <span className="flex items-center gap-1.5 uppercase font-bold text-white">
                <Maximize2 className="h-3.5 w-3.5 text-hud-cyan" />
                Select 1.0-Second Verified Micro-Window (2,000 Raw Discrete Samples)
              </span>
              <span className="text-[11px] text-hud-amber font-mono">
                Actual Measured Accelerometer Data • Zero Fabrication
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
              {microWindowsFile.windows.map((win) => {
                const isSelected = (activeMicroWindowId === win.id) || (!activeMicroWindowId && win.id === 'anomaly_site_1');
                const isNormal = win.type === 'NORMAL';

                return (
                  <button
                    key={win.id}
                    onClick={() => selectMicroWindow(win.id)}
                    className={`p-2 rounded-lg border text-left transition-all font-mono text-xs ${
                      isSelected
                        ? isNormal
                          ? 'border-emerald-500 bg-emerald-500/15 text-white ring-1 ring-emerald-500'
                          : 'border-rose-500 bg-rose-500/15 text-white ring-1 ring-rose-500'
                        : 'border-railway-border/70 bg-railway-elevated/60 text-railway-muted hover:border-railway-border hover:text-white'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-[11px] truncate">
                        {win.site_id ? `Site ${win.site_id}` : 'Baseline'}
                      </span>
                      {isNormal ? (
                        <CheckCircle2 className="h-3 w-3 text-emerald-400 shrink-0" />
                      ) : (
                        <AlertOctagon className="h-3 w-3 text-rose-400 shrink-0" />
                      )}
                    </div>
                    <div className="text-[10px] text-railway-muted truncate mt-0.5">{win.title}</div>
                    <div className="text-[9px] text-hud-cyan mt-1">
                      {win.chainage_m.toFixed(0)}m • t={win.time_start_s.toFixed(1)}s
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Channels & Oscilloscope Adjustment Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-lg bg-railway-darkest/70 border border-railway-border/60 text-xs font-mono">
          {/* Channel Solo Selector */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-railway-muted text-[10px] uppercase mr-1">Channel:</span>
            {channels.map((ch) => (
              <button
                key={ch.id}
                onClick={() => setActiveChannel(ch.id)}
                className={`px-2 py-1 rounded text-[11px] font-bold border transition-colors ${
                  activeChannel === ch.id
                    ? 'border-hud-cyan text-railway-darkest font-extrabold'
                    : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
                }`}
                style={{
                  backgroundColor: activeChannel === ch.id ? ch.color : undefined
                }}
              >
                CH {ch.id}
              </button>
            ))}

            {viewMode === 'microwindow' && (
              <button
                onClick={() => setDisplayAllChannels(!displayAllChannels)}
                className={`ml-2 px-2.5 py-1 rounded border text-[11px] font-bold transition-colors flex items-center gap-1 ${
                  displayAllChannels
                    ? 'bg-hud-amber text-railway-darkest border-hud-amber'
                    : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
                }`}
              >
                <Layers className="h-3 w-3" />
                {displayAllChannels ? 'OVERLAY ALL ACTIVE' : 'OVERLAY ALL'}
              </button>
            )}
          </div>

          {/* Amplitude Range Selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-railway-muted text-[10px] uppercase flex items-center gap-1">
              <Sliders className="h-3 w-3 text-hud-cyan" /> Scale:
            </span>
            {(['auto', '0.5', '1.0', '2.5', '5.0'] as const).map((sc) => (
              <button
                key={sc}
                onClick={() => setAmplitudeScale(sc)}
                className={`px-2 py-0.5 rounded border text-[10px] transition-colors ${
                  amplitudeScale === sc
                    ? 'bg-hud-cyan text-railway-darkest font-bold border-hud-cyan'
                    : 'bg-railway-elevated text-railway-muted border-railway-border hover:text-white'
                }`}
              >
                {sc === 'auto' ? 'AUTO' : `±${sc}g`}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main ECharts Canvas Display */}
      <div className="relative rounded-lg border border-railway-border bg-[#07080a] overflow-hidden">
        {/* Active Channel Legend Badge */}
        <div className="absolute top-2 left-3 z-10 flex items-center gap-2 pointer-events-none text-xs font-mono bg-railway-darkest/90 px-2 py-1 rounded border border-railway-border/60">
          <span style={{ backgroundColor: chDef.color }} className="h-2 w-2 rounded-full" />
          <span className="font-bold text-white">
            CH {chDef.id}: {chDef.label}
          </span>
          <span className="text-[10px] text-railway-muted">({chDef.filtering})</span>
        </div>

        {/* ECharts Rendering Component */}
        <ReactECharts
          option={chartOption}
          style={{ height: '360px', width: '100%' }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>

      {/* Forensic Inspection Footer Note */}
      <div className="flex flex-wrap items-center justify-between text-[11px] font-mono text-railway-muted px-1">
        <span className="flex items-center gap-1">
          <ZoomIn className="h-3 w-3 text-hud-cyan" />
          Scroll or drag on chart or slider to zoom & pan microsecond transients
        </span>
        <span className="text-hud-cyan">
          {viewMode === 'microwindow' ? '2,000 Hz Discrete Accelerometer Waveform' : '20 Hz Peak-Preserving Min-Max Envelope'}
        </span>
      </div>
    </div>
  );
};
