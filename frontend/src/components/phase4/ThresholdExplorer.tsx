import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { ThresholdExplorerModel, ThresholdStepMetrics } from '../../types';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import {
  Sliders,
  AlertTriangle
} from 'lucide-react';

interface ThresholdExplorerProps {
  explorerData: Record<string, ThresholdExplorerModel>;
}

export const ThresholdExplorer: React.FC<ThresholdExplorerProps> = ({ explorerData }) => {
  const [selectedModelKey, setSelectedModelKey] = useState<string>('cnn');

  const activeModel = explorerData[selectedModelKey] || explorerData.cnn;

  // Initialize threshold to calibrated validation threshold
  const [interactiveThreshold, setInteractiveThreshold] = useState<number>(() => {
    return activeModel?.val_calibrated_threshold || 0.5;
  });

  // When model changes, update interactive threshold to that model's calibrated threshold
  const handleModelChange = (key: string) => {
    setSelectedModelKey(key);
    const m = explorerData[key];
    if (m) {
      setInteractiveThreshold(m.val_calibrated_threshold);
    }
  };

  // Find the closest metrics step for the current threshold
  const currentMetrics: ThresholdStepMetrics = useMemo(() => {
    if (!activeModel || !activeModel.threshold_sweep || activeModel.threshold_sweep.length === 0) {
      return {
        threshold: interactiveThreshold,
        tp: 0,
        fp: 0,
        tn: 0,
        fn: 0,
        accuracy: 0,
        precision: 0,
        recall: 0,
        specificity: 0,
        f1: 0,
        balanced_accuracy: 0,
        mcc: 0
      };
    }

    let minDiff = Infinity;
    let closest = activeModel.threshold_sweep[0];
    for (const step of activeModel.threshold_sweep) {
      const diff = Math.abs(step.threshold - interactiveThreshold);
      if (diff < minDiff) {
        minDiff = diff;
        closest = step;
      }
    }
    return closest;
  }, [activeModel, interactiveThreshold]);

  const total = currentMetrics.tp + currentMetrics.fp + currentMetrics.tn + currentMetrics.fn;

  // Interactive Operating Point Curve (PR Curve with current point)
  const interactiveChartOption = useMemo(() => {
    if (!activeModel || !activeModel.pr_curve) return {};

    const prData = activeModel.pr_curve.map((pt) => [pt.recall, pt.precision]);

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'Dynamic Operating Point on Precision-Recall Curve',
        subtext: `Interactive Threshold: ${interactiveThreshold.toFixed(4)} • Precision: ${(currentMetrics.precision * 100).toFixed(1)}% • Recall: ${(currentMetrics.recall * 100).toFixed(1)}%`,
        textStyle: { color: '#f8fafc', fontSize: 13, fontFamily: 'monospace' },
        subtextStyle: { color: '#94a3b8', fontSize: 11, fontFamily: 'monospace' },
        top: 0,
        left: 0
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#08090c',
        borderColor: '#06b6d4',
        textStyle: { color: '#f8fafc', fontFamily: 'monospace', fontSize: 11 }
      },
      grid: { left: 45, right: 25, top: 55, bottom: 35 },
      xAxis: {
        type: 'value',
        min: 0,
        max: 1.0,
        name: 'Recall',
        nameLocation: 'middle',
        nameGap: 20,
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontFamily: 'monospace', fontSize: 10 }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1.0,
        name: 'Precision',
        nameLocation: 'middle',
        nameGap: 30,
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontFamily: 'monospace', fontSize: 10 }
      },
      series: [
        {
          name: 'PR Curve',
          type: 'line',
          smooth: true,
          data: prData,
          lineStyle: { color: '#06b6d4', width: 2.2 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(6, 182, 212, 0.35)' },
                { offset: 1, color: 'rgba(6, 182, 212, 0.02)' }
              ]
            }
          },
          markPoint: {
            data: [
              {
                name: 'Current Threshold',
                coord: [currentMetrics.recall, currentMetrics.precision],
                symbol: 'circle',
                symbolSize: 14,
                itemStyle: { color: '#ef4444', borderColor: '#fff', borderWidth: 2 },
                label: {
                  formatter: `T=${interactiveThreshold.toFixed(2)}`,
                  position: 'top',
                  color: '#fff',
                  fontFamily: 'monospace',
                  fontSize: 10,
                  fontWeight: 'bold'
                }
              },
              {
                name: 'Calibrated Threshold',
                coord: [
                  activeModel.threshold_sweep.find((s) => Math.abs(s.threshold - activeModel.val_calibrated_threshold) < 0.02)?.recall || 0.89,
                  activeModel.threshold_sweep.find((s) => Math.abs(s.threshold - activeModel.val_calibrated_threshold) < 0.02)?.precision || 0.19
                ],
                symbol: 'triangle',
                symbolSize: 14,
                itemStyle: { color: '#f59e0b' },
                label: {
                  formatter: 'Calibrated (Val)',
                  position: 'bottom',
                  color: '#f59e0b',
                  fontFamily: 'monospace',
                  fontSize: 9
                }
              }
            ]
          }
        }
      ]
    };
  }, [activeModel, interactiveThreshold, currentMetrics]);

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Header and Controls Card */}
      <div className="p-4 rounded-xl bg-railway-surface/50 border border-railway-border/80 backdrop-blur space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Sliders className="h-4 w-4 text-hud-cyan" />
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Interactive Decision Threshold Explorer
            </h2>
            <DataIntegrityBadge type="ML_PREDICTION" size="sm" />
          </div>

          {/* Model Switcher */}
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-railway-muted">MODEL:</span>
            <div className="flex items-center gap-1">
              {['cnn', 'rf', 'tcn'].map((key) => (
                <button
                  key={key}
                  onClick={() => handleModelChange(key)}
                  className={`px-3 py-1 rounded text-xs font-bold transition-all ${
                    selectedModelKey === key
                      ? 'bg-hud-cyan text-railway-darkest shadow-md'
                      : 'bg-railway-darkest/80 text-railway-muted hover:text-white border border-railway-border/70'
                  }`}
                >
                  {key.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Disclaimer Callout */}
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[11px] font-sans flex items-start gap-2.5">
          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <b>Scientific Protocol Integrity Notice:</b> Decision thresholds must be calibrated exclusively on validation
            folds (Max-F1 or ROC tangent), never on the test partition. Adjusting this slider recalculates real-time classification
            metrics from stored model prediction probabilities to illustrate precision-recall trade-offs. It does <b>NOT</b> constitute
            an independently validated operating point.
          </div>
        </div>

        {/* Slider Controls */}
        <div className="p-3.5 rounded-lg bg-railway-darkest/80 border border-railway-border/70 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-3">
              <span className="text-white font-bold text-sm">
                Threshold: <span className="text-hud-cyan font-mono text-base">{interactiveThreshold.toFixed(4)}</span>
              </span>
              <span className="text-railway-muted text-[11px]">
                (Calibrated Val: <b className="text-amber-400">{activeModel.val_calibrated_threshold.toFixed(4)}</b> • Default: <b className="text-slate-400">{activeModel.default_threshold.toFixed(2)}</b>)
              </span>
            </div>

            {/* Quick Preset Buttons */}
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setInteractiveThreshold(activeModel.val_calibrated_threshold)}
                className="px-2.5 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 border border-amber-500/40 text-[10px] font-bold transition-all"
              >
                Reset to Calibrated (Val)
              </button>
              <button
                onClick={() => setInteractiveThreshold(0.5)}
                className="px-2.5 py-1 rounded bg-railway-surface/70 hover:bg-railway-surface text-railway-text border border-railway-border text-[10px] font-bold transition-all"
              >
                Reset to Default (0.50)
              </button>
            </div>
          </div>

          {/* Range Input Slider */}
          <input
            type="range"
            min="0.01"
            max="0.99"
            step="0.005"
            value={interactiveThreshold}
            onChange={(e) => setInteractiveThreshold(parseFloat(e.target.value))}
            className="w-full accent-hud-cyan h-2 bg-railway-surface rounded-lg cursor-pointer"
          />

          <div className="flex justify-between text-[10px] text-railway-muted font-mono">
            <span>0.01 (High Sensitivity / Many False Alarms)</span>
            <span>0.50 (Nominal Default)</span>
            <span>0.99 (High Specificity / Many Missed Defects)</span>
          </div>
        </div>
      </div>

      {/* Recalculated Confusion Matrix & Metrics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Dynamic Confusion Matrix */}
        <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">
              Recalculated Confusion Matrix @ T = {interactiveThreshold.toFixed(4)}
            </h3>
            <span className="text-[10px] text-railway-muted">{total} Test Samples</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <div className="p-3.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40">
              <span className="text-[10px] text-emerald-400 font-bold block">TRUE NEGATIVE (TN)</span>
              <span className="text-xl font-bold text-white">{currentMetrics.tn}</span>
              <span className="text-[10px] text-emerald-300/70 block">
                {total > 0 ? `${((currentMetrics.tn / total) * 100).toFixed(1)}%` : '0%'}
              </span>
            </div>

            <div className="p-3.5 rounded-lg bg-rose-950/40 border border-rose-500/40">
              <span className="text-[10px] text-rose-400 font-bold block">FALSE POSITIVE (FP)</span>
              <span className="text-xl font-bold text-rose-300">{currentMetrics.fp}</span>
              <span className="text-[10px] text-rose-300/70 block">
                {total > 0 ? `${((currentMetrics.fp / total) * 100).toFixed(1)}%` : '0%'}
              </span>
            </div>

            <div className="p-3.5 rounded-lg bg-amber-950/40 border border-amber-500/40">
              <span className="text-[10px] text-amber-400 font-bold block">FALSE NEGATIVE (FN)</span>
              <span className="text-xl font-bold text-amber-300">{currentMetrics.fn}</span>
              <span className="text-[10px] text-amber-300/70 block">
                {total > 0 ? `${((currentMetrics.fn / total) * 100).toFixed(1)}%` : '0%'}
              </span>
            </div>

            <div className="p-3.5 rounded-lg bg-cyan-950/40 border border-hud-cyan/50">
              <span className="text-[10px] text-hud-cyan font-bold block">TRUE POSITIVE (TP)</span>
              <span className="text-xl font-bold text-white">{currentMetrics.tp}</span>
              <span className="text-[10px] text-hud-cyan/70 block">
                {total > 0 ? `${((currentMetrics.tp / total) * 100).toFixed(1)}%` : '0%'}
              </span>
            </div>
          </div>

          {/* Dynamic 8-Metric Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-railway-border/60">
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">PRECISION</span>
              <span className="text-sm font-bold text-hud-cyan">{currentMetrics.precision.toFixed(4)}</span>
            </div>
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">RECALL</span>
              <span className="text-sm font-bold text-emerald-400">{currentMetrics.recall.toFixed(4)}</span>
            </div>
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">F1 SCORE</span>
              <span className="text-sm font-bold text-white">{currentMetrics.f1.toFixed(4)}</span>
            </div>
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">SPECIFICITY</span>
              <span className="text-sm font-bold text-white">{currentMetrics.specificity.toFixed(4)}</span>
            </div>
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">ACCURACY</span>
              <span className="text-sm font-bold text-white">{currentMetrics.accuracy.toFixed(4)}</span>
            </div>
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">BALANCED ACC</span>
              <span className="text-sm font-bold text-white">{currentMetrics.balanced_accuracy.toFixed(4)}</span>
            </div>
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">MCC</span>
              <span className="text-sm font-bold text-white">{currentMetrics.mcc.toFixed(4)}</span>
            </div>
            <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[9px] text-railway-muted block">PREVALENCE</span>
              <span className="text-sm font-bold text-white">
                {total > 0 ? `${(((currentMetrics.tp + currentMetrics.fn) / total) * 100).toFixed(1)}%` : '0%'}
              </span>
            </div>
          </div>
        </div>

        {/* Dynamic Curve Visualizer */}
        <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur">
          <ReactECharts option={interactiveChartOption} style={{ height: '360px', width: '100%' }} />
        </div>
      </div>
    </div>
  );
};
