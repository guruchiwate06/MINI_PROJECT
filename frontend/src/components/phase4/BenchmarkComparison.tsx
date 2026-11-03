import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { EvaluatedExperiment, ThresholdExplorerModel } from '../../types';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import {
  BarChart3,
  TrendingUp,
  Layers,
  Info
} from 'lucide-react';

interface BenchmarkComparisonProps {
  experiments: EvaluatedExperiment[];
  explorerData: Record<string, ThresholdExplorerModel>;
  selectedExperimentId: string;
  onSelectExperiment: (id: string) => void;
}

export const BenchmarkComparison: React.FC<BenchmarkComparisonProps> = ({
  experiments,
  explorerData,
  selectedExperimentId,
  onSelectExperiment
}) => {
  const [activeChartTab, setActiveChartTab] = useState<'pr-roc' | 'f1-bar' | 'confusion'>('pr-roc');

  // Filter out non-test-evaluated models from test comparisons
  const testModels = useMemo(() => {
    return experiments.filter((e) => e.is_test_evaluated !== false && e.method !== 'track_ot_pp');
  }, [experiments]);

  // Currently selected experiment (default to first test model if active is invalid)
  const activeExp = useMemo(() => {
    return testModels.find((e) => e.id === selectedExperimentId) || testModels[0] || experiments[0];
  }, [testModels, selectedExperimentId, experiments]);

  // Explorer model matching activeExp
  const activeExplorer = useMemo(() => {
    if (!explorerData) return null;
    if (activeExp.method === 'cnn') return explorerData.cnn;
    if (activeExp.method === 'rf') return explorerData.rf;
    if (activeExp.method === 'tcn' || activeExp.method === 'ssl_tcn') return explorerData.tcn;
    return explorerData.cnn;
  }, [explorerData, activeExp]);

  // 1. Precision-Recall Curve Option
  const prCurveOption = useMemo(() => {
    if (!activeExplorer || !activeExplorer.pr_curve) return {};
    const prData = activeExplorer.pr_curve.map((pt) => [pt.recall, pt.precision]);

    return {
      backgroundColor: 'transparent',
      title: {
        text: `${activeExplorer.model_name} — Precision-Recall Curve`,
        subtext: `Test Partition • Average Precision (AP): ${activeExp.average_precision !== null && activeExp.average_precision !== undefined ? activeExp.average_precision.toFixed(4) : 'UNAVAILABLE'} (Continuous Scores)`,
        textStyle: { color: '#f8fafc', fontSize: 13, fontFamily: 'monospace' },
        subtextStyle: { color: '#94a3b8', fontSize: 11, fontFamily: 'monospace' },
        top: 0,
        left: 0
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#08090c',
        borderColor: '#06b6d4',
        textStyle: { color: '#f8fafc', fontFamily: 'monospace', fontSize: 11 },
        formatter: (params: any) => {
          const pt = params[0].value;
          return `Recall: <b>${pt[0].toFixed(3)}</b><br/>Precision: <b>${pt[1].toFixed(3)}</b>`;
        }
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
          lineStyle: { color: '#06b6d4', width: 2.5 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(6, 182, 212, 0.45)' },
                { offset: 1, color: 'rgba(6, 182, 212, 0.02)' }
              ]
            }
          },
          markPoint: {
            data: [
              {
                coord: [activeExp.recall ?? 0.89, activeExp.precision ?? 0.19],
                symbol: 'pin',
                symbolSize: 32,
                itemStyle: { color: '#f59e0b' },
                label: {
                  formatter: 'Calibrated',
                  fontFamily: 'monospace',
                  fontSize: 9,
                  color: '#000',
                  fontWeight: 'bold'
                }
              }
            ]
          }
        }
      ]
    };
  }, [activeExplorer, activeExp]);

  // 2. ROC Curve Option
  const rocCurveOption = useMemo(() => {
    if (!activeExplorer || !activeExplorer.roc_curve) return {};
    const rocData = activeExplorer.roc_curve.map((pt) => [pt.fpr, pt.tpr]);

    return {
      backgroundColor: 'transparent',
      title: {
        text: `${activeExplorer.model_name} — Receiver Operating Characteristic (ROC)`,
        subtext: `ROC-AUC: ${activeExp.roc_auc !== null && activeExp.roc_auc !== undefined ? activeExp.roc_auc.toFixed(4) : 'UNAVAILABLE'} • Continuous Probabilities`,
        textStyle: { color: '#f8fafc', fontSize: 13, fontFamily: 'monospace' },
        subtextStyle: { color: '#94a3b8', fontSize: 11, fontFamily: 'monospace' },
        top: 0,
        left: 0
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#08090c',
        borderColor: '#10b981',
        textStyle: { color: '#f8fafc', fontFamily: 'monospace', fontSize: 11 },
        formatter: (params: any) => {
          const pt = params[0].value;
          return `FPR: <b>${pt[0].toFixed(3)}</b><br/>TPR (Recall): <b>${pt[1].toFixed(3)}</b>`;
        }
      },
      grid: { left: 45, right: 25, top: 55, bottom: 35 },
      xAxis: {
        type: 'value',
        min: 0,
        max: 1.0,
        name: 'False Positive Rate (FPR)',
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
        name: 'True Positive Rate (TPR)',
        nameLocation: 'middle',
        nameGap: 30,
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontFamily: 'monospace', fontSize: 10 }
      },
      series: [
        {
          name: 'ROC Curve',
          type: 'line',
          smooth: true,
          data: rocData,
          lineStyle: { color: '#10b981', width: 2.5 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(16, 185, 129, 0.35)' },
                { offset: 1, color: 'rgba(16, 185, 129, 0.02)' }
              ]
            }
          }
        },
        {
          name: 'Random Chance Baseline',
          type: 'line',
          data: [
            [0, 0],
            [1, 1]
          ],
          lineStyle: { color: '#64748b', type: 'dashed', width: 1.5 },
          symbol: 'none'
        }
      ]
    };
  }, [activeExplorer, activeExp]);

  // 3. F1 Comparison Bar Chart
  const f1BarOption = useMemo(() => {
    // Select top 10 representative test models
    const topModels = testModels
      .filter((e) => e.f1 !== null && e.f1 > 0)
      .sort((a, b) => (b.f1 || 0) - (a.f1 || 0))
      .slice(0, 10);

    const categories = topModels.map((m) => m.id);
    const f1Values = topModels.map((m) => m.f1);
    const apValues = topModels.map((m) => m.average_precision || 0);

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'Top 10 Test Models by F1 Score on Spatial Holdout Partitions',
        subtext: 'F1 Score (Threshold-Calibrated) vs Average Precision (AP, Continuous Step-Curve)',
        textStyle: { color: '#f8fafc', fontSize: 13, fontFamily: 'monospace' },
        subtextStyle: { color: '#94a3b8', fontSize: 11, fontFamily: 'monospace' },
        top: 0,
        left: 0
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: '#08090c',
        borderColor: '#06b6d4',
        textStyle: { color: '#f8fafc', fontFamily: 'monospace', fontSize: 11 }
      },
      legend: {
        data: ['F1 Score', 'Average Precision (AP)'],
        textStyle: { color: '#94a3b8', fontFamily: 'monospace' },
        top: 25,
        right: 10
      },
      grid: { left: 55, right: 25, top: 70, bottom: 55 },
      xAxis: {
        type: 'category',
        data: categories,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: {
          color: '#94a3b8',
          fontFamily: 'monospace',
          fontSize: 9,
          interval: 0,
          rotate: 25
        }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 0.7,
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontFamily: 'monospace', fontSize: 10 }
      },
      series: [
        {
          name: 'F1 Score',
          type: 'bar',
          data: f1Values,
          itemStyle: { color: '#06b6d4', borderRadius: [4, 4, 0, 0] }
        },
        {
          name: 'Average Precision (AP)',
          type: 'bar',
          data: apValues,
          itemStyle: { color: '#f59e0b', borderRadius: [4, 4, 0, 0] }
        }
      ]
    };
  }, [testModels]);

  // Confusion Matrix stats
  const cm = activeExp.confusion_matrix || [
    [Number(activeExp.tn) || 0, Number(activeExp.fp) || 0],
    [Number(activeExp.fn) || 0, Number(activeExp.tp) || 0]
  ];
  const tn = Number(cm[0][0]) || 0;
  const fp = Number(cm[0][1]) || 0;
  const fn = Number(cm[1][0]) || 0;
  const tp = Number(cm[1][1]) || 0;
  const total = tn + fp + fn + tp;

  return (
    <div className="space-y-4">
      {/* Sub-Tabs for Chart View */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur font-mono text-xs">
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setActiveChartTab('pr-roc')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
              activeChartTab === 'pr-roc'
                ? 'bg-hud-cyan text-railway-darkest shadow-md'
                : 'text-railway-muted hover:text-white hover:bg-railway-darkest/60'
            }`}
          >
            <TrendingUp className="h-3.5 w-3.5" />
            <span>PR & ROC CURVES</span>
          </button>

          <button
            onClick={() => setActiveChartTab('f1-bar')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
              activeChartTab === 'f1-bar'
                ? 'bg-hud-cyan text-railway-darkest shadow-md'
                : 'text-railway-muted hover:text-white hover:bg-railway-darkest/60'
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>TOP MODELS (F1 & AP)</span>
          </button>

          <button
            onClick={() => setActiveChartTab('confusion')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
              activeChartTab === 'confusion'
                ? 'bg-hud-cyan text-railway-darkest shadow-md'
                : 'text-railway-muted hover:text-white hover:bg-railway-darkest/60'
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            <span>CONFUSION MATRIX & METRICS</span>
          </button>
        </div>

        {/* Active Model Selector */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-railway-muted">MODEL UNDER TEST:</span>
          <select
            value={activeExp.id}
            onChange={(e) => onSelectExperiment(e.target.value)}
            className="px-2.5 py-1 rounded bg-railway-darkest border border-railway-border text-white text-xs font-mono focus:outline-none focus:border-hud-cyan cursor-pointer"
          >
            {testModels.map((e) => (
              <option key={e.id} value={e.id}>
                {e.model_name} ({e.id})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Chart Canvas */}
      {activeChartTab === 'pr-roc' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur">
              <ReactECharts option={prCurveOption} style={{ height: '320px', width: '100%' }} />
            </div>
            <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur">
              <ReactECharts option={rocCurveOption} style={{ height: '320px', width: '100%' }} />
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-railway-surface/40 border border-railway-border/70 text-xs font-mono flex items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-railway-muted">
              <Info className="h-4 w-4 text-hud-cyan shrink-0" />
              <span>
                <b>Average Precision (AP) Implementation Note:</b> AP evaluates the area under the PR step curve via exact scikit-learn convention: step-wise precision weighted by recall increments. Unlike trapezoidal PR-AUC, this avoids non-linear interpolation distortion between adjacent recall points.
              </span>
            </div>
            <div className="text-[11px] text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/30 whitespace-nowrap">
              Continuous Probabilities Used
            </div>
          </div>
        </div>
      )}

      {activeChartTab === 'f1-bar' && (
        <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur space-y-3">
          <ReactECharts option={f1BarOption} style={{ height: '340px', width: '100%' }} />
          <div className="p-2.5 rounded bg-railway-darkest/70 border border-railway-border/50 text-[11px] text-railway-muted font-mono flex items-center gap-2">
            <Info className="h-3.5 w-3.5 text-hud-cyan shrink-0" />
            <span>F1 score is evaluated at the frozen validation-calibrated decision threshold. AP is threshold-independent across all continuous operating points.</span>
          </div>
        </div>
      )}

      {activeChartTab === 'confusion' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Confusion Matrix Visualizer */}
          <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur font-mono space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Audited Confusion Matrix ({activeExp.id})
              </h3>
              <DataIntegrityBadge type="ML_PREDICTION" size="sm" />
            </div>

            <p className="text-[11px] text-railway-muted font-sans">
              Evaluated on {total} unpeeking test windows. Threshold calibrated strictly on validation partition: <b>{activeExp.calibrated_threshold?.toFixed(4) ?? 'UNAVAILABLE'}</b>.
            </p>

            <div className="grid grid-cols-2 gap-2 text-center text-xs mt-2">
              {/* True Negative */}
              <div className="p-3.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40 space-y-1">
                <span className="text-[10px] text-emerald-400 font-bold block">TRUE NEGATIVE (TN)</span>
                <span className="text-xl font-bold text-white">{tn}</span>
                <span className="text-[10px] text-emerald-300/70 block">
                  {total > 0 ? `${((tn / total) * 100).toFixed(1)}%` : '0%'}
                </span>
                <span className="text-[9px] text-railway-muted block">Correct Nominal Track</span>
              </div>

              {/* False Positive */}
              <div className="p-3.5 rounded-lg bg-rose-950/40 border border-rose-500/40 space-y-1">
                <span className="text-[10px] text-rose-400 font-bold block">FALSE POSITIVE (FP)</span>
                <span className="text-xl font-bold text-rose-300">{fp}</span>
                <span className="text-[10px] text-rose-300/70 block">
                  {total > 0 ? `${((fp / total) * 100).toFixed(1)}%` : '0%'}
                </span>
                <span className="text-[9px] text-railway-muted block">False Alarm Over Track</span>
              </div>

              {/* False Negative */}
              <div className="p-3.5 rounded-lg bg-amber-950/40 border border-amber-500/40 space-y-1">
                <span className="text-[10px] text-amber-400 font-bold block">FALSE NEGATIVE (FN)</span>
                <span className="text-xl font-bold text-amber-300">{fn}</span>
                <span className="text-[10px] text-amber-300/70 block">
                  {total > 0 ? `${((fn / total) * 100).toFixed(1)}%` : '0%'}
                </span>
                <span className="text-[9px] text-railway-muted block">Missed Physical Defect</span>
              </div>

              {/* True Positive */}
              <div className="p-3.5 rounded-lg bg-cyan-950/40 border border-hud-cyan/50 space-y-1">
                <span className="text-[10px] text-hud-cyan font-bold block">TRUE POSITIVE (TP)</span>
                <span className="text-xl font-bold text-white">{tp}</span>
                <span className="text-[10px] text-hud-cyan/70 block">
                  {total > 0 ? `${((tp / total) * 100).toFixed(1)}%` : '0%'}
                </span>
                <span className="text-[9px] text-railway-muted block">Verified Defect Catch</span>
              </div>
            </div>
          </div>

          {/* Derived Classification Metrics Card */}
          <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur font-mono space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Full 12-Metric Classification Evaluation
              </h3>
              <span className="text-[10px] text-hud-cyan">{activeExp.split_strategy.split('(')[0].trim()}</span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">ACCURACY</span>
                <span className="text-sm font-bold text-white">{activeExp.accuracy !== null && activeExp.accuracy !== undefined ? activeExp.accuracy.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">BALANCED ACCURACY</span>
                <span className="text-sm font-bold text-white">{activeExp.balanced_accuracy !== null && activeExp.balanced_accuracy !== undefined ? activeExp.balanced_accuracy.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">PRECISION (PPV)</span>
                <span className="text-sm font-bold text-white">{activeExp.precision !== null && activeExp.precision !== undefined ? activeExp.precision.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">RECALL / SENSITIVITY</span>
                <span className="text-sm font-bold text-white">{activeExp.recall !== null && activeExp.recall !== undefined ? activeExp.recall.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">SPECIFICITY (TNR)</span>
                <span className="text-sm font-bold text-white">{activeExp.specificity !== null && activeExp.specificity !== undefined ? activeExp.specificity.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">F1 SCORE</span>
                <span className="text-sm font-bold text-hud-cyan">{activeExp.f1 !== null && activeExp.f1 !== undefined ? activeExp.f1.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">MATTHEWS CORR (MCC)</span>
                <span className="text-sm font-bold text-white">{activeExp.mcc !== null && activeExp.mcc !== undefined ? activeExp.mcc.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[10px] text-railway-muted block">BRIER SCORE</span>
                <span className="text-sm font-bold text-white">{activeExp.brier_score !== null && activeExp.brier_score !== undefined ? activeExp.brier_score.toFixed(4) : 'UNAVAILABLE'}</span>
              </div>
            </div>

            {/* Operational Event Metrics */}
            <div className="p-2.5 rounded bg-railway-darkest/80 border border-railway-border/60 text-[11px] space-y-1">
              <span className="text-[10px] text-hud-amber font-bold uppercase block">Railway Operational Event Metrics</span>
              <div className="flex justify-between">
                <span className="text-railway-muted">Event Recall (Anomaly Hits):</span>
                <span className="text-white font-bold">{activeExp.event_recall !== null && activeExp.event_recall !== undefined ? `${(activeExp.event_recall * 100).toFixed(1)}%` : 'UNAVAILABLE'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-railway-muted">False Alarms per Track-km:</span>
                <span className="text-white font-bold">{activeExp.false_alarms_per_km !== null && activeExp.false_alarms_per_km !== undefined ? `${activeExp.false_alarms_per_km} /km` : 'UNAVAILABLE'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-railway-muted">Localization MAE:</span>
                <span className="text-white font-bold">{activeExp.localization_mae_m !== null && activeExp.localization_mae_m !== undefined ? `±${activeExp.localization_mae_m} m` : 'UNAVAILABLE'}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};