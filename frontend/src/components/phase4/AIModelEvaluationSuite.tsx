import React, { useState, useEffect } from 'react';
import { EvaluatedExperiment, SiteHoldoutMetric, ThresholdExplorerModel } from '../../types';
import { ModelLeaderboard } from './ModelLeaderboard';
import { BenchmarkComparison } from './BenchmarkComparison';
import { SpatialHoldoutEvaluation } from './SpatialHoldoutEvaluation';
import { ThresholdExplorer } from './ThresholdExplorer';
import { ModelPerformanceDeployment } from './ModelPerformanceDeployment';
import { ExperimentRegistryView } from './ExperimentRegistryView';
import {
  Trophy,
  BarChart3,
  MapPin,
  Sliders,
  Zap,
  Database
} from 'lucide-react';

interface AIModelEvaluationSuiteProps {
  onNavigateToTab?: (tab: string) => void;
  onNavigateToSite3D?: (siteId: number, chainage: number) => void;
  onNavigateToSignalWindow?: (siteId: number) => void;
}

type Phase4SubTab =
  | 'leaderboard'
  | 'comparison'
  | 'spatial-holdout'
  | 'threshold-explorer'
  | 'deployment'
  | 'registry';

export const AIModelEvaluationSuite: React.FC<AIModelEvaluationSuiteProps> = ({
  onNavigateToSite3D,
  onNavigateToSignalWindow
}) => {
  const [activeSubTab, setActiveSubTab] = useState<Phase4SubTab>('leaderboard');
  const [experiments, setExperiments] = useState<EvaluatedExperiment[]>([]);
  const [siteMetrics, setSiteMetrics] = useState<SiteHoldoutMetric[]>([]);
  const [explorerData, setExplorerData] = useState<Record<string, ThresholdExplorerModel>>({});
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>('fold1_cnn_s42');
  const trackPilot = experiments.find(e => e.id === 'track_ot_pp_fold1_full_s42');

  // Fetch compiled Phase 4 experiment datasets
  useEffect(() => {
    async function loadData() {
      try {
        const [expRes, siteRes, teRes] = await Promise.all([
          fetch('/data/experiment_registry.json'),
          fetch('/data/spatial_holdout_site_breakdown.json'),
          fetch('/data/threshold_explorer_data.json')
        ]);

        if (expRes.ok) {
          const expData = await expRes.json();
          setExperiments(expData);
        }
        if (siteRes.ok) {
          const sData = await siteRes.json();
          setSiteMetrics(sData);
        }
        if (teRes.ok) {
          const teData = await teRes.json();
          setExplorerData(teData);
        }
      } catch (err) {
        console.error('Failed to load Phase 4 evaluation data:', err);
      }
    }

    loadData();
  }, []);

  return (
    <div className="space-y-6">
      {/* 1. HUD Executive Summary Header */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 gap-3 font-mono text-xs">
        <div className="p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block">Total Model Runs</span>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-white">{experiments.length || 50}</span>
            <span className="text-[10px] text-hud-cyan font-bold">Audited</span>
          </div>
          <span className="text-[10px] text-railway-muted block">37 Folds + 12 Baselines + 1 OT++</span>
        </div>

        <div className="p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block">Best Spatial F1</span>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-hud-cyan">0.4825</span>
            <span className="text-[10px] text-emerald-400 font-bold">Fold 1 RF</span>
          </div>
          <span className="text-[10px] text-railway-muted block">1D CNN: 0.4286 (Deep Neural)</span>
        </div>

        <div className="p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block">Leakage Inflation Penalty</span>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-rose-400">+13.43%</span>
            <span className="text-[10px] text-rose-300 font-bold">PR-AUC</span>
          </div>
          <span className="text-[10px] text-railway-muted block">Random 0.473 vs Block 0.417</span>
        </div>

        <div className="p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur space-y-1">
          <span className="text-[10px] text-railway-muted uppercase block">Fastest Edge Inference</span>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-white">0.22 ms</span>
            <span className="text-[10px] text-emerald-400 font-bold">4,452 win/s</span>
          </div>
          <span className="text-[10px] text-railway-muted block">1D Dilated CNN (262 kB)</span>
        </div>

        <div className="p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur space-y-1 col-span-2 sm:col-span-4 md:col-span-1">
          <span className="text-[10px] text-railway-muted uppercase block">TRACK-OT++ Status</span>
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-bold text-amber-400">{trackPilot?.validation_ap?.toFixed(4) ?? 'Unreported'} AP</span>
            <span className="text-[9px] px-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold">
              VAL ONLY
            </span>
          </div>
          <span className="text-[10px] text-railway-muted block truncate" title="Epoch 1 Completed on RTX 5070 GPU">
            {trackPilot?.status.replace(/_/g, ' ') ?? 'Awaiting uploaded status'}
          </span>
        </div>
      </div>

      {/* 2. Sub-Navigation Bar */}
      <div className="flex items-center gap-1.5 p-1.5 rounded-xl bg-railway-dark/90 border border-railway-border/80 backdrop-blur font-mono text-xs overflow-x-auto shadow-lg">
        <button
          onClick={() => setActiveSubTab('leaderboard')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg font-bold transition-all whitespace-nowrap ${
            activeSubTab === 'leaderboard'
              ? 'bg-hud-cyan text-railway-darkest shadow-md'
              : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
          }`}
        >
          <Trophy className="h-3.5 w-3.5" />
          <span>MODEL LEADERBOARD</span>
        </button>

        <button
          onClick={() => setActiveSubTab('comparison')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg font-bold transition-all whitespace-nowrap ${
            activeSubTab === 'comparison'
              ? 'bg-hud-cyan text-railway-darkest shadow-md'
              : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
          }`}
        >
          <BarChart3 className="h-3.5 w-3.5" />
          <span>BENCHMARK COMPARISON</span>
        </button>

        <button
          onClick={() => setActiveSubTab('spatial-holdout')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg font-bold transition-all whitespace-nowrap ${
            activeSubTab === 'spatial-holdout'
              ? 'bg-hud-cyan text-railway-darkest shadow-md'
              : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
          }`}
        >
          <MapPin className="h-3.5 w-3.5" />
          <span>SPATIAL HOLDOUT SITES</span>
        </button>

        <button
          onClick={() => setActiveSubTab('threshold-explorer')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg font-bold transition-all whitespace-nowrap ${
            activeSubTab === 'threshold-explorer'
              ? 'bg-hud-cyan text-railway-darkest shadow-md'
              : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
          }`}
        >
          <Sliders className="h-3.5 w-3.5" />
          <span>THRESHOLD EXPLORER</span>
        </button>

        <button
          onClick={() => setActiveSubTab('deployment')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg font-bold transition-all whitespace-nowrap ${
            activeSubTab === 'deployment'
              ? 'bg-hud-cyan text-railway-darkest shadow-md'
              : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
          }`}
        >
          <Zap className="h-3.5 w-3.5" />
          <span>DEPLOYMENT & LATENCY</span>
        </button>

        <button
          onClick={() => setActiveSubTab('registry')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg font-bold transition-all whitespace-nowrap ${
            activeSubTab === 'registry'
              ? 'bg-hud-cyan text-railway-darkest shadow-md'
              : 'text-railway-muted hover:text-white hover:bg-railway-surface/60'
          }`}
        >
          <Database className="h-3.5 w-3.5" />
          <span>EXPERIMENT REGISTRY</span>
        </button>
      </div>

      {/* 3. Sub-Tab Content Rendering */}
      <div>
        {activeSubTab === 'leaderboard' && (
          <ModelLeaderboard
            experiments={experiments}
            selectedExperimentId={selectedExperimentId}
            onSelectExperiment={setSelectedExperimentId}
          />
        )}

        {activeSubTab === 'comparison' && (
          <BenchmarkComparison
            experiments={experiments}
            explorerData={explorerData}
            selectedExperimentId={selectedExperimentId}
            onSelectExperiment={setSelectedExperimentId}
          />
        )}

        {activeSubTab === 'spatial-holdout' && (
          <SpatialHoldoutEvaluation
            siteMetrics={siteMetrics}
            onNavigateToSite3D={onNavigateToSite3D}
            onNavigateToSignalWindow={onNavigateToSignalWindow}
          />
        )}

        {activeSubTab === 'threshold-explorer' && (
          <ThresholdExplorer explorerData={explorerData} />
        )}

        {activeSubTab === 'deployment' && (
          <ModelPerformanceDeployment experiments={experiments} />
        )}

        {activeSubTab === 'registry' && (
          <ExperimentRegistryView
            experiments={experiments}
            onSelectExperiment={(id) => {
              setSelectedExperimentId(id);
              setActiveSubTab('comparison');
            }}
          />
        )}
      </div>
    </div>
  );
};
