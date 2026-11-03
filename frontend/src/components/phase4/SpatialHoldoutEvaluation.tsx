import React, { useState } from 'react';
import { SiteHoldoutMetric } from '../../types';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import {
  MapPin,
  Box,
  Activity
} from 'lucide-react';

interface SpatialHoldoutEvaluationProps {
  siteMetrics: SiteHoldoutMetric[];
  onNavigateToSite3D?: (siteId: number, chainage: number) => void;
  onNavigateToSignalWindow?: (siteId: number) => void;
}

export const SpatialHoldoutEvaluation: React.FC<SpatialHoldoutEvaluationProps> = ({
  siteMetrics,
  onNavigateToSite3D,
  onNavigateToSignalWindow
}) => {
  const [selectedSiteId, setSelectedSiteId] = useState<number>(1);

  const activeSite = siteMetrics.find((s) => s.site_id === selectedSiteId) || siteMetrics[0];

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Overview Banner */}
      <div className="p-4 rounded-xl bg-railway-surface/50 border border-railway-border/80 backdrop-blur space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <MapPin className="h-4 w-4 text-hud-cyan" />
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Spatial Holdout Site-by-Site Evaluation
            </h2>
            <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
          </div>
          <span className="text-[11px] text-railway-muted bg-railway-darkest/80 px-2.5 py-1 rounded border border-railway-border/60">
            3-Fold Cross-Validation • 18m Buffer Rings Enforced
          </span>
        </div>
        <p className="text-[11px] text-railway-muted font-sans max-w-4xl">
          In spatial holdout evaluation, defect sites are strictly quarantined from training. Folds evaluate generalization
          to physically unseen track defects across the Orbisonia 1.4-km testbed. Notice that Sites 4 & 5 represent complex
          Turnout frog switches with Black cones, while Sites 1–3 and 6–9 are Rail Joint expansion gaps with Orange cones.
        </p>
      </div>

      {/* Grid of 9 Sites Selector */}
      <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-2">
        {siteMetrics.map((site) => {
          const isSelected = site.site_id === selectedSiteId;

          return (
            <button
              key={site.site_id}
              onClick={() => setSelectedSiteId(site.site_id)}
              className={`p-2.5 rounded-lg border text-left transition-all ${
                isSelected
                  ? 'bg-hud-cyan/20 border-hud-cyan shadow-md shadow-hud-cyan/15 text-white'
                  : 'bg-railway-surface/40 border-railway-border/60 hover:bg-railway-surface/70 text-railway-muted'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-extrabold text-sm text-white">S{site.site_id}</span>
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    site.cone_color === 'Black' ? 'bg-slate-400' : 'bg-orange-500'
                  }`}
                  title={`${site.cone_color} Cone`}
                />
              </div>
              <span className="text-[10px] text-railway-muted block mt-1 truncate">
                {site.chainage_m.toFixed(0)}m
              </span>
              <span
                className={`text-[9px] font-bold block mt-1 ${
                  site.f1 >= 0.5 ? 'text-emerald-400' : site.f1 >= 0.3 ? 'text-amber-400' : 'text-rose-400'
                }`}
              >
                F1: {site.f1.toFixed(2)}
              </span>
            </button>
          );
        })}
      </div>

      {/* Selected Site Detail Card */}
      {activeSite && (
        <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur space-y-4">
          {/* Header */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-railway-border/60">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-white">
                  Site {activeSite.site_id}: {activeSite.defect_type}
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    activeSite.cone_color === 'Black'
                      ? 'bg-slate-700 text-slate-200 border border-slate-500'
                      : 'bg-orange-500/20 text-orange-400 border border-orange-500/40'
                  }`}
                >
                  {activeSite.cone_color} Survey Cone
                </span>
                <span className="text-[10px] text-railway-muted bg-railway-darkest/70 px-2 py-0.5 rounded">
                  Chainage: {activeSite.chainage_m.toFixed(1)} m
                </span>
              </div>
              <span className="text-[11px] text-railway-muted font-sans block mt-1">
                Evaluated under {activeSite.fold.toUpperCase()} test partition ({activeSite.model})
              </span>
            </div>

            {/* Cross-Phase Navigation Actions */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => onNavigateToSite3D?.(activeSite.site_id, activeSite.chainage_m)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-hud-cyan/15 hover:bg-hud-cyan/30 text-hud-cyan border border-hud-cyan/40 font-bold transition-all shadow-sm"
              >
                <Box className="h-3.5 w-3.5" />
                <span>INSPECT IN 3D TWIN</span>
              </button>

              <button
                onClick={() => onNavigateToSignalWindow?.(activeSite.site_id)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-railway-darkest/80 hover:bg-railway-surface/80 text-railway-text border border-railway-border font-bold transition-all"
              >
                <Activity className="h-3.5 w-3.5 text-hud-amber" />
                <span>VIEW OSCILLOSCOPE</span>
              </button>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
            <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[10px] text-railway-muted block">AVERAGE PRECISION (AP)</span>
              <span className="text-base font-bold text-hud-cyan">{activeSite.average_precision.toFixed(4)}</span>
            </div>
            <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[10px] text-railway-muted block">F1 SCORE</span>
              <span className="text-base font-bold text-white">{activeSite.f1.toFixed(4)}</span>
            </div>
            <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[10px] text-railway-muted block">RECALL (DETECTION)</span>
              <span className="text-base font-bold text-emerald-400">{(activeSite.recall * 100).toFixed(1)}%</span>
            </div>
            <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[10px] text-railway-muted block">PRECISION</span>
              <span className="text-base font-bold text-white">{(activeSite.precision * 100).toFixed(1)}%</span>
            </div>
            <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[10px] text-railway-muted block">ROC-AUC</span>
              <span className="text-base font-bold text-white">{activeSite.roc_auc.toFixed(4)}</span>
            </div>
            <div className="p-3 rounded-lg bg-railway-darkest/70 border border-railway-border/50">
              <span className="text-[10px] text-railway-muted block">CALIBRATED THRESHOLD</span>
              <span className="text-base font-bold text-amber-400">{activeSite.calibrated_threshold.toFixed(4)}</span>
            </div>
          </div>

          {/* Site Sample & Confusion Counts */}
          <div className="p-3.5 rounded-lg bg-railway-darkest/80 border border-railway-border/60">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div>
                <span className="text-[10px] text-railway-muted block">EVALUATED WINDOWS</span>
                <span className="text-lg font-bold text-white">{activeSite.observations}</span>
              </div>
              <div>
                <span className="text-[10px] text-hud-cyan block">TRUE POSITIVES (TP)</span>
                <span className="text-lg font-bold text-hud-cyan">{activeSite.tp}</span>
              </div>
              <div>
                <span className="text-[10px] text-rose-400 block">FALSE POSITIVES (FP)</span>
                <span className="text-lg font-bold text-rose-400">{activeSite.fp}</span>
              </div>
              <div>
                <span className="text-[10px] text-amber-400 block">FALSE NEGATIVES (FN)</span>
                <span className="text-lg font-bold text-amber-400">{activeSite.fn}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
