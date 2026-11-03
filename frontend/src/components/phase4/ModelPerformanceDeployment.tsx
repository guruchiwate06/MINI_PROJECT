import React from 'react';
import { EvaluatedExperiment } from '../../types';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import {
  Cpu,
  Server
} from 'lucide-react';

interface ModelPerformanceDeploymentProps {
  experiments: EvaluatedExperiment[];
}

export const ModelPerformanceDeployment: React.FC<ModelPerformanceDeploymentProps> = ({ experiments }) => {
  const ids = ['fold1_cnn_s42', 'fold1_tcn_s42', 'fold1_rf_s42', 'track_ot_pp_fold1_full_s42'];
  const deploymentProfiles = ids.map(id => experiments.find(e => e.id === id))
    .filter((e): e is EvaluatedExperiment => e !== undefined)
    .map(e => ({
      id: e.id, name: e.model_name, tier: 'Uploaded experiment record',
      size: e.checkpoint_size_kb == null ? 'Unreported' : `${(e.checkpoint_size_kb / 1024).toFixed(2)} MB`,
      latency: e.latency_ms == null ? 'Unmeasured' : `${e.latency_ms.toFixed(3)} ms`,
      throughput: e.throughput_windows_sec == null ? 'Unmeasured' : `${e.throughput_windows_sec.toFixed(1)} win/s`,
      memory: 'Training peak unverified', device: e.device,
      suitability: 'Research evaluation; deployment readiness has not been established',
      status: e.status, color: e.method === 'track_ot_pp' ? '#8b5cf6' : '#06b6d4'
    }));

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Header Profile Card */}
      <div className="p-4 rounded-xl bg-railway-surface/50 border border-railway-border/80 backdrop-blur space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-hud-cyan" />
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Model Performance, Latency & Edge Deployment Feasibility
            </h2>
            <DataIntegrityBadge type="ML_PREDICTION" size="sm" />
          </div>
          <span className="text-[11px] text-railway-muted bg-railway-darkest/80 px-2.5 py-1 rounded border border-railway-border/60">
            Hardware: per-run uploaded metadata
          </span>
        </div>
        <p className="text-[11px] text-railway-muted font-sans max-w-4xl">
          Values below come from the uploaded experiment registry. Missing values remain unmeasured.
          Validation scores and failed training attempts do not establish test performance or production readiness.
        </p>
      </div>

      {/* Deployment Tier Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {deploymentProfiles.map((p) => (
          <div
            key={p.id}
            className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80 backdrop-blur space-y-3 relative overflow-hidden"
          >
            <div
              className="absolute top-0 left-0 right-0 h-1"
              style={{ backgroundColor: p.color }}
            />

            <div className="flex items-start justify-between gap-2">
              <div>
                <span className="text-[10px] text-railway-muted uppercase block font-sans">{p.tier}</span>
                <h3 className="text-sm font-bold text-white mt-0.5">{p.name}</h3>
                <span className="text-[10px] text-railway-muted">{p.id}</span>
              </div>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  p.status === 'PRODUCTION_READY'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : p.status === 'TRAINING_IN_PROGRESS'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    : 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                }`}
              >
                {p.status.replace(/_/g, ' ')}
              </span>
            </div>

            {/* Metrics Matrix */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-railway-border/50 text-[11px]">
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[9px] text-railway-muted block">LATENCY</span>
                <span className="text-xs font-bold text-hud-cyan">{p.latency}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[9px] text-railway-muted block">THROUGHPUT</span>
                <span className="text-xs font-bold text-white">{p.throughput}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[9px] text-railway-muted block">MODEL SIZE</span>
                <span className="text-xs font-bold text-white">{p.size}</span>
              </div>
              <div className="p-2 rounded bg-railway-darkest/70 border border-railway-border/50">
                <span className="text-[9px] text-railway-muted block">MEMORY</span>
                <span className="text-xs font-bold text-white">{p.memory}</span>
              </div>
            </div>

            {/* Suitability Footer */}
            <div className="p-2.5 rounded bg-railway-darkest/80 border border-railway-border/60 text-[11px] font-sans">
              <span className="text-railway-muted">Target Deployment: </span>
              <span className="text-white font-medium">{p.suitability}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="p-4 rounded-xl bg-railway-surface/40 border border-railway-border/80">
        <Server className="h-4 w-4 text-hud-amber" />
        <p className="text-[11px] text-railway-muted mt-2">
          GPU allocation is not GPU utilization. Use the recovery profiler to measure loader, forward,
          auxiliary-loss and backward costs on the training device.
        </p>
      </div>
    </div>
  );
};
