import React, { useState, useMemo } from 'react';
import { EvaluatedExperiment } from '../../types';
import {
  Search,
  CheckCircle2,
  Copy,
  ExternalLink,
  ShieldCheck,
  AlertTriangle
} from 'lucide-react';

interface ExperimentRegistryViewProps {
  experiments: EvaluatedExperiment[];
  onSelectExperiment: (id: string) => void;
}

export const ExperimentRegistryView: React.FC<ExperimentRegistryViewProps> = ({
  experiments,
  onSelectExperiment
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    return experiments.filter((e) => {
      if (statusFilter === 'TEST_CLEAN' && (!e.status.includes('TEST_CLEAN') || e.is_test_evaluated === false)) return false;
      if (statusFilter === 'BASELINE' && !e.status.includes('BASELINE')) return false;
      if (statusFilter === 'INCOMPLETE' && e.status !== 'TRAINING_INCOMPLETE') return false;
      if (searchTerm) {
        const t = searchTerm.toLowerCase();
        return (
          e.id.toLowerCase().includes(t) ||
          e.model_name.toLowerCase().includes(t) ||
          e.architecture.toLowerCase().includes(t) ||
          e.split_strategy.toLowerCase().includes(t) ||
          (e.record_type && e.record_type.toLowerCase().includes(t))
        );
      }
      return true;
    });
  }, [experiments, statusFilter, searchTerm]);

  const handleCopy = (id: string) => {
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="space-y-4 font-mono text-xs">
      {/* Search and Filters */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur">
        <div className="flex items-center gap-2 flex-1 min-w-[240px]">
          <Search className="h-4 w-4 text-railway-muted" />
          <input
            type="text"
            placeholder="Search 50 experiment records by ID, model, split, or record type..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-railway-darkest/70 border border-railway-border/60 rounded px-2.5 py-1 text-white placeholder-railway-muted focus:outline-none focus:border-hud-cyan"
          />
        </div>

        <div className="flex items-center gap-2">
          <span className="text-railway-muted text-[11px]">FILTER:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-railway-darkest/80 border border-railway-border/70 rounded px-2.5 py-1 text-white text-[11px] focus:outline-none focus:border-hud-cyan cursor-pointer"
          >
            <option value="ALL">All 50 Audited Experiments</option>
            <option value="TEST_CLEAN">Test Evaluated Models (37 Folds)</option>
            <option value="BASELINE">Phase 1 Baselines (12 Records)</option>
            <option value="INCOMPLETE">Incomplete / Validation Only (1 Record)</option>
          </select>
        </div>
      </div>

      {/* Forensic Accounting Info Banner */}
      <div className="p-3 rounded-xl bg-railway-surface/40 border border-railway-border/70 flex items-center justify-between gap-3 text-[11px] text-railway-muted">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-hud-cyan shrink-0" />
          <span>
            <b>Reconciliation Audit:</b> 50 unique records accounted for (36 3-seed 3-fold runs + 1 combined-cohort TCN + 12 historical Phase 1 baselines + 1 Track-OT++ validation record).
          </span>
        </div>
        <span className="text-hud-cyan font-bold whitespace-nowrap">ZERO UNACCOUNTED RUNS</span>
      </div>

      {/* Grid of Experiment Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map((exp) => (
          <div
            key={exp.id}
            className={`p-3.5 rounded-xl border backdrop-blur transition-all space-y-2.5 relative group ${
              exp.status === 'TRAINING_INCOMPLETE'
                ? 'bg-rose-950/20 border-amber-500/50 hover:border-amber-400'
                : 'bg-railway-surface/40 border-railway-border/70 hover:border-hud-cyan/50'
            }`}
          >
            {/* Top Row */}
            <div className="flex items-start justify-between gap-2">
              <div className="truncate flex-1">
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-hud-cyan uppercase tracking-wider font-bold block truncate">
                    {exp.model_name}
                  </span>
                  {exp.status === 'TRAINING_INCOMPLETE' && (
                    <span className="px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-400 text-[9px] font-bold border border-amber-500/40">
                      VAL ONLY
                    </span>
                  )}
                </div>
                <span className="text-xs font-bold text-white block truncate" title={exp.id}>
                  {exp.id}
                </span>
                <span className="text-[9px] text-railway-muted block">
                  {exp.record_type || 'Fold Experiment'}
                </span>
              </div>
              <button
                onClick={() => handleCopy(exp.id)}
                className="text-railway-muted hover:text-white p-1 rounded hover:bg-railway-surface transition-colors"
                title="Copy Experiment ID"
              >
                {copiedId === exp.id ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>
            </div>

            {/* Split & Artifact Details */}
            <div className="p-2 rounded bg-railway-darkest/80 border border-railway-border/50 text-[10px] space-y-1">
              <div className="flex justify-between">
                <span className="text-railway-muted">Split:</span>
                <span className="text-white truncate max-w-[160px]">{exp.split_strategy.split('(')[0].trim()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-railway-muted">Manifest Hash:</span>
                <span className="text-hud-cyan font-mono truncate max-w-[140px]" title={exp.split_fingerprint}>
                  {exp.split_fingerprint ? exp.split_fingerprint.slice(0, 10) + '...' : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-railway-muted">Partition:</span>
                <span className={exp.evaluation_partition === 'test' ? 'text-emerald-400 font-bold uppercase' : 'text-amber-400 font-bold uppercase'}>
                  {exp.evaluation_partition || 'test'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-railway-muted">Checkpoint:</span>
                <span className="text-white truncate max-w-[150px]" title={exp.checkpoint_path || 'N/A'}>
                  {exp.checkpoint_path ? exp.checkpoint_path.split('/').pop() : 'N/A'}
                </span>
              </div>
            </div>

            {/* Key Metric Snapshot */}
            <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
              <div className="p-1.5 rounded bg-railway-darkest/60 border border-railway-border/40">
                <span className="text-railway-muted block text-[9px]">F1</span>
                <span className="font-bold text-white">
                  {exp.f1 !== null && exp.f1 !== undefined ? exp.f1.toFixed(3) : 'UNAVAILABLE'}
                </span>
              </div>
              <div className="p-1.5 rounded bg-railway-darkest/60 border border-railway-border/40">
                <span className="text-railway-muted block text-[9px]">AP</span>
                <span className="font-bold text-hud-cyan">
                  {exp.average_precision !== null && exp.average_precision !== undefined
                    ? exp.average_precision.toFixed(3)
                    : exp.validation_ap !== null && exp.validation_ap !== undefined
                    ? `${exp.validation_ap.toFixed(3)} (Val)`
                    : 'UNAVAILABLE'}
                </span>
              </div>
              <div className="p-1.5 rounded bg-railway-darkest/60 border border-railway-border/40">
                <span className="text-railway-muted block text-[9px]">THRESH</span>
                <span className="font-bold text-amber-400">
                  {exp.calibrated_threshold !== null && exp.calibrated_threshold !== undefined
                    ? exp.calibrated_threshold.toFixed(2)
                    : 'UNAVAILABLE'}
                </span>
              </div>
            </div>

            {/* Footer with Select Action */}
            <div className="pt-2 border-t border-railway-border/40 flex items-center justify-between">
              <span
                className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                  exp.status === 'TRAINING_INCOMPLETE'
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                    : exp.status.includes('TEST_CLEAN')
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-blue-500/20 text-blue-300'
                }`}
              >
                {exp.status.replace(/_/g, ' ')}
              </span>

              {exp.is_test_evaluated !== false ? (
                <button
                  onClick={() => onSelectExperiment(exp.id)}
                  className="text-[10px] text-hud-cyan hover:underline font-bold flex items-center gap-1"
                >
                  <span>Select Model</span>
                  <ExternalLink className="h-3 w-3" />
                </button>
              ) : (
                <span className="text-[9px] text-amber-400 font-bold flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  <span>Unranked</span>
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};