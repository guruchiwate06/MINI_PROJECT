import React, { useState, useMemo } from 'react';
import { EvaluatedExperiment } from '../../types';
import {
  Search,
  CheckCircle2,
  Copy,
  ExternalLink
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
      if (statusFilter !== 'ALL' && !e.status.includes(statusFilter)) return false;
      if (searchTerm) {
        const t = searchTerm.toLowerCase();
        return (
          e.id.toLowerCase().includes(t) ||
          e.model_name.toLowerCase().includes(t) ||
          e.split_strategy.toLowerCase().includes(t)
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
            placeholder="Search experiment registry by ID, model, or split..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-railway-darkest/70 border border-railway-border/60 rounded px-2.5 py-1 text-white placeholder-railway-muted focus:outline-none focus:border-hud-cyan"
          />
        </div>

        <div className="flex items-center gap-2">
          <span className="text-railway-muted text-[11px]">STATUS:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-railway-darkest/80 border border-railway-border/70 rounded px-2.5 py-1 text-white text-[11px] focus:outline-none focus:border-hud-cyan cursor-pointer"
          >
            <option value="ALL">All Statuses ({experiments.length})</option>
            <option value="TEST_CLEAN">Test Evaluated (37)</option>
            <option value="BASELINE">Phase 1 Baselines (12)</option>
            <option value="FAILED">Failed runs ({experiments.filter(e => e.status.includes('FAILED')).length})</option>
          </select>
        </div>
      </div>

      {/* Grid of Experiment Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map((exp) => (
          <div
            key={exp.id}
            className="p-3.5 rounded-xl bg-railway-surface/40 border border-railway-border/70 backdrop-blur hover:border-hud-cyan/50 transition-all space-y-2.5 relative group"
          >
            {/* Top Row */}
            <div className="flex items-start justify-between gap-2">
              <div className="truncate flex-1">
                <span className="text-[10px] text-hud-cyan uppercase tracking-wider font-bold block">
                  {exp.model_name}
                </span>
                <span className="text-xs font-bold text-white block truncate" title={exp.id}>
                  {exp.id}
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

            {/* Split Strategy */}
            <div className="p-2 rounded bg-railway-darkest/80 border border-railway-border/50 text-[10px] space-y-1">
              <div className="flex justify-between">
                <span className="text-railway-muted">Split:</span>
                <span className="text-white truncate max-w-[160px]">{exp.split_strategy.split('(')[0].trim()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-railway-muted">Dataset:</span>
                <span className="text-emerald-400">{exp.dataset_version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-railway-muted">Test Partition:</span>
                <span className="text-white">
                  {Array.isArray(exp.test_sites) ? `Sites [${exp.test_sites.join(', ')}]` : exp.test_sites}
                </span>
              </div>
            </div>

            {/* Key Metric Snapshot */}
            <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
              <div className="p-1.5 rounded bg-railway-darkest/60 border border-railway-border/40">
                <span className="text-railway-muted block text-[9px]">F1</span>
                <span className="font-bold text-white">{exp.f1 !== null ? exp.f1.toFixed(3) : 'Pending'}</span>
              </div>
              <div className="p-1.5 rounded bg-railway-darkest/60 border border-railway-border/40">
                <span className="text-railway-muted block text-[9px]">AP</span>
                <span className="font-bold text-hud-cyan">{exp.average_precision !== null ? exp.average_precision.toFixed(3) : '—'}</span>
              </div>
              <div className="p-1.5 rounded bg-railway-darkest/60 border border-railway-border/40">
                <span className="text-railway-muted block text-[9px]">THRESH</span>
                <span className="font-bold text-amber-400">{exp.calibrated_threshold?.toFixed(2)}</span>
              </div>
            </div>

            {/* Footer with Select Action */}
            <div className="pt-2 border-t border-railway-border/40 flex items-center justify-between">
              <span
                className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                  exp.status.includes('TEST_CLEAN')
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : exp.status.includes('PROGRESS')
                    ? 'bg-amber-500/20 text-amber-300'
                    : 'bg-blue-500/20 text-blue-300'
                }`}
              >
                {exp.status.replace(/_/g, ' ')}
              </span>

              <button
                onClick={() => onSelectExperiment(exp.id)}
                className="text-[10px] text-hud-cyan hover:underline font-bold flex items-center gap-1"
              >
                <span>Select Model</span>
                <ExternalLink className="h-3 w-3" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
