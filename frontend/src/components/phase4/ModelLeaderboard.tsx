import React, { useState, useMemo } from 'react';
import { EvaluatedExperiment } from '../../types';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import {
  ArrowUpDown,
  Search,
  Info
} from 'lucide-react';

interface ModelLeaderboardProps {
  experiments: EvaluatedExperiment[];
  selectedExperimentId: string;
  onSelectExperiment: (id: string) => void;
}

type SortField =
  | 'model_name'
  | 'f1'
  | 'average_precision'
  | 'roc_auc'
  | 'precision'
  | 'recall'
  | 'balanced_accuracy'
  | 'throughput_windows_sec';

export const ModelLeaderboard: React.FC<ModelLeaderboardProps> = ({
  experiments,
  selectedExperimentId,
  onSelectExperiment
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [splitFilter, setSplitFilter] = useState<string>('ALL');
  const [architectureFilter, setArchitectureFilter] = useState<string>('ALL');
  const [sortField, setSortField] = useState<SortField>('average_precision');
  const [sortAsc, setSortAsc] = useState(false);

  // Filter and sort experiments
  const filteredExperiments = useMemo(() => {
    return experiments.filter((exp) => {
      // Split filter
      if (splitFilter === 'SPATIAL' && !exp.split_strategy.includes('Spatial')) return false;
      if (splitFilter === 'BLOCK' && !exp.split_strategy.includes('Block')) return false;
      if (splitFilter === 'RANDOM' && !exp.split_strategy.includes('Random')) return false;
      if (splitFilter === 'RUN' && !exp.split_strategy.includes('Run_Held_Out')) return false;

      // Architecture filter
      if (architectureFilter !== 'ALL' && exp.method !== architectureFilter) return false;

      // Search term
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        return (
          exp.id.toLowerCase().includes(term) ||
          exp.model_name.toLowerCase().includes(term) ||
          exp.architecture.toLowerCase().includes(term) ||
          exp.split_strategy.toLowerCase().includes(term)
        );
      }
      return true;
    });
  }, [experiments, splitFilter, architectureFilter, searchTerm]);

  const sortedExperiments = useMemo(() => {
    return [...filteredExperiments].sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      // Handle nulls
      if (aVal === null || aVal === undefined) aVal = -Infinity;
      if (bVal === null || bVal === undefined) bVal = -Infinity;

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }

      return sortAsc ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    });
  }, [filteredExperiments, sortField, sortAsc]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Search & Filters Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur font-mono text-xs">
        <div className="flex items-center gap-2 flex-1 min-w-[240px]">
          <Search className="h-4 w-4 text-railway-muted" />
          <input
            type="text"
            placeholder="Search 50 models, architectures, or experiment IDs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-railway-darkest/70 border border-railway-border/60 rounded px-2.5 py-1 text-white placeholder-railway-muted focus:outline-none focus:border-hud-cyan"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Split Strategy Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-railway-muted text-[11px]">SPLIT:</span>
            <select
              value={splitFilter}
              onChange={(e) => setSplitFilter(e.target.value)}
              className="bg-railway-darkest/80 border border-railway-border/70 rounded px-2 py-1 text-white text-[11px] focus:outline-none focus:border-hud-cyan cursor-pointer"
            >
              <option value="ALL">All Evaluation Splits ({experiments.length})</option>
              <option value="SPATIAL">Spatial Holdout (Guarded Sites)</option>
              <option value="BLOCK">Block-Grouped (Overlap-Free)</option>
              <option value="RANDOM">Random-Window (Leaky Control)</option>
              <option value="RUN">Run-Held-Out (Transit Blind)</option>
            </select>
          </div>

          {/* Architecture Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-railway-muted text-[11px]">MODEL:</span>
            <select
              value={architectureFilter}
              onChange={(e) => setArchitectureFilter(e.target.value)}
              className="bg-railway-darkest/80 border border-railway-border/70 rounded px-2 py-1 text-white text-[11px] focus:outline-none focus:border-hud-cyan cursor-pointer"
            >
              <option value="ALL">All Architectures</option>
              <option value="cnn">1D Dilated CNN (Deep Temporal)</option>
              <option value="tcn">Residual TCN (Causal Conv)</option>
              <option value="ssl_tcn">Self-Supervised TCN (SimCLR)</option>
              <option value="rf">Random Forest (Spectral Tree)</option>
              <option value="track_ot_pp">TRACK-OT++ (Multimodal Transf.)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Leaderboard Table Container */}
      <div className="rounded-xl bg-railway-surface/40 border border-railway-border/80 overflow-hidden shadow-2xl backdrop-blur">
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs border-collapse">
            <thead>
              <tr className="bg-railway-darkest/90 border-b border-railway-border/80 text-railway-muted select-none">
                <th className="py-3 px-3.5 font-bold uppercase tracking-wider text-[11px]">Rank & Model</th>
                <th className="py-3 px-3 font-bold uppercase tracking-wider text-[11px]">Architecture</th>
                <th className="py-3 px-3 font-bold uppercase tracking-wider text-[11px]">Evaluation Split</th>
                <th
                  onClick={() => handleSort('average_precision')}
                  className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] cursor-pointer hover:text-hud-cyan text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>PR-AUC / AP</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('f1')}
                  className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] cursor-pointer hover:text-hud-cyan text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>F1 Score</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('precision')}
                  className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] cursor-pointer hover:text-hud-cyan text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Precision</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('recall')}
                  className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] cursor-pointer hover:text-hud-cyan text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Recall</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('roc_auc')}
                  className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] cursor-pointer hover:text-hud-cyan text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>ROC-AUC</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] text-center">Threshold</th>
                <th className="py-3 px-3.5 font-bold uppercase tracking-wider text-[11px] text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-railway-border/40">
              {sortedExperiments.map((exp, idx) => {
                const isSelected = exp.id === selectedExperimentId;
                const isOngoingTrackOT = exp.id.includes('track_ot') && exp.average_precision == null;
                const isBaseline = exp.status.includes('BASELINE');

                return (
                  <tr
                    key={exp.id}
                    onClick={() => onSelectExperiment(exp.id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-hud-cyan/15 border-l-4 border-l-hud-cyan'
                        : 'hover:bg-railway-surface/70'
                    }`}
                  >
                    {/* Model & ID */}
                    <td className="py-2.5 px-3.5 font-bold">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-railway-muted w-5 text-right">{idx + 1}.</span>
                        <div>
                          <div className="text-white flex items-center gap-1.5">
                            <span>{exp.model_name}</span>
                            {isOngoingTrackOT && (
                              <span className="text-[9px] px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-400 font-extrabold border border-amber-500/40 animate-pulse">
                                GPU RUNNING
                              </span>
                            )}
                          </div>
                          <span className="text-[10px] text-railway-muted block">{exp.id}</span>
                        </div>
                      </div>
                    </td>

                    {/* Architecture */}
                    <td className="py-2.5 px-3 text-[11px] text-railway-text/90">
                      <span className="truncate block max-w-[180px]" title={exp.architecture}>
                        {exp.architecture}
                      </span>
                      <span className="text-[9px] text-railway-muted">
                        Size: {exp.checkpoint_size_kb == null ? 'unreported' : exp.checkpoint_size_kb > 1024 ? `${(exp.checkpoint_size_kb / 1024).toFixed(1)} MB` : `${exp.checkpoint_size_kb} kB`}
                      </span>
                    </td>

                    {/* Split Strategy */}
                    <td className="py-2.5 px-3 text-[11px]">
                      <span
                        className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          exp.split_strategy.includes('Spatial')
                            ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                            : exp.split_strategy.includes('Random')
                            ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                            : 'bg-hud-cyan/15 text-hud-cyan border border-hud-cyan/30'
                        }`}
                      >
                        {exp.split_strategy.split('(')[0].trim()}
                      </span>
                      <span className="text-[9px] text-railway-muted block mt-0.5">
                        Fold: {exp.fold} {exp.seed ? `• Seed ${exp.seed}` : ''}
                      </span>
                    </td>

                    {/* PR-AUC / AP */}
                    <td className="py-2.5 px-3 text-right">
                      {isOngoingTrackOT ? (
                        <div className="flex flex-col items-end">
                          <span className="text-amber-400 font-bold text-xs">{exp.validation_ap?.toFixed(4) ?? 'unreported'}</span>
                          <span className="text-[9px] text-amber-300/80 font-sans">Validation AP (epoch {exp.validation_epoch ?? '?'})</span>
                        </div>
                      ) : exp.average_precision !== null ? (
                        <span
                          className={`font-bold ${
                            exp.average_precision >= 0.5
                              ? 'text-hud-cyan'
                              : exp.average_precision >= 0.35
                              ? 'text-white'
                              : 'text-railway-muted'
                          }`}
                        >
                          {exp.average_precision.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-railway-muted">—</span>
                      )}
                    </td>

                    {/* F1 Score */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.f1 !== null ? (
                        <span className="text-white font-bold">{exp.f1.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">Pending</span>
                      )}
                    </td>

                    {/* Precision */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.precision !== null ? (
                        <span className="text-railway-text">{exp.precision.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">—</span>
                      )}
                    </td>

                    {/* Recall */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.recall !== null ? (
                        <span className="text-railway-text">{exp.recall.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">—</span>
                      )}
                    </td>

                    {/* ROC-AUC */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.roc_auc !== null ? (
                        <span className="text-railway-text">{exp.roc_auc.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">—</span>
                      )}
                    </td>

                    {/* Threshold */}
                    <td className="py-2.5 px-3 text-center">
                      <span className="px-1.5 py-0.5 rounded bg-railway-darkest/80 text-railway-muted text-[10px]">
                        {exp.calibrated_threshold?.toFixed(4)}
                      </span>
                    </td>

                    {/* Status Badge */}
                    <td className="py-2.5 px-3.5 text-right whitespace-nowrap">
                      {isOngoingTrackOT ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          {exp.status.startsWith('FAILED') ? 'FAILED · VALIDATION ONLY' : exp.status.replace(/_/g, ' ')}
                        </span>
                      ) : isBaseline ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/40">
                          PHASE 1 BASELINE
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                          TEST EVALUATED
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Footer Note */}
        <div className="p-3 bg-railway-darkest/90 border-t border-railway-border/80 flex flex-wrap items-center justify-between gap-2 text-[11px] text-railway-muted font-mono">
          <div className="flex items-center gap-2">
            <Info className="h-3.5 w-3.5 text-hud-cyan" />
            <span>Showing {sortedExperiments.length} of {experiments.length} audited model checkpoints.</span>
          </div>
          <div className="flex items-center gap-2">
            <DataIntegrityBadge type="ML_PREDICTION" size="sm" />
            <span className="text-[10px] text-railway-muted font-sans">Strictly zero test-set threshold peeking.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
