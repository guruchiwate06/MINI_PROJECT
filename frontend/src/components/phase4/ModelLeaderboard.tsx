import React, { useState, useMemo } from 'react';
import { EvaluatedExperiment } from '../../types';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';
import {
  ArrowUpDown,
  Search,
  Info,
  AlertTriangle,
  XCircle,
  ShieldAlert
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

  // 1. Separate Track-OT++ from test-evaluated models
  const { testEvaluatedModels, trackOtModel } = useMemo(() => {
    const testModels: EvaluatedExperiment[] = [];
    let otModel: EvaluatedExperiment | undefined;

    for (const exp of experiments) {
      if (exp.method === 'track_ot_pp' || exp.id.includes('track_ot') || exp.is_test_evaluated === false) {
        otModel = exp;
      } else {
        testModels.push(exp);
      }
    }
    return { testEvaluatedModels: testModels, trackOtModel: otModel };
  }, [experiments]);

  // Filter test-evaluated models
  const filteredExperiments = useMemo(() => {
    return testEvaluatedModels.filter((exp) => {
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
  }, [testEvaluatedModels, splitFilter, architectureFilter, searchTerm]);

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
    <div className="space-y-6">
      {/* Search & Filters Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-railway-surface/60 border border-railway-border/80 backdrop-blur font-mono text-xs">
        <div className="flex items-center gap-2 flex-1 min-w-[240px]">
          <Search className="h-4 w-4 text-railway-muted" />
          <input
            type="text"
            placeholder="Search test-evaluated models, architectures, or experiment IDs..."
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
              <option value="ALL">All Test Partitions ({testEvaluatedModels.length})</option>
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
              <option value="ALL">All Test Architectures</option>
              <option value="cnn">1D Dilated CNN (Deep Temporal)</option>
              <option value="tcn">Residual TCN (Causal Conv)</option>
              <option value="ssl_tcn">Self-Supervised TCN (SimCLR)</option>
              <option value="rf">Random Forest (Spectral Tree)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Protocol Notice Banner */}
      <div className="p-3 rounded-xl bg-railway-surface/40 border border-railway-border/70 flex items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center gap-2 text-railway-muted">
          <Info className="h-4 w-4 text-hud-cyan shrink-0" />
          <span>
            Ranked leaderboard displays <b>held-out test-evaluated models only</b> (37 fold runs + 12 baselines). Track-OT++ is quarantined in the validation section below.
          </span>
        </div>
        <div className="flex items-center gap-1 text-[11px] text-amber-300 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/30 whitespace-nowrap">
          <AlertTriangle className="h-3 w-3" />
          <span>AP ≠ PR-AUC (Step-Interpolated)</span>
        </div>
      </div>

      {/* Ranked Test Leaderboard Table */}
      <div className="rounded-xl bg-railway-surface/40 border border-railway-border/80 overflow-hidden shadow-2xl backdrop-blur">
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs border-collapse">
            <thead>
              <tr className="bg-railway-darkest/90 border-b border-railway-border/80 text-railway-muted select-none">
                <th className="py-3 px-3.5 font-bold uppercase tracking-wider text-[11px]">Rank & Model</th>
                <th className="py-3 px-3 font-bold uppercase tracking-wider text-[11px]">Architecture</th>
                <th className="py-3 px-3 font-bold uppercase tracking-wider text-[11px]">Evaluation Partition</th>
                <th
                  onClick={() => handleSort('average_precision')}
                  className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] cursor-pointer hover:text-hud-cyan text-right"
                  title="Average Precision (AP) computed via step-function interpolation of continuous predictions. Distinct from trapezoidal PR-AUC."
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Average Precision (AP)</span>
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
                  title="Area Under ROC Curve computed from continuous predicted probabilities."
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>ROC-AUC</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th className="py-3 px-3 font-bold uppercase tracking-wider text-[11px] text-center" title="Threshold calibrated strictly on validation partition without test set peeking.">
                  Calibrated Threshold
                </th>
                <th className="py-3 px-3.5 font-bold uppercase tracking-wider text-[11px] text-right">Evaluation Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-railway-border/40">
              {sortedExperiments.map((exp, idx) => {
                const isSelected = exp.id === selectedExperimentId;
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
                          </div>
                          <span className="text-[10px] text-railway-muted block font-normal">{exp.id}</span>
                        </div>
                      </div>
                    </td>

                    {/* Architecture */}
                    <td className="py-2.5 px-3 text-[11px] text-railway-text/90">
                      <span className="truncate block max-w-[180px]" title={exp.architecture}>
                        {exp.architecture}
                      </span>
                      <span className="text-[9px] text-railway-muted">
                        Size: {exp.checkpoint_size_kb == null ? 'UNAVAILABLE' : exp.checkpoint_size_kb > 1024 ? `${(exp.checkpoint_size_kb / 1024).toFixed(1)} MB` : `${exp.checkpoint_size_kb} kB`}
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

                    {/* Average Precision (AP) */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.average_precision !== null && exp.average_precision !== undefined ? (
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
                        <span className="text-amber-400 text-[10px]">UNAVAILABLE</span>
                      )}
                    </td>

                    {/* F1 Score */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.f1 !== null && exp.f1 !== undefined ? (
                        <span className="text-white font-bold">{exp.f1.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">UNAVAILABLE</span>
                      )}
                    </td>

                    {/* Precision */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.precision !== null && exp.precision !== undefined ? (
                        <span className="text-railway-text">{exp.precision.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">UNAVAILABLE</span>
                      )}
                    </td>

                    {/* Recall */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.recall !== null && exp.recall !== undefined ? (
                        <span className="text-railway-text">{exp.recall.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">UNAVAILABLE</span>
                      )}
                    </td>

                    {/* ROC-AUC */}
                    <td className="py-2.5 px-3 text-right">
                      {exp.roc_auc !== null && exp.roc_auc !== undefined ? (
                        <span className="text-railway-text">{exp.roc_auc.toFixed(4)}</span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">UNAVAILABLE</span>
                      )}
                    </td>

                    {/* Calibrated Threshold */}
                    <td className="py-2.5 px-3 text-center">
                      {exp.calibrated_threshold !== null && exp.calibrated_threshold !== undefined ? (
                        <span className="px-1.5 py-0.5 rounded bg-railway-darkest/80 text-amber-400 font-mono text-[10px] border border-railway-border/60">
                          {exp.calibrated_threshold.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-railway-muted text-[10px]">UNAVAILABLE</span>
                      )}
                    </td>

                    {/* Status Badge */}
                    <td className="py-2.5 px-3.5 text-right whitespace-nowrap">
                      {isBaseline ? (
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
            <span>Showing {sortedExperiments.length} test-evaluated models (37 fold runs + 12 baselines).</span>
          </div>
          <div className="flex items-center gap-2">
            <DataIntegrityBadge type="ML_PREDICTION" size="sm" />
            <span className="text-[10px] text-railway-muted font-sans">Strictly zero test-set threshold peeking.</span>
          </div>
        </div>
      </div>

      {/* 2. Separate Training / Validation Status Section for Track-OT++ */}
      {trackOtModel && (
        <div className="rounded-xl border-2 border-amber-500/60 bg-gradient-to-br from-amber-950/20 via-railway-surface/60 to-railway-darkest/90 p-5 shadow-2xl space-y-4 font-mono">
          <div className="flex flex-wrap items-start justify-between gap-3 pb-3 border-b border-amber-500/30">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <ShieldAlert className="h-5 w-5 text-amber-400 shrink-0" />
                <h3 className="text-sm font-bold text-white tracking-wider uppercase">
                  Separate Training & Validation Status: TRACK-OT++ (Multimodal Transformer)
                </h3>
              </div>
              <p className="text-[11px] text-amber-200/80 font-sans max-w-3xl">
                This model is quarantined from test rankings. Evaluation is strictly confined to Epoch 1 validation logs. Training was interrupted prior to convergence; held-out test evaluation was never conducted.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40 text-[10px] font-bold flex items-center gap-1">
                <XCircle className="h-3 w-3" />
                STATUS: TRAINING_INCOMPLETE
              </span>
              <span className="px-2.5 py-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[10px] font-bold">
                NOT PRODUCTION READY
              </span>
            </div>
          </div>

          {/* Forensic Metadata & Metrics Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3">
            <div className="p-3 rounded-lg bg-railway-darkest/90 border border-amber-500/30 space-y-1">
              <span className="text-[10px] text-railway-muted uppercase block">Training Status</span>
              <span className="text-sm font-bold text-rose-400 block">TRAINING_INCOMPLETE</span>
              <span className="text-[10px] text-railway-muted block">Run halted at Epoch 2</span>
            </div>

            <div className="p-3 rounded-lg bg-railway-darkest/90 border border-amber-500/30 space-y-1">
              <span className="text-[10px] text-railway-muted uppercase block">Epoch 1 Validation AP</span>
              <div className="flex items-baseline gap-1.5">
                <span className="text-xl font-bold text-amber-400">0.7916</span>
                <span className="text-[10px] text-amber-300 font-bold">AP</span>
              </div>
              <span className="text-[10px] text-railway-muted block">Validation partition only</span>
            </div>

            <div className="p-3 rounded-lg bg-railway-darkest/90 border border-amber-500/30 space-y-1">
              <span className="text-[10px] text-railway-muted uppercase block">Epoch 2 Interruption</span>
              <span className="text-sm font-bold text-white block">Batch 870 / 1743</span>
              <span className="text-[10px] text-railway-muted block">Progress: 49.9% of epoch</span>
            </div>

            <div className="p-3 rounded-lg bg-railway-darkest/90 border border-amber-500/30 space-y-1">
              <span className="text-[10px] text-railway-muted uppercase block">Failure Root Cause</span>
              <span className="text-[11px] font-bold text-rose-300 block truncate" title="non-finite gradient norm during gradient clipping">
                non-finite gradient norm
              </span>
              <span className="text-[9px] text-railway-muted block">during gradient clipping</span>
            </div>

            <div className="p-3 rounded-lg bg-railway-darkest/90 border border-amber-500/30 space-y-1">
              <span className="text-[10px] text-railway-muted uppercase block">Held-out Test Metrics</span>
              <span className="text-base font-bold text-rose-400 block">UNAVAILABLE</span>
              <span className="text-[10px] text-railway-muted block">Zero test peeking executed</span>
            </div>
          </div>

          {/* Audit Rule Adherence Box */}
          <div className="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/40 text-[11px] text-amber-200/90 font-sans space-y-1.5">
            <div className="flex items-center gap-1.5 font-bold text-amber-300 font-mono text-xs">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>FORENSIC SCIENTIFIC AUDIT RULES ENFORCED</span>
            </div>
            <ul className="list-disc pl-5 space-y-1 text-[11px]">
              <li>
                <b>No Test Ranking:</b> Validation AP (0.7916) is never compared, sorted, or ranked against other models' held-out test metrics.
              </li>
              <li>
                <b>No Missing Defaults:</b> Test metrics are displayed strictly as <code>UNAVAILABLE</code>, never substituted with zeroes or synthetic estimates.
              </li>
              <li>
                <b>Production Disclaimer:</b> This architecture remains an incomplete research artifact undergoing numerical gradient recovery in float32. It cannot be deployed to trackside edge inference.
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};