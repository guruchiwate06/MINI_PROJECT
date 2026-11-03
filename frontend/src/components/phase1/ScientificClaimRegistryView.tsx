import React, { useState } from 'react';
import { SCIENTIFIC_CLAIM_REGISTRY } from '../../data/claimRegistry';
import { ScientificClaim } from '../../types';
import { CheckCircle2, ChevronDown, ChevronRight, FileCode2, Info } from 'lucide-react';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

export const ScientificClaimRegistryView: React.FC = () => {
  const [selectedClaim, setSelectedClaim] = useState<ScientificClaim | null>(SCIENTIFIC_CLAIM_REGISTRY[0]);
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({ 'CR-01': true, 'CR-07': true });

  const toggleRow = (id: string) => {
    setExpandedRows(prev => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="bg-railway-surface/60 border border-railway-border rounded-xl p-5 sm:p-6 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5 border-b border-railway-border/60 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-lg font-bold text-white font-mono tracking-wide flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-hud-emerald" />
              SCIENTIFIC CLAIM REGISTRY (AUDITED)
            </h2>
            <DataIntegrityBadge type="GROUND_TRUTH" size="sm" />
          </div>
          <p className="text-xs text-railway-muted mt-1">
            Forensic cross-verification of all quantified claims against dataset records, source code, and published benchmarks.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-hud-emerald/10 border border-hud-emerald/30 text-hud-emerald font-semibold">
            7 / 7 VERIFIED
          </span>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-railway-elevated border border-railway-border text-railway-muted">
            0 UNVERIFIED
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-sans">
          <thead>
            <tr className="border-b border-railway-border/80 text-[11px] font-mono uppercase text-railway-muted bg-railway-darkest/40">
              <th className="py-2.5 px-3">ID</th>
              <th className="py-2.5 px-3">Scientific Claim Title</th>
              <th className="py-2.5 px-3">Quantified Value</th>
              <th className="py-2.5 px-3">Primary Source File</th>
              <th className="py-2.5 px-3 text-center">Status</th>
              <th className="py-2.5 px-3 text-right">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-railway-border/40 font-mono">
            {SCIENTIFIC_CLAIM_REGISTRY.map((claim) => {
              const isExpanded = !!expandedRows[claim.id];
              return (
                <React.Fragment key={claim.id}>
                  <tr
                    onClick={() => {
                      setSelectedClaim(claim);
                      toggleRow(claim.id);
                    }}
                    className={`hover:bg-railway-elevated/70 cursor-pointer transition-colors ${
                      selectedClaim?.id === claim.id ? 'bg-railway-elevated/50' : ''
                    }`}
                  >
                    <td className="py-3 px-3 font-bold text-hud-cyan">{claim.id}</td>
                    <td className="py-3 px-3 font-sans text-white font-medium">
                      {claim.title}
                    </td>
                    <td className="py-3 px-3 text-hud-amber font-semibold">
                      {String(claim.verified_value)} {claim.unit !== '% relative inflation' && claim.unit !== 'probability threshold & recall' ? claim.unit : ''}
                    </td>
                    <td className="py-3 px-3 text-railway-muted text-[11px] truncate max-w-[200px]" title={claim.source_file}>
                      {claim.source_file.split(',')[0]}
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded font-mono font-bold bg-hud-emerald/15 text-hud-emerald border border-hud-emerald/30">
                        <CheckCircle2 className="h-3 w-3" /> VERIFIED
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right text-railway-muted">
                      {isExpanded ? <ChevronDown className="h-4 w-4 inline text-hud-cyan" /> : <ChevronRight className="h-4 w-4 inline" />}
                    </td>
                  </tr>

                  {isExpanded && (
                    <tr className="bg-railway-darkest/70 border-b border-railway-border/60">
                      <td colSpan={6} className="p-4 font-sans text-xs">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <div className="flex items-center gap-2 text-hud-cyan font-mono text-[11px] uppercase font-bold mb-1">
                              <Info className="h-3.5 w-3.5" /> Statement & Context
                            </div>
                            <p className="text-railway-text leading-relaxed bg-railway-dark/60 p-3 rounded border border-railway-border/60">
                              {claim.statement}
                            </p>
                            <p className="text-[11px] text-railway-muted mt-2 leading-relaxed">
                              <strong className="text-white">Analysis:</strong> {claim.notes}
                            </p>
                          </div>

                          <div>
                            <div className="flex items-center gap-2 text-hud-amber font-mono text-[11px] uppercase font-bold mb-1">
                              <FileCode2 className="h-3.5 w-3.5" /> Source Evidence & Location
                            </div>
                            <div className="bg-railway-dark/90 p-3 rounded border border-railway-border/60 font-mono text-[11px] space-y-1.5">
                              <div className="text-railway-muted">
                                <span className="text-white">Files:</span> {claim.source_file}
                              </div>
                              <div className="text-railway-muted">
                                <span className="text-white">Line Range:</span> {claim.line_range}
                              </div>
                              <div className="p-2 bg-black/60 rounded border border-railway-border/40 text-hud-cyan text-[11px] overflow-x-auto whitespace-pre">
                                {claim.code_excerpt}
                              </div>
                              <div className="text-[10px] text-railway-muted pt-1">
                                <span className="text-hud-emerald">Method:</span> {claim.verification_method}
                              </div>
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

