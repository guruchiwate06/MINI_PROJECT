import React from 'react';
import { DataIntegrityClassification } from '../../types';

interface Props {
  type: DataIntegrityClassification;
  size?: 'sm' | 'md';
}

const BADGE_CONFIG: Record<DataIntegrityClassification, { label: string; bg: string; text: string; border: string; desc: string }> = {
  MEASURED: {
    label: '[MEASURED]',
    bg: 'bg-hud-cyan/10',
    text: 'text-hud-cyan',
    border: 'border-hud-cyan/30',
    desc: 'Empirical 2 kHz telemetry digitized directly on the vehicle'
  },
  GROUND_TRUTH: {
    label: '[GROUND_TRUTH]',
    bg: 'bg-hud-amber/10',
    text: 'text-hud-amber',
    border: 'border-hud-amber/30',
    desc: 'Table 2 physically verified coordinates from Nature Scientific Data (2026)'
  },
  ML_PREDICTION: {
    label: '[ML_PREDICTION]',
    bg: 'bg-purple-500/10',
    text: 'text-purple-400',
    border: 'border-purple-500/30',
    desc: 'Audited model inference output from run_benchmarks.py'
  },
  SIMULATED: {
    label: '[SIMULATED]',
    bg: 'bg-blue-500/10',
    text: 'text-blue-400',
    border: 'border-blue-500/30',
    desc: 'Virtual kinematic playback and track trajectory simulation'
  },
  PROCEDURAL: {
    label: '[PROCEDURAL]',
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    border: 'border-emerald-500/30',
    desc: 'Real-time Web Audio API procedural acoustic synthesis'
  },
  ILLUSTRATIVE: {
    label: '[ILLUSTRATIVE]',
    bg: 'bg-gray-500/10',
    text: 'text-gray-400',
    border: 'border-gray-500/30',
    desc: 'Visual buffer guide or sensor frustum overlay'
  }
};

export const DataIntegrityBadge: React.FC<Props> = ({ type, size = 'sm' }) => {
  const cfg = BADGE_CONFIG[type];
  const sizeClasses = size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2.5 py-1 text-xs';

  return (
    <span
      title={cfg.desc}
      className={`inline-flex items-center font-mono font-medium rounded border ${cfg.bg} ${cfg.text} ${cfg.border} ${sizeClasses} tracking-wider transition-colors`}
    >
      {cfg.label}
    </span>
  );
};

