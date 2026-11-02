import React, { useState } from 'react';
import { HARDWARE_SPECS } from '../../data/hardwareSpecs';
import { GROUND_TRUTH_ANOMALIES } from '../../data/groundTruthAnomalies';
import { Cpu, Gauge, Layers } from 'lucide-react';
import { DataIntegrityBadge } from '../common/DataIntegrityBadge';

export const DatasetArchitectureView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'anomalies' | 'hardware' | 'runs'>('anomalies');

  return (
    <div className="bg-railway-surface/60 border border-railway-border rounded-xl p-5 sm:p-6 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5 border-b border-railway-border/60 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-lg font-bold text-white font-mono tracking-wide flex items-center gap-2">
              <Layers className="h-5 w-5 text-hud-cyan" />
              PHYSICAL TESTBED & DATASET ARCHITECTURE
            </h2>
            <DataIntegrityBadge type="MEASURED" size="sm" />
          </div>
          <p className="text-xs text-railway-muted mt-1">
            Empirical hardware rig, transit parameters, and the 9 verified physical track anomalies at Orbisonia, PA.
          </p>
        </div>

        {/* Tab switchers */}
        <div className="flex rounded-lg bg-railway-darkest p-1 border border-railway-border/70 text-xs font-mono">
          <button
            onClick={() => setActiveTab('anomalies')}
            className={`px-3 py-1.5 rounded transition-colors ${
              activeTab === 'anomalies'
                ? 'bg-railway-elevated text-hud-cyan font-bold border border-hud-cyan/40 shadow-sm'
                : 'text-railway-muted hover:text-white'
            }`}
          >
            Ground Truth (N=9)
          </button>
          <button
            onClick={() => setActiveTab('hardware')}
            className={`px-3 py-1.5 rounded transition-colors ${
              activeTab === 'hardware'
                ? 'bg-railway-elevated text-hud-cyan font-bold border border-hud-cyan/40 shadow-sm'
                : 'text-railway-muted hover:text-white'
            }`}
          >
            Sensor Rig & Class 230
          </button>
          <button
            onClick={() => setActiveTab('runs')}
            className={`px-3 py-1.5 rounded transition-colors ${
              activeTab === 'runs'
                ? 'bg-railway-elevated text-hud-cyan font-bold border border-hud-cyan/40 shadow-sm'
                : 'text-railway-muted hover:text-white'
            }`}
          >
            20 Experimental Runs
          </button>
        </div>
      </div>

      {/* Tab: 9 Ground-Truth Anomalies */}
      {activeTab === 'anomalies' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-3 bg-railway-dark/60 rounded-lg border border-railway-border/60">
              <span className="text-[10px] text-railway-muted uppercase font-mono tracking-wider block">Mechanical Joint Gaps</span>
              <span className="text-xl font-bold font-mono text-hud-amber">7 Locations</span>
              <span className="text-[11px] text-railway-muted block mt-0.5">Expansion gaps 2.0 to 3.0 cm</span>
            </div>
            <div className="p-3 bg-railway-dark/60 rounded-lg border border-railway-border/60">
              <span className="text-[10px] text-railway-muted uppercase font-mono tracking-wider block">Track Switches (Turnouts)</span>
              <span className="text-xl font-bold font-mono text-hud-cyan">2 Locations</span>
              <span className="text-[11px] text-railway-muted block mt-0.5">1 Divergence, 1 Convergence</span>
            </div>
            <div className="p-3 bg-railway-dark/60 rounded-lg border border-railway-border/60">
              <span className="text-[10px] text-railway-muted uppercase font-mono tracking-wider block">Track Span & Spacing</span>
              <span className="text-xl font-bold font-mono text-white">1,245.6 m</span>
              <span className="text-[11px] text-railway-muted block mt-0.5">Cumulative offset Anom 1 to 9</span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-railway-border/80 text-[10px] uppercase text-railway-muted bg-railway-darkest/40">
                  <th className="py-2 px-3">ID</th>
                  <th className="py-2 px-3">Physical Defect Type</th>
                  <th className="py-2 px-3">Dimension</th>
                  <th className="py-2 px-3">Latitude / Longitude</th>
                  <th className="py-2 px-3">Offset from Anom 1</th>
                  <th className="py-2 px-3">Marker Cone</th>
                  <th className="py-2 px-3 text-right">3D Scan Available</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-railway-border/30">
                {GROUND_TRUTH_ANOMALIES.map((a) => (
                  <tr key={a.id} className="hover:bg-railway-elevated/40 transition-colors">
                    <td className="py-2.5 px-3 font-bold text-hud-cyan">A{a.id}</td>
                    <td className="py-2.5 px-3 text-white font-sans">{a.type}</td>
                    <td className="py-2.5 px-3 text-hud-amber font-semibold">
                      {a.size_cm !== null ? `${a.size_cm.toFixed(1)} cm` : 'Turnout geometry'}
                    </td>
                    <td className="py-2.5 px-3 text-railway-muted text-[11px]">
                      {a.lat.toFixed(5)}° N, {a.lon.toFixed(5)}° W
                    </td>
                    <td className="py-2.5 px-3 text-railway-muted">
                      +{a.cum_track_offset_m.toFixed(1)} m
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        a.cone_color === 'Orange' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 'bg-gray-700 text-gray-300'
                      }`}>
                        ▲ {a.cone_color} Cone
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      {a.point_cloud_available ? (
                        <span className="text-hud-emerald text-[11px] font-semibold" title={a.scan_file}>
                          ✓ Available (.fls)
                        </span>
                      ) : (
                        <span className="text-railway-muted text-[11px]">✗ Missing in archive</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab: Hardware Specifications */}
      {activeTab === 'hardware' && (
        <div className="space-y-4">
          <div className="p-4 bg-railway-dark/80 rounded-lg border border-railway-border/60">
            <div className="flex items-center gap-2 text-sm font-bold text-white font-mono mb-2">
              <Gauge className="h-4 w-4 text-hud-cyan" />
              TESTBED VEHICLE: {HARDWARE_SPECS.train.model}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono pt-1">
              <div>
                <span className="text-railway-muted text-[10px] block">Operator / Line</span>
                <span className="text-white">{HARDWARE_SPECS.train.operator}</span>
              </div>
              <div>
                <span className="text-railway-muted text-[10px] block">Location</span>
                <span className="text-white">{HARDWARE_SPECS.train.track_location}</span>
              </div>
              <div>
                <span className="text-railway-muted text-[10px] block">Track Length</span>
                <span className="text-white">{HARDWARE_SPECS.train.track_length_m} m</span>
              </div>
              <div>
                <span className="text-railway-muted text-[10px] block">Speed Matrix</span>
                <span className="text-hud-cyan font-bold">{HARDWARE_SPECS.train.speed_range_kmh}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {HARDWARE_SPECS.sensors.map((sensor, idx) => (
              <div key={idx} className="p-4 bg-railway-dark/50 rounded-lg border border-railway-border/60 space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="text-xs font-bold text-white font-mono flex items-center gap-1.5">
                      <Cpu className="h-3.5 w-3.5 text-hud-cyan" />
                      {sensor.name}
                    </h3>
                    <span className="text-[10px] text-railway-muted font-sans">{sensor.type}</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-hud-cyan/10 text-hud-cyan border border-hud-cyan/30">
                    {sensor.sampling_rate}
                  </span>
                </div>

                <div className="text-xs font-mono space-y-1 pt-1">
                  <div className="text-railway-text text-[11px]">
                    <span className="text-railway-muted">Mounting:</span> {sensor.location}
                  </div>
                  <div className="text-hud-amber text-[11px]">
                    <span className="text-railway-muted">Channels:</span> {sensor.channels.join(', ')}
                  </div>
                  <div className="text-[10px] text-railway-muted pt-1 border-t border-railway-border/40 font-sans">
                    {sensor.specs}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab: 20 Experimental Runs Matrix */}
      {activeTab === 'runs' && (
        <div className="space-y-3 font-mono text-xs">
          <div className="p-3 bg-railway-dark/60 rounded-lg border border-railway-border/60 flex flex-wrap items-center justify-between gap-2">
            <div>
              <span className="text-white font-bold">20 Transits Recorded across 2 Target Dates</span>
              <p className="text-[11px] text-railway-muted font-sans mt-0.5">
                April 11, 2025 (Runs 1-3) and April 18, 2025 (Run 4). 9,401,255 total rows digitized at 2 kHz.
              </p>
            </div>
            <div className="flex gap-2">
              <span className="px-2 py-1 rounded bg-hud-cyan/15 text-hud-cyan text-[11px] font-bold">
                8 Uphill Transits (AtoB)
              </span>
              <span className="px-2 py-1 rounded bg-hud-amber/15 text-hud-amber text-[11px] font-bold">
                12 Downhill Transits (BtoA)
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-1">
            <div className="p-3 bg-railway-dark/40 rounded border border-railway-border/50">
              <span className="text-hud-cyan font-bold block text-sm">AtoB 20.0 km/h</span>
              <span className="text-[11px] text-railway-muted">4 Runs (Index 1-4)</span>
              <span className="text-[10px] text-railway-muted block mt-1">~288s avg transit</span>
            </div>
            <div className="p-3 bg-railway-dark/40 rounded border border-railway-border/50">
              <span className="text-hud-cyan font-bold block text-sm">AtoB 22.5 km/h</span>
              <span className="text-[11px] text-railway-muted">4 Runs (Index 1-4)</span>
              <span className="text-[10px] text-railway-muted block mt-1">~246s avg transit</span>
            </div>
            <div className="p-3 bg-railway-dark/40 rounded border border-railway-border/50">
              <span className="text-hud-amber font-bold block text-sm">BtoA 22.5 km/h</span>
              <span className="text-[11px] text-railway-muted">4 Runs (Index 1-4)</span>
              <span className="text-[10px] text-railway-muted block mt-1">~259s avg transit</span>
            </div>
            <div className="p-3 bg-railway-dark/40 rounded border border-railway-border/50">
              <span className="text-hud-amber font-bold block text-sm">BtoA 25.0 km/h</span>
              <span className="text-[11px] text-railway-muted">4 Runs (Index 1-4)</span>
              <span className="text-[10px] text-railway-muted block mt-1">~203s avg transit</span>
            </div>
            <div className="p-3 bg-railway-dark/40 rounded border border-railway-border/50">
              <span className="text-hud-amber font-bold block text-sm">BtoA 27.5 km/h</span>
              <span className="text-[11px] text-railway-muted">4 Runs (Index 1-4)</span>
              <span className="text-[10px] text-railway-muted block mt-1">~191s avg transit</span>
            </div>
            <div className="p-3 bg-railway-dark/40 rounded border border-railway-border/50 col-span-1 sm:col-span-3 flex items-center justify-between px-4">
              <div>
                <span className="text-white font-bold block">Zero Missingness Guarantee</span>
                <span className="text-[11px] text-hud-emerald">0.00% NaNs detected across all 9.4M rows</span>
              </div>
              <span className="text-railway-muted text-[10px]">Silicon Designs 16-bit DAQ</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
