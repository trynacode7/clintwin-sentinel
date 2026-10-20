import { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';

/**
 * Custom Tooltip for What-If Trajectory Chart
 */
function TrajectoryTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="bg-slate-900/95 border border-slate-700/80 rounded-xl p-3 shadow-xl backdrop-blur-md text-xs min-w-[160px]">
      <p className="text-slate-400 font-mono mb-2 pb-1 border-b border-slate-800">
        Time: <span className="text-white font-semibold">{label}s</span>
      </p>
      <div className="space-y-1.5">
        {payload.map((entry, idx) => (
          <div key={idx} className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="text-slate-300 font-medium">{entry.name}:</span>
            </div>
            <span className="font-mono font-bold text-slate-100">
              {entry.value !== undefined && entry.value !== null
                ? Number(entry.value).toFixed(2)
                : '—'}{' '}
              <span className="text-[10px] text-slate-400 font-normal">{entry.unit}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Marker style configurations for up to 3 distinct sequential interventions
const INTERVENTION_MARKER_STYLES = [
  {
    startStroke: '#10b981', // Emerald
    startFill: '#34d399',
    endStroke: '#f59e0b',   // Amber
    endFill: '#fbbf24',
    name: 'Bolus 1',
  },
  {
    startStroke: '#06b6d4', // Cyan
    startFill: '#38bdf8',
    endStroke: '#a855f7',   // Purple
    endFill: '#c084fc',
    name: 'Bolus 2',
  },
  {
    startStroke: '#3b82f6', // Blue
    startFill: '#60a5fa',
    endStroke: '#ec4899',   // Pink
    endFill: '#f472b6',
    name: 'Bolus 3',
  },
];

/**
 * WhatIfTrajectoryChart — Physiological time-series trajectory visualizer.
 * Displays returned simulation data points without inventing intermediate values.
 * Strictly separates different units onto labeled Y-axes.
 * Supports distinct event markers and legend for 1, 2, or 3 sequential interventions.
 */
export default function WhatIfTrajectoryChart({
  trajectory,
  milestones = [],
  actionConfig = null,
  actions = null,
}) {
  const [activeTab, setActiveTab] = useState('hemodynamics'); // 'hemodynamics' | 'volume_flow'

  const dataPoints = trajectory?.data_points || [];

  // Compute marker list for 1 to 3 interventions
  const bolusMarkers = useMemo(() => {
    // If multi-action array provided in props
    if (Array.isArray(actions) && actions.length > 0) {
      return actions.map((act, idx) => {
        const style = INTERVENTION_MARKER_STYLES[idx % INTERVENTION_MARKER_STYLES.length];
        const actionType = act.type || (act.substance === 'Norepinephrine' ? 'norepinephrine' : 'saline');
        const isNorepi = actionType === 'norepinephrine';
        const vol = act.bag_volume_ml ?? act.volume_ml ?? act.volumeMl;
        const start = Number(act.start_time_s ?? act.startTimeS);
        const duration = Number(act.delivery_duration_s ?? act.duration_s ?? (isNorepi ? 120 : (vol ? (vol / 50.0) * 60 : 120)));
        const end = start + duration;
        const interventionNum = act.action_index !== undefined ? act.action_index + 1 : idx + 1;
        const name = isNorepi ? `Norepi ${interventionNum}` : `Bolus ${interventionNum}`;

        let startLabel = '';
        let endLabel = '';
        if (isNorepi) {
          startLabel = actions.length === 1 ? 'Norepi Start (14µg)' : `NE${interventionNum} Start (14µg)`;
          endLabel = actions.length === 1 ? 'Norepi End' : `NE${interventionNum} End`;
        } else {
          startLabel = actions.length === 1 ? 'Bolus Start' : `B${interventionNum} Start (${vol}mL)`;
          endLabel = actions.length === 1 ? 'Bolus End' : `B${interventionNum} End`;
        }

        return {
          id: `action_${interventionNum}`,
          name,
          type: actionType,
          vol: isNorepi ? null : vol,
          startTime: start,
          endTime: end,
          startLabel,
          endLabel,
          style,
        };
      });
    }

    // Fallback: single actionConfig or milestones extraction (preserving Step F backward compatibility)
    const startTime = actionConfig?.start_time_s ?? milestones.find(m => m.category === 'baseline')?.target_time_s;
    const endTime = actionConfig
      ? actionConfig.start_time_s + actionConfig.delivery_duration_s
      : milestones.find(m => m.phase?.startsWith('intervention_') || m.category === 'intervention')?.target_time_s;

    if (typeof startTime === 'number' || typeof endTime === 'number') {
      const vol = actionConfig?.bag_volume_ml ?? 100;
      return [
        {
          id: 'bolus_1',
          name: 'Bolus 1',
          vol,
          startTime,
          endTime,
          startLabel: 'Bolus Start',
          endLabel: 'Bolus End',
          style: INTERVENTION_MARKER_STYLES[0],
        },
      ];
    }

    return [];
  }, [actions, actionConfig, milestones]);

  if (!dataPoints || dataPoints.length === 0) {
    return (
      <div className="p-8 text-center bg-slate-900/40 rounded-xl border border-slate-800/60 text-slate-400 text-xs">
        <svg className="w-8 h-8 mx-auto mb-2 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
        <p className="font-medium text-slate-300">No trajectory data available</p>
        <p className="text-[11px] text-slate-500 mt-1">
          The simulation response did not include time-series data points.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-800/80 p-4">
      {/* Chart Header & Signal Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            Continuous Physiological Trajectory
            <span className="text-[10px] font-mono text-cyan-400/90 bg-cyan-950/40 border border-cyan-800/40 px-2 py-0.5 rounded">
              {dataPoints.length} points • 0.2 Hz
            </span>
            {bolusMarkers.length > 1 && (
              <span className="text-[10px] font-mono text-purple-300 bg-purple-950/40 border border-purple-800/40 px-2 py-0.5 rounded">
                {bolusMarkers.length} Interventions
              </span>
            )}
          </h4>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Real time-series output directly parsed from BioGears solver engine
          </p>
        </div>

        {/* View Switcher Buttons */}
        <div className="flex items-center gap-1 bg-slate-950/80 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab('hemodynamics')}
            className={`px-3 py-1 rounded-md transition-all font-medium ${
              activeTab === 'hemodynamics'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Hemodynamics (HR & MAP)
          </button>
          <button
            onClick={() => setActiveTab('volume_flow')}
            className={`px-3 py-1 rounded-md transition-all font-medium ${
              activeTab === 'volume_flow'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Volume & Perfusion (BV & CO)
          </button>
        </div>
      </div>

      {/* Main Chart */}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {activeTab === 'hemodynamics' ? (
            <ComposedChart
              data={dataPoints}
              margin={{ top: 10, right: 25, left: 10, bottom: 20 }}
            >
              <defs>
                <linearGradient id="hrLineGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#f43f5e" />
                  <stop offset="100%" stopColor="#fb7185" />
                </linearGradient>
                <linearGradient id="mapLineGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#06b6d4" />
                  <stop offset="100%" stopColor="#38bdf8" />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.08)" />

              <XAxis
                dataKey="time_s"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 10, fill: '#64748b' }}
                tickLine={false}
                axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
                unit="s"
                label={{
                  value: 'Continuous Simulation Time (seconds)',
                  position: 'insideBottom',
                  offset: -12,
                  fill: '#64748b',
                  fontSize: 10,
                }}
              />

              {/* Left Y Axis for Heart Rate (bpm) */}
              <YAxis
                yAxisId="hr"
                orientation="left"
                domain={['auto', 'auto']}
                tick={{ fontSize: 10, fill: '#f43f5e' }}
                tickLine={false}
                axisLine={{ stroke: 'rgba(244, 63, 94, 0.3)' }}
                width={45}
                unit=" bpm"
              />

              {/* Right Y Axis for MAP (mmHg) */}
              <YAxis
                yAxisId="map"
                orientation="right"
                domain={['auto', 'auto']}
                tick={{ fontSize: 10, fill: '#06b6d4' }}
                tickLine={false}
                axisLine={{ stroke: 'rgba(6, 182, 212, 0.3)' }}
                width={50}
                unit=" mmHg"
              />

              <Tooltip content={<TrajectoryTooltip />} />

              {/* Sequential Intervention Reference Lines */}
              {bolusMarkers.map((marker) => (
                <g key={marker.id}>
                  {typeof marker.startTime === 'number' && (
                    <ReferenceLine
                      key={`${marker.id}_start`}
                      x={marker.startTime}
                      stroke={marker.style.startStroke}
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                      label={{
                        value: marker.startLabel,
                        position: 'insideTopLeft',
                        fill: marker.style.startFill,
                        fontSize: 10,
                      }}
                    />
                  )}
                  {typeof marker.endTime === 'number' && (
                    <ReferenceLine
                      key={`${marker.id}_end`}
                      x={marker.endTime}
                      stroke={marker.style.endStroke}
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                      label={{
                        value: marker.endLabel,
                        position: 'insideTopRight',
                        fill: marker.style.endFill,
                        fontSize: 10,
                      }}
                    />
                  )}
                </g>
              ))}

              <Line
                yAxisId="hr"
                type="monotone"
                dataKey="heart_rate_bpm"
                name="Heart Rate"
                unit="bpm"
                stroke="url(#hrLineGrad)"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 5, fill: '#f43f5e', stroke: '#fff', strokeWidth: 2 }}
                isAnimationActive={false}
              />

              <Line
                yAxisId="map"
                type="monotone"
                dataKey="mean_arterial_pressure_mmhg"
                name="MAP"
                unit="mmHg"
                stroke="url(#mapLineGrad)"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 5, fill: '#06b6d4', stroke: '#fff', strokeWidth: 2 }}
                isAnimationActive={false}
              />
            </ComposedChart>
          ) : (
            <ComposedChart
              data={dataPoints}
              margin={{ top: 10, right: 25, left: 10, bottom: 20 }}
            >
              <defs>
                <linearGradient id="bvLineGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="100%" stopColor="#34d399" />
                </linearGradient>
                <linearGradient id="coLineGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#8b5cf6" />
                  <stop offset="100%" stopColor="#a78bfa" />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.08)" />

              <XAxis
                dataKey="time_s"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 10, fill: '#64748b' }}
                tickLine={false}
                axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
                unit="s"
                label={{
                  value: 'Continuous Simulation Time (seconds)',
                  position: 'insideBottom',
                  offset: -12,
                  fill: '#64748b',
                  fontSize: 10,
                }}
              />

              {/* Left Y Axis for Blood Volume (L) */}
              <YAxis
                yAxisId="bv"
                orientation="left"
                domain={['auto', 'auto']}
                tick={{ fontSize: 10, fill: '#10b981' }}
                tickLine={false}
                axisLine={{ stroke: 'rgba(16, 185, 129, 0.3)' }}
                width={45}
                unit=" L"
              />

              {/* Right Y Axis for Cardiac Output (L/min) */}
              <YAxis
                yAxisId="co"
                orientation="right"
                domain={['auto', 'auto']}
                tick={{ fontSize: 10, fill: '#8b5cf6' }}
                tickLine={false}
                axisLine={{ stroke: 'rgba(139, 92, 246, 0.3)' }}
                width={50}
                unit=" L/min"
              />

              <Tooltip content={<TrajectoryTooltip />} />

              {/* Sequential Intervention Reference Lines */}
              {bolusMarkers.map((marker) => (
                <g key={marker.id}>
                  {typeof marker.startTime === 'number' && (
                    <ReferenceLine
                      key={`${marker.id}_start`}
                      x={marker.startTime}
                      stroke={marker.style.startStroke}
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                      label={{
                        value: marker.startLabel,
                        position: 'insideTopLeft',
                        fill: marker.style.startFill,
                        fontSize: 10,
                      }}
                    />
                  )}
                  {typeof marker.endTime === 'number' && (
                    <ReferenceLine
                      key={`${marker.id}_end`}
                      x={marker.endTime}
                      stroke={marker.style.endStroke}
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                      label={{
                        value: marker.endLabel,
                        position: 'insideTopRight',
                        fill: marker.style.endFill,
                        fontSize: 10,
                      }}
                    />
                  )}
                </g>
              ))}

              <Line
                yAxisId="bv"
                type="monotone"
                dataKey="blood_volume_l"
                name="Blood Volume"
                unit="L"
                stroke="url(#bvLineGrad)"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 5, fill: '#10b981', stroke: '#fff', strokeWidth: 2 }}
                isAnimationActive={false}
              />

              <Line
                yAxisId="co"
                type="monotone"
                dataKey="cardiac_output_l_min"
                name="Cardiac Output"
                unit="L/min"
                stroke="url(#coLineGrad)"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 5, fill: '#8b5cf6', stroke: '#fff', strokeWidth: 2 }}
                isAnimationActive={false}
              />
            </ComposedChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Chart Footer Legend / Info */}
      <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 pt-3 border-t border-slate-800/80 mt-2 gap-2">
        <div className="flex items-center gap-4">
          {activeTab === 'hemodynamics' ? (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-0.5 bg-rose-500 rounded" />
                <span className="text-slate-300">Heart Rate (bpm) — Left Axis</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-0.5 bg-cyan-400 rounded" />
                <span className="text-slate-300">MAP (mmHg) — Right Axis</span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-0.5 bg-emerald-500 rounded" />
                <span className="text-slate-300">Blood Volume (L) — Left Axis</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-0.5 bg-purple-400 rounded" />
                <span className="text-slate-300">Cardiac Output (L/min) — Right Axis</span>
              </div>
            </>
          )}
        </div>

        {/* Multi-bolus event markers legend */}
        <div className="flex flex-wrap items-center gap-3 text-[10px] text-slate-400">
          {bolusMarkers.length > 1 ? (
            bolusMarkers.map((bm) => (
              <div key={bm.id} className="flex items-center gap-1.5 font-mono">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: bm.style.startStroke }} />
                <span>
                  {bm.name} ({bm.type === 'norepinephrine' ? '14µg/mL' : `${bm.vol}mL`}: {bm.startTime}s–{bm.endTime}s)
                </span>
              </div>
            ))
          ) : (
            <span>Dashed vertical lines indicate intervention delivery window</span>
          )}
        </div>
      </div>
    </div>
  );
}
