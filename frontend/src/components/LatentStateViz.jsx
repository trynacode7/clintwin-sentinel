import { motion } from 'framer-motion';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

/**
 * LatentStateViz — 2D projection of z_t (digital twin state)
 * 
 * Visualizes the patient's trajectory through the latent state space.
 * z_t is the model's learned representation of the patient's physiological
 * state — projected to 2D via PCA for visualization.
 * 
 * The trajectory shows how the patient's state evolves over time:
 * - Stable patients cluster in one region
 * - Deteriorating patients drift toward high-risk zones
 * - Color gradient from green (low risk) to red (high risk)
 */

function getRiskColor(risk) {
  if (risk < 0.3) return '#10b981';
  if (risk < 0.5) return '#f59e0b';
  if (risk < 0.7) return '#f97316';
  return '#ef4444';
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <p className="text-xs text-slate-400 mb-1">Timestep {d.t_idx}</p>
      <p className="text-sm font-medium" style={{ color: getRiskColor(d.risk_score) }}>
        Risk: {(d.risk_score * 100).toFixed(0)}%
      </p>
      <p className="text-xs text-slate-500">
        z₁: {d.x?.toFixed(2)}, z₂: {d.y?.toFixed(2)}
      </p>
    </div>
  );
}

function StateCategoryLegend({ twinState }) {
  const simAvailable = twinState?.simulated?.available;
  return (
    <div className="flex items-center gap-1.5 mb-3 text-[9px]">
      <span
        className="px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
        title={twinState?.observed?.description}
      >
        OBSERVED · real vitals
      </span>
      <span
        className="px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400 border border-cyan-500/30"
        title={twinState?.predicted?.description}
      >
        PREDICTED · this chart
      </span>
      <span
        className={`px-2 py-0.5 rounded-full border ${
          simAvailable
            ? 'bg-violet-500/15 text-violet-400 border-violet-500/30'
            : 'bg-slate-800/60 text-slate-600 border-slate-700/50'
        }`}
        title={twinState?.simulated?.description}
      >
        SIMULATED · {simAvailable ? 'available' : 'not yet available'}
      </span>
      <span
        className="px-2 py-0.5 rounded-full bg-slate-800/60 text-slate-600 border border-slate-700/50"
        title="A real ML counterfactual (same model, re-run on a hypothetically modified input) is available on demand from the What-If panel below — it applies to a proposed scenario, not this window's actual state, so it isn't shown here."
      >
        COUNTERFACTUAL · see What-If panel
      </span>
    </div>
  );
}

export default function LatentStateViz({ trajectory, disclaimer, twinState }) {
  if (!trajectory?.length) {
    return (
      <motion.div
        className="glass-panel p-5 flex items-center justify-center h-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <p className="text-sm text-slate-500">No trajectory data available</p>
      </motion.div>
    );
  }

  // Transform trajectory for scatter chart
  const scatterData = trajectory.map((pt, i) => ({
    x: pt.z_2d?.[0] || 0,
    y: pt.z_2d?.[1] || 0,
    risk_score: pt.risk_score || 0,
    t_idx: pt.t_idx ?? i,
    size: i === trajectory.length - 1 ? 120 : 40 + (i / trajectory.length) * 30,
  }));

  // Compute bounds with padding
  const xs = scatterData.map(d => d.x);
  const ys = scatterData.map(d => d.y);
  const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.15 || 1;
  const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.15 || 1;

  return (
    <motion.div
      className="glass-panel p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Latent State Trajectory
        </h2>
        <span className="text-xs text-slate-600">
          z_t = PREDICTED latent state (PCA 2D)
        </span>
      </div>

      <StateCategoryLegend twinState={twinState} />

      {/* Legend */}
      <div className="flex items-center gap-4 mb-3 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-emerald-500" />
          <span className="text-slate-500">Low Risk</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-amber-500" />
          <span className="text-slate-500">Medium</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-red-500" />
          <span className="text-slate-500">High Risk</span>
        </div>
        <div className="flex items-center gap-1.5 ml-auto">
          <svg width="24" height="8">
            <line x1="0" y1="4" x2="24" y2="4" stroke="#475569" strokeWidth="1.5" strokeDasharray="3 2" />
          </svg>
          <span className="text-slate-500">Trajectory</span>
        </div>
      </div>

      <div className="h-64 relative">
        {/* Trajectory path overlay */}
        <svg
          className="absolute inset-0 pointer-events-none z-10"
          width="100%"
          height="100%"
          viewBox={`${Math.min(...xs) - xPad} ${-(Math.max(...ys) + yPad)} ${Math.max(...xs) - Math.min(...xs) + 2 * xPad} ${Math.max(...ys) - Math.min(...ys) + 2 * yPad}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ opacity: 0.4 }}
        >
          <polyline
            points={scatterData.map(d => `${d.x},${-d.y}`).join(' ')}
            fill="none"
            stroke="#475569"
            strokeWidth="0.03"
            strokeDasharray="0.05 0.03"
          />
        </svg>

        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, left: -15, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.08)" />
            <XAxis
              dataKey="x"
              type="number"
              domain={[Math.min(...xs) - xPad, Math.max(...xs) + xPad]}
              tick={{ fontSize: 9, fill: '#64748b' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(148, 163, 184, 0.1)' }}
              name="z₁"
            />
            <YAxis
              dataKey="y"
              type="number"
              domain={[Math.min(...ys) - yPad, Math.max(...ys) + yPad]}
              tick={{ fontSize: 9, fill: '#64748b' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(148, 163, 184, 0.1)' }}
              name="z₂"
            />
            <Tooltip content={<CustomTooltip />} />
            <Scatter data={scatterData} name="Patient State">
              {scatterData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={getRiskColor(entry.risk_score)}
                  fillOpacity={0.3 + (i / scatterData.length) * 0.7}
                  stroke={i === scatterData.length - 1 ? '#fff' : getRiskColor(entry.risk_score)}
                  strokeWidth={i === scatterData.length - 1 ? 2 : 0.5}
                  r={i === scatterData.length - 1 ? 7 : 3 + (i / scatterData.length) * 2}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Current state info */}
      {scatterData.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-800 flex items-center justify-between text-xs">
          <span className="text-slate-500">
            {scatterData.length} timesteps tracked
          </span>
          <span className="text-slate-400 font-mono">
            Current: z₁={scatterData[scatterData.length - 1].x.toFixed(2)},
            z₂={scatterData[scatterData.length - 1].y.toFixed(2)}
          </span>
        </div>
      )}

      {/* Disclaimer text — comes verbatim from the backend
          (backend/inference_service.py get_twin_state), not restated or
          reworded here, so it stays in sync with whatever the backend
          actually says this representation is. */}
      {disclaimer && (
        <p className="mt-2 text-[10px] text-slate-600 leading-relaxed">{disclaimer}</p>
      )}
    </motion.div>
  );
}
