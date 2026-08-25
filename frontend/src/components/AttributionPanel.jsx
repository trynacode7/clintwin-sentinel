import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

/**
 * AttributionPanel — Feature attribution visualization
 *
 * Phase D/K: renamed from ShapPanel. The values shown here are REAL
 * input×gradient saliency computed by the trained model
 * (backend/explainability.py::gradient_x_input_attribution), never
 * fabricated — but they are deliberately never called "SHAP" anywhere
 * in this component, because they aren't SHAP (no Shapley-value game-
 * theoretic computation is involved). See PHASE_D_REPORT.md.
 *
 * Displays feature attribution as:
 * 1. Horizontal bar chart (per-window importance, sorted)
 * 2. A short list showing contribution direction
 *
 * Red bars = pushes toward higher risk
 * Blue bars = pushes toward lower risk (protective)
 */

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <p className="text-xs font-medium text-slate-300 mb-1">{d.feature}</p>
      <p className="text-sm" style={{ color: d.value > 0 ? '#f87171' : '#60a5fa' }}>
        {d.value > 0 ? '+' : ''}{d.value?.toFixed(3)}
      </p>
      <p className="text-xs text-slate-500 mt-0.5">
        {d.value > 0 ? '↑ Increases risk' : '↓ Decreases risk'}
      </p>
    </div>
  );
}

export default function AttributionPanel({ attributionData, method }) {
  if (!attributionData?.length) {
    return (
      <motion.div className="glass-panel p-5 flex items-center justify-center h-full">
        <p className="text-sm text-slate-500">No attribution data available</p>
      </motion.div>
    );
  }

  // Sort by absolute importance
  const sorted = [...attributionData]
    .sort((a, b) => Math.abs(b.value ?? b.importance) - Math.abs(a.value ?? a.importance))
    .slice(0, 12);

  const chartData = sorted.map((d) => ({
    feature: d.feature,
    value: d.value ?? (d.direction === 'risk' ? d.importance : -d.importance),
    importance: d.importance ?? Math.abs(d.value),
  }));

  return (
    <motion.div
      className="glass-panel p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Feature Attribution
        </h2>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: '#f87171' }} />
            <span className="text-slate-500">Risk ↑</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: '#60a5fa' }} />
            <span className="text-slate-500">Protective ↓</span>
          </div>
        </div>
      </div>

      {/* Bar Chart */}
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 15, left: 5, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.08)" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 9, fill: '#64748b' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(148, 163, 184, 0.1)' }}
              domain={['auto', 'auto']}
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(148, 163, 184, 0.1)' }}
              width={110}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(148, 163, 184, 0.05)' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
              {chartData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.value > 0 ? 'rgba(248, 113, 113, 0.7)' : 'rgba(96, 165, 250, 0.7)'}
                  stroke={entry.value > 0 ? '#f87171' : '#60a5fa'}
                  strokeWidth={1}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary list */}
      <div className="mt-3 pt-3 border-t border-slate-800">
        <div className="space-y-2">
          {chartData.slice(0, 5).map((item, i) => (
            <motion.div
              key={item.feature}
              className="flex items-center justify-between text-xs"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
            >
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <div
                  className="w-1.5 h-4 rounded-full flex-shrink-0"
                  style={{ background: item.value > 0 ? '#f87171' : '#60a5fa' }}
                />
                <span className="text-slate-400 truncate">{item.feature}</span>
              </div>
              <div className="flex items-center gap-2 ml-2">
                <span className="font-mono text-slate-300" style={{ color: item.value > 0 ? '#fca5a5' : '#93c5fd' }}>
                  {item.value > 0 ? '+' : ''}{item.value.toFixed(3)}
                </span>
                <div
                  className="h-1.5 rounded-full"
                  style={{
                    width: `${Math.abs(item.value) * 200}px`,
                    background: item.value > 0
                      ? 'linear-gradient(90deg, rgba(248,113,113,0.3), rgba(248,113,113,0.8))'
                      : 'linear-gradient(90deg, rgba(96,165,250,0.3), rgba(96,165,250,0.8))',
                    maxWidth: '80px',
                  }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Method disclosure — real backend-authored text, not restated here */}
      {method && (
        <p className="mt-3 text-[10px] text-slate-600 leading-relaxed">
          {method}
        </p>
      )}
    </motion.div>
  );
}
