import { motion } from 'framer-motion';
import { useMemo } from 'react';

/**
 * AttentionHeatmap - Matrix visualization of self-attention weights
 * OR Physiological Attribution Breakdown when attention is not supported by the model architecture.
 *
 * - When a Transformer model is active, displays the multi-head self-attention matrix.
 * - When the active model is BiLSTM (which processes sequences recurrently without attention weights),
 *   displays an accurate, useful breakdown of the model's directional input-gradient physiological drivers,
 *   never relabeling them as attention.
 * - If neither is available, displays a clean "Feature attribution unavailable" state.
 */

function getHeatColor(value, maxVal) {
  const normalized = value / (maxVal || 1);
  if (normalized < 0.2) return 'rgba(15, 23, 42, 0.8)';
  if (normalized < 0.4) return `rgba(59, 130, 246, ${0.2 + normalized * 0.5})`;
  if (normalized < 0.6) return `rgba(6, 182, 212, ${0.3 + normalized * 0.5})`;
  if (normalized < 0.8) return `rgba(245, 158, 11, ${0.4 + normalized * 0.5})`;
  return `rgba(239, 68, 68, ${0.5 + normalized * 0.5})`;
}

export default function AttentionHeatmap({ attention, attributionData }) {
  const { matrix, labels } = attention || {};

  // Transformer self-attention calculation (if present)
  const { maxVal, columnImportance } = useMemo(() => {
    if (!matrix?.length) return { maxVal: 1, columnImportance: [] };

    let max = 0;
    const colSums = new Array(matrix[0]?.length || 0).fill(0);

    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        max = Math.max(max, matrix[i][j]);
        colSums[j] += matrix[i][j];
      }
    }

    const avgCol = colSums.map((s) => s / matrix.length);
    const colMean = avgCol.reduce((a, b) => a + b, 0) / avgCol.length;

    return {
      maxVal: max,
      columnImportance: avgCol.map((v) => v > colMean * 1.3),
    };
  }, [matrix]);

  // Case 1: Genuine Transformer attention weights exist
  if (matrix?.length > 0) {
    const size = matrix.length;
    const cellSize = Math.max(16, Math.min(28, 400 / size));

    return (
      <motion.div
        className="glass-panel p-5"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
            Attention Heatmap
          </h2>
          <span className="text-xs text-slate-600">Self-attention weights</span>
        </div>

        <div className="flex items-center gap-2 mb-4 text-xs text-slate-500">
          <span>Low</span>
          <div className="flex-1 h-2 rounded-full overflow-hidden flex">
            {[0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1].map((v, i) => (
              <div
                key={i}
                className="flex-1"
                style={{ background: getHeatColor(v, 1) }}
              />
            ))}
          </div>
          <span>High</span>
        </div>

        <div className="overflow-x-auto">
          <div className="inline-block">
            <div className="flex ml-12">
              {labels?.slice(0, size).map((label, j) => (
                <div
                  key={j}
                  className={`text-center ${columnImportance[j] ? 'font-semibold' : ''}`}
                  style={{
                    width: cellSize,
                    fontSize: '8px',
                    color: columnImportance[j] ? '#f59e0b' : '#475569',
                    transform: 'rotate(-45deg)',
                    transformOrigin: 'center',
                    height: 30,
                    display: 'flex',
                    alignItems: 'flex-end',
                    justifyContent: 'center',
                  }}
                >
                  {label}
                </div>
              ))}
            </div>

            {matrix.map((row, i) => (
              <motion.div
                key={i}
                className="flex items-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 + i * 0.02 }}
              >
                <div
                  className="text-right pr-2 flex-shrink-0"
                  style={{ width: 48, fontSize: '8px', color: '#475569' }}
                >
                  {labels?.[i] || `T${i}`}
                </div>

                {row.map((val, j) => (
                  <motion.div
                    key={j}
                    className="flex-shrink-0 rounded-sm cursor-pointer transition-all duration-150"
                    style={{
                      width: cellSize - 1,
                      height: cellSize - 1,
                      margin: 0.5,
                      background: getHeatColor(val, maxVal),
                      border: columnImportance[j] && val > maxVal * 0.6
                        ? '1px solid rgba(245, 158, 11, 0.4)'
                        : '1px solid transparent',
                    }}
                    whileHover={{
                      scale: 1.8,
                      zIndex: 10,
                      boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                    }}
                    title={`Query: ${labels?.[i] || i}, Key: ${labels?.[j] || j}\nWeight: ${val.toFixed(4)}`}
                  />
                ))}
              </motion.div>
            ))}
          </div>
        </div>

        {columnImportance.some((v) => v) && (
          <div className="mt-3 pt-3 border-t border-slate-800">
            <div className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              <span className="text-slate-400">
                Critical timesteps:{' '}
                {labels?.filter((_, i) => columnImportance[i]).join(', ')}
              </span>
            </div>
          </div>
        )}
      </motion.div>
    );
  }

  // Case 2: Attention not produced by active BiLSTM model, but feature attribution is available
  if (attributionData && attributionData.length > 0) {
    const sorted = [...attributionData]
      .sort((a, b) => Math.abs(b.importance ?? b.value ?? 0) - Math.abs(a.importance ?? a.value ?? 0))
      .slice(0, 6);

    const maxImportance = Math.max(...sorted.map((s) => Math.abs(s.importance ?? s.value ?? 0.0001)), 0.0001);

    return (
      <motion.div
        className="glass-panel p-5"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
              Physiological Drivers
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/70 border border-cyan-800/40 text-cyan-300 normal-case font-medium">
                Input × Gradient
              </span>
            </h2>
            <p className="text-[10px] text-slate-500 mt-0.5">
              BiLSTM recurrent model • Key directional gradient drivers
            </p>
          </div>
        </div>

        {/* Informative model notice */}
        <div className="mb-3.5 px-3 py-2 rounded-lg bg-slate-900/60 border border-slate-800/70 text-[11px] text-slate-400 leading-snug">
          BiLSTM architecture models temporal dependencies via recurrent hidden states (<span className="text-slate-300 font-mono">z_t</span>), without Transformer query-key attention matrices.
        </div>

        {/* Ranked Driver Breakdown */}
        <div className="space-y-2.5">
          {sorted.map((item, idx) => {
            const rawVal = item.value ?? item.importance ?? 0;
            const isProtective = item.direction === 'protective' || rawVal < 0;
            const magnitude = Math.abs(rawVal);
            const percentWidth = Math.min(100, Math.max(12, Math.round((magnitude / maxImportance) * 100)));
            const displayName = item.clinical_name || item.feature?.replace(/_/g, ' ') || 'Feature';

            return (
              <div key={item.feature || idx} className="p-2 rounded-lg bg-slate-900/40 border border-slate-800/40 space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span
                      className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: isProtective ? '#60a5fa' : '#f87171' }}
                    />
                    <span className="font-medium text-slate-200 capitalize truncate">
                      {displayName}
                    </span>
                  </div>
                  <span
                    className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded"
                    style={{
                      backgroundColor: isProtective ? 'rgba(96, 165, 250, 0.1)' : 'rgba(248, 113, 113, 0.1)',
                      color: isProtective ? '#93c5fd' : '#fca5a5',
                    }}
                  >
                    {isProtective ? 'Stabilizing' : 'Risk Driver'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-1 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${percentWidth}%`,
                        backgroundColor: isProtective ? '#60a5fa' : '#f87171',
                      }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">
                    {magnitude.toExponential(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </motion.div>
    );
  }

  // Case 3: Neither attention nor feature attribution is available
  return (
    <motion.div
      className="glass-panel p-5 flex flex-col items-center justify-center h-full text-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="w-8 h-8 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center mb-2 text-slate-500">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <p className="text-xs font-medium text-slate-400">Feature attribution unavailable for this prediction.</p>
      <p className="text-[10px] text-slate-500 mt-1">BiLSTM model does not generate self-attention weights.</p>
    </motion.div>
  );
}
