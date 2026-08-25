import { motion } from 'framer-motion';
import { useMemo } from 'react';

/**
 * AttentionHeatmap — Matrix visualization of self-attention weights
 * 
 * Shows which timesteps the Transformer model attends to when making
 * predictions. Brighter cells = higher attention weight.
 * 
 * Critical timesteps (high attention from many queries) are highlighted,
 * revealing which moments in the patient's history the model considers
 * most important for risk prediction.
 */

function getHeatColor(value, maxVal) {
  const normalized = value / (maxVal || 1);
  
  if (normalized < 0.2) return `rgba(15, 23, 42, 0.8)`;
  if (normalized < 0.4) return `rgba(59, 130, 246, ${0.2 + normalized * 0.5})`;
  if (normalized < 0.6) return `rgba(6, 182, 212, ${0.3 + normalized * 0.5})`;
  if (normalized < 0.8) return `rgba(245, 158, 11, ${0.4 + normalized * 0.5})`;
  return `rgba(239, 68, 68, ${0.5 + normalized * 0.5})`;
}

export default function AttentionHeatmap({ attention }) {
  const { matrix, labels } = attention || {};

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
    
    const avgCol = colSums.map(s => s / matrix.length);
    const colMean = avgCol.reduce((a, b) => a + b, 0) / avgCol.length;
    
    return {
      maxVal: max,
      columnImportance: avgCol.map(v => v > colMean * 1.3),
    };
  }, [matrix]);

  if (!matrix?.length) {
    return (
      <motion.div className="glass-panel p-5 flex flex-col items-center justify-center h-full text-center gap-1">
        <p className="text-sm text-slate-500">Not applicable for this model</p>
        <p className="text-xs text-slate-600 max-w-[220px]">
          The active model is a BiLSTM, which has no self-attention mechanism —
          this panel needs the Transformer model to show real attention weights.
        </p>
      </motion.div>
    );
  }

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

      {/* Color scale legend */}
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

      {/* Heatmap */}
      <div className="overflow-x-auto">
        <div className="inline-block">
          {/* Column labels */}
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

          {/* Matrix rows */}
          {matrix.map((row, i) => (
            <motion.div
              key={i}
              className="flex items-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 + i * 0.02 }}
            >
              {/* Row label */}
              <div
                className="text-right pr-2 flex-shrink-0"
                style={{ width: 48, fontSize: '8px', color: '#475569' }}
              >
                {labels?.[i] || `T${i}`}
              </div>

              {/* Cells */}
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

      {/* Critical timesteps */}
      {columnImportance.some(v => v) && (
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
