import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * WhatIfPanel — pre-BioGears What-If / TRUE ML counterfactual UI.
 *
 * Structural piece of the pre-BioGears architecture (see
 * backend/whatif.py and docs/BIOGEARS_INTEGRATION_BOUNDARY.md): lets
 * the user build a real, validated scenario request against this
 * patient's real data, submits it to the real backend, and shows
 * exactly what comes back — a real baseline, a REAL ML counterfactual
 * (the same trained model re-run on the modified input through the
 * real feature-engineering pipeline), and an honest "physiological
 * simulation requires BioGears integration" boundary. This component
 * never invents a counterfactual or simulated result itself; it only
 * renders whatever the backend actually computed.
 *
 * Labeling rule (do not violate): the counterfactual section is always
 * labeled "ML Counterfactual" — NEVER "BioGears Simulation" or
 * "Physiological Simulation". Those are a different, still-unavailable
 * thing (see the separate "Physiological Simulation" section below).
 */
export default function WhatIfPanel({ intervenableFeatures, onRunScenario, stayId }) {
  const [feature, setFeature] = useState('');
  const [value, setValue] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    // Reset the scenario form/result when the selected patient changes —
    // a stale result from a different patient must never linger.
    setResult(null);
    setError(null);
    if (intervenableFeatures?.length && !feature) {
      setFeature(intervenableFeatures[0].feature);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stayId, intervenableFeatures]);

  const selected = intervenableFeatures?.find((f) => f.feature === feature);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selected) return;
    const numValue = parseFloat(value);
    if (Number.isNaN(numValue)) {
      setError('Enter a numeric value.');
      return;
    }
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const res = await onRunScenario([{ feature, value: numValue }], 'biogears');
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  if (!intervenableFeatures?.length) {
    return (
      <motion.div className="glass-panel p-5 flex items-center justify-center h-full">
        <p className="text-sm text-slate-500">Loading intervention options…</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="glass-panel p-5"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          What-If Scenario
        </h2>
        <span className="text-xs text-slate-600">Pre-BioGears architecture</span>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3 mb-4">
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-slate-500 uppercase tracking-wide">Intervention</label>
          <select
            value={feature}
            onChange={(e) => {
              setFeature(e.target.value);
              setValue('');
            }}
            className="bg-slate-800/60 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-300"
          >
            {intervenableFeatures.map((f) => (
              <option key={f.feature} value={f.feature}>
                {f.display_name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-slate-500 uppercase tracking-wide">
            Value {selected && `(${selected.unit})`}
          </label>
          <input
            type="number"
            step="any"
            min={selected?.min}
            max={selected?.max}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={selected ? `${selected.min}–${selected.max}` : ''}
            className="w-32 bg-slate-800/60 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-300"
          />
        </div>

        <motion.button
          type="submit"
          disabled={submitting || !value}
          whileTap={{ scale: 0.96 }}
          className="px-3 py-1.5 rounded-lg bg-cyan-600/80 hover:bg-cyan-600 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium text-white transition-colors"
        >
          {submitting ? 'Running…' : 'Run Scenario'}
        </motion.button>
      </form>

      {selected?.clinical_note && (
        <p className="text-[10px] text-slate-600 mb-3">{selected.clinical_note}</p>
      )}

      <AnimatePresence mode="wait">
        {error && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-300"
          >
            {error}
          </motion.div>
        )}

        {result && !error && (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            <div className="flex items-center justify-between text-xs px-3 py-2 rounded-lg bg-slate-800/40 border border-slate-700/50">
              <span className="text-slate-400">Baseline</span>
              <span className="font-mono text-slate-200">
                {(result.baseline.risk_score * 100).toFixed(0)}% · {result.baseline.status}
              </span>
            </div>

            {result.counterfactual && (
              <div className="px-3 py-3 rounded-lg bg-cyan-500/10 border border-cyan-500/25 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold text-cyan-300 uppercase tracking-wide">
                    ML Counterfactual
                  </span>
                  <span className="text-[9px] text-cyan-200/50">not a BioGears/physiological simulation</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Counterfactual risk</span>
                  <span className="font-mono text-slate-100">
                    {(result.counterfactual.risk_score * 100).toFixed(0)}% · {result.counterfactual.status}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Change</span>
                  <span
                    className={`font-mono font-semibold ${
                      result.counterfactual.risk_change > 0 ? 'text-red-300' : result.counterfactual.risk_change < 0 ? 'text-emerald-300' : 'text-slate-300'
                    }`}
                  >
                    {result.counterfactual.risk_change >= 0 ? '+' : ''}
                    {(result.counterfactual.risk_change * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-[10px] text-cyan-200/60 leading-relaxed">
                  Same trained model, re-run on this patient's real window with
                  the requested intervention applied and features
                  re-engineered — not a physiological simulation of the
                  patient's body.
                </p>
              </div>
            )}

            <div className="px-3 py-3 rounded-lg bg-amber-500/10 border border-amber-500/25">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-semibold text-amber-300 uppercase tracking-wide">
                  Physiological Simulation
                </span>
              </div>
              <p className="text-xs font-medium text-amber-300 mb-1">
                {result.simulation.message}
              </p>
              <p className="text-[10px] text-amber-200/70 leading-relaxed">
                {result.simulation.description}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
