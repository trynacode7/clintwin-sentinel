import { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  SUPPORTED_SALINE_VOLUMES,
  DEFAULT_BASELINE_OBSERVATION_S,
  DEFAULT_POST_OBSERVATION_S,
  MAX_SIMULATION_DURATION_S,
  calculateDeliveryDuration,
  validateSequentialSimulation,
  executeSequentialWhatIf,
  COMPARISON_PALETTE,
  alignComparisonTrajectories,
  computeScenarioSummaryTable,
  computeReferenceDifferences,
  DEFAULT_STEP_J_STRATEGIES,
  calculateStrategyTotalSaline,
  formatStrategyActionSummary,
  VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
  VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
  VERIFIED_NOREPINEPHRINE_DURATION_S,
} from '../services/biogearsApi';
import ScenarioComparisonChart from './ScenarioComparisonChart';

export default function ScenarioComparisonWorkspace() {
  const [strategies, setStrategies] = useState(DEFAULT_STEP_J_STRATEGIES);
  const [running, setRunning] = useState(false);
  const [activeRunningIndex, setActiveRunningIndex] = useState(null);
  const [currentElapsedSec, setCurrentElapsedSec] = useState(0);

  // Per-strategy execution results: map of { [strategyId]: { result, status, error, wallClockSeconds } }
  const [resultsMap, setResultsMap] = useState({});

  // Reference scenario selection for difference analysis
  const [referenceId, setReferenceId] = useState('strategy_1');

  // Overall error banner
  const [generalError, setGeneralError] = useState(null);

  const abortControllerRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, []);

  // Strategy validation for all configured strategies
  const strategyValidations = useMemo(() => {
    return strategies.map((strat) => {
      const res = validateSequentialSimulation({
        actions: strat.actions,
        baselineObservationS: strat.baseline_s,
        postObservationS: strat.post_s,
      });
      return {
        id: strat.id,
        valid: res.valid,
        error: res.error,
        actionErrors: res.actionErrors,
        totalDuration: res.totalDuration,
      };
    });
  }, [strategies]);

  const hasAnyValidationErrors = strategyValidations.some((v) => !v.valid);

  // Completed scenarios list for charts and tables
  const completedScenariosForCharts = useMemo(() => {
    return strategies
      .filter((s) => resultsMap[s.id]?.status === 'completed' && resultsMap[s.id]?.result)
      .map((s) => ({
        id: s.id,
        name: s.name,
        color: s.color,
        actions: s.actions,
        result: resultsMap[s.id].result,
        wallClockSeconds: resultsMap[s.id].wallClockSeconds,
      }));
  }, [strategies, resultsMap]);

  // Aligned trajectory calculations
  const alignedTrajectories = useMemo(() => {
    return alignComparisonTrajectories(completedScenariosForCharts);
  }, [completedScenariosForCharts]);

  // Scenario Summary Table data
  const summaryTableRows = useMemo(() => {
    const list = strategies.map((s) => {
      const exec = resultsMap[s.id];
      const val = strategyValidations.find((v) => v.id === s.id);
      return {
        id: s.id,
        name: s.name,
        color: s.color,
        actions: s.actions,
        totalSimDuration: val?.totalDuration ?? null,
        result: exec?.result ?? null,
        status: exec?.status ?? (running ? 'pending' : 'idle'),
        error: exec?.error ?? null,
        wallClockSeconds: exec?.wallClockSeconds ?? null,
      };
    });
    return computeScenarioSummaryTable(list);
  }, [strategies, resultsMap, running, strategyValidations]);

  // Reference differences analysis
  const referenceAnalysis = useMemo(() => {
    return computeReferenceDifferences(summaryTableRows, referenceId);
  }, [summaryTableRows, referenceId]);

  // Handlers for Strategy Configuration
  const handleAddStrategy = () => {
    if (strategies.length >= 3) return;
    const nextIdx = strategies.length;
    const newId = `strategy_${Date.now()}`;
    const nextColor = COMPARISON_PALETTE[nextIdx]?.color || '#f59e0b';
    const letter = String.fromCharCode(65 + nextIdx); // 'A', 'B', 'C'

    setStrategies((prev) => [
      ...prev,
      {
        id: newId,
        name: `Strategy ${letter}: Custom Strategy`,
        color: nextColor,
        baseline_s: 60,
        post_s: 60,
        actions: [{ id: `act_${Date.now()}`, type: 'saline', volume_ml: 100, start_time_s: 60 }],
      },
    ]);
  };

  const handleRemoveStrategy = (idToRemove) => {
    if (strategies.length <= 1) return;
    setStrategies((prev) => prev.filter((s) => s.id !== idToRemove));
    setResultsMap((prev) => {
      const copy = { ...prev };
      delete copy[idToRemove];
      return copy;
    });
    if (referenceId === idToRemove) {
      const remaining = strategies.filter((s) => s.id !== idToRemove);
      if (remaining.length > 0) setReferenceId(remaining[0].id);
    }
  };

  const handleResetPresets = () => {
    if (running) return;
    setStrategies(DEFAULT_STEP_J_STRATEGIES);
    setResultsMap({});
    setReferenceId('strategy_1');
    setGeneralError(null);
  };

  const handleUpdateStrategyName = (id, newName) => {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, name: newName } : s))
    );
  };

  const handleAddIntervention = (strategyId, type = 'saline') => {
    setStrategies((prev) =>
      prev.map((strat) => {
        if (strat.id !== strategyId || strat.actions.length >= 3) return strat;
        const lastAction = strat.actions[strat.actions.length - 1];
        const lastDuration = (lastAction?.type === 'norepinephrine' || lastAction?.substance === 'Norepinephrine')
          ? VERIFIED_NOREPINEPHRINE_DURATION_S
          : calculateDeliveryDuration(lastAction?.volume_ml || 100);
        const lastEnd = Number(lastAction?.start_time_s || 60) + lastDuration;
        const newStart = Math.max(lastEnd, lastEnd + 30);

        const newAction = type === 'norepinephrine'
          ? {
              id: `act_${Date.now()}`,
              type: 'norepinephrine',
              concentration_ug_ml: VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
              rate_ml_per_min: VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
              duration_s: VERIFIED_NOREPINEPHRINE_DURATION_S,
              start_time_s: newStart,
            }
          : {
              id: `act_${Date.now()}`,
              type: 'saline',
              volume_ml: 100,
              start_time_s: newStart,
            };

        return {
          ...strat,
          actions: [...strat.actions, newAction],
        };
      })
    );
  };

  const handleRemoveIntervention = (strategyId, actionIndex) => {
    setStrategies((prev) =>
      prev.map((strat) => {
        if (strat.id !== strategyId || strat.actions.length <= 1) return strat;
        const newActions = strat.actions.filter((_, idx) => idx !== actionIndex);
        return { ...strat, actions: newActions };
      })
    );
  };

  const handleMoveIntervention = (strategyId, actionIndex, direction) => {
    setStrategies((prev) =>
      prev.map((strat) => {
        if (strat.id !== strategyId) return strat;
        const targetIndex = actionIndex + direction;
        if (targetIndex < 0 || targetIndex >= strat.actions.length) return strat;
        const newActions = [...strat.actions];
        const temp = newActions[actionIndex];
        newActions[actionIndex] = newActions[targetIndex];
        newActions[targetIndex] = temp;
        return { ...strat, actions: newActions };
      })
    );
  };

  const handleUpdateIntervention = (strategyId, actionIndex, field, value) => {
    setStrategies((prev) =>
      prev.map((strat) => {
        if (strat.id !== strategyId) return strat;
        const newActions = [...strat.actions];
        const current = { ...newActions[actionIndex] };

        if (field === 'type') {
          if (value === 'norepinephrine') {
            current.type = 'norepinephrine';
            current.concentration_ug_ml = VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML;
            current.rate_ml_per_min = VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN;
            current.duration_s = VERIFIED_NOREPINEPHRINE_DURATION_S;
            delete current.volume_ml;
          } else {
            current.type = 'saline';
            current.volume_ml = 100;
            delete current.concentration_ug_ml;
            delete current.rate_ml_per_min;
            delete current.duration_s;
          }
        } else if (field === 'volume_ml' || field === 'start_time_s') {
          current[field] = Number(value);
        } else {
          current[field] = value;
        }

        newActions[actionIndex] = current;
        return { ...strat, actions: newActions };
      })
    );
  };

  // Execution: Sequential simulation loop
  const handleRunComparison = async () => {
    if (running || hasAnyValidationErrors || strategies.length === 0) return;

    setRunning(true);
    setGeneralError(null);
    abortControllerRef.current = new AbortController();

    // Start wall-clock timer
    setCurrentElapsedSec(0);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setCurrentElapsedSec((prev) => prev + 1);
    }, 1000);

    for (let i = 0; i < strategies.length; i++) {
      const strat = strategies[i];
      setActiveRunningIndex(i);

      // Mark strategy as simulating in resultsMap
      setResultsMap((prev) => ({
        ...prev,
        [strat.id]: {
          status: 'simulating',
          result: prev[strat.id]?.result || null,
          error: null,
          wallClockSeconds: null,
        },
      }));

      const startTimeMs = performance.now();

      try {
        const result = await executeSequentialWhatIf({
          actions: strat.actions,
          baselineObservationS: strat.baseline_s,
          postObservationS: strat.post_s,
          signal: abortControllerRef.current.signal,
        });

        const elapsedWallClock = Math.round((performance.now() - startTimeMs) / 100) / 10;

        setResultsMap((prev) => ({
          ...prev,
          [strat.id]: {
            status: 'completed',
            result,
            error: null,
            wallClockSeconds: elapsedWallClock,
          },
        }));
      } catch (err) {
        if (err.message?.includes('cancelled')) {
          setGeneralError('Comparison simulation was aborted by user.');
          break;
        }

        // Record error for this strategy but allow other completed strategies to remain visible
        setResultsMap((prev) => ({
          ...prev,
          [strat.id]: {
            status: 'error',
            result: null,
            error: err.message || 'Simulation execution failed.',
            wallClockSeconds: null,
          },
        }));
      }
    }

    if (timerRef.current) clearInterval(timerRef.current);
    setRunning(false);
    setActiveRunningIndex(null);
  };

  const handleCancelComparison = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (timerRef.current) clearInterval(timerRef.current);
    setRunning(false);
    setActiveRunningIndex(null);
  };

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-5 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-xl backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
              <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                BioGears Scenario Comparison Workspace
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/70 text-cyan-300 border border-cyan-800/40 uppercase font-semibold">
                  Step J
                </span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Configure, execute, and compare up to three sequential intervention strategies (Saline & Norepinephrine) side-by-side using real BioGears physiological trajectory data.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleResetPresets}
            disabled={running}
            className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 text-xs font-medium transition-all disabled:opacity-50 cursor-pointer"
            title="Load default Strategies A (Fluid-only), B (Sequential), and C (Repeated boluses)"
          >
            Reset Presets
          </button>
          <button
            onClick={handleAddStrategy}
            disabled={strategies.length >= 3 || running}
            className="px-3 py-1.5 rounded-lg border border-cyan-500/40 bg-cyan-950/40 hover:bg-cyan-900/50 text-cyan-300 text-xs font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1.5"
          >
            <span>+</span> Add Strategy ({strategies.length}/3)
          </button>
        </div>
      </div>

      {/* General Error Banner */}
      {generalError && (
        <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/50 text-rose-300 text-xs flex items-center justify-between">
          <span>{generalError}</span>
          <button
            onClick={() => setGeneralError(null)}
            className="text-rose-400 hover:text-rose-200 font-bold ml-2 cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* Strategy Configuration Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {strategies.map((strat, idx) => {
          const validation = strategyValidations.find((v) => v.id === strat.id);
          const exec = resultsMap[strat.id];
          const isCurrentRunning = running && activeRunningIndex === idx;
          const totalSaline = calculateStrategyTotalSaline(strat.actions);
          const scheduleSummary = formatStrategyActionSummary(strat.actions);

          return (
            <div
              key={strat.id}
              className="p-4 rounded-xl bg-slate-900/60 border transition-all flex flex-col justify-between"
              style={{
                borderColor: strat.color ? `${strat.color}55` : 'rgba(148, 163, 184, 0.2)',
                boxShadow: isCurrentRunning ? `0 0 15px ${strat.color}33` : 'none',
              }}
            >
              <div>
                {/* Card Header: Color chip, editable title, remove button */}
                <div className="flex items-center justify-between gap-2 mb-2 pb-2 border-b border-slate-800">
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <span
                      className="w-3.5 h-3.5 rounded-md flex-shrink-0"
                      style={{ backgroundColor: strat.color }}
                    />
                    <input
                      type="text"
                      value={strat.name}
                      onChange={(e) => handleUpdateStrategyName(strat.id, e.target.value)}
                      disabled={running}
                      className="bg-transparent font-semibold text-xs text-slate-100 hover:bg-slate-800/40 focus:bg-slate-950 px-1.5 py-0.5 rounded border border-transparent focus:border-slate-700 outline-none w-full truncate"
                    />
                  </div>

                  {strategies.length > 1 && (
                    <button
                      onClick={() => handleRemoveStrategy(strat.id)}
                      disabled={running}
                      className="text-slate-500 hover:text-rose-400 text-xs px-1.5 py-0.5 rounded transition-all disabled:opacity-40 cursor-pointer"
                      title="Remove strategy"
                    >
                      ✕
                    </button>
                  )}
                </div>

                {/* Experimental Control Badge: Total Saline & Schedule Summary */}
                <div className="mb-3 space-y-1">
                  <div className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg bg-slate-950/70 border border-slate-800 text-[11px]">
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-400">Total Saline:</span>
                      <span className="font-mono font-bold text-cyan-300">
                        {totalSaline} mL
                      </span>
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">
                      {strat.actions.some((a) => a.type === 'norepinephrine') ? 'Fluid + Vasopressor' : 'Fluid-Only'}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono px-1 truncate" title={scheduleSummary}>
                    Schedule: <span className="text-slate-300">{scheduleSummary}</span>
                  </div>
                </div>

                {/* Sequential Interventions Configuration List */}
                <div className="space-y-2.5 mb-3">
                  <div className="flex items-center justify-between text-[11px] text-slate-400 font-medium">
                    <span>Sequential Actions ({strat.actions.length}/3)</span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => handleAddIntervention(strat.id, 'saline')}
                        disabled={strat.actions.length >= 3 || running}
                        className="text-cyan-400 hover:text-cyan-300 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-[10px] px-1.5 py-0.5 rounded bg-cyan-950/50 border border-cyan-800/50 cursor-pointer"
                        title="Add Saline Bolus"
                      >
                        + Saline
                      </button>
                      <button
                        onClick={() => handleAddIntervention(strat.id, 'norepinephrine')}
                        disabled={strat.actions.length >= 3 || running}
                        className="text-purple-400 hover:text-purple-300 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-[10px] px-1.5 py-0.5 rounded bg-purple-950/50 border border-purple-800/50 cursor-pointer"
                        title="Add Norepinephrine Infusion"
                      >
                        + Norepi
                      </button>
                    </div>
                  </div>

                  {strat.actions.map((act, aIdx) => {
                    const isNorepi = act.type === 'norepinephrine';
                    const dur = isNorepi
                      ? VERIFIED_NOREPINEPHRINE_DURATION_S
                      : calculateDeliveryDuration(act.volume_ml || 100);
                    const aError = validation?.actionErrors?.[aIdx];

                    return (
                      <div
                        key={act.id || aIdx}
                        className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/90 space-y-2"
                      >
                        <div className="flex items-center justify-between text-[11px]">
                          <div className="flex items-center gap-1.5">
                            <span className="font-semibold text-slate-300">
                              #{aIdx + 1}
                            </span>
                            <span
                              className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                                isNorepi
                                  ? 'bg-purple-950/80 text-purple-300 border border-purple-800/50'
                                  : 'bg-cyan-950/80 text-cyan-300 border border-cyan-800/50'
                              }`}
                            >
                              {isNorepi ? 'Norepinephrine' : 'Saline Bolus'}
                            </span>
                          </div>

                          <div className="flex items-center gap-1.5">
                            <span className="font-mono text-[10px] text-slate-500">
                              {dur}s ({dur / 60}m)
                            </span>

                            {/* Reordering controls */}
                            <button
                              onClick={() => handleMoveIntervention(strat.id, aIdx, -1)}
                              disabled={aIdx === 0 || running}
                              className="text-slate-500 hover:text-slate-200 disabled:opacity-20 text-[10px] px-1 py-0.5 rounded hover:bg-slate-800 cursor-pointer"
                              title="Move action earlier in sequence"
                            >
                              ↑
                            </button>
                            <button
                              onClick={() => handleMoveIntervention(strat.id, aIdx, 1)}
                              disabled={aIdx === strat.actions.length - 1 || running}
                              className="text-slate-500 hover:text-slate-200 disabled:opacity-20 text-[10px] px-1 py-0.5 rounded hover:bg-slate-800 cursor-pointer"
                              title="Move action later in sequence"
                            >
                              ↓
                            </button>

                            {strat.actions.length > 1 && (
                              <button
                                onClick={() => handleRemoveIntervention(strat.id, aIdx)}
                                disabled={running}
                                className="text-slate-500 hover:text-rose-400 text-[10px] px-1 py-0.5 cursor-pointer"
                                title="Remove intervention"
                              >
                                ✕
                              </button>
                            )}
                          </div>
                        </div>

                        {/* Action Type & Start Time */}
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <label className="block text-[10px] text-slate-400 mb-1">
                              Intervention Type
                            </label>
                            <select
                              value={act.type || 'saline'}
                              onChange={(e) =>
                                handleUpdateIntervention(strat.id, aIdx, 'type', e.target.value)
                              }
                              disabled={running}
                              className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 text-xs focus:border-cyan-500 outline-none"
                            >
                              <option value="saline">Normal Saline</option>
                              <option value="norepinephrine">Norepinephrine</option>
                            </select>
                          </div>

                          <div>
                            <label className="block text-[10px] text-slate-400 mb-1">
                              Start Time (s)
                            </label>
                            <input
                              type="number"
                              min="0"
                              max="1800"
                              value={act.start_time_s}
                              onChange={(e) =>
                                handleUpdateIntervention(strat.id, aIdx, 'start_time_s', e.target.value)
                              }
                              disabled={running}
                              className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 text-xs font-mono focus:border-cyan-500 outline-none"
                            />
                          </div>
                        </div>

                        {/* Dose / Volume Parameters */}
                        {isNorepi ? (
                          <div className="p-1.5 rounded bg-purple-950/30 border border-purple-900/40 text-[10px] text-purple-300 font-mono flex items-center justify-between">
                            <span>Locked Verified Dose:</span>
                            <span>14 µg/mL @ 1 mL/min (120s)</span>
                          </div>
                        ) : (
                          <div>
                            <label className="block text-[10px] text-slate-400 mb-1">
                              Bolus Volume (at 50 mL/min)
                            </label>
                            <select
                              value={act.volume_ml || 100}
                              onChange={(e) =>
                                handleUpdateIntervention(strat.id, aIdx, 'volume_ml', e.target.value)
                              }
                              disabled={running}
                              className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 text-xs focus:border-cyan-500 outline-none"
                            >
                              {SUPPORTED_SALINE_VOLUMES.map((v) => (
                                <option key={v} value={v}>
                                  {v} mL
                                </option>
                              ))}
                            </select>
                          </div>
                        )}

                        {aError && (
                          <p className="text-[10px] text-rose-400 mt-1 leading-tight">
                            {aError}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Validation message if invalid */}
                {!validation?.valid && validation?.error && (
                  <div className="p-2 rounded bg-rose-950/40 border border-rose-900/60 text-rose-400 text-[10px] mb-2 leading-tight">
                    {validation.error}
                  </div>
                )}
              </div>

              {/* Card Footer: Status and summary */}
              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
                <div className="text-slate-400 font-mono text-[10px]">
                  Total Sim: <span className="text-slate-200">{validation?.totalDuration || 0}s</span>
                </div>

                <div>
                  {isCurrentRunning ? (
                    <span className="px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 animate-pulse text-[10px] font-semibold flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                      Simulating...
                    </span>
                  ) : exec?.status === 'completed' ? (
                    <span className="px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800 text-[10px] font-semibold flex items-center gap-1">
                      ✓ {exec.result?.metadata?.is_cached ? 'Cached' : 'Fresh BioGears'}
                    </span>
                  ) : exec?.status === 'error' ? (
                    <span className="px-2 py-0.5 rounded-full bg-rose-950 text-rose-300 border border-rose-800 text-[10px] font-semibold">
                      Failed
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 text-[10px]">
                      Ready
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Run Action Bar */}
      <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={handleRunComparison}
            disabled={running || hasAnyValidationErrors}
            className={`px-5 py-2.5 rounded-xl text-xs font-semibold shadow-lg transition-all flex items-center gap-2 cursor-pointer ${
              running || hasAnyValidationErrors
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed shadow-none'
                : 'bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white shadow-emerald-950/50'
            }`}
          >
            {running ? (
              <>
                <svg className="w-4 h-4 animate-spin text-cyan-200" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span>
                  Running Comparison (Strategy {activeRunningIndex !== null ? activeRunningIndex + 1 : 1} of {strategies.length})...
                </span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4 text-emerald-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Run Scenario Comparison ({strategies.length} Strategies)</span>
              </>
            )}
          </button>

          {running && (
            <button
              onClick={handleCancelComparison}
              className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-rose-950 hover:text-rose-300 text-slate-300 text-xs font-medium border border-slate-700 transition-all cursor-pointer"
            >
              Cancel Run
            </button>
          )}

          {running && (
            <span className="font-mono text-xs text-slate-400">
              Elapsed: <span className="text-white font-semibold">{currentElapsedSec}s</span>
            </span>
          )}
        </div>

        <div className="text-[11px] text-slate-400 max-w-md">
          <p>
            <span className="text-cyan-400 font-semibold">Note:</span> Scenarios execute sequentially to prevent engine collisions. Cached runs return in milliseconds; fresh live runs take ~2.5–3.5 minutes each.
          </p>
        </div>
      </div>

      {/* Trajectory Comparison Chart */}
      {completedScenariosForCharts.length > 0 && (
        <ScenarioComparisonChart
          alignedData={alignedTrajectories.alignedPoints}
          scenarios={completedScenariosForCharts}
          commonInterval={alignedTrajectories.commonInterval}
          maxDuration={alignedTrajectories.maxDuration}
          interventionMarkers={alignedTrajectories.interventionMarkers}
        />
      )}

      {/* Scenario Summary Table */}
      <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
              Scenario Comparison Summary Table
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Exact initial, final, and absolute changes (Δ = final - initial) across verified BioGears runs.
            </p>
          </div>
          <span className="text-[10px] font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
            {summaryTableRows.filter((r) => r.status === 'completed').length} / {summaryTableRows.length} Strategies Completed
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-semibold bg-slate-950/60 text-[11px]">
                <th className="py-2.5 px-3">Strategy</th>
                <th className="py-2.5 px-3">Interventions</th>
                <th className="py-2.5 px-3">Total Saline</th>
                <th className="py-2.5 px-3">Sim Duration</th>
                <th className="py-2.5 px-3">Status / Source</th>
                <th className="py-2.5 px-3 text-right">Heart Rate (bpm)</th>
                <th className="py-2.5 px-3 text-right">MAP (mmHg)</th>
                <th className="py-2.5 px-3 text-right">Blood Vol (L)</th>
                <th className="py-2.5 px-3 text-right">Cardiac Out (L/min)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
              {summaryTableRows.map((row) => {
                const isDone = row.status === 'completed';
                const hr = row.vitals?.hr;
                const map = row.vitals?.map;
                const bv = row.vitals?.bv;
                const co = row.vitals?.co;

                return (
                  <tr key={row.id} className="hover:bg-slate-800/30 transition-colors">
                    {/* Strategy Name */}
                    <td className="py-2.5 px-3 font-sans font-semibold text-slate-200">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                          style={{ backgroundColor: row.color }}
                        />
                        <span className="truncate max-w-[170px]" title={row.name}>
                          {row.name}
                        </span>
                      </div>
                    </td>

                    {/* Number of Interventions */}
                    <td className="py-2.5 px-3 text-slate-300">
                      <span>{row.numInterventions} action{row.numInterventions > 1 ? 's' : ''}</span>
                      <span className="text-[10px] text-slate-500 block font-sans">
                        {row.hasNorepinephrine ? 'Saline + Norepi' : 'Fluid-only'}
                      </span>
                    </td>

                    {/* Total Volume */}
                    <td className="py-2.5 px-3 text-cyan-300 font-bold">
                      {row.totalVolumeMl} mL
                    </td>

                    {/* Simulated Duration vs Wall Clock */}
                    <td className="py-2.5 px-3 text-slate-400">
                      {row.simulatedDurationS ? `${row.simulatedDurationS}s` : '—'}
                      {typeof row.wallClockSeconds === 'number' && (
                        <span className="text-[10px] text-slate-500 block font-sans">
                          Wall: {row.wallClockSeconds}s
                        </span>
                      )}
                    </td>

                    {/* Status / Cache Indicator */}
                    <td className="py-2.5 px-3">
                      {isDone ? (
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-sans font-semibold border ${
                            row.isCached
                              ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800/60'
                              : 'bg-cyan-950/80 text-cyan-300 border-cyan-800/60'
                          }`}
                        >
                          {row.isCached ? '⚡ Cache Hit' : '🔬 Live BioGears'}
                        </span>
                      ) : row.status === 'error' ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-sans font-semibold bg-rose-950 text-rose-300 border border-rose-800">
                          Error
                        </span>
                      ) : row.status === 'simulating' ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-sans font-semibold bg-amber-950 text-amber-300 border border-amber-800 animate-pulse">
                          Simulating...
                        </span>
                      ) : (
                        <span className="text-slate-500 font-sans text-[10px]">
                          Pending Run
                        </span>
                      )}
                    </td>

                    {/* Heart Rate */}
                    <td className="py-2.5 px-3 text-right">
                      {isDone && hr?.initial !== null && hr?.final !== null ? (
                        <div>
                          <span className="text-slate-400">{hr.initial.toFixed(1)} → </span>
                          <span className="text-slate-100 font-bold">{hr.final.toFixed(1)}</span>
                          <span
                            className={`ml-1 text-[10px] ${
                              hr.delta > 0 ? 'text-rose-400' : hr.delta < 0 ? 'text-emerald-400' : 'text-slate-400'
                            }`}
                          >
                            ({hr.delta > 0 ? '+' : ''}{hr.delta.toFixed(1)})
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-500">Unavailable</span>
                      )}
                    </td>

                    {/* MAP */}
                    <td className="py-2.5 px-3 text-right">
                      {isDone && map?.initial !== null && map?.final !== null ? (
                        <div>
                          <span className="text-slate-400">{map.initial.toFixed(1)} → </span>
                          <span className="text-slate-100 font-bold">{map.final.toFixed(1)}</span>
                          <span
                            className={`ml-1 text-[10px] ${
                              map.delta > 0 ? 'text-cyan-400' : map.delta < 0 ? 'text-amber-400' : 'text-slate-400'
                            }`}
                          >
                            ({map.delta > 0 ? '+' : ''}{map.delta.toFixed(1)})
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-500">Unavailable</span>
                      )}
                    </td>

                    {/* Blood Volume */}
                    <td className="py-2.5 px-3 text-right">
                      {isDone && bv?.initial !== null && bv?.final !== null ? (
                        <div>
                          <span className="text-slate-400">{bv.initial.toFixed(2)} → </span>
                          <span className="text-slate-100 font-bold">{bv.final.toFixed(2)}</span>
                          <span className="ml-1 text-[10px] text-emerald-400">
                            ({bv.delta > 0 ? '+' : ''}{bv.delta.toFixed(2)})
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-500">Unavailable</span>
                      )}
                    </td>

                    {/* Cardiac Output */}
                    <td className="py-2.5 px-3 text-right">
                      {isDone && co?.initial !== null && co?.final !== null ? (
                        <div>
                          <span className="text-slate-400">{co.initial.toFixed(2)} → </span>
                          <span className="text-slate-100 font-bold">{co.final.toFixed(2)}</span>
                          <span
                            className={`ml-1 text-[10px] ${
                              co.delta > 0 ? 'text-purple-400' : co.delta < 0 ? 'text-amber-400' : 'text-slate-400'
                            }`}
                          >
                            ({co.delta > 0 ? '+' : ''}{co.delta.toFixed(2)})
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-500">Unavailable</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Reference Scenario Difference Analysis */}
      {completedScenariosForCharts.length >= 2 && referenceAnalysis && (
        <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                Reference Scenario Relative Differences
                <span className="text-[10px] font-mono text-purple-300 bg-purple-950/70 border border-purple-800/50 px-2 py-0.5 rounded">
                  Δ vs Reference
                </span>
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Descriptive simulated difference: (Strategy Metric – Reference Metric).
              </p>
            </div>

            {/* Reference Scenario Selector */}
            <div className="flex items-center gap-2 text-xs">
              <label className="text-slate-400 font-medium">Select Reference:</label>
              <select
                value={referenceId}
                onChange={(e) => setReferenceId(e.target.value)}
                className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200 text-xs focus:border-purple-500 outline-none"
              >
                {completedScenariosForCharts.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Comparative Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {referenceAnalysis.comparisonRows
              .filter((row) => !row.isReference && row.status === 'completed')
              .map((row) => {
                const diffs = row.differences;

                return (
                  <div
                    key={row.id}
                    className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2.5 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-200">{row.name}</span>
                      <span className="text-[10px] font-mono text-slate-400">
                        vs. {referenceAnalysis.referenceScenario?.name}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                      {/* Heart Rate Diff */}
                      <div className="p-2 rounded bg-slate-900/60 border border-slate-800/80">
                        <span className="text-[10px] text-slate-400 block font-sans">
                          HR Final Diff
                        </span>
                        <span className="text-slate-100 font-bold">
                          {diffs?.hr?.finalDiff !== null
                            ? `${diffs.hr.finalDiff > 0 ? '+' : ''}${diffs.hr.finalDiff.toFixed(1)} bpm`
                            : '—'}
                        </span>
                        <span className="text-[9px] text-slate-500 block font-sans">
                          ΔΔ: {diffs?.hr?.deltaDiff !== null ? `${diffs.hr.deltaDiff > 0 ? '+' : ''}${diffs.hr.deltaDiff.toFixed(1)}` : '—'}
                        </span>
                      </div>

                      {/* MAP Diff */}
                      <div className="p-2 rounded bg-slate-900/60 border border-slate-800/80">
                        <span className="text-[10px] text-slate-400 block font-sans">
                          MAP Final Diff
                        </span>
                        <span className="text-slate-100 font-bold">
                          {diffs?.map?.finalDiff !== null
                            ? `${diffs.map.finalDiff > 0 ? '+' : ''}${diffs.map.finalDiff.toFixed(1)} mmHg`
                            : '—'}
                        </span>
                        <span className="text-[9px] text-slate-500 block font-sans">
                          ΔΔ: {diffs?.map?.deltaDiff !== null ? `${diffs.map.deltaDiff > 0 ? '+' : ''}${diffs.map.deltaDiff.toFixed(1)}` : '—'}
                        </span>
                      </div>

                      {/* Blood Volume Diff */}
                      <div className="p-2 rounded bg-slate-900/60 border border-slate-800/80">
                        <span className="text-[10px] text-slate-400 block font-sans">
                          Blood Volume Final Diff
                        </span>
                        <span className="text-slate-100 font-bold">
                          {diffs?.bv?.finalDiff !== null
                            ? `${diffs.bv.finalDiff > 0 ? '+' : ''}${diffs.bv.finalDiff.toFixed(2)} L`
                            : '—'}
                        </span>
                        <span className="text-[9px] text-slate-500 block font-sans">
                          ΔΔ: {diffs?.bv?.deltaDiff !== null ? `${diffs.bv.deltaDiff > 0 ? '+' : ''}${diffs.bv.deltaDiff.toFixed(2)}` : '—'}
                        </span>
                      </div>

                      {/* Cardiac Output Diff */}
                      <div className="p-2 rounded bg-slate-900/60 border border-slate-800/80">
                        <span className="text-[10px] text-slate-400 block font-sans">
                          Cardiac Output Final Diff
                        </span>
                        <span className="text-slate-100 font-bold">
                          {diffs?.co?.finalDiff !== null
                            ? `${diffs.co.finalDiff > 0 ? '+' : ''}${diffs.co.finalDiff.toFixed(2)} L/min`
                            : '—'}
                        </span>
                        <span className="text-[9px] text-slate-500 block font-sans">
                          ΔΔ: {diffs?.co?.deltaDiff !== null ? `${diffs.co.deltaDiff > 0 ? '+' : ''}${diffs.co.deltaDiff.toFixed(2)}` : '—'}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Clinical & Scientific Disclaimer Notice */}
      <div className="rounded-xl bg-slate-950/90 border border-amber-900/30 p-4 text-xs">
        <div className="flex items-start gap-3">
          <div className="w-5 h-5 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
            <svg className="w-3 h-3 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="space-y-1.5 text-slate-400">
            <p className="font-semibold text-slate-300">
              Physiological Simulation & Research Disclaimer
            </p>
            <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-400">
              <li>
                <span className="text-slate-300">Model Limitation:</span> All scenarios are solved using the BioGears engine configured with <span className="font-mono text-slate-200">StandardMale.xml</span>. This is a generic physiological reference model and is not calibrated to the individual ICU patient.
              </li>
              <li>
                <span className="text-slate-300">Comparative Differences:</span> Reported deltas and relative differences are strictly descriptive outputs from the mechanistic ODE vascular model. They do not constitute clinical recommendations or evidence of therapeutic superiority.
              </li>
              <li>
                <span className="text-slate-300">Sepsis Risk Separation:</span> This scenario comparison workspace simulates physiological fluid dynamics and does not alter or recompute the BiLSTM sepsis risk prediction model.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
