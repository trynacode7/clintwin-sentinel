import { useState, useRef, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  SUPPORTED_SALINE_VOLUMES,
  DEFAULT_BASELINE_OBSERVATION_S,
  DEFAULT_POST_OBSERVATION_S,
  SALINE_INFUSION_RATE_ML_PER_MIN,
  MAX_SIMULATION_DURATION_S,
  VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
  VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
  VERIFIED_NOREPINEPHRINE_DURATION_S,
  SUPPORTED_INTERVENTION_TYPES,
  calculateDeliveryDuration,
  validateSequentialSimulation,
  deduplicateMilestones,
  computePhysiologicalDeltas,
  computeConsecutiveMilestoneDeltas,
  executeSequentialWhatIf,
} from '../services/biogearsApi';
import WhatIfTrajectoryChart from './WhatIfTrajectoryChart';
import ScenarioComparisonWorkspace from './ScenarioComparisonWorkspace';

export default function WhatIfSimulationPanel() {
  // Top-level workspace mode: 'comparison' (Step H) or 'single' (Step G & I)
  const [workspaceMode, setWorkspaceMode] = useState('comparison');

  // Number of interventions: 1, 2, or 3 (default 1)
  const [numInterventions, setNumInterventions] = useState(1);

  // Intervention actions list: array of { id, type, volume_ml, start_time_s, concentration_ug_ml, ... }
  // Default is 1 saline bolus of 100 mL starting at 60s (preserving Step F single-bolus workflow)
  const [actions, setActions] = useState([
    { id: 'bolus_1', type: 'saline', volume_ml: 100, start_time_s: DEFAULT_BASELINE_OBSERVATION_S },
  ]);

  // Observation periods
  const [baselinePeriod, setBaselinePeriod] = useState(DEFAULT_BASELINE_OBSERVATION_S);
  const [postPeriod, setPostPeriod] = useState(DEFAULT_POST_OBSERVATION_S);

  // Simulation execution state
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [simulationResult, setSimulationResult] = useState(null);
  const [clientFetchTimeMs, setClientFetchTimeMs] = useState(null);
  const [error, setError] = useState(null);
  const [validationError, setValidationError] = useState(null);

  // Selected milestone comparison view tab: 'overall' | 'consecutive'
  const [deltaViewMode, setDeltaViewMode] = useState('overall');
  const [selectedConsecutiveStep, setSelectedConsecutiveStep] = useState(0);

  // Timer reference for elapsed time display
  const timerRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, []);

  // Helper to determine delivery duration for any action
  const getActionDuration = (act) => {
    if (!act) return 120;
    if (act.type === 'norepinephrine') return VERIFIED_NOREPINEPHRINE_DURATION_S;
    return calculateDeliveryDuration(act.volume_ml ?? 100);
  };

  // Handle switching the number of interventions (1, 2, or 3)
  const handleSetNumInterventions = (count) => {
    if (count === numInterventions || loading) return;
    setValidationError(null);

    setActions((prevActions) => {
      if (count === 1) {
        return [
          prevActions[0] || { id: 'bolus_1', type: 'saline', volume_ml: 100, start_time_s: baselinePeriod },
        ];
      }

      if (count === 2) {
        const act1 = prevActions[0] || { id: 'bolus_1', type: 'saline', volume_ml: 100, start_time_s: baselinePeriod };
        const act1End = Number(act1.start_time_s) + getActionDuration(act1);
        const act2 = prevActions[1] || {
          id: 'action_2',
          type: 'norepinephrine',
          concentration_ug_ml: VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
          rate_ml_per_min: VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
          duration_s: VERIFIED_NOREPINEPHRINE_DURATION_S,
          start_time_s: Math.max(act1End + 30, 210),
        };
        return [act1, act2];
      }

      if (count === 3) {
        const act1 = prevActions[0] || { id: 'bolus_1', type: 'saline', volume_ml: 100, start_time_s: baselinePeriod };
        const act1End = Number(act1.start_time_s) + getActionDuration(act1);
        const act2 = prevActions[1] || {
          id: 'action_2',
          type: 'norepinephrine',
          concentration_ug_ml: VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
          rate_ml_per_min: VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
          duration_s: VERIFIED_NOREPINEPHRINE_DURATION_S,
          start_time_s: Math.max(act1End + 30, 210),
        };
        const act2End = Number(act2.start_time_s) + getActionDuration(act2);
        const act3 = prevActions[2] || {
          id: 'action_3',
          type: 'saline',
          volume_ml: 250,
          start_time_s: Math.max(act2End + 30, 360),
        };
        return [act1, act2, act3];
      }

      return prevActions;
    });

    setNumInterventions(count);
  };

  // Toggle action type between saline bolus and norepinephrine infusion
  const handleToggleActionType = (index, newType) => {
    setValidationError(null);
    setActions((prev) => {
      const updated = [...prev];
      const cur = updated[index];
      if (newType === 'norepinephrine') {
        const next = {
          id: cur.id || `action_${index + 1}`,
          type: 'norepinephrine',
          concentration_ug_ml: VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
          rate_ml_per_min: VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
          duration_s: VERIFIED_NOREPINEPHRINE_DURATION_S,
          start_time_s: cur.start_time_s,
        };
        updated[index] = next;
      } else {
        const next = {
          id: cur.id || `action_${index + 1}`,
          type: 'saline',
          volume_ml: 100,
          start_time_s: cur.start_time_s,
        };
        updated[index] = next;
      }
      return updated;
    });
  };

  // Update a specific action's field (volume_ml or start_time_s)
  const handleUpdateAction = (index, field, value) => {
    setValidationError(null);
    setActions((prev) => {
      const updated = [...prev];
      updated[index] = { ...updated[index], [field]: value };
      return updated;
    });
  };

  // Live client-side scenario validation status
  const currentValidation = useMemo(() => {
    return validateSequentialSimulation({
      actions,
      baselineObservationS: baselinePeriod,
      postObservationS: postPeriod,
    });
  }, [actions, baselinePeriod, postPeriod]);

  // Execute simulation against backend
  const handleRunSimulation = async () => {
    // 1. Validate complete scenario before sending
    const validation = validateSequentialSimulation({
      actions,
      baselineObservationS: baselinePeriod,
      postObservationS: postPeriod,
    });

    if (!validation.valid) {
      setValidationError(validation.error);
      return;
    }
    setValidationError(null);

    // 2. Prevent duplicate submissions while in flight
    if (loading) return;

    setLoading(true);
    setError(null);
    setElapsedSeconds(0);
    const startTime = performance.now();

    // Start live elapsed timer
    timerRef.current = setInterval(() => {
      setElapsedSeconds((prev) => +(prev + 0.5).toFixed(1));
    }, 500);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const data = await executeSequentialWhatIf({
        actions,
        baselineObservationS: baselinePeriod,
        postObservationS: postPeriod,
        signal: controller.signal,
      });

      const totalTimeMs = Math.round(performance.now() - startTime);
      setClientFetchTimeMs(totalTimeMs);
      setSimulationResult(data);
      setSelectedConsecutiveStep(0);
    } catch (err) {
      console.error('BioGears simulation execution error:', err);
      setError(err.message || 'An unexpected error occurred during simulation.');
    } finally {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setLoading(false);
    }
  };

  // Overall Baseline-to-Final physiological deltas (Step F requirement)
  const overallDeltas = simulationResult
    ? computePhysiologicalDeltas(
        simulationResult.baseline_snapshot,
        simulationResult.final_snapshot
      )
    : null;

  // Deduplicate milestones without discarding distinct intervention delivery endpoints
  const deduplicatedMilestones = simulationResult
    ? deduplicateMilestones(
        simulationResult.baseline_snapshot,
        simulationResult.intermediate_milestones,
        simulationResult.final_snapshot
      )
    : [];

  // Consecutive milestone comparisons (Step G requirement: Delta x = x_curr - x_prev)
  const consecutiveComparisons = useMemo(() => {
    return computeConsecutiveMilestoneDeltas(deduplicatedMilestones);
  }, [deduplicatedMilestones]);

  const metadata = simulationResult?.metadata;
  const returnedActions = metadata?.actions || [];

  return (
    <motion.div
      className="glass-panel p-5 relative overflow-hidden"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Background ambient glow */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Top Workspace Tab Switcher */}
      <div className="flex items-center gap-2 mb-5 p-1 rounded-xl bg-slate-950/80 border border-slate-800 w-fit">
        <button
          onClick={() => setWorkspaceMode('comparison')}
          className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-2 cursor-pointer ${
            workspaceMode === 'comparison'
              ? 'bg-gradient-to-r from-emerald-600/30 to-cyan-600/30 text-cyan-200 shadow-sm border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <span>Scenario Comparison Workspace (Step H)</span>
        </button>
        <button
          onClick={() => setWorkspaceMode('single')}
          className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-2 cursor-pointer ${
            workspaceMode === 'single'
              ? 'bg-slate-800 text-white shadow-sm border border-slate-700'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <span>Single Scenario Simulation (Step G)</span>
        </button>
      </div>

      {workspaceMode === 'comparison' ? (
        <ScenarioComparisonWorkspace />
      ) : (
        <>
          {/* Header Section */}
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30 flex items-center justify-center">
            <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <div>
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              Sequential BioGears Physiological Simulation
              <span className="text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full uppercase tracking-wider">
                Continuous Multi-Action Timeline
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Intervention modeling on BioGears StandardMale.xml virtual subject • Mechanistic ODE solver via POST /biogears/whatif/sequential
            </p>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center gap-3">
          <button
            onClick={handleRunSimulation}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white text-xs font-semibold shadow-lg shadow-emerald-950/40 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {loading ? (
              <>
                <motion.div
                  className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
                />
                <span>Simulating BioGears ({elapsedSeconds}s)...</span>
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Run {numInterventions > 1 ? `${numInterventions}-Action` : 'What-If'} Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Intervention Builder Controls */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 mb-5 space-y-4">
        {/* Row 1: Number of Interventions Selector + Timeline Summary */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
              Number of Interventions:
            </span>
            <div className="flex items-center gap-1.5 bg-slate-950/80 p-1 rounded-lg border border-slate-800">
              {[1, 2, 3].map((count) => {
                const isActive = numInterventions === count;
                return (
                  <button
                    key={count}
                    type="button"
                    disabled={loading}
                    onClick={() => handleSetNumInterventions(count)}
                    className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                      isActive
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    } disabled:opacity-50`}
                  >
                    {count} {count === 1 ? 'Action' : 'Actions'}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-slate-950/60 border border-slate-800 font-mono text-[11px]">
              <span className="text-slate-500">Timeline Duration:</span>
              <span className={`font-semibold ${currentValidation.totalDuration > MAX_SIMULATION_DURATION_S ? 'text-rose-400' : 'text-emerald-300'}`}>
                {currentValidation.totalDuration}s
              </span>
              <span className="text-slate-500">/ {MAX_SIMULATION_DURATION_S}s max</span>
            </div>
            <div className="flex items-center gap-2 text-slate-400 text-[11px]">
              <span className="flex items-center gap-1 font-mono text-[10px] text-cyan-300 bg-cyan-950/40 px-1.5 py-0.5 rounded border border-cyan-800/40">
                Saline (50 mL/min)
              </span>
              <span className="flex items-center gap-1 font-mono text-[10px] text-purple-300 bg-purple-950/40 px-1.5 py-0.5 rounded border border-purple-800/40">
                Norepi (14 µg/mL)
              </span>
            </div>
          </div>
        </div>

        {/* Row 2: Configured Sequential Intervention Rows */}
        <div className="space-y-3">
          {actions.map((act, idx) => {
            const actionType = act.type || 'saline';
            const isNorepi = actionType === 'norepinephrine';
            const vol = Number(act.volume_ml ?? 100);
            const deliveryDuration = isNorepi ? VERIFIED_NOREPINEPHRINE_DURATION_S : calculateDeliveryDuration(vol);
            const startTime = Number(act.start_time_s);
            const endTime = startTime + deliveryDuration;
            const rowError = currentValidation.actionErrors[idx];

            return (
              <div
                key={act.id || idx}
                className={`p-3.5 rounded-lg border transition-all ${
                  rowError
                    ? 'bg-amber-950/20 border-amber-800/50'
                    : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                }`}
              >
                <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                  {/* Action Index & Type Selector */}
                  <div className="md:col-span-4 flex flex-col gap-1.5">
                    <div className="flex items-center gap-2">
                      <span className={`w-6 h-6 rounded-md flex items-center justify-center text-xs font-mono font-bold ${
                        isNorepi
                          ? 'bg-purple-500/15 border border-purple-500/30 text-purple-400'
                          : 'bg-emerald-500/15 border border-emerald-500/30 text-emerald-400'
                      }`}>
                        {idx + 1}
                      </span>
                      <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-xs">
                        <button
                          type="button"
                          disabled={loading}
                          onClick={() => handleToggleActionType(idx, 'saline')}
                          className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all ${
                            !isNorepi
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Saline Bolus
                        </button>
                        <button
                          type="button"
                          disabled={loading}
                          onClick={() => handleToggleActionType(idx, 'norepinephrine')}
                          className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all ${
                            isNorepi
                              ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Norepinephrine
                        </button>
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-400 font-mono ml-8">
                      Window: {deliveryDuration}s ({startTime}s → {endTime}s)
                    </p>
                  </div>

                  {/* Volume Selection (Saline) OR Fixed Protocol Display (Norepinephrine) */}
                  <div className="md:col-span-5">
                    {!isNorepi ? (
                      <div>
                        <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                          Bolus {idx + 1} Saline Volume
                        </label>
                        <div className="flex items-center gap-1.5">
                          {SUPPORTED_SALINE_VOLUMES.map((v) => {
                            const isSelected = vol === v;
                            return (
                              <button
                                key={v}
                                type="button"
                                disabled={loading}
                                onClick={() => handleUpdateAction(idx, 'volume_ml', v)}
                                className={`flex-1 py-1 px-2 text-xs font-semibold rounded-md border transition-all ${
                                  isSelected
                                    ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300 shadow-sm'
                                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                                } disabled:opacity-50`}
                              >
                                {v} mL
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                          Infusion {idx + 1} Verified Protocol
                        </label>
                        <div className="flex items-center gap-1.5 bg-slate-900/90 border border-purple-800/40 px-2.5 py-1 rounded-md text-[11px] font-mono">
                          <span className="text-purple-300 font-semibold">14.0 µg/mL</span>
                          <span className="text-slate-500">•</span>
                          <span className="text-slate-300">1.0 mL/min</span>
                          <span className="text-slate-500">•</span>
                          <span className="text-slate-300">120s duration</span>
                          <span className="text-emerald-400 text-[10px] ml-auto">Clean Stop</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Start Time Input (seconds) */}
                  <div className="md:col-span-3">
                    <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                      Start Time (simulated sec)
                    </label>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min="0"
                        max="1700"
                        step="10"
                        disabled={loading}
                        value={act.start_time_s}
                        onChange={(e) => handleUpdateAction(idx, 'start_time_s', Number(e.target.value))}
                        className={`w-full bg-slate-900 border rounded-md px-3 py-1 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500 disabled:opacity-50 ${
                          rowError ? 'border-amber-600/80 text-amber-200' : 'border-slate-800'
                        }`}
                      />
                      <span className="text-xs text-slate-500 font-mono">sec</span>
                    </div>
                  </div>
                </div>

                {/* Inline Action Validation Warning */}
                {rowError && (
                  <div className="mt-2 pt-2 border-t border-amber-900/40 text-[11px] text-amber-300 flex items-center gap-1.5">
                    <svg className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span>{rowError}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Row 3: Shared Observation Windows */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-slate-800/80">
          <div>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Baseline Stabilization Observation
            </label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min="10"
                max="600"
                step="10"
                disabled={loading}
                value={baselinePeriod}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setBaselinePeriod(val);
                  setValidationError(null);
                }}
                className="w-full bg-slate-950/70 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500 disabled:opacity-50"
              />
              <span className="text-xs text-slate-500 font-mono">sec</span>
            </div>
          </div>

          <div>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Final Post-Intervention Observation
            </label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min="10"
                max="600"
                step="10"
                disabled={loading}
                value={postPeriod}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setPostPeriod(val);
                  setValidationError(null);
                }}
                className="w-full bg-slate-950/70 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500 disabled:opacity-50"
              />
              <span className="text-xs text-slate-500 font-mono">sec</span>
            </div>
          </div>
        </div>
      </div>

      {/* Validation Error Message Alert */}
      {validationError && (
        <div className="mb-4 p-3 rounded-lg bg-amber-950/40 border border-amber-800/40 text-amber-300 text-xs flex items-center justify-between">
          <span className="flex items-center gap-2">
            <svg className="w-4 h-4 text-amber-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>{validationError}</span>
          </span>
          <button onClick={() => setValidationError(null)} className="text-amber-400 hover:text-amber-200 text-xs cursor-pointer">
            Dismiss
          </button>
        </div>
      )}

      {/* Backend / Network Error Message with Retry */}
      {error && (
        <div className="mb-5 p-3 rounded-lg bg-rose-950/40 border border-rose-800/40 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-rose-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{error}</span>
          </div>
          <button
            onClick={handleRunSimulation}
            disabled={loading}
            className="px-2.5 py-1 rounded bg-rose-800/60 hover:bg-rose-700 text-white font-medium text-xs transition-colors cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading In-Progress State Banner */}
      {loading && (
        <div className="mb-5 p-4 rounded-xl bg-cyan-950/30 border border-cyan-800/40 flex items-center gap-4">
          <motion.div
            className="w-7 h-7 border-3 border-cyan-500/30 border-t-cyan-400 rounded-full flex-shrink-0"
            animate={{ rotate: 360 }}
            transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
          />
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-cyan-300">
                Executing BioGears Sequential Mechanistic Engine...
              </p>
              <span className="font-mono text-xs text-cyan-400 font-bold">{elapsedSeconds}s</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Simulating continuous cardiovascular, fluid, and respiratory ODEs for virtual subject StandardMale.xml: {actions.length} {actions.length === 1 ? 'bolus' : 'boluses'} ({actions.map((a, i) => `B${i + 1}: ${a.volume_ml}mL @ ${a.start_time_s}s`).join(', ')}). Fresh live executions may take 30–90 seconds.
            </p>
          </div>
        </div>
      )}

      {/* Simulation Results Display */}
      {simulationResult && !loading && (
        <div className="space-y-5">
          {/* A. Simulation Status & Scenario Provenance Summary */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-3.5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-3">
                {/* Cache vs Live Badge */}
                <div
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${
                    metadata?.is_cached
                      ? 'bg-purple-500/15 text-purple-300 border-purple-500/30'
                      : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      metadata?.is_cached ? 'bg-purple-400' : 'bg-emerald-400'
                    }`}
                  />
                  <span>{metadata?.is_cached ? 'Cached Result (Cache Hit)' : 'Fresh BioGears Execution (Live)'}</span>
                </div>

                <div className="flex items-center gap-1.5 text-xs text-slate-300">
                  <span className="text-slate-500 font-medium">Scenario:</span>
                  <span className="font-mono font-medium text-slate-200">{metadata?.scenario_name}</span>
                </div>

                <div className="flex items-center gap-1.5 text-xs text-slate-300">
                  <span className="text-slate-500 font-medium">Subject:</span>
                  <span className="font-mono text-slate-200">{metadata?.patient_file}</span>
                </div>

                <div className="flex items-center gap-1.5 text-xs text-slate-300">
                  <span className="text-slate-500 font-medium">Interventions:</span>
                  <span className="font-mono font-semibold text-emerald-300">
                    {returnedActions.length || actions.length}
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
                <div>
                  <span className="text-slate-500">Simulated: </span>
                  <span className="font-mono font-medium text-slate-200">{metadata?.simulated_duration_s}s</span>
                </div>
                <div>
                  <span className="text-slate-500">Data Points: </span>
                  <span className="font-mono font-medium text-slate-200">{metadata?.row_count}</span>
                </div>
                <div>
                  <span className="text-slate-500">Runtime: </span>
                  {metadata?.wall_clock_seconds !== null && metadata?.wall_clock_seconds !== undefined ? (
                    metadata.wall_clock_seconds > 0 ? (
                      <span className="font-mono font-medium text-slate-200">{metadata.wall_clock_seconds.toFixed(2)}s</span>
                    ) : (
                      <span className="font-mono text-slate-400" title="Cached run without wall clock benchmark recorded">
                        0.0s (Cached artifact, fetched in {clientFetchTimeMs}ms)
                      </span>
                    )
                  ) : (
                    <span className="font-mono text-slate-400">{clientFetchTimeMs}ms (client)</span>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  <span className="text-emerald-400 font-medium">Completed</span>
                </div>
              </div>
            </div>

            {/* Configured Interventions List Badge Chips */}
            {returnedActions.length > 0 && (
              <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-slate-500 font-medium">Ordered Actions:</span>
                {returnedActions.map((act, i) => {
                  const isNE = act.type === 'norepinephrine' || act.substance === 'Norepinephrine';
                  return (
                    <div
                      key={act.action_index ?? i}
                      className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-slate-950/80 border border-slate-800 font-mono text-[11px]"
                    >
                      <span className={isNE ? 'text-purple-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                        {isNE ? `NE${i + 1}:` : `B${i + 1}:`}
                      </span>
                      <span className="text-slate-200">
                        {isNE ? `${act.concentration_ug_ml || 14} µg/mL` : `${act.bag_volume_ml} mL`}
                      </span>
                      <span className="text-slate-500">@</span>
                      <span className="text-slate-300">{act.start_time_s}s</span>
                      <span className="text-slate-500">
                        ({act.delivery_duration_s ?? act.duration_s ?? 120}s)
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* B. Physiological Snapshots & Deltas */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Baseline Physiology Card */}
            <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-4">
              <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-slate-400" />
                  1. Baseline Physiology
                </span>
                <span className="text-[10px] text-slate-500 font-mono">
                  t = {simulationResult.baseline_snapshot.actual_time_s}s
                </span>
              </div>

              <div className="space-y-2 text-xs">
                {overallDeltas?.map((d) => (
                  <div key={d.key} className="flex justify-between items-center py-1 border-b border-slate-800/40">
                    <span className="text-slate-400">{d.label}</span>
                    <span className="font-mono font-medium text-slate-200">
                      {d.formattedBase} <span className="text-[10px] text-slate-500 font-normal">{d.unit}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Final Physiology Card */}
            <div className="rounded-xl bg-emerald-950/20 border border-emerald-800/40 p-4">
              <div className="flex items-center justify-between mb-3 pb-2 border-b border-emerald-900/40">
                <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  2. Final Snapshot (Post-Intervention)
                </span>
                <span className="text-[10px] text-emerald-400/80 font-mono">
                  t = {simulationResult.final_snapshot.actual_time_s}s
                </span>
              </div>

              <div className="space-y-2 text-xs">
                {overallDeltas?.map((d) => (
                  <div key={d.key} className="flex justify-between items-center py-1 border-b border-emerald-900/20">
                    <span className="text-slate-400">{d.label}</span>
                    <span className="font-mono font-medium text-emerald-200">
                      {d.formattedFinal} <span className="text-[10px] text-emerald-500/70 font-normal">{d.unit}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* C. Physiological Changes (Deltas) Card — Switcher for Overall vs Consecutive */}
            <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    3. Physiological Changes (Δ)
                  </span>

                  {/* Switch between Overall Δ and Consecutive Milestone Δ */}
                  {consecutiveComparisons.length > 1 && (
                    <div className="flex items-center gap-1 bg-slate-950 p-0.5 rounded border border-slate-800 text-[10px]">
                      <button
                        type="button"
                        onClick={() => setDeltaViewMode('overall')}
                        className={`px-1.5 py-0.5 rounded font-medium transition-all ${
                          deltaViewMode === 'overall'
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Net (Final–Base)
                      </button>
                      <button
                        type="button"
                        onClick={() => setDeltaViewMode('consecutive')}
                        className={`px-1.5 py-0.5 rounded font-medium transition-all ${
                          deltaViewMode === 'consecutive'
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Step-by-Step ({consecutiveComparisons.length})
                      </button>
                    </div>
                  )}
                </div>

                {/* Sub-selector for consecutive comparison steps */}
                {deltaViewMode === 'consecutive' && consecutiveComparisons.length > 0 && (
                  <div className="mb-2.5">
                    <div className="flex items-center gap-1 overflow-x-auto pb-1">
                      {consecutiveComparisons.map((comp, idx) => (
                        <button
                          key={comp.stepIndex}
                          type="button"
                          onClick={() => setSelectedConsecutiveStep(idx)}
                          className={`px-2 py-0.5 text-[10px] rounded font-mono whitespace-nowrap transition-all ${
                            selectedConsecutiveStep === idx
                              ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/40 font-bold'
                              : 'bg-slate-950/60 text-slate-400 border border-slate-800 hover:text-slate-200'
                          }`}
                        >
                          Step {comp.stepIndex}: {comp.fromTime}s→{comp.toTime}s
                        </button>
                      ))}
                    </div>
                    <p className="text-[10px] text-slate-400 font-medium truncate mt-1" title={consecutiveComparisons[selectedConsecutiveStep]?.label}>
                      {consecutiveComparisons[selectedConsecutiveStep]?.label}
                    </p>
                  </div>
                )}

                {/* Delta values display */}
                {(() => {
                  const activeDeltas = deltaViewMode === 'consecutive' && consecutiveComparisons[selectedConsecutiveStep]
                    ? consecutiveComparisons[selectedConsecutiveStep].deltas
                    : overallDeltas;

                  return (
                    <>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {activeDeltas?.slice(0, 4).map((d) => {
                          const isPositive = d.delta !== null && d.delta > 0;
                          const isNegative = d.delta !== null && d.delta < 0;
                          const colorClass = isPositive
                            ? 'text-emerald-400'
                            : isNegative
                            ? 'text-cyan-300'
                            : 'text-slate-300';

                          return (
                            <div key={d.key} className="bg-slate-800/40 p-2 rounded-lg border border-slate-700/40">
                              <p className="text-[10px] text-slate-400 truncate">{d.label}</p>
                              <p className={`text-sm font-mono font-bold ${colorClass}`}>
                                {d.formattedDelta}{' '}
                                <span className="text-[10px] font-normal text-slate-400">{d.unit}</span>
                              </p>
                            </div>
                          );
                        })}
                      </div>

                      {activeDeltas && activeDeltas.length > 4 && (
                        <div className="mt-2.5 pt-2 border-t border-slate-800/80 space-y-1 text-[11px]">
                          {activeDeltas.slice(4).map((d) => (
                            <div key={d.key} className="flex justify-between items-center py-0.5">
                              <span className="text-slate-400">{d.label}:</span>
                              <span className="font-mono text-cyan-300 font-medium">
                                {d.formattedDelta} {d.unit}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>

              <div className="pt-3 border-t border-slate-800 mt-2">
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  Signed absolute changes (Δx = x_current - x_previous) computed from continuous mechanistic ODE differential equations.
                </p>
              </div>
            </div>
          </div>

          {/* D. Scenario Milestones Timeline (Deduplicated Chronological Milestones) */}
          <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-4">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
              <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                Scenario Milestones Timeline
                <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                  {deduplicatedMilestones.length} milestones
                </span>
              </h4>
              <span className="text-[10px] text-slate-500">Deduplicated chronological milestones</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {deduplicatedMilestones.map((m, idx) => {
                const snap = m.snapshot;
                const isBaseline = m.category === 'baseline';
                const isFinal = m.category === 'final';
                const isHold = m.category === 'hold' || m.phase.startsWith('intermediate_hold');
                const borderColor = isBaseline
                  ? 'border-slate-700/60'
                  : isFinal
                  ? 'border-emerald-700/50'
                  : isHold
                  ? 'border-amber-700/40'
                  : 'border-cyan-700/50';
                const badgeColor = isBaseline
                  ? 'bg-slate-800 text-slate-300'
                  : isFinal
                  ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-700/50'
                  : isHold
                  ? 'bg-amber-950/40 text-amber-300 border border-amber-700/40'
                  : 'bg-cyan-950/60 text-cyan-300 border border-cyan-700/50';

                return (
                  <div
                    key={`${m.phase}_${m.target_time_s}_${idx}`}
                    className={`rounded-lg bg-slate-950/60 border ${borderColor} p-3 text-xs flex flex-col justify-between`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-wider ${badgeColor}`}>
                          {m.phase.replace(/_/g, ' ')}
                        </span>
                        <span className="font-mono text-[10px] text-slate-400">
                          t = {m.actual_time_s}s
                        </span>
                      </div>
                      <p className="font-medium text-slate-300 mb-2 truncate" title={m.description}>
                        {m.description}
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] pt-2 border-t border-slate-800/60 font-mono">
                      <div className="text-slate-400">
                        HR: <span className="text-slate-200">{snap?.heart_rate_bpm?.toFixed(1) ?? '—'}</span>
                      </div>
                      <div className="text-slate-400">
                        MAP: <span className="text-slate-200">{snap?.mean_arterial_pressure_mmhg?.toFixed(1) ?? '—'}</span>
                      </div>
                      <div className="text-slate-400">
                        BV: <span className="text-slate-200">{snap?.blood_volume_l?.toFixed(2) ?? '—'}L</span>
                      </div>
                      <div className="text-slate-400">
                        CO: <span className="text-slate-200">{snap?.cardiac_output_l_min?.toFixed(2) ?? '—'}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* E. Physiological Trajectory Visualizer (Multi-Intervention Chart) */}
          <WhatIfTrajectoryChart
            trajectory={simulationResult.trajectory}
            milestones={deduplicatedMilestones}
            actions={returnedActions.length > 0 ? returnedActions : actions}
          />

          {/* F. Clinical and Simulation Disclaimer */}
          <div className="rounded-xl bg-slate-950/90 border border-amber-900/30 p-4 text-xs">
            <div className="flex items-start gap-3">
              <div className="w-5 h-5 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-3 h-3 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="space-y-1.5 text-slate-400">
                <p className="font-semibold text-slate-300">Clinical & Simulation Disclaimer</p>
                <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-400">
                  <li>
                    The simulation currently uses BioGears' <span className="text-slate-300 font-mono">StandardMale.xml</span> patient configuration.
                  </li>
                  <li>
                    It is a mechanistic physiological simulation, not an individualized model of the selected dashboard patient.
                  </li>
                  <li>
                    Simulated physiological changes do not establish clinical effectiveness or safety.
                  </li>
                  <li>
                    This endpoint returns simulated physiological results; it has not independently recalculated or validated the BiLSTM sepsis-risk prediction.
                  </li>
                </ul>
                {simulationResult.provenance_disclaimer && (
                  <p className="text-[10px] text-slate-500 pt-1.5 border-t border-slate-800 font-mono">
                    Provenance: {simulationResult.provenance_disclaimer}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Initial Empty State before any simulation has run */}
      {!simulationResult && !loading && !error && (
        <div className="p-8 text-center bg-slate-900/30 rounded-xl border border-slate-800/60">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-3">
            <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <h3 className="text-sm font-semibold text-slate-200 mb-1">
            Ready to Simulate Saline Intervention{numInterventions > 1 ? 's' : ''}
          </h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto mb-4">
            Configure one, two, or three sequential saline boluses (100, 250, or 500 mL) and click &quot;Run {numInterventions > 1 ? `${numInterventions}-Bolus` : 'What-If'} Simulation&quot; to solve the continuous physiological trajectory using the verified BioGears mathematical model.
          </p>
          <button
            onClick={handleRunSimulation}
            className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-lg shadow-emerald-950/40 transition-all cursor-pointer"
          >
            Run BioGears Simulation ({actions.map((a) => `${a.volume_ml}mL`).join(' + ')} Saline)
          </button>
        </div>
      )}
        </>
      )}
    </motion.div>
  );
}

