/**
 * biogearsApi.js — BioGears Sequential What-If Simulation API Service
 * ==================================================================
 * Handles communication with POST /biogears/whatif/sequential, payload construction,
 * input validation, milestone deduplication, physiological delta calculations,
 * and error classification for single and multi-intervention scenarios.
 */

export const API_BASE = import.meta?.env?.VITE_API_BASE || 'http://localhost:8000';

export const SUPPORTED_SALINE_VOLUMES = [100, 250, 500];
export const DEFAULT_BASELINE_OBSERVATION_S = 60;
export const DEFAULT_POST_OBSERVATION_S = 60;
export const SALINE_INFUSION_RATE_ML_PER_MIN = 50.0;
export const MAX_SIMULATION_DURATION_S = 1800.0;
export const MAX_OBSERVATION_PERIOD_S = 600.0;
export const MIN_ACTIONS = 1;
export const MAX_ACTIONS = 3;

// Verified Norepinephrine Infusion constants
export const VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML = 14.0;
export const VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN = 1.0;
export const VERIFIED_NOREPINEPHRINE_DURATION_S = 120.0;
export const SUPPORTED_INTERVENTION_TYPES = ['saline', 'norepinephrine'];

/**
 * Calculate the active delivery duration in seconds for an intervention.
 * - Saline bolus: (volumeMl / 50 mL/min) * 60s (e.g. 100 mL -> 120s, 250 mL -> 300s, 500 mL -> 600s)
 * - Norepinephrine infusion: locked verified 120s duration.
 */
export function calculateDeliveryDuration(volumeOrAction) {
  if (typeof volumeOrAction === 'object' && volumeOrAction !== null) {
    if (volumeOrAction.type === 'norepinephrine') {
      return Number(volumeOrAction.duration_s ?? volumeOrAction.durationS ?? VERIFIED_NOREPINEPHRINE_DURATION_S);
    }
    const vol = Number(volumeOrAction.volume_ml ?? volumeOrAction.volumeMl);
    return calculateDeliveryDuration(vol);
  }
  const vol = Number(volumeOrAction);
  if (!Number.isFinite(vol) || vol <= 0) return 0;
  return (vol / SALINE_INFUSION_RATE_ML_PER_MIN) * 60.0;
}

/**
 * Validate a multi-intervention sequential scenario before submitting to backend.
 * Supports both Saline boluses and Norepinephrine infusions.
 * Checks:
 *   1. Exactly 1 to 3 sequential interventions.
 *   2. For saline: volume is verified (100, 250, or 500 mL).
 *   3. For norepinephrine: verified concentration (14 ug/mL), rate (1 mL/min), duration (120s).
 *   4. Each start time is numeric, valid, and non-negative.
 *   5. First intervention occurs at or after baseline stabilization.
 *   6. Subsequent interventions do not overlap prior delivery windows.
 *   7. Baseline and post-observation periods are within [10, 600]s.
 *   8. Total simulated duration does not exceed the backend limit (1800s).
 *
 * Returns: { valid: boolean, error: string | null, actionErrors: (string | null)[], totalDuration: number }
 */
export function validateSequentialSimulation({
  actions = [],
  baselineObservationS = DEFAULT_BASELINE_OBSERVATION_S,
  postObservationS = DEFAULT_POST_OBSERVATION_S,
} = {}) {
  const baseline = Number(baselineObservationS);
  if (!Number.isFinite(baseline) || baseline <= 0) {
    return {
      valid: false,
      error: 'Baseline observation period must be a positive number of seconds (e.g. 60).',
      actionErrors: [],
      totalDuration: 0,
    };
  }
  if (baseline > MAX_OBSERVATION_PERIOD_S) {
    return {
      valid: false,
      error: `Baseline observation period exceeds maximum recommended limit (${MAX_OBSERVATION_PERIOD_S}s).`,
      actionErrors: [],
      totalDuration: 0,
    };
  }

  const post = Number(postObservationS);
  if (!Number.isFinite(post) || post <= 0) {
    return {
      valid: false,
      error: 'Post-intervention observation period must be a positive number of seconds (e.g. 60).',
      actionErrors: [],
      totalDuration: 0,
    };
  }
  if (post > MAX_OBSERVATION_PERIOD_S) {
    return {
      valid: false,
      error: `Post-intervention observation period exceeds maximum recommended limit (${MAX_OBSERVATION_PERIOD_S}s).`,
      actionErrors: [],
      totalDuration: 0,
    };
  }

  if (!Array.isArray(actions) || actions.length < MIN_ACTIONS || actions.length > MAX_ACTIONS) {
    return {
      valid: false,
      error: `Scenarios must configure between 1 and 3 sequential interventions (got ${Array.isArray(actions) ? actions.length : 0}).`,
      actionErrors: [],
      totalDuration: 0,
    };
  }

  const actionErrors = new Array(actions.length).fill(null);
  let currentSimTime = baseline;
  let overallError = null;

  for (let i = 0; i < actions.length; i++) {
    const act = actions[i];
    const actionType = act.type || 'saline';
    const start = Number(act.start_time_s ?? act.startTimeS);

    if (!SUPPORTED_INTERVENTION_TYPES.includes(actionType)) {
      const msg = `Intervention ${i + 1}: Unsupported intervention type '${actionType}'. Supported: ${SUPPORTED_INTERVENTION_TYPES.join(', ')}.`;
      actionErrors[i] = msg;
      if (!overallError) overallError = msg;
      continue;
    }

    let deliveryDuration = 0;
    if (actionType === 'saline') {
      const vol = Number(act.volume_ml ?? act.volumeMl);
      if (!SUPPORTED_SALINE_VOLUMES.includes(vol)) {
        const msg = `Bolus ${i + 1}: Invalid saline volume (${act.volume_ml ?? act.volumeMl} mL). Supported volumes: ${SUPPORTED_SALINE_VOLUMES.join(', ')} mL.`;
        actionErrors[i] = msg;
        if (!overallError) overallError = msg;
        continue;
      }
      deliveryDuration = calculateDeliveryDuration(vol);
    } else if (actionType === 'norepinephrine') {
      const conc = Number(act.concentration_ug_ml ?? act.concentrationUgMl ?? VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML);
      const rate = Number(act.rate_ml_per_min ?? act.rateMlPerMin ?? VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN);
      const dur = Number(act.duration_s ?? act.durationS ?? VERIFIED_NOREPINEPHRINE_DURATION_S);

      if (conc !== VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML) {
        const msg = `Infusion ${i + 1}: Unsupported concentration (${conc} ug/mL). Supported: ${VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML} ug/mL.`;
        actionErrors[i] = msg;
        if (!overallError) overallError = msg;
        continue;
      }
      if (rate !== VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN) {
        const msg = `Infusion ${i + 1}: Unsupported rate (${rate} mL/min). Supported: ${VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN} mL/min.`;
        actionErrors[i] = msg;
        if (!overallError) overallError = msg;
        continue;
      }
      if (dur !== VERIFIED_NOREPINEPHRINE_DURATION_S) {
        const msg = `Infusion ${i + 1}: Unsupported duration (${dur} s). Supported: ${VERIFIED_NOREPINEPHRINE_DURATION_S} s.`;
        actionErrors[i] = msg;
        if (!overallError) overallError = msg;
        continue;
      }
      deliveryDuration = dur;
    }

    if (!Number.isFinite(start) || start < 0) {
      const msg = `Intervention ${i + 1}: Start time must be a non-negative numeric value.`;
      actionErrors[i] = msg;
      if (!overallError) overallError = msg;
      continue;
    }

    if (start < currentSimTime) {
      const label = actionType === 'saline' ? 'Bolus' : 'Infusion';
      const conflictMsg = i === 0
        ? `${label} 1 start time (${start}s) precedes baseline stabilization completion (${baseline}s). Increase start time to at least ${baseline}s.`
        : `${label} ${i + 1} start time (${start}s) precedes previous bolus completion (${currentSimTime}s). Interventions must be sequential and non-overlapping. Increase start time to at least ${currentSimTime}s.`;
      actionErrors[i] = conflictMsg;
      if (!overallError) overallError = conflictMsg;
      continue;
    }

    currentSimTime = start + deliveryDuration;
  }

  const totalSimDuration = currentSimTime + post;

  if (totalSimDuration > MAX_SIMULATION_DURATION_S) {
    const exceedsMsg = `Total simulated duration is ${totalSimDuration}s, exceeding the ${MAX_SIMULATION_DURATION_S}s maximum limit. Please reduce intervention start times, decrease bolus volumes, or reduce the post-observation duration.`;
    return {
      valid: false,
      error: exceedsMsg,
      actionErrors,
      totalDuration: totalSimDuration,
    };
  }

  if (overallError) {
    return {
      valid: false,
      error: overallError,
      actionErrors,
      totalDuration: totalSimDuration,
    };
  }

  return {
    valid: true,
    error: null,
    actionErrors,
    totalDuration: totalSimDuration,
  };
}

/**
 * Validate intervention inputs before submitting to the backend.
 * Supports both legacy single-bolus signature:
 *   validateSimulationInputs(volumeMl, baselineObservationS, postObservationS)
 * and multi-bolus array signature:
 *   validateSimulationInputs(actionsArray, baselineObservationS, postObservationS)
 * Returns { valid: boolean, error: string | null }
 */
export function validateSimulationInputs(volumeOrActions, baselineObservationS = DEFAULT_BASELINE_OBSERVATION_S, postObservationS = DEFAULT_POST_OBSERVATION_S) {
  // If passed an array of actions or object with actions
  if (Array.isArray(volumeOrActions)) {
    const res = validateSequentialSimulation({
      actions: volumeOrActions,
      baselineObservationS,
      postObservationS,
    });
    return { valid: res.valid, error: res.error };
  }

  if (typeof volumeOrActions === 'object' && volumeOrActions !== null && Array.isArray(volumeOrActions.actions)) {
    const res = validateSequentialSimulation(volumeOrActions);
    return { valid: res.valid, error: res.error };
  }

  // Legacy single-volume validation preserving exact Step F behavior & error messages
  const vol = Number(volumeOrActions);
  if (!SUPPORTED_SALINE_VOLUMES.includes(vol)) {
    return {
      valid: false,
      error: `Invalid saline volume: ${volumeOrActions} mL. Supported volumes are: ${SUPPORTED_SALINE_VOLUMES.join(', ')} mL.`,
    };
  }

  const baseline = Number(baselineObservationS);
  if (!Number.isFinite(baseline) || baseline <= 0) {
    return {
      valid: false,
      error: 'Baseline observation period must be a positive number of seconds (e.g. 60).',
    };
  }
  if (baseline > MAX_OBSERVATION_PERIOD_S) {
    return {
      valid: false,
      error: 'Baseline observation period exceeds maximum recommended limit (600s).',
    };
  }

  const post = Number(postObservationS);
  if (!Number.isFinite(post) || post <= 0) {
    return {
      valid: false,
      error: 'Post-intervention observation period must be a positive number of seconds (e.g. 60).',
    };
  }
  if (post > MAX_OBSERVATION_PERIOD_S) {
    return {
      valid: false,
      error: 'Post-intervention observation period exceeds maximum recommended limit (600s).',
    };
  }

  return { valid: true, error: null };
}

/**
 * Construct the SequentialWhatIfRequest payload for single or multi-action saline simulation.
 * Supports legacy single-bolus signature:
 *   buildSequentialPayload(volumeMl, baselineObservationS, postObservationS)
 * as well as multi-action array signature:
 *   buildSequentialPayload(actionsArray, baselineObservationS, postObservationS)
 */
export function buildSequentialPayload(actionsOrVolume, baselineObservationS = 60, postObservationS = 60) {
  // Single bolus numeric call (backward compatibility with Step F)
  if (typeof actionsOrVolume === 'number' || typeof actionsOrVolume === 'string') {
    const vol = Number(actionsOrVolume);
    const baseline = Number(baselineObservationS);
    const post = Number(postObservationS);

    return {
      scenario_name: `SeqSaline_1x${vol}mL`,
      patient_file: 'StandardMale.xml',
      baseline_stabilization_s: baseline,
      post_observation_s: post,
      samples_per_second: 0.2,
      actions: [
        {
          type: 'saline',
          volume_ml: vol,
          start_time_s: baseline,
        },
      ],
    };
  }

  // Multi-action array or object
  const actionsList = Array.isArray(actionsOrVolume)
    ? actionsOrVolume
    : (actionsOrVolume?.actions || []);
  const baseline = Number(baselineObservationS ?? actionsOrVolume?.baselineObservationS ?? 60);
  const post = Number(postObservationS ?? actionsOrVolume?.postObservationS ?? 60);

  const mappedActions = actionsList.map((act) => {
    const actionType = act.type || 'saline';
    if (actionType === 'norepinephrine') {
      return {
        type: 'norepinephrine',
        concentration_ug_ml: Number(act.concentration_ug_ml ?? act.concentrationUgMl ?? VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML),
        rate_ml_per_min: Number(act.rate_ml_per_min ?? act.rateMlPerMin ?? VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN),
        duration_s: Number(act.duration_s ?? act.durationS ?? VERIFIED_NOREPINEPHRINE_DURATION_S),
        start_time_s: Number(act.start_time_s ?? act.startTimeS),
      };
    }
    return {
      type: 'saline',
      volume_ml: Number(act.volume_ml ?? act.volumeMl),
      start_time_s: Number(act.start_time_s ?? act.startTimeS),
    };
  });

  const hasNorepi = mappedActions.some((a) => a.type === 'norepinephrine');
  const hasSaline = mappedActions.some((a) => a.type === 'saline');
  let scenarioName = '';
  if (hasNorepi && hasSaline) {
    const summary = mappedActions.map((a) => a.type === 'saline' ? `${Math.round(a.volume_ml)}mL` : `${Math.round(a.concentration_ug_ml)}ugNE`).join('_');
    scenarioName = `SeqMixed_${mappedActions.length}actions_${summary}`;
  } else if (hasNorepi) {
    const summary = mappedActions.map((a) => `${Math.round(a.concentration_ug_ml)}ugNE`).join('_');
    scenarioName = `SeqNorepi_${mappedActions.length}actions_${summary}`;
  } else {
    const volSummary = mappedActions.map((a) => `${Math.round(a.volume_ml)}mL`).join('_');
    scenarioName = `SeqSaline_${mappedActions.length}actions_${volSummary}`;
  }

  return {
    scenario_name: scenarioName,
    patient_file: 'StandardMale.xml',
    baseline_stabilization_s: baseline,
    post_observation_s: post,
    samples_per_second: 0.2,
    actions: mappedActions,
  };
}

/**
 * Deduplicate and normalize milestones across baseline_snapshot,
 * intermediate_milestones, and final_snapshot.
 *
 * Backend returns:
 *   - baseline_snapshot (milestone_type: "baseline", at t = baseline_s)
 *   - intermediate_milestones (array containing "baseline", "intervention_0", "post_observation", etc.)
 *   - final_snapshot (milestone_type: "final_state", at t = final_s)
 *
 * This function guarantees:
 *   1. The baseline is NOT displayed twice.
 *   2. The final snapshot is NOT displayed twice.
 *   3. Distinct intervention delivery milestones are preserved.
 *   4. Milestones are in strictly non-decreasing chronological order.
 */
export function deduplicateMilestones(baselineSnapshot, intermediateMilestones = [], finalSnapshot) {
  const milestones = [];
  const seenTimes = new Set();

  const addMilestone = (entry, category) => {
    if (!entry) return;
    const snap = entry.snapshot || entry;
    const targetTime = Number(entry.target_time_s ?? snap.target_time_s ?? 0);
    const actualTime = Number(entry.actual_time_s ?? snap.actual_time_s ?? targetTime);
    const key = `${category}_${Math.round(targetTime * 10) / 10}`;

    if (seenTimes.has(key)) return;
    seenTimes.add(key);

    milestones.push({
      category,
      phase: entry.phase || category,
      description: entry.description || snap.description || `${category} (t = ${targetTime}s)`,
      target_time_s: targetTime,
      actual_time_s: actualTime,
      snapshot: snap,
    });
  };

  // 1. Add baseline snapshot
  if (baselineSnapshot) {
    addMilestone({
      phase: 'baseline',
      description: baselineSnapshot.description || `Baseline Stabilization (${baselineSnapshot.target_time_s}s)`,
      target_time_s: baselineSnapshot.target_time_s,
      actual_time_s: baselineSnapshot.actual_time_s,
      snapshot: baselineSnapshot,
    }, 'baseline');
  }

  // 2. Add intermediate milestones (skipping if duplicate of baseline or post_observation/final)
  if (Array.isArray(intermediateMilestones)) {
    for (const m of intermediateMilestones) {
      const p = (m.phase || '').toLowerCase();
      const isBaseline = p === 'baseline' || m.snapshot?.milestone_type === 'baseline';
      const isFinal = p === 'post_observation' || m.snapshot?.milestone_type === 'final_observation' || m.snapshot?.milestone_type === 'final_state';

      const timeMatchBaseline = baselineSnapshot && Math.abs(m.target_time_s - baselineSnapshot.target_time_s) < 0.5;
      const timeMatchFinal = finalSnapshot && Math.abs(m.target_time_s - finalSnapshot.target_time_s) < 0.5;

      if (isBaseline || timeMatchBaseline) {
        continue;
      }
      if (isFinal || timeMatchFinal) {
        continue;
      }

      const cat = p.startsWith('intermediate_hold') ? 'hold' : 'intervention';
      addMilestone(m, cat);
    }
  }

  // 3. Add final snapshot
  if (finalSnapshot) {
    addMilestone({
      phase: 'final_state',
      description: finalSnapshot.description || `Final Post-Intervention State (${finalSnapshot.target_time_s}s)`,
      target_time_s: finalSnapshot.target_time_s,
      actual_time_s: finalSnapshot.actual_time_s,
      snapshot: finalSnapshot,
    }, 'final');
  }

  // Sort chronologically
  milestones.sort((a, b) => a.target_time_s - b.target_time_s);
  return milestones;
}

/**
 * Safely calculate physiological changes (Delta = current - previous)
 * for all recognized physiological metrics between two snapshots.
 */
export function computePhysiologicalDeltas(previousSnapshot, currentSnapshot) {
  if (!previousSnapshot || !currentSnapshot) return null;

  const calculateDelta = (currVal, prevVal) => {
    if (typeof currVal !== 'number' || typeof prevVal !== 'number') return null;
    if (isNaN(currVal) || isNaN(prevVal)) return null;
    return currVal - prevVal;
  };

  const metrics = [
    {
      key: 'heart_rate_bpm',
      label: 'Heart Rate',
      unit: 'bpm',
      base: previousSnapshot.heart_rate_bpm,
      final: currentSnapshot.heart_rate_bpm,
      precision: 1,
    },
    {
      key: 'mean_arterial_pressure_mmhg',
      label: 'Mean Arterial Pressure (MAP)',
      unit: 'mmHg',
      base: previousSnapshot.mean_arterial_pressure_mmhg,
      final: currentSnapshot.mean_arterial_pressure_mmhg,
      precision: 1,
    },
    {
      key: 'blood_volume_l',
      label: 'Intravascular Blood Volume',
      unit: 'L',
      base: previousSnapshot.blood_volume_l,
      final: currentSnapshot.blood_volume_l,
      precision: 3,
    },
    {
      key: 'cardiac_output_l_min',
      label: 'Cardiac Output',
      unit: 'L/min',
      base: previousSnapshot.cardiac_output_l_min,
      final: currentSnapshot.cardiac_output_l_min,
      precision: 2,
    },
    {
      key: 'systolic_bp_mmhg',
      label: 'Systolic Arterial Pressure',
      unit: 'mmHg',
      base: previousSnapshot.systolic_bp_mmhg,
      final: currentSnapshot.systolic_bp_mmhg,
      precision: 1,
      optional: true,
    },
    {
      key: 'diastolic_bp_mmhg',
      label: 'Diastolic Arterial Pressure',
      unit: 'mmHg',
      base: previousSnapshot.diastolic_bp_mmhg,
      final: currentSnapshot.diastolic_bp_mmhg,
      precision: 1,
      optional: true,
    },
    {
      key: 'oxygen_saturation',
      label: 'Oxygen Saturation (SpO₂)',
      unit: '%',
      base: typeof previousSnapshot.oxygen_saturation === 'number'
        ? (previousSnapshot.oxygen_saturation <= 1.0 ? previousSnapshot.oxygen_saturation * 100 : previousSnapshot.oxygen_saturation)
        : null,
      final: typeof currentSnapshot.oxygen_saturation === 'number'
        ? (currentSnapshot.oxygen_saturation <= 1.0 ? currentSnapshot.oxygen_saturation * 100 : currentSnapshot.oxygen_saturation)
        : null,
      precision: 1,
      optional: true,
    },
    {
      key: 'respiration_rate_bpm',
      label: 'Respiration Rate',
      unit: 'breaths/min',
      base: previousSnapshot.respiration_rate_bpm,
      final: currentSnapshot.respiration_rate_bpm,
      precision: 1,
      optional: true,
    },
  ];

  return metrics
    .filter((m) => !m.optional || (typeof m.base === 'number' && typeof m.final === 'number'))
    .map((m) => {
      const delta = calculateDelta(m.final, m.base);
      return {
        ...m,
        delta,
        formattedBase: typeof m.base === 'number' ? m.base.toFixed(m.precision) : '—',
        formattedFinal: typeof m.final === 'number' ? m.final.toFixed(m.precision) : '—',
        formattedDelta: delta !== null
          ? `${delta > 0 ? '+' : ''}${delta.toFixed(m.precision)}`
          : '—',
      };
    });
}

/**
 * Compute signed absolute physiological changes between each pair of consecutive milestones.
 * Delta = x_current - x_previous.
 * Returns an array of comparison objects with actual timestamps, descriptions, and metric deltas.
 */
export function computeConsecutiveMilestoneDeltas(milestones = []) {
  if (!Array.isArray(milestones) || milestones.length < 2) return [];

  const comparisons = [];
  for (let i = 0; i < milestones.length - 1; i++) {
    const prev = milestones[i];
    const curr = milestones[i + 1];

    const deltas = computePhysiologicalDeltas(prev.snapshot, curr.snapshot);
    if (!deltas) continue;

    const fromTime = prev.actual_time_s ?? prev.target_time_s;
    const toTime = curr.actual_time_s ?? curr.target_time_s;
    const fromLabel = prev.description || prev.phase;
    const toLabel = curr.description || curr.phase;

    comparisons.push({
      stepIndex: i + 1,
      fromIndex: i,
      toIndex: i + 1,
      fromTime,
      toTime,
      fromLabel,
      toLabel,
      label: `${fromLabel} (${fromTime}s) → ${toLabel} (${toTime}s)`,
      deltas,
    });
  }

  return comparisons;
}

/**
 * Execute sequential saline simulation against the backend.
 * Handles timeouts and formats HTTP errors.
 * Accepts either { actions: [...] } or { volumeMl: 100 } for backward compatibility.
 */
export async function executeSequentialWhatIf({
  volumeMl,
  actions,
  baselineObservationS = DEFAULT_BASELINE_OBSERVATION_S,
  postObservationS = DEFAULT_POST_OBSERVATION_S,
  signal,
}) {
  const isMultiAction = Array.isArray(actions) && actions.length > 0;
  const configuredActions = isMultiAction
    ? actions
    : [
        {
          type: 'saline',
          volume_ml: Number(volumeMl ?? 100),
          start_time_s: Number(baselineObservationS),
        },
      ];

  const validation = validateSimulationInputs(configuredActions, baselineObservationS, postObservationS);
  if (!validation.valid) {
    throw new Error(validation.error);
  }

  const payload = buildSequentialPayload(configuredActions, baselineObservationS, postObservationS);

  let response;
  try {
    response = await fetch(`${API_BASE}/biogears/whatif/sequential`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('Simulation request was cancelled.');
    }
    throw new Error(
      `Network error: Unable to connect to backend server at ${API_BASE}. ` +
      'Please verify that the FastAPI backend server is running.'
    );
  }

  if (!response.ok) {
    let errorDetail = '';
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || JSON.stringify(errorJson);
    } catch {
      errorDetail = await response.text();
    }

    if (response.status === 422) {
      throw new Error(`Validation Error (HTTP 422): ${errorDetail}`);
    } else if (response.status === 503) {
      throw new Error(`BioGears Engine Unavailable (HTTP 503): ${errorDetail}`);
    } else if (response.status === 504) {
      throw new Error(`Simulation Timeout (HTTP 504): ${errorDetail}`);
    } else {
      throw new Error(`Simulation Request Failed (HTTP ${response.status}): ${errorDetail || response.statusText}`);
    }
  }

  const result = await response.json();

  // Validate response contract integrity
  if (!result || !result.metadata || !result.baseline_snapshot || !result.final_snapshot) {
    throw new Error('Malformed API response: missing required metadata or snapshot objects.');
  }

  return result;
}

// ============================================================================
// STEP H: Scenario Comparison Workspace Services & Trajectory Alignment
// ============================================================================

export const COMPARISON_PALETTE = [
  {
    id: 'strategy_1',
    color: '#10b981', // Emerald
    lightColor: '#34d399',
    bg: 'rgba(16, 185, 129, 0.1)',
    border: 'rgba(16, 185, 129, 0.4)',
    name: 'Strategy A',
  },
  {
    id: 'strategy_2',
    color: '#0284c7', // Sky / Cyan
    lightColor: '#38bdf8',
    bg: 'rgba(2, 132, 199, 0.1)',
    border: 'rgba(2, 132, 199, 0.4)',
    name: 'Strategy B',
  },
  {
    id: 'strategy_3',
    color: '#8b5cf6', // Violet
    lightColor: '#a78bfa',
    bg: 'rgba(139, 92, 246, 0.1)',
    border: 'rgba(139, 92, 246, 0.4)',
    name: 'Strategy C',
  },
];

/**
 * Aligns continuous physiological trajectories from multiple scenarios onto a shared timeline.
 * 
 * Rules:
 *   1. Extracts the union of all discrete simulated timestamps across all scenarios.
 *   2. For each scenario and timestamp, locates the matching measurement within tolerance.
 *   3. Does NOT extrapolate or invent values beyond each scenario's actual observed range.
 *      Periods outside a scenario's simulation duration receive null values so lines end cleanly.
 *   4. Identifies the strictly common time interval [minCommonStart, minCommonEnd] where all scenarios overlap.
 */
export function alignComparisonTrajectories(scenarios = [], toleranceSeconds = 2.6) {
  if (!Array.isArray(scenarios) || scenarios.length === 0) {
    return { alignedPoints: [], commonInterval: [0, 0], maxDuration: 0, scenariosMeta: [] };
  }

  const validScenarios = scenarios.filter(
    (s) => s && s.result?.trajectory?.data_points && Array.isArray(s.result.trajectory.data_points) && s.result.trajectory.data_points.length > 0
  );

  if (validScenarios.length === 0) {
    return { alignedPoints: [], commonInterval: [0, 0], maxDuration: 0, scenariosMeta: [] };
  }

  const scenariosMeta = validScenarios.map((s, idx) => {
    const dps = s.result.trajectory.data_points;
    const minTime = Number(dps[0]?.time_s ?? 0);
    const maxTime = Number(dps[dps.length - 1]?.time_s ?? 0);
    const color = s.color || COMPARISON_PALETTE[idx % COMPARISON_PALETTE.length].color;
    return {
      id: s.id || `scenario_${idx + 1}`,
      name: s.name || s.result.metadata?.scenario_name || `Strategy ${idx + 1}`,
      color,
      minTime,
      maxTime,
      dataPoints: dps,
      actions: (s.actions && s.actions.length > 0) ? s.actions : (s.result.metadata?.actions || []),
    };
  });

  const commonMin = Math.max(...scenariosMeta.map((m) => m.minTime));
  const commonMax = Math.min(...scenariosMeta.map((m) => m.maxTime));
  const overallMax = Math.max(...scenariosMeta.map((m) => m.maxTime));

  // Extract intervention markers for timeline visualization
  const interventionMarkers = [];
  scenariosMeta.forEach((meta) => {
    (meta.actions || []).forEach((act, actIdx) => {
      const startTime = Number(act.start_time_s ?? act.startTimeS ?? 0);
      const isNorepi = act.type === 'norepinephrine' || act.substance === 'Norepinephrine';
      const vol = act.bag_volume_ml ?? act.volume_ml;
      const label = isNorepi ? 'Norepi (14µg/mL)' : `${vol || 100}mL Saline`;
      interventionMarkers.push({
        scenarioId: meta.id,
        scenarioName: meta.name,
        color: meta.color,
        time_s: startTime,
        type: isNorepi ? 'norepinephrine' : 'saline',
        label: `${meta.name}: ${label}`,
        actionIndex: actIdx,
      });
    });
  });

  const timeSet = new Set();
  scenariosMeta.forEach((m) => {
    m.dataPoints.forEach((pt) => {
      if (typeof pt.time_s === 'number' && Number.isFinite(pt.time_s)) {
        timeSet.add(Math.round(pt.time_s * 100) / 100);
      }
    });
  });

  const sortedTimes = Array.from(timeSet).sort((a, b) => a - b);

  const alignedPoints = sortedTimes.map((t) => {
    const row = {
      time_s: t,
      isCommonInterval: t >= commonMin && t <= commonMax,
    };

    scenariosMeta.forEach((meta) => {
      // If outside this scenario's actual observed range, do NOT interpolate or extrapolate
      if (t < meta.minTime - 0.05 || t > meta.maxTime + 0.05) {
        row[`hr_${meta.id}`] = null;
        row[`map_${meta.id}`] = null;
        row[`bv_${meta.id}`] = null;
        row[`co_${meta.id}`] = null;
        return;
      }

      let closestPt = null;
      let minDiff = Infinity;

      for (let i = 0; i < meta.dataPoints.length; i++) {
        const pt = meta.dataPoints[i];
        const diff = Math.abs(pt.time_s - t);
        if (diff < minDiff) {
          minDiff = diff;
          closestPt = pt;
        } else if (diff > minDiff) {
          break;
        }
      }

      if (closestPt && minDiff <= toleranceSeconds) {
        row[`hr_${meta.id}`] = typeof closestPt.heart_rate_bpm === 'number' ? closestPt.heart_rate_bpm : null;
        row[`map_${meta.id}`] = typeof closestPt.mean_arterial_pressure_mmhg === 'number' ? closestPt.mean_arterial_pressure_mmhg : null;
        row[`bv_${meta.id}`] = typeof closestPt.blood_volume_l === 'number' ? closestPt.blood_volume_l : null;
        row[`co_${meta.id}`] = typeof closestPt.cardiac_output_l_min === 'number' ? closestPt.cardiac_output_l_min : null;
      } else {
        row[`hr_${meta.id}`] = null;
        row[`map_${meta.id}`] = null;
        row[`bv_${meta.id}`] = null;
        row[`co_${meta.id}`] = null;
      }
    });

    return row;
  });

  return {
    alignedPoints,
    commonInterval: [commonMin, commonMax],
    maxDuration: overallMax,
    scenariosMeta,
    interventionMarkers,
  };
}

/**
 * Builds the Scenario Summary Table comparing configuration, execution metadata,
 * initial/final observations, and signed absolute deltas (Δ = final - initial)
 * for all submitted comparison scenarios.
 */
export function computeScenarioSummaryTable(scenarios = []) {
  if (!Array.isArray(scenarios)) return [];

  return scenarios.map((s, idx) => {
    const meta = s.result?.metadata;
    const baseSnap = s.result?.baseline_snapshot;
    const finalSnap = s.result?.final_snapshot;

    const name = s.name || meta?.scenario_name || `Strategy ${idx + 1}`;
    const status = s.status || (s.result ? 'completed' : (s.error ? 'error' : 'idle'));
    const isCached = meta?.is_cached ?? (meta?.source === 'cache');
    const source = meta?.source || (isCached ? 'cache' : 'live');
    const wallClock = typeof meta?.wall_clock_seconds === 'number'
      ? meta.wall_clock_seconds
      : (typeof s.wallClockSeconds === 'number' ? s.wallClockSeconds : null);

    const actions = (s.actions && s.actions.length > 0) ? s.actions : (meta?.actions || []);
    const numInterventions = actions.length;
    const totalVolumeMl = actions.reduce(
      (sum, a) => sum + (a.type === 'norepinephrine' || a.substance === 'Norepinephrine' ? 0 : Number(a.bag_volume_ml ?? a.volume_ml ?? 0)),
      0
    );
    const hasNorepinephrine = actions.some((a) => a.type === 'norepinephrine' || a.substance === 'Norepinephrine');
    const simulatedDurationS = meta?.simulated_duration_s ?? (s.totalSimDuration || null);

    const baseHR = typeof baseSnap?.heart_rate_bpm === 'number' ? baseSnap.heart_rate_bpm : null;
    const finalHR = typeof finalSnap?.heart_rate_bpm === 'number' ? finalSnap.heart_rate_bpm : null;
    const deltaHR = (finalHR !== null && baseHR !== null) ? finalHR - baseHR : null;

    const baseMAP = typeof baseSnap?.mean_arterial_pressure_mmhg === 'number' ? baseSnap.mean_arterial_pressure_mmhg : null;
    const finalMAP = typeof finalSnap?.mean_arterial_pressure_mmhg === 'number' ? finalSnap.mean_arterial_pressure_mmhg : null;
    const deltaMAP = (finalMAP !== null && baseMAP !== null) ? finalMAP - baseMAP : null;

    const baseBV = typeof baseSnap?.blood_volume_l === 'number' ? baseSnap.blood_volume_l : null;
    const finalBV = typeof finalSnap?.blood_volume_l === 'number' ? finalSnap.blood_volume_l : null;
    const deltaBV = (finalBV !== null && baseBV !== null) ? finalBV - baseBV : null;

    const baseCO = typeof baseSnap?.cardiac_output_l_min === 'number' ? baseSnap.cardiac_output_l_min : null;
    const finalCO = typeof finalSnap?.cardiac_output_l_min === 'number' ? finalSnap.cardiac_output_l_min : null;
    const deltaCO = (finalCO !== null && baseCO !== null) ? finalCO - baseCO : null;

    return {
      id: s.id || `strategy_${idx + 1}`,
      name,
      status,
      error: s.error || null,
      isCached,
      source,
      wallClockSeconds: wallClock,
      simulatedDurationS,
      numInterventions,
      totalVolumeMl,
      hasNorepinephrine,
      actions,
      color: s.color || COMPARISON_PALETTE[idx % COMPARISON_PALETTE.length].color,
      vitals: {
        hr: {
          initial: baseHR,
          final: finalHR,
          delta: deltaHR,
          unit: 'bpm',
          precision: 1,
        },
        map: {
          initial: baseMAP,
          final: finalMAP,
          delta: deltaMAP,
          unit: 'mmHg',
          precision: 1,
        },
        bv: {
          initial: baseBV,
          final: finalBV,
          delta: deltaBV,
          unit: 'L',
          precision: 2,
        },
        co: {
          initial: baseCO,
          final: finalCO,
          delta: deltaCO,
          unit: 'L/min',
          precision: 2,
        },
      },
    };
  });
}

/**
 * Calculates relative differences (Δ_ref = Scenario_vital - Reference_vital)
 * against a user-selected reference scenario.
 */
export function computeReferenceDifferences(summaryRows = [], referenceId) {
  if (!Array.isArray(summaryRows) || summaryRows.length === 0 || !referenceId) {
    return null;
  }

  const refRow = summaryRows.find((r) => r.id === referenceId && r.status === 'completed');
  if (!refRow) return null;

  const comparisonRows = summaryRows.map((row) => {
    const isReference = row.id === referenceId;
    if (row.status !== 'completed') {
      return {
        id: row.id,
        name: row.name,
        isReference,
        status: row.status,
        differences: null,
      };
    }

    if (isReference) {
      return {
        id: row.id,
        name: row.name,
        isReference: true,
        status: 'completed',
        differences: {
          hr: { initialDiff: 0, finalDiff: 0, deltaDiff: 0, unit: 'bpm', precision: 1 },
          map: { initialDiff: 0, finalDiff: 0, deltaDiff: 0, unit: 'mmHg', precision: 1 },
          bv: { initialDiff: 0, finalDiff: 0, deltaDiff: 0, unit: 'L', precision: 2 },
          co: { initialDiff: 0, finalDiff: 0, deltaDiff: 0, unit: 'L/min', precision: 2 },
        },
      };
    }

    const diffFor = (key) => {
      const rowV = row.vitals[key];
      const refV = refRow.vitals[key];
      if (!rowV || !refV) return null;

      const initialDiff = (rowV.initial !== null && refV.initial !== null)
        ? rowV.initial - refV.initial
        : null;
      const finalDiff = (rowV.final !== null && refV.final !== null)
        ? rowV.final - refV.final
        : null;
      const deltaDiff = (rowV.delta !== null && refV.delta !== null)
        ? rowV.delta - refV.delta
        : null;

      return {
        initialDiff,
        finalDiff,
        deltaDiff,
        unit: rowV.unit,
        precision: rowV.precision,
      };
    };

    return {
      id: row.id,
      name: row.name,
      isReference: false,
      status: 'completed',
      differences: {
        hr: diffFor('hr'),
        map: diffFor('map'),
        bv: diffFor('bv'),
        co: diffFor('co'),
      },
    };
  });

  return {
    referenceScenario: refRow,
    comparisonRows,
  };
}

/**
 * Step J: Standard default experimental comparison strategies
 * Strategy A: Fluid-only reference (100mL Saline @ 60s)
 * Strategy B: Sequential fluid & vasopressor (100mL Saline @ 60s -> Norepinephrine @ 210s)
 * Strategy C: Repeated fluid boluses & vasopressor (100mL Saline @ 60s -> 100mL Saline @ 210s -> Norepinephrine @ 360s)
 */
export const DEFAULT_STEP_J_STRATEGIES = [
  {
    id: 'strategy_1',
    name: 'Strategy A: Fluid-Only Reference',
    color: COMPARISON_PALETTE[0].color,
    baseline_s: 60,
    post_s: 60,
    actions: [
      { id: 'act_a1', type: 'saline', volume_ml: 100, start_time_s: 60 },
    ],
  },
  {
    id: 'strategy_2',
    name: 'Strategy B: Sequential Fluid & Vasopressor',
    color: COMPARISON_PALETTE[1].color,
    baseline_s: 60,
    post_s: 60,
    actions: [
      { id: 'act_b1', type: 'saline', volume_ml: 100, start_time_s: 60 },
      {
        id: 'act_b2',
        type: 'norepinephrine',
        concentration_ug_ml: 14.0,
        rate_ml_per_min: 1.0,
        duration_s: 120,
        start_time_s: 210,
      },
    ],
  },
  {
    id: 'strategy_3',
    name: 'Strategy C: Repeated Fluid & Vasopressor',
    color: COMPARISON_PALETTE[2].color,
    baseline_s: 60,
    post_s: 60,
    actions: [
      { id: 'act_c1', type: 'saline', volume_ml: 100, start_time_s: 60 },
      { id: 'act_c2', type: 'saline', volume_ml: 100, start_time_s: 210 },
      {
        id: 'act_c3',
        type: 'norepinephrine',
        concentration_ug_ml: 14.0,
        rate_ml_per_min: 1.0,
        duration_s: 120,
        start_time_s: 360,
      },
    ],
  },
];

/**
 * Calculates total saline volume administered across all actions in a strategy
 */
export function calculateStrategyTotalSaline(actions = []) {
  if (!Array.isArray(actions)) return 0;
  return actions.reduce((sum, act) => {
    const isSaline = act.type === 'saline' || !act.type;
    return sum + (isSaline ? Number(act.volume_ml ?? act.bag_volume_ml ?? 0) : 0);
  }, 0);
}

/**
 * Formats a concise summary of a strategy's action schedule
 */
export function formatStrategyActionSummary(actions = []) {
  if (!Array.isArray(actions) || actions.length === 0) return 'No interventions';
  return actions
    .map((act) => {
      const isNorepi = act.type === 'norepinephrine' || act.substance === 'Norepinephrine';
      const start = act.start_time_s ?? act.startTimeS ?? 0;
      if (isNorepi) {
        return `Norepi (14µg/mL) @ ${start}s`;
      }
      const vol = act.volume_ml ?? act.bag_volume_ml ?? 100;
      return `${vol}mL Saline @ ${start}s`;
    })
    .join(' → ');
}
