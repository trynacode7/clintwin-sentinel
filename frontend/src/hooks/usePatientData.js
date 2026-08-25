import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

/**
 * usePatientData — Phase K frontend wiring.
 *
 * REPLACES the previous version of this file entirely. The old
 * implementation fabricated every value shown on this dashboard —
 * vitals, risk scores, gradient/"SHAP" values, attention weights,
 * latent trajectories, patient lists — client-side, via a seeded
 * pseudo-random generator, and NEVER called the backend at all (see
 * PHASE_B_REPORT.md / PHASE_D_REPORT.md for the audit that found this).
 *
 * This version calls the real backend endpoints built in Phases B-D
 * (backend/api.py, backed by backend/inference_service.py — real model
 * inference on real synthetic ICU stays) and derives every value shown
 * from what the backend actually returns. Nothing here invents a
 * clinical value. The only "derived, not fetched" value is `risk_trend`
 * (increasing/decreasing/stable), computed directly from the last two
 * REAL risk scores in the real trajectory — see deriveTrend() below.
 *
 * If the backend is unreachable or returns an error, this hook surfaces
 * that as a real error state — it does NOT fall back to fabricated data.
 */

async function apiFetch(path) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`);
  } catch (networkErr) {
    throw new Error(
      `Could not reach the backend at ${API_BASE}${path}. Is \`uvicorn backend.api:app\` running? (${networkErr.message})`
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(`${res.status} ${path}: ${detail}`);
  }
  return res.json();
}

function hoursObserved(rosterPatient) {
  const first = new Date(rosterPatient.first_charttime);
  const last = new Date(rosterPatient.last_charttime);
  const diffH = (last.getTime() - first.getTime()) / (1000 * 60 * 60);
  return Number.isFinite(diffH) ? diffH : 0;
}

/**
 * Real, derived (not fabricated) trend: compares the last two REAL
 * risk scores in the trajectory. A small dead-zone (±0.02) avoids
 * flapping between "increasing"/"decreasing" on floating-point noise
 * between two nearly-identical real predictions.
 */
function deriveTrend(trajectory) {
  if (!trajectory || trajectory.length < 2) return 'stable';
  const delta =
    trajectory[trajectory.length - 1].risk_score - trajectory[trajectory.length - 2].risk_score;
  if (delta > 0.02) return 'increasing';
  if (delta < -0.02) return 'decreasing';
  return 'stable';
}

async function fetchRoster() {
  const body = await apiFetch('/api/roster');
  return body.patients
    .slice()
    .sort((a, b) => (b.latest_risk_score ?? 0) - (a.latest_risk_score ?? 0))
    .map((p) => ({
      id: p.stay_id,
      name: `Patient ${p.stay_id}`,
      status: p.latest_status || 'Unknown',
      risk_score: p.latest_risk_score ?? 0,
      hours_in_icu: hoursObserved(p),
    }));
}

async function fetchPatientDetail(stayId) {
  const [current, explanation, trajectoryBody, twin] = await Promise.all([
    apiFetch(`/api/patients/${stayId}`),
    apiFetch(`/api/patients/${stayId}/explanation`),
    apiFetch(`/api/patients/${stayId}/trajectory`),
    apiFetch(`/api/patients/${stayId}/twin-state`),
  ]);

  const trajectory = trajectoryBody.trajectory;

  // Real vitals over real time, straight from the trajectory's per-window
  // observed_vitals (see inference_service.py) — same values the model
  // actually saw, in original clinical units (scaler.inverse_transform'd
  // server-side), never regenerated here.
  const vitals = {
    timestamps: trajectory.map((p) => p.window_end_time),
    heart_rate: trajectory.map((p) => p.observed_vitals.heart_rate),
    map: trajectory.map((p) => p.observed_vitals.map),
    spo2: trajectory.map((p) => p.observed_vitals.spo2),
    resp_rate: trajectory.map((p) => p.observed_vitals.resp_rate),
    lactate: trajectory.map((p) => p.observed_vitals.lactate),
  };

  // Real latent-space trajectory (z_2d per window) — this is what makes
  // the Digital Twin panel show an actual patient path through the
  // model's learned state space, not an invented spiral.
  const latentTrajectory = trajectory.map((p) => ({
    t_idx: p.window_idx,
    z_2d: p.z_2d,
    risk_score: p.risk_score,
  }));

  // Real, signed, directional local attribution (gradient×input — see
  // PHASE_D_REPORT.md for why this is never called "SHAP"). Using the
  // LOCAL (per-window) values here, not the global/magnitude-only ones,
  // because the panel needs direction (risk-increasing vs protective).
  const attribution = (explanation.local_gradient_attribution || []).map((item) => ({
    feature: item.clinical_name || item.feature,
    value: item.attribution_value,
    importance: Math.abs(item.attribution_value),
    direction: item.direction === 'risk_increasing' ? 'risk' : 'protective',
  }));

  return {
    current_patient: {
      id: stayId,
      risk_score: current.prediction.risk_score,
      status: current.prediction.status,
      risk_trend: deriveTrend(trajectory),
    },
    patient_meta: current.patient,
    vitals,
    attribution,
    attribution_method: explanation.explanation_method,
    // LSTM has no self-attention mechanism (it's a BiLSTM, not a
    // Transformer) — the real backend correctly returns null here, and
    // this stays null rather than being papered over with invented
    // weights. See AttentionHeatmap's honest empty-state message.
    attention: null,
    latent_trajectory: latentTrajectory,
    twin_disclaimer: twin.disclaimer,
    // Pre-BioGears redesign: the backend now explicitly separates the
    // twin's state into OBSERVED (real vitals) / PREDICTED (this
    // model's real risk + latent state) / SIMULATED (BioGears forward
    // physiology — not yet available). Passed through verbatim so the
    // UI can show the same three-way distinction instead of implying
    // z_t is a physiological simulation. See digital_twin.py.
    twin_state: { observed: twin.observed, predicted: twin.predicted, simulated: twin.simulated },
    explanation: explanation.explanation_text,
    model: {
      type: current.prediction.model_type,
      epoch: current.prediction.model_epoch,
      checkpoint: current.prediction.model_checkpoint,
    },
  };
}

/**
 * POST a validated What-If scenario for the given stay and return the
 * real response (real baseline + honest "simulation not yet available"
 * boundary — see backend/whatif.py). Throws with the backend's real
 * validation message on 400 (e.g. an out-of-range value) rather than
 * swallowing it.
 */
async function postWhatIf(stayId, interventions, engine = 'biogears') {
  const res = await fetch(`${API_BASE}/api/patients/${stayId}/whatif`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ interventions, engine }),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(body?.detail || `${res.status} ${res.statusText}`);
  }
  return body;
}

export function usePatientData() {
  const [patients, setPatients] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [intervenableFeatures, setIntervenableFeatures] = useState([]);
  const requestIdRef = useRef(0);

  // Fetched once — the real, single source of truth for what a What-If
  // scenario can ask about (backend/whatif.py::INTERVENABLE_FEATURES).
  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/whatif/interventions')
      .then((body) => {
        if (!cancelled) setIntervenableFeatures(body.interventions || []);
      })
      .catch(() => {
        // Non-fatal: the What-If panel just shows its own error state
        // if this list never arrives; the rest of the dashboard doesn't
        // depend on it.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const loadDetail = useCallback(async (stayId) => {
    if (stayId == null) return;
    const requestId = ++requestIdRef.current;
    setDetailLoading(true);
    setError(null);
    try {
      const detail = await fetchPatientDetail(stayId);
      // Ignore stale responses if the user switched patients again
      // before this request resolved.
      if (requestId === requestIdRef.current) {
        setData(detail);
      }
    } catch (err) {
      if (requestId === requestIdRef.current) {
        setError(err.message);
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setDetailLoading(false);
      }
    }
  }, []);

  // Initial load: real roster, then the real detail for the top patient.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const roster = await fetchRoster();
        if (cancelled) return;
        setPatients(roster);
        if (roster.length > 0) {
          setSelectedPatient(roster[0].id);
        } else {
          setError('The backend returned an empty roster — no patients to display.');
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // When the selected patient changes, fetch real detail for that patient.
  useEffect(() => {
    if (selectedPatient != null) {
      loadDetail(selectedPatient);
    }
  }, [selectedPatient, loadDetail]);

  return {
    data: data ? { ...data, patients } : patients ? { patients } : null,
    loading,
    detailLoading,
    error,
    selectedPatient,
    setSelectedPatient,
    refresh: () => loadDetail(selectedPatient),
    intervenableFeatures,
    runWhatIf: (interventions, engine) => {
      if (selectedPatient == null) {
        return Promise.reject(new Error('No patient selected.'));
      }
      return postWhatIf(selectedPatient, interventions, engine);
    },
  };
}
