import { useState, useEffect, useRef, useCallback } from 'react';

export const API_BASE = import.meta?.env?.VITE_API_BASE || 'http://localhost:8000';
// Timeout for each fetch in milliseconds — fail fast so fallback kicks in quickly
const FETCH_TIMEOUT_MS = 4000;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
/** Check backend health endpoint */
export async function checkBackendHealth() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/health`, {}, 2500);
    return res && res.status === 'ok';
  } catch {
    try {
      const res = await fetchWithTimeout(`${API_BASE}/`, {}, 2500);
      return res && res.status === 'ok';
    } catch {
      return false;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────

/** Fetch with an AbortController-based timeout. */
async function fetchWithTimeout(url, options = {}, timeoutMs = FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.json();
  } catch (err) {
    clearTimeout(timer);
    throw err;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Local fallback demo generation
// Used when the FastAPI backend is unavailable.
// ─────────────────────────────────────────────────────────────────────────────

function seededRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

/**
 * Single source of truth for fallback patient data.
 * Mirrors the backend _PATIENT_PROFILES exactly so that the sidebar and
 * gauge show the same values whether the backend is up or down.
 */
const PATIENT_REGISTRY = [
  { id: 'patient_001', name: 'Patient 001', baseRisk: 0.86, status: 'Critical',  archetype: 'critical',   trend: 'increasing', hours: 18.4 },
  { id: 'patient_002', name: 'Patient 002', baseRisk: 0.89, status: 'Warning',   archetype: 'critical',   trend: 'increasing', hours: 94.1 },
  { id: 'patient_003', name: 'Patient 003', baseRisk: 0.40, status: 'Warning',   archetype: 'warning',    trend: 'increasing', hours: 14.2 },
  { id: 'patient_004', name: 'Patient 004', baseRisk: 0.07, status: 'Stable',    archetype: 'stable',     trend: 'stable',     hours: 48.7 },
  { id: 'patient_005', name: 'Patient 005', baseRisk: 0.69, status: 'Warning',   archetype: 'warning',    trend: 'increasing', hours: 67.3 },
  { id: 'patient_006', name: 'Patient 006', baseRisk: 0.72, status: 'Critical',  archetype: 'recovering', trend: 'decreasing', hours: 82.1 },
  { id: 'patient_007', name: 'Patient 007', baseRisk: 0.57, status: 'Warning',   archetype: 'warning',    trend: 'increasing', hours: 10.5 },
  { id: 'patient_008', name: 'Patient 008', baseRisk: 0.71, status: 'Critical',  archetype: 'recovering', trend: 'decreasing', hours: 110.2 },
  { id: 'patient_009', name: 'Patient 009', baseRisk: 0.27, status: 'Stable',    archetype: 'stable',     trend: 'stable',     hours: 87.6 },
  { id: 'patient_010', name: 'Patient 010', baseRisk: 0.15, status: 'Stable',    archetype: 'stable',     trend: 'stable',     hours: 55.3 },
];

function getPatientInfo(patientId) {
  return PATIENT_REGISTRY.find(p => p.id === patientId) || PATIENT_REGISTRY[0];
}

/** Generate deterministic demo data for a single patient (fallback). */
function generateFallbackPatientData(patientId) {
  const seed = hashString(patientId);
  const rand = seededRandom(seed);
  const patientInfo = getPatientInfo(patientId);
  const archetype = patientInfo.archetype;

  const profiles = {
    stable:     { hrBase: 72, mapBase: 85, spo2Base: 97, rrBase: 14, lacBase: 0.8, detStart: 999 },
    warning:    { hrBase: 88, mapBase: 72, spo2Base: 95, rrBase: 18, lacBase: 1.4, detStart: 50 },
    critical:   { hrBase: 78, mapBase: 82, spo2Base: 97, rrBase: 16, lacBase: 1.0, detStart: 40 },
    recovering: { hrBase: 90, mapBase: 68, spo2Base: 94, rrBase: 20, lacBase: 2.0, detStart: 25 },
  };
  const p = profiles[archetype];
  const n = 72;
  const timestamps = Array.from({ length: n }, (_, i) => `T-${n - i}`);

  const vitals = { timestamps, heart_rate: [], map: [], spo2: [], resp_rate: [], lactate: [] };
  for (let t = 0; t < n; t++) {
    const severity = t > p.detStart ? Math.min(1, (t - p.detStart) / (n - p.detStart)) : 0;
    vitals.heart_rate.push(+(p.hrBase + 40 * severity + (rand() - 0.5) * 6).toFixed(1));
    vitals.map.push(+(p.mapBase - 25 * severity + (rand() - 0.5) * 8).toFixed(1));
    vitals.spo2.push(+(p.spo2Base - 5 * severity + (rand() - 0.5) * 2).toFixed(1));
    vitals.resp_rate.push(+(p.rrBase + 10 * severity + (rand() - 0.5) * 4).toFixed(1));
    vitals.lactate.push(
      rand() > 0.55 ? +(p.lacBase + 3 * severity + (rand() - 0.5) * 0.6).toFixed(2) : null
    );
  }

  const shapTemplates = {
    stable: [
      { feature: 'heart_rate', value: -0.05 }, { feature: 'MAP', value: -0.12 },
      { feature: 'lactate', value: -0.03 }, { feature: 'SpO2', value: -0.08 },
      { feature: 'shock_index', value: -0.04 }, { feature: 'resp_rate', value: 0.02 },
      { feature: 'HR_delta', value: -0.01 }, { feature: 'MAP_delta', value: -0.02 },
      { feature: 'time_in_ICU', value: 0.03 }, { feature: 'WBC', value: 0.01 },
    ],
    warning: [
      { feature: 'heart_rate', value: 0.15 }, { feature: 'MAP', value: -0.18 },
      { feature: 'lactate', value: 0.12 }, { feature: 'shock_index', value: 0.14 },
      { feature: 'resp_rate', value: 0.10 }, { feature: 'SpO2', value: -0.06 },
      { feature: 'HR_delta', value: 0.09 }, { feature: 'MAP_delta', value: -0.11 },
      { feature: 'lactate_delta', value: 0.08 }, { feature: 'time_in_ICU', value: 0.05 },
    ],
    critical: [
      { feature: 'lactate', value: 0.31 }, { feature: 'heart_rate', value: 0.23 },
      { feature: 'MAP', value: -0.19 }, { feature: 'shock_index', value: 0.18 },
      { feature: 'HR_delta', value: 0.15 }, { feature: 'MAP_delta', value: -0.14 },
      { feature: 'resp_rate', value: 0.12 }, { feature: 'lactate_delta', value: 0.22 },
      { feature: 'SpO2', value: -0.08 }, { feature: 'HR_rolling_std', value: 0.09 },
    ],
    recovering: [
      { feature: 'lactate', value: 0.10 }, { feature: 'MAP', value: -0.08 },
      { feature: 'heart_rate', value: 0.06 }, { feature: 'SpO2', value: -0.12 },
      { feature: 'lactate_delta', value: -0.09 }, { feature: 'MAP_delta', value: 0.07 },
      { feature: 'shock_index', value: 0.05 }, { feature: 'resp_rate', value: 0.03 },
      { feature: 'HR_delta', value: -0.04 }, { feature: 'time_in_ICU', value: 0.06 },
    ],
  };
  const shap = (shapTemplates[archetype] || shapTemplates.stable).map(s => ({
    ...s,
    importance: Math.abs(s.value),
    direction: s.value > 0 ? 'risk' : 'protective',
  })).sort((a, b) => b.importance - a.importance);

  const attnSize = 18;
  const attnLabels = Array.from({ length: attnSize }, (_, i) => `T-${Math.round(n - (i * n / attnSize))}`);
  const attnMatrix = [];
  const attnRand = seededRandom(seed + 777);
  for (let i = 0; i < attnSize; i++) {
    const row = [];
    for (let j = 0; j < attnSize; j++) {
      const dist = Math.abs(i - j);
      const base = Math.exp(-dist * 0.3) * 0.5;
      const recency = (i / attnSize) * 0.3;
      row.push(+(base + recency + attnRand() * 0.1).toFixed(3));
    }
    const sum = row.reduce((a, b) => a + b, 0);
    attnMatrix.push(row.map(v => +(v / sum).toFixed(4)));
  }

  const trajectory = [];
  const trajRand = seededRandom(seed + 333);
  const trajAngle = trajRand() * Math.PI * 2;
  for (let t = 0; t < 50; t++) {
    const sev = t > 25 ? Math.max(0, (t - 25) / 25) : 0;
    const direction = archetype === 'stable' ? 0.1 : archetype === 'recovering' ? -0.3 : 1.0;
    trajectory.push({
      t_idx: t,
      z_2d: [
        Math.cos(trajAngle) * (-2 + 3 * sev * direction) + (trajRand() - 0.5) * 0.4,
        Math.sin(trajAngle) * (1 - 2 * sev * direction) + (trajRand() - 0.5) * 0.4,
      ],
      risk_score: Math.min(1, Math.max(0, patientInfo.baseRisk * (t / 49))),
    });
  }

  const explanations = {
    stable: `Patient is hemodynamically stable with a risk score of ${patientInfo.baseRisk.toFixed(2)}. Vitals are within normal limits. MAP and heart rate are well-controlled. No significant deterioration pattern detected.`,
    warning: `Risk assessed at ${patientInfo.baseRisk.toFixed(2)} (Warning) due to mildly elevated heart rate and trending decrease in MAP. Shock index is approaching a concerning threshold. Recommend increased monitoring frequency.`,
    critical: `Risk increased to ${patientInfo.baseRisk.toFixed(2)} (Critical) due to rising lactate (tissue hypoxia), declining MAP (hypotension), and sustained tachycardia. Elevated shock index indicates hemodynamic instability.`,
    recovering: `Risk declining to ${patientInfo.baseRisk.toFixed(2)} following intervention. Lactate is trending downward, MAP stabilizing. Heart rate remains mildly elevated but improving. Continue current treatment protocol.`,
  };

  return {
    patient: {
      id: patientId,
      name: patientInfo.name,
      risk_score: patientInfo.baseRisk,
      status: patientInfo.status,
      hours_in_icu: patientInfo.hours,
      risk_trend: patientInfo.trend,
    },
    vitals,
    shap,
    attribution: shap,
    attention: { matrix: attnMatrix, labels: attnLabels },
    latent_trajectory: trajectory,
    explanation: explanations[archetype] || explanations.stable,
  };
}

/** Fallback patient list (mirrors backend _PATIENT_PROFILES). */
function generateFallbackPatientList() {
  return PATIENT_REGISTRY.map(p => ({
    id: p.id,
    name: p.name,
    status: p.status,
    risk_score: p.baseRisk,
    hours_in_icu: p.hours,
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
// normalisePatientResponse
// Maps the /patient/{id} API response to the shape the dashboard expects.
// ─────────────────────────────────────────────────────────────────────────────
function normalisePatientResponse(apiResp) {
  const pat = apiResp.patient || {};
  const pred = apiResp.prediction || {};
  const attr = apiResp.gradient_attribution || apiResp.attribution || apiResp.shap;

  const patientId = pat.id ?? pat.stay_id;
  const riskScore = pred.risk_score ?? pat.risk_score ?? pat.latest_risk_score ?? 0;
  const status = pred.status ?? pat.status ?? pat.latest_status ?? 'Stable';

  let vitals = apiResp.vitals;
  if (!vitals && Array.isArray(apiResp.trajectory) && apiResp.trajectory.length > 0) {
    vitals = {
      timestamps: apiResp.trajectory.map(w => w.window_end_time || `w${w.window_idx}`),
      heart_rate: apiResp.trajectory.map(w => w.observed_vitals?.heart_rate),
      map: apiResp.trajectory.map(w => w.observed_vitals?.map),
      spo2: apiResp.trajectory.map(w => w.observed_vitals?.spo2),
      resp_rate: apiResp.trajectory.map(w => w.observed_vitals?.resp_rate),
      lactate: apiResp.trajectory.map(w => w.observed_vitals?.lactate),
    };
  } else if (!vitals && apiResp.observed_vitals) {
    const ov = apiResp.observed_vitals;
    vitals = {
      timestamps: ['Current'],
      heart_rate: [ov.heart_rate],
      map: [ov.map],
      spo2: [ov.spo2],
      resp_rate: [ov.resp_rate],
      lactate: [ov.lactate],
    };
  }

  return {
    current_patient: {
      id: patientId,
      name: pat.name || (patientId ? 'Patient ' + patientId : 'Patient'),
      risk_score: riskScore,
      status: status,
      risk_trend: pat.risk_trend || (riskScore >= 0.5 ? 'increasing' : 'stable'),
      hours_in_icu: pat.hours_in_icu ?? pat.n_observed_timesteps ?? 24,
    },
    vitals: vitals,
    shap: apiResp.shap || attr,
    attribution: attr,
    attention: apiResp.attention,
    latent_trajectory: apiResp.latent_trajectory || apiResp.trajectory,
    explanation: apiResp.explanation,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// usePatientData — main hook
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetches patient data from the FastAPI backend.
 *
 * Flow:
 *   1. On mount, GET /patients  → populates the sidebar patient list.
 *   2. On patient selection, GET /patient/{id}  → populates all panels.
 *   3. If the backend is unreachable, falls back to local demo generation
 *      and sets `demoMode = true` so the UI can display a clear label.
 *
 * Patient switching:
 *   Call setSelectedPatient(id) → triggers a new /patient/{id} fetch →
 *   all six panels re-render with the new patient's data.
 */
export function usePatientData() {
  const [patients, setPatients] = useState(null);       // sidebar list
  const [data, setData] = useState(null);               // current patient dashboard data
  const [loading, setLoading] = useState(true);
  const [patientLoading, setPatientLoading] = useState(false);
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(false);      // true = backend unavailable
  const [mode, setMode] = useState('demo');             // 'demo' vs 'real_ml'
  const [selectedPatient, setSelectedPatient] = useState('patient_001');

  // Track whether the backend was reachable on first attempt
  const backendAvailableRef = useRef(null); // null = unknown, true/false = checked

  // ── Step 1: Load patient list ──────────────────────────────────────────────
  const loadPatientList = useCallback(async () => {
    try {
      const json = await fetchWithTimeout(`${API_BASE}/patients`);
      const rawList = json.patients || [];
      const list = rawList.map(p => ({
        id: p.id ?? p.stay_id,
        stay_id: p.stay_id ?? p.id,
        name: p.name || `Patient ${p.stay_id ?? p.id}`,
        status: p.status || p.latest_status || 'Stable',
        risk_score: p.risk_score ?? p.latest_risk_score ?? 0,
        hours_in_icu: p.hours_in_icu ?? p.n_observed_timesteps ?? 24,
        risk_trend: p.risk_trend || (p.latest_risk_score >= 0.5 ? 'increasing' : 'stable'),
      }));
      setPatients(list);
      backendAvailableRef.current = true;
      setDemoMode(false);
      setError(null);
      return list;
    } catch (err) {
      console.warn('[ClinTwin] /patients unreachable, using fallback demo mode:', err.message);
      backendAvailableRef.current = false;
      setDemoMode(true);
      const fallback = generateFallbackPatientList();
      setPatients(fallback);
      return fallback;
    }
  }, []);

  // ── Step 2: Load a single patient ─────────────────────────────────────────
  const loadPatient = useCallback(async (patientId, isInitial = false, overrideMode = null) => {
    const activeMode = overrideMode || mode;
    if (isInitial) {
      setLoading(true);
    } else {
      setPatientLoading(true);
    }
    setError(null);

    // If we already know the backend is down, go straight to fallback
    if (backendAvailableRef.current === false) {
      const fallback = generateFallbackPatientData(patientId);
      const normalised = normalisePatientResponse(fallback);
      setData(prev => ({ ...normalised, mode: 'demo', is_fallback: true, patients: prev?.patients || generateFallbackPatientList() }));
      setLoading(false);
      setPatientLoading(false);
      return;
    }

    try {
      const apiResp = await fetchWithTimeout(`${API_BASE}/patient/${patientId}`);
      let normalised = normalisePatientResponse(apiResp);

      // In Dynamic ML Mode, invoke POST /predict with patient trajectory vitals
      if (activeMode === 'real_ml') {
        let patientData = [];
        if (Array.isArray(apiResp.trajectory) && apiResp.trajectory.length > 0) {
          patientData = apiResp.trajectory.map((w, idx) => {
            const ov = w.observed_vitals || {};
            return {
              charttime: w.window_end_time || new Date(Date.now() - (apiResp.trajectory.length - 1 - idx) * 3600000).toISOString(),
              heart_rate: ov.heart_rate != null ? Number(ov.heart_rate) : null,
              map: ov.map != null ? Number(ov.map) : null,
              resp_rate: ov.resp_rate != null ? Number(ov.resp_rate) : null,
              spo2: ov.spo2 != null ? Number(ov.spo2) : null,
              temp: ov.temp != null ? Number(ov.temp) : 37.0,
              lactate: ov.lactate != null ? Number(ov.lactate) : null,
              wbc: ov.wbc != null ? Number(ov.wbc) : 7.0,
            };
          });
        } else if (normalised.vitals?.timestamps?.length > 0) {
          const nv = normalised.vitals;
          patientData = nv.timestamps.map((t, idx) => ({
            charttime: typeof t === 'string' && t.length > 10 ? t : new Date(Date.now() - (nv.timestamps.length - 1 - idx) * 3600000).toISOString(),
            heart_rate: nv.heart_rate?.[idx] != null ? Number(nv.heart_rate[idx]) : null,
            map: nv.map?.[idx] != null ? Number(nv.map[idx]) : null,
            resp_rate: nv.resp_rate?.[idx] != null ? Number(nv.resp_rate[idx]) : null,
            spo2: nv.spo2?.[idx] != null ? Number(nv.spo2[idx]) : null,
            temp: 37.0,
            lactate: nv.lactate?.[idx] != null ? Number(nv.lactate[idx]) : null,
            wbc: 7.0,
          }));
        }

        try {
          const predictResp = await fetchWithTimeout(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              patient_data: patientData,
              patient_id: String(patientId),
              model_type: 'lstm',
            }),
          });

          // Map live gradient attribution / SHAP values
          const rawShap = predictResp.gradient_attribution || predictResp.attribution || predictResp.shap_values || [];
          const formattedShap = rawShap.map(item => ({
            feature: item.clinical_name || item.feature,
            value: item.importance,
            importance: Math.abs(item.importance),
            direction: item.importance >= 0 ? 'risk' : 'protective',
          })).sort((a, b) => b.importance - a.importance);

          normalised = {
            ...normalised,
            current_patient: {
              ...normalised.current_patient,
              risk_score: predictResp.risk_score,
              status: predictResp.status,
              risk_trend: predictResp.risk_score >= 0.5 ? 'increasing' : 'stable',
              model_type: 'BiLSTM (256-dim z_t)',
              is_real_ml: true,
            },
            shap: formattedShap.length > 0 ? formattedShap : normalised.shap,
            attribution: formattedShap.length > 0 ? formattedShap : (normalised.attribution || normalised.shap),
            explanation: predictResp.explanation || normalised.explanation,
            latent_trajectory: predictResp.latent_trajectory && predictResp.latent_trajectory.length > 0
              ? predictResp.latent_trajectory
              : normalised.latent_trajectory,
            mode: 'real_ml',
          };
        } catch (predictErr) {
          console.error('[ClinTwin] Dynamic ML /predict error:', predictErr.message);
          setError(`Dynamic ML inference failed: ${predictErr.message}. Displaying precomputed cohort baseline.`);
          normalised = {
            ...normalised,
            mode: 'demo',
            current_patient: {
              ...normalised.current_patient,
              is_real_ml: false,
            },
          };
        }
      } else {
        normalised.mode = 'demo';
      }

      backendAvailableRef.current = true;
      setDemoMode(false);
      setData(prev => ({ ...normalised, patients: prev?.patients || patients }));
    } catch (err) {
      console.warn(`[ClinTwin] /patient/${patientId} failed:`, err.message);
      const isHealthy = await checkBackendHealth();
      if (!isHealthy) {
        backendAvailableRef.current = false;
        setDemoMode(true);
      }
      setError(`Failed to retrieve patient: ${err.message}`);
      const fallback = generateFallbackPatientData(patientId);
      const normalised = normalisePatientResponse(fallback);
      setData(prev => ({ ...normalised, mode: 'demo', is_fallback: true, patients: prev?.patients || generateFallbackPatientList() }));
    } finally {
      setLoading(false);
      setPatientLoading(false);
    }
  }, [patients, mode]);

  // ── Initial mount: load list then first patient ────────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function init() {
      const list = await loadPatientList();
      if (cancelled) return;

      const firstId = list[0]?.id ?? list[0]?.stay_id ?? 'patient_001';
      setSelectedPatient(firstId);
      await loadPatient(firstId, true);
    }

    init();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Patient selection change ───────────────────────────────────────────────
  useEffect(() => {
    if (data === null && loading) return;
    loadPatient(selectedPatient, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPatient]);

  // ── Manual refresh ────────────────────────────────────────────────────────
  const refresh = useCallback(() => {
    loadPatient(selectedPatient, false);
  }, [selectedPatient, loadPatient]);

  // ── Switch mode ───────────────────────────────────────────────────────────
  const setDashboardMode = useCallback((newMode) => {
    setMode(newMode);
    loadPatient(selectedPatient, false, newMode);
  }, [selectedPatient, loadPatient]);

  // ── Compose final data object consumed by App.jsx ─────────────────────────
  const mergedData = data
    ? { ...data, patients: patients || data.patients || [] }
    : null;

  return {
    data: mergedData,
    loading,
    patientLoading,
    error,
    demoMode,
    mode,
    setMode: setDashboardMode,
    selectedPatient,
    setSelectedPatient,
    refresh,
  };
}
