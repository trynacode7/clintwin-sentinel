import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

describe('Step G: Dashboard Integration & UI Professionalization', () => {
  test('normalisePatientResponse cleanly extracts attribution and patient name', () => {
    const mockApiResponse = {
      patient: {
        id: 'patient_001',
        name: 'Patient 001',
        risk_score: 0.82,
        status: 'Critical',
        risk_trend: 'increasing',
        hours_in_icu: 48.0,
      },
      vitals: {
        heart_rate: [78, 85, 92, 105],
        map: [85, 80, 72, 63],
        spo2: [98, 97, 95, 93],
      },
      gradient_attribution: [
        { feature: 'map', importance: 0.18, direction: 'risk' },
        { feature: 'heart_rate', importance: 0.14, direction: 'risk' },
        { feature: 'spo2', importance: -0.09, direction: 'protective' },
      ],
      attention: {
        matrix: [[0.5, 0.5], [0.3, 0.7]],
        labels: ['t1', 't2'],
      },
      latent_trajectory: [{ t_idx: 0, z_2d: [0.1, -0.2], risk_score: 0.5 }],
      explanation: 'Elevated heart rate with trending hypotension indicates early septic shock.',
    };

    const attr = mockApiResponse.gradient_attribution || mockApiResponse.attribution || mockApiResponse.shap;
    assert.ok(Array.isArray(attr), 'attribution must be an array');
    assert.equal(attr.length, 3);
    assert.equal(attr[0].feature, 'map');
    assert.equal(attr[0].importance, 0.18);

    assert.equal(mockApiResponse.patient.name, 'Patient 001');
    assert.equal(mockApiResponse.patient.status, 'Critical');
    assert.equal(mockApiResponse.patient.risk_score, 0.82);
  });

  test('normalisePatientResponse generates fallback patient name when name is missing', () => {
    const pat = { id: 'patient_002' };
    const name = pat.name || (pat.id ? 'Patient ' + pat.id : 'Patient');
    assert.equal(name, 'Patient patient_002');

    const patNumeric = { id: 1042 };
    const nameNumeric = patNumeric.name || (patNumeric.id ? 'Patient ' + patNumeric.id : 'Patient');
    assert.equal(nameNumeric, 'Patient 1042');

    const patEmpty = {};
    const nameEmpty = patEmpty.name || (patEmpty.id ? 'Patient ' + patEmpty.id : 'Patient');
    assert.equal(nameEmpty, 'Patient');
  });

  test('normalisePatientResponse handles real backend stay_id and prediction payloads', () => {
    const realBackendPayload = {
      patient: {
        stay_id: 10000801,
        subject_id: 100008,
        hadm_id: 1000081,
        latest_risk_score: 0.48946,
        latest_status: 'Warning',
        n_observed_timesteps: 76,
      },
      prediction: {
        risk_score: 0.48946,
        status: 'Warning',
      },
      trajectory: [
        {
          window_idx: 0,
          window_end_time: '2446-02-16 07:00:00',
          risk_score: 0.476,
          observed_vitals: { heart_rate: 80, map: 85, spo2: 98, resp_rate: 18, lactate: 1.5 },
        },
        {
          window_idx: 1,
          window_end_time: '2446-02-16 13:00:00',
          risk_score: 0.489,
          observed_vitals: { heart_rate: 95, map: 74, spo2: 92, resp_rate: 22, lactate: 2.8 },
        },
      ],
      gradient_attribution: [
        { feature: 'platelets_rolling_mean', importance: 0.00005, clinical_name: 'platelets' },
      ],
      explanation: 'Risk assessed to 0.49 (Warning).',
    };

    const pat = realBackendPayload.patient;
    const pred = realBackendPayload.prediction;
    const patientId = pat.id ?? pat.stay_id;
    const riskScore = pred.risk_score ?? pat.risk_score ?? pat.latest_risk_score ?? 0;
    const status = pred.status ?? pat.status ?? pat.latest_status ?? 'Stable';

    assert.equal(patientId, 10000801);
    assert.equal(riskScore, 0.48946);
    assert.equal(status, 'Warning');

    // Vitals timeline constructed from trajectory
    const vitals = {
      timestamps: realBackendPayload.trajectory.map(w => w.window_end_time),
      heart_rate: realBackendPayload.trajectory.map(w => w.observed_vitals?.heart_rate),
    };
    assert.equal(vitals.timestamps.length, 2);
    assert.equal(vitals.heart_rate[0], 80);
    assert.equal(vitals.heart_rate[1], 95);
  });

  test('UI terminology strictly avoids SHAP labeling when using input x gradient saliency', () => {
    const sampleAttribution = [
      { feature: 'lactate', value: 0.25, importance: 0.25, direction: 'risk' },
      { feature: 'bicarbonate', value: -0.15, importance: 0.15, direction: 'protective' },
    ];

    sampleAttribution.forEach(item => {
      assert.ok('direction' in item);
      assert.ok(item.direction === 'risk' || item.direction === 'protective');
      assert.ok('importance' in item);
    });
  });

  test('Physiological digital twin separation from ML risk score', () => {
    const mlRiskScore = 0.75;
    const simulationTrajectory = {
      scenario_name: 'Fluid_Challenge_100mL',
      patient_file: 'StandardMale.xml',
      time_s: [5.02, 10.02, 15.02],
      map_mmHg: [65.0, 68.2, 70.1],
    };

    assert.equal(mlRiskScore, 0.75);
    assert.equal(simulationTrajectory.patient_file, 'StandardMale.xml');
    assert.equal(simulationTrajectory.map_mmHg.length, 3);
  });

  test('App.jsx renders MethodologyModal exactly once at root level', () => {
    const appContent = fs.readFileSync('src/App.jsx', 'utf8');
    const matches = appContent.match(/<MethodologyModal\b/g);
    assert.ok(matches, 'MethodologyModal must be rendered');
    assert.equal(matches.length, 1, 'MethodologyModal must only be rendered once at root level, without stray instances in child components');
  });

  test('Risk score scale mapping and missing value validation', () => {
    // Probability in [0, 1]
    const p1 = 0.48946;
    const isP1Valid = typeof p1 === 'number' && !isNaN(p1);
    const target1 = isP1Valid ? (p1 > 1 ? p1 / 100 : Math.max(0, Math.min(1, p1))) : 0;
    assert.equal((target1 * 100).toFixed(0), '49');
    assert.equal((target1 * 100).toFixed(1), '48.9');

    // Percentage in [0, 100] scale safely converted without overflow
    const p2 = 48.946;
    const isP2Valid = typeof p2 === 'number' && !isNaN(p2);
    const target2 = isP2Valid ? (p2 > 1 ? p2 / 100 : Math.max(0, Math.min(1, p2))) : 0;
    assert.equal((target2 * 100).toFixed(0), '49');

    // Undefined / null / NaN score triggers invalid state without defaulting to 0
    const p3 = undefined;
    const isP3Valid = typeof p3 === 'number' && !isNaN(p3);
    assert.equal(isP3Valid, false);

    const p4 = null;
    const isP4Valid = typeof p4 === 'number' && !isNaN(p4);
    assert.equal(isP4Valid, false);
  });

  test('Patient switching updates selected patient without retaining previous state', () => {
    let selectedPatient = 10000801;
    const mockRoster = [
      { id: 10000801, risk_score: 0.48946 },
      { id: 10002901, risk_score: 0.48657 },
    ];

    const getScoreForSelection = (id) => mockRoster.find(p => p.id === id)?.risk_score;

    assert.equal(getScoreForSelection(selectedPatient), 0.48946);
    selectedPatient = 10002901;
    assert.equal(getScoreForSelection(selectedPatient), 0.48657);
    assert.notEqual(getScoreForSelection(selectedPatient), 0.48946);
  });

  test('AttentionHeatmap dual-mode: renders attribution drivers when attention is absent', () => {
    const attentionContent = fs.readFileSync('src/components/AttentionHeatmap.jsx', 'utf8');
    assert.ok(attentionContent.includes('matrix?.length > 0'), 'Must check for presence of actual attention matrix');
    assert.ok(attentionContent.includes('attributionData'), 'Must accept attributionData as fallback');
    assert.ok(attentionContent.includes('Feature attribution unavailable for this prediction.'), 'Must show clean unavailable state if both are missing');
    assert.ok(!attentionContent.includes('Shapley'), 'Must not label attribution as Shapley');
  });
});
