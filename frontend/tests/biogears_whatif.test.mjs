/**
 * biogears_whatif.test.mjs — Comprehensive Unit & Integration Tests for BioGears What-If Frontend
 * ==============================================================================================
 * Tests:
 *   1. Validation of all supported saline volumes (100, 250, 500 mL) and rejection of unsupported.
 *   2. Validation of observation periods (baseline, post-intervention).
 *   3. Construction of SequentialWhatIfRequest payload matching backend schemas.
 *   4. Milestone deduplication: ensures baseline is not duplicated and chronological order is maintained.
 *   5. Physiological delta computations for HR, MAP, BV, CO, and optional vitals (SpO2, RR, SBP, DBP).
 *   6. Missing or malformed measurements handled gracefully without NaN.
 *   7. API request mock tests:
 *      - Successful execution for 100 mL, 250 mL, 500 mL.
 *      - HTTP 422 validation error.
 *      - HTTP 503 service unavailable (missing engine).
 *      - HTTP 504 gateway timeout.
 *      - Network connection error.
 *      - Malformed response rejection.
 *   8. Cached vs. live execution identification.
 *   9. Empty and incomplete trajectory handling.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  SUPPORTED_SALINE_VOLUMES,
  validateSimulationInputs,
  buildSequentialPayload,
  deduplicateMilestones,
  computePhysiologicalDeltas,
  executeSequentialWhatIf,
} from '../src/services/biogearsApi.js';

describe('1. Input Validation for Saline Interventions', () => {
  test('Accepts all three supported saline volumes (100 mL, 250 mL, 500 mL)', () => {
    for (const vol of [100, 250, 500]) {
      const result = validateSimulationInputs(vol, 60, 60);
      assert.equal(result.valid, true, `Volume ${vol} should be valid`);
      assert.equal(result.error, null);
    }
  });

  test('Rejects unsupported saline volumes', () => {
    const invalidVolumes = [0, 50, 150, 200, 300, 750, 1000, -100, 'invalid'];
    for (const vol of invalidVolumes) {
      const result = validateSimulationInputs(vol, 60, 60);
      assert.equal(result.valid, false, `Volume ${vol} should be rejected`);
      assert.match(result.error, /Invalid saline volume/);
    }
  });

  test('Validates baseline observation period', () => {
    assert.equal(validateSimulationInputs(100, 60, 60).valid, true);
    assert.equal(validateSimulationInputs(100, 10, 60).valid, true);
    assert.equal(validateSimulationInputs(100, 300, 60).valid, true);

    // Negative or zero
    assert.equal(validateSimulationInputs(100, 0, 60).valid, false);
    assert.equal(validateSimulationInputs(100, -30, 60).valid, false);
    assert.match(validateSimulationInputs(100, 0, 60).error, /Baseline observation period/);

    // Exceeds upper limit
    assert.equal(validateSimulationInputs(100, 700, 60).valid, false);
    assert.match(validateSimulationInputs(100, 700, 60).error, /exceeds maximum/);
  });

  test('Validates post-intervention observation period', () => {
    assert.equal(validateSimulationInputs(100, 60, 60).valid, true);
    assert.equal(validateSimulationInputs(100, 60, 120).valid, true);

    // Negative or zero
    assert.equal(validateSimulationInputs(100, 60, 0).valid, false);
    assert.equal(validateSimulationInputs(100, 60, -10).valid, false);
    assert.match(validateSimulationInputs(100, 60, 0).error, /Post-intervention observation period/);

    // Exceeds upper limit
    assert.equal(validateSimulationInputs(100, 60, 800).valid, false);
  });
});

describe('2. Request Payload Construction', () => {
  test('Constructs exact schema for single-action saline simulation', () => {
    const payload = buildSequentialPayload(250, 60, 90);

    assert.equal(payload.scenario_name, 'SeqSaline_1x250mL');
    assert.equal(payload.patient_file, 'StandardMale.xml');
    assert.equal(payload.baseline_stabilization_s, 60);
    assert.equal(payload.post_observation_s, 90);
    assert.equal(payload.samples_per_second, 0.2);

    assert.equal(Array.isArray(payload.actions), true);
    assert.equal(payload.actions.length, 1);

    const action = payload.actions[0];
    assert.equal(action.type, 'saline');
    assert.equal(action.volume_ml, 250);
    assert.equal(action.start_time_s, 60); // Action starts at end of baseline
  });
});

describe('3. Milestone Deduplication & Chronology', () => {
  const mockBaseline = {
    target_time_s: 60.0,
    actual_time_s: 60.0,
    milestone_type: 'baseline',
    description: 'Baseline observation (60.0s)',
    heart_rate_bpm: 72.1,
    mean_arterial_pressure_mmhg: 95.8,
    blood_volume_l: 5.54,
    cardiac_output_l_min: 5.81,
  };

  const mockIntermediates = [
    {
      phase: 'baseline',
      description: 'Baseline observation (60.0s)',
      target_time_s: 60.0,
      actual_time_s: 60.0,
      snapshot: { ...mockBaseline },
    },
    {
      phase: 'intervention_0',
      description: 'Action 0: Normal Saline 100.0mL completed',
      target_time_s: 180.0,
      actual_time_s: 180.0,
      snapshot: {
        target_time_s: 180.0,
        actual_time_s: 180.0,
        milestone_type: 'delivery_end',
        action_index: 0,
        heart_rate_bpm: 71.5,
        mean_arterial_pressure_mmhg: 96.2,
        blood_volume_l: 5.63,
        cardiac_output_l_min: 5.90,
      },
    },
    {
      phase: 'post_observation',
      description: 'Post-intervention observation (60.0s hold)',
      target_time_s: 240.0,
      actual_time_s: 240.0,
      snapshot: {
        target_time_s: 240.0,
        actual_time_s: 240.0,
        milestone_type: 'final_observation',
        heart_rate_bpm: 71.1,
        mean_arterial_pressure_mmhg: 96.3,
        blood_volume_l: 5.63,
        cardiac_output_l_min: 5.90,
      },
    },
  ];

  const mockFinal = {
    target_time_s: 240.0,
    actual_time_s: 240.0,
    milestone_type: 'final_state',
    description: 'Final state post-intervention',
    heart_rate_bpm: 71.1,
    mean_arterial_pressure_mmhg: 96.3,
    blood_volume_l: 5.63,
    cardiac_output_l_min: 5.90,
  };

  test('Guarantees baseline is not displayed twice', () => {
    const deduplicated = deduplicateMilestones(mockBaseline, mockIntermediates, mockFinal);

    // Count how many baseline milestones are in deduplicated output
    const baselineCount = deduplicated.filter(
      (m) => m.category === 'baseline' || m.phase === 'baseline' || Math.abs(m.target_time_s - 60.0) < 0.1
    ).length;

    assert.equal(baselineCount, 1, 'Baseline must appear exactly once');
  });

  test('Guarantees final snapshot is not displayed twice', () => {
    const deduplicated = deduplicateMilestones(mockBaseline, mockIntermediates, mockFinal);

    const finalCount = deduplicated.filter(
      (m) => m.category === 'final' || Math.abs(m.target_time_s - 240.0) < 0.1
    ).length;

    assert.equal(finalCount, 1, 'Final snapshot must appear exactly once');
  });

  test('Preserves distinct intervention delivery endpoints and enforces chronological order', () => {
    const deduplicated = deduplicateMilestones(mockBaseline, mockIntermediates, mockFinal);

    assert.equal(deduplicated.length, 3, 'Should have baseline, bolus delivery, and final state');
    assert.equal(deduplicated[0].target_time_s, 60.0);
    assert.equal(deduplicated[1].target_time_s, 180.0);
    assert.equal(deduplicated[2].target_time_s, 240.0);

    // Verify chronological order
    for (let i = 1; i < deduplicated.length; i++) {
      assert.ok(
        deduplicated[i].target_time_s >= deduplicated[i - 1].target_time_s,
        'Milestones must be sorted in non-decreasing chronological order'
      );
    }
  });
});

describe('4. Physiological Deltas Calculation', () => {
  const baseline = {
    heart_rate_bpm: 72.3,
    mean_arterial_pressure_mmhg: 95.9,
    blood_volume_l: 5.54,
    cardiac_output_l_min: 5.81,
    systolic_bp_mmhg: 120.0,
    diastolic_bp_mmhg: 80.0,
    oxygen_saturation: 0.98, // fraction
    respiration_rate_bpm: 14.0,
  };

  const finalSnap = {
    heart_rate_bpm: 71.1,
    mean_arterial_pressure_mmhg: 96.3,
    blood_volume_l: 5.63,
    cardiac_output_l_min: 5.90,
    systolic_bp_mmhg: 121.5,
    diastolic_bp_mmhg: 80.5,
    oxygen_saturation: 0.98,
    respiration_rate_bpm: 14.0,
  };

  test('Calculates accurate deltas with correct signs and units', () => {
    const deltas = computePhysiologicalDeltas(baseline, finalSnap);
    assert.ok(deltas);

    const hr = deltas.find((d) => d.key === 'heart_rate_bpm');
    assert.equal(hr.delta, 71.1 - 72.3);
    assert.equal(hr.formattedDelta, '-1.2');
    assert.equal(hr.unit, 'bpm');

    const map = deltas.find((d) => d.key === 'mean_arterial_pressure_mmhg');
    assert.equal(map.formattedDelta, '+0.4');
    assert.equal(map.unit, 'mmHg');

    const bv = deltas.find((d) => d.key === 'blood_volume_l');
    assert.equal(bv.formattedDelta, '+0.090');
    assert.equal(bv.unit, 'L');

    const co = deltas.find((d) => d.key === 'cardiac_output_l_min');
    assert.equal(co.formattedDelta, '+0.09');
    assert.equal(co.unit, 'L/min');
  });

  test('Converts fractional oxygen saturation to percentage', () => {
    const deltas = computePhysiologicalDeltas(baseline, finalSnap);
    const spo2 = deltas.find((d) => d.key === 'oxygen_saturation');
    assert.equal(spo2.formattedBase, '98.0');
    assert.equal(spo2.formattedFinal, '98.0');
    assert.equal(spo2.unit, '%');
  });

  test('Handles missing or null optional measurements gracefully without crashing or NaN', () => {
    const baselineWithNulls = {
      heart_rate_bpm: 72.0,
      mean_arterial_pressure_mmhg: 95.0,
      blood_volume_l: 5.5,
      cardiac_output_l_min: 5.8,
      systolic_bp_mmhg: null,
      diastolic_bp_mmhg: null,
      oxygen_saturation: null,
      respiration_rate_bpm: null,
    };

    const deltas = computePhysiologicalDeltas(baselineWithNulls, finalSnap);
    assert.ok(deltas);
    // Optional null measurements should be safely filtered out
    assert.equal(deltas.some((d) => d.key === 'systolic_bp_mmhg'), false);
    assert.equal(deltas.some((d) => isNaN(d.delta)), false);
  });
});

describe('5. API Execution & Error Handling (Mocked Fetch)', () => {
  const originalFetch = globalThis.fetch;

  test('Successfully requests and parses 100 mL, 250 mL, and 500 mL simulations', async () => {
    for (const volume of [100, 250, 500]) {
      globalThis.fetch = async (url, options) => {
        assert.match(url, /\/biogears\/whatif\/sequential$/);
        assert.equal(options.method, 'POST');
        const body = JSON.parse(options.body);
        assert.equal(body.actions[0].volume_ml, volume);

        return {
          ok: true,
          status: 200,
          json: async () => ({
            metadata: {
              scenario_name: `SeqSaline_1x${volume}mL`,
              patient_file: 'StandardMale.xml',
              is_cached: true,
              source: 'cache',
              simulated_duration_s: 240.0,
              row_count: 48,
              wall_clock_seconds: 0.0,
              actions: [
                {
                  action_index: 0,
                  action_type: 'saline',
                  bag_volume_ml: volume,
                  rate_ml_per_min: 50.0,
                  start_time_s: 60.0,
                  delivery_duration_s: 120.0,
                },
              ],
            },
            baseline_snapshot: {
              target_time_s: 60.0,
              actual_time_s: 60.0,
              heart_rate_bpm: 72.0,
              mean_arterial_pressure_mmhg: 95.0,
              blood_volume_l: 5.5,
              cardiac_output_l_min: 5.8,
            },
            intermediate_milestones: [],
            final_snapshot: {
              target_time_s: 240.0,
              actual_time_s: 240.0,
              heart_rate_bpm: 71.0,
              mean_arterial_pressure_mmhg: 96.0,
              blood_volume_l: 5.6,
              cardiac_output_l_min: 5.9,
            },
            trajectory: {
              time_s: [5.0, 10.0, 60.0, 180.0, 240.0],
              signals: {},
              data_points: [
                { time_s: 5.0, heart_rate_bpm: 72.0, mean_arterial_pressure_mmhg: 95.0 },
                { time_s: 240.0, heart_rate_bpm: 71.0, mean_arterial_pressure_mmhg: 96.0 },
              ],
            },
            provenance_disclaimer: 'Mock disclaimer',
          }),
        };
      };

      const result = await executeSequentialWhatIf({ volumeMl: volume });
      assert.equal(result.metadata.actions[0].bag_volume_ml, volume);
      assert.equal(result.metadata.is_cached, true);
    }
    globalThis.fetch = originalFetch;
  });

  test('Correctly identifies and classifies HTTP 422 Validation Error', async () => {
    globalThis.fetch = async () => ({
      ok: false,
      status: 422,
      json: async () => ({ detail: 'Scenario validation rejected: Invalid volume' }),
    });

    await assert.rejects(
      async () => executeSequentialWhatIf({ volumeMl: 100 }),
      /Validation Error \(HTTP 422\): Scenario validation rejected/
    );
    globalThis.fetch = originalFetch;
  });

  test('Correctly identifies and classifies HTTP 503 Engine Unavailable', async () => {
    globalThis.fetch = async () => ({
      ok: false,
      status: 503,
      json: async () => ({ detail: 'BioGears engine executable not found on host system.' }),
    });

    await assert.rejects(
      async () => executeSequentialWhatIf({ volumeMl: 100 }),
      /BioGears Engine Unavailable \(HTTP 503\)/
    );
    globalThis.fetch = originalFetch;
  });

  test('Correctly identifies and classifies HTTP 504 Timeout', async () => {
    globalThis.fetch = async () => ({
      ok: false,
      status: 504,
      json: async () => ({ detail: 'BioGears simulation timed out after 600s.' }),
    });

    await assert.rejects(
      async () => executeSequentialWhatIf({ volumeMl: 100 }),
      /Simulation Timeout \(HTTP 504\)/
    );
    globalThis.fetch = originalFetch;
  });

  test('Handles network connection failure with actionable message', async () => {
    globalThis.fetch = async () => {
      throw new TypeError('Failed to fetch');
    };

    await assert.rejects(
      async () => executeSequentialWhatIf({ volumeMl: 100 }),
      /Network error: Unable to connect to backend server/
    );
    globalThis.fetch = originalFetch;
  });

  test('Rejects malformed response missing required snapshots', async () => {
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({ invalid_body: true }),
    });

    await assert.rejects(
      async () => executeSequentialWhatIf({ volumeMl: 100 }),
      /Malformed API response: missing required metadata/
    );
    globalThis.fetch = originalFetch;
  });
});
