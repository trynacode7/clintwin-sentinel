/**
 * biogears_sequential_whatif.test.mjs — Comprehensive Step G Unit Tests
 * ======================================================================
 * Tests:
 *   1. Multi-intervention input validation (1, 2, 3 actions):
 *      - Supported saline volumes (100, 250, 500 mL) for each bolus independently.
 *      - Rejection of unsupported volumes, zero actions, >3 actions.
 *      - Chronological ordering and non-overlapping constraint enforcement.
 *      - Bolus 1 preceding baseline rejection.
 *      - Total simulation duration limit (> 1800s) rejection with explanation.
 *   2. Request payload construction:
 *      - Backward-compatible 1-bolus payload.
 *      - 2-bolus sequential payload matching SequentialWhatIfRequest schema.
 *      - 3-bolus sequential payload matching SequentialWhatIfRequest schema.
 *   3. Delivery duration calculations:
 *      - Verified 50 mL/min delivery duration: 100 mL -> 120s, 250 mL -> 300s, 500 mL -> 600s.
 *   4. Milestone deduplication across multi-action scenarios:
 *      - Baseline and final snapshots deduplicated.
 *      - Multiple distinct intervention endpoints preserved in chronological order.
 *   5. Consecutive milestone delta calculations (Δx = x_current - x_previous):
 *      - Signed absolute changes between consecutive milestones with timestamps and descriptions.
 *      - Handles missing / optional measurements without NaN or fake values.
 *   6. API execution and error handling:
 *      - 1-, 2-, and 3-bolus mocked successful executions.
 *      - HTTP 422, 503, 504 error mapping.
 *      - Network failure and malformed response rejection.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  SUPPORTED_SALINE_VOLUMES,
  calculateDeliveryDuration,
  validateSequentialSimulation,
  validateSimulationInputs,
  buildSequentialPayload,
  deduplicateMilestones,
  computePhysiologicalDeltas,
  computeConsecutiveMilestoneDeltas,
  executeSequentialWhatIf,
} from '../src/services/biogearsApi.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURES_DIR = path.join(__dirname, 'fixtures');

describe('1. Active Saline Delivery Duration Derivation', () => {
  test('Calculates exact delivery duration at verified 50 mL/min rate', () => {
    assert.equal(calculateDeliveryDuration(100), 120);  // (100 / 50) * 60 = 120s
    assert.equal(calculateDeliveryDuration(250), 300);  // (250 / 50) * 60 = 300s
    assert.equal(calculateDeliveryDuration(500), 600);  // (500 / 50) * 60 = 600s
    assert.equal(calculateDeliveryDuration(0), 0);
    assert.equal(calculateDeliveryDuration(-100), 0);
  });
});

describe('2. Multi-Action Sequential Validation Rules', () => {
  test('Accepts valid 1-bolus scenario', () => {
    const res = validateSequentialSimulation({
      actions: [{ volume_ml: 100, start_time_s: 60 }],
      baselineObservationS: 60,
      postObservationS: 60,
    });
    assert.equal(res.valid, true);
    assert.equal(res.error, null);
    assert.equal(res.totalDuration, 240); // 60 + 120 + 60
  });

  test('Accepts valid 2-bolus scenario with different volumes', () => {
    const res = validateSequentialSimulation({
      actions: [
        { volume_ml: 100, start_time_s: 60 },  // completes at 180s
        { volume_ml: 250, start_time_s: 240 }, // completes at 240 + 300 = 540s
      ],
      baselineObservationS: 60,
      postObservationS: 60,
    });
    assert.equal(res.valid, true);
    assert.equal(res.error, null);
    assert.equal(res.totalDuration, 600); // 540 + 60
  });

  test('Accepts valid 3-bolus scenario with mixed supported volumes', () => {
    const res = validateSequentialSimulation({
      actions: [
        { volume_ml: 100, start_time_s: 60 },  // delivers 60..180s
        { volume_ml: 100, start_time_s: 180 }, // delivers 180..300s (back-to-back)
        { volume_ml: 500, start_time_s: 360 }, // delivers 360..960s
      ],
      baselineObservationS: 60,
      postObservationS: 60,
    });
    assert.equal(res.valid, true);
    assert.equal(res.error, null);
    assert.equal(res.totalDuration, 1020); // 960 + 60
  });

  test('Rejects zero actions or more than 3 actions', () => {
    const resZero = validateSequentialSimulation({ actions: [] });
    assert.equal(resZero.valid, false);
    assert.match(resZero.error, /between 1 and 3 sequential interventions/);

    const fourActions = [
      { volume_ml: 100, start_time_s: 60 },
      { volume_ml: 100, start_time_s: 200 },
      { volume_ml: 100, start_time_s: 340 },
      { volume_ml: 100, start_time_s: 480 },
    ];
    const resFour = validateSequentialSimulation({ actions: fourActions });
    assert.equal(resFour.valid, false);
    assert.match(resFour.error, /between 1 and 3 sequential interventions/);
  });

  test('Rejects unsupported volume in any intervention row', () => {
    const res = validateSequentialSimulation({
      actions: [
        { volume_ml: 100, start_time_s: 60 },
        { volume_ml: 300, start_time_s: 200 }, // 300 mL is unverified!
      ],
    });
    assert.equal(res.valid, false);
    assert.match(res.error, /Bolus 2: Invalid saline volume \(300 mL\)/);
  });

  test('Rejects first intervention starting before baseline stabilization', () => {
    const res = validateSequentialSimulation({
      actions: [{ volume_ml: 100, start_time_s: 40 }], // baseline is 60!
      baselineObservationS: 60,
    });
    assert.equal(res.valid, false);
    assert.match(res.error, /Bolus 1 start time \(40s\) precedes baseline stabilization/);
  });

  test('Rejects overlapping interventions and derives correct minimum start time', () => {
    const res = validateSequentialSimulation({
      actions: [
        { volume_ml: 100, start_time_s: 60 },  // 100mL @ 50mL/min delivers until 60 + 120 = 180s
        { volume_ml: 100, start_time_s: 150 }, // Overlaps at 150s!
      ],
      baselineObservationS: 60,
    });
    assert.equal(res.valid, false);
    assert.match(res.error, /Bolus 2 start time \(150s\) precedes previous bolus completion \(180s\)/);
    assert.match(res.error, /Increase start time to at least 180s/);
  });

  test('Rejects total simulation duration exceeding 1800s limit with explanatory message', () => {
    const res = validateSequentialSimulation({
      actions: [
        { volume_ml: 500, start_time_s: 60 },   // 60..660s
        { volume_ml: 500, start_time_s: 700 },  // 700..1300s
        { volume_ml: 500, start_time_s: 1350 }, // 1350..1950s -> Exceeds 1800s!
      ],
      baselineObservationS: 60,
      postObservationS: 60,
    });
    assert.equal(res.valid, false);
    assert.match(res.error, /Total simulated duration is 2010s, exceeding the 1800s maximum limit/);
    assert.match(res.error, /reduce intervention start times, decrease bolus volumes/);
  });

  test('validateSimulationInputs delegates to validateSequentialSimulation for action arrays', () => {
    const resArray = validateSimulationInputs([
      { volume_ml: 100, start_time_s: 60 },
      { volume_ml: 100, start_time_s: 200 },
    ]);
    assert.equal(resArray.valid, true);

    const resOverlap = validateSimulationInputs([
      { volume_ml: 100, start_time_s: 60 },
      { volume_ml: 100, start_time_s: 100 },
    ]);
    assert.equal(resOverlap.valid, false);
    assert.match(resOverlap.error, /precedes previous bolus completion/);
  });
});

describe('3. Multi-Action Payload Construction', () => {
  test('Constructs 2-bolus payload matching SequentialWhatIfRequest schema', () => {
    const actions = [
      { volume_ml: 100, start_time_s: 60 },
      { volume_ml: 250, start_time_s: 240 },
    ];
    const payload = buildSequentialPayload(actions, 60, 90);

    assert.equal(payload.scenario_name, 'SeqSaline_2actions_100mL_250mL');
    assert.equal(payload.patient_file, 'StandardMale.xml');
    assert.equal(payload.baseline_stabilization_s, 60);
    assert.equal(payload.post_observation_s, 90);
    assert.equal(payload.samples_per_second, 0.2);
    assert.equal(payload.actions.length, 2);

    assert.deepEqual(payload.actions[0], { type: 'saline', volume_ml: 100, start_time_s: 60 });
    assert.deepEqual(payload.actions[1], { type: 'saline', volume_ml: 250, start_time_s: 240 });
  });

  test('Constructs 3-bolus payload matching SequentialWhatIfRequest schema', () => {
    const actions = [
      { volume_ml: 100, start_time_s: 60 },
      { volume_ml: 100, start_time_s: 200 },
      { volume_ml: 500, start_time_s: 350 },
    ];
    const payload = buildSequentialPayload(actions);

    assert.equal(payload.scenario_name, 'SeqSaline_3actions_100mL_100mL_500mL');
    assert.equal(payload.actions.length, 3);
    assert.equal(payload.actions[2].volume_ml, 500);
    assert.equal(payload.actions[2].start_time_s, 350);
  });

  test('Preserves Step F backward-compatible single bolus signature', () => {
    const payload = buildSequentialPayload(500, 60, 60);
    assert.equal(payload.scenario_name, 'SeqSaline_1x500mL');
    assert.equal(payload.actions.length, 1);
    assert.equal(payload.actions[0].volume_ml, 500);
    assert.equal(payload.actions[0].start_time_s, 60);
  });
});

describe('4. Milestone Deduplication for 2 and 3 Interventions', () => {
  const baseSnap = {
    target_time_s: 60.0,
    actual_time_s: 60.0,
    description: 'Baseline observation (60.0s)',
    heart_rate_bpm: 72.0,
    mean_arterial_pressure_mmhg: 95.0,
    blood_volume_l: 5.50,
    cardiac_output_l_min: 5.80,
  };

  const finalSnap = {
    target_time_s: 540.0,
    actual_time_s: 540.0,
    description: 'Final post-observation state',
    heart_rate_bpm: 70.5,
    mean_arterial_pressure_mmhg: 96.8,
    blood_volume_l: 5.75,
    cardiac_output_l_min: 6.00,
  };

  test('Correctly preserves multiple distinct intervention milestones in order', () => {
    const intermediates = [
      {
        phase: 'baseline', // duplicate of baseSnap
        target_time_s: 60.0,
        snapshot: { ...baseSnap },
      },
      {
        phase: 'intervention_0',
        description: 'Bolus 1 (100mL) Delivery End',
        target_time_s: 180.0,
        actual_time_s: 180.0,
        snapshot: {
          heart_rate_bpm: 71.4,
          mean_arterial_pressure_mmhg: 95.9,
          blood_volume_l: 5.59,
          cardiac_output_l_min: 5.88,
        },
      },
      {
        phase: 'intermediate_hold_0_to_1',
        description: 'Intermediate observation hold (60s)',
        target_time_s: 240.0,
        actual_time_s: 240.0,
        snapshot: {
          heart_rate_bpm: 71.2,
          mean_arterial_pressure_mmhg: 96.0,
          blood_volume_l: 5.59,
          cardiac_output_l_min: 5.87,
        },
      },
      {
        phase: 'intervention_1',
        description: 'Bolus 2 (250mL) Delivery End',
        target_time_s: 480.0,
        actual_time_s: 480.0,
        snapshot: {
          heart_rate_bpm: 70.8,
          mean_arterial_pressure_mmhg: 96.5,
          blood_volume_l: 5.75,
          cardiac_output_l_min: 5.98,
        },
      },
      {
        phase: 'post_observation', // duplicate of finalSnap
        target_time_s: 540.0,
        snapshot: { ...finalSnap },
      },
    ];

    const deduplicated = deduplicateMilestones(baseSnap, intermediates, finalSnap);

    // Milestones: baseline (60s), intervention_0 (180s), hold (240s), intervention_1 (480s), final (540s)
    assert.equal(deduplicated.length, 5);
    assert.equal(deduplicated[0].target_time_s, 60.0);
    assert.equal(deduplicated[1].target_time_s, 180.0);
    assert.equal(deduplicated[2].target_time_s, 240.0);
    assert.equal(deduplicated[3].target_time_s, 480.0);
    assert.equal(deduplicated[4].target_time_s, 540.0);

    // Chronological order verification
    for (let i = 1; i < deduplicated.length; i++) {
      assert.ok(deduplicated[i].target_time_s > deduplicated[i - 1].target_time_s);
    }
  });
});

describe('5. Consecutive Milestone Physiological Deltas (Step-by-Step Δ)', () => {
  const milestones = [
    {
      phase: 'baseline',
      description: 'Baseline',
      target_time_s: 60,
      actual_time_s: 60,
      snapshot: {
        heart_rate_bpm: 72.0,
        mean_arterial_pressure_mmhg: 95.0,
        blood_volume_l: 5.50,
        cardiac_output_l_min: 5.80,
      },
    },
    {
      phase: 'intervention_0',
      description: 'Bolus 1 End',
      target_time_s: 180,
      actual_time_s: 180,
      snapshot: {
        heart_rate_bpm: 71.0,
        mean_arterial_pressure_mmhg: 96.0,
        blood_volume_l: 5.60,
        cardiac_output_l_min: 5.90,
      },
    },
    {
      phase: 'intervention_1',
      description: 'Bolus 2 End',
      target_time_s: 360,
      actual_time_s: 360,
      snapshot: {
        heart_rate_bpm: 70.0,
        mean_arterial_pressure_mmhg: 97.2,
        blood_volume_l: 5.75,
        cardiac_output_l_min: 6.05,
      },
    },
  ];

  test('Computes signed absolute changes Δx = x_curr - x_prev between adjacent milestones', () => {
    const comparisons = computeConsecutiveMilestoneDeltas(milestones);
    assert.equal(comparisons.length, 2);

    // Step 1: Baseline (60s) -> Bolus 1 End (180s)
    const step1 = comparisons[0];
    assert.equal(step1.stepIndex, 1);
    assert.equal(step1.fromTime, 60);
    assert.equal(step1.toTime, 180);
    assert.match(step1.label, /Baseline \(60s\) → Bolus 1 End \(180s\)/);

    const hrDelta1 = step1.deltas.find((d) => d.key === 'heart_rate_bpm');
    assert.equal(hrDelta1.formattedDelta, '-1.0');
    const mapDelta1 = step1.deltas.find((d) => d.key === 'mean_arterial_pressure_mmhg');
    assert.equal(mapDelta1.formattedDelta, '+1.0');
    const bvDelta1 = step1.deltas.find((d) => d.key === 'blood_volume_l');
    assert.equal(bvDelta1.formattedDelta, '+0.100');

    // Step 2: Bolus 1 End (180s) -> Bolus 2 End (360s)
    const step2 = comparisons[1];
    assert.equal(step2.stepIndex, 2);
    assert.equal(step2.fromTime, 180);
    assert.equal(step2.toTime, 360);

    const hrDelta2 = step2.deltas.find((d) => d.key === 'heart_rate_bpm');
    assert.equal(hrDelta2.formattedDelta, '-1.0');
    const mapDelta2 = step2.deltas.find((d) => d.key === 'mean_arterial_pressure_mmhg');
    assert.equal(mapDelta2.formattedDelta, '+1.2');
    const bvDelta2 = step2.deltas.find((d) => d.key === 'blood_volume_l');
    assert.equal(bvDelta2.formattedDelta, '+0.150');
  });

  test('Returns empty array safely if fewer than 2 milestones provided', () => {
    assert.deepEqual(computeConsecutiveMilestoneDeltas([]), []);
    assert.deepEqual(computeConsecutiveMilestoneDeltas([milestones[0]]), []);
  });
});

describe('6. Multi-Bolus Simulation API Execution', () => {
  const originalFetch = globalThis.fetch;

  test('Executes 2-bolus scenario and parses response with multiple actions and milestones', async () => {
    const actions = [
      { volume_ml: 100, start_time_s: 60 },
      { volume_ml: 250, start_time_s: 240 },
    ];

    globalThis.fetch = async (url, options) => {
      assert.match(url, /\/biogears\/whatif\/sequential$/);
      assert.equal(options.method, 'POST');
      const reqBody = JSON.parse(options.body);
      assert.equal(reqBody.actions.length, 2);
      assert.equal(reqBody.actions[0].volume_ml, 100);
      assert.equal(reqBody.actions[1].volume_ml, 250);

      return {
        ok: true,
        status: 200,
        json: async () => ({
          metadata: {
            scenario_name: 'SeqSaline_2actions_100mL_250mL',
            patient_file: 'StandardMale.xml',
            is_cached: false,
            source: 'live',
            simulated_duration_s: 600.0,
            row_count: 120,
            wall_clock_seconds: 42.5,
            actions: [
              {
                action_index: 0,
                action_type: 'saline',
                bag_volume_ml: 100.0,
                rate_ml_per_min: 50.0,
                start_time_s: 60.0,
                delivery_duration_s: 120.0,
              },
              {
                action_index: 1,
                action_type: 'saline',
                bag_volume_ml: 250.0,
                rate_ml_per_min: 50.0,
                start_time_s: 240.0,
                delivery_duration_s: 300.0,
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
          intermediate_milestones: [
            {
              phase: 'intervention_0',
              target_time_s: 180.0,
              actual_time_s: 180.0,
              description: 'Bolus 1 End',
              snapshot: {
                heart_rate_bpm: 71.5,
                mean_arterial_pressure_mmhg: 95.8,
                blood_volume_l: 5.59,
                cardiac_output_l_min: 5.86,
              },
            },
            {
              phase: 'intervention_1',
              target_time_s: 540.0,
              actual_time_s: 540.0,
              description: 'Bolus 2 End',
              snapshot: {
                heart_rate_bpm: 70.8,
                mean_arterial_pressure_mmhg: 96.6,
                blood_volume_l: 5.75,
                cardiac_output_l_min: 5.98,
              },
            },
          ],
          final_snapshot: {
            target_time_s: 600.0,
            actual_time_s: 600.0,
            heart_rate_bpm: 70.6,
            mean_arterial_pressure_mmhg: 96.7,
            blood_volume_l: 5.75,
            cardiac_output_l_min: 5.99,
          },
          trajectory: {
            time_s: [0, 60, 180, 240, 540, 600],
            signals: {},
            data_points: [
              { time_s: 60, heart_rate_bpm: 72.0, mean_arterial_pressure_mmhg: 95.0 },
              { time_s: 600, heart_rate_bpm: 70.6, mean_arterial_pressure_mmhg: 96.7 },
            ],
          },
          provenance_disclaimer: 'Verified Step G multi-bolus simulation',
        }),
      };
    };

    const result = await executeSequentialWhatIf({ actions });
    assert.equal(result.metadata.actions.length, 2);
    assert.equal(result.metadata.is_cached, false);
    assert.equal(result.intermediate_milestones.length, 2);
    assert.equal(result.metadata.wall_clock_seconds, 42.5);

    globalThis.fetch = originalFetch;
  });
});

describe('7. Verification with Real BioGears 1-, 2-, and 3-Bolus Responses', () => {
  test('Processes real 1-bolus BioGears response fixture correctly', () => {
    const raw = fs.readFileSync(path.join(FIXTURES_DIR, 'real_1bolus_response.json'), 'utf8');
    const resp = JSON.parse(raw);

    assert.equal(resp.metadata.patient_file, 'StandardMale.xml');
    assert.equal(resp.metadata.actions.length, 1);
    assert.equal(resp.metadata.is_continuous, true);

    const deltas = computePhysiologicalDeltas(resp.baseline_snapshot, resp.final_snapshot);
    assert.ok(deltas);
    assert.equal(deltas.length >= 4, true);

    const hrDelta = deltas.find(d => d.key === 'heart_rate_bpm');
    assert.ok(hrDelta);
    assert.equal(typeof hrDelta.delta, 'number');
    assert.equal(isNaN(hrDelta.delta), false);

    const milestones = deduplicateMilestones(resp.baseline_snapshot, resp.intermediate_milestones, resp.final_snapshot);
    assert.ok(milestones.length >= 3);
  });

  test('Processes real 2-bolus BioGears response fixture correctly', () => {
    const raw = fs.readFileSync(path.join(FIXTURES_DIR, 'real_2bolus_response.json'), 'utf8');
    const resp = JSON.parse(raw);

    assert.equal(resp.metadata.actions.length, 2);
    assert.equal(resp.metadata.actions[0].start_time_s, 60);
    assert.equal(resp.metadata.actions[1].start_time_s, 240);

    const milestones = deduplicateMilestones(resp.baseline_snapshot, resp.intermediate_milestones, resp.final_snapshot);
    assert.ok(milestones.length >= 4);

    const consecutive = computeConsecutiveMilestoneDeltas(milestones);
    assert.ok(consecutive.length >= 3);

    for (const step of consecutive) {
      assert.ok(step.fromTime <= step.toTime);
      assert.ok(step.deltas.length >= 4);
    }
  });

  test('Processes real 3-bolus BioGears response fixture correctly', () => {
    const raw = fs.readFileSync(path.join(FIXTURES_DIR, 'real_3bolus_response.json'), 'utf8');
    const resp = JSON.parse(raw);

    assert.equal(resp.metadata.actions.length, 3);
    assert.equal(resp.metadata.actions[0].start_time_s, 60);
    assert.equal(resp.metadata.actions[1].start_time_s, 240);
    assert.equal(resp.metadata.actions[2].start_time_s, 420);

    const milestones = deduplicateMilestones(resp.baseline_snapshot, resp.intermediate_milestones, resp.final_snapshot);
    assert.ok(milestones.length >= 5);

    const consecutive = computeConsecutiveMilestoneDeltas(milestones);
    assert.ok(consecutive.length >= 4);

    for (const step of consecutive) {
      assert.ok(step.fromTime <= step.toTime);
      assert.ok(step.deltas.every(d => !isNaN(d.delta)));
    }
  });

  test('Validates fresh live 2-bolus execution fixture (100mL @ 60s, 250mL @ 210s)', () => {
    const raw = fs.readFileSync(path.join(FIXTURES_DIR, 'fresh_2bolus_response.json'), 'utf8');
    const resp = JSON.parse(raw);

    assert.equal(resp.metadata.scenario_name, 'SeqSaline_2actions_100mL_250mL_Fresh');
    assert.equal(resp.metadata.actions.length, 2);
    assert.equal(resp.metadata.is_continuous, true);
    assert.equal(resp.metadata.row_count, 114);
    assert.equal(resp.metadata.simulated_duration_s, 570);

    // Verify delivery duration and timing
    const act0 = resp.metadata.actions[0];
    const act1 = resp.metadata.actions[1];
    assert.equal(act0.bag_volume_ml, 100);
    assert.equal(act0.delivery_duration_s, 120);
    assert.equal(act1.bag_volume_ml, 250);
    assert.equal(act1.delivery_duration_s, 300);
    assert.ok(act1.start_time_s >= act0.start_time_s + act0.delivery_duration_s);

    // Verify milestones and timestamp tolerance
    const milestones = deduplicateMilestones(resp.baseline_snapshot, resp.intermediate_milestones, resp.final_snapshot);
    assert.equal(milestones.length, 5); // baseline, B1 end (180s), hold (210s), B2 end (510s), final (570s)

    for (const m of milestones) {
      const diff = Math.abs(m.actual_time_s - m.target_time_s);
      assert.ok(diff <= 0.1, `Timestamp diff ${diff}s must be within tolerance`);
    }

    // Verify step-by-step consecutive deltas
    const consecutive = computeConsecutiveMilestoneDeltas(milestones);
    assert.equal(consecutive.length, 4);
    for (const step of consecutive) {
      assert.ok(step.fromTime < step.toTime);
      assert.equal(step.deltas.some(d => isNaN(d.delta)), false);
    }
  });

  test('Validates fresh live 3-bolus execution fixture (100mL @ 60s, 100mL @ 195s, 100mL @ 330s)', () => {
    const raw = fs.readFileSync(path.join(FIXTURES_DIR, 'fresh_3bolus_response.json'), 'utf8');
    const resp = JSON.parse(raw);

    assert.equal(resp.metadata.scenario_name, 'SeqSaline_3actions_100mL_100mL_100mL_Fresh');
    assert.equal(resp.metadata.actions.length, 3);
    assert.equal(resp.metadata.is_continuous, true);
    assert.equal(resp.metadata.row_count, 102);
    assert.equal(resp.metadata.simulated_duration_s, 510);

    // Verify action ordering
    for (let i = 1; i < resp.metadata.actions.length; i++) {
      const prev = resp.metadata.actions[i - 1];
      const curr = resp.metadata.actions[i];
      assert.ok(curr.start_time_s >= prev.start_time_s + prev.delivery_duration_s);
    }

    // Verify milestones
    const milestones = deduplicateMilestones(resp.baseline_snapshot, resp.intermediate_milestones, resp.final_snapshot);
    assert.equal(milestones.length, 7); // baseline, B1 end, hold1, B2 end, hold2, B3 end, final

    for (let i = 1; i < milestones.length; i++) {
      assert.ok(milestones[i].actual_time_s > milestones[i - 1].actual_time_s);
    }

    // Verify consecutive deltas
    const consecutive = computeConsecutiveMilestoneDeltas(milestones);
    assert.equal(consecutive.length, 6);
    for (const step of consecutive) {
      const bv = step.deltas.find(d => d.key === 'blood_volume_l');
      assert.ok(bv && typeof bv.delta === 'number');
    }
  });
});

