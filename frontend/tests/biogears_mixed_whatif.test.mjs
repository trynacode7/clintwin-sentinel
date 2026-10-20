/**
 * Automated Verification Suite for ClinTwin Sentinel Mixed Interventions
 * (Saline Bolus + Norepinephrine Infusion)
 * =====================================================================
 * Tests:
 *   1. Mixed Intervention Array Validation Rules
 *   2. Mixed Request Payload Construction
 *   3. Scenario Summary Table Handling with Norepinephrine
 *   4. Trajectory Data Alignment with Mixed Scenarios
 *   5. API Execution & Snapshot Deltas for Mixed Scenarios
 */

import { test, describe, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  SUPPORTED_SALINE_VOLUMES,
  VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
  VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
  VERIFIED_NOREPINEPHRINE_DURATION_S,
  validateSequentialSimulation,
  buildSequentialPayload,
  computeScenarioSummaryTable,
  alignComparisonTrajectories,
  deduplicateMilestones,
  computePhysiologicalDeltas,
  executeSequentialWhatIf,
} from '../src/services/biogearsApi.js';

describe('STEP I: Mixed Sequential Interventions Frontend Suite', () => {
  describe('1. Mixed Intervention Array Validation', () => {
    test('Accepts valid mixed saline + norepinephrine scenario', () => {
      const actions = [
        { type: 'saline', volume_ml: 100, start_time_s: 60 },
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 120.0,
          start_time_s: 210,
        },
      ];

      const res = validateSequentialSimulation({
        actions,
        baselineObservationS: 60,
        postObservationS: 60,
      });

      assert.equal(res.valid, true);
      assert.equal(res.error, null);
      // Timeline: 0-60 baseline, 60-180 saline, 180-210 pause, 210-330 norepi, 330-390 post = 390s
      assert.equal(res.totalDuration, 390);
    });

    test('Rejects unsupported intervention type', () => {
      const actions = [
        { type: 'epinephrine', concentration_ug_ml: 1.0, start_time_s: 60 },
      ];

      const res = validateSequentialSimulation({ actions });
      assert.equal(res.valid, false);
      assert.match(res.error, /Unsupported intervention type 'epinephrine'/);
    });

    test('Rejects invalid norepinephrine concentration', () => {
      const actions = [
        {
          type: 'norepinephrine',
          concentration_ug_ml: 25.0,
          rate_ml_per_min: 1.0,
          duration_s: 120.0,
          start_time_s: 60,
        },
      ];

      const res = validateSequentialSimulation({ actions });
      assert.equal(res.valid, false);
      assert.match(res.error, /Unsupported concentration \(25 ug\/mL\)/);
    });

    test('Rejects invalid norepinephrine infusion rate', () => {
      const actions = [
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 5.0,
          duration_s: 120.0,
          start_time_s: 60,
        },
      ];

      const res = validateSequentialSimulation({ actions });
      assert.equal(res.valid, false);
      assert.match(res.error, /Unsupported rate \(5 mL\/min\)/);
    });

    test('Rejects invalid norepinephrine infusion duration', () => {
      const actions = [
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 60.0,
          start_time_s: 60,
        },
      ];

      const res = validateSequentialSimulation({ actions });
      assert.equal(res.valid, false);
      assert.match(res.error, /Unsupported duration \(60 s\)/);
    });

    test('Rejects overlapping interventions in mixed scenario', () => {
      const actions = [
        { type: 'saline', volume_ml: 100, start_time_s: 60 }, // finishes at 60 + 120 = 180s
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 120.0,
          start_time_s: 150, // Conflict: 150 < 180s
        },
      ];

      const res = validateSequentialSimulation({ actions });
      assert.equal(res.valid, false);
      assert.match(res.error, /Infusion 2 start time \(150s\) precedes previous bolus completion \(180s\)/);
    });
  });

  describe('2. Mixed Payload Construction', () => {
    test('Constructs descriptive mixed scenario payload', () => {
      const actions = [
        { type: 'saline', volume_ml: 100, start_time_s: 60 },
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 120.0,
          start_time_s: 210,
        },
      ];

      const payload = buildSequentialPayload(actions, 60, 60);

      assert.equal(payload.scenario_name, 'SeqMixed_2actions_100mL_14ugNE');
      assert.equal(payload.patient_file, 'StandardMale.xml');
      assert.equal(payload.baseline_stabilization_s, 60);
      assert.equal(payload.post_observation_s, 60);
      assert.equal(payload.actions.length, 2);
      assert.deepEqual(payload.actions[0], {
        type: 'saline',
        volume_ml: 100,
        start_time_s: 60,
      });
      assert.deepEqual(payload.actions[1], {
        type: 'norepinephrine',
        concentration_ug_ml: 14.0,
        rate_ml_per_min: 1.0,
        duration_s: 120.0,
        start_time_s: 210,
      });
    });

    test('Constructs pure norepinephrine scenario payload', () => {
      const actions = [
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 120.0,
          start_time_s: 60,
        },
      ];

      const payload = buildSequentialPayload(actions, 60, 60);
      assert.equal(payload.scenario_name, 'SeqNorepi_1actions_14ugNE');
      assert.equal(payload.actions.length, 1);
      assert.equal(payload.actions[0].type, 'norepinephrine');
    });
  });

  describe('3. Scenario Summary Table Handling with Norepinephrine', () => {
    test('Computes total volume excluding norepinephrine and sets hasNorepinephrine flag', () => {
      const scenarios = [
        {
          id: 'strat_mixed',
          name: 'Saline + Norepi',
          status: 'completed',
          result: {
            metadata: {
              scenario_name: 'SeqMixed_Saline100_Norepi14ug',
              is_cached: true,
              wall_clock_seconds: 0.05,
              simulated_duration_s: 390,
              actions: [
                { type: 'saline', bag_volume_ml: 100, start_time_s: 60 },
                { type: 'norepinephrine', substance: 'Norepinephrine', concentration_ug_ml: 14.0, start_time_s: 210 },
              ],
            },
            baseline_snapshot: {
              heart_rate_bpm: 72.0,
              mean_arterial_pressure_mmhg: 85.0,
              blood_volume_l: 5.2,
              cardiac_output_l_min: 5.6,
            },
            final_snapshot: {
              heart_rate_bpm: 70.0,
              mean_arterial_pressure_mmhg: 88.0,
              blood_volume_l: 5.3,
              cardiac_output_l_min: 5.7,
            },
          },
        },
      ];

      const table = computeScenarioSummaryTable(scenarios);
      assert.equal(table.length, 1);
      const row = table[0];

      assert.equal(row.totalVolumeMl, 100); // 100mL Saline, 0mL from norepinephrine
      assert.equal(row.hasNorepinephrine, true);
      assert.equal(row.numInterventions, 2);
      assert.equal(row.vitals.map.delta, 3.0); // 88.0 - 85.0 = +3.0 mmHg
      assert.equal(row.vitals.hr.delta, -2.0);  // 70.0 - 72.0 = -2.0 bpm
    });
  });

  describe('4. Trajectory Alignment for Mixed Scenarios', () => {
    test('Aligns mixed scenario with pure saline scenario cleanly without extrapolation', () => {
      const s1 = {
        id: 'strat_saline',
        name: 'Pure Saline 100mL',
        result: {
          trajectory: {
            data_points: [
              { time_s: 5.0, heart_rate_bpm: 72, mean_arterial_pressure_mmhg: 85 },
              { time_s: 60.0, heart_rate_bpm: 72, mean_arterial_pressure_mmhg: 85 },
              { time_s: 240.0, heart_rate_bpm: 71, mean_arterial_pressure_mmhg: 86 },
            ],
          },
        },
      };

      const s2 = {
        id: 'strat_mixed',
        name: 'Saline + Norepi',
        result: {
          trajectory: {
            data_points: [
              { time_s: 5.0, heart_rate_bpm: 72, mean_arterial_pressure_mmhg: 85 },
              { time_s: 60.0, heart_rate_bpm: 72, mean_arterial_pressure_mmhg: 85 },
              { time_s: 240.0, heart_rate_bpm: 71, mean_arterial_pressure_mmhg: 86 },
              { time_s: 390.0, heart_rate_bpm: 70, mean_arterial_pressure_mmhg: 88 },
            ],
          },
        },
      };

      const aligned = alignComparisonTrajectories([s1, s2]);
      assert.equal(aligned.commonInterval[0], 5.0);
      assert.equal(aligned.commonInterval[1], 240.0);
      assert.equal(aligned.maxDuration, 390.0);

      // Verify at t = 390.0, strat_saline is null (not extrapolated) while strat_mixed has value
      const pt390 = aligned.alignedPoints.find((p) => p.time_s === 390.0);
      assert.ok(pt390);
      assert.equal(pt390.map_strat_saline, null);
      assert.equal(pt390.map_strat_mixed, 88);
    });
  });

  describe('5. Mock API Execution with Mixed Actions', () => {
    const originalFetch = globalThis.fetch;

    afterEach(() => {
      globalThis.fetch = originalFetch;
    });

    test('Executes mixed scenario and verifies deltas reflect both fluid volume and vasoconstriction', async () => {
      const mockResponse = {
        metadata: {
          scenario_name: 'SeqMixed_Saline100_Norepi14ug_Live',
          patient_file: 'StandardMale.xml',
          is_cached: true,
          wall_clock_seconds: 0.05,
          simulated_duration_s: 390.0,
          row_count: 78,
          actions: [
            {
              action_index: 0,
              type: 'saline',
              bag_volume_ml: 100,
              start_time_s: 60,
              delivery_duration_s: 120,
            },
            {
              action_index: 1,
              type: 'norepinephrine',
              substance: 'Norepinephrine',
              concentration_ug_ml: 14.0,
              rate_ml_per_min: 1.0,
              duration_s: 120,
              start_time_s: 210,
              delivery_duration_s: 120,
            },
          ],
        },
        baseline_snapshot: {
          target_time_s: 60.0,
          actual_time_s: 60.02,
          heart_rate_bpm: 72.1,
          mean_arterial_pressure_mmhg: 85.3,
          blood_volume_l: 5.25,
          cardiac_output_l_min: 5.62,
        },
        intermediate_milestones: [
          {
            phase: 'intervention_0',
            description: 'Saline Bolus 1 Complete',
            target_time_s: 180.0,
            actual_time_s: 180.02,
            snapshot: {
              target_time_s: 180.0,
              actual_time_s: 180.02,
              heart_rate_bpm: 71.8,
              mean_arterial_pressure_mmhg: 85.9,
              blood_volume_l: 5.35,
              cardiac_output_l_min: 5.75,
            },
          },
          {
            phase: 'intervention_1',
            description: 'Norepinephrine Infusion Complete',
            target_time_s: 330.0,
            actual_time_s: 330.02,
            snapshot: {
              target_time_s: 330.0,
              actual_time_s: 330.02,
              heart_rate_bpm: 69.8,
              mean_arterial_pressure_mmhg: 88.7,
              blood_volume_l: 5.35,
              cardiac_output_l_min: 5.68,
            },
          },
        ],
        final_snapshot: {
          target_time_s: 390.0,
          actual_time_s: 390.02,
          heart_rate_bpm: 71.0,
          mean_arterial_pressure_mmhg: 86.8,
          blood_volume_l: 5.35,
          cardiac_output_l_min: 5.65,
        },
      };

      globalThis.fetch = async (url, options) => {
        const body = JSON.parse(options.body);
        assert.equal(body.actions.length, 2);
        assert.equal(body.actions[0].type, 'saline');
        assert.equal(body.actions[1].type, 'norepinephrine');

        return {
          ok: true,
          status: 200,
          json: async () => mockResponse,
        };
      };

      const result = await executeSequentialWhatIf({
        actions: [
          { type: 'saline', volume_ml: 100, start_time_s: 60 },
          {
            type: 'norepinephrine',
            concentration_ug_ml: 14.0,
            rate_ml_per_min: 1.0,
            duration_s: 120.0,
            start_time_s: 210,
          },
        ],
        baselineObservationS: 60,
        postObservationS: 60,
      });

      assert.ok(result);
      assert.equal(result.metadata.scenario_name, 'SeqMixed_Saline100_Norepi14ug_Live');

      // Verify physiological deltas
      const overallDeltas = computePhysiologicalDeltas(result.baseline_snapshot, result.final_snapshot);
      const mapDelta = overallDeltas.find((d) => d.key === 'mean_arterial_pressure_mmhg');
      const bvDelta = overallDeltas.find((d) => d.key === 'blood_volume_l');

      assert.ok(mapDelta);
      assert.ok(bvDelta);
      assert.equal(mapDelta.delta.toFixed(1), '1.5'); // 86.8 - 85.3 = +1.5 mmHg
      assert.equal(bvDelta.delta.toFixed(2), '0.10');  // +100 mL blood volume
    });
  });
});
