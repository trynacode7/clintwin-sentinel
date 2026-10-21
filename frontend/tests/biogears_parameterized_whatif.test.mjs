/**
 * biogears_parameterized_whatif.test.mjs ? Parameterized What-If Sandbox Tests (Step K2)
 * ======================================================================================
 * Tests:
 *   1. Parameterized saline bounds (50 to 1000 mL, 10 to 100 mL/min).
 *   2. Rejection of out-of-bounds volume and rate.
 *   3. Delivery duration calculations (250mL @ 50mL/min = 300s; 250mL @ 25mL/min = 600s).
 *   4. Timing semantics & suggested start times:
 *      - Overlap error when second action starts early.
 *      - Recalculation of suggested earliest valid start time.
 *      - Exact boundary acceptance when start equals preceding action completion.
 *   5. Disabled vs enabled norepinephrine intervention handling.
 *   6. Multi-action payload construction with configurable rates.
 *   7. Summary text and timeline markers.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateDeliveryDuration,
  validateSequentialSimulation,
  buildSequentialPayload,
  alignComparisonTrajectories,
  formatStrategyActionSummary,
  MIN_SALINE_VOLUME_ML,
  MAX_SALINE_VOLUME_ML,
  MIN_SALINE_RATE_ML_PER_MIN,
  MAX_SALINE_RATE_ML_PER_MIN,
  DEFAULT_SALINE_VOLUME_ML,
  DEFAULT_SALINE_RATE_ML_PER_MIN,
  VERIFIED_NOREPINEPHRINE_CONCENTRATION_UG_ML,
  VERIFIED_NOREPINEPHRINE_RATE_ML_PER_MIN,
  VERIFIED_NOREPINEPHRINE_DURATION_S,
} from '../src/services/biogearsApi.js';

describe('1. Parameterized Delivery Duration Math', () => {
  test('Calculates accurate duration for 250 mL @ 50 mL/min -> 300s', () => {
    assert.equal(calculateDeliveryDuration(250, 50), 300);
    assert.equal(
      calculateDeliveryDuration({ type: 'saline', volume_ml: 250, rate_ml_per_min: 50 }),
      300
    );
  });

  test('Calculates accurate duration for 250 mL @ 25 mL/min -> 600s', () => {
    assert.equal(calculateDeliveryDuration(250, 25), 600);
    assert.equal(
      calculateDeliveryDuration({ type: 'saline', volume_ml: 250, rate_ml_per_min: 25 }),
      600
    );
  });

  test('Calculates accurate duration for 1000 mL @ 100 mL/min -> 600s', () => {
    assert.equal(calculateDeliveryDuration(1000, 100), 600);
  });

  test('Returns fixed 120s for norepinephrine regardless of rate parameter', () => {
    assert.equal(
      calculateDeliveryDuration({ type: 'norepinephrine', duration_s: 120 }),
      120
    );
  });
});

describe('2. Parameterized Saline Validation Bounds', () => {
  test('Accepts valid volume boundaries [50, 1000] mL', () => {
    for (const vol of [50, 100, 250, 500, 750, 1000]) {
      const res = validateSequentialSimulation({
        actions: [{ type: 'saline', volume_ml: vol, rate_ml_per_min: 50, start_time_s: 60 }],
        parameterized: true,
      });
      assert.equal(res.valid, true, `Volume ${vol} should be valid`);
    }
  });

  test('Rejects volume below 50 mL or above 1000 mL', () => {
    for (const badVol of [0, 49, -10, 1001, 2000]) {
      const res = validateSequentialSimulation({
        actions: [{ type: 'saline', volume_ml: badVol, rate_ml_per_min: 50, start_time_s: 60 }],
        parameterized: true,
      });
      assert.equal(res.valid, false);
      assert.match(res.error, /Invalid saline volume|Saline volume must be a positive number/);
    }
  });

  test('Accepts valid infusion rate boundaries [10, 100] mL/min', () => {
    for (const rate of [10, 25, 50, 75, 100]) {
      const res = validateSequentialSimulation({
        actions: [{ type: 'saline', volume_ml: 250, rate_ml_per_min: rate, start_time_s: 60 }],
        parameterized: true,
      });
      assert.equal(res.valid, true, `Rate ${rate} should be valid`);
    }
  });

  test('Rejects infusion rate below 10 or above 100 mL/min', () => {
    for (const badRate of [0, 9, -5, 101, 150]) {
      const res = validateSequentialSimulation({
        actions: [{ type: 'saline', volume_ml: 250, rate_ml_per_min: badRate, start_time_s: 60 }],
        parameterized: true,
      });
      assert.equal(res.valid, false);
      assert.match(res.error, /Invalid saline infusion rate/);
    }
  });
});

describe('3. Timing Semantics, Overlap Detection & Suggested Start Times', () => {
  test('Detects overlap for 250mL @ 50mL/min starting at 60s when Norepi requested at 210s', () => {
    // Saline runs 60..360s. Norepi requested at 210s -> overlap by 150s!
    const res = validateSequentialSimulation({
      actions: [
        { type: 'saline', volume_ml: 250, rate_ml_per_min: 50, start_time_s: 60 },
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 120,
          start_time_s: 210,
        },
      ],
      parameterized: true,
    });

    assert.equal(res.valid, false);
    assert.match(res.error, /Infusion 2 start time \(210s\) precedes previous bolus completion \(360s\)/);
    assert.equal(res.suggestedStartTimes[1], 360, 'Suggests earliest technically valid start of 360s');
  });

  test('Accepts exact completion boundary at 360s without overlap error', () => {
    const res = validateSequentialSimulation({
      actions: [
        { type: 'saline', volume_ml: 250, rate_ml_per_min: 50, start_time_s: 60 },
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 120,
          start_time_s: 360,
        },
      ],
      parameterized: true,
    });

    assert.equal(res.valid, true);
    assert.equal(res.error, null);
    // 60 baseline + 300 delivery + 120 norepi + 60 post = 540s
    assert.equal(res.totalDuration, 540);
  });

  test('Recalculates suggested start time when saline rate changes', () => {
    // If saline rate changed to 25 mL/min, duration becomes 600s, ends at 660s.
    const res = validateSequentialSimulation({
      actions: [
        { type: 'saline', volume_ml: 250, rate_ml_per_min: 25, start_time_s: 60 },
        {
          type: 'norepinephrine',
          concentration_ug_ml: 14.0,
          rate_ml_per_min: 1.0,
          duration_s: 120,
          start_time_s: 210,
        },
      ],
      parameterized: true,
    });

    assert.equal(res.valid, false);
    assert.equal(res.suggestedStartTimes[1], 660, 'Recalculates suggested start time to 660s');
  });

  test('Saline-only scenario executes cleanly when norepinephrine is omitted/disabled', () => {
    const res = validateSequentialSimulation({
      actions: [
        { type: 'saline', volume_ml: 250, rate_ml_per_min: 50, start_time_s: 60 },
      ],
      parameterized: true,
    });

    assert.equal(res.valid, true);
    assert.equal(res.error, null);
    // 60 baseline + 300 delivery + 60 post = 420s
    assert.equal(res.totalDuration, 420);
  });
});

describe('4. Payload Construction and Summary Formatting', () => {
  test('buildSequentialPayload maps custom saline rate correctly', () => {
    const payload = buildSequentialPayload([
      { type: 'saline', volume_ml: 250, rate_ml_per_min: 25, start_time_s: 60 },
    ]);
    assert.equal(payload.actions[0].volume_ml, 250);
    assert.equal(payload.actions[0].rate_ml_per_min, 25);
    assert.equal(payload.actions[0].start_time_s, 60);
  });

  test('formatStrategyActionSummary reflects action order and types', () => {
    const summary = formatStrategyActionSummary([
      { type: 'saline', volume_ml: 250, start_time_s: 60 },
      { type: 'norepinephrine', start_time_s: 360 },
    ]);
    assert.equal(summary, '250mL Saline @ 60s → Norepi (14µg/mL) @ 360s');
  });

  test('alignComparisonTrajectories generates correct timeline markers for sandbox', () => {
    const fakeScenario = {
      id: 'strat_sandbox',
      name: 'Sandbox',
      color: '#06b6d4',
      actions: [
        { type: 'saline', volume_ml: 250, rate_ml_per_min: 50, start_time_s: 60 },
        { type: 'norepinephrine', start_time_s: 360 },
      ],
      result: {
        trajectory: {
          data_points: [
            { time_s: 0, heart_rate_bpm: 72, mean_arterial_pressure_mmhg: 85 },
            { time_s: 500, heart_rate_bpm: 75, mean_arterial_pressure_mmhg: 90 },
          ],
        },
      },
    };

    const aligned = alignComparisonTrajectories([fakeScenario]);
    assert.equal(aligned.interventionMarkers.length, 2);
    assert.equal(aligned.interventionMarkers[0].time_s, 60);
    assert.equal(aligned.interventionMarkers[0].type, 'saline');
    assert.equal(aligned.interventionMarkers[1].time_s, 360);
    assert.equal(aligned.interventionMarkers[1].type, 'norepinephrine');
  });
});
