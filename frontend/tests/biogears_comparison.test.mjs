/**
 * biogears_comparison.test.mjs — Step H Scenario Comparison Workspace Tests
 * =========================================================================
 * Tests:
 *   1. Strategy Configuration & Multi-Strategy Validation
 *   2. Trajectory Time-Alignment across Different Durations (No Extrapolation)
 *   3. Scenario Summary Table Computation (Initial, Final, Absolute Deltas, Cache Status)
 *   4. Reference Scenario Relative Difference Analysis
 *   5. Real BioGears Fixture Verification (1-Bolus, 2-Bolus, 3-Bolus)
 *   6. Partial Failure Resilience (Completed Scenarios Preserved on Error)
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
  alignComparisonTrajectories,
  computeScenarioSummaryTable,
  computeReferenceDifferences,
  COMPARISON_PALETTE,
} from '../src/services/biogearsApi.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURES_DIR = path.join(__dirname, 'fixtures');

// Load real BioGears response fixtures
const real1Bolus = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'real_1bolus_response.json'), 'utf8'));
const real2Bolus = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'real_2bolus_response.json'), 'utf8'));
const real3Bolus = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'real_3bolus_response.json'), 'utf8'));

describe('1. Strategy Configuration & Validation', () => {
  test('Validates 1-, 2-, and 3-bolus strategies using supported volumes', () => {
    // Strategy A: 1x 100mL
    const stratA = validateSequentialSimulation({
      actions: [{ volume_ml: 100, start_time_s: 60 }],
      baselineObservationS: 60,
      postObservationS: 60,
    });
    assert.equal(stratA.valid, true);
    assert.equal(stratA.totalDuration, 240); // 60 + 120 + 60

    // Strategy B: 2x 250mL
    const stratB = validateSequentialSimulation({
      actions: [
        { volume_ml: 250, start_time_s: 60 },
        { volume_ml: 250, start_time_s: 390 }, // 60 + 300 = 360, start @ 390
      ],
      baselineObservationS: 60,
      postObservationS: 60,
    });
    assert.equal(stratB.valid, true);
    assert.equal(stratB.totalDuration, 750); // 390 + 300 + 60

    // Strategy C: 3x 100mL
    const stratC = validateSequentialSimulation({
      actions: [
        { volume_ml: 100, start_time_s: 60 },
        { volume_ml: 100, start_time_s: 195 },
        { volume_ml: 100, start_time_s: 330 },
      ],
      baselineObservationS: 60,
      postObservationS: 60,
    });
    assert.equal(stratC.valid, true);
    assert.equal(stratC.totalDuration, 510); // 330 + 120 + 60
  });

  test('Rejects invalid volumes in comparison strategies', () => {
    const invalid = validateSequentialSimulation({
      actions: [{ volume_ml: 350, start_time_s: 60 }],
    });
    assert.equal(invalid.valid, false);
    assert.match(invalid.error, /Invalid saline volume/);
  });

  test('Rejects overlapping boluses within any comparison strategy', () => {
    const overlapping = validateSequentialSimulation({
      actions: [
        { volume_ml: 250, start_time_s: 60 }, // runs 60..360
        { volume_ml: 100, start_time_s: 300 }, // overlaps prior delivery!
      ],
    });
    assert.equal(overlapping.valid, false);
    assert.match(overlapping.error, /precedes previous bolus completion/);
  });

  test('Enforces maximum 1800s simulation duration limit', () => {
    const exceeds = validateSequentialSimulation({
      actions: [
        { volume_ml: 500, start_time_s: 60 },   // 60..660
        { volume_ml: 500, start_time_s: 800 },  // 800..1400
        { volume_ml: 500, start_time_s: 1500 }, // 1500..2100 -> >1800s
      ],
    });
    assert.equal(exceeds.valid, false);
    assert.match(exceeds.error, /exceeding the 1800s maximum limit/);
  });
});

describe('2. Trajectory Time-Alignment across Different Durations', () => {
  const scenarioShort = {
    id: 'strat_short',
    name: 'Strategy Short',
    color: '#10b981',
    result: {
      metadata: { scenario_name: 'Short' },
      trajectory: {
        data_points: [
          { time_s: 0, heart_rate_bpm: 72.0, mean_arterial_pressure_mmhg: 85.0, blood_volume_l: 5.0, cardiac_output_l_min: 5.5 },
          { time_s: 5, heart_rate_bpm: 72.2, mean_arterial_pressure_mmhg: 85.1, blood_volume_l: 5.05, cardiac_output_l_min: 5.52 },
          { time_s: 10, heart_rate_bpm: 72.5, mean_arterial_pressure_mmhg: 85.3, blood_volume_l: 5.1, cardiac_output_l_min: 5.55 },
        ],
      },
    },
  };

  const scenarioLong = {
    id: 'strat_long',
    name: 'Strategy Long',
    color: '#0284c7',
    result: {
      metadata: { scenario_name: 'Long' },
      trajectory: {
        data_points: [
          { time_s: 0, heart_rate_bpm: 72.0, mean_arterial_pressure_mmhg: 85.0, blood_volume_l: 5.0, cardiac_output_l_min: 5.5 },
          { time_s: 5, heart_rate_bpm: 73.0, mean_arterial_pressure_mmhg: 86.0, blood_volume_l: 5.1, cardiac_output_l_min: 5.6 },
          { time_s: 10, heart_rate_bpm: 73.5, mean_arterial_pressure_mmhg: 87.0, blood_volume_l: 5.2, cardiac_output_l_min: 5.7 },
          { time_s: 15, heart_rate_bpm: 74.0, mean_arterial_pressure_mmhg: 88.0, blood_volume_l: 5.3, cardiac_output_l_min: 5.8 },
          { time_s: 20, heart_rate_bpm: 74.5, mean_arterial_pressure_mmhg: 89.0, blood_volume_l: 5.4, cardiac_output_l_min: 5.9 },
        ],
      },
    },
  };

  test('Correctly computes common time interval and overall max duration', () => {
    const aligned = alignComparisonTrajectories([scenarioShort, scenarioLong]);
    assert.deepEqual(aligned.commonInterval, [0, 10]);
    assert.equal(aligned.maxDuration, 20);
    assert.equal(aligned.alignedPoints.length, 5);
  });

  test('Guarantees strictly no extrapolation beyond individual scenario duration', () => {
    const aligned = alignComparisonTrajectories([scenarioShort, scenarioLong]);
    const pts = aligned.alignedPoints;

    // At t = 5 (within both): both have numbers
    const pt5 = pts.find((p) => p.time_s === 5);
    assert.equal(pt5.isCommonInterval, true);
    assert.equal(pt5.hr_strat_short, 72.2);
    assert.equal(pt5.hr_strat_long, 73.0);

    // At t = 15 (beyond strat_short maxTime of 10): strat_short MUST be null!
    const pt15 = pts.find((p) => p.time_s === 15);
    assert.equal(pt15.isCommonInterval, false);
    assert.equal(pt15.hr_strat_short, null);
    assert.equal(pt15.map_strat_short, null);
    assert.equal(pt15.bv_strat_short, null);
    assert.equal(pt15.co_strat_short, null);
    assert.equal(pt15.hr_strat_long, 74.0);
  });

  test('Handles empty and invalid inputs gracefully', () => {
    const empty = alignComparisonTrajectories([]);
    assert.deepEqual(empty.alignedPoints, []);
    assert.deepEqual(empty.commonInterval, [0, 0]);

    const nulls = alignComparisonTrajectories([null, undefined, {}]);
    assert.deepEqual(nulls.alignedPoints, []);
  });
});

describe('3. Scenario Summary Table Computation', () => {
  test('Extracts initial/final vitals, signed absolute changes, and metadata', () => {
    const scenarios = [
      {
        id: 'strat_1',
        name: 'Conservative 100mL',
        status: 'completed',
        color: '#10b981',
        actions: [{ volume_ml: 100, start_time_s: 60 }],
        wallClockSeconds: 1.2,
        result: real1Bolus,
      },
      {
        id: 'strat_2',
        name: 'Moderate 2-Bolus',
        status: 'completed',
        color: '#0284c7',
        actions: [
          { volume_ml: 100, start_time_s: 60 },
          { volume_ml: 250, start_time_s: 210 },
        ],
        wallClockSeconds: 0.05,
        result: real2Bolus,
      },
    ];

    const table = computeScenarioSummaryTable(scenarios);
    assert.equal(table.length, 2);

    const row1 = table[0];
    assert.equal(row1.id, 'strat_1');
    assert.equal(row1.numInterventions, 1);
    assert.equal(row1.totalVolumeMl, 100);
    assert.equal(row1.isCached, true);
    assert.equal(row1.source, 'cache');
    assert.equal(typeof row1.vitals.hr.initial, 'number');
    assert.equal(typeof row1.vitals.hr.final, 'number');
    assert.equal(typeof row1.vitals.hr.delta, 'number');

    // Delta formula check: delta === final - initial
    const hrDeltaExpected = row1.vitals.hr.final - row1.vitals.hr.initial;
    assert.ok(Math.abs(row1.vitals.hr.delta - hrDeltaExpected) < 1e-6);

    const bvDeltaExpected = row1.vitals.bv.final - row1.vitals.bv.initial;
    assert.ok(Math.abs(row1.vitals.bv.delta - bvDeltaExpected) < 1e-6);
  });

  test('Marks unavailable measurements gracefully without NaN', () => {
    const partialScenario = [
      {
        id: 'strat_partial',
        name: 'Partial Data',
        status: 'completed',
        result: {
          metadata: {
            scenario_name: 'Partial',
            actions: [],
            is_cached: false,
            source: 'live',
            wall_clock_seconds: 145.2,
          },
          baseline_snapshot: {
            heart_rate_bpm: 80,
            mean_arterial_pressure_mmhg: null, // missing MAP
            blood_volume_l: 5.0,
            cardiac_output_l_min: 5.0,
          },
          final_snapshot: {
            heart_rate_bpm: null, // missing final HR
            mean_arterial_pressure_mmhg: 90,
            blood_volume_l: 5.2,
            cardiac_output_l_min: 5.2,
          },
        },
      },
    ];

    const table = computeScenarioSummaryTable(partialScenario);
    const row = table[0];
    assert.equal(row.vitals.hr.initial, 80);
    assert.equal(row.vitals.hr.final, null);
    assert.equal(row.vitals.hr.delta, null); // Cannot compute delta if final is null

    assert.equal(row.vitals.map.initial, null);
    assert.equal(row.vitals.map.final, 90);
    assert.equal(row.vitals.map.delta, null);
  });
});

describe('4. Reference Scenario Relative Difference Analysis', () => {
  test('Computes relative differences against reference scenario', () => {
    const summaryRows = [
      {
        id: 'ref',
        name: 'Reference Strategy',
        status: 'completed',
        vitals: {
          hr: { initial: 72.0, final: 70.0, delta: -2.0, unit: 'bpm', precision: 1 },
          map: { initial: 85.0, final: 87.0, delta: 2.0, unit: 'mmHg', precision: 1 },
          bv: { initial: 5.0, final: 5.1, delta: 0.1, unit: 'L', precision: 2 },
          co: { initial: 5.5, final: 5.6, delta: 0.1, unit: 'L/min', precision: 2 },
        },
      },
      {
        id: 'comp_1',
        name: 'High Volume Strategy',
        status: 'completed',
        vitals: {
          hr: { initial: 72.0, final: 68.0, delta: -4.0, unit: 'bpm', precision: 1 },
          map: { initial: 85.0, final: 91.0, delta: 6.0, unit: 'mmHg', precision: 1 },
          bv: { initial: 5.0, final: 5.4, delta: 0.4, unit: 'L', precision: 2 },
          co: { initial: 5.5, final: 5.9, delta: 0.4, unit: 'L/min', precision: 2 },
        },
      },
    ];

    const diffAnalysis = computeReferenceDifferences(summaryRows, 'ref');
    assert.ok(diffAnalysis);
    assert.equal(diffAnalysis.referenceScenario.id, 'ref');

    // Self-reference comparison yields 0
    const refDiff = diffAnalysis.comparisonRows.find((r) => r.id === 'ref');
    assert.equal(refDiff.isReference, true);
    assert.equal(refDiff.differences.hr.finalDiff, 0);

    // Comparator difference
    const compDiff = diffAnalysis.comparisonRows.find((r) => r.id === 'comp_1');
    assert.equal(compDiff.isReference, false);
    // HR final diff: 68 - 70 = -2
    assert.equal(compDiff.differences.hr.finalDiff, -2);
    // HR delta diff: -4 - (-2) = -2
    assert.equal(compDiff.differences.hr.deltaDiff, -2);
    // MAP final diff: 91 - 87 = +4
    assert.equal(compDiff.differences.map.finalDiff, 4);
    // BV final diff: 5.4 - 5.1 = +0.3
    assert.ok(Math.abs(compDiff.differences.bv.finalDiff - 0.3) < 1e-6);
  });

  test('Returns null safely if reference scenario is not found', () => {
    const analysis = computeReferenceDifferences([], 'non_existent');
    assert.equal(analysis, null);
  });
});

describe('5. Real BioGears Fixture 3-Scenario Comparison', () => {
  test('Simulates side-by-side 1-, 2-, and 3-bolus BioGears trajectory alignment', () => {
    const scenarios = [
      { id: 's1', name: '1-Bolus (100mL)', color: COMPARISON_PALETTE[0].color, result: real1Bolus },
      { id: 's2', name: '2-Bolus (100+250mL)', color: COMPARISON_PALETTE[1].color, result: real2Bolus },
      { id: 's3', name: '3-Bolus (3x100mL)', color: COMPARISON_PALETTE[2].color, result: real3Bolus },
    ];

    const aligned = alignComparisonTrajectories(scenarios);
    assert.equal(aligned.scenariosMeta.length, 3);
    assert.ok(aligned.alignedPoints.length > 30);

    // Common interval should be bounded by the shortest scenario (real1Bolus)
    const [cMin, cMax] = aligned.commonInterval;
    assert.equal(cMin, real1Bolus.trajectory.data_points[0].time_s);
    assert.equal(cMax, real1Bolus.trajectory.data_points[real1Bolus.trajectory.data_points.length - 1].time_s);

    // Overall duration is bounded by the longest scenario across all 3
    const expectedMaxDuration = Math.max(...[real1Bolus, real2Bolus, real3Bolus].map(s => s.trajectory.data_points[s.trajectory.data_points.length - 1].time_s));
    assert.equal(aligned.maxDuration, expectedMaxDuration);

    // Check points before 240s have valid values across all 3
    const midPoint = aligned.alignedPoints.find((p) => Math.abs(p.time_s - 120) < 2.5);
    assert.ok(midPoint);
    assert.equal(midPoint.isCommonInterval, true);
    assert.ok(typeof midPoint.hr_s1 === 'number');
    assert.ok(typeof midPoint.hr_s2 === 'number');
    assert.ok(typeof midPoint.hr_s3 === 'number');

    // Check points beyond 240s have s1 as null while longer scenarios continue
    const latePoint = aligned.alignedPoints.find((p) => p.time_s > 300 && p.time_s < 400);
    assert.ok(latePoint);
    assert.equal(latePoint.isCommonInterval, false);
    assert.equal(latePoint.hr_s1, null);
    assert.ok(typeof latePoint.hr_s2 === 'number' || typeof latePoint.hr_s3 === 'number');
  });
});

describe('6. Partial Failure Resilience', () => {
  test('Preserves completed scenarios when another scenario fails', () => {
    const mixedScenarios = [
      {
        id: 'strat_ok_1',
        name: 'Strategy Success 1',
        color: '#10b981',
        status: 'completed',
        result: real1Bolus,
      },
      {
        id: 'strat_failed',
        name: 'Strategy Failure',
        color: '#f59e0b',
        status: 'error',
        error: 'Simulation Timeout (HTTP 504)',
        result: null,
      },
      {
        id: 'strat_ok_2',
        name: 'Strategy Success 2',
        color: '#0284c7',
        status: 'completed',
        result: real2Bolus,
      },
    ];

    // Trajectory alignment filters out failed scenario and aligns the successful ones
    const aligned = alignComparisonTrajectories(mixedScenarios);
    assert.equal(aligned.scenariosMeta.length, 2);
    assert.equal(aligned.scenariosMeta[0].id, 'strat_ok_1');
    assert.equal(aligned.scenariosMeta[1].id, 'strat_ok_2');

    // Summary table presents all 3 with their respective status
    const table = computeScenarioSummaryTable(mixedScenarios);
    assert.equal(table.length, 3);
    assert.equal(table[0].status, 'completed');
    assert.equal(table[1].status, 'error');
    assert.equal(table[1].error, 'Simulation Timeout (HTTP 504)');
    assert.equal(table[2].status, 'completed');
  });
});
