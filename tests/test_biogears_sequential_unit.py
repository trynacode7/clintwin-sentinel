"""
test_biogears_sequential_unit.py — Unit tests for sequential execution & state extraction (Step C)
==================================================================================================
SCOPE: Fast, offline unit tests for backend/biogears_sequential_execution.py.
       No BioGears installation required. Tests synthetic fixtures and saved Step B CSVs.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.biogears_bridge import (
    BioGearsColumn,
    BioGearsOutputParseError,
    BioGearsProcessError,
    BioGearsResult,
    BioGearsTimeSeries,
    BioGearsTimeoutError,
)
from backend.biogears_scenario_builder import (
    BioGearsNorepinephrineInfusion,
    BioGearsSalineBolus,
    BioGearsScenarioBuilderError,
    BioGearsSequentialScenario,
)
from backend.biogears_sequential_execution import (
    DEFAULT_TIMESTAMP_TOLERANCE_S,
    BioGearsExtractionError,
    PhysiologicalSnapshot,
    execute_sequential_saline_scenario,
    extract_scenario_snapshots,
    extract_snapshot_at_time,
    extract_snapshots_from_csv,
    verify_temporal_continuity,
)


# ─── Fixtures & Helpers ───────────────────────────────────────────────────────


def _make_mock_biogears_result(
    times: list,
    bv_values: list = None,
    map_values: list = None,
    hr_values: list = None,
    co_values: list = None,
    include_optional: bool = True,
) -> BioGearsResult:
    """Helper to synthesize a mock BioGearsResult for offline unit testing."""
    n = len(times)
    bv = bv_values if bv_values is not None else [5.54] * n
    map_ = map_values if map_values is not None else [95.9] * n
    hr = hr_values if hr_values is not None else [72.3] * n
    co = co_values if co_values is not None else [5.81] * n

    def _ts(name, unit, vals):
        col = BioGearsColumn(raw_header=f"{name}({unit})" if unit else name, name=name, unit=unit)
        return BioGearsTimeSeries(column=col, times_s=times, values=vals)

    columns = {
        "BloodVolume(L)": _ts("BloodVolume", "L", bv),
        "MeanArterialPressure(mmHg)": _ts("MeanArterialPressure", "mmHg", map_),
        "HeartRate(1/min)": _ts("HeartRate", "1/min", hr),
        "CardiacOutput(L/min)": _ts("CardiacOutput", "L/min", co),
    }

    if include_optional:
        columns["SystolicArterialPressure(mmHg)"] = _ts("SystolicArterialPressure", "mmHg", [115.0] * n)
        columns["DiastolicArterialPressure(mmHg)"] = _ts("DiastolicArterialPressure", "mmHg", [75.0] * n)
        columns["OxygenSaturation"] = _ts("OxygenSaturation", "", [0.98] * n)
        columns["RespirationRate(1/min)"] = _ts("RespirationRate", "1/min", [16.0] * n)

    final_state = {k: v.values[-1] for k, v in columns.items() if v.values}

    return BioGearsResult(
        scenario_name="MockScenario",
        time_s=times,
        columns=columns,
        final_state=final_state,
        row_count=len(times),
        time_range_s=(times[0], times[-1]) if times else (0.0, 0.0),
        returncode=0,
        output_csv_path="mock_results.csv",
    )


# ─── Snapshot Extraction: Timestamp Matching & Tolerances (Step C) ───────────


class TestTimestampMatchingAndTolerance:
    def test_exact_timestamp_match(self):
        times = [5.0, 30.0, 60.0, 90.0, 120.0]
        res = _make_mock_biogears_result(times)
        snap = extract_snapshot_at_time(res, target_time_s=60.0)
        assert snap.target_time_s == 60.0
        assert snap.actual_time_s == 60.0
        assert snap.time_s == 60.0

    def test_small_timestamp_offset_handled(self):
        # BioGears outputs 60.02s for a 60.0s target due to phase offset
        times = [5.02, 30.02, 60.02, 90.02, 120.02]
        res = _make_mock_biogears_result(times)
        snap = extract_snapshot_at_time(res, target_time_s=60.0, tolerance_s=5.0)
        assert snap.target_time_s == 60.0
        assert snap.actual_time_s == 60.02

    def test_nearest_sample_selection(self):
        times = [10.0, 20.0, 30.0]
        res = _make_mock_biogears_result(times)
        # 18.0 is closer to 20.0 than 10.0
        snap1 = extract_snapshot_at_time(res, target_time_s=18.0)
        assert snap1.actual_time_s == 20.0

        # 12.0 is closer to 10.0 than 20.0
        snap2 = extract_snapshot_at_time(res, target_time_s=12.0)
        assert snap2.actual_time_s == 10.0

    def test_sample_outside_allowed_tolerance_raises(self):
        times = [10.0, 20.0, 30.0]
        res = _make_mock_biogears_result(times)
        # Gap between 10 and 20 is 10s. Target 15.0 has diff 5.0. If tolerance is 2.0s, must raise.
        with pytest.raises(BioGearsExtractionError, match="exceeds allowed tolerance"):
            extract_snapshot_at_time(res, target_time_s=15.0, tolerance_s=2.0)

    def test_out_of_range_target_before_simulation_raises(self):
        times = [10.0, 20.0, 30.0]
        res = _make_mock_biogears_result(times)
        with pytest.raises(BioGearsExtractionError, match="outside available simulation interval"):
            extract_snapshot_at_time(res, target_time_s=2.0, tolerance_s=5.0)

    def test_out_of_range_target_after_simulation_raises(self):
        times = [10.0, 20.0, 30.0]
        res = _make_mock_biogears_result(times)
        with pytest.raises(BioGearsExtractionError, match="outside available simulation interval"):
            extract_snapshot_at_time(res, target_time_s=50.0, tolerance_s=5.0)


# ─── Snapshot Extraction: Validation & Required Columns ───────────────────────


class TestSnapshotValidationAndColumns:
    def test_missing_required_blood_volume_raises(self):
        times = [5.0, 10.0]
        res = _make_mock_biogears_result(times)
        del res.columns["BloodVolume(L)"]
        with pytest.raises(BioGearsExtractionError, match="Missing required physiological signal 'blood_volume'"):
            extract_snapshot_at_time(res, target_time_s=10.0)

    def test_missing_required_heart_rate_raises(self):
        times = [5.0, 10.0]
        res = _make_mock_biogears_result(times)
        del res.columns["HeartRate(1/min)"]
        with pytest.raises(BioGearsExtractionError, match="Missing required physiological signal 'heart_rate'"):
            extract_snapshot_at_time(res, target_time_s=10.0)

    def test_missing_required_map_raises(self):
        times = [5.0, 10.0]
        res = _make_mock_biogears_result(times)
        del res.columns["MeanArterialPressure(mmHg)"]
        with pytest.raises(BioGearsExtractionError, match="Missing required physiological signal 'mean_arterial_pressure'"):
            extract_snapshot_at_time(res, target_time_s=10.0)

    def test_missing_required_cardiac_output_raises(self):
        times = [5.0, 10.0]
        res = _make_mock_biogears_result(times)
        del res.columns["CardiacOutput(L/min)"]
        with pytest.raises(BioGearsExtractionError, match="Missing required physiological signal 'cardiac_output'"):
            extract_snapshot_at_time(res, target_time_s=10.0)

    def test_missing_optional_measurements_represented_as_none(self):
        times = [5.0, 10.0]
        res = _make_mock_biogears_result(times, include_optional=False)
        snap = extract_snapshot_at_time(res, target_time_s=10.0)
        assert snap.blood_volume_l == 5.54
        assert snap.heart_rate_bpm == 72.3
        # Optional measurements must be None, NOT fabricated as 0.0
        assert snap.systolic_bp_mmhg is None
        assert snap.diastolic_bp_mmhg is None
        assert snap.oxygen_saturation is None
        assert snap.respiration_rate_bpm is None

    def test_empty_result_raises(self):
        res = _make_mock_biogears_result([])
        with pytest.raises(BioGearsExtractionError, match="zero rows"):
            extract_snapshot_at_time(res, target_time_s=60.0)

    def test_non_monotonic_timestamps_raises(self):
        times = [5.0, 10.0, 8.0, 20.0]
        res = _make_mock_biogears_result(times)
        with pytest.raises(BioGearsExtractionError, match="Non-monotonic"):
            extract_snapshot_at_time(res, target_time_s=10.0)

    def test_malformed_nan_timestamp_raises(self):
        times = [5.0, float("nan"), 15.0]
        res = _make_mock_biogears_result(times)
        with pytest.raises(BioGearsExtractionError, match="Malformed timestamp"):
            extract_snapshot_at_time(res, target_time_s=5.0)


# ─── Multi-Action Scenario Milestone Extraction & Sequences (Step C) ──────────


class TestScenarioMilestoneSequences:
    def test_one_action_milestone_sequence(self):
        sc = BioGearsSequentialScenario(
            scenario_name="Seq1",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )
        # Expected sequence: baseline (60s), delivery_end 0 (180s), final_observation (240s)
        times = [float(t) for t in range(5, 245, 5)]
        res = _make_mock_biogears_result(times)
        report = extract_scenario_snapshots(res, sc)

        assert len(report.milestone_snapshots) == 3
        # Milestone 0: baseline (60s)
        assert report.milestone_snapshots[0].milestone_type == "baseline"
        assert report.milestone_snapshots[0].target_time_s == 60.0
        # Milestone 1: delivery_end (180s)
        assert report.milestone_snapshots[1].milestone_type == "delivery_end"
        assert report.milestone_snapshots[1].target_time_s == 180.0
        assert report.milestone_snapshots[1].action_index == 0
        # Milestone 2: final_observation (240s)
        assert report.milestone_snapshots[2].milestone_type == "final_observation"
        assert report.milestone_snapshots[2].target_time_s == 240.0

    def test_two_action_milestone_sequence(self):
        sc = BioGearsSequentialScenario(
            scenario_name="Seq2",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(100.0, start_time_s=240.0),
            ],
        )
        times = [float(t) for t in range(5, 425, 5)]
        res = _make_mock_biogears_result(times)
        report = extract_scenario_snapshots(res, sc)

        assert len(report.milestone_snapshots) == 5
        targets = [s.target_time_s for s in report.milestone_snapshots]
        assert targets == [60.0, 180.0, 240.0, 360.0, 420.0]

        types = [s.milestone_type for s in report.milestone_snapshots]
        assert types == [
            "baseline",
            "delivery_end",
            "intermediate_observation",
            "delivery_end",
            "final_observation",
        ]
        assert len(report.delivery_end_snapshots) == 2
        assert len(report.intermediate_observation_snapshots) == 1

    def test_three_action_milestone_sequence(self):
        """
        Verify the expected milestone sequence for a 3-action scenario:
          - baseline at 60 seconds
          - delivery ends at 180, 360, and 540 seconds
          - intermediate observations at 240 and 420 seconds
          - final observation at 600 seconds
        """
        sc = BioGearsSequentialScenario(
            scenario_name="Seq3",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(100.0, start_time_s=240.0),
                BioGearsSalineBolus(100.0, start_time_s=420.0),
            ],
        )
        times = [float(t) for t in range(5, 605, 5)]
        res = _make_mock_biogears_result(times)
        report = extract_scenario_snapshots(res, sc)

        assert len(report.milestone_snapshots) == 7
        targets = [s.target_time_s for s in report.milestone_snapshots]
        assert targets == [60.0, 180.0, 240.0, 360.0, 420.0, 540.0, 600.0]

        types = [s.milestone_type for s in report.milestone_snapshots]
        assert types == [
            "baseline",
            "delivery_end",
            "intermediate_observation",
            "delivery_end",
            "intermediate_observation",
            "delivery_end",
            "final_observation",
        ]

        # Verify delivery ends at 180, 360, and 540 seconds
        assert [s.target_time_s for s in report.delivery_end_snapshots] == [180.0, 360.0, 540.0]
        assert [s.action_index for s in report.delivery_end_snapshots] == [0, 1, 2]

        # Verify intermediate observations at 240 and 420 seconds
        assert [s.target_time_s for s in report.intermediate_observation_snapshots] == [240.0, 420.0]

        # Verify baseline at 60s and final observation at 600s
        assert report.baseline_snapshot.target_time_s == 60.0
        assert report.final_snapshot.target_time_s == 600.0


# ─── Validation Against Saved Step B Simulation Results (Step C) ──────────────


class TestExtractionAgainstSavedStepBResults:
    """Validate extraction directly against actual BioGears CSVs without rerunning engine."""

    RUNS_DIR = Path(__file__).parent.parent / "biogears_runs"

    def test_saved_1x100ml_results_extraction(self):
        csv_path = self.RUNS_DIR / "seq_saline_1x100ml_results.csv"
        if not csv_path.exists():
            pytest.skip(f"Saved CSV not found at {csv_path}")

        sc = BioGearsSequentialScenario(
            scenario_name="SeqSaline_1x100mL_Live",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )

        report = extract_snapshots_from_csv(csv_path, sc)
        assert len(report.milestone_snapshots) == 3

        # Baseline: 60.02s
        base = report.baseline_snapshot
        assert base.target_time_s == 60.0
        assert base.actual_time_s == pytest.approx(60.02, abs=0.01)
        assert base.blood_volume_l == pytest.approx(5.53, abs=0.02)
        assert base.mean_arterial_pressure_mmhg == pytest.approx(95.9, abs=0.5)

        # Delivery end 0: 180.02s
        action0 = report.delivery_end_snapshots[0]
        assert action0.target_time_s == 180.0
        assert action0.actual_time_s == pytest.approx(180.02, abs=0.01)
        assert action0.blood_volume_l == pytest.approx(5.63, abs=0.02)

        # Final observation: 240.02s
        final = report.final_snapshot
        assert final.target_time_s == 240.0
        assert final.actual_time_s == pytest.approx(240.02, abs=0.01)
        assert final.blood_volume_l == pytest.approx(5.63, abs=0.02)

    def test_saved_2x100ml_results_extraction(self):
        csv_path = self.RUNS_DIR / "seq_saline_2x100ml_results.csv"
        if not csv_path.exists():
            pytest.skip(f"Saved CSV not found at {csv_path}")

        sc = BioGearsSequentialScenario(
            scenario_name="SeqSaline_2x100mL_Live",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(100.0, start_time_s=240.0),
            ],
        )

        report = extract_snapshots_from_csv(csv_path, sc)
        assert len(report.milestone_snapshots) == 5

        # Cumulative volume expansion
        base_bv = report.baseline_snapshot.blood_volume_l
        act0_bv = report.delivery_end_snapshots[0].blood_volume_l
        act1_bv = report.delivery_end_snapshots[1].blood_volume_l
        final_bv = report.final_snapshot.blood_volume_l

        assert act0_bv > base_bv
        assert act1_bv > act0_bv
        assert (final_bv - base_bv) == pytest.approx(0.20, abs=0.03)

    def test_saved_3x100ml_results_extraction(self):
        csv_path = self.RUNS_DIR / "seq_saline_3x100ml_results.csv"
        if not csv_path.exists():
            pytest.skip(f"Saved CSV not found at {csv_path}")

        sc = BioGearsSequentialScenario(
            scenario_name="SeqSaline_3x100mL_Live",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(100.0, start_time_s=240.0),
                BioGearsSalineBolus(100.0, start_time_s=420.0),
            ],
        )

        report = extract_snapshots_from_csv(csv_path, sc)
        assert len(report.milestone_snapshots) == 7

        # Check exact target timeline sequence: 60, 180, 240, 360, 420, 540, 600
        targets = [s.target_time_s for s in report.milestone_snapshots]
        assert targets == [60.0, 180.0, 240.0, 360.0, 420.0, 540.0, 600.0]

        # Cumulative volume expansion across 3 sequential infusions
        base_bv = report.baseline_snapshot.blood_volume_l
        act0_bv = report.delivery_end_snapshots[0].blood_volume_l
        act1_bv = report.delivery_end_snapshots[1].blood_volume_l
        act2_bv = report.delivery_end_snapshots[2].blood_volume_l
        final_bv = report.final_snapshot.blood_volume_l

        assert act0_bv > base_bv
        assert act1_bv > act0_bv
        assert act2_bv > act1_bv
        assert (final_bv - base_bv) == pytest.approx(0.31, abs=0.04)


# ─── Temporal Continuity Tests ────────────────────────────────────────────────


class TestTemporalContinuity:
    def test_continuous_timeline_passes(self):
        times = [float(t) for t in range(5, 245, 5)]
        assert verify_temporal_continuity(times, expected_duration_s=240.0) is True

    def test_empty_timeline_raises(self):
        with pytest.raises(BioGearsOutputParseError, match="Empty time series"):
            verify_temporal_continuity([], expected_duration_s=240.0)

    def test_non_monotonic_timeline_raises(self):
        times = [5.0, 10.0, 15.0, 12.0, 20.0]
        with pytest.raises(BioGearsOutputParseError, match="Non-monotonic"):
            verify_temporal_continuity(times, expected_duration_s=20.0)

    def test_repeated_timestamp_raises(self):
        times = [5.0, 10.0, 10.0, 15.0]
        with pytest.raises(BioGearsOutputParseError, match="Non-monotonic"):
            verify_temporal_continuity(times, expected_duration_s=15.0)

    def test_truncated_timeline_raises(self):
        times = [float(t) for t in range(5, 125, 5)]
        with pytest.raises(BioGearsOutputParseError, match="truncated prematurely"):
            verify_temporal_continuity(times, expected_duration_s=240.0)


# ─── Execution Scope & Guard Tests ────────────────────────────────────────────


class TestExecutionGuards:
    def test_non_sequential_scenario_rejected(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="BioGearsSequentialScenario"):
            execute_sequential_saline_scenario("not_a_scenario")  # type: ignore[arg-type]

    def test_mixed_intervention_strictly_blocked(self):
        sc = BioGearsSequentialScenario(
            scenario_name="BlockedMixed",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsNorepinephrineInfusion(start_time_s=240.0),
            ],
        )
        with pytest.raises(BioGearsScenarioBuilderError, match=r"\[BLOCKED\]"):
            execute_sequential_saline_scenario(sc)

    def test_non_saline_action_rejected(self):
        sc = BioGearsSequentialScenario(
            scenario_name="NorepiOnlyInStepB",
            actions=[BioGearsNorepinephrineInfusion(start_time_s=60.0)],
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="saline-only"):
            execute_sequential_saline_scenario(sc)


# ─── Execution Failure Handling Tests ─────────────────────────────────────────


class TestExecutionFailureHandling:
    def test_timeout_propagates(self):
        sc = BioGearsSequentialScenario(
            scenario_name="TimeoutTest",
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )
        mock_bridge = MagicMock()
        mock_bridge.run_scenario_xml.side_effect = BioGearsTimeoutError("Timeout", timeout_seconds=10)

        with pytest.raises(BioGearsTimeoutError) as exc_info:
            execute_sequential_saline_scenario(sc, bridge=mock_bridge)
        assert exc_info.value.timeout_seconds == 10

    def test_process_error_propagates(self):
        sc = BioGearsSequentialScenario(
            scenario_name="ProcessErrorTest",
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )
        mock_bridge = MagicMock()
        mock_bridge.run_scenario_xml.side_effect = BioGearsProcessError(
            "Crash", returncode=1, stdout="err", stderr="err"
        )

        with pytest.raises(BioGearsProcessError) as exc_info:
            execute_sequential_saline_scenario(sc, bridge=mock_bridge)
        assert exc_info.value.returncode == 1

    def test_happy_path_mocked_execution(self):
        sc = BioGearsSequentialScenario(
            scenario_name="HappyPath",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )
        times = [float(t) for t in range(5, 245, 5)]
        bv = [5.54 + 0.000375 * t for t in times]
        map_ = [95.9 + 0.001 * t for t in times]
        hr = [72.3 - 0.005 * t for t in times]
        co = [5.81 + 0.0004 * t for t in times]

        mock_res = _make_mock_biogears_result(times, bv, map_, hr, co)
        mock_bridge = MagicMock()
        mock_bridge.config.executable = r"C:\BioGears\bin\bg-scenario.exe"
        mock_bridge.run_scenario_xml.return_value = mock_res

        exec_res = execute_sequential_saline_scenario(sc, bridge=mock_bridge)
        assert exec_res.exit_status == 0
        assert exec_res.row_count == len(times)
        assert exec_res.is_continuous is True
        assert exec_res.baseline_state.target_time_s == 60.0
        assert exec_res.final_state.target_time_s == 240.0
        assert len(exec_res.intermediate_states) == 3
