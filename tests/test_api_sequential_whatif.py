"""
test_api_sequential_whatif.py — Tests for Sequential BioGears What-If API (Phase D Step E)
==========================================================================================
Validates:
  1. Valid single-action 100 mL saline request.
  2. Valid two-action sequential saline request.
  3. Valid three-action sequential saline request.
  4. Response schema adherence and chronological milestone ordering.
  5. Correct mapping of optional missing measurements to JSON null.
  6. Cache-hit response with is_cached=True and source='cache'.
  7. Fresh-execution response with is_cached=False and source='live'.
  8. Validation rejection of unsupported volumes (e.g., 150 mL, 0 mL).
  9. Validation rejection of unsupported intervention types (e.g., 'norepinephrine').
  10. Validation rejection of empty actions or >3 actions.
  11. Validation rejection of overlapping or out-of-order action timelines.
  12. Validation rejection of scenarios exceeding 1800s total duration.
  13. BioGears timeout mapped to HTTP 504.
  14. BioGears engine process failure mapped to HTTP 503.
  15. Snapshot extraction failure mapped to HTTP 500 without fabricating vitals.
  16. Unexpected internal failures safely caught as HTTP 500.
  17. Preservation of existing API endpoints (GET /, GET /patients, GET /biogears/intervention).
  18. Complete serialization of all snapshot fields.
  19. Concurrency delegation to cache locking layer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.biogears_bridge import (
    BioGearsBridge,
    BioGearsColumn,
    BioGearsNotFoundError,
    BioGearsProcessError,
    BioGearsResult,
    BioGearsTimeoutError,
    BioGearsTimeSeries,
)
from backend.biogears_scenario_builder import (
    BioGearsSalineBolus,
    BioGearsSequentialScenario,
)
from backend.biogears_sequential_execution import (
    BioGearsExtractionError,
    PhysiologicalSnapshot,
    SequentialExecutionResult,
    extract_scenario_snapshots,
    verify_temporal_continuity,
)

SAVED_STEP_B_CSV = Path("biogears_runs/seq_saline_1x100ml_results.csv")


@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient."""
    with TestClient(app) as c:
        yield c


def _create_mock_execution_result(
    scenario: BioGearsSequentialScenario,
    is_cached: bool = False,
    fingerprint: str = "test_fingerprint_abc123",
    csv_path: Optional[Path] = None,
) -> SequentialExecutionResult:
    """Helper to build a deterministic in-memory SequentialExecutionResult test fixture."""
    if csv_path is not None and Path(csv_path).exists():
        bridge = BioGearsBridge()
        biogears_result = bridge._parse_csv(
            csv_path=str(csv_path),
            scenario_name=scenario.scenario_name,
            returncode=0,
        )
    else:
        # Deterministic in-memory synthetic fixture covering the entire configured duration
        total_duration = max(scenario.total_duration_s, 60.0)
        time_s = [float(t) for t in range(int(total_duration) + 5)]

        cols: Dict[str, BioGearsTimeSeries] = {
            "Time(s)": BioGearsTimeSeries(
                BioGearsColumn("Time(s)", "Time", "s"),
                time_s,
                list(time_s),
            ),
            "BloodVolume(L)": BioGearsTimeSeries(
                BioGearsColumn("BloodVolume(L)", "BloodVolume", "L"),
                time_s,
                [5.0] * len(time_s),
            ),
            "MeanArterialPressure(mmHg)": BioGearsTimeSeries(
                BioGearsColumn("MeanArterialPressure(mmHg)", "MeanArterialPressure", "mmHg"),
                time_s,
                [90.0] * len(time_s),
            ),
            "HeartRate(1/min)": BioGearsTimeSeries(
                BioGearsColumn("HeartRate(1/min)", "HeartRate", "1/min"),
                time_s,
                [75.0] * len(time_s),
            ),
            "CardiacOutput(L/min)": BioGearsTimeSeries(
                BioGearsColumn("CardiacOutput(L/min)", "CardiacOutput", "L/min"),
                time_s,
                [5.5] * len(time_s),
            ),
            "SystolicArterialPressure(mmHg)": BioGearsTimeSeries(
                BioGearsColumn("SystolicArterialPressure(mmHg)", "SystolicArterialPressure", "mmHg"),
                time_s,
                [120.0] * len(time_s),
            ),
            "DiastolicArterialPressure(mmHg)": BioGearsTimeSeries(
                BioGearsColumn("DiastolicArterialPressure(mmHg)", "DiastolicArterialPressure", "mmHg"),
                time_s,
                [80.0] * len(time_s),
            ),
            "OxygenSaturation": BioGearsTimeSeries(
                BioGearsColumn("OxygenSaturation", "OxygenSaturation", ""),
                time_s,
                [0.98] * len(time_s),
            ),
            "RespirationRate(1/min)": BioGearsTimeSeries(
                BioGearsColumn("RespirationRate(1/min)", "RespirationRate", "1/min"),
                time_s,
                [14.0] * len(time_s),
            ),
        }
        final_state = {k: v.values[-1] for k, v in cols.items()}
        biogears_result = BioGearsResult(
            scenario_name=scenario.scenario_name,
            time_s=time_s,
            columns=cols,
            final_state=final_state,
            row_count=len(time_s),
            time_range_s=(time_s[0], time_s[-1]),
            returncode=0,
            output_csv_path="mock_synthetic_test_fixture.csv",
        )

    is_continuous = verify_temporal_continuity(
        biogears_result.time_s,
        expected_duration_s=scenario.total_duration_s,
    )
    report = extract_scenario_snapshots(
        biogears_result=biogears_result,
        scenario=scenario,
    )
    milestones = scenario.get_timeline_milestones()
    intermediate_states = [
        {
            "milestone": m,
            "target_time_s": snap.target_time_s,
            "actual_time_s": snap.actual_time_s,
            "snapshot": snap,
        }
        for m, snap in zip(milestones, report.milestone_snapshots)
    ]
    return SequentialExecutionResult(
        scenario_name=scenario.scenario_name,
        scenario=scenario,
        scenario_xml="<mock_synthetic_scenario_xml/>",
        command_executed="mock biogears executable (synthetic test fixture)",
        exit_status=0,
        wall_clock_seconds=0.1,
        row_count=biogears_result.row_count,
        time_range_s=biogears_result.time_range_s,
        output_csv_path=biogears_result.output_csv_path or "mock_synthetic_test_fixture.csv",
        is_continuous=is_continuous,
        milestones=milestones,
        baseline_state=report.baseline_snapshot,
        intermediate_states=intermediate_states,
        final_state=report.final_snapshot,
        biogears_result=biogears_result,
        is_cached=is_cached,
        cache_fingerprint=fingerprint,
    )


# ─── 1–4. Valid Scenarios & Schema Adherence ──────────────────────────────────


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_valid_single_action_100ml_request(client):
    """POST /biogears/whatif/sequential handles valid 1x100mL saline scenario."""
    payload = {
        "scenario_name": "SeqSaline_1x100mL_Live",
        "patient_file": "StandardMale.xml",
        "baseline_stabilization_s": 60.0,
        "post_observation_s": 60.0,
        "samples_per_second": 0.2,
        "actions": [
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}
        ],
    }
    resp = client.post("/biogears/whatif/sequential", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["metadata"]["scenario_name"] == "SeqSaline_1x100mL_Live"
    assert data["metadata"]["patient_file"] == "StandardMale.xml"
    assert len(data["metadata"]["actions"]) == 1
    assert data["metadata"]["actions"][0]["bag_volume_ml"] == 100.0
    assert data["metadata"]["is_continuous"] is True
    assert "trajectory" in data
    assert len(data["trajectory"]["time_s"]) > 0


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_valid_two_action_sequential_request(client):
    """POST /biogears/whatif/sequential handles valid 2-action sequential scenario."""
    payload = {
        "scenario_name": "SeqSaline_2x100mL_Live",
        "actions": [
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0},
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 240.0},
        ],
    }
    resp = client.post("/biogears/whatif/sequential", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert len(data["metadata"]["actions"]) == 2
    assert data["metadata"]["actions"][0]["start_time_s"] == 60.0
    assert data["metadata"]["actions"][1]["start_time_s"] == 240.0


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_valid_three_action_sequential_request(client):
    """POST /biogears/whatif/sequential handles valid 3-action sequential scenario."""
    payload = {
        "scenario_name": "SeqSaline_3x100mL_Live",
        "actions": [
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0},
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 240.0},
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 420.0},
        ],
    }
    resp = client.post("/biogears/whatif/sequential", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert len(data["metadata"]["actions"]) == 3
    assert len(data["intermediate_milestones"]) >= 3


def test_response_milestone_ordering_and_schema(client):
    """Milestones must follow chronological timeline order with correct schema."""
    scen = BioGearsSequentialScenario(
        scenario_name="OrderingTest",
        actions=[
            BioGearsSalineBolus(100.0, 50.0, 60.0),
            BioGearsSalineBolus(100.0, 50.0, 240.0),
        ],
    )
    mock_res = _create_mock_execution_result(scen, is_cached=True)

    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        return_value=mock_res,
    ):
        payload = {
            "actions": [
                {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0},
                {"type": "saline", "volume_ml": 100.0, "start_time_s": 240.0},
            ]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Check milestones are chronologically non-decreasing
        target_times = [m["target_time_s"] for m in data["intermediate_milestones"]]
        assert target_times == sorted(target_times)

        # Baseline precedes all milestones
        assert data["baseline_snapshot"]["target_time_s"] <= target_times[0]
        # Final succeeds all milestones
        assert data["final_snapshot"]["target_time_s"] >= target_times[-1]


# ─── 5. Optional Missing Measurements Representation ──────────────────────────


def test_optional_measurements_serialized_as_null(client):
    """Measurements not present in output must serialize as JSON null, not 0."""
    scen = BioGearsSequentialScenario(
        scenario_name="NullVitalsTest",
        actions=[BioGearsSalineBolus(100.0, 50.0, 60.0)],
    )
    # Create snapshot where systolic/diastolic are None
    custom_snap = PhysiologicalSnapshot(
        target_time_s=60.0,
        actual_time_s=60.0,
        milestone_type="baseline",
        heart_rate_bpm=72.0,
        mean_arterial_pressure_mmhg=95.0,
        blood_volume_l=5.5,
        cardiac_output_l_min=5.8,
        systolic_bp_mmhg=None,
        diastolic_bp_mmhg=None,
        oxygen_saturation=None,
        respiration_rate_bpm=None,
    )
    mock_res = _create_mock_execution_result(scen)
    mock_res.baseline_state = custom_snap

    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        return_value=mock_res,
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 200
        base = resp.json()["baseline_snapshot"]

        assert base["systolic_bp_mmhg"] is None
        assert base["diastolic_bp_mmhg"] is None
        assert base["oxygen_saturation"] is None
        assert base["respiration_rate_bpm"] is None


# ─── 6–7. Cache Hit vs Fresh Execution Provenance ─────────────────────────────


def test_cache_hit_response_marking(client):
    """Cache-hit response must be clearly marked with is_cached=True and source='cache'."""
    scen = BioGearsSequentialScenario(
        scenario_name="CacheHitTest",
        actions=[BioGearsSalineBolus(100.0, 50.0, 60.0)],
    )
    mock_res = _create_mock_execution_result(scen, is_cached=True, fingerprint="cached_sha_123")

    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        return_value=mock_res,
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["is_cached"] is True
        assert meta["source"] == "cache"
        assert meta["fingerprint"] == "cached_sha_123"


def test_fresh_execution_response_marking(client):
    """Fresh execution response must be clearly marked with is_cached=False and source='live'."""
    scen = BioGearsSequentialScenario(
        scenario_name="FreshExecTest",
        actions=[BioGearsSalineBolus(100.0, 50.0, 60.0)],
    )
    mock_res = _create_mock_execution_result(scen, is_cached=False, fingerprint="live_sha_456")

    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        return_value=mock_res,
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["is_cached"] is False
        assert meta["source"] == "live"
        assert meta["fingerprint"] == "live_sha_456"


# ─── 8–12. Validation & Parameter Rejections (HTTP 422) ────────────────────────


def test_invalid_volume_rejected(client):
    """Unsupported bag volumes must be rejected with HTTP 422."""
    for bad_vol in [0, 49, 1001, 2000, -100]:
        payload = {
            "actions": [{"type": "saline", "volume_ml": bad_vol, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 422, f"Expected 422 for volume {bad_vol}"


def test_unsupported_intervention_type_rejected(client):
    """Unsupported intervention types must be rejected with HTTP 422."""
    for bad_type in ["norepinephrine", "vasopressor", "blood", "colloid", "unknown"]:
        payload = {
            "actions": [{"type": bad_type, "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 422


def test_zero_actions_and_excessive_actions_rejected(client):
    """Scenarios with 0 actions or >3 actions must be rejected with HTTP 422."""
    # Zero actions
    resp_zero = client.post("/biogears/whatif/sequential", json={"actions": []})
    assert resp_zero.status_code == 422

    # 4 actions
    four_actions = [
        {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0 + i * 180.0}
        for i in range(4)
    ]
    resp_four = client.post("/biogears/whatif/sequential", json={"actions": four_actions})
    assert resp_four.status_code == 422


def test_overlapping_actions_rejected(client):
    """Overlapping actions must be rejected with HTTP 422."""
    payload = {
        "actions": [
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0},  # delivers 60..180s
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 120.0}, # overlaps at 120s!
        ]
    }
    resp = client.post("/biogears/whatif/sequential", json=payload)
    assert resp.status_code == 422
    assert "precedes previous action completion" in str(resp.json())


def test_action_before_baseline_rejected(client):
    """Action starting before baseline completion must be rejected with HTTP 422."""
    payload = {
        "baseline_stabilization_s": 60.0,
        "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 30.0}],
    }
    resp = client.post("/biogears/whatif/sequential", json=payload)
    assert resp.status_code == 422


def test_scenario_exceeding_1800s_rejected(client):
    """Scenario with total simulated duration > 1800s must be rejected with HTTP 422."""
    payload = {
        "actions": [
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0},
            {"type": "saline", "volume_ml": 100.0, "start_time_s": 1700.0}, # 1700 + 120 + 60 = 1880s!
        ]
    }
    resp = client.post("/biogears/whatif/sequential", json=payload)
    assert resp.status_code == 422
    assert "Maximum simulated duration is 1800s" in str(resp.json())


# ─── 13–16. Engine Failures & HTTP Status Mapping ──────────────────────────────


def test_biogears_timeout_maps_to_504(client):
    """BioGears timeout error must be mapped to HTTP 504 Gateway Timeout."""
    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        side_effect=BioGearsTimeoutError("Timeout after 600s", timeout_seconds=600),
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 504
        assert "timed out after 600s" in resp.json()["detail"]


def test_biogears_process_error_maps_to_503(client):
    """BioGears engine process failure must be mapped to HTTP 503 Service Unavailable."""
    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        side_effect=BioGearsProcessError("Process crash", returncode=1),
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 503
        assert "non-zero exit code" in resp.json()["detail"]


def test_biogears_not_found_maps_to_503(client):
    """Missing BioGears executable must be mapped to HTTP 503."""
    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        side_effect=BioGearsNotFoundError("Executable missing"),
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 503


def test_snapshot_extraction_failure_maps_to_500(client):
    """Physiological extraction failure must be mapped to HTTP 500."""
    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        side_effect=BioGearsExtractionError("Required column BloodVolume missing"),
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 500
        assert "Failed to extract physiological snapshots" in resp.json()["detail"]


def test_unexpected_internal_exception_maps_to_500(client):
    """Unexpected runtime exceptions must be mapped to HTTP 500 safely."""
    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        side_effect=RuntimeError("Unexpected internal memory fault"),
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 500
        assert "Internal server error" in resp.json()["detail"]


# ─── 17. Preservation of Existing Endpoints ───────────────────────────────────


def test_preservation_of_root_health_check(client):
    """GET / remains operational."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_preservation_of_patients_endpoint(client):
    """GET /patients remains operational."""
    resp = client.get("/patients")
    assert resp.status_code == 200
    assert "patients" in resp.json()


def test_preservation_of_biogears_intervention_endpoint(client):
    """GET /biogears/intervention and GET /intervention/simulate remain operational."""
    resp = client.get("/biogears/intervention")
    assert resp.status_code == 200
    data = resp.json()
    assert "baseline" in data
    assert "post_intervention" in data
    assert "deltas" in data


# ─── 18–19. Serialization and Concurrency Delegation ──────────────────────────


def test_complete_snapshot_serialization(client):
    """All required fields of snapshot are properly formatted in response."""
    scen = BioGearsSequentialScenario(
        scenario_name="SerializationTest",
        actions=[BioGearsSalineBolus(100.0, 50.0, 60.0)],
    )
    mock_res = _create_mock_execution_result(scen)

    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario",
        return_value=mock_res,
    ):
        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        for snap_key in ["baseline_snapshot", "final_snapshot"]:
            snap = data[snap_key]
            assert isinstance(snap["heart_rate_bpm"], float)
            assert isinstance(snap["mean_arterial_pressure_mmhg"], float)
            assert isinstance(snap["blood_volume_l"], float)
            assert isinstance(snap["cardiac_output_l_min"], float)
            assert isinstance(snap["target_time_s"], float)
            assert isinstance(snap["actual_time_s"], float)
            assert isinstance(snap["milestone_type"], str)


def test_cache_concurrency_delegation(client):
    """Sequential API delegates execution with use_cache=True."""
    with patch(
        "backend.biogears_sequential_execution.execute_sequential_saline_scenario"
    ) as mock_exec:
        scen = BioGearsSequentialScenario(
            scenario_name="DelegationTest",
            actions=[BioGearsSalineBolus(100.0, 50.0, 60.0)],
        )
        mock_exec.return_value = _create_mock_execution_result(scen)

        payload = {
            "actions": [{"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0}]
        }
        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 200
        # Verify execute_sequential_saline_scenario was called with use_cache=True
        mock_exec.assert_called_once()
        _, kwargs = mock_exec.call_args
        assert kwargs.get("use_cache") is True
