"""
test_biogears_scenario_cache.py — Tests for Scenario Fingerprinting and Result Caching (Phase D Step D)
=======================================================================================================
Validates:
  1. Stable SHA-256 fingerprints for equivalent configurations.
  2. Fingerprint sensitivity to every simulation-affecting parameter.
  3. Patient-file content changes invalidating the cache fingerprint.
  4. Successful cache persistence and cache hit retrieval.
  5. Structural equivalence between fresh and cached SequentialExecutionResult.
  6. Graceful cache miss handling on missing artifacts.
  7. Graceful cache miss handling on corrupted artifacts (JSON, CSV, schema, metadata).
  8. Strict guarantee that failed/timed-out simulations are never cached.
  9. Strict physiological snapshot and continuity revalidation on cache hits.
  10. Atomic write staging and cleanup behavior.
  11. Thread-safe concurrency control preventing duplicate simulation runs.
  12. Configurable cache directories and directory auto-creation.
"""

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest

from backend.biogears_bridge import (
    BioGearsBridge,
    BioGearsConfig,
    BioGearsProcessError,
    BioGearsTimeoutError,
)
from backend.biogears_scenario_builder import (
    BioGearsNorepinephrineInfusion,
    BioGearsSalineBolus,
    BioGearsSequentialScenario,
)
from backend.biogears_scenario_cache import (
    CACHE_SCHEMA_VERSION,
    BioGearsScenarioCache,
    canonicalize_scenario_config,
    compute_scenario_fingerprint,
    get_patient_file_hash,
    resolve_patient_file,
)
from backend.biogears_sequential_execution import (
    BioGearsExtractionError,
    PhysiologicalSnapshot,
    SequentialExecutionResult,
    execute_sequential_saline_scenario,
    extract_scenario_snapshots,
    verify_temporal_continuity,
)

SAVED_STEP_B_CSV = Path("biogears_runs/seq_saline_1x100ml_results.csv")


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary cache directory."""
    cache_d = tmp_path / "scenario_cache"
    cache_d.mkdir(parents=True, exist_ok=True)
    return cache_d


@pytest.fixture
def temp_patient(tmp_path: Path) -> Tuple[Path, str]:
    """Create a temporary valid patient XML file."""
    patient_file = tmp_path / "TestPatient.xml"
    content = "<Patient><Name>TestSubject</Name><Weight>70</Weight></Patient>"
    patient_file.write_text(content, encoding="utf-8")
    return tmp_path, "TestPatient.xml"


@pytest.fixture
def sample_scenario() -> BioGearsSequentialScenario:
    """Create a standard 1x100mL saline sequential scenario."""
    return BioGearsSequentialScenario(
        scenario_name="UnitTest_SeqSaline_1x100mL",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[
            BioGearsSalineBolus(
                bag_volume_ml=100.0,
                rate_ml_per_min=50.0,
                start_time_s=60.0,
            )
        ],
    )


def _build_mock_execution_result(
    scenario: BioGearsSequentialScenario,
    csv_path: Path,
) -> SequentialExecutionResult:
    """Construct a verified SequentialExecutionResult using a valid CSV."""
    bridge = BioGearsBridge()
    biogears_result = bridge._parse_csv(
        csv_path=str(csv_path),
        scenario_name=scenario.scenario_name,
        returncode=0,
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
        scenario_xml="<Scenario><Name>Test</Name></Scenario>",
        command_executed="mock bg-scenario.exe",
        exit_status=0,
        wall_clock_seconds=12.34,
        row_count=biogears_result.row_count,
        time_range_s=biogears_result.time_range_s,
        output_csv_path=str(csv_path),
        is_continuous=is_continuous,
        milestones=milestones,
        baseline_state=report.baseline_snapshot,
        intermediate_states=intermediate_states,
        final_state=report.final_snapshot,
        biogears_result=biogears_result,
        is_cached=False,
    )


# ─── 1. Fingerprint Stability & Parameter Sensitivity ─────────────────────────


def test_stable_fingerprint_for_equivalent_configurations(sample_scenario):
    """Equivalent configurations must produce the identical SHA-256 fingerprint."""
    scen1 = sample_scenario
    scen2 = BioGearsSequentialScenario(
        scenario_name="UnitTest_SeqSaline_1x100mL",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[
            BioGearsSalineBolus(
                bag_volume_ml=100.0,
                rate_ml_per_min=50.0,
                start_time_s=60.0,
            )
        ],
    )

    fp1 = compute_scenario_fingerprint(scen1)
    fp2 = compute_scenario_fingerprint(scen2)

    assert fp1 == fp2
    assert len(fp1) == 64
    assert int(fp1, 16) > 0  # Valid hex string


def test_fingerprint_changes_on_any_simulation_parameter():
    """A change to any simulation-affecting parameter must produce a different fingerprint."""
    base = BioGearsSequentialScenario(
        scenario_name="Base_Scenario",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0)],
    )
    base_fp = compute_scenario_fingerprint(base)

    # 1. Bag volume change
    diff_vol = BioGearsSequentialScenario(
        scenario_name="Base_Scenario",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=250.0, start_time_s=60.0)],
    )
    assert compute_scenario_fingerprint(diff_vol) != base_fp

    # 2. Start time change
    diff_start = BioGearsSequentialScenario(
        scenario_name="Base_Scenario",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=90.0)],
    )
    assert compute_scenario_fingerprint(diff_start) != base_fp

    # 3. Baseline duration change
    diff_base = BioGearsSequentialScenario(
        scenario_name="Base_Scenario",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=120.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=120.0)],
    )
    assert compute_scenario_fingerprint(diff_base) != base_fp

    # 4. Post-observation duration change
    diff_post = BioGearsSequentialScenario(
        scenario_name="Base_Scenario",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=180.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0)],
    )
    assert compute_scenario_fingerprint(diff_post) != base_fp

    # 5. Sampling rate change
    diff_sampling = BioGearsSequentialScenario(
        scenario_name="Base_Scenario",
        patient_file="StandardMale.xml",
        samples_per_second=1.0,
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0)],
    )
    assert compute_scenario_fingerprint(diff_sampling) != base_fp

    # 6. Additional action
    diff_action_count = BioGearsSequentialScenario(
        scenario_name="Base_Scenario",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[
            BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0),
            BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=240.0),
        ],
    )
    assert compute_scenario_fingerprint(diff_action_count) != base_fp

    # 7. Scenario name change
    diff_name = BioGearsSequentialScenario(
        scenario_name="Different_Scenario_Name",
        patient_file="StandardMale.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0)],
    )
    assert compute_scenario_fingerprint(diff_name) != base_fp


# ─── 2. Patient File Invalidation ─────────────────────────────────────────────


def test_patient_file_content_change_invalidates_fingerprint(tmp_path: Path):
    """Modifying the content of a patient file must change the fingerprint."""
    p_file = tmp_path / "CustomPatient.xml"
    p_file.write_text("<Patient version='1.0'/>", encoding="utf-8")

    scen = BioGearsSequentialScenario(
        scenario_name="PatientHashTest",
        patient_file="CustomPatient.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0)],
    )

    fp_initial = compute_scenario_fingerprint(scen, patient_dir=tmp_path)

    # Modify content of the patient file
    p_file.write_text("<Patient version='2.0' modified='true'/>", encoding="utf-8")

    fp_modified = compute_scenario_fingerprint(scen, patient_dir=tmp_path)

    assert fp_initial != fp_modified, "Fingerprint failed to change after patient file modification!"


def test_missing_patient_file_raises_error():
    """Attempting to fingerprint an unresolvable patient file raises FileNotFoundError."""
    scen = BioGearsSequentialScenario(
        scenario_name="GhostPatient",
        patient_file="NonExistentGhostPatient_xyz.xml",
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0)],
    )
    with pytest.raises(FileNotFoundError, match="could not be found"):
        compute_scenario_fingerprint(scen)


# ─── 3. Cache Persistence & Retrieval ─────────────────────────────────────────


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_successful_cache_write_and_hit(temp_cache_dir: Path, sample_scenario):
    """Cache put writes all artifacts; cache get retrieves and revalidates successfully."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)
    result = _build_mock_execution_result(sample_scenario, SAVED_STEP_B_CSV)

    fp = cache.compute_fingerprint(sample_scenario)
    entry_dir = cache.put(result, fingerprint=fp)

    # Verify all artifacts were persisted
    assert entry_dir.is_dir()
    assert (entry_dir / "scenario.xml").is_file()
    assert (entry_dir / "results.csv").is_file()
    assert (entry_dir / "snapshots.json").is_file()
    assert (entry_dir / "metadata.json").is_file()

    # Verify metadata content
    with open(entry_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["fingerprint"] == fp
    assert meta["completed"] is True
    assert meta["schema_version"] == CACHE_SCHEMA_VERSION

    # Retrieve from cache
    cached_result = cache.get(sample_scenario, fingerprint=fp)
    assert cached_result is not None
    assert cached_result.is_cached is True
    assert cached_result.cache_fingerprint == fp
    assert cached_result.row_count == result.row_count
    assert cached_result.is_continuous is True


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_identical_result_structure_between_cached_and_fresh(temp_cache_dir: Path, sample_scenario):
    """The cached result structure must match the fresh execution result exactly."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)
    fresh = _build_mock_execution_result(sample_scenario, SAVED_STEP_B_CSV)

    fp = cache.compute_fingerprint(sample_scenario)
    cache.put(fresh, fingerprint=fp)
    cached = cache.get(sample_scenario, fingerprint=fp)

    assert cached is not None
    assert cached.scenario_name == fresh.scenario_name
    assert cached.row_count == fresh.row_count
    assert cached.time_range_s == fresh.time_range_s
    assert cached.exit_status == fresh.exit_status
    assert cached.is_continuous == fresh.is_continuous

    # Verify baseline and final snapshots match
    assert cached.baseline_state.blood_volume_l == fresh.baseline_state.blood_volume_l
    assert cached.baseline_state.heart_rate_bpm == fresh.baseline_state.heart_rate_bpm
    assert cached.baseline_state.mean_arterial_pressure_mmhg == fresh.baseline_state.mean_arterial_pressure_mmhg
    assert cached.final_state.blood_volume_l == fresh.final_state.blood_volume_l
    assert len(cached.intermediate_states) == len(fresh.intermediate_states)

    # Clearly distinguish source
    assert fresh.is_cached is False
    assert cached.is_cached is True


# ─── 4. Missing & Corrupted Artifact Handling ─────────────────────────────────


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_missing_artifact_triggers_cache_miss(temp_cache_dir: Path, sample_scenario):
    """Deleting any required artifact must cause cache.get to return None gracefully."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)
    result = _build_mock_execution_result(sample_scenario, SAVED_STEP_B_CSV)
    fp = cache.compute_fingerprint(sample_scenario)
    entry_dir = cache.put(result, fingerprint=fp)

    # 1. Missing results.csv
    csv_file = entry_dir / "results.csv"
    csv_backup = entry_dir / "results.csv.bak"
    shutil.move(csv_file, csv_backup)
    assert cache.get(sample_scenario, fingerprint=fp) is None

    # Restore and test missing metadata.json
    shutil.move(csv_backup, csv_file)
    meta_file = entry_dir / "metadata.json"
    meta_backup = entry_dir / "metadata.json.bak"
    shutil.move(meta_file, meta_backup)
    assert cache.get(sample_scenario, fingerprint=fp) is None


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_corrupted_artifacts_triggers_cache_miss(temp_cache_dir: Path, sample_scenario):
    """Corrupted metadata or incomplete marker must trigger a cache miss."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)
    result = _build_mock_execution_result(sample_scenario, SAVED_STEP_B_CSV)
    fp = cache.compute_fingerprint(sample_scenario)
    entry_dir = cache.put(result, fingerprint=fp)

    meta_file = entry_dir / "metadata.json"

    # Corrupt JSON syntax
    meta_file.write_text("{ incomplete json ...", encoding="utf-8")
    assert cache.get(sample_scenario, fingerprint=fp) is None

    # Incomplete flag
    meta_file.write_text(json.dumps({"completed": False, "fingerprint": fp}), encoding="utf-8")
    assert cache.get(sample_scenario, fingerprint=fp) is None

    # Schema version mismatch
    meta_file.write_text(
        json.dumps({"completed": True, "fingerprint": fp, "schema_version": "99.0.0"}),
        encoding="utf-8",
    )
    assert cache.get(sample_scenario, fingerprint=fp) is None


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_snapshot_revalidation_fails_on_tampered_csv(temp_cache_dir: Path, sample_scenario):
    """If the cached CSV is corrupted so that required channels are missing, get returns None."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)
    result = _build_mock_execution_result(sample_scenario, SAVED_STEP_B_CSV)
    fp = cache.compute_fingerprint(sample_scenario)
    entry_dir = cache.put(result, fingerprint=fp)

    # Overwrite CSV with missing BloodVolume column
    csv_file = entry_dir / "results.csv"
    csv_file.write_text("Time(s),HeartRate(1/min)\n10.0,72.0\n", encoding="utf-8")

    assert cache.get(sample_scenario, fingerprint=fp) is None


# ─── 5. Failed Simulations Are Never Cached ───────────────────────────────────


def test_failed_runs_never_cached(temp_cache_dir: Path, sample_scenario):
    """Simulations failing with timeout or exit errors must never be cached."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)

    mock_bridge = MagicMock()
    mock_bridge.config.executable = r"C:\BioGears\bin\bg-scenario.exe"
    mock_bridge.run_scenario_xml.side_effect = BioGearsTimeoutError(
        "Simulation timed out after 600s", timeout_seconds=600
    )

    with pytest.raises(BioGearsTimeoutError):
        execute_sequential_saline_scenario(
            scenario=sample_scenario,
            bridge=mock_bridge,
            cache=cache,
            use_cache=True,
        )

    fp = cache.compute_fingerprint(sample_scenario)
    assert not cache.has(fp)
    assert not cache.get_entry_dir(fp).exists()


def test_process_error_never_cached(temp_cache_dir: Path, sample_scenario):
    """Non-zero exit codes must raise BioGearsProcessError and not be cached."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)

    mock_bridge = MagicMock()
    mock_bridge.config.executable = r"C:\BioGears\bin\bg-scenario.exe"
    mock_bridge.run_scenario_xml.side_effect = BioGearsProcessError(
        "Simulation failed with returncode 1", returncode=1
    )

    with pytest.raises(BioGearsProcessError):
        execute_sequential_saline_scenario(
            scenario=sample_scenario,
            bridge=mock_bridge,
            cache=cache,
            use_cache=True,
        )

    fp = cache.compute_fingerprint(sample_scenario)
    assert not cache.has(fp)


# ─── 6. Atomic Write Behavior & Staging Cleanup ───────────────────────────────


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_atomic_write_cleans_up_staging_on_error(temp_cache_dir: Path, sample_scenario):
    """If an exception occurs during cache put, staging directories are cleaned up."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)
    result = _build_mock_execution_result(sample_scenario, SAVED_STEP_B_CSV)

    with patch("json.dump", side_effect=IOError("Simulated disk full write failure")):
        with pytest.raises(IOError, match="Simulated disk full"):
            cache.put(result)

    # Verify no partial directory exists
    fp = cache.compute_fingerprint(sample_scenario)
    assert not cache.get_entry_dir(fp).exists()
    staging_dirs = list(temp_cache_dir.glob(".tmp_*"))
    assert len(staging_dirs) == 0, f"Staging directory leaked: {staging_dirs}"


# ─── 7. Concurrency Control ───────────────────────────────────────────────────


@pytest.mark.skipif(not SAVED_STEP_B_CSV.exists(), reason="Requires saved Step B CSV")
def test_concurrent_requests_execute_simulation_only_once(temp_cache_dir: Path, sample_scenario):
    """Multiple concurrent requests for the same scenario run BioGears exactly ONCE."""
    cache = BioGearsScenarioCache(cache_dir=temp_cache_dir)
    mock_result = _build_mock_execution_result(sample_scenario, SAVED_STEP_B_CSV)

    call_count = 0
    call_lock = threading.Lock()

    def mock_uncached_execution(*args, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        time.sleep(0.1)  # Simulate wall-clock simulation time
        return mock_result

    results = []
    threads = []
    num_threads = 5

    def worker():
        res = execute_sequential_saline_scenario(
            scenario=sample_scenario,
            cache=cache,
            use_cache=True,
        )
        results.append(res)

    with patch(
        "backend.biogears_sequential_execution._execute_sequential_saline_uncached",
        side_effect=mock_uncached_execution,
    ):
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    # The simulation must have executed EXACTLY ONCE
    assert call_count == 1, f"Expected 1 simulation call, got {call_count}"
    assert len(results) == num_threads

    # Exactly 1 result was fresh, the remaining 4 were cache hits
    cached_flags = [r.is_cached for r in results]
    assert cached_flags.count(False) == 1
    assert cached_flags.count(True) == num_threads - 1


# ─── 8. Configurable Directory Creation ───────────────────────────────────────


def test_custom_cache_directory_auto_created(tmp_path: Path):
    """Cache manager automatically creates nested custom directories."""
    custom_path = tmp_path / "deeply" / "nested" / "biogears_cache"
    assert not custom_path.exists()

    cache = BioGearsScenarioCache(cache_dir=custom_path)
    assert custom_path.is_dir()
    assert cache.cache_dir == custom_path
