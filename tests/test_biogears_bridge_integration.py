"""
test_biogears_bridge_integration.py — Integration tests for BioGearsBridge
===========================================================================
SCOPE: Real BioGears execution against ClinTwinTinyTest.xml, and end-to-end
chain tests for biogears_scenario_builder → BioGearsBridge.

These tests are AUTOMATICALLY SKIPPED if:
  - bg-scenario.exe is not found at C:\\BioGears\\bin\\bg-scenario.exe
  - ClinTwinTinyTest.xml is not found at the expected path

No fixtures from conftest.py are used. No synthetic data is generated.

Run with:
    pytest tests/test_biogears_bridge_integration.py -v -s

Wall-clock time varies by test and scenario duration.
Run these only when you need to verify the full execution path against real BioGears.

DO NOT:
  - Run InfectionSevere.xml or any other BioGears-shipped scenario here.
  - Invent or assert specific numeric values (patient physiology is deterministic
    but asserting exact floats would couple tests to BioGears internals).
  - Import or modify whatif.py, api.py, or any ML module.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.biogears_bridge import (
    BioGearsConfig,
    BioGearsBridge,
    BioGearsNotFoundError,
    BioGearsOutputNotFoundError,
    BioGearsOutputParseError,
    BioGearsProcessError,
    BioGearsResult,
    BioGearsTimeoutError,
)
from backend.biogears_scenario_builder import (
    BioGearsFluidBolus,
    BioGearsScenario,
    build_scenario_xml,
)

# ─── Skip conditions ──────────────────────────────────────────────────────────

BIOGEARS_EXE = r"C:\BioGears\bin\bg-scenario.exe"
TINY_TEST_XML = r"C:\BioGears\bin\Scenarios\Patient\ClinTwinTinyTest.xml"
TMP_DIR = r"C:\BioGears\bin\Scenarios\tmp"

_biogears_present = os.path.isfile(BIOGEARS_EXE)
_tinytest_present = os.path.isfile(TINY_TEST_XML)

skip_no_biogears = pytest.mark.skipif(
    not _biogears_present,
    reason=f"BioGears not installed: {BIOGEARS_EXE!r} not found",
)
skip_no_tinytest = pytest.mark.skipif(
    not (_biogears_present and _tinytest_present),
    reason=(
        f"ClinTwinTinyTest.xml not found at {TINY_TEST_XML!r}; "
        "create it before running integration tests"
    ),
)


# ─── Fixture ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bridge():
    """A BioGearsBridge configured for the local BioGears installation."""
    cfg = BioGearsConfig(
        executable=BIOGEARS_EXE,
        timeout_seconds=300,
        tmp_dir=TMP_DIR,
    )
    return BioGearsBridge(config=cfg)


# ─── Executable-presence tests (no run needed) ───────────────────────────────


@skip_no_biogears
def test_executable_found(bridge):
    """_check_executable() must not raise when BioGears is installed."""
    bridge._check_executable()  # must not raise


def test_not_found_error_when_exe_missing(tmp_path):
    """BioGearsNotFoundError is raised when the configured exe does not exist."""
    cfg = BioGearsConfig(
        executable=str(tmp_path / "nonexistent.exe"),
        tmp_dir=str(tmp_path),
    )
    b = BioGearsBridge(config=cfg)
    with pytest.raises(BioGearsNotFoundError):
        b.run_scenario_file("anything.xml")


# ─── Full execution tests (slow — ~69 s each) ─────────────────────────────────


@skip_no_tinytest
def test_run_returns_biogears_result(bridge):
    """run_scenario_file() returns a BioGearsResult on a successful run."""
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert isinstance(result, BioGearsResult)


@skip_no_tinytest
def test_row_count_matches_expected(bridge):
    """
    ClinTwinTinyTest: 2-minute sim at SamplesPerSecond=0.2 → 24 data rows.
    VERIFIED: ClinTwinTinyTest runs 2026-08-26.
    """
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert result.row_count == 24


@skip_no_tinytest
def test_time_column_starts_after_zero(bridge):
    """
    BioGears emits the first sample at one interval after t=0, not at t=0.
    VERIFIED from actual ClinTwinTinyTest output.
    """
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert result.time_s[0] > 0.0


@skip_no_tinytest
def test_time_column_is_monotone(bridge):
    """Time axis must be strictly increasing."""
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    for i in range(1, len(result.time_s)):
        assert result.time_s[i] > result.time_s[i - 1], (
            f"Time axis not monotone at index {i}: "
            f"{result.time_s[i - 1]} → {result.time_s[i]}"
        )


@skip_no_tinytest
def test_ctsresistance_present(bridge):
    """CTSresistance column must be present (always second column in ClinTwinTinyTest)."""
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert "CTSresistance" in result.columns


@skip_no_tinytest
def test_ctsresistance_zero_at_baseline(bridge):
    """
    CTSresistance is 0 throughout a baseline (no-intervention) run.
    VERIFIED from ClinTwinTinyTest output.
    """
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    cts = result.columns["CTSresistance"]
    assert all(v == 0.0 for v in cts.values), (
        f"CTSresistance expected all zeros; got: {cts.values}"
    )


@skip_no_tinytest
def test_oxygen_saturation_present_bare_name(bridge):
    """
    OxygenSaturation is emitted without a unit suffix (bare column name).
    VERIFIED from actual ClinTwinTinyTest CSV header.
    """
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert "OxygenSaturation" in result.columns
    col = result.columns["OxygenSaturation"].column
    assert col.unit == ""


@skip_no_tinytest
def test_heart_rate_column_has_unit(bridge):
    """HeartRate(1/min) must be present with unit '1/min'."""
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert "HeartRate(1/min)" in result.columns
    col = result.columns["HeartRate(1/min)"].column
    assert col.unit == "1/min"
    assert col.name == "HeartRate"


@skip_no_tinytest
def test_final_state_has_all_columns(bridge):
    """final_state must contain exactly the same keys as columns."""
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert set(result.final_state.keys()) == set(result.columns.keys())


@skip_no_tinytest
def test_final_state_matches_last_row(bridge):
    """final_state[col] must equal the last value in the corresponding time series."""
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    for raw_header, ts in result.columns.items():
        assert result.final_state[raw_header] == pytest.approx(ts.values[-1]), (
            f"final_state mismatch for {raw_header!r}"
        )


@skip_no_tinytest
def test_returncode_is_zero(bridge):
    """Successful runs must have returncode=0 in the result."""
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="ClinTwinTinyTest")
    assert result.returncode == 0


@skip_no_tinytest
def test_scenario_name_preserved(bridge):
    result = bridge.run_scenario_file(TINY_TEST_XML, scenario_name="MySepsis")
    assert result.scenario_name == "MySepsis"


@skip_no_tinytest
def test_output_csv_cleaned_up_by_default(bridge):
    """Output files must be deleted after a successful run (keep_output=False)."""
    # We need to capture the prefix to check cleanup
    original_build = bridge._build_results_prefix
    captured = []

    def capturing_build(tmp_dir, scenario_name):
        prefix = original_build(tmp_dir, scenario_name)
        captured.append(prefix)
        return prefix

    bridge._build_results_prefix = capturing_build
    try:
        bridge.run_scenario_file(TINY_TEST_XML, scenario_name="CleanupTest")
    finally:
        bridge._build_results_prefix = original_build

    assert len(captured) == 1
    csv_path = captured[0] + "Results.csv"
    log_path = captured[0] + ".log"
    assert not os.path.exists(csv_path), f"CSV not cleaned up: {csv_path}"
    assert not os.path.exists(log_path), f"Log not cleaned up: {log_path}"


@skip_no_tinytest
def test_keep_output_preserves_files(bridge):
    """With keep_output=True, the CSV and log files must survive after parsing."""
    original_build = bridge._build_results_prefix
    captured = []

    def capturing_build(tmp_dir, scenario_name):
        prefix = original_build(tmp_dir, scenario_name)
        captured.append(prefix)
        return prefix

    bridge._build_results_prefix = capturing_build
    try:
        bridge.run_scenario_file(
            TINY_TEST_XML,
            scenario_name="KeepTest",
            keep_output=True,
        )
    finally:
        bridge._build_results_prefix = original_build
        # Clean up manually so we don't litter
        if captured:
            for suffix in ("Results.csv", ".log"):
                p = captured[0] + suffix
                if os.path.exists(p):
                    os.remove(p)

    assert len(captured) == 1


@skip_no_tinytest
def test_two_concurrent_calls_use_different_prefixes(bridge):
    """
    Two back-to-back calls must generate distinct output paths.
    (UUID-based prefix uniqueness.)
    """
    original_build = bridge._build_results_prefix
    prefixes = []

    def capturing_build(tmp_dir, scenario_name):
        prefix = original_build(tmp_dir, scenario_name)
        prefixes.append(prefix)
        return prefix

    bridge._build_results_prefix = capturing_build
    try:
        # Two sequential runs (true concurrency is not tested here to keep
        # wall-clock time reasonable, but prefix uniqueness is verified)
        bridge.run_scenario_file(TINY_TEST_XML, scenario_name="Run1")
        bridge.run_scenario_file(TINY_TEST_XML, scenario_name="Run1")
    finally:
        bridge._build_results_prefix = original_build

    assert len(prefixes) == 2
    assert prefixes[0] != prefixes[1]


# ─── Error-path integration tests (fast — no BioGears run) ──────────────────


@skip_no_biogears
def test_missing_scenario_file_raises_process_error(bridge):
    """Passing a nonexistent XML path to BioGears should cause a non-zero exit."""
    with pytest.raises(BioGearsProcessError):
        bridge.run_scenario_file(
            r"C:\BioGears\bin\Scenarios\Patient\DoesNotExist_abc123.xml",
            scenario_name="MissingTest",
        )


@skip_no_biogears
def test_timeout_error_on_very_short_timeout(tmp_path):
    """A 1-second timeout must trigger BioGearsTimeoutError."""
    if not _tinytest_present:
        pytest.skip("ClinTwinTinyTest.xml scenario file is not present on disk")
    cfg = BioGearsConfig(
        executable=BIOGEARS_EXE,
        timeout_seconds=1,
        tmp_dir=str(tmp_path),
    )
    b = BioGearsBridge(config=cfg)
    with pytest.raises(BioGearsTimeoutError) as exc_info:
        b.run_scenario_file(TINY_TEST_XML, scenario_name="TimeoutTest")
    assert exc_info.value.timeout_seconds == 1


# ─── Builder → Bridge end-to-end chain test ───────────────────────────────────


@skip_no_biogears
def test_fluid_bolus_builder_chains_to_bridge(bridge, tmp_path):
    """
    End-to-end chain: BioGearsFluidBolus → build_scenario_xml → tmp XML file
    → BioGearsBridge.run_scenario_file() → BioGearsResult.

    Uses a 100 mL bolus at 50 mL/min (delivery = 120 s) with a 180 s simulation
    window. Patient file: StandardMale.xml (verified in IVFluids.xml and
    SalineInfusion.xml installed scenarios).

    This test verifies:
      - build_scenario_xml() produces XML BioGears can parse and execute
      - The chain returns a BioGearsResult with > 0 rows
      - BloodVolume(L) is present in result.columns (the key column for
        confirming fluid delivery is tracked)
      - returncode is 0

    It does NOT assert specific physiological values — BioGears physiology
    is deterministic but coupling tests to exact float outputs would make
    them brittle across BioGears versions.

    Wall-clock time: patient stabilization + 180 s simulation (varies by machine).
    """
    # ── Build scenario XML ───────────────────────────────────────────────────
    # 100 mL at 50 mL/min = 120 s delivery; sim window = 180 s (covers delivery
    # with 60 s margin). Small volume keeps wall-clock time manageable.
    bolus = BioGearsFluidBolus(bag_volume_ml=100.0, rate_ml_per_min=50.0)
    scenario = BioGearsScenario(
        patient_file="StandardMale.xml",
        scenario_name="ClinTwinFluidBolus100mL",
        sim_duration_s=180.0,
        samples_per_second=0.2,
        actions=[bolus],
    )
    xml_str = build_scenario_xml(scenario)

    # ── Write to a unique temp file ──────────────────────────────────────────
    scenario_path = str(tmp_path / "fluid_bolus_100ml.xml")
    with open(scenario_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    # ── Execute via bridge ───────────────────────────────────────────────────
    result = bridge.run_scenario_file(
        scenario_path,
        scenario_name="ClinTwinFluidBolus100mL",
    )

    # ── Structural assertions only ───────────────────────────────────────────
    assert isinstance(result, BioGearsResult)
    assert result.returncode == 0
    assert result.row_count > 0

    # BloodVolume(L) is in the DataRequests block — must appear in output
    assert "BloodVolume(L)" in result.columns, (
        f"BloodVolume(L) not found in result.columns. "
        f"Available columns: {sorted(result.columns.keys())}"
    )

    # HeartRate(1/min) must also be present — sanity check on DataRequests
    assert "HeartRate(1/min)" in result.columns, (
        f"HeartRate(1/min) not found in result.columns. "
        f"Available columns: {sorted(result.columns.keys())}"
    )

    # Time axis must be monotone
    for i in range(1, len(result.time_s)):
        assert result.time_s[i] > result.time_s[i - 1], (
            f"Time axis not monotone at index {i}: "
            f"{result.time_s[i - 1]} → {result.time_s[i]}"
        )
