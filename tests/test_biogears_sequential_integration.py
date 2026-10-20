"""
test_biogears_sequential_integration.py — Live BioGears sequential execution integration tests
=============================================================================================
SCOPE: Live execution of continuous 1-, 2-, and 3-action sequential saline scenarios
against the installed C:\\BioGears\\bin\\bg-scenario.exe.

Automatically skipped if BioGears is not installed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.biogears_bridge import BioGearsBridge, BioGearsConfig
from backend.biogears_scenario_builder import (
    BioGearsSalineBolus,
    BioGearsSequentialScenario,
)
from backend.biogears_sequential_execution import (
    SequentialExecutionResult,
    execute_sequential_saline_scenario,
)

BIOGEARS_EXE = r"C:\BioGears\bin\bg-scenario.exe"
TMP_DIR = r"C:\BioGears\bin\Scenarios\tmp"

skip_no_biogears = pytest.mark.skipif(
    not os.path.isfile(BIOGEARS_EXE),
    reason=f"BioGears executable not found at {BIOGEARS_EXE!r}",
)


@pytest.fixture(scope="module")
def live_bridge():
    cfg = BioGearsConfig(
        executable=BIOGEARS_EXE,
        timeout_seconds=600,
        tmp_dir=TMP_DIR,
    )
    return BioGearsBridge(config=cfg)


@skip_no_biogears
def test_live_single_saline_100ml_execution(live_bridge):
    """
    Case 1: Live execution of a single 100 mL saline bolus on a continuous timeline.
    Timeline:
      - 0..60s: Baseline observation (60s)
      - 60..180s: Action 0 delivery (100 mL @ 50 mL/min = 120s delivery)
      - 180..240s: Final observation (60s)
      - Total duration: 240s
    """
    sc = BioGearsSequentialScenario(
        scenario_name="SeqSaline_1x100mL",
        patient_file="StandardMale.xml",
        samples_per_second=0.2,
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
    )

    res: SequentialExecutionResult = execute_sequential_saline_scenario(
        scenario=sc,
        bridge=live_bridge,
        keep_output=True,
    )

    assert res.exit_status == 0
    assert res.row_count > 0
    assert res.is_continuous is True
    assert res.time_range_s[1] >= 235.0

    # Baseline physiological values (StandardMale resting baseline)
    assert res.baseline_state.blood_volume_l == pytest.approx(5.54, abs=0.03)
    assert res.baseline_state.mean_arterial_pressure_mmhg == pytest.approx(95.9, abs=2.0)
    assert res.baseline_state.heart_rate_bpm == pytest.approx(72.3, abs=2.0)

    # Post-intervention state: Blood volume expands by ~0.09 L (90 mL)
    bv_delta = res.final_state.blood_volume_l - res.baseline_state.blood_volume_l
    assert bv_delta == pytest.approx(0.09, abs=0.02)
    assert res.final_state.blood_volume_l >= 5.61


@skip_no_biogears
def test_live_two_sequential_saline_100ml_execution(live_bridge):
    """
    Case 2: Live execution of two sequential 100 mL saline boluses on a continuous timeline.
    Timeline:
      - 0..60s: Baseline observation (60s)
      - 60..180s: Action 0 delivery (100 mL @ 50 mL/min = 120s delivery)
      - 180..240s: Intermediate observation gap (60s)
      - 240..360s: Action 1 delivery (100 mL @ 50 mL/min = 120s delivery)
      - 360..420s: Final observation (60s)
      - Total duration: 420s
    """
    sc = BioGearsSequentialScenario(
        scenario_name="SeqSaline_2x100mL",
        patient_file="StandardMale.xml",
        samples_per_second=0.2,
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[
            BioGearsSalineBolus(100.0, start_time_s=60.0),
            BioGearsSalineBolus(100.0, start_time_s=240.0),
        ],
    )

    res: SequentialExecutionResult = execute_sequential_saline_scenario(
        scenario=sc,
        bridge=live_bridge,
        keep_output=True,
    )

    assert res.exit_status == 0
    assert res.row_count > 0
    assert res.is_continuous is True
    assert res.time_range_s[1] >= 415.0

    # Milestone snapshots:
    # Intermediate after Action 0 (t=180s)
    # Intermediate after Gap (t=240s)
    # Intermediate after Action 1 (t=360s)
    # Final (t=420s)
    bv_base = res.baseline_state.blood_volume_l
    bv_after_action0 = res.intermediate_states[1]["snapshot"].blood_volume_l
    bv_after_action1 = res.intermediate_states[3]["snapshot"].blood_volume_l
    bv_final = res.final_state.blood_volume_l

    # Action 0 expands volume
    assert bv_after_action0 > bv_base
    assert (bv_after_action0 - bv_base) == pytest.approx(0.09, abs=0.03)

    # Action 1 expands volume further from the evolved state
    assert bv_after_action1 > bv_after_action0
    total_delta = bv_final - bv_base
    assert total_delta == pytest.approx(0.18, abs=0.03)


@skip_no_biogears
def test_live_three_sequential_saline_100ml_execution(live_bridge):
    """
    Case 3: Live execution of three sequential 100 mL saline boluses on a continuous timeline.
    Timeline:
      - 0..60s: Baseline observation (60s)
      - 60..180s: Action 0 delivery (100 mL @ 50 mL/min = 120s delivery)
      - 180..240s: Intermediate observation gap 1 (60s)
      - 240..360s: Action 1 delivery (100 mL @ 50 mL/min = 120s delivery)
      - 360..420s: Intermediate observation gap 2 (60s)
      - 420..540s: Action 2 delivery (100 mL @ 50 mL/min = 120s delivery)
      - 540..600s: Final observation (60s)
      - Total duration: 600s
    """
    sc = BioGearsSequentialScenario(
        scenario_name="SeqSaline_3x100mL",
        patient_file="StandardMale.xml",
        samples_per_second=0.2,
        baseline_stabilization_s=60.0,
        post_observation_s=60.0,
        actions=[
            BioGearsSalineBolus(100.0, start_time_s=60.0),
            BioGearsSalineBolus(100.0, start_time_s=240.0),
            BioGearsSalineBolus(100.0, start_time_s=420.0),
        ],
    )

    res: SequentialExecutionResult = execute_sequential_saline_scenario(
        scenario=sc,
        bridge=live_bridge,
        keep_output=True,
    )

    assert res.exit_status == 0
    assert res.row_count > 0
    assert res.is_continuous is True
    assert res.time_range_s[1] >= 595.0

    bv_base = res.baseline_state.blood_volume_l
    bv_after_action0 = res.intermediate_states[1]["snapshot"].blood_volume_l
    bv_after_action1 = res.intermediate_states[3]["snapshot"].blood_volume_l
    bv_after_action2 = res.intermediate_states[5]["snapshot"].blood_volume_l
    bv_final = res.final_state.blood_volume_l

    # Each intervention cumulatively expands intravascular blood volume
    assert bv_after_action0 > bv_base
    assert bv_after_action1 > bv_after_action0
    assert bv_after_action2 > bv_after_action1

    total_delta = bv_final - bv_base
    assert total_delta == pytest.approx(0.27, abs=0.04)
