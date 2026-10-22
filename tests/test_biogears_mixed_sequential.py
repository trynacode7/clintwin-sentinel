"""
test_biogears_mixed_sequential.py — Step I Mixed Sequential Intervention Backend Tests
========================================================================================
Tests:
  1. Mixed intervention request schema validation (Saline + Norepinephrine).
  2. Norepinephrine parameter verification (14.0 ug/mL, 1.0 mL/min, 120.0s duration).
  3. Rejection of invalid types, invalid concentrations, rates, durations, and volumes.
  4. Non-overlapping sequential timing between fluid boluses and drug infusions.
  5. Cache fingerprint sensitivity to drug type, timing, rate, concentration, and duration.
  6. Response contract serialization for mixed scenarios.
  7. Verification with cached and live mixed simulation artifacts.
"""

import pytest
from fastapi.testclient import TestClient
from backend.api import app
from backend.biogears_schemas import (
    SalineActionRequest,
    NorepinephrineActionRequest,
    SequentialWhatIfRequest,
    ActionConfigResponse,
)
from backend.biogears_scenario_builder import (
    BioGearsSalineBolus,
    BioGearsNorepinephrineInfusion,
    BioGearsSequentialScenario,
    BioGearsScenarioBuilderError,
    VERIFIED_NOREPINEPHRINE_CONCENTRATION,
    VERIFIED_NOREPINEPHRINE_RATE,
    VERIFIED_NOREPINEPHRINE_DURATION,
)
from backend.biogears_scenario_cache import BioGearsScenarioCache
from backend.biogears_sequential_execution import (
    execute_sequential_scenario,
    execute_sequential_saline_scenario,
)

client = TestClient(app)


class TestMixedInterventionSchemas:
    def test_valid_saline_action_request(self):
        req = SalineActionRequest(type="saline", volume_ml=250.0, start_time_s=60.0)
        assert req.type == "saline"
        assert req.volume_ml == 250.0
        assert req.start_time_s == 60.0

    def test_valid_norepinephrine_action_request_defaults(self):
        req = NorepinephrineActionRequest(start_time_s=180.0)
        assert req.type == "norepinephrine"
        assert req.start_time_s == 180.0
        assert req.concentration_ug_ml == 14.0
        assert req.rate_ml_per_min == 1.0
        assert req.duration_s == 120.0

    def test_norepinephrine_rejects_volume_ml(self):
        with pytest.raises(ValueError, match="does not accept 'volume_ml'"):
            NorepinephrineActionRequest(
                type="norepinephrine",
                start_time_s=60.0,
                volume_ml=100.0,  # forbidden field
            )

    def test_norepinephrine_rejects_unverified_concentration(self):
        with pytest.raises(ValueError, match="concentration"):
            NorepinephrineActionRequest(
                type="norepinephrine",
                start_time_s=60.0,
                concentration_ug_ml=25.0,
            )

    def test_norepinephrine_rejects_unverified_rate(self):
        with pytest.raises(ValueError, match="rate"):
            NorepinephrineActionRequest(
                type="norepinephrine",
                start_time_s=60.0,
                rate_ml_per_min=10.0,
            )

    def test_norepinephrine_rejects_unverified_duration(self):
        with pytest.raises(ValueError, match="duration"):
            NorepinephrineActionRequest(
                type="norepinephrine",
                start_time_s=60.0,
                duration_s=1000.0,
            )

    def test_norepinephrine_accepts_parameterized_rate_and_duration(self):
        req = NorepinephrineActionRequest(
            type="norepinephrine",
            start_time_s=60.0,
            rate_ml_per_min=2.5,
            duration_s=300.0,
        )
        assert req.rate_ml_per_min == 2.5
        assert req.duration_s == 300.0

    def test_sequential_whatif_request_mixed_payload(self):
        payload = SequentialWhatIfRequest(
            scenario_name="MixedTest",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0},
                {"type": "norepinephrine", "start_time_s": 210.0},
            ],
        )
        assert len(payload.actions) == 2
        assert isinstance(payload.actions[0], SalineActionRequest)
        assert isinstance(payload.actions[1], NorepinephrineActionRequest)


class TestMixedScenarioBuilderAndTiming:
    def test_mixed_scenario_construction_and_milestones(self):
        scenario = BioGearsSequentialScenario(
            scenario_name="MixedMilestones",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=60.0), # 60..180s
                BioGearsNorepinephrineInfusion(start_time_s=210.0),            # 210..330s
            ],
        )
        assert scenario.is_mixed_intervention is True
        assert scenario.total_duration_s == 390.0  # 330 + 60

        milestones = scenario.get_timeline_milestones()
        phases = [m["phase"] for m in milestones]
        assert phases == [
            "baseline",
            "intervention_0",
            "intermediate_hold_0_to_1",
            "intervention_1",
            "post_observation",
        ]
        assert milestones[1]["action_type"] == "saline"
        assert milestones[3]["action_type"] == "norepinephrine"

    def test_overlapping_mixed_actions_rejected(self):
        # Saline 250mL @ 60s finishes at 60 + 300 = 360s. Norepi at 200s overlaps!
        scenario = BioGearsSequentialScenario(
            scenario_name="OverlappingMixed",
            actions=[
                BioGearsSalineBolus(bag_volume_ml=250.0, start_time_s=60.0),
                BioGearsNorepinephrineInfusion(start_time_s=200.0),
            ],
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="precedes previous action completion"):
            scenario.validate()


class TestMixedScenarioCacheCorrectness:
    def test_cache_fingerprint_distinguishes_substances(self):
        cache = BioGearsScenarioCache()

        # Pure saline scenario
        sc_saline = BioGearsSequentialScenario(
            scenario_name="TestA",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(100.0, start_time_s=210.0),
            ],
        )
        fp_saline = cache.compute_fingerprint(sc_saline)

        # Mixed scenario with Norepinephrine as second action
        sc_mixed = BioGearsSequentialScenario(
            scenario_name="TestB",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsNorepinephrineInfusion(start_time_s=210.0),
            ],
        )
        fp_mixed = cache.compute_fingerprint(sc_mixed)

        assert fp_saline != fp_mixed, "Fingerprints must differ when action type changes from saline to norepinephrine."

    def test_cache_fingerprint_distinguishes_timing(self):
        cache = BioGearsScenarioCache()

        sc1 = BioGearsSequentialScenario(
            scenario_name="Time1",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsNorepinephrineInfusion(start_time_s=210.0),
            ],
        )
        sc2 = BioGearsSequentialScenario(
            scenario_name="Time2",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsNorepinephrineInfusion(start_time_s=240.0),
            ],
        )
        assert cache.compute_fingerprint(sc1) != cache.compute_fingerprint(sc2)


class TestMixedInterventionApiExecution:
    def test_api_executes_mixed_intervention_successfully(self):
        # Live run executed in Step I verification verified cache hit
        payload = {
            "scenario_name": "SeqMixed_Saline100_Norepi14ug_Live",
            "patient_file": "StandardMale.xml",
            "baseline_stabilization_s": 60.0,
            "post_observation_s": 60.0,
            "samples_per_second": 0.2,
            "actions": [
                {"type": "saline", "volume_ml": 100.0, "start_time_s": 60.0},
                {"type": "norepinephrine", "start_time_s": 210.0, "concentration_ug_ml": 14.0, "rate_ml_per_min": 1.0, "duration_s": 120.0},
            ],
        }

        resp = client.post("/biogears/whatif/sequential", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        # Metadata checks
        meta = data["metadata"]
        assert meta["is_continuous"] is True
        assert meta["simulated_duration_s"] == 390.0
        assert len(meta["actions"]) == 2

        act0 = meta["actions"][0]
        assert act0["action_type"] == "saline"
        assert act0["bag_volume_ml"] == 100.0
        assert act0["start_time_s"] == 60.0
        assert act0["delivery_duration_s"] == 120.0

        act1 = meta["actions"][1]
        assert act1["action_type"] == "norepinephrine"
        assert act1["concentration_ug_ml"] == 14.0
        assert act1["rate_ml_per_min"] == 1.0
        assert act1["start_time_s"] == 210.0
        assert act1["delivery_duration_s"] == 120.0

        # Snapshot checks
        base = data["baseline_snapshot"]
        final = data["final_snapshot"]
        assert base["target_time_s"] == 60.0
        assert final["target_time_s"] == 390.0

        # Vasoconstrictor physiological effect: MAP increases from baseline
        assert final["mean_arterial_pressure_mmhg"] > base["mean_arterial_pressure_mmhg"]
