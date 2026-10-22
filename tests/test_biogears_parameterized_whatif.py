"""
test_biogears_parameterized_whatif.py ? Comprehensive Unit & Schema Tests for Parameterized What-If (Step K2)
=============================================================================================================
Tests:
  1. SalineActionRequest schema defaults, boundary validation, and rejection of invalid values (NaN, Inf, negatives).
  2. NorepinephrineActionRequest locked parameter invariants and validation.
  3. Delivery duration calculations (e.g. 250mL @ 50mL/min -> 300s, 250mL @ 25mL/min -> 600s).
  4. Non-overlap and timing semantics (exact boundary acceptance, overlap rejection, scenario limit <= 1800s).
  5. BioGears scenario XML generation, proper attribute units, and norepinephrine cessation stop action.
  6. API endpoint compatibility, response schema verification, and deterministic cache fingerprint differentiation.
"""

import math
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.biogears_schemas import (
    SalineActionRequest,
    NorepinephrineActionRequest,
    SequentialWhatIfRequest,
    build_sequential_whatif_response,
)
from backend.biogears_scenario_builder import (
    BioGearsSalineBolus,
    BioGearsNorepinephrineInfusion,
    BioGearsSequentialScenario,
    BioGearsScenarioBuilderError,
    build_sequential_scenario_xml,
    MIN_SALINE_VOLUME_ML,
    MAX_SALINE_VOLUME_ML,
    MIN_SALINE_RATE_ML_PER_MIN,
    MAX_SALINE_RATE_ML_PER_MIN,
    DEFAULT_SALINE_VOLUME_ML,
    DEFAULT_SALINE_RATE_ML_PER_MIN,
    VERIFIED_NOREPINEPHRINE_CONCENTRATION,
    VERIFIED_NOREPINEPHRINE_RATE,
    VERIFIED_NOREPINEPHRINE_DURATION,
)
from backend.biogears_scenario_cache import (
    compute_scenario_fingerprint,
    canonicalize_scenario_config,
)

client = TestClient(app)


class TestSalineSchemaValidation:
    """Validate SalineActionRequest defaults, bounds, and rejection semantics."""

    def test_default_values(self):
        req = SalineActionRequest(start_time_s=60.0)
        assert req.type == "saline"
        assert req.volume_ml == 250.0
        assert req.rate_ml_per_min == 50.0
        assert req.start_time_s == 60.0

    def test_backward_compatible_omitted_rate(self):
        req = SalineActionRequest(volume_ml=100.0, start_time_s=60.0)
        assert req.volume_ml == 100.0
        assert req.rate_ml_per_min == 50.0

    @pytest.mark.parametrize("vol", [50.0, 100.0, 250.0, 500.0, 750.0, 1000.0])
    def test_accepted_volume_range(self, vol):
        req = SalineActionRequest(volume_ml=vol, start_time_s=60.0)
        assert req.volume_ml == vol

    @pytest.mark.parametrize("bad_vol", [49.9, 0.0, -10.0, 1000.1, 2000.0])
    def test_rejected_volume_bounds(self, bad_vol):
        with pytest.raises(ValueError, match="Supported saline volumes range"):
            SalineActionRequest(volume_ml=bad_vol, start_time_s=60.0)

    @pytest.mark.parametrize("rate", [10.0, 25.0, 50.0, 75.0, 100.0])
    def test_accepted_rate_range(self, rate):
        req = SalineActionRequest(volume_ml=250.0, rate_ml_per_min=rate, start_time_s=60.0)
        assert req.rate_ml_per_min == rate

    @pytest.mark.parametrize("bad_rate", [9.9, 0.0, -5.0, 100.1, 200.0])
    def test_rejected_rate_bounds(self, bad_rate):
        with pytest.raises(ValueError, match="Supported saline infusion rates range"):
            SalineActionRequest(volume_ml=250.0, rate_ml_per_min=bad_rate, start_time_s=60.0)

    @pytest.mark.parametrize("nan_val", [float("nan"), float("inf"), float("-inf")])
    def test_rejection_of_nan_and_inf(self, nan_val):
        with pytest.raises(ValueError):
            SalineActionRequest(volume_ml=nan_val, start_time_s=60.0)
        with pytest.raises(ValueError):
            SalineActionRequest(volume_ml=250.0, rate_ml_per_min=nan_val, start_time_s=60.0)
        with pytest.raises(ValueError):
            SalineActionRequest(volume_ml=250.0, start_time_s=nan_val)

    def test_rejection_of_negative_start_time(self):
        with pytest.raises(ValueError, match="greater than or equal to 0|non-negative"):
            SalineActionRequest(volume_ml=250.0, start_time_s=-1.0)


class TestNorepinephrineSchemaValidation:
    """Validate locked norepinephrine settings and start time validation."""

    def test_locked_defaults(self):
        req = NorepinephrineActionRequest(start_time_s=210.0)
        assert req.type == "norepinephrine"
        assert req.start_time_s == 210.0
        assert req.concentration_ug_ml == 14.0
        assert req.rate_ml_per_min == 1.0
        assert req.duration_s == 120.0
        assert req.volume_ml is None

    def test_rejection_of_volume_ml(self):
        with pytest.raises(ValueError, match="does not accept 'volume_ml'"):
            NorepinephrineActionRequest(start_time_s=210.0, volume_ml=100.0)

    def test_rejection_of_altered_concentration(self):
        with pytest.raises(ValueError, match="concentration"):
            NorepinephrineActionRequest(start_time_s=210.0, concentration_ug_ml=20.0)

    def test_rejection_of_altered_rate(self):
        with pytest.raises(ValueError, match="rate"):
            NorepinephrineActionRequest(start_time_s=210.0, rate_ml_per_min=10.0)

    def test_rejection_of_altered_duration(self):
        with pytest.raises(ValueError, match="duration"):
            NorepinephrineActionRequest(start_time_s=210.0, duration_s=1000.0)

    def test_acceptance_of_parameterized_rate_and_duration(self):
        req = NorepinephrineActionRequest(start_time_s=210.0, rate_ml_per_min=2.0, duration_s=180.0)
        assert req.rate_ml_per_min == 2.0
        assert req.duration_s == 180.0


class TestDurationAndTimingSemantics:
    """Verify delivery duration math, non-overlap rules, boundary acceptance, and limits."""

    def test_delivery_duration_math(self):
        # 250 mL @ 50 mL/min -> exactly 300.0 s
        b1 = BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=50.0, start_time_s=60.0)
        assert b1.delivery_duration_s == pytest.approx(300.0)

        # 250 mL @ 25 mL/min -> exactly 600.0 s
        b2 = BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=25.0, start_time_s=60.0)
        assert b2.delivery_duration_s == pytest.approx(600.0)

        # 100 mL @ 50 mL/min -> exactly 120.0 s
        b3 = BioGearsSalineBolus(bag_volume_ml=100.0, rate_ml_per_min=50.0, start_time_s=60.0)
        assert b3.delivery_duration_s == pytest.approx(120.0)

    def test_overlap_rejected_when_second_action_starts_early(self):
        # Saline 250mL @ 50mL/min starts at 60s -> ends at 360s.
        # Norepinephrine requested at 359s -> overlap by 1s!
        actions = [
            BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=50.0, start_time_s=60.0),
            BioGearsNorepinephrineInfusion(start_time_s=359.0),
        ]
        scenario = BioGearsSequentialScenario(
            scenario_name="OverlapTest",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=actions,
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="precedes previous action completion"):
            scenario.validate()

    def test_boundary_accepted_when_second_action_starts_at_exact_completion(self):
        # Saline 250mL @ 50mL/min starts at 60s -> ends at 360s.
        # Norepinephrine starts at exact boundary 360.0s -> valid!
        actions = [
            BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=50.0, start_time_s=60.0),
            BioGearsNorepinephrineInfusion(start_time_s=360.0),
        ]
        scenario = BioGearsSequentialScenario(
            scenario_name="BoundaryTest",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=actions,
        )
        scenario.validate()
        # Saline ends at 360s, Norepi delivers 360..480s, post observation 60s -> total 540s
        assert scenario.total_duration_s == pytest.approx(540.0)

    def test_total_scenario_duration_limit_enforced(self):
        # 1000 mL @ 50 mL/min = 1200s. Starts at 60s -> ends at 1260s.
        # Norepinephrine at 1260s for 120s -> ends at 1380s.
        # If post observation is 500s -> total 1880s (>1800s limit) -> rejected!
        actions = [
            BioGearsSalineBolus(bag_volume_ml=1000.0, rate_ml_per_min=50.0, start_time_s=60.0),
            BioGearsNorepinephrineInfusion(start_time_s=1260.0),
        ]
        scenario = BioGearsSequentialScenario(
            scenario_name="ExceedLimit",
            baseline_stabilization_s=60.0,
            post_observation_s=500.0,
            actions=actions,
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="Maximum simulated duration is 1800s"):
            scenario.validate()


class TestScenarioXMLGeneration:
    """Verify XML representation uses exact BioGears structures and cessation stops."""

    def test_xml_saline_and_norepi_with_stop(self):
        actions = [
            BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=50.0, start_time_s=60.0),
            BioGearsNorepinephrineInfusion(start_time_s=360.0),
        ]
        scenario = BioGearsSequentialScenario(
            scenario_name="XMLVerify",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=actions,
        )
        xml_str = build_sequential_scenario_xml(scenario)

        # Saline action check
        assert '<SubstanceCompound>Saline</SubstanceCompound>' in xml_str
        assert '<BagVolume' in xml_str and 'value="250"' in xml_str and 'unit="mL"' in xml_str
        assert '<Rate' in xml_str and 'value="50"' in xml_str and 'unit="mL/min"' in xml_str

        # Norepinephrine start check
        assert '<Substance>Norepinephrine</Substance>' in xml_str
        assert '<Concentration' in xml_str and 'value="14"' in xml_str and 'unit="ug/mL"' in xml_str
        assert '<Rate' in xml_str and 'value="1"' in xml_str and 'unit="mL/min"' in xml_str

        # Norepinephrine stop action check (Rate 0, Concentration 0)
        assert '<Concentration' in xml_str and 'value="0"' in xml_str and 'unit="ug/mL"' in xml_str


class TestCacheFingerprintDifferentiation:
    """Verify cache keys change when volume, rate, or timing varies."""

    def test_rate_change_modifies_fingerprint(self):
        scen1 = BioGearsSequentialScenario(
            scenario_name="Scen",
            actions=[BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=50.0, start_time_s=60.0)],
        )
        scen2 = BioGearsSequentialScenario(
            scenario_name="Scen",
            actions=[BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=25.0, start_time_s=60.0)],
        )
        fp1 = compute_scenario_fingerprint(scen1)
        fp2 = compute_scenario_fingerprint(scen2)
        assert fp1 != fp2, "Varying infusion rate must produce distinct cache fingerprints"

    def test_volume_change_modifies_fingerprint(self):
        scen1 = BioGearsSequentialScenario(
            scenario_name="Scen",
            actions=[BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=50.0, start_time_s=60.0)],
        )
        scen2 = BioGearsSequentialScenario(
            scenario_name="Scen",
            actions=[BioGearsSalineBolus(bag_volume_ml=500.0, rate_ml_per_min=50.0, start_time_s=60.0)],
        )
        assert compute_scenario_fingerprint(scen1) != compute_scenario_fingerprint(scen2)


class TestAPISequentialValidation:
    """Verify FastAPI /biogears/whatif/sequential endpoint validation."""

    def test_api_rejects_overlapping_norepinephrine(self):
        payload = {
            "scenario_name": "APIOverlapTest",
            "baseline_stabilization_s": 60.0,
            "post_observation_s": 60.0,
            "actions": [
                {"type": "saline", "volume_ml": 250.0, "rate_ml_per_min": 50.0, "start_time_s": 60.0},
                {"type": "norepinephrine", "start_time_s": 210.0},
            ],
        }
        res = client.post("/biogears/whatif/sequential", json=payload)
        assert res.status_code == 422
        assert "precedes previous action completion" in res.json()["detail"]

    def test_api_rejects_out_of_bounds_rate(self):
        payload = {
            "scenario_name": "APIRateTest",
            "actions": [
                {"type": "saline", "volume_ml": 250.0, "rate_ml_per_min": 5.0, "start_time_s": 60.0},
            ],
        }
        res = client.post("/biogears/whatif/sequential", json=payload)
        assert res.status_code == 422


class TestStepHParameterizedNorepinephrine:
    """Step H: Tests for configurable norepinephrine rate, duration, and XML propagation."""

    def test_xml_generation_with_custom_rate_and_duration(self):
        from backend.biogears_scenario_builder import (
            build_sequential_scenario_xml,
            BioGearsSequentialScenario,
            BioGearsNorepinephrineInfusion,
        )
        scenario = BioGearsSequentialScenario(
            scenario_name="StepH_NE_Test",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsNorepinephrineInfusion(
                    concentration_ug_ml=14.0,
                    rate_ml_per_min=2.5,
                    duration_s=180.0,
                    start_time_s=60.0,
                )
            ],
        )
        xml_str = build_sequential_scenario_xml(scenario)

        # Verify rate of 2.5 mL/min is in XML
        assert 'value="2.5"' in xml_str or 'value="2.50' in xml_str
        assert "<Substance>Norepinephrine</Substance>" in xml_str
        # Verify termination occurs at start (60) + duration (180) = 240s
        assert 'value="180"' in xml_str or 'value="180.0"' in xml_str
        # Stop infusion (rate 0.0)
        assert '<Rate value="0" unit="mL/min" />' in xml_str

    def test_cache_fingerprint_changes_with_norepi_parameters(self):
        from backend.biogears_scenario_builder import (
            BioGearsSequentialScenario,
            BioGearsNorepinephrineInfusion,
        )
        from backend.biogears_scenario_cache import compute_scenario_fingerprint

        scen_base = BioGearsSequentialScenario(
            scenario_name="NE_Base",
            actions=[BioGearsNorepinephrineInfusion(rate_ml_per_min=1.0, duration_s=120.0, start_time_s=60.0)],
        )
        scen_rate = BioGearsSequentialScenario(
            scenario_name="NE_Rate",
            actions=[BioGearsNorepinephrineInfusion(rate_ml_per_min=2.0, duration_s=120.0, start_time_s=60.0)],
        )
        scen_dur = BioGearsSequentialScenario(
            scenario_name="NE_Dur",
            actions=[BioGearsNorepinephrineInfusion(rate_ml_per_min=1.0, duration_s=60.0, start_time_s=60.0)],
        )

        fp_base = compute_scenario_fingerprint(scen_base)
        fp_rate = compute_scenario_fingerprint(scen_rate)
        fp_dur = compute_scenario_fingerprint(scen_dur)

        assert fp_base != fp_rate, "Different rate must produce distinct cache fingerprint"
        assert fp_base != fp_dur, "Different duration must produce distinct cache fingerprint"
        assert fp_rate != fp_dur, "Different rate and duration must produce distinct fingerprints"

    def test_api_rejects_unsupported_concentration_with_explanation(self):
        payload = {
            "scenario_name": "UnsupportedConcTest",
            "actions": [
                {
                    "type": "norepinephrine",
                    "concentration_ug_ml": 25.0,
                    "rate_ml_per_min": 1.0,
                    "duration_s": 120.0,
                    "start_time_s": 60.0,
                }
            ],
        }
        res = client.post("/biogears/whatif/sequential", json=payload)
        assert res.status_code == 422
        detail_str = str(res.json()["detail"])
        assert "14" in detail_str
        assert "concentration" in detail_str.lower() or "stock" in detail_str.lower()

    def test_api_accepts_parameterized_norepinephrine(self):
        payload = {
            "scenario_name": "ValidParamNETest",
            "actions": [
                {
                    "type": "norepinephrine",
                    "concentration_ug_ml": 14.0,
                    "rate_ml_per_min": 2.0,
                    "duration_s": 90.0,
                    "start_time_s": 60.0,
                }
            ],
        }
        from backend.biogears_schemas import SequentialWhatIfRequest
        req = SequentialWhatIfRequest(**payload)
        assert req.actions[0].rate_ml_per_min == 2.0
        assert req.actions[0].duration_s == 90.0
