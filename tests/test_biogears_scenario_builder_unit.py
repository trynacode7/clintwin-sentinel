"""
test_biogears_scenario_builder_unit.py — Unit tests for biogears_scenario_builder
===================================================================================
SCOPE: Fast, fully offline tests. No BioGears installation required.
       No file I/O, no subprocess calls.

Run with:
    pytest tests/test_biogears_scenario_builder_unit.py -v
"""

import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.biogears_scenario_builder import (
    BioGearsFluidBolus,
    BioGearsScenario,
    BioGearsScenarioBuilderError,
    build_scenario_xml,
    BioGearsSalineBolus,
    BioGearsNorepinephrineInfusion,
    BioGearsSequentialScenario,
    build_sequential_scenario_xml,
    VERIFIED_SALINE_VOLUMES,
    VERIFIED_SALINE_RATE,
    VERIFIED_NOREPINEPHRINE_CONCENTRATION,
    VERIFIED_NOREPINEPHRINE_RATE,
    VERIFIED_NOREPINEPHRINE_DURATION,
)

# ─── Namespaces (must match the builder's registered values) ──────────────────

_NS = "uri:/mil/tatrc/physiology/datamodel"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"

_NS_MAP = {
    "bg": _NS,
    "xsi": _XSI,
}


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _bolus(volume=500.0, rate=50.0) -> BioGearsFluidBolus:
    return BioGearsFluidBolus(bag_volume_ml=volume, rate_ml_per_min=rate)


def _scenario(
    boluses=None,
    patient_file="StandardMale.xml",
    scenario_name="TestScenario",
    sim_duration_s=700.0,
    samples_per_second=0.2,
) -> BioGearsScenario:
    actions = boluses if boluses is not None else [_bolus()]
    return BioGearsScenario(
        patient_file=patient_file,
        scenario_name=scenario_name,
        sim_duration_s=sim_duration_s,
        samples_per_second=samples_per_second,
        actions=actions,
    )


def _parse(xml_str: str) -> ET.Element:
    """Parse an XML string and return the root element."""
    return ET.fromstring(xml_str.split("\n", 1)[1])  # strip XML declaration


# ─── BioGearsFluidBolus ───────────────────────────────────────────────────────


class TestBioGearsFluidBolus:
    def test_default_rate_is_50(self):
        b = BioGearsFluidBolus(bag_volume_ml=500.0)
        assert b.rate_ml_per_min == 50.0

    def test_custom_rate_stored(self):
        b = BioGearsFluidBolus(bag_volume_ml=250.0, rate_ml_per_min=100.0)
        assert b.rate_ml_per_min == 100.0

    def test_bag_volume_stored(self):
        b = BioGearsFluidBolus(bag_volume_ml=1000.0)
        assert b.bag_volume_ml == 1000.0

    def test_frozen_cannot_set_volume(self):
        b = BioGearsFluidBolus(bag_volume_ml=500.0)
        with pytest.raises((AttributeError, TypeError)):
            b.bag_volume_ml = 600.0  # type: ignore[misc]

    def test_frozen_cannot_set_rate(self):
        b = BioGearsFluidBolus(bag_volume_ml=500.0)
        with pytest.raises((AttributeError, TypeError)):
            b.rate_ml_per_min = 100.0  # type: ignore[misc]

    def test_delivery_time_s(self):
        # 500 mL at 50 mL/min = 10 min = 600 s
        b = BioGearsFluidBolus(bag_volume_ml=500.0, rate_ml_per_min=50.0)
        assert b.delivery_time_s == pytest.approx(600.0)

    def test_delivery_time_s_different_rate(self):
        # 100 mL at 100 mL/min = 1 min = 60 s
        b = BioGearsFluidBolus(bag_volume_ml=100.0, rate_ml_per_min=100.0)
        assert b.delivery_time_s == pytest.approx(60.0)


# ─── BioGearsScenario ─────────────────────────────────────────────────────────


class TestBioGearsScenario:
    def test_all_fields_stored(self):
        actions = [_bolus()]
        sc = BioGearsScenario(
            patient_file="StandardMale.xml",
            scenario_name="MySepsis",
            sim_duration_s=700.0,
            samples_per_second=0.2,
            actions=actions,
        )
        assert sc.patient_file == "StandardMale.xml"
        assert sc.scenario_name == "MySepsis"
        assert sc.sim_duration_s == 700.0
        assert sc.samples_per_second == 0.2
        assert sc.actions is actions

    def test_frozen(self):
        sc = _scenario()
        with pytest.raises((AttributeError, TypeError)):
            sc.patient_file = "Other.xml"  # type: ignore[misc]


# ─── BioGearsScenarioBuilderError ─────────────────────────────────────────────


class TestBioGearsScenarioBuilderError:
    def test_is_exception_subclass(self):
        assert issubclass(BioGearsScenarioBuilderError, Exception)

    def test_carries_message(self):
        exc = BioGearsScenarioBuilderError("something went wrong")
        assert str(exc) == "something went wrong"


# ─── Validation errors ────────────────────────────────────────────────────────


class TestValidationErrors:
    # ── BioGearsFluidBolus field errors ──

    def test_bag_volume_zero_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="bag_volume_ml"):
            build_scenario_xml(_scenario(boluses=[_bolus(volume=0.0)]))

    def test_bag_volume_negative_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="bag_volume_ml"):
            build_scenario_xml(_scenario(boluses=[_bolus(volume=-1.0)]))

    def test_bag_volume_over_2000_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="2000"):
            build_scenario_xml(_scenario(boluses=[_bolus(volume=2001.0)]))

    def test_bag_volume_exactly_2000_ok(self):
        sc = _scenario(boluses=[_bolus(volume=2000.0)], sim_duration_s=2500.0)
        xml_str = build_scenario_xml(sc)
        assert xml_str  # no exception

    def test_rate_zero_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="rate_ml_per_min"):
            build_scenario_xml(_scenario(boluses=[_bolus(rate=0.0)]))

    def test_rate_negative_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="rate_ml_per_min"):
            build_scenario_xml(_scenario(boluses=[_bolus(rate=-5.0)]))

    # ── BioGearsScenario field errors ──

    def test_empty_patient_file_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="patient_file"):
            build_scenario_xml(_scenario(patient_file=""))

    def test_whitespace_patient_file_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="patient_file"):
            build_scenario_xml(_scenario(patient_file="   "))

    def test_empty_scenario_name_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="scenario_name"):
            build_scenario_xml(_scenario(scenario_name=""))

    def test_sim_duration_zero_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="sim_duration_s"):
            build_scenario_xml(_scenario(sim_duration_s=0.0))

    def test_sim_duration_negative_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="sim_duration_s"):
            build_scenario_xml(_scenario(sim_duration_s=-1.0))

    def test_samples_per_second_zero_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="samples_per_second"):
            build_scenario_xml(_scenario(samples_per_second=0.0))

    def test_samples_per_second_negative_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="samples_per_second"):
            build_scenario_xml(_scenario(samples_per_second=-0.1))

    def test_empty_actions_raises(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="actions"):
            build_scenario_xml(_scenario(boluses=[]))

    # ── sim_duration vs bolus delivery time ──

    def test_sim_duration_shorter_than_delivery_raises(self):
        # 500 mL at 50 mL/min = 600 s — sim of 599 s too short
        bolus = _bolus(volume=500.0, rate=50.0)
        sc = _scenario(boluses=[bolus], sim_duration_s=599.0)
        with pytest.raises(BioGearsScenarioBuilderError, match="600"):
            build_scenario_xml(sc)

    def test_sim_duration_exactly_equal_to_delivery_ok(self):
        # 500 mL at 50 mL/min = 600 s — sim of exactly 600 s is OK
        bolus = _bolus(volume=500.0, rate=50.0)
        sc = _scenario(boluses=[bolus], sim_duration_s=600.0)
        xml_str = build_scenario_xml(sc)
        assert xml_str  # no exception

    def test_sim_duration_longer_than_delivery_ok(self):
        bolus = _bolus(volume=500.0, rate=50.0)
        sc = _scenario(boluses=[bolus], sim_duration_s=700.0)
        xml_str = build_scenario_xml(sc)
        assert xml_str

    def test_sim_duration_error_mentions_action_index(self):
        # Second action is the short one
        bolus1 = _bolus(volume=100.0, rate=50.0)   # delivery = 120 s
        bolus2 = _bolus(volume=500.0, rate=50.0)   # delivery = 600 s
        sc = _scenario(boluses=[bolus1, bolus2], sim_duration_s=200.0)
        with pytest.raises(BioGearsScenarioBuilderError, match=r"actions\[1\]"):
            build_scenario_xml(sc)

    # ── Unsupported action type ──

    def test_unsupported_action_type_raises(self):
        class FakeAction:
            pass

        sc = BioGearsScenario(
            patient_file="StandardMale.xml",
            scenario_name="Bad",
            sim_duration_s=700.0,
            samples_per_second=0.2,
            actions=[FakeAction()],  # type: ignore[list-item]
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="FakeAction"):
            build_scenario_xml(sc)

    def test_unsupported_action_error_mentions_phase_1(self):
        class FakeAction:
            pass

        sc = BioGearsScenario(
            patient_file="StandardMale.xml",
            scenario_name="Bad",
            sim_duration_s=700.0,
            samples_per_second=0.2,
            actions=[FakeAction()],  # type: ignore[list-item]
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="Phase 1"):
            build_scenario_xml(sc)


# ─── build_scenario_xml — good-path XML structure ────────────────────────────


class TestBuildScenarioXmlGoodPath:
    """All tests parse the XML string to assert structural properties."""

    def _build(self, **kwargs) -> str:
        return build_scenario_xml(_scenario(**kwargs))

    def test_returns_string(self):
        assert isinstance(self._build(), str)

    def test_parses_as_valid_xml(self):
        xml_str = self._build()
        # Strip XML declaration before parsing
        _parse(xml_str)  # must not raise

    def test_xml_declaration_present(self):
        xml_str = self._build()
        assert xml_str.startswith("<?xml")

    def test_xml_declaration_has_standalone_yes(self):
        xml_str = self._build()
        assert 'standalone="yes"' in xml_str.splitlines()[0]

    def test_root_tag_is_scenario(self):
        root = _parse(self._build())
        assert root.tag == f"{{{_NS}}}Scenario"

    def test_content_version_attribute(self):
        root = _parse(self._build())
        assert root.get("contentVersion") == "BioGears_6.3.0-beta"

    def test_xsi_schema_location_attribute_present(self):
        root = _parse(self._build())
        assert root.get(f"{{{_XSI}}}schemaLocation") is not None

    def test_name_element(self):
        root = _parse(self._build(scenario_name="FluidTest"))
        name_el = root.find(f"{{{_NS}}}Name")
        assert name_el is not None
        assert name_el.text == "FluidTest"

    def test_patient_file_element(self):
        root = _parse(self._build(patient_file="StandardMale.xml"))
        pf = root.find(f".//{{{_NS}}}PatientFile")
        assert pf is not None
        assert pf.text == "StandardMale.xml"

    def test_samples_per_second_attribute(self):
        root = _parse(self._build(samples_per_second=0.5))
        dr = root.find(f"{{{_NS}}}DataRequests")
        assert dr is not None
        assert dr.get("SamplesPerSecond") == "0.5"

    def test_samples_per_second_whole_number_format(self):
        root = _parse(self._build(samples_per_second=1.0))
        dr = root.find(f"{{{_NS}}}DataRequests")
        assert dr.get("SamplesPerSecond") == "1"

    # ── DataRequests ──

    def test_data_requests_element_present(self):
        root = _parse(self._build())
        assert root.find(f"{{{_NS}}}DataRequests") is not None

    def test_heart_rate_data_request_present(self):
        root = _parse(self._build())
        drs = root.findall(f".//{{{_NS}}}DataRequest")
        names = [dr.get("Name") for dr in drs]
        assert "HeartRate" in names

    def test_blood_volume_data_request_present(self):
        root = _parse(self._build())
        drs = root.findall(f".//{{{_NS}}}DataRequest")
        names = [dr.get("Name") for dr in drs]
        assert "BloodVolume" in names

    def test_data_request_has_xsi_type(self):
        root = _parse(self._build())
        dr = root.find(f".//{{{_NS}}}DataRequest")
        assert dr is not None
        assert dr.get(f"{{{_XSI}}}type") == "PhysiologyDataRequestData"

    def test_at_least_two_data_requests(self):
        root = _parse(self._build())
        drs = root.findall(f".//{{{_NS}}}DataRequest")
        assert len(drs) >= 2

    # ── Actions ──

    def test_actions_element_present(self):
        root = _parse(self._build())
        assert root.find(f"{{{_NS}}}Actions") is not None

    def test_substance_compound_action_present(self):
        root = _parse(self._build())
        actions = root.findall(f".//{{{_NS}}}Action")
        types = [a.get(f"{{{_XSI}}}type") for a in actions]
        assert "SubstanceCompoundInfusionData" in types

    def test_substance_compound_is_saline(self):
        root = _parse(self._build())
        compound = root.find(f".//{{{_NS}}}SubstanceCompound")
        assert compound is not None
        assert compound.text == "Saline"

    def test_bag_volume_value(self):
        # 750 mL at 50 mL/min = 900 s delivery; use 1000 s sim window
        root = _parse(self._build(boluses=[_bolus(volume=750.0)],
                                  sim_duration_s=1000.0))
        bag = root.find(f".//{{{_NS}}}BagVolume")
        assert bag is not None
        assert float(bag.get("value")) == pytest.approx(750.0)

    def test_bag_volume_unit_is_ml(self):
        root = _parse(self._build())
        bag = root.find(f".//{{{_NS}}}BagVolume")
        assert bag.get("unit") == "mL"

    def test_rate_value_default_50(self):
        root = _parse(self._build())
        rate = root.find(f".//{{{_NS}}}Rate")
        assert rate is not None
        assert float(rate.get("value")) == pytest.approx(50.0)

    def test_rate_unit_is_ml_per_min(self):
        root = _parse(self._build())
        rate = root.find(f".//{{{_NS}}}Rate")
        assert rate.get("unit") == "mL/min"

    def test_rate_value_custom(self):
        root = _parse(self._build(boluses=[_bolus(rate=100.0)]))
        rate = root.find(f".//{{{_NS}}}Rate")
        assert float(rate.get("value")) == pytest.approx(100.0)

    def test_advance_time_action_present(self):
        root = _parse(self._build())
        actions = root.findall(f".//{{{_NS}}}Action")
        types = [a.get(f"{{{_XSI}}}type") for a in actions]
        assert "AdvanceTimeData" in types

    def test_exactly_one_advance_time_action(self):
        """No stabilization pause — only the simulation window AdvanceTime."""
        root = _parse(self._build())
        actions = root.findall(f".//{{{_NS}}}Action")
        advance_times = [
            a for a in actions
            if a.get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        ]
        assert len(advance_times) == 1

    def test_advance_time_value_matches_sim_duration(self):
        root = _parse(self._build(sim_duration_s=800.0))
        actions = root.findall(f".//{{{_NS}}}Action")
        for a in actions:
            if a.get(f"{{{_XSI}}}type") == "AdvanceTimeData":
                time_el = a.find(f"{{{_NS}}}Time")
                assert time_el is not None
                assert float(time_el.get("value")) == pytest.approx(800.0)
                assert time_el.get("unit") == "s"
                break

    # ── Multiple bolus actions ──

    def test_two_bolus_actions_both_emitted(self):
        boluses = [_bolus(volume=250.0), _bolus(volume=500.0)]
        root = _parse(build_scenario_xml(_scenario(
            boluses=boluses, sim_duration_s=700.0
        )))
        actions = root.findall(f".//{{{_NS}}}Action")
        infusion_actions = [
            a for a in actions
            if a.get(f"{{{_XSI}}}type") == "SubstanceCompoundInfusionData"
        ]
        assert len(infusion_actions) == 2

    def test_two_bolus_volumes_preserved(self):
        boluses = [_bolus(volume=250.0), _bolus(volume=750.0)]
        root = _parse(build_scenario_xml(_scenario(
            boluses=boluses, sim_duration_s=1000.0
        )))
        bag_els = root.findall(f".//{{{_NS}}}BagVolume")
        volumes = sorted(float(b.get("value")) for b in bag_els)
        assert volumes == pytest.approx([250.0, 750.0])

    # ── Purity ──

    def test_pure_function_same_input_same_output(self):
        sc = _scenario()
        assert build_scenario_xml(sc) == build_scenario_xml(sc)

    def test_different_volumes_different_xml(self):
        xml_a = build_scenario_xml(_scenario(boluses=[_bolus(volume=500.0)]))
        xml_b = build_scenario_xml(_scenario(boluses=[_bolus(volume=1000.0)],
                                             sim_duration_s=1300.0))
        assert xml_a != xml_b

    # ── Namespace ──

    def test_default_namespace_in_output(self):
        xml_str = self._build()
        assert _NS in xml_str

    def test_xsi_namespace_in_output(self):
        xml_str = self._build()
        assert _XSI in xml_str

    def test_xsi_type_attribute_in_output(self):
        xml_str = self._build()
        assert "SubstanceCompoundInfusionData" in xml_str


# ─── Phase D Step A: Sequential Scenario Builder Unit Tests ───────────────────


class TestBioGearsSalineBolus:
    """Tests for BioGearsSalineBolus locked parameter verification."""

    def test_all_supported_volumes(self):
        for vol in (100.0, 250.0, 500.0):
            bolus = BioGearsSalineBolus(bag_volume_ml=vol)
            assert bolus.bag_volume_ml == vol
            assert bolus.rate_ml_per_min == VERIFIED_SALINE_RATE

    def test_duration_calculation_100ml(self):
        bolus = BioGearsSalineBolus(bag_volume_ml=100.0)
        # 100 mL @ 50 mL/min = 2 min = 120 s
        assert bolus.delivery_duration_s == pytest.approx(120.0)
        assert bolus.delivery_time_s == pytest.approx(120.0)

    def test_duration_calculation_250ml(self):
        bolus = BioGearsSalineBolus(bag_volume_ml=250.0)
        # 250 mL @ 50 mL/min = 5 min = 300 s
        assert bolus.delivery_duration_s == pytest.approx(300.0)
        assert bolus.delivery_time_s == pytest.approx(300.0)

    def test_duration_calculation_500ml(self):
        bolus = BioGearsSalineBolus(bag_volume_ml=500.0)
        # 500 mL @ 50 mL/min = 10 min = 600 s
        assert bolus.delivery_duration_s == pytest.approx(600.0)
        assert bolus.delivery_time_s == pytest.approx(600.0)

    def test_rejection_of_unsupported_volumes(self):
        for unsupp_vol in (0.0, 49.9, 1000.1, 2000.0, -100.0):
            with pytest.raises(BioGearsScenarioBuilderError, match="outside approved bounds|positive finite number"):
                BioGearsSalineBolus(bag_volume_ml=unsupp_vol)

    def test_acceptance_of_parameterized_volumes(self):
        for vol in (50.0, 100.0, 150.0, 250.0, 500.0, 750.0, 1000.0):
            b = BioGearsSalineBolus(bag_volume_ml=vol)
            assert b.bag_volume_ml == vol

    def test_rejection_of_zero_volume(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="positive finite number"):
            BioGearsSalineBolus(bag_volume_ml=0.0)

    def test_rejection_of_negative_volume(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="positive finite number"):
            BioGearsSalineBolus(bag_volume_ml=-100.0)

    def test_rejection_of_unsupported_rate(self):
        for unsupp_rate in (0.0, 9.9, 100.1, -50.0):
            with pytest.raises(BioGearsScenarioBuilderError, match="outside approved bounds|positive finite number"):
                BioGearsSalineBolus(bag_volume_ml=100.0, rate_ml_per_min=unsupp_rate)

    def test_acceptance_of_parameterized_rates(self):
        for rate in (10.0, 25.0, 50.0, 75.0, 100.0):
            b = BioGearsSalineBolus(bag_volume_ml=250.0, rate_ml_per_min=rate)
            assert b.rate_ml_per_min == rate

    def test_rejection_of_negative_start_time(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="start_time_s"):
            BioGearsSalineBolus(bag_volume_ml=100.0, start_time_s=-1.0)

    def test_frozen_immutability(self):
        bolus = BioGearsSalineBolus(bag_volume_ml=100.0)
        with pytest.raises((AttributeError, TypeError)):
            bolus.bag_volume_ml = 250.0  # type: ignore[misc]


class TestBioGearsNorepinephrineInfusion:
    """Tests for BioGearsNorepinephrineInfusion locked parameter verification."""

    def test_locked_default_parameters(self):
        inf = BioGearsNorepinephrineInfusion()
        assert inf.concentration_ug_ml == VERIFIED_NOREPINEPHRINE_CONCENTRATION  # 14.0
        assert inf.rate_ml_per_min == VERIFIED_NOREPINEPHRINE_RATE                # 1.0
        assert inf.duration_s == VERIFIED_NOREPINEPHRINE_DURATION                # 120.0
        assert inf.delivery_duration_s == 120.0
        assert inf.delivery_time_s == 120.0

    def test_explicit_start_time(self):
        inf = BioGearsNorepinephrineInfusion(start_time_s=180.0)
        assert inf.start_time_s == 180.0

    def test_positional_start_time(self):
        inf = BioGearsNorepinephrineInfusion(180.0)
        assert inf.start_time_s == 180.0

    def test_rejection_of_modified_concentration(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="concentration_ug_ml"):
            BioGearsNorepinephrineInfusion(concentration_ug_ml=20.0)

    def test_rejection_of_modified_rate(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="rate_ml_per_min"):
            BioGearsNorepinephrineInfusion(rate_ml_per_min=10.0)

    def test_rejection_of_modified_duration(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="duration_s"):
            BioGearsNorepinephrineInfusion(duration_s=700.0)

    def test_acceptance_of_configurable_rate_and_duration(self):
        inf = BioGearsNorepinephrineInfusion(rate_ml_per_min=2.0, duration_s=60.0)
        assert inf.rate_ml_per_min == 2.0
        assert inf.duration_s == 60.0

    def test_rejection_of_zero_duration(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="duration_s"):
            BioGearsNorepinephrineInfusion(duration_s=0.0)

    def test_rejection_of_negative_duration(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="duration_s"):
            BioGearsNorepinephrineInfusion(duration_s=-120.0)

    def test_rejection_of_negative_start_time(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="start_time_s"):
            BioGearsNorepinephrineInfusion(start_time_s=-10.0)

    def test_frozen_immutability(self):
        inf = BioGearsNorepinephrineInfusion()
        with pytest.raises((AttributeError, TypeError)):
            inf.start_time_s = 120.0  # type: ignore[misc]


class TestBioGearsSequentialScenarioValidation:
    """Tests for multi-action scenario validation rules and limits."""

    def test_empty_actions_rejected(self):
        sc = BioGearsSequentialScenario(scenario_name="Empty", actions=[])
        with pytest.raises(BioGearsScenarioBuilderError, match="at least 1 action"):
            build_sequential_scenario_xml(sc)

    def test_none_actions_rejected(self):
        sc = BioGearsSequentialScenario(scenario_name="Empty", actions=None)
        with pytest.raises(BioGearsScenarioBuilderError, match="at least 1 action"):
            build_sequential_scenario_xml(sc)

    def test_one_action_accepted(self):
        sc = BioGearsSequentialScenario(
            scenario_name="OneAction",
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )
        sc.validate()  # must not raise

    def test_two_actions_accepted(self):
        sc = BioGearsSequentialScenario(
            scenario_name="TwoActions",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),   # 60s..180s
                BioGearsSalineBolus(250.0, start_time_s=240.0),  # 240s..540s
            ],
        )
        sc.validate()  # must not raise

    def test_three_actions_accepted(self):
        sc = BioGearsSequentialScenario(
            scenario_name="ThreeActions",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),   # 60s..180s
                BioGearsSalineBolus(100.0, start_time_s=200.0),  # 200s..320s
                BioGearsSalineBolus(100.0, start_time_s=340.0),  # 340s..460s
            ],
        )
        sc.validate()  # must not raise

    def test_more_than_three_actions_rejected(self):
        actions = [
            BioGearsSalineBolus(100.0, start_time_s=60.0),
            BioGearsSalineBolus(100.0, start_time_s=200.0),
            BioGearsSalineBolus(100.0, start_time_s=340.0),
            BioGearsSalineBolus(100.0, start_time_s=480.0),
        ]
        sc = BioGearsSequentialScenario(scenario_name="FourActions", actions=actions)
        with pytest.raises(BioGearsScenarioBuilderError, match="maximum of 3 actions"):
            sc.validate()

    def test_unsupported_action_type_rejected(self):
        # Phase 1 BioGearsFluidBolus is not valid in sequential scenario
        legacy_bolus = BioGearsFluidBolus(bag_volume_ml=500.0)
        sc = BioGearsSequentialScenario(
            scenario_name="LegacyInSequential",
            actions=[legacy_bolus],  # type: ignore[list-item]
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="Unsupported action type"):
            sc.validate()

    def test_action_starting_before_baseline_stabilization_rejected(self):
        # baseline is 60s, but action starts at 30s
        sc = BioGearsSequentialScenario(
            scenario_name="PreBaselineAction",
            baseline_stabilization_s=60.0,
            actions=[BioGearsSalineBolus(100.0, start_time_s=30.0)],
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="precedes previous action completion or baseline"):
            sc.validate()

    def test_overlapping_actions_rejected(self):
        # Action 0: 60s..180s (100 mL @ 50 mL/min = 120s delivery)
        # Action 1 starts at 150s (overlap of 30s)
        sc = BioGearsSequentialScenario(
            scenario_name="OverlappingActions",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(100.0, start_time_s=150.0),
            ],
        )
        with pytest.raises(BioGearsScenarioBuilderError, match="strictly sequential and non-overlapping"):
            sc.validate()

    def test_zero_baseline_stabilization_rejected(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="baseline_stabilization_s"):
            BioGearsSequentialScenario(
                scenario_name="ZeroBaseline",
                baseline_stabilization_s=0.0,
                actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
            )

    def test_negative_baseline_stabilization_rejected(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="baseline_stabilization_s"):
            BioGearsSequentialScenario(
                scenario_name="NegBaseline",
                baseline_stabilization_s=-10.0,
                actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
            )

    def test_zero_post_observation_rejected(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="post_observation_s"):
            BioGearsSequentialScenario(
                scenario_name="ZeroPostObs",
                post_observation_s=0.0,
                actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
            )

    def test_negative_post_observation_rejected(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="post_observation_s"):
            BioGearsSequentialScenario(
                scenario_name="NegPostObs",
                post_observation_s=-30.0,
                actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
            )

    def test_empty_scenario_name_rejected(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="scenario_name"):
            BioGearsSequentialScenario(
                scenario_name="",
                actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
            )

    def test_empty_patient_file_rejected(self):
        with pytest.raises(BioGearsScenarioBuilderError, match="patient_file"):
            BioGearsSequentialScenario(
                scenario_name="Valid",
                patient_file="",
                actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
            )

    def test_simulation_duration_limit_within_1800s_accepted(self):
        # 60s baseline + 500 mL (600s) + 500 mL (600s) + 250 mL (300s) + 60s post = 1620s <= 1800s
        sc = BioGearsSequentialScenario(
            scenario_name="WithinLimit",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(500.0, start_time_s=60.0),    # 60..660s
                BioGearsSalineBolus(500.0, start_time_s=660.0),   # 660..1260s
                BioGearsSalineBolus(250.0, start_time_s=1260.0),  # 1260..1560s
            ],
        )
        assert sc.total_duration_s == 1620.0
        sc.validate()  # must not raise

    def test_simulation_duration_limit_exceeding_1800s_rejected(self):
        # 60s baseline + 3 * 500 mL (3 * 600s = 1800s) + 60s post = 1920s > 1800s
        sc = BioGearsSequentialScenario(
            scenario_name="ExceedsLimit",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(500.0, start_time_s=60.0),    # 60..660s
                BioGearsSalineBolus(500.0, start_time_s=660.0),   # 660..1260s
                BioGearsSalineBolus(500.0, start_time_s=1260.0),  # 1260..1860s (+ 60s post = 1920s)
            ],
        )
        assert sc.total_duration_s == 1920.0
        with pytest.raises(BioGearsScenarioBuilderError, match="Maximum simulated duration is 1800s"):
            sc.validate()

    def test_mixed_intervention_flagging_and_status(self):
        # Saline + Norepinephrine: defined in builder, but marked [BLOCKED] for execution
        sc_mixed = BioGearsSequentialScenario(
            scenario_name="MixedIntervention",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsNorepinephrineInfusion(start_time_s=240.0),
            ],
        )
        assert sc_mixed.is_mixed_intervention is True
        assert sc_mixed.execution_status == "[BLOCKED]"

        # Pure Saline: unverified for sequential execution until Step B
        sc_saline = BioGearsSequentialScenario(
            scenario_name="PureSaline",
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(250.0, start_time_s=240.0),
            ],
        )
        assert sc_saline.is_mixed_intervention is False
        assert sc_saline.execution_status == "[UNVERIFIED]"


class TestSequentialScenarioXmlGeneration:
    """Tests for generated XML structure, chronological order, and advance-time segments."""

    def test_single_saline_sequential_xml(self):
        sc = BioGearsSequentialScenario(
            scenario_name="SeqSaline100",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )
        xml_str = build_sequential_scenario_xml(sc)
        root = _parse(xml_str)

        # Baseline advance time: 60s
        # Action 0: Saline 100 mL @ 50 mL/min
        # Action 0 delivery advance time: 120s
        # Post-observation advance time: 60s
        # Total sim time = 240s
        assert sc.total_duration_s == 240.0

        actions_el = root.find(f"{{{_NS}}}Actions")
        actions = list(actions_el)
        assert len(actions) == 4

        # 1. Baseline AdvanceTimeData (60s)
        assert actions[0].get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        assert actions[0].find(f"{{{_NS}}}Time").get("value") == "60"

        # 2. Saline SubstanceCompoundInfusionData (100 mL @ 50 mL/min)
        assert actions[1].get(f"{{{_XSI}}}type") == "SubstanceCompoundInfusionData"
        assert actions[1].find(f"{{{_NS}}}SubstanceCompound").text == "Saline"
        assert actions[1].find(f"{{{_NS}}}BagVolume").get("value") == "100"
        assert actions[1].find(f"{{{_NS}}}Rate").get("value") == "50"

        # 3. Delivery AdvanceTimeData (120s)
        assert actions[2].get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        assert actions[2].find(f"{{{_NS}}}Time").get("value") == "120"

        # 4. Post-observation AdvanceTimeData (60s)
        assert actions[3].get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        assert actions[3].find(f"{{{_NS}}}Time").get("value") == "60"

    def test_single_norepinephrine_sequential_xml(self):
        sc = BioGearsSequentialScenario(
            scenario_name="SeqNorepi",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[BioGearsNorepinephrineInfusion(start_time_s=60.0)],
        )
        xml_str = build_sequential_scenario_xml(sc)
        root = _parse(xml_str)

        actions_el = root.find(f"{{{_NS}}}Actions")
        actions = list(actions_el)
        assert len(actions) == 5

        # 1. Baseline AdvanceTimeData (60s)
        assert actions[0].get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        assert actions[0].find(f"{{{_NS}}}Time").get("value") == "60"

        # 2. Norepinephrine SubstanceInfusionData (14 ug/mL @ 1.0 mL/min)
        assert actions[1].get(f"{{{_XSI}}}type") == "SubstanceInfusionData"
        assert actions[1].find(f"{{{_NS}}}Substance").text == "Norepinephrine"
        assert actions[1].find(f"{{{_NS}}}Concentration").get("value") == "14"
        assert actions[1].find(f"{{{_NS}}}Rate").get("value") == "1"

        # 3. Infusion duration AdvanceTimeData (120s)
        assert actions[2].get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        assert actions[2].find(f"{{{_NS}}}Time").get("value") == "120"

        # 4. Stop infusion (rate=0, conc=0)
        assert actions[3].get(f"{{{_XSI}}}type") == "SubstanceInfusionData"
        assert actions[3].find(f"{{{_NS}}}Rate").get("value") == "0"
        assert actions[3].find(f"{{{_NS}}}Concentration").get("value") == "0"

        # 5. Post-observation AdvanceTimeData (60s)
        assert actions[4].get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        assert actions[4].find(f"{{{_NS}}}Time").get("value") == "60"

    def test_multi_action_with_advance_time_gap(self):
        # Baseline: 60s
        # Action 0: Saline 100 mL, starts 60s, delivers in 120s (ends 180s)
        # Advance gap: 60s (action 1 starts at 240s)
        # Action 1: Saline 250 mL, starts 240s, delivers in 300s (ends 540s)
        # Post-observation: 60s (ends 600s)
        sc = BioGearsSequentialScenario(
            scenario_name="TwoBolusesWithGap",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsSalineBolus(250.0, start_time_s=240.0),
            ],
        )
        assert sc.total_duration_s == 600.0

        xml_str = build_sequential_scenario_xml(sc)
        root = _parse(xml_str)
        actions = list(root.find(f"{{{_NS}}}Actions"))

        # Expected sequence:
        # 0: AdvanceTime (60s) - Baseline
        # 1: Saline 100 mL
        # 2: AdvanceTime (120s) - Action 0 delivery
        # 3: AdvanceTime (60s) - Gap between Action 0 and Action 1
        # 4: Saline 250 mL
        # 5: AdvanceTime (300s) - Action 1 delivery
        # 6: AdvanceTime (60s) - Post-observation
        assert len(actions) == 7

        assert actions[0].find(f"{{{_NS}}}Time").get("value") == "60"
        assert actions[1].find(f"{{{_NS}}}BagVolume").get("value") == "100"
        assert actions[2].find(f"{{{_NS}}}Time").get("value") == "120"
        assert actions[3].find(f"{{{_NS}}}Time").get("value") == "60"
        assert actions[4].find(f"{{{_NS}}}BagVolume").get("value") == "250"
        assert actions[5].find(f"{{{_NS}}}Time").get("value") == "300"
        assert actions[6].find(f"{{{_NS}}}Time").get("value") == "60"

        # Verify sum of all advance times equals total_duration_s
        advance_times = [
            float(a.find(f"{{{_NS}}}Time").get("value"))
            for a in actions
            if a.get(f"{{{_XSI}}}type") == "AdvanceTimeData"
        ]
        assert sum(advance_times) == pytest.approx(600.0)

    def test_timeline_milestones_structure(self):
        sc = BioGearsSequentialScenario(
            scenario_name="MilestoneTest",
            baseline_stabilization_s=60.0,
            post_observation_s=60.0,
            actions=[
                BioGearsSalineBolus(100.0, start_time_s=60.0),
                BioGearsNorepinephrineInfusion(start_time_s=240.0),
            ],
        )
        milestones = sc.get_timeline_milestones()
        assert len(milestones) == 5

        # Milestone 0: baseline (0..60s)
        assert milestones[0]["phase"] == "baseline"
        assert milestones[0]["start_s"] == 0.0
        assert milestones[0]["end_s"] == 60.0

        # Milestone 1: saline intervention (60..180s)
        assert milestones[1]["phase"] == "intervention_0"
        assert milestones[1]["start_s"] == 60.0
        assert milestones[1]["end_s"] == 180.0
        assert milestones[1]["action_type"] == "saline"

        # Milestone 2: intermediate hold (180..240s)
        assert milestones[2]["start_s"] == 180.0
        assert milestones[2]["end_s"] == 240.0
        assert milestones[2]["duration_s"] == 60.0

        # Milestone 3: norepi intervention (240..360s)
        assert milestones[3]["phase"] == "intervention_1"
        assert milestones[3]["start_s"] == 240.0
        assert milestones[3]["end_s"] == 360.0
        assert milestones[3]["action_type"] == "norepinephrine"

        # Milestone 4: post-intervention observation (360..420s)
        assert milestones[4]["phase"] == "post_observation"
        assert milestones[4]["start_s"] == 360.0
        assert milestones[4]["end_s"] == 420.0
        assert milestones[4]["duration_s"] == 60.0

    def test_build_scenario_xml_dispatches_sequential_scenario(self):
        sc = BioGearsSequentialScenario(
            scenario_name="DispatchTest",
            actions=[BioGearsSalineBolus(100.0, start_time_s=60.0)],
        )
        xml_via_main = build_scenario_xml(sc)
        xml_via_seq = build_sequential_scenario_xml(sc)
        assert xml_via_main == xml_via_seq

