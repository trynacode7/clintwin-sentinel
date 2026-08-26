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
