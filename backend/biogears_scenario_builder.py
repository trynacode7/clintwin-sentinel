"""
biogears_scenario_builder.py — BioGears Scenario XML Builder (Phase 1)
=======================================================================
SCOPE: Translate ClinTwin intervention dataclasses into valid BioGears
scenario XML strings. This is a pure XML builder — no file I/O, no
subprocess calls, no BioGears execution.

Phase 1 supports BioGearsFluidBolus ONLY. Any other action type passed to
build_scenario_xml() raises BioGearsScenarioBuilderError immediately.

Decision baked in (approved 2026-08-26):
  Fluid infusion rate default: 50.0 mL/min
  Source: C:\\BioGears\\bin\\Scenarios\\Patient\\SepticShock_Treatment.xml
          <Rate value="50" unit="mL/min"/> (500 mL bolus actions)

XML namespace and element structure verified from installed BioGears files:
  AcuteRespiratoryDistress_Ventilation.xml, IVFluids.xml, SalineInfusion.xml,
  SepticShock_Treatment.xml, MainstemIntubation.xml (namespace / contentVersion)

This module does NOT:
  - Execute BioGears (use BioGearsBridge.run_scenario_file() for that)
  - Modify whatif.py, api.py, the frontend, or any ML code
  - Generate vasopressor, ventilator, or FiO2 actions (pending Phase 2+)
  - Invent clinical values not present in verified BioGears example files
  - Prepend a stabilization AdvanceTimeData action (BioGears performs its
    own patient stabilization; verified by ClinTwinTinyTest runs 2026-08-26)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

# ─── Namespaces ───────────────────────────────────────────────────────────────
# Verified from installed BioGears scenario XML files (contentVersion BioGears_6.3.0-beta)

_NS = "uri:/mil/tatrc/physiology/datamodel"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"

ET.register_namespace("", _NS)
ET.register_namespace("xsi", _XSI)


def _tag(name: str) -> str:
    """Return a namespace-qualified element name for the BioGears data model."""
    return f"{{{_NS}}}{name}"


def _xsi(attr: str) -> str:
    """Return a namespace-qualified xsi attribute name."""
    return f"{{{_XSI}}}{attr}"


# ─── Exception ────────────────────────────────────────────────────────────────


class BioGearsScenarioBuilderError(Exception):
    """Raised when a BioGears scenario cannot be built from the given inputs.

    Always carries a specific, actionable message identifying which field
    failed and why. Callers should treat this as a programming error in the
    scenario specification, not a transient failure.
    """


# ─── Intervention dataclass — Phase 1 ────────────────────────────────────────


@dataclass(frozen=True)
class BioGearsFluidBolus:
    """IV saline bolus — maps to BioGears SubstanceCompoundInfusionData.

    This is the Phase 1 intervention. It is the only action type accepted
    by build_scenario_xml() in this version.

    Attributes:
        bag_volume_ml: Total fluid volume in mL.
            Mapped directly from ClinTwin's fluid_bolus_ml feature.
            Valid range: (0, 2000] mL  (ClinTwin's declared max is 2000 mL).

        rate_ml_per_min: Infusion rate in mL/min.
            Default: 50.0 mL/min
            Source: SepticShock_Treatment.xml — <Rate value="50" unit="mL/min"/>
                    Used for 500 mL bolus actions in that scenario (approved
                    2026-08-26 as the verified reference value for Phase 1).
            Valid range: > 0 mL/min.

    Substance compound is always "Saline" (Normal Saline).
    Verified: C:\\BioGears\\bin\\substances\\Saline.xml
              Na 0.354 g/dL, Cl 0.546 g/dL.
    """

    bag_volume_ml: float
    rate_ml_per_min: float = 50.0  # VERIFIED: SepticShock_Treatment.xml

    @property
    def delivery_time_s(self) -> float:
        """Seconds required to deliver the full bag at the configured rate."""
        return (self.bag_volume_ml / self.rate_ml_per_min) * 60.0


# ─── Scenario dataclass ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class BioGearsScenario:
    """A complete Phase 1 BioGears scenario ready to be serialized to XML.

    Attributes:
        patient_file: Filename of the BioGears patient file, e.g.
            "StandardMale.xml". BioGears resolves it relative to its own
            patient directory. Must be non-empty.

        scenario_name: Human-readable name embedded in the <Name> element.
            Must be non-empty.

        sim_duration_s: Total simulation time in seconds that BioGears will
            advance after the intervention action(s) are applied.
            Must be > 0 and must be ≥ the delivery time of every contained
            BioGearsFluidBolus (bag_volume_ml / rate_ml_per_min * 60).
            If sim_duration_s is shorter than the bolus delivery, BioGears
            would terminate the infusion mid-bag; build_scenario_xml() raises
            BioGearsScenarioBuilderError rather than silently accepting this.

        samples_per_second: DataRequests sampling rate. Must be > 0.
            Lower values (e.g. 0.2) produce fewer rows in the output CSV.

        actions: Ordered list of BioGearsFluidBolus instances.
            Phase 1 only — any other type in this list causes
            build_scenario_xml() to raise BioGearsScenarioBuilderError.
            Must contain at least one action.
    """

    patient_file: str
    scenario_name: str
    sim_duration_s: float
    samples_per_second: float
    actions: List[BioGearsFluidBolus]


# ─── DataRequests columns ─────────────────────────────────────────────────────
# Core physiology columns included in all Phase 1 fluid-bolus scenarios.
# Verified present in IVFluids.xml DataRequests block.
# Format: (Name, Unit, Precision, xsi:type value)

_FLUID_DATA_REQUESTS = [
    ("HeartRate",                "1/min",   "2", "PhysiologyDataRequestData"),
    ("BloodVolume",              "L",       "2", "PhysiologyDataRequestData"),
    ("MeanArterialPressure",     "mmHg",    "1", "PhysiologyDataRequestData"),
    ("SystolicArterialPressure", "mmHg",    "0", "PhysiologyDataRequestData"),
    ("DiastolicArterialPressure","mmHg",    "1", "PhysiologyDataRequestData"),
    ("CardiacOutput",            "L/min",   "2", "PhysiologyDataRequestData"),
    ("OxygenSaturation",         "unitless","3", "PhysiologyDataRequestData"),
    ("RespirationRate",          "1/min",   "2", "PhysiologyDataRequestData"),
]


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    """Format a float for a BioGears XML value= attribute.

    Produces the minimal representation: whole numbers without a decimal
    point (matching the BioGears installed-file convention), fractional
    values with full precision.

    Examples:
        50.0  → "50"
        500.0 → "500"
        0.016 → "0.016"
        600.0 → "600"
    """
    if abs(v) < 1e15 and v == int(v):
        return str(int(v))
    return repr(v)


def _validate_fluid_bolus(bolus: BioGearsFluidBolus, index: int) -> None:
    """Validate one BioGearsFluidBolus. Raises BioGearsScenarioBuilderError."""
    prefix = f"BioGearsFluidBolus (actions[{index}])"
    if bolus.bag_volume_ml <= 0:
        raise BioGearsScenarioBuilderError(
            f"{prefix}.bag_volume_ml must be > 0, got {bolus.bag_volume_ml}."
        )
    if bolus.bag_volume_ml > 2000:
        raise BioGearsScenarioBuilderError(
            f"{prefix}.bag_volume_ml must be ≤ 2000 mL "
            f"(ClinTwin declared max), got {bolus.bag_volume_ml}."
        )
    if bolus.rate_ml_per_min <= 0:
        raise BioGearsScenarioBuilderError(
            f"{prefix}.rate_ml_per_min must be > 0, got {bolus.rate_ml_per_min}."
        )


def _validate_scenario(scenario: BioGearsScenario) -> None:
    """Validate all fields of a BioGearsScenario.

    Raises BioGearsScenarioBuilderError with a specific message on failure.
    Checks fields in dependency order so the first failure is the most
    actionable one.
    """
    if not scenario.patient_file or not scenario.patient_file.strip():
        raise BioGearsScenarioBuilderError(
            "BioGearsScenario.patient_file must be a non-empty string."
        )
    if not scenario.scenario_name or not scenario.scenario_name.strip():
        raise BioGearsScenarioBuilderError(
            "BioGearsScenario.scenario_name must be a non-empty string."
        )
    if scenario.sim_duration_s <= 0:
        raise BioGearsScenarioBuilderError(
            f"BioGearsScenario.sim_duration_s must be > 0, "
            f"got {scenario.sim_duration_s}."
        )
    if scenario.samples_per_second <= 0:
        raise BioGearsScenarioBuilderError(
            f"BioGearsScenario.samples_per_second must be > 0, "
            f"got {scenario.samples_per_second}."
        )
    if not scenario.actions:
        raise BioGearsScenarioBuilderError(
            "BioGearsScenario.actions must contain at least one action."
        )

    for i, action in enumerate(scenario.actions):
        if not isinstance(action, BioGearsFluidBolus):
            raise BioGearsScenarioBuilderError(
                f"Unsupported action type {type(action).__name__!r} at "
                f"actions[{i}]. Phase 1 of the BioGears scenario builder "
                f"supports BioGearsFluidBolus only. Other action types "
                f"(vasopressor, ventilator, FiO2) will be added in future "
                f"phases once their clinical parameter decisions are resolved."
            )
        _validate_fluid_bolus(action, index=i)

    # Verify sim_duration_s covers complete delivery of every bolus.
    # A sim that ends before the bag is empty would truncate the infusion;
    # this is almost certainly a caller error, so we reject it explicitly.
    for i, action in enumerate(scenario.actions):
        delivery_s = action.delivery_time_s
        if scenario.sim_duration_s < delivery_s:
            raise BioGearsScenarioBuilderError(
                f"BioGearsScenario.sim_duration_s ({scenario.sim_duration_s}s) "
                f"is less than the time required to deliver actions[{i}] "
                f"({action.bag_volume_ml} mL at {action.rate_ml_per_min} mL/min "
                f"= {delivery_s:.1f}s). Increase sim_duration_s to at least "
                f"{delivery_s:.1f}s so the bolus completes within the simulation."
            )


def _emit_fluid_bolus(actions_el: ET.Element, bolus: BioGearsFluidBolus) -> None:
    """Append a SubstanceCompoundInfusionData action element to actions_el.

    XML structure verified from SepticShock_Treatment.xml and IVFluids.xml:
      <Action xsi:type="SubstanceCompoundInfusionData">
        <SubstanceCompound>Saline</SubstanceCompound>
        <BagVolume value="..." unit="mL"/>
        <Rate value="..." unit="mL/min"/>
      </Action>
    """
    action_el = ET.SubElement(actions_el, _tag("Action"))
    action_el.set(_xsi("type"), "SubstanceCompoundInfusionData")
    ET.SubElement(action_el, _tag("SubstanceCompound")).text = "Saline"
    bag_el = ET.SubElement(action_el, _tag("BagVolume"))
    bag_el.set("value", _fmt(bolus.bag_volume_ml))
    bag_el.set("unit", "mL")
    rate_el = ET.SubElement(action_el, _tag("Rate"))
    rate_el.set("value", _fmt(bolus.rate_ml_per_min))
    rate_el.set("unit", "mL/min")


# ─── Primary public API ───────────────────────────────────────────────────────


def build_scenario_xml(scenario: BioGearsScenario) -> str:
    """Build a complete BioGears scenario XML string from a BioGearsScenario.

    This is a PURE FUNCTION — no file I/O, no subprocess calls, no side
    effects. The same input always produces the same output. Tests can call
    it without any filesystem or BioGears dependency.

    The returned XML string can be written to a file and passed directly to
    BioGearsBridge.run_scenario_file().

    No stabilization AdvanceTimeData is prepended. BioGears performs patient
    stabilization internally before executing scenario actions. This was
    verified by ClinTwinTinyTest runs conducted 2026-08-26.

    XML structure produced:
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Scenario xmlns="uri:/mil/tatrc/physiology/datamodel"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  contentVersion="BioGears_6.3.0-beta"
                  xsi:schemaLocation="">
          <Name>{scenario_name}</Name>
          <Description>...</Description>
          <InitialParameters>
            <PatientFile>{patient_file}</PatientFile>
          </InitialParameters>
          <DataRequests SamplesPerSecond="{samples_per_second}">
            <!-- HeartRate, BloodVolume, MAP, SBP, DBP,
                 CardiacOutput, OxygenSaturation, RespirationRate -->
          </DataRequests>
          <Actions>
            <!-- one SubstanceCompoundInfusionData per BioGearsFluidBolus -->
            <Action xsi:type="AdvanceTimeData">
              <Time value="{sim_duration_s}" unit="s"/>
            </Action>
          </Actions>
        </Scenario>

    Args:
        scenario: Fully specified BioGearsScenario. All actions must be
            BioGearsFluidBolus instances.

    Returns:
        Complete, well-formed BioGears scenario XML as a string.

    Raises:
        BioGearsScenarioBuilderError: on any validation failure.
    """
    _validate_scenario(scenario)

    # ── Root ─────────────────────────────────────────────────────────────────
    root = ET.Element(_tag("Scenario"))
    root.set("contentVersion", "BioGears_6.3.0-beta")
    root.set(_xsi("schemaLocation"), "")

    ET.SubElement(root, _tag("Name")).text = scenario.scenario_name
    ET.SubElement(root, _tag("Description")).text = (
        "ClinTwin BioGears fluid bolus scenario - "
        "generated by biogears_scenario_builder.py"
    )

    # ── InitialParameters ────────────────────────────────────────────────────
    init_params = ET.SubElement(root, _tag("InitialParameters"))
    ET.SubElement(init_params, _tag("PatientFile")).text = scenario.patient_file

    # ── DataRequests ─────────────────────────────────────────────────────────
    data_requests_el = ET.SubElement(root, _tag("DataRequests"))
    data_requests_el.set("SamplesPerSecond", _fmt(scenario.samples_per_second))

    for name, unit, precision, req_type in _FLUID_DATA_REQUESTS:
        dr = ET.SubElement(data_requests_el, _tag("DataRequest"))
        dr.set(_xsi("type"), req_type)
        dr.set("Name", name)
        dr.set("Unit", unit)
        dr.set("Precision", precision)

    # ── Actions ───────────────────────────────────────────────────────────────
    actions_el = ET.SubElement(root, _tag("Actions"))

    for action in scenario.actions:
        # Type already checked in _validate_scenario(); this guards against
        # future callers who bypass validation or subclass the dataclass.
        if not isinstance(action, BioGearsFluidBolus):  # pragma: no branch
            raise BioGearsScenarioBuilderError(
                f"Unsupported action type {type(action).__name__!r}. "
                f"Phase 1 supports BioGearsFluidBolus only."
            )
        _emit_fluid_bolus(actions_el, action)

    # ── AdvanceTimeData — simulation window ───────────────────────────────────
    advance_el = ET.SubElement(actions_el, _tag("Action"))
    advance_el.set(_xsi("type"), "AdvanceTimeData")
    time_el = ET.SubElement(advance_el, _tag("Time"))
    time_el.set("value", _fmt(scenario.sim_duration_s))
    time_el.set("unit", "s")

    # ── Serialize ─────────────────────────────────────────────────────────────
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + body
