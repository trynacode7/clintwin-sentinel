"""
biogears_scenario_builder.py — BioGears Scenario XML Builder (Phase D)
=======================================================================
SCOPE: Translate ClinTwin intervention dataclasses into valid BioGears
scenario XML strings. This is a pure XML builder — no file I/O, no
subprocess calls, no BioGears execution.

Phase D Step A: Generalized BioGears Scenario Builder
Supports:
  1. Phase 1 legacy single-bolus: BioGearsScenario + BioGearsFluidBolus (100% preserved)
  2. Sequential multi-action: BioGearsSequentialScenario supporting 1 to 3 sequential actions
     with baseline observation, intermediate observation gaps, and post-intervention observation.

Epistemic Status:
  [VERIFIED]:
    - BioGears 100 mL Saline bolus parameter set (100 mL @ 50 mL/min, 180s sim window)
      verified in Phase B live execution.
    - Norepinephrine parameter values (concentration = 14 ug/mL, rate = 1.0 mL/min)
      verified from Phase B reference configuration.
  [UNVERIFIED]:
    - Norepinephrine infusion duration termination via rate=0 requires controlled
      BioGears verification in Phase D Step H before relying on it for bounded infusion.
    - Multi-action sequential execution in live BioGears requires Step B verification.
  [BLOCKED]:
    - Mixed interventions (Saline -> Norepinephrine) execution and UI exposure
      remain BLOCKED pending explicit execution and verification in Phase D Step I.

Runtime & Simulation Limits:
  - Maximum configured simulated duration: 1800.0s (simulated time limit, NOT a wall-clock guarantee).
  - "Maximum simulated duration is 1800s, but wall-clock runtime is scenario-dependent
     and may exceed several minutes."
    (Phase B benchmarks: 180s sim ~87.7s, 600s sim ~187.3s, 1800s sim timed out at 600s).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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
    by build_scenario_xml() for BioGearsScenario in this version.

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


# ─── Sequential Interventions Dataclasses (Phase D) ──────────────────────────


VERIFIED_SALINE_VOLUMES = {100.0, 250.0, 500.0}
VERIFIED_SALINE_RATE = 50.0

MIN_SALINE_VOLUME_ML = 50.0
MAX_SALINE_VOLUME_ML = 1000.0
MIN_SALINE_RATE_ML_PER_MIN = 10.0
MAX_SALINE_RATE_ML_PER_MIN = 100.0
DEFAULT_SALINE_VOLUME_ML = 250.0
DEFAULT_SALINE_RATE_ML_PER_MIN = 50.0

VERIFIED_NOREPINEPHRINE_CONCENTRATION = 14.0  # ug/mL (standard stock dilution from SepticShock reference)
DEFAULT_NOREPINEPHRINE_CONCENTRATION_UG_ML = 14.0

MIN_NOREPINEPHRINE_RATE_ML_PER_MIN = 0.1
MAX_NOREPINEPHRINE_RATE_ML_PER_MIN = 5.0
DEFAULT_NOREPINEPHRINE_RATE_ML_PER_MIN = 1.0
VERIFIED_NOREPINEPHRINE_RATE = 1.0           # backward-compatibility alias

MIN_NOREPINEPHRINE_DURATION_S = 10.0
MAX_NOREPINEPHRINE_DURATION_S = 600.0
DEFAULT_NOREPINEPHRINE_DURATION_S = 120.0
VERIFIED_NOREPINEPHRINE_DURATION = 120.0     # backward-compatibility alias


@dataclass(frozen=True)
class BioGearsSalineBolus:
    """IV Saline Bolus for sequential What-If scenarios.

    Maps to BioGears SubstanceCompoundInfusionData.
    Supported bag volumes: 50.0 to 1000.0 mL (verified presets: 100, 250, 500 mL).
    Supported infusion rate: 10.0 to 100.0 mL/min (default: 50.0 mL/min).
    """

    bag_volume_ml: float = DEFAULT_SALINE_VOLUME_ML
    rate_ml_per_min: float = DEFAULT_SALINE_RATE_ML_PER_MIN
    start_time_s: float = 60.0

    def __post_init__(self):
        import math
        if not isinstance(self.bag_volume_ml, (int, float)) or math.isnan(self.bag_volume_ml) or math.isinf(self.bag_volume_ml) or self.bag_volume_ml <= 0:
            raise BioGearsScenarioBuilderError(
                f"BioGearsSalineBolus: bag_volume_ml must be a positive finite number, got {self.bag_volume_ml!r}."
            )
        if self.bag_volume_ml < MIN_SALINE_VOLUME_ML or self.bag_volume_ml > MAX_SALINE_VOLUME_ML:
            raise BioGearsScenarioBuilderError(
                f"BioGearsSalineBolus: bag_volume_ml {self.bag_volume_ml} is outside approved bounds "
                f"[{MIN_SALINE_VOLUME_ML}, {MAX_SALINE_VOLUME_ML}] mL."
            )
        if not isinstance(self.rate_ml_per_min, (int, float)) or math.isnan(self.rate_ml_per_min) or math.isinf(self.rate_ml_per_min) or self.rate_ml_per_min <= 0:
            raise BioGearsScenarioBuilderError(
                f"BioGearsSalineBolus: rate_ml_per_min must be a positive finite number, got {self.rate_ml_per_min!r}."
            )
        if self.rate_ml_per_min < MIN_SALINE_RATE_ML_PER_MIN or self.rate_ml_per_min > MAX_SALINE_RATE_ML_PER_MIN:
            raise BioGearsScenarioBuilderError(
                f"BioGearsSalineBolus: rate_ml_per_min {self.rate_ml_per_min} is outside approved bounds "
                f"[{MIN_SALINE_RATE_ML_PER_MIN}, {MAX_SALINE_RATE_ML_PER_MIN}] mL/min."
            )
        if not isinstance(self.start_time_s, (int, float)) or math.isnan(self.start_time_s) or math.isinf(self.start_time_s) or self.start_time_s < 0:
            raise BioGearsScenarioBuilderError(
                f"BioGearsSalineBolus: start_time_s must be non-negative and finite, got {self.start_time_s!r}."
            )

    @property
    def delivery_duration_s(self) -> float:
        """Seconds required to deliver the full bag at 50 mL/min."""
        return (self.bag_volume_ml / self.rate_ml_per_min) * 60.0

    @property
    def delivery_time_s(self) -> float:
        """Alias for delivery_duration_s for compatibility with BioGearsFluidBolus."""
        return self.delivery_duration_s


@dataclass(frozen=True)
class BioGearsNorepinephrineInfusion:
    """Configurable Norepinephrine Infusion for sequential What-If scenarios.

    Maps to BioGears SubstanceInfusionData with Norepinephrine substance.
    Supported infusion rates: 0.1 to 5.0 mL/min (default: 1.0 mL/min).
    Supported infusion durations: 10.0 to 600.0 s (default: 120.0 s).
    Concentration is standard stock dilution locked to 14.0 ug/mL.
    """

    start_time_s: float = 60.0
    concentration_ug_ml: float = DEFAULT_NOREPINEPHRINE_CONCENTRATION_UG_ML
    rate_ml_per_min: float = DEFAULT_NOREPINEPHRINE_RATE_ML_PER_MIN
    duration_s: float = DEFAULT_NOREPINEPHRINE_DURATION_S

    def __post_init__(self):
        import math
        if not isinstance(self.start_time_s, (int, float)) or math.isnan(self.start_time_s) or math.isinf(self.start_time_s) or self.start_time_s < 0:
            raise BioGearsScenarioBuilderError(
                f"BioGearsNorepinephrineInfusion: start_time_s must be non-negative and finite, got {self.start_time_s!r}."
            )
        if self.concentration_ug_ml != VERIFIED_NOREPINEPHRINE_CONCENTRATION:
            raise BioGearsScenarioBuilderError(
                f"BioGearsNorepinephrineInfusion: concentration_ug_ml {self.concentration_ug_ml} is unsupported. "
                f"BioGears substance models are calibrated to standard stock dilution of {VERIFIED_NOREPINEPHRINE_CONCENTRATION} ug/mL. "
                f"Titration must be achieved via rate_ml_per_min and duration_s."
            )
        if not isinstance(self.rate_ml_per_min, (int, float)) or math.isnan(self.rate_ml_per_min) or math.isinf(self.rate_ml_per_min) or self.rate_ml_per_min <= 0:
            raise BioGearsScenarioBuilderError(
                f"BioGearsNorepinephrineInfusion: rate_ml_per_min must be a positive finite number, got {self.rate_ml_per_min!r}."
            )
        if self.rate_ml_per_min < MIN_NOREPINEPHRINE_RATE_ML_PER_MIN or self.rate_ml_per_min > MAX_NOREPINEPHRINE_RATE_ML_PER_MIN:
            raise BioGearsScenarioBuilderError(
                f"BioGearsNorepinephrineInfusion: rate_ml_per_min {self.rate_ml_per_min} is outside supported range "
                f"[{MIN_NOREPINEPHRINE_RATE_ML_PER_MIN}, {MAX_NOREPINEPHRINE_RATE_ML_PER_MIN}] mL/min."
            )
        if not isinstance(self.duration_s, (int, float)) or math.isnan(self.duration_s) or math.isinf(self.duration_s) or self.duration_s <= 0:
            raise BioGearsScenarioBuilderError(
                f"BioGearsNorepinephrineInfusion: duration_s must be a positive finite number, got {self.duration_s!r}."
            )
        if self.duration_s < MIN_NOREPINEPHRINE_DURATION_S or self.duration_s > MAX_NOREPINEPHRINE_DURATION_S:
            raise BioGearsScenarioBuilderError(
                f"BioGearsNorepinephrineInfusion: duration_s {self.duration_s} is outside supported range "
                f"[{MIN_NOREPINEPHRINE_DURATION_S}, {MAX_NOREPINEPHRINE_DURATION_S}] s."
            )

    @property
    def delivery_duration_s(self) -> float:
        """Active infusion duration in seconds."""
        return self.duration_s

    @property
    def delivery_time_s(self) -> float:
        """Alias for delivery_duration_s for compatibility with BioGearsFluidBolus."""
        return self.duration_s


# Union of valid sequential actions
SequentialAction = Union[BioGearsSalineBolus, BioGearsNorepinephrineInfusion]


@dataclass(frozen=True)
class BioGearsSequentialScenario:
    """A multi-intervention BioGears scenario with sequential timing.

    Supports:
      - Baseline observation period.
      - 1 to 3 sequential interventions on a continuous BioGears timeline.
      - Explicit intervention start times and durations.
      - Advance-time segments between interventions.
      - Intermediate observation points and a final observation period.

    Timeline progresses continuously without resetting or restarting the engine.
    Maximum simulated duration is 1800s, but wall-clock runtime is
    scenario-dependent and may exceed several minutes.
    """

    scenario_name: str
    patient_file: str = "StandardMale.xml"
    samples_per_second: float = 0.2
    baseline_stabilization_s: float = 60.0
    post_observation_s: float = 60.0
    actions: List[SequentialAction] = None

    def __post_init__(self):
        if self.actions is None:
            object.__setattr__(self, "actions", [])
        if not isinstance(self.scenario_name, str) or not self.scenario_name.strip():
            raise BioGearsScenarioBuilderError("scenario_name must be a non-empty string.")
        if not isinstance(self.patient_file, str) or not self.patient_file.strip():
            raise BioGearsScenarioBuilderError("patient_file must be a non-empty string.")
        if not isinstance(self.baseline_stabilization_s, (int, float)) or self.baseline_stabilization_s <= 0:
            raise BioGearsScenarioBuilderError(
                f"baseline_stabilization_s must be positive (> 0), got {self.baseline_stabilization_s!r}."
            )
        if not isinstance(self.post_observation_s, (int, float)) or self.post_observation_s <= 0:
            raise BioGearsScenarioBuilderError(
                f"post_observation_s must be positive (> 0), got {self.post_observation_s!r}."
            )
        if not isinstance(self.samples_per_second, (int, float)) or self.samples_per_second <= 0:
            raise BioGearsScenarioBuilderError(
                f"samples_per_second must be > 0, got {self.samples_per_second!r}."
            )

    def validate(self) -> None:
        """Validate all rules for this sequential scenario."""
        _validate_sequential_scenario(self)

    @property
    def execution_status(self) -> str:
        """Return the epistemic status of this scenario specification for execution.

        Returns:
          '[BLOCKED]' if mixed-intervention (Saline + Norepinephrine),
          '[UNVERIFIED]' for sequential execution pending live BioGears execution verification.
        """
        if self.is_mixed_intervention:
            return "[BLOCKED]"
        return "[UNVERIFIED]"

    @property
    def total_duration_s(self) -> float:
        """Total configured simulated time in seconds."""
        if not self.actions:
            return float(self.baseline_stabilization_s + self.post_observation_s)
        last_action = self.actions[-1]
        return float(last_action.start_time_s + last_action.delivery_duration_s + self.post_observation_s)

    @property
    def is_mixed_intervention(self) -> bool:
        """Return True if scenario contains both Saline and Norepinephrine actions.

        NOTE: Mixed intervention execution in live BioGears is [BLOCKED] pending
        explicit execution verification in Phase D Step I. The builder defines the
        structures, but mixed execution is NOT yet verified.
        """
        has_saline = any(isinstance(a, BioGearsSalineBolus) for a in self.actions)
        has_norepi = any(isinstance(a, BioGearsNorepinephrineInfusion) for a in self.actions)
        return bool(has_saline and has_norepi)

    def get_timeline_milestones(self) -> List[Dict[str, Any]]:
        """Return structured timeline milestones for intermediate observations and transitions.

        Returns a list of milestone dictionaries detailing:
          - baseline observation period
          - each intervention's active delivery window
          - advance-time observation gaps between interventions
          - final post-intervention observation window
        """
        milestones = []
        milestones.append({
            "phase": "baseline",
            "start_s": 0.0,
            "end_s": float(self.baseline_stabilization_s),
            "duration_s": float(self.baseline_stabilization_s),
            "description": f"Baseline observation ({self.baseline_stabilization_s}s)",
        })

        current_t = float(self.baseline_stabilization_s)
        for i, action in enumerate(self.actions):
            if action.start_time_s > current_t:
                gap = float(action.start_time_s - current_t)
                milestones.append({
                    "phase": f"intermediate_hold_{i-1}_to_{i}" if i > 0 else "pre_action_0_hold",
                    "start_s": current_t,
                    "end_s": float(action.start_time_s),
                    "duration_s": gap,
                    "description": f"Intermediate observation hold before action {i} ({gap}s)",
                })
                current_t = float(action.start_time_s)

            action_type = "saline" if isinstance(action, BioGearsSalineBolus) else "norepinephrine"
            delivery_s = float(action.delivery_duration_s)
            end_t = current_t + delivery_s
            milestones.append({
                "phase": f"intervention_{i}",
                "action_index": i,
                "action_type": action_type,
                "start_s": current_t,
                "end_s": end_t,
                "duration_s": delivery_s,
                "action": action,
                "description": f"Action {i} ({action_type}): active delivery ({delivery_s}s)",
            })
            current_t = end_t

        if self.post_observation_s > 0:
            milestones.append({
                "phase": "post_observation",
                "start_s": current_t,
                "end_s": float(current_t + self.post_observation_s),
                "duration_s": float(self.post_observation_s),
                "description": f"Final post-intervention observation ({self.post_observation_s}s)",
            })

        return milestones


# ─── Scenario dataclass (Phase 1 legacy) ──────────────────────────────────────



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
    if isinstance(scenario, BioGearsSequentialScenario):
        return build_sequential_scenario_xml(scenario)

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


# ─── Sequential Scenario Building & Validation ───────────────────────────────


def _validate_sequential_scenario(scenario: BioGearsSequentialScenario) -> None:
    """Validate all constraints of a BioGearsSequentialScenario.

    Rules:
      1. patient_file and scenario_name non-empty.
      2. samples_per_second > 0.
      3. Exactly 1 to 3 actions (maximum 3 interventions).
      4. Saline volume strictly in {100.0, 250.0, 500.0} mL, rate = 50.0 mL/min.
      5. Norepinephrine conc = 14.0 ug/mL, rate = 1.0 mL/min, duration = 120.0s.
      6. Chronological, non-overlapping action progression.
      7. Maximum simulated duration <= 1800s.
         (Maximum simulated duration is 1800s, but wall-clock runtime is
          scenario-dependent and may exceed several minutes.)
    """
    if not scenario.patient_file or not scenario.patient_file.strip():
        raise BioGearsScenarioBuilderError("patient_file must be a non-empty string.")
    if not scenario.scenario_name or not scenario.scenario_name.strip():
        raise BioGearsScenarioBuilderError("scenario_name must be a non-empty string.")
    if scenario.samples_per_second <= 0:
        raise BioGearsScenarioBuilderError(
            f"samples_per_second must be > 0, got {scenario.samples_per_second}."
        )
    if scenario.baseline_stabilization_s <= 0:
        raise BioGearsScenarioBuilderError(
            f"baseline_stabilization_s must be > 0, got {scenario.baseline_stabilization_s}."
        )
    if scenario.post_observation_s <= 0:
        raise BioGearsScenarioBuilderError(
            f"post_observation_s must be > 0, got {scenario.post_observation_s}."
        )
    if not scenario.actions or len(scenario.actions) == 0:
        raise BioGearsScenarioBuilderError(
            "BioGearsSequentialScenario must contain at least 1 action."
        )
    if len(scenario.actions) > 3:
        raise BioGearsScenarioBuilderError(
            f"BioGearsSequentialScenario supports a maximum of 3 actions, got {len(scenario.actions)}."
        )

    current_t = scenario.baseline_stabilization_s
    for i, action in enumerate(scenario.actions):
        if isinstance(action, BioGearsSalineBolus):
            if action.bag_volume_ml < MIN_SALINE_VOLUME_ML or action.bag_volume_ml > MAX_SALINE_VOLUME_ML:
                raise BioGearsScenarioBuilderError(
                    f"actions[{i}]: bag_volume_ml {action.bag_volume_ml} is outside approved bounds "
                    f"[{MIN_SALINE_VOLUME_ML}, {MAX_SALINE_VOLUME_ML}] mL."
                )
            if action.rate_ml_per_min < MIN_SALINE_RATE_ML_PER_MIN or action.rate_ml_per_min > MAX_SALINE_RATE_ML_PER_MIN:
                raise BioGearsScenarioBuilderError(
                    f"actions[{i}]: rate_ml_per_min {action.rate_ml_per_min} is outside approved bounds "
                    f"[{MIN_SALINE_RATE_ML_PER_MIN}, {MAX_SALINE_RATE_ML_PER_MIN}] mL/min."
                )
        elif isinstance(action, BioGearsNorepinephrineInfusion):
            if action.concentration_ug_ml != VERIFIED_NOREPINEPHRINE_CONCENTRATION:
                raise BioGearsScenarioBuilderError(
                    f"actions[{i}]: concentration_ug_ml {action.concentration_ug_ml} is unsupported. "
                    f"Supported concentration: {VERIFIED_NOREPINEPHRINE_CONCENTRATION} ug/mL."
                )
            if action.rate_ml_per_min < MIN_NOREPINEPHRINE_RATE_ML_PER_MIN or action.rate_ml_per_min > MAX_NOREPINEPHRINE_RATE_ML_PER_MIN:
                raise BioGearsScenarioBuilderError(
                    f"actions[{i}]: rate_ml_per_min {action.rate_ml_per_min} is outside supported range "
                    f"[{MIN_NOREPINEPHRINE_RATE_ML_PER_MIN}, {MAX_NOREPINEPHRINE_RATE_ML_PER_MIN}] mL/min."
                )
            if action.duration_s < MIN_NOREPINEPHRINE_DURATION_S or action.duration_s > MAX_NOREPINEPHRINE_DURATION_S:
                raise BioGearsScenarioBuilderError(
                    f"actions[{i}]: duration_s {action.duration_s} is outside supported range "
                    f"[{MIN_NOREPINEPHRINE_DURATION_S}, {MAX_NOREPINEPHRINE_DURATION_S}] s."
                )
        else:
            raise BioGearsScenarioBuilderError(
                f"actions[{i}]: Unsupported action type {type(action).__name__!r}. "
                f"Expected BioGearsSalineBolus or BioGearsNorepinephrineInfusion."
            )

        if action.start_time_s < current_t:
            raise BioGearsScenarioBuilderError(
                f"actions[{i}]: start_time_s ({action.start_time_s}s) precedes previous action completion "
                f"or baseline stabilization ({current_t}s). Actions must be strictly sequential and non-overlapping."
            )
        current_t = action.start_time_s + action.delivery_duration_s

    total_sim_s = current_t + scenario.post_observation_s
    if total_sim_s > 1800.0:
        raise BioGearsScenarioBuilderError(
            f"Maximum simulated duration is 1800s, but wall-clock runtime is scenario-dependent "
            f"and may exceed several minutes. Requested scenario requires {total_sim_s}s."
        )


def _emit_saline_action(actions_el: ET.Element, action: BioGearsSalineBolus) -> None:
    action_el = ET.SubElement(actions_el, _tag("Action"))
    action_el.set(_xsi("type"), "SubstanceCompoundInfusionData")
    ET.SubElement(action_el, _tag("SubstanceCompound")).text = "Saline"
    bag_el = ET.SubElement(action_el, _tag("BagVolume"))
    bag_el.set("value", _fmt(action.bag_volume_ml))
    bag_el.set("unit", "mL")
    rate_el = ET.SubElement(action_el, _tag("Rate"))
    rate_el.set("value", _fmt(action.rate_ml_per_min))
    rate_el.set("unit", "mL/min")


def _emit_norepi_action(actions_el: ET.Element, action: BioGearsNorepinephrineInfusion) -> None:
    action_el = ET.SubElement(actions_el, _tag("Action"))
    action_el.set(_xsi("type"), "SubstanceInfusionData")
    ET.SubElement(action_el, _tag("Substance")).text = "Norepinephrine"
    conc_el = ET.SubElement(action_el, _tag("Concentration"))
    conc_el.set("value", _fmt(action.concentration_ug_ml))
    conc_el.set("unit", "ug/mL")
    rate_el = ET.SubElement(action_el, _tag("Rate"))
    rate_el.set("value", _fmt(action.rate_ml_per_min))
    rate_el.set("unit", "mL/min")


def _emit_norepi_stop(actions_el: ET.Element) -> None:
    """Stop Norepinephrine infusion by setting rate and concentration to 0.

    [UNVERIFIED]: Direct BioGears verification in Phase D Step H is required
    to confirm whether setting rate to 0 reliably terminates the infusion without
    lingering clearance or accumulation artifacts. Do not assume setting rate to 0
    is sufficient until explicitly tested.
    """
    action_el = ET.SubElement(actions_el, _tag("Action"))
    action_el.set(_xsi("type"), "SubstanceInfusionData")
    ET.SubElement(action_el, _tag("Substance")).text = "Norepinephrine"
    conc_el = ET.SubElement(action_el, _tag("Concentration"))
    conc_el.set("value", "0")
    conc_el.set("unit", "ug/mL")
    rate_el = ET.SubElement(action_el, _tag("Rate"))
    rate_el.set("value", "0")
    rate_el.set("unit", "mL/min")


def _emit_advance_time(actions_el: ET.Element, seconds: float) -> None:
    if seconds <= 0:
        return
    advance_el = ET.SubElement(actions_el, _tag("Action"))
    advance_el.set(_xsi("type"), "AdvanceTimeData")
    time_el = ET.SubElement(advance_el, _tag("Time"))
    time_el.set("value", _fmt(seconds))
    time_el.set("unit", "s")


def build_sequential_scenario_xml(scenario: BioGearsSequentialScenario) -> str:
    """Build a complete multi-action sequential BioGears scenario XML string.

    This generates a single continuous scenario timeline without resetting
    or restarting the physiology engine.
    """
    _validate_sequential_scenario(scenario)

    root = ET.Element(_tag("Scenario"))
    root.set("contentVersion", "BioGears_6.3.0-beta")
    root.set(_xsi("schemaLocation"), "")

    ET.SubElement(root, _tag("Name")).text = scenario.scenario_name
    ET.SubElement(root, _tag("Description")).text = (
        "ClinTwin BioGears sequential intervention scenario - "
        "generated by biogears_scenario_builder.py"
    )

    init_params = ET.SubElement(root, _tag("InitialParameters"))
    ET.SubElement(init_params, _tag("PatientFile")).text = scenario.patient_file

    data_requests_el = ET.SubElement(root, _tag("DataRequests"))
    data_requests_el.set("SamplesPerSecond", _fmt(scenario.samples_per_second))

    for name, unit, precision, req_type in _FLUID_DATA_REQUESTS:
        dr = ET.SubElement(data_requests_el, _tag("DataRequest"))
        dr.set(_xsi("type"), req_type)
        dr.set("Name", name)
        dr.set("Unit", unit)
        dr.set("Precision", precision)

    actions_el = ET.SubElement(root, _tag("Actions"))

    current_sim_time = 0.0
    if scenario.baseline_stabilization_s > 0:
        _emit_advance_time(actions_el, scenario.baseline_stabilization_s)
        current_sim_time = scenario.baseline_stabilization_s

    for action in scenario.actions:
        if action.start_time_s > current_sim_time:
            gap = action.start_time_s - current_sim_time
            _emit_advance_time(actions_el, gap)
            current_sim_time = action.start_time_s

        if isinstance(action, BioGearsSalineBolus):
            _emit_saline_action(actions_el, action)
            _emit_advance_time(actions_el, action.delivery_duration_s)
            current_sim_time += action.delivery_duration_s
        elif isinstance(action, BioGearsNorepinephrineInfusion):
            _emit_norepi_action(actions_el, action)
            _emit_advance_time(actions_el, action.duration_s)
            current_sim_time += action.duration_s
            _emit_norepi_stop(actions_el)

    if scenario.post_observation_s > 0:
        _emit_advance_time(actions_el, scenario.post_observation_s)

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + body

