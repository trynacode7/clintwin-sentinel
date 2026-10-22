"""
biogears_schemas.py — Request & Response Schemas for Sequential BioGears What-If (Phase D Step E)
==================================================================================================
Pydantic v2 schemas and trajectory serialization helpers for the sequential BioGears What-If API.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator

from backend.biogears_bridge import BioGearsResult
from backend.biogears_scenario_builder import (
    BioGearsSalineBolus,
    BioGearsNorepinephrineInfusion,
    VERIFIED_SALINE_RATE,
    VERIFIED_SALINE_VOLUMES,
    MIN_SALINE_VOLUME_ML,
    MAX_SALINE_VOLUME_ML,
    MIN_SALINE_RATE_ML_PER_MIN,
    MAX_SALINE_RATE_ML_PER_MIN,
    DEFAULT_SALINE_VOLUME_ML,
    DEFAULT_SALINE_RATE_ML_PER_MIN,
    VERIFIED_NOREPINEPHRINE_CONCENTRATION,
    VERIFIED_NOREPINEPHRINE_RATE,
    VERIFIED_NOREPINEPHRINE_DURATION,
    MIN_NOREPINEPHRINE_RATE_ML_PER_MIN,
    MAX_NOREPINEPHRINE_RATE_ML_PER_MIN,
    DEFAULT_NOREPINEPHRINE_RATE_ML_PER_MIN,
    MIN_NOREPINEPHRINE_DURATION_S,
    MAX_NOREPINEPHRINE_DURATION_S,
    DEFAULT_NOREPINEPHRINE_DURATION_S,
    DEFAULT_NOREPINEPHRINE_CONCENTRATION_UG_ML,
)
from backend.biogears_sequential_execution import (
    PhysiologicalSnapshot,
    SequentialExecutionResult,
)


# ─── Request Schemas ──────────────────────────────────────────────────────────


class SalineActionRequest(BaseModel):
    """Specification for a single saline bolus action on the continuous timeline."""

    type: Literal["saline"] = Field("saline", description="Action type: strictly 'saline'.")
    volume_ml: float = Field(
        DEFAULT_SALINE_VOLUME_ML,
        description=f"Volume in mL. Supported range: {MIN_SALINE_VOLUME_ML} to {MAX_SALINE_VOLUME_ML} mL.",
    )
    rate_ml_per_min: float = Field(
        DEFAULT_SALINE_RATE_ML_PER_MIN,
        description=f"Infusion rate in mL/min. Supported range: {MIN_SALINE_RATE_ML_PER_MIN} to {MAX_SALINE_RATE_ML_PER_MIN} mL/min.",
    )
    start_time_s: float = Field(..., ge=0, description="Start timestamp on the continuous timeline in seconds.")

    @field_validator("start_time_s")
    @classmethod
    def validate_saline_start_time(cls, v: float) -> float:
        import math
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("start_time_s must be a finite number.")
        if v < 0:
            raise ValueError("start_time_s must be non-negative.")
        return float(v)

    @field_validator("volume_ml")
    @classmethod
    def validate_volume(cls, v: float) -> float:
        import math
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("volume_ml must be a finite number.")
        if v < MIN_SALINE_VOLUME_ML or v > MAX_SALINE_VOLUME_ML:
            raise ValueError(
                f"Invalid volume_ml {v}. Supported saline volumes range from {MIN_SALINE_VOLUME_ML} to {MAX_SALINE_VOLUME_ML} mL."
            )
        return float(v)

    @field_validator("rate_ml_per_min")
    @classmethod
    def validate_rate(cls, v: float) -> float:
        import math
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("rate_ml_per_min must be a finite number.")
        if v < MIN_SALINE_RATE_ML_PER_MIN or v > MAX_SALINE_RATE_ML_PER_MIN:
            raise ValueError(
                f"Invalid rate_ml_per_min {v}. Supported saline infusion rates range from {MIN_SALINE_RATE_ML_PER_MIN} to {MAX_SALINE_RATE_ML_PER_MIN} mL/min."
            )
        return float(v)

    @field_validator("start_time_s")
    @classmethod
    def validate_norepi_start_time(cls, v: float) -> float:
        import math
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("start_time_s must be a finite number.")
        if v < 0:
            raise ValueError("start_time_s must be non-negative.")
        return float(v)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v != "saline":
            raise ValueError(
                f"Unsupported action type '{v}'. Expected 'saline'."
            )
        return v


class NorepinephrineActionRequest(BaseModel):
    """Specification for a norepinephrine substance infusion action on the continuous timeline."""

    type: Literal["norepinephrine"] = Field("norepinephrine", description="Action type: strictly 'norepinephrine'.")
    start_time_s: float = Field(..., ge=0, description="Start timestamp on the continuous timeline in seconds.")
    concentration_ug_ml: float = Field(
        DEFAULT_NOREPINEPHRINE_CONCENTRATION_UG_ML,
        description=f"Concentration in ug/mL. Standard stock dilution: {DEFAULT_NOREPINEPHRINE_CONCENTRATION_UG_ML} ug/mL.",
    )
    rate_ml_per_min: float = Field(
        DEFAULT_NOREPINEPHRINE_RATE_ML_PER_MIN,
        description=f"Infusion rate in mL/min. Supported range: {MIN_NOREPINEPHRINE_RATE_ML_PER_MIN} to {MAX_NOREPINEPHRINE_RATE_ML_PER_MIN} mL/min.",
    )
    duration_s: float = Field(
        DEFAULT_NOREPINEPHRINE_DURATION_S,
        description=f"Active infusion duration in seconds. Supported range: {MIN_NOREPINEPHRINE_DURATION_S} to {MAX_NOREPINEPHRINE_DURATION_S} s.",
    )
    volume_ml: Optional[float] = Field(None, description="Forbidden for norepinephrine infusions.")

    @field_validator("volume_ml")
    @classmethod
    def validate_no_volume(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            raise ValueError(
                "Norepinephrine infusion does not accept 'volume_ml'. "
                "Specify 'rate_ml_per_min' and 'duration_s' instead."
            )
        return v

    @field_validator("concentration_ug_ml")
    @classmethod
    def validate_concentration(cls, v: float) -> float:
        import math
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("concentration_ug_ml must be a finite number.")
        if v != VERIFIED_NOREPINEPHRINE_CONCENTRATION:
            raise ValueError(
                f"Invalid concentration_ug_ml {v}. Supported standard stock concentration is {VERIFIED_NOREPINEPHRINE_CONCENTRATION} ug/mL. "
                "BioGears substance models are calibrated to standard stock dilution; adjust rate_ml_per_min or duration_s to titrate dose."
            )
        return float(v)

    @field_validator("rate_ml_per_min")
    @classmethod
    def validate_rate(cls, v: float) -> float:
        import math
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("rate_ml_per_min must be a finite number.")
        if v < MIN_NOREPINEPHRINE_RATE_ML_PER_MIN or v > MAX_NOREPINEPHRINE_RATE_ML_PER_MIN:
            raise ValueError(
                f"Invalid rate_ml_per_min {v}. Supported norepinephrine infusion rates range from "
                f"{MIN_NOREPINEPHRINE_RATE_ML_PER_MIN} to {MAX_NOREPINEPHRINE_RATE_ML_PER_MIN} mL/min."
            )
        return float(v)

    @field_validator("duration_s")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        import math
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError("duration_s must be a finite number.")
        if v < MIN_NOREPINEPHRINE_DURATION_S or v > MAX_NOREPINEPHRINE_DURATION_S:
            raise ValueError(
                f"Invalid duration_s {v}. Supported norepinephrine infusion durations range from "
                f"{MIN_NOREPINEPHRINE_DURATION_S} to {MAX_NOREPINEPHRINE_DURATION_S} s."
            )
        return float(v)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v != "norepinephrine":
            raise ValueError(
                f"Unsupported action type '{v}'. Expected 'norepinephrine'."
            )
        return v


ActionRequest = Annotated[
    Union[SalineActionRequest, NorepinephrineActionRequest],
    Field(discriminator="type"),
]


class SequentialWhatIfRequest(BaseModel):
    """Request payload for POST /biogears/whatif/sequential."""

    scenario_name: Optional[str] = Field(None, description="Optional scenario identifier. Deterministic default used if omitted.")
    patient_file: str = Field("StandardMale.xml", description="Patient definition XML file (defaults to StandardMale.xml).")
    baseline_stabilization_s: float = Field(60.0, gt=0, description="Initial baseline observation in seconds.")
    post_observation_s: float = Field(60.0, gt=0, description="Post-intervention observation duration in seconds.")
    samples_per_second: float = Field(0.2, gt=0, description="Sampling rate in Hz (samples per second).")
    actions: List[ActionRequest] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="Sequential actions (1 to 3 non-overlapping saline or norepinephrine interventions).",
    )


# ─── Response Schemas ─────────────────────────────────────────────────────────


class ActionConfigResponse(BaseModel):
    """Summary of a configured scenario action."""

    action_index: int
    action_type: str
    bag_volume_ml: Optional[float] = None
    rate_ml_per_min: float
    start_time_s: float
    delivery_duration_s: float
    concentration_ug_ml: Optional[float] = None
    duration_s: Optional[float] = None
    substance: Optional[str] = None


class PhysiologicalSnapshotResponse(BaseModel):
    """Physiological vitals at a specific scenario milestone."""

    target_time_s: float
    actual_time_s: float
    milestone_type: str
    action_index: Optional[int] = None
    description: Optional[str] = None
    heart_rate_bpm: float
    mean_arterial_pressure_mmhg: float
    blood_volume_l: float
    cardiac_output_l_min: float
    systolic_bp_mmhg: Optional[float] = None
    diastolic_bp_mmhg: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    respiration_rate_bpm: Optional[float] = None


class MilestoneStateResponse(BaseModel):
    """Intermediate scenario milestone containing structured snapshot."""

    phase: str
    description: str
    target_time_s: float
    actual_time_s: float
    snapshot: PhysiologicalSnapshotResponse


class TrajectorySignal(BaseModel):
    """Single physiological signal time-series with unit."""

    name: str
    unit: str
    values: List[float]


class TrajectoryResponse(BaseModel):
    """Full physiological time-series trajectory."""

    time_s: List[float]
    signals: Dict[str, TrajectorySignal]
    data_points: List[Dict[str, Any]]


class ScenarioMetadataResponse(BaseModel):
    """Complete provenance and configuration metadata for the scenario."""

    scenario_name: str
    patient_file: str
    actions: List[ActionConfigResponse]
    baseline_stabilization_s: float
    post_observation_s: float
    simulated_duration_s: float
    samples_per_second: float
    fingerprint: str
    source: Literal["cache", "live"]
    is_cached: bool
    wall_clock_seconds: Optional[float] = None
    time_range_s: List[float]
    row_count: int
    is_continuous: bool


class SequentialWhatIfResponse(BaseModel):
    """Structured response for POST /biogears/whatif/sequential."""

    metadata: ScenarioMetadataResponse
    baseline_snapshot: PhysiologicalSnapshotResponse
    intermediate_milestones: List[MilestoneStateResponse]
    final_snapshot: PhysiologicalSnapshotResponse
    trajectory: TrajectoryResponse
    provenance_disclaimer: str


# ─── Serialization Helpers ───────────────────────────────────────────────────


_SIGNAL_LOOKUP = {
    "blood_volume_l": (["BloodVolume(L)", "BloodVolume"], "BloodVolume", "L"),
    "mean_arterial_pressure_mmhg": (["MeanArterialPressure(mmHg)", "MeanArterialPressure"], "MeanArterialPressure", "mmHg"),
    "heart_rate_bpm": (["HeartRate(1/min)", "HeartRate"], "HeartRate", "1/min"),
    "cardiac_output_l_min": (["CardiacOutput(L/min)", "CardiacOutput"], "CardiacOutput", "L/min"),
    "systolic_bp_mmhg": (["SystolicArterialPressure(mmHg)", "SystolicArterialPressure"], "SystolicArterialPressure", "mmHg"),
    "diastolic_bp_mmhg": (["DiastolicArterialPressure(mmHg)", "DiastolicArterialPressure"], "DiastolicArterialPressure", "mmHg"),
    "oxygen_saturation": (["OxygenSaturation"], "OxygenSaturation", "unitless"),
    "respiration_rate_bpm": (["RespirationRate(1/min)", "RespirationRate"], "RespirationRate", "1/min"),
}


def build_trajectory_response(biogears_result: BioGearsResult) -> TrajectoryResponse:
    """Extract structured trajectory signals and columnar data points from BioGearsResult."""
    time_s = [round(t, 2) for t in biogears_result.time_s]
    n_rows = len(time_s)

    signals: Dict[str, TrajectorySignal] = {}
    matched_cols: Dict[str, str] = {}

    for signal_key, (candidate_headers, default_name, default_unit) in _SIGNAL_LOOKUP.items():
        found_header = None
        for h in candidate_headers:
            if h in biogears_result.columns:
                found_header = h
                break
        if found_header:
            matched_cols[signal_key] = found_header
            ts = biogears_result.columns[found_header]
            signals[signal_key] = TrajectorySignal(
                name=ts.column.name or default_name,
                unit=ts.column.unit or default_unit,
                values=[round(v, 3) for v in ts.values],
            )

    # Build row-based data_points for straightforward charting
    data_points: List[Dict[str, Any]] = []
    for i in range(n_rows):
        pt: Dict[str, Any] = {"time_s": time_s[i]}
        for signal_key in matched_cols:
            pt[signal_key] = signals[signal_key].values[i]
        data_points.append(pt)

    return TrajectoryResponse(
        time_s=time_s,
        signals=signals,
        data_points=data_points,
    )


def serialize_snapshot_to_response(snap: PhysiologicalSnapshot) -> PhysiologicalSnapshotResponse:
    """Convert domain PhysiologicalSnapshot to validated Pydantic model."""
    d = snap.to_dict()
    return PhysiologicalSnapshotResponse(**d)


def build_sequential_whatif_response(
    result: SequentialExecutionResult,
) -> SequentialWhatIfResponse:
    """Convert SequentialExecutionResult into a fully-validated SequentialWhatIfResponse."""
    # Metadata actions summary
    action_summaries: List[ActionConfigResponse] = []
    for i, act in enumerate(result.scenario.actions):
        if isinstance(act, BioGearsSalineBolus):
            action_summaries.append(
                ActionConfigResponse(
                    action_index=i,
                    action_type="saline",
                    substance="Saline",
                    bag_volume_ml=float(act.bag_volume_ml),
                    rate_ml_per_min=float(act.rate_ml_per_min),
                    start_time_s=float(act.start_time_s),
                    delivery_duration_s=float(act.delivery_duration_s),
                )
            )
        elif isinstance(act, BioGearsNorepinephrineInfusion):
            action_summaries.append(
                ActionConfigResponse(
                    action_index=i,
                    action_type="norepinephrine",
                    substance="Norepinephrine",
                    bag_volume_ml=None,
                    rate_ml_per_min=float(act.rate_ml_per_min),
                    start_time_s=float(act.start_time_s),
                    delivery_duration_s=float(act.delivery_duration_s),
                    concentration_ug_ml=float(act.concentration_ug_ml),
                    duration_s=float(act.duration_s),
                )
            )

    source: Literal["cache", "live"] = "cache" if result.is_cached else "live"

    metadata = ScenarioMetadataResponse(
        scenario_name=result.scenario_name,
        patient_file=result.scenario.patient_file,
        actions=action_summaries,
        baseline_stabilization_s=float(result.scenario.baseline_stabilization_s),
        post_observation_s=float(result.scenario.post_observation_s),
        simulated_duration_s=float(result.scenario.total_duration_s),
        samples_per_second=float(result.scenario.samples_per_second),
        fingerprint=result.cache_fingerprint or "unknown",
        source=source,
        is_cached=result.is_cached,
        wall_clock_seconds=round(result.wall_clock_seconds, 3) if result.wall_clock_seconds is not None else None,
        time_range_s=[round(t, 2) for t in result.time_range_s],
        row_count=result.row_count,
        is_continuous=result.is_continuous,
    )

    baseline_snap = serialize_snapshot_to_response(result.baseline_state)
    final_snap = serialize_snapshot_to_response(result.final_state)

    intermediate_milestones: List[MilestoneStateResponse] = []
    for item in result.intermediate_states:
        snap_resp = serialize_snapshot_to_response(item["snapshot"])
        m_info = item["milestone"]
        intermediate_milestones.append(
            MilestoneStateResponse(
                phase=m_info.get("phase", "milestone"),
                description=m_info.get("description", snap_resp.description or ""),
                target_time_s=round(item["target_time_s"], 2),
                actual_time_s=round(item["actual_time_s"], 2),
                snapshot=snap_resp,
            )
        )

    trajectory = build_trajectory_response(result.biogears_result)

    disclaimer = (
        f"Mechanistic physiological simulation executed via BioGears engine using virtual subject "
        f"'{result.scenario.patient_file}'. Outputs reflect verified mathematical physiology models "
        f"and are not personalized to dashboard ICU patient electronic health record profiles."
    )

    return SequentialWhatIfResponse(
        metadata=metadata,
        baseline_snapshot=baseline_snap,
        intermediate_milestones=intermediate_milestones,
        final_snapshot=final_snap,
        trajectory=trajectory,
        provenance_disclaimer=disclaimer,
    )
