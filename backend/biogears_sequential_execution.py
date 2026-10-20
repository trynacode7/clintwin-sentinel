"""
biogears_sequential_execution.py — BioGears Sequential Scenario Execution & State Extraction (Phase D)
====================================================================================================
SCOPE: Execution orchestration and robust physiological state extraction for continuous
sequential BioGears scenarios.

Phase D Step C: Robust Physiological State Extraction
Supports:
  1. Structured PhysiologicalSnapshot with target vs actual sampled timestamps,
     milestone types, action indices, and explicit representation of unavailable vitals.
  2. Documented timestamp-matching policy with configurable tolerance (default 5.0s).
  3. Rejection of non-monotonic timelines, out-of-range milestones, missing required
     physiological columns, and empty or malformed data.
  4. Direct extraction from BioGearsResult or from saved output CSV files.
  5. Continuous sequential saline execution with strict failure propagation.

Epistemic Status:
  - Saline sequential execution: [VERIFIED] in Step B.
  - Physiological state extraction logic: [VERIFIED] via unit tests and saved CSV analysis.
  - Mixed interventions (Saline + Norepinephrine): strictly [BLOCKED].
  - Norepinephrine sequential execution: [UNVERIFIED] (scheduled for later step).
"""

import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.biogears_bridge import (
    BioGearsBridge,
    BioGearsBridgeError,
    BioGearsConfig,
    BioGearsNotFoundError,
    BioGearsOutputNotFoundError,
    BioGearsOutputParseError,
    BioGearsProcessError,
    BioGearsResult,
    BioGearsTimeoutError,
)
from backend.biogears_scenario_builder import (
    BioGearsSalineBolus,
    BioGearsScenarioBuilderError,
    BioGearsSequentialScenario,
    build_sequential_scenario_xml,
)

logger = logging.getLogger(__name__)

# ─── Constants & Policies ─────────────────────────────────────────────────────

# Default tolerance for matching target milestone timestamps to sampled rows.
# At 0.2 Hz (5-second intervals), the maximum expected distance between any target
# time and the nearest sample is 2.5 seconds (plus ~0.02s BioGears phase offset).
# Setting default tolerance to 5.0s ensures single-step capture while rejecting distant samples.
DEFAULT_TIMESTAMP_TOLERANCE_S = 5.0

# Core physiological signals that MUST be present in the BioGears output.
REQUIRED_PHYSIOLOGICAL_SIGNALS = {
    "blood_volume": ["BloodVolume(L)", "BloodVolume"],
    "mean_arterial_pressure": ["MeanArterialPressure(mmHg)", "MeanArterialPressure"],
    "heart_rate": ["HeartRate(1/min)", "HeartRate"],
    "cardiac_output": ["CardiacOutput(L/min)", "CardiacOutput"],
}

# Supported optional physiological signals extracted only when present in the CSV.
OPTIONAL_PHYSIOLOGICAL_SIGNALS = {
    "systolic_bp": ["SystolicArterialPressure(mmHg)", "SystolicArterialPressure"],
    "diastolic_bp": ["DiastolicArterialPressure(mmHg)", "DiastolicArterialPressure"],
    "oxygen_saturation": ["OxygenSaturation"],
    "respiration_rate": ["RespirationRate(1/min)", "RespirationRate"],
}


# ─── Exceptions ───────────────────────────────────────────────────────────────


class BioGearsExtractionError(BioGearsBridgeError):
    """Raised when physiological state extraction from BioGears output fails.

    Carries actionable information explaining whether the failure was due to
    out-of-range milestones, tolerance exceeded, non-monotonic timestamps,
    or missing required physiological columns.
    """


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PhysiologicalSnapshot:
    """Physiological measurements at a specific milestone timestamp.

    Attributes:
        target_time_s: Target milestone timestamp on the scenario timeline.
        actual_time_s: Actual sampled timestamp from the simulation output.
        milestone_type: Type of milestone ('baseline', 'delivery_end',
            'intermediate_observation', 'final_observation', or 'custom').
        heart_rate_bpm: Heart rate in beats/min (REQUIRED).
        mean_arterial_pressure_mmhg: MAP in mmHg (REQUIRED).
        blood_volume_l: Blood volume in Liters (REQUIRED).
        cardiac_output_l_min: Cardiac output in L/min (REQUIRED).
        action_index: 0-based index of intervention action, if applicable.
        systolic_bp_mmhg: Systolic arterial pressure in mmHg (Optional).
        diastolic_bp_mmhg: Diastolic arterial pressure in mmHg (Optional).
        oxygen_saturation: SpO2 as unitless fraction [0, 1] (Optional).
        respiration_rate_bpm: Respiration rate in breaths/min (Optional).
        description: Human-readable description of milestone (Optional).
    """

    target_time_s: float
    actual_time_s: float
    milestone_type: str
    heart_rate_bpm: float
    mean_arterial_pressure_mmhg: float
    blood_volume_l: float
    cardiac_output_l_min: float
    action_index: Optional[int] = None
    systolic_bp_mmhg: Optional[float] = None
    diastolic_bp_mmhg: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    respiration_rate_bpm: Optional[float] = None
    description: Optional[str] = None

    @property
    def time_s(self) -> float:
        """Backward-compatibility alias for actual_time_s."""
        return self.actual_time_s

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to JSON-serializable dictionary."""
        return {
            "target_time_s": round(self.target_time_s, 2),
            "actual_time_s": round(self.actual_time_s, 2),
            "milestone_type": self.milestone_type,
            "action_index": self.action_index,
            "description": self.description,
            "blood_volume_l": round(self.blood_volume_l, 3),
            "mean_arterial_pressure_mmhg": round(self.mean_arterial_pressure_mmhg, 1),
            "heart_rate_bpm": round(self.heart_rate_bpm, 1),
            "cardiac_output_l_min": round(self.cardiac_output_l_min, 2),
            "systolic_bp_mmhg": round(self.systolic_bp_mmhg, 1) if self.systolic_bp_mmhg is not None else None,
            "diastolic_bp_mmhg": round(self.diastolic_bp_mmhg, 1) if self.diastolic_bp_mmhg is not None else None,
            "oxygen_saturation": round(self.oxygen_saturation, 3) if self.oxygen_saturation is not None else None,
            "respiration_rate_bpm": round(self.respiration_rate_bpm, 1) if self.respiration_rate_bpm is not None else None,
        }


@dataclass
class ScenarioMilestoneReport:
    """Structured report of all extracted milestones for a scenario."""

    baseline_snapshot: PhysiologicalSnapshot
    milestone_snapshots: List[PhysiologicalSnapshot]
    delivery_end_snapshots: List[PhysiologicalSnapshot]
    intermediate_observation_snapshots: List[PhysiologicalSnapshot]
    final_snapshot: PhysiologicalSnapshot


@dataclass
class SequentialExecutionResult:
    """Complete verified results from executing a sequential BioGears scenario."""

    scenario_name: str
    scenario: BioGearsSequentialScenario
    scenario_xml: str
    command_executed: str
    exit_status: int
    wall_clock_seconds: float
    row_count: int
    time_range_s: Tuple[float, float]
    output_csv_path: Optional[str]
    is_continuous: bool
    milestones: List[Dict[str, Any]]
    baseline_state: PhysiologicalSnapshot
    intermediate_states: List[Dict[str, Any]]
    final_state: PhysiologicalSnapshot
    biogears_result: BioGearsResult
    is_cached: bool = False
    cache_fingerprint: Optional[str] = None


# ─── Snapshot Extraction Helpers ──────────────────────────────────────────────


def _resolve_column(columns: Dict[str, Any], candidate_names: List[str]) -> Optional[str]:
    """Find the first matching raw column header in the result columns."""
    for name in candidate_names:
        if name in columns:
            return name
    return None


def extract_snapshot_at_time(
    biogears_result: BioGearsResult,
    target_time_s: float,
    milestone_type: str = "custom",
    action_index: Optional[int] = None,
    description: Optional[str] = None,
    tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
) -> PhysiologicalSnapshot:
    """Extract a PhysiologicalSnapshot closest to target_time_s within tolerance.

    Rules:
      1. Reject empty time series.
      2. Verify non-monotonic and valid timestamps.
      3. Verify all required physiological columns exist.
      4. Reject target times outside the simulation interval + tolerance.
      5. Match nearest sample row; reject if nearest sample exceeds tolerance_s.
      6. Record both target_time_s and actual_time_s.
      7. Represent unavailable optional measurements as None (never fabricate).

    Raises:
        BioGearsExtractionError on any validation or matching failure.
    """
    if not biogears_result.time_s:
        raise BioGearsExtractionError("Cannot extract snapshot from BioGearsResult with zero rows.")

    time_s = biogears_result.time_s

    # Verify monotonic and finite timestamps
    for i in range(len(time_s)):
        t_val = time_s[i]
        if not isinstance(t_val, (int, float)) or math.isnan(t_val) or math.isinf(t_val):
            raise BioGearsExtractionError(
                f"Malformed timestamp at row {i}: {t_val!r}. Timestamps must be finite numbers."
            )
        if i > 0 and t_val <= time_s[i - 1]:
            raise BioGearsExtractionError(
                f"Non-monotonic timestamp at row {i}: {time_s[i - 1]} -> {t_val}."
            )

    # Verify required physiological columns
    resolved_required: Dict[str, str] = {}
    for sig_name, candidates in REQUIRED_PHYSIOLOGICAL_SIGNALS.items():
        found = _resolve_column(biogears_result.columns, candidates)
        if found is None:
            raise BioGearsExtractionError(
                f"Missing required physiological signal '{sig_name}' (checked candidates {candidates}). "
                f"Available columns: {sorted(biogears_result.columns.keys())}"
            )
        resolved_required[sig_name] = found

    # Verify out-of-range target times
    t_min = time_s[0]
    t_max = time_s[-1]
    if target_time_s < (t_min - tolerance_s) or target_time_s > (t_max + tolerance_s):
        raise BioGearsExtractionError(
            f"Target milestone time {target_time_s}s is outside available simulation interval "
            f"[{t_min:.2f}s, {t_max:.2f}s] by more than allowed tolerance ({tolerance_s}s)."
        )

    # Nearest-sample selection
    best_idx = 0
    min_diff = abs(time_s[0] - target_time_s)
    for idx in range(1, len(time_s)):
        diff = abs(time_s[idx] - target_time_s)
        if diff < min_diff:
            min_diff = diff
            best_idx = idx

    if min_diff > tolerance_s:
        raise BioGearsExtractionError(
            f"Nearest sample to target {target_time_s}s is at {time_s[best_idx]:.2f}s "
            f"(difference {min_diff:.2f}s exceeds allowed tolerance {tolerance_s}s)."
        )

    actual_time = time_s[best_idx]

    # Extract required values
    bv = float(biogears_result.columns[resolved_required["blood_volume"]].values[best_idx])
    map_v = float(biogears_result.columns[resolved_required["mean_arterial_pressure"]].values[best_idx])
    hr = float(biogears_result.columns[resolved_required["heart_rate"]].values[best_idx])
    co = float(biogears_result.columns[resolved_required["cardiac_output"]].values[best_idx])

    # Extract optional values if present, else None (explicitly represented, never fabricated)
    def _opt(sig_name: str) -> Optional[float]:
        candidates = OPTIONAL_PHYSIOLOGICAL_SIGNALS[sig_name]
        col_hdr = _resolve_column(biogears_result.columns, candidates)
        if col_hdr is not None:
            val = biogears_result.columns[col_hdr].values[best_idx]
            if val is not None and not math.isnan(val):
                return float(val)
        return None

    return PhysiologicalSnapshot(
        target_time_s=target_time_s,
        actual_time_s=actual_time,
        milestone_type=milestone_type,
        action_index=action_index,
        description=description,
        blood_volume_l=bv,
        mean_arterial_pressure_mmhg=map_v,
        heart_rate_bpm=hr,
        cardiac_output_l_min=co,
        systolic_bp_mmhg=_opt("systolic_bp"),
        diastolic_bp_mmhg=_opt("diastolic_bp"),
        oxygen_saturation=_opt("oxygen_saturation"),
        respiration_rate_bpm=_opt("respiration_rate"),
    )


def extract_scenario_snapshots(
    biogears_result: BioGearsResult,
    scenario: BioGearsSequentialScenario,
    tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
) -> ScenarioMilestoneReport:
    """Extract all scenario milestones into structured PhysiologicalSnapshot objects.

    Preserves exact chronological order and milestone metadata across:
      - Baseline observation period
      - Active intervention delivery endpoints
      - Intermediate advance-time observation holds
      - Final post-intervention observation period
    """
    milestones = scenario.get_timeline_milestones()
    all_snapshots: List[PhysiologicalSnapshot] = []
    delivery_ends: List[PhysiologicalSnapshot] = []
    intermediate_obs: List[PhysiologicalSnapshot] = []

    # 1. Baseline snapshot (at baseline_stabilization_s)
    baseline_snap = extract_snapshot_at_time(
        biogears_result=biogears_result,
        target_time_s=scenario.baseline_stabilization_s,
        milestone_type="baseline",
        description=f"Baseline observation ({scenario.baseline_stabilization_s}s)",
        tolerance_s=tolerance_s,
    )

    # 2. Extract milestone snapshots in chronological order
    for m in milestones:
        phase = m["phase"]
        end_s = float(m["end_s"])
        desc = m.get("description", phase)

        if phase == "baseline":
            m_type = "baseline"
            act_idx = None
        elif phase.startswith("intervention_"):
            m_type = "delivery_end"
            act_idx = m.get("action_index")
        elif "intermediate_hold" in phase or "pre_action" in phase:
            m_type = "intermediate_observation"
            act_idx = None
        elif phase == "post_observation":
            m_type = "final_observation"
            act_idx = None
        else:
            m_type = "custom"
            act_idx = None

        snap = extract_snapshot_at_time(
            biogears_result=biogears_result,
            target_time_s=end_s,
            milestone_type=m_type,
            action_index=act_idx,
            description=desc,
            tolerance_s=tolerance_s,
        )
        all_snapshots.append(snap)

        if m_type == "delivery_end":
            delivery_ends.append(snap)
        elif m_type == "intermediate_observation":
            intermediate_obs.append(snap)

    final_snap = all_snapshots[-1]

    return ScenarioMilestoneReport(
        baseline_snapshot=baseline_snap,
        milestone_snapshots=all_snapshots,
        delivery_end_snapshots=delivery_ends,
        intermediate_observation_snapshots=intermediate_obs,
        final_snapshot=final_snap,
    )


def extract_snapshots_from_csv(
    csv_path: str | Path,
    scenario: BioGearsSequentialScenario,
    tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
) -> ScenarioMilestoneReport:
    """Parse a BioGears simulation CSV file and extract all scenario milestones.

    Allows validating state extraction directly against saved simulation results
    without rerunning expensive engine simulations.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise BioGearsOutputNotFoundError(
            f"BioGears results CSV not found at {csv_file}",
            expected_path=str(csv_file),
        )

    bridge = BioGearsBridge()
    result = bridge._parse_csv(
        csv_path=str(csv_file),
        scenario_name=scenario.scenario_name,
        returncode=0,
    )
    return extract_scenario_snapshots(result, scenario, tolerance_s=tolerance_s)


# ─── Continuity Verification Helper ──────────────────────────────────────────


def verify_temporal_continuity(
    time_s: List[float],
    expected_duration_s: float,
    tolerance_s: float = 6.0,
) -> bool:
    """Verify that timestamps are strictly monotonic and reach the expected duration."""
    if not time_s:
        raise BioGearsOutputParseError(
            "Empty time series — simulation produced no timestamps.",
            csv_path="",
            reason="zero rows",
        )

    for i in range(1, len(time_s)):
        if time_s[i] <= time_s[i - 1]:
            raise BioGearsOutputParseError(
                f"Non-monotonic timestamps detected at index {i}: "
                f"{time_s[i - 1]} -> {time_s[i]}. Timeline was interrupted or reset.",
                csv_path="",
                reason="non-monotonic time axis",
            )

    last_time = time_s[-1]
    if last_time < (expected_duration_s - tolerance_s):
        raise BioGearsOutputParseError(
            f"Simulation truncated prematurely: last observed timestamp was {last_time}s, "
            f"but scenario configured duration is {expected_duration_s}s.",
            csv_path="",
            reason="truncated simulation output",
        )

    return True


# ─── Execution Function ───────────────────────────────────────────────────────


def _execute_sequential_uncached(
    scenario: BioGearsSequentialScenario,
    bridge: Optional[BioGearsBridge] = None,
    timeout_seconds: int = 600,
    tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
    keep_output: bool = True,
) -> SequentialExecutionResult:
    """Execute scenario directly via BioGearsBridge without checking or writing cache."""
    if not isinstance(scenario, BioGearsSequentialScenario):
        raise BioGearsScenarioBuilderError(
            f"Expected BioGearsSequentialScenario, got {type(scenario).__name__}."
        )

    scenario.validate()
    scenario_xml = build_sequential_scenario_xml(scenario)

    if bridge is None:
        cfg = BioGearsConfig(timeout_seconds=timeout_seconds)
        bridge = BioGearsBridge(config=cfg)

    command_executed = f'"{bridge.config.executable}" scenario <xml> results <prefix>'

    t_start = time.perf_counter()
    biogears_result = bridge.run_scenario_xml(
        scenario_xml_content=scenario_xml,
        scenario_name=scenario.scenario_name,
        keep_output=keep_output,
    )
    wall_clock_seconds = time.perf_counter() - t_start

    is_continuous = verify_temporal_continuity(
        time_s=biogears_result.time_s,
        expected_duration_s=scenario.total_duration_s,
    )

    report = extract_scenario_snapshots(
        biogears_result=biogears_result,
        scenario=scenario,
        tolerance_s=tolerance_s,
    )

    milestones = scenario.get_timeline_milestones()
    intermediate_states: List[Dict[str, Any]] = [
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
        scenario_xml=scenario_xml,
        command_executed=command_executed,
        exit_status=biogears_result.returncode,
        wall_clock_seconds=wall_clock_seconds,
        row_count=biogears_result.row_count,
        time_range_s=biogears_result.time_range_s,
        output_csv_path=biogears_result.output_csv_path,
        is_continuous=is_continuous,
        milestones=milestones,
        baseline_state=report.baseline_snapshot,
        intermediate_states=intermediate_states,
        final_state=report.final_snapshot,
        biogears_result=biogears_result,
        is_cached=False,
        cache_fingerprint=None,
    )


def _execute_sequential_saline_uncached(
    scenario: BioGearsSequentialScenario,
    bridge: Optional[BioGearsBridge] = None,
    timeout_seconds: int = 600,
    tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
    keep_output: bool = True,
) -> SequentialExecutionResult:
    """Execute scenario directly via BioGearsBridge enforcing saline-only preconditions."""
    if not isinstance(scenario, BioGearsSequentialScenario):
        raise BioGearsScenarioBuilderError(
            f"Expected BioGearsSequentialScenario, got {type(scenario).__name__}."
        )

    if scenario.is_mixed_intervention:
        raise BioGearsScenarioBuilderError(
            "Mixed intervention execution (Saline followed by Norepinephrine) is [BLOCKED] "
            "pending verification of continuous single-substance execution in Phase D Step I."
        )

    for i, action in enumerate(scenario.actions):
        if not isinstance(action, BioGearsSalineBolus):
            raise BioGearsScenarioBuilderError(
                f"Action {i} is {type(action).__name__}, not BioGearsSalineBolus. "
                f"Phase D Step B/C supports saline-only sequential scenarios."
            )

    return _execute_sequential_uncached(
        scenario=scenario,
        bridge=bridge,
        timeout_seconds=timeout_seconds,
        tolerance_s=tolerance_s,
        keep_output=keep_output,
    )


def execute_sequential_scenario(
    scenario: BioGearsSequentialScenario,
    bridge: Optional[BioGearsBridge] = None,
    timeout_seconds: int = 600,
    tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
    keep_output: bool = True,
    use_cache: bool = True,
    cache: Optional[Any] = None,
) -> SequentialExecutionResult:
    """Execute a multi-action sequential BioGears scenario (saline, norepinephrine, or mixed) with result caching.

    On a cache hit:
      - Validates fingerprint, metadata, required artifacts, continuity, and snapshot channels.
      - Returns cached SequentialExecutionResult with is_cached=True.

    On a cache miss:
      - Synchronizes concurrent executions on the same fingerprint using a per-fingerprint lock.
      - Double-checks the cache inside the lock before running simulation.
      - Executes scenario via BioGearsBridge on a continuous timeline.
      - Validates temporal continuity and extracts physiological milestone snapshots.
      - Persists complete artifacts atomically.
      - Returns SequentialExecutionResult with is_cached=False.
    """
    if not isinstance(scenario, BioGearsSequentialScenario):
        raise BioGearsScenarioBuilderError(
            f"Expected BioGearsSequentialScenario, got {type(scenario).__name__}."
        )

    scenario.validate()

    if not use_cache:
        if not scenario.is_mixed_intervention and all(isinstance(a, BioGearsSalineBolus) for a in scenario.actions):
            return _execute_sequential_saline_uncached(
                scenario=scenario,
                bridge=bridge,
                timeout_seconds=timeout_seconds,
                tolerance_s=tolerance_s,
                keep_output=keep_output,
            )
        return _execute_sequential_uncached(
            scenario=scenario,
            bridge=bridge,
            timeout_seconds=timeout_seconds,
            tolerance_s=tolerance_s,
            keep_output=keep_output,
        )

    from backend.biogears_scenario_cache import BioGearsScenarioCache

    if cache is None:
        cache = BioGearsScenarioCache()

    fingerprint = cache.compute_fingerprint(scenario)

    with cache.get_lock(fingerprint):
        cached_result = cache.get(scenario, fingerprint=fingerprint, tolerance_s=tolerance_s)
        if cached_result is not None:
            logger.info(
                "Cache HIT for scenario '%s' (fingerprint: %s).",
                scenario.scenario_name,
                fingerprint,
            )
            return cached_result

        logger.info(
            "Cache MISS for scenario '%s' (fingerprint: %s). Running BioGears...",
            scenario.scenario_name,
            fingerprint,
        )
        if not scenario.is_mixed_intervention and all(isinstance(a, BioGearsSalineBolus) for a in scenario.actions):
            fresh_result = _execute_sequential_saline_uncached(
                scenario=scenario,
                bridge=bridge,
                timeout_seconds=timeout_seconds,
                tolerance_s=tolerance_s,
                keep_output=keep_output,
            )
        else:
            fresh_result = _execute_sequential_uncached(
                scenario=scenario,
                bridge=bridge,
                timeout_seconds=timeout_seconds,
                tolerance_s=tolerance_s,
                keep_output=keep_output,
            )

        cache.put(fresh_result, fingerprint=fingerprint)
        fresh_result.cache_fingerprint = fingerprint
        fresh_result.is_cached = False
        return fresh_result


def execute_sequential_saline_scenario(
    scenario: BioGearsSequentialScenario,
    bridge: Optional[BioGearsBridge] = None,
    timeout_seconds: int = 600,
    tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
    keep_output: bool = True,
    use_cache: bool = True,
    cache: Optional[Any] = None,
) -> SequentialExecutionResult:
    """Execute a saline-only sequential BioGears scenario with deterministic result caching.
    Preserved for strict Phase D Step B/C/G backward-compatibility and testing."""
    if not isinstance(scenario, BioGearsSequentialScenario):
        raise BioGearsScenarioBuilderError(
            f"Expected BioGearsSequentialScenario, got {type(scenario).__name__}."
        )

    if scenario.is_mixed_intervention:
        raise BioGearsScenarioBuilderError(
            "Mixed intervention execution (Saline followed by Norepinephrine) is [BLOCKED] "
            "pending verification of continuous single-substance execution in Phase D Step I."
        )

    for i, action in enumerate(scenario.actions):
        if not isinstance(action, BioGearsSalineBolus):
            raise BioGearsScenarioBuilderError(
                f"Action {i} is {type(action).__name__}, not BioGearsSalineBolus. "
                f"Phase D Step B/C supports saline-only sequential scenarios."
            )

    return execute_sequential_scenario(
        scenario=scenario,
        bridge=bridge,
        timeout_seconds=timeout_seconds,
        tolerance_s=tolerance_s,
        keep_output=keep_output,
        use_cache=use_cache,
        cache=cache,
    )
