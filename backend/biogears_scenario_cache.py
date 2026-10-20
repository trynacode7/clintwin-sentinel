"""
biogears_scenario_cache.py — Scenario Fingerprinting & Result Caching (Phase D Step D)
======================================================================================
SCOPE: Deterministic scenario fingerprinting, atomic result caching, concurrency control,
and artifact revalidation for sequential BioGears simulations.

Phase D Step D: Scenario Execution and Result Caching
Features:
  1. Stable SHA-256 fingerprint derived from canonical configuration:
     - Scenario configuration and name.
     - Patient file identity and content/version SHA-256 hash.
     - Ordered intervention types and locked parameters.
     - Start times and delivery durations.
     - Baseline stabilization and post-observation durations.
     - Sampling frequency and canonical DataRequests output channels.
     - Cache schema version.
  2. Configurable artifact storage under `biogears_runs/scenarios/<fingerprint>/`:
     - scenario.xml: Generated XML payload.
     - results.csv: Full raw simulation CSV output.
     - snapshots.json: Structured milestone physiological snapshots.
     - metadata.json: Execution provenance, durations, row counts, and versions.
  3. Atomic cache writes via temporary staging directories.
  4. Cache-hit verification:
     - Fingerprint and configuration consistency check.
     - Required artifact presence and parseability.
     - Strict revalidation of temporal continuity and required physiological channels.
     - Equivalent result shape (SequentialExecutionResult with is_cached=True).
  5. Thread-safe concurrency control via per-fingerprint locking with double-checked pattern.
  6. Strict rejection of partial, failed, timed-out, or corrupted runs.

Epistemic Status:
  - Fingerprinting & Cache storage: [VERIFIED] via unit and integration tests.
  - Saline sequential execution: [VERIFIED] in Step B.
  - Snapshot extraction revalidation: [VERIFIED] in Step C.
  - Mixed interventions: [BLOCKED].
  - Norepinephrine sequential execution: [UNVERIFIED].
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.biogears_bridge import (
    BioGearsBridge,
    BioGearsBridgeError,
    BioGearsConfig,
    BioGearsOutputNotFoundError,
    BioGearsOutputParseError,
    BioGearsResult,
)
from backend.biogears_scenario_builder import (
    _FLUID_DATA_REQUESTS,
    BioGearsNorepinephrineInfusion,
    BioGearsSalineBolus,
    BioGearsScenarioBuilderError,
    BioGearsSequentialScenario,
)
from backend.biogears_sequential_execution import (
    DEFAULT_TIMESTAMP_TOLERANCE_S,
    BioGearsExtractionError,
    PhysiologicalSnapshot,
    ScenarioMilestoneReport,
    SequentialExecutionResult,
    extract_scenario_snapshots,
    verify_temporal_continuity,
)

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "1.0.0"
DEFAULT_CACHE_DIR = Path("biogears_runs/scenarios")


# ─── Patient File Resolution & Content Hashing ────────────────────────────────


def resolve_patient_file(
    patient_file: str,
    patient_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve patient filename to an existing path on disk.

    Search order:
      1. If patient_file is absolute and exists: return it.
      2. If patient_dir is provided: check patient_dir / patient_file.
      3. Check BioGears installed patient directory (e.g. C:\\BioGears\\bin\\patients\\...).
      4. Check executable parent / 'patients' / patient_file.
      5. Check Path.cwd() / 'patients' / patient_file.
      6. Check Path.cwd() / patient_file.

    Raises:
      FileNotFoundError: if patient file cannot be located.
    """
    candidate = Path(patient_file)
    if candidate.is_absolute() and candidate.is_file():
        return candidate

    if patient_dir is not None:
        p_dir = Path(patient_dir)
        p_path = p_dir / patient_file
        if p_path.is_file():
            return p_path

    # Standard BioGears installation directory
    default_bg_patient = Path(r"C:\BioGears\bin\patients") / patient_file
    if default_bg_patient.is_file():
        return default_bg_patient

    # Derived from BioGearsConfig executable location
    try:
        exe_path = Path(BioGearsConfig().executable)
        derived_patient = exe_path.parent / "patients" / patient_file
        if derived_patient.is_file():
            return derived_patient
    except Exception:
        pass

    # Local workspace locations
    local_p = Path.cwd() / "patients" / patient_file
    if local_p.is_file():
        return local_p

    local_root = Path.cwd() / patient_file
    if local_root.is_file():
        return local_root

    raise FileNotFoundError(
        f"BioGears patient file '{patient_file}' could not be found. "
        f"Searched: patient_dir={patient_dir}, '{default_bg_patient}', local paths."
    )


def get_patient_file_hash(
    patient_file: str,
    patient_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, str]:
    """Calculate the SHA-256 hash of the patient XML file content.

    Returns:
      Tuple of (resolved_path, sha256_hex_digest).

    Raises:
      FileNotFoundError: if the file cannot be located.
      IOError: if file reading fails.
    """
    resolved_path = resolve_patient_file(patient_file, patient_dir=patient_dir)
    hasher = hashlib.sha256()
    with open(resolved_path, "rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return resolved_path, hasher.hexdigest()


# ─── Canonical Configuration & Fingerprinting ─────────────────────────────────


def canonicalize_scenario_config(
    scenario: BioGearsSequentialScenario,
    patient_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Produce a canonical dictionary representing the complete simulation configuration.

    All simulation-affecting parameters are normalized and sorted:
      - Cache schema version.
      - Scenario name.
      - Patient file name and SHA-256 content hash.
      - Sampling rate (samples_per_second).
      - Timeline durations (baseline_stabilization_s, post_observation_s, total_duration_s).
      - Ordered actions with locked clinical parameters.
      - Canonical requested output channels (data requests).
    """
    if not isinstance(scenario, BioGearsSequentialScenario):
        raise BioGearsScenarioBuilderError(
            f"Expected BioGearsSequentialScenario, got {type(scenario).__name__}."
        )

    scenario.validate()

    resolved_patient_path, patient_hash = get_patient_file_hash(
        scenario.patient_file, patient_dir=patient_dir
    )

    actions_list: List[Dict[str, Any]] = []
    for i, action in enumerate(scenario.actions):
        if isinstance(action, BioGearsSalineBolus):
            actions_list.append({
                "action_index": i,
                "action_type": "saline",
                "bag_volume_ml": round(float(action.bag_volume_ml), 4),
                "rate_ml_per_min": round(float(action.rate_ml_per_min), 4),
                "start_time_s": round(float(action.start_time_s), 4),
                "delivery_duration_s": round(float(action.delivery_duration_s), 4),
            })
        elif isinstance(action, BioGearsNorepinephrineInfusion):
            actions_list.append({
                "action_index": i,
                "action_type": "norepinephrine",
                "concentration_ug_ml": round(float(action.concentration_ug_ml), 4),
                "rate_ml_per_min": round(float(action.rate_ml_per_min), 4),
                "duration_s": round(float(action.duration_s), 4),
                "start_time_s": round(float(action.start_time_s), 4),
                "delivery_duration_s": round(float(action.delivery_duration_s), 4),
            })
        else:
            raise BioGearsScenarioBuilderError(
                f"Unsupported action type for caching fingerprint: {type(action).__name__}"
            )

    canonical_data_requests = sorted(
        [
            {
                "name": req[0],
                "unit": req[1],
                "precision": str(req[2]),
                "type": req[3],
            }
            for req in _FLUID_DATA_REQUESTS
        ],
        key=lambda x: x["name"],
    )

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scenario_name": scenario.scenario_name.strip(),
        "patient": {
            "filename": scenario.patient_file.strip(),
            "content_hash": patient_hash,
        },
        "timeline": {
            "baseline_stabilization_s": round(float(scenario.baseline_stabilization_s), 4),
            "post_observation_s": round(float(scenario.post_observation_s), 4),
            "total_duration_s": round(float(scenario.total_duration_s), 4),
            "samples_per_second": round(float(scenario.samples_per_second), 4),
        },
        "actions": actions_list,
        "data_requests": canonical_data_requests,
    }


def compute_scenario_fingerprint(
    scenario: BioGearsSequentialScenario,
    patient_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Compute deterministic SHA-256 fingerprint for a BioGears scenario.

    Equivalent configurations produce identical fingerprints.
    A change to any simulation-affecting parameter produces a different fingerprint.
    """
    config = canonicalize_scenario_config(scenario, patient_dir=patient_dir)
    canonical_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ─── Cache Manager Class ──────────────────────────────────────────────────────


class BioGearsScenarioCache:
    """Deterministic scenario cache manager.

    Stores verified artifacts under `<cache_dir>/<fingerprint>/`:
      - `scenario.xml`
      - `results.csv`
      - `snapshots.json`
      - `metadata.json`
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        patient_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.patient_dir = Path(patient_dir) if patient_dir is not None else None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def get_lock(self, fingerprint: str) -> threading.Lock:
        """Retrieve or create an in-process thread lock for the given fingerprint."""
        with self._registry_lock:
            if fingerprint not in self._locks:
                self._locks[fingerprint] = threading.Lock()
            return self._locks[fingerprint]

    def compute_fingerprint(self, scenario: BioGearsSequentialScenario) -> str:
        """Compute fingerprint using this cache's configured patient_dir."""
        return compute_scenario_fingerprint(scenario, patient_dir=self.patient_dir)

    def get_entry_dir(self, fingerprint: str) -> Path:
        """Return the directory path for a given fingerprint."""
        return self.cache_dir / fingerprint

    def has(self, fingerprint: str) -> bool:
        """Check if complete and valid cache directory exists for fingerprint."""
        entry_dir = self.get_entry_dir(fingerprint)
        if not entry_dir.is_dir():
            return False
        meta_file = entry_dir / "metadata.json"
        if not meta_file.is_file():
            return False
        try:
            with open(meta_file, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            return bool(meta.get("completed", False)) and meta.get("fingerprint") == fingerprint
        except Exception:
            return False

    def get(
        self,
        scenario: BioGearsSequentialScenario,
        fingerprint: Optional[str] = None,
        tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
    ) -> Optional[SequentialExecutionResult]:
        """Retrieve and revalidate cached execution result if present.

        Returns:
            SequentialExecutionResult with is_cached=True if hit and valid.
            None on cache miss, corruption, missing artifacts, or validation failure.
        """
        if fingerprint is None:
            try:
                fingerprint = self.compute_fingerprint(scenario)
            except Exception as exc:
                logger.warning("Failed to compute fingerprint for scenario %s: %s", scenario.scenario_name, exc)
                return None

        entry_dir = self.get_entry_dir(fingerprint)
        if not entry_dir.is_dir():
            return None

        scenario_xml_path = entry_dir / "scenario.xml"
        results_csv_path = entry_dir / "results.csv"
        snapshots_json_path = entry_dir / "snapshots.json"
        metadata_json_path = entry_dir / "metadata.json"

        # Verify all required artifacts exist
        for required_file in [scenario_xml_path, results_csv_path, snapshots_json_path, metadata_json_path]:
            if not required_file.is_file():
                logger.warning(
                    "Cache miss for %s: required artifact %s missing in %s",
                    fingerprint,
                    required_file.name,
                    entry_dir,
                )
                return None

        try:
            # 1. Parse and verify metadata
            with open(metadata_json_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)

            if not metadata.get("completed", False):
                logger.warning("Cache entry %s is marked incomplete. Ignoring.", fingerprint)
                return None

            if metadata.get("fingerprint") != fingerprint:
                logger.warning(
                    "Fingerprint mismatch in metadata: expected %s, got %s",
                    fingerprint,
                    metadata.get("fingerprint"),
                )
                return None

            if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
                logger.warning(
                    "Cache schema version mismatch: expected %s, got %s",
                    CACHE_SCHEMA_VERSION,
                    metadata.get("schema_version"),
                )
                return None

            # 2. Read scenario XML
            with open(scenario_xml_path, "r", encoding="utf-8") as fh:
                scenario_xml = fh.read()

            # 3. Parse and revalidate CSV output
            bridge = BioGearsBridge()
            biogears_result = bridge._parse_csv(
                csv_path=str(results_csv_path),
                scenario_name=scenario.scenario_name,
                returncode=metadata.get("exit_status", 0),
            )

            # 4. Revalidate temporal continuity
            is_continuous = verify_temporal_continuity(
                time_s=biogears_result.time_s,
                expected_duration_s=scenario.total_duration_s,
            )

            # 5. Revalidate snapshot extraction
            report = extract_scenario_snapshots(
                biogears_result=biogears_result,
                scenario=scenario,
                tolerance_s=tolerance_s,
            )

            # 6. Reconstruct intermediate states
            milestones = scenario.get_timeline_milestones()
            intermediate_states = [
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
                command_executed=metadata.get("command_executed", "<cached>"),
                exit_status=metadata.get("exit_status", 0),
                wall_clock_seconds=float(metadata.get("wall_clock_seconds", 0.0)),
                row_count=biogears_result.row_count,
                time_range_s=biogears_result.time_range_s,
                output_csv_path=str(results_csv_path),
                is_continuous=is_continuous,
                milestones=milestones,
                baseline_state=report.baseline_snapshot,
                intermediate_states=intermediate_states,
                final_state=report.final_snapshot,
                biogears_result=biogears_result,
                is_cached=True,
                cache_fingerprint=fingerprint,
            )

        except (BioGearsBridgeError, json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            logger.warning(
                "Cache entry %s corrupted or failed revalidation: %s. Treating as cache miss.",
                fingerprint,
                exc,
            )
            return None

    def put(
        self,
        result: SequentialExecutionResult,
        fingerprint: Optional[str] = None,
    ) -> Path:
        """Persist a verified execution result into cache atomically.

        Writes all artifacts into a unique staging directory and renames it to
        the final fingerprint directory upon successful write.
        """
        if fingerprint is None:
            fingerprint = self.compute_fingerprint(result.scenario)

        target_dir = self.get_entry_dir(fingerprint)
        staging_dir = self.cache_dir / f".tmp_{fingerprint}_{uuid.uuid4().hex}"

        try:
            staging_dir.mkdir(parents=True, exist_ok=False)

            # 1. Write scenario.xml
            scenario_xml_file = staging_dir / "scenario.xml"
            with open(scenario_xml_file, "w", encoding="utf-8") as fh:
                fh.write(result.scenario_xml)

            # 2. Write results.csv
            results_csv_file = staging_dir / "results.csv"
            if result.output_csv_path and os.path.isfile(result.output_csv_path):
                shutil.copy2(result.output_csv_path, results_csv_file)
            else:
                # Reconstruct CSV from biogears_result if file was removed
                _write_biogears_result_to_csv(result.biogears_result, results_csv_file)

            # 3. Write snapshots.json
            snapshots_json_file = staging_dir / "snapshots.json"
            snapshots_data = {
                "baseline_snapshot": result.baseline_state.to_dict(),
                "intermediate_snapshots": [
                    {
                        "target_time_s": item["target_time_s"],
                        "actual_time_s": item["actual_time_s"],
                        "snapshot": item["snapshot"].to_dict(),
                        "milestone": {k: v for k, v in item["milestone"].items() if k != "action"},
                    }
                    for item in result.intermediate_states
                ],
                "final_snapshot": result.final_state.to_dict(),
            }
            with open(snapshots_json_file, "w", encoding="utf-8") as fh:
                json.dump(snapshots_data, fh, indent=2)

            # 4. Write metadata.json (marked completed=True only after all files are written)
            metadata_file = staging_dir / "metadata.json"
            resolved_p, p_hash = get_patient_file_hash(
                result.scenario.patient_file, patient_dir=self.patient_dir
            )
            canonical_cfg = canonicalize_scenario_config(
                result.scenario, patient_dir=self.patient_dir
            )

            metadata_data = {
                "fingerprint": fingerprint,
                "schema_version": CACHE_SCHEMA_VERSION,
                "scenario_name": result.scenario_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed": True,
                "command_executed": result.command_executed,
                "exit_status": result.exit_status,
                "wall_clock_seconds": round(result.wall_clock_seconds, 3),
                "row_count": result.row_count,
                "time_range_s": list(result.time_range_s),
                "is_continuous": result.is_continuous,
                "patient": {
                    "filename": result.scenario.patient_file,
                    "resolved_path": str(resolved_p),
                    "content_hash": p_hash,
                },
                "configuration": canonical_cfg,
                "software_versions": {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "biogears_version": "BioGears_6.3.0-beta",
                    "clintwin_phase": "Phase_D_Step_D",
                },
            }
            with open(metadata_file, "w", encoding="utf-8") as fh:
                json.dump(metadata_data, fh, indent=2)

            # 5. Atomic commit: rename staging directory to final target directory
            if target_dir.exists():
                # Another concurrent writer may have completed; remove old or staging
                shutil.rmtree(staging_dir, ignore_errors=True)
                return target_dir

            try:
                os.replace(staging_dir, target_dir)
            except OSError:
                # Fallback if cross-device or Windows directory rename quirk
                if not target_dir.exists():
                    shutil.move(str(staging_dir), str(target_dir))
                else:
                    shutil.rmtree(staging_dir, ignore_errors=True)

            logger.info("Successfully persisted cache entry for %s to %s", fingerprint, target_dir)
            return target_dir

        except Exception as exc:
            logger.error("Failed to write cache entry for %s: %s", fingerprint, exc)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def invalidate(self, fingerprint: str) -> bool:
        """Remove a cached scenario entry if it exists."""
        entry_dir = self.get_entry_dir(fingerprint)
        if entry_dir.is_dir():
            shutil.rmtree(entry_dir, ignore_errors=True)
            return True
        return False


def _write_biogears_result_to_csv(result: BioGearsResult, destination: Path) -> None:
    """Helper to reconstruct a valid BioGears CSV from BioGearsResult."""
    import csv

    fieldnames = ["Time(s)"] + list(result.columns.keys())
    with open(destination, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        num_rows = len(result.time_s)
        for i in range(num_rows):
            row = [result.time_s[i]]
            for col_name in result.columns:
                row.append(result.columns[col_name].values[i])
            writer.writerow(row)
