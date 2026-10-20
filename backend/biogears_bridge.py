"""
biogears_bridge.py — BioGears 6.3.0-beta Execution and Output-Parsing Bridge
==============================================================================
SCOPE: Execution infrastructure and CSV output parsing ONLY.

This module:
  - Accepts a BioGears scenario XML (as a file path or a content string)
  - Invokes C:\\BioGears\\bin\\bg-scenario.exe via subprocess with absolute paths
  - Captures return code, stdout, and stderr
  - Locates and parses the output CSV into structured Python dataclasses
  - Manages temporary files with UUID-based naming to avoid collisions
  - Raises typed exceptions on every failure mode

This module does NOT:
  - Generate BioGears scenario XML (no action types, no patient parameters)
  - Map ClinTwin features (vasopressor_dose, fluid_bolus_ml, fio2, ventilated)
    to BioGears action types — those mappings require pending clinical domain
    decisions and will live in a separate future module
  - Modify whatif.py, api.py, the frontend, config.py, or any ML code
  - Fabricate, interpolate, or invent simulation results
  - Run InfectionSevere.xml or any existing BioGears scenario

VERIFIED FACTS (source: biogears_integration_investigation.md, 2026-08-26,
and ClinTwinTinyTest verification runs conducted the same date):

  Executable location:
    C:\\BioGears\\bin\\bg-scenario.exe  (694,784 bytes)

  CLI invocation (VERIFIED — 3 independent runs):
    bg-scenario.exe scenario <abs_xml_path> results <abs_prefix>

  CWD requirement (VERIFIED — run 3):
    NOT required to be C:\\BioGears\\bin when absolute paths are supplied.
    bg-scenario.exe resolves patients/ and substances/ relative to the
    executable directory, not the CWD.

  Output naming rule (VERIFIED — all 3 runs with results= parameter):
    Output CSV  = {results_prefix}Results.csv
    Output log  = {results_prefix}.log
    No double-extension quirk when results= parameter is supplied.

  Wall-clock time (VERIFIED — ClinTwinTinyTest, 2-minute simulated scenario):
    ~69 seconds total:
      ~56 seconds: patient stabilization (fixed cost, every fresh run)
      ~13 seconds: actual simulation of 2 simulated minutes

  CSV structure (VERIFIED — ClinTwinTinyTest, SamplesPerSecond=0.2, 2-min sim):
    Row 0:    header, comma-separated column names
    Rows 1+:  data rows, comma-separated floats
    Row count: 24 data rows (= 2min × 60s/min × 0.2 samples/s)
    Time(s):  always first column; first value at one sampling interval
              after t=0 (not at t=0 itself)
    CTSresistance: always second column; value 0 at baseline

  Column name format (VERIFIED from actual output):
    Most columns:  "Name(Unit)"  e.g. HeartRate(1/min), MeanArterialPressure(mmHg)
    Some columns:  bare "Name"  e.g. OxygenSaturation  (no unit suffix in header)
    Both forms are preserved exactly as BioGears emits them.

  Exit codes (VERIFIED):
    0   = success
    ≠0  = failure (details in stdout)
"""

import csv
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Exceptions ───────────────────────────────────────────────────────────────


class BioGearsBridgeError(Exception):
    """Base class for all bridge errors. Always carries an informative message.

    Subclass by failure mode so callers can react specifically (e.g. retry on
    timeout, alert on missing executable, log-and-fallback on process error).
    """


class BioGearsNotFoundError(BioGearsBridgeError):
    """bg-scenario.exe not found at the configured path.

    Most likely cause: BioGears is not installed, or BioGearsConfig.executable
    points to the wrong location.
    """


class BioGearsProcessError(BioGearsBridgeError):
    """bg-scenario.exe exited with a non-zero return code.

    Attributes:
        returncode (int): The process exit code (always non-zero here).
        stdout (str): Full captured stdout from bg-scenario.exe.
        stderr (str): Full captured stderr from bg-scenario.exe.
    """

    def __init__(
        self,
        message: str,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class BioGearsTimeoutError(BioGearsBridgeError):
    """bg-scenario.exe did not complete within the configured timeout.

    Attributes:
        timeout_seconds (int): The timeout value (BioGearsConfig.timeout_seconds)
            that was exceeded.
    """

    def __init__(self, message: str, timeout_seconds: int) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class BioGearsOutputNotFoundError(BioGearsBridgeError):
    """bg-scenario.exe exited 0 but the expected output CSV was not produced.

    This can happen if the scenario XML contains no <DataRequests> block or
    if BioGears wrote output to an unexpected path.

    Attributes:
        expected_path (str): The absolute path where the CSV was expected.
    """

    def __init__(self, message: str, expected_path: str) -> None:
        super().__init__(message)
        self.expected_path = expected_path


class BioGearsOutputParseError(BioGearsBridgeError):
    """The output CSV was found but could not be parsed into a valid result.

    Attributes:
        csv_path (str): Path to the CSV file that failed to parse.
        reason (str): Human-readable explanation of the parse failure.
    """

    def __init__(self, message: str, csv_path: str, reason: str) -> None:
        super().__init__(message)
        self.csv_path = csv_path
        self.reason = reason


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BioGearsColumn:
    """One column from a BioGears output CSV, with name and unit cleanly separated.

    BioGears emits column headers in two forms (both VERIFIED from actual output):
      1. "Name(Unit)"  e.g. "HeartRate(1/min)", "MeanArterialPressure(mmHg)"
      2. Bare "Name"   e.g. "OxygenSaturation", "CTSresistance"

    Both forms are stored here. raw_header is the exact string from the CSV
    header — it is never modified, so callers can use it as a lossless key
    back into the original CSV structure.
    """

    raw_header: str  # Exact string from the CSV header row, e.g. "HeartRate(1/min)"
    name: str        # Signal name, e.g. "HeartRate" or "OxygenSaturation"
    unit: str        # Unit string, e.g. "1/min" or "mmHg"; empty string if no unit suffix


@dataclass
class BioGearsTimeSeries:
    """A single time-series variable from one BioGears scenario run.

    `times_s` and `values` are parallel lists of equal length. The time axis
    is shared with all other columns in the same BioGearsResult (they all
    derive from the same Time(s) column).

    Values are floats. BioGears occasionally emits blank cells on convergence
    failure; the parser will raise BioGearsOutputParseError on non-numeric
    cells rather than silently substituting NaN — the caller should not need
    to guess whether a value is real or fabricated.
    """

    column: BioGearsColumn
    times_s: List[float]  # Shared time axis: Time(s) column values
    values: List[float]   # Per-row values for this column


@dataclass
class BioGearsResult:
    """Complete parsed output from one BioGears scenario run.

    All time-series columns except Time(s) are available in `columns`, keyed
    by their raw CSV header string (e.g. "HeartRate(1/min)", "OxygenSaturation").
    CTSresistance is included verbatim — the bridge never silently drops
    columns BioGears emits.

    `final_state` is a convenience lookup for the last row's values.
    It is exactly {raw_header: ts.values[-1]} for every column in `columns`.

    `returncode` is always 0 when a BioGearsResult is returned (the bridge
    raises BioGearsProcessError on non-zero exit). It is included for
    informational completeness.

    stdout is NOT stored here on success. It is available via
    BioGearsProcessError.stdout when a run fails, and via keep_output=True
    for manual debugging of successful runs (inspect the .log file).

    VERIFIED output structure (ClinTwinTinyTest, 2-min sim, SamplesPerSecond=0.2):
      row_count:    24  (= 2min × 60s/min × 0.2 samples/s)
      time_range_s: (5.02, 120.02)
      column count (excl. Time): 7  (CTSresistance + 6 DataRequest columns)
    """

    scenario_name: str
    time_s: List[float]
    columns: Dict[str, BioGearsTimeSeries]  # raw_header → BioGearsTimeSeries (excl. Time(s))
    final_state: Dict[str, float]           # raw_header → last-row value (convenience)
    row_count: int
    time_range_s: Tuple[float, float]       # (first_time_s, last_time_s)
    returncode: int                         # Always 0 when this object exists
    output_csv_path: Optional[str] = None


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BioGearsConfig:
    """Runtime configuration for BioGearsBridge.

    All path fields should be absolute paths. No network access, no ClinTwin
    ML model knowledge, and no clinical intervention logic live here.

    VERIFIED facts baked into the defaults:
      executable: confirmed present at C:\\BioGears\\bin\\bg-scenario.exe
      timeout_seconds: 300s (5 min) gives a generous margin over the observed
          ~69s for a 2-minute simulated scenario on this machine
      tmp_dir: dedicated subdirectory under BioGears\\ keeps temp files out of
          the main Scenarios\\Patient\\ browser view; created automatically
    """

    executable: str = r"C:\BioGears\bin\bg-scenario.exe"
    timeout_seconds: int = 300
    tmp_dir: str = r"C:\BioGears\bin\Scenarios\tmp"


# ─── Bridge class ─────────────────────────────────────────────────────────────


class BioGearsBridge:
    """Execute a BioGears scenario XML file and return parsed output.

    Primary public method: run_scenario_file(scenario_xml_path)

    A thin secondary method run_scenario_xml() accepts an XML string, writes
    it to a temp file, delegates to run_scenario_file(), and cleans up the
    temp XML in a finally block. It exists because the integration tests need
    to supply XML content without managing temp files themselves.

    This class is intentionally minimal for Phase 1:
      - It executes a caller-supplied scenario XML and returns parsed data.
      - It does NOT generate scenario XML.
      - It does NOT know about ClinTwin features or interventions.
      - It does NOT modify any other ClinTwin module.

    Thread safety: The config is read-only after construction. Each call to
    run_scenario_file() / run_scenario_xml() generates a unique UUID prefix
    and is independently safe — concurrent calls will not collide on outputs.
    """

    def __init__(self, config: Optional[BioGearsConfig] = None) -> None:
        self._config = config if config is not None else BioGearsConfig()

    @property
    def config(self) -> BioGearsConfig:
        """The configuration this bridge instance was created with."""
        return self._config

    @property
    def is_available(self) -> bool:
        """True if the BioGears executable exists and is a file."""
        return os.path.isfile(self._config.executable)

    # ─── Primary public method ────────────────────────────────────────────

    def run_scenario_file(
        self,
        scenario_xml_path: str,
        *,
        scenario_name: str = "BioGearsScenario",
        keep_output: bool = False,
    ) -> BioGearsResult:
        """Run a pre-existing BioGears scenario XML file and return parsed output.

        The caller is responsible for providing a valid, readable BioGears
        scenario XML at `scenario_xml_path`. This method does NOT create or
        delete that file — it only manages the output CSV and log that
        BioGears produces.

        A UUID-based results prefix is generated on every call so that
        concurrent calls with the same scenario file are safe.

        Args:
            scenario_xml_path: Absolute path to an existing BioGears scenario
                XML file. Must be readable. Does not have to reside inside
                C:\\BioGears\\bin\\ (VERIFIED: absolute path resolution works
                from any CWD).
            scenario_name: Human-readable tag included in the results filename
                prefix. Does not affect BioGears execution — only makes the
                output files identifiable in logs and temp directories.
                Default "BioGearsScenario".
            keep_output: If True, the output CSV and .log files produced by
                BioGears are NOT deleted after parsing. Useful for manual
                inspection. Default False (auto-cleanup).

        Returns:
            BioGearsResult — fully parsed time-series and final state.
            All column names are preserved exactly as BioGears emits them.

        Raises:
            BioGearsNotFoundError:       executable missing at config path
            BioGearsProcessError:        bg-scenario.exe exited non-zero
            BioGearsTimeoutError:        run exceeded config.timeout_seconds
            BioGearsOutputNotFoundError: exit 0 but CSV not produced
            BioGearsOutputParseError:    CSV found but unreadable/malformed
        """
        self._check_executable()

        tmp_dir = self._ensure_tmp_dir()
        results_prefix = self._build_results_prefix(tmp_dir, scenario_name)
        output_csv = results_prefix + "Results.csv"
        output_log = results_prefix + ".log"

        returncode, _stdout, _stderr = self._invoke_biogears(
            scenario_xml_path=scenario_xml_path,
            results_prefix=results_prefix,
        )

        try:
            csv_path = self._find_output_csv(results_prefix)
            result = self._parse_csv(
                csv_path=csv_path,
                scenario_name=scenario_name,
                returncode=returncode,
            )
        finally:
            if not keep_output:
                self._cleanup_outputs(output_csv, output_log)

        return result

    # ─── Secondary public method (thin wrapper) ───────────────────────────

    def run_scenario_xml(
        self,
        scenario_xml_content: str,
        scenario_name: str = "ClinTwinScenario",
        *,
        keep_output: bool = False,
    ) -> BioGearsResult:
        """Write a scenario XML string to a temp file and run it.

        The temporary XML file is always deleted in a finally block regardless
        of success or failure. Output CSV and log are deleted after parsing
        unless keep_output=True.

        This is a thin wrapper around run_scenario_file() for callers that
        have scenario XML as a string rather than an on-disk file path.

        Args:
            scenario_xml_content: Full BioGears scenario XML as a string.
            scenario_name: Used as a prefix in the temp XML filename and the
                results prefix. Default "ClinTwinScenario".
            keep_output: Passed through to run_scenario_file().

        Returns:
            BioGearsResult — same as run_scenario_file().

        Raises: same exceptions as run_scenario_file().
        """
        tmp_dir = self._ensure_tmp_dir()
        unique_tag = uuid.uuid4().hex[:12]
        safe_name = _sanitize_name(scenario_name)
        tmp_xml_path = os.path.join(tmp_dir, f"{safe_name}_{unique_tag}.xml")

        try:
            with open(tmp_xml_path, "w", encoding="utf-8") as fh:
                fh.write(scenario_xml_content)
            logger.debug("Wrote temporary scenario XML: %s", tmp_xml_path)
            return self.run_scenario_file(
                tmp_xml_path,
                scenario_name=scenario_name,
                keep_output=keep_output,
            )
        finally:
            if os.path.exists(tmp_xml_path):
                try:
                    os.remove(tmp_xml_path)
                    logger.debug("Removed temporary scenario XML: %s", tmp_xml_path)
                except OSError as exc:
                    logger.warning(
                        "Could not remove temporary scenario XML %s: %s",
                        tmp_xml_path,
                        exc,
                    )

    # ─── Private helpers ──────────────────────────────────────────────────

    def _check_executable(self) -> None:
        """Raise BioGearsNotFoundError if the executable is missing."""
        if not os.path.isfile(self._config.executable):
            raise BioGearsNotFoundError(
                f"BioGears executable not found at: {self._config.executable!r}. "
                f"Verify that BioGears 6.3.0-beta is installed and that "
                f"BioGearsConfig.executable points to the correct bg-scenario.exe path."
            )

    def _ensure_tmp_dir(self) -> str:
        """Create the tmp_dir if it does not exist and return its path."""
        os.makedirs(self._config.tmp_dir, exist_ok=True)
        return self._config.tmp_dir

    def _build_results_prefix(self, tmp_dir: str, scenario_name: str) -> str:
        """Return a unique, absolute results prefix for this run.

        VERIFIED naming rule: BioGears appends "Results.csv" to the prefix
        supplied via the results= CLI parameter.

        The prefix encodes the scenario name (sanitized) and a UUID fragment
        so that multiple concurrent runs and repeated runs of the same scenario
        never collide on the filesystem.
        """
        unique_tag = uuid.uuid4().hex[:12]
        safe_name = _sanitize_name(scenario_name)
        return os.path.join(tmp_dir, f"{safe_name}_{unique_tag}")

    def _invoke_biogears(
        self,
        scenario_xml_path: str,
        results_prefix: str,
    ) -> Tuple[int, str, str]:
        """Invoke bg-scenario.exe and return (returncode, stdout, stderr).

        VERIFIED invocation (from investigation and ClinTwinTinyTest runs):
          bg-scenario.exe scenario <abs_xml_path> results <abs_prefix>

        CWD does not matter because absolute paths are used for both
        scenario= and results=. VERIFIED: run from C:\\Users\\liss produced
        identical output to run from C:\\BioGears\\bin.
        """
        abs_xml = os.path.abspath(scenario_xml_path)
        abs_prefix = os.path.abspath(results_prefix)

        cmd = [
            self._config.executable,
            "scenario", abs_xml,
            "results", abs_prefix,
        ]
        logger.info(
            "Invoking BioGears: %s",
            " ".join(f'"{c}"' if " " in c else c for c in cmd),
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise BioGearsTimeoutError(
                f"BioGears did not complete within {self._config.timeout_seconds}s. "
                f"Scenario: {abs_xml!r}. "
                f"Consider increasing BioGearsConfig.timeout_seconds for longer "
                f"simulated durations (VERIFIED: 2-min simulated time takes ~69s "
                f"wall-clock including patient stabilization).",
                timeout_seconds=self._config.timeout_seconds,
            ) from exc

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            # Include the last 500 chars of stdout — BioGears writes diagnostic
            # information there, not to stderr.
            snippet = stdout[-500:] if stdout else "<empty>"
            raise BioGearsProcessError(
                f"BioGears exited with code {proc.returncode}. "
                f"Scenario: {abs_xml!r}. "
                f"Stdout tail: {snippet!r}",
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        logger.info("BioGears completed successfully (returncode=0).")
        return proc.returncode, stdout, stderr

    def _find_output_csv(self, results_prefix: str) -> str:
        """Locate the output CSV produced by BioGears after a successful run.

        VERIFIED naming rule: {results_prefix}Results.csv
        No double-extension quirk when the results= CLI parameter is used.
        """
        expected = results_prefix + "Results.csv"
        if not os.path.isfile(expected):
            raise BioGearsOutputNotFoundError(
                f"BioGears exited successfully (returncode=0) but the expected "
                f"output CSV was not found at: {expected!r}. "
                f"This may indicate that the scenario XML contains no "
                f"<DataRequests> block, or that BioGears wrote output to an "
                f"unexpected path (check the .log file for details).",
                expected_path=expected,
            )
        return expected

    def _parse_csv(
        self,
        csv_path: str,
        scenario_name: str,
        returncode: int,
    ) -> BioGearsResult:
        """Parse a BioGears output CSV into a BioGearsResult.

        VERIFIED CSV structure (from ClinTwinTinyTest run, 2026-08-26):
          - Row 0: header row, comma-separated column name strings
          - Rows 1+: data rows, comma-separated float strings
          - Time(s): always the first column
          - CTSresistance: always the second column (0 at baseline)
          - OxygenSaturation: appears without unit suffix in header (verified)
          - All other columns: "Name(Unit)" format

        Raises BioGearsOutputParseError on any structural problem:
          empty file, missing Time(s) column, header-only file, column count
          mismatch, or non-numeric cell.
        """
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                try:
                    raw_headers = next(reader)
                except StopIteration:
                    raise BioGearsOutputParseError(
                        f"BioGears output CSV is completely empty (no header row): "
                        f"{csv_path!r}",
                        csv_path=csv_path,
                        reason="empty file",
                    )
                data_rows = list(reader)
        except OSError as exc:
            raise BioGearsOutputParseError(
                f"Could not open or read BioGears output CSV {csv_path!r}: {exc}",
                csv_path=csv_path,
                reason=str(exc),
            ) from exc

        # ── Header validation ──
        if not raw_headers:
            raise BioGearsOutputParseError(
                f"BioGears output CSV has an empty header row: {csv_path!r}",
                csv_path=csv_path,
                reason="empty header row",
            )

        first_col = raw_headers[0].strip()
        if first_col != "Time(s)":
            raise BioGearsOutputParseError(
                f"Expected first column to be 'Time(s)', got {first_col!r}: "
                f"{csv_path!r}",
                csv_path=csv_path,
                reason=f"unexpected first column: {first_col!r}",
            )

        if not data_rows:
            raise BioGearsOutputParseError(
                f"BioGears output CSV has a header but no data rows: {csv_path!r}",
                csv_path=csv_path,
                reason="header present but zero data rows",
            )

        # ── Parse column headers ──
        parsed_cols = [
            self._parse_column_header(h.strip()) for h in raw_headers
        ]
        n_cols = len(parsed_cols)

        # ── Parse data rows into per-column float lists ──
        col_values: List[List[float]] = [[] for _ in range(n_cols)]

        for row_idx, row in enumerate(data_rows):
            if len(row) != n_cols:
                raise BioGearsOutputParseError(
                    f"Row {row_idx + 1} has {len(row)} fields but the header "
                    f"has {n_cols}: {csv_path!r}",
                    csv_path=csv_path,
                    reason=(
                        f"column count mismatch at data row {row_idx + 1}: "
                        f"expected {n_cols}, got {len(row)}"
                    ),
                )
            for col_idx, cell in enumerate(row):
                try:
                    col_values[col_idx].append(float(cell.strip()))
                except ValueError as exc:
                    raise BioGearsOutputParseError(
                        f"Non-numeric value in column "
                        f"{parsed_cols[col_idx].raw_header!r} at data row "
                        f"{row_idx + 1}: {cell!r}: {csv_path!r}",
                        csv_path=csv_path,
                        reason=(
                            f"non-numeric cell at row {row_idx + 1}, "
                            f"col {col_idx} ({parsed_cols[col_idx].raw_header!r}): "
                            f"{cell!r}"
                        ),
                    ) from exc

        time_s = col_values[0]
        row_count = len(time_s)

        # ── Build BioGearsTimeSeries for every column except Time(s) ──
        # Time(s) becomes the shared time axis; all other columns get their
        # own BioGearsTimeSeries. CTSresistance is included verbatim.
        columns: Dict[str, BioGearsTimeSeries] = {}
        for col_idx in range(1, n_cols):
            col_def = parsed_cols[col_idx]
            columns[col_def.raw_header] = BioGearsTimeSeries(
                column=col_def,
                times_s=time_s,
                values=col_values[col_idx],
            )

        final_state: Dict[str, float] = {
            raw_header: ts.values[-1]
            for raw_header, ts in columns.items()
        }

        return BioGearsResult(
            scenario_name=scenario_name,
            time_s=time_s,
            columns=columns,
            final_state=final_state,
            row_count=row_count,
            time_range_s=(time_s[0], time_s[-1]),
            returncode=returncode,
            output_csv_path=csv_path,
        )

    @staticmethod
    def _parse_column_header(raw_header: str) -> BioGearsColumn:
        """Parse a single BioGears CSV column header into a BioGearsColumn.

        Handles both "Name(Unit)" and bare "Name" (no unit) forms.
        The closing ")" is stripped; the opening "(" index is used to split.

        Examples (VERIFIED from actual ClinTwinTinyTest output):
            "HeartRate(1/min)"              → name="HeartRate",                    unit="1/min"
            "MeanArterialPressure(mmHg)"    → name="MeanArterialPressure",         unit="mmHg"
            "RespirationRate(1/min)"        → name="RespirationRate",              unit="1/min"
            "OxygenSaturation"              → name="OxygenSaturation",             unit=""
            "CoreTemperature(degC)"         → name="CoreTemperature",              unit="degC"
            "Time(s)"                       → name="Time",                         unit="s"
            "CTSresistance"                 → name="CTSresistance",                unit=""
            "Lactate-BloodConcentration(mg/dL)" → name="Lactate-BloodConcentration", unit="mg/dL"
        """
        if "(" in raw_header and raw_header.endswith(")"):
            paren_idx = raw_header.index("(")
            name = raw_header[:paren_idx]
            unit = raw_header[paren_idx + 1 : -1]  # strip "(" prefix and ")" suffix
        else:
            name = raw_header
            unit = ""
        return BioGearsColumn(raw_header=raw_header, name=name, unit=unit)

    def _cleanup_outputs(self, *paths: str) -> None:
        """Delete each output file if it exists. Never raises on failure.

        Cleanup failures are logged as warnings only — a cleanup failure must
        not mask a successful simulation result.
        """
        for path in paths:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                    logger.debug("Cleaned up BioGears output file: %s", path)
                except OSError as exc:
                    logger.warning(
                        "Could not remove BioGears output file %s: %s", path, exc
                    )


# ─── Module-level helpers ─────────────────────────────────────────────────────


def _sanitize_name(name: str) -> str:
    """Return a filesystem-safe version of a scenario name.

    Replaces any character that is not alphanumeric, '-', or '_' with '_'.
    Used for building temp filenames and results prefixes.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
