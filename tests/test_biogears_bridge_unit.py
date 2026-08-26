"""
test_biogears_bridge_unit.py — Unit tests for BioGearsBridge
=============================================================
SCOPE: Fast, offline tests — no BioGears installation required.
subprocess is mocked throughout so these pass on any developer machine.

Run with:
    pytest tests/test_biogears_bridge_unit.py -v
"""

import os
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.biogears_bridge import (
    BioGearsConfig,
    BioGearsBridge,
    BioGearsColumn,
    BioGearsBridgeError,
    BioGearsNotFoundError,
    BioGearsOutputNotFoundError,
    BioGearsOutputParseError,
    BioGearsProcessError,
    BioGearsResult,
    BioGearsTimeSeries,
    BioGearsTimeoutError,
    _sanitize_name,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_csv_file(directory: str, content: str, filename: str = "out.csv") -> str:
    """Write content to a file inside directory and return the absolute path."""
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(content)
    return path


def _make_bridge(tmp_dir: str, exe: str = r"C:\BioGears\bin\bg-scenario.exe") -> BioGearsBridge:
    cfg = BioGearsConfig(executable=exe, timeout_seconds=60, tmp_dir=tmp_dir)
    return BioGearsBridge(config=cfg)


# ─── _sanitize_name ───────────────────────────────────────────────────────────


class TestSanitizeName:
    def test_alphanumeric_unchanged(self):
        assert _sanitize_name("abc123") == "abc123"

    def test_dash_and_underscore_preserved(self):
        assert _sanitize_name("my-scenario_1") == "my-scenario_1"

    def test_spaces_replaced(self):
        assert _sanitize_name("Hello World") == "Hello_World"

    def test_special_chars_replaced(self):
        assert _sanitize_name("a/b\\c:d") == "a_b_c_d"

    def test_empty_string(self):
        assert _sanitize_name("") == ""


# ─── BioGearsColumn / _parse_column_header ────────────────────────────────────


class TestParseColumnHeader:
    def _parse(self, header: str) -> BioGearsColumn:
        return BioGearsBridge._parse_column_header(header)

    def test_name_unit_form(self):
        col = self._parse("HeartRate(1/min)")
        assert col.raw_header == "HeartRate(1/min)"
        assert col.name == "HeartRate"
        assert col.unit == "1/min"

    def test_mean_arterial_pressure(self):
        col = self._parse("MeanArterialPressure(mmHg)")
        assert col.name == "MeanArterialPressure"
        assert col.unit == "mmHg"

    def test_bare_name_no_unit(self):
        col = self._parse("OxygenSaturation")
        assert col.raw_header == "OxygenSaturation"
        assert col.name == "OxygenSaturation"
        assert col.unit == ""

    def test_ctsresistance_bare(self):
        col = self._parse("CTSresistance")
        assert col.name == "CTSresistance"
        assert col.unit == ""

    def test_time_column(self):
        col = self._parse("Time(s)")
        assert col.name == "Time"
        assert col.unit == "s"

    def test_complex_unit_with_slash(self):
        col = self._parse("RespirationRate(1/min)")
        assert col.name == "RespirationRate"
        assert col.unit == "1/min"

    def test_unit_with_slash_in_name(self):
        col = self._parse("Lactate-BloodConcentration(mg/dL)")
        assert col.name == "Lactate-BloodConcentration"
        assert col.unit == "mg/dL"

    def test_paren_not_at_end_treated_as_bare(self):
        col = self._parse("Weird(Name")
        assert col.name == "Weird(Name"
        assert col.unit == ""

    def test_raw_header_always_preserved(self):
        raw = "CoreTemperature(degC)"
        col = self._parse(raw)
        assert col.raw_header == raw


# ─── BioGearsConfig defaults ──────────────────────────────────────────────────


class TestBioGearsConfigDefaults:
    def test_default_executable(self):
        cfg = BioGearsConfig()
        assert cfg.executable == r"C:\BioGears\bin\bg-scenario.exe"

    def test_default_timeout(self):
        cfg = BioGearsConfig()
        assert cfg.timeout_seconds == 300

    def test_default_tmp_dir(self):
        cfg = BioGearsConfig()
        assert cfg.tmp_dir == r"C:\BioGears\bin\Scenarios\tmp"

    def test_custom_values(self):
        cfg = BioGearsConfig(executable="x.exe", timeout_seconds=42, tmp_dir="C:\\tmp")
        assert cfg.executable == "x.exe"
        assert cfg.timeout_seconds == 42
        assert cfg.tmp_dir == "C:\\tmp"


# ─── BioGearsBridge.config property ──────────────────────────────────────────


class TestBridgeConfigProperty:
    def test_config_round_trips(self, tmp_path):
        cfg = BioGearsConfig(executable="x.exe", timeout_seconds=99, tmp_dir=str(tmp_path))
        bridge = BioGearsBridge(config=cfg)
        assert bridge.config is cfg

    def test_default_config_used_when_none(self):
        bridge = BioGearsBridge()
        assert isinstance(bridge.config, BioGearsConfig)
        assert bridge.config.executable == r"C:\BioGears\bin\bg-scenario.exe"


# ─── _check_executable ────────────────────────────────────────────────────────


class TestCheckExecutable:
    def test_raises_not_found_when_exe_missing(self, tmp_path):
        bridge = _make_bridge(str(tmp_path), exe=str(tmp_path / "nonexistent.exe"))
        with pytest.raises(BioGearsNotFoundError) as exc_info:
            bridge._check_executable()
        assert "nonexistent.exe" in str(exc_info.value)

    def test_no_raise_when_exe_present(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        bridge._check_executable()  # should not raise


# ─── _ensure_tmp_dir ──────────────────────────────────────────────────────────


class TestEnsureTmpDir:
    def test_creates_directory_if_absent(self, tmp_path):
        new_dir = str(tmp_path / "deep" / "nested")
        cfg = BioGearsConfig(tmp_dir=new_dir)
        bridge = BioGearsBridge(config=cfg)
        result = bridge._ensure_tmp_dir()
        assert os.path.isdir(new_dir)
        assert result == new_dir

    def test_idempotent_when_already_exists(self, tmp_path):
        bridge = _make_bridge(str(tmp_path))
        bridge._ensure_tmp_dir()
        bridge._ensure_tmp_dir()  # second call must not raise
        assert os.path.isdir(str(tmp_path))


# ─── _build_results_prefix ────────────────────────────────────────────────────


class TestBuildResultsPrefix:
    def test_prefix_is_inside_tmp_dir(self, tmp_path):
        bridge = _make_bridge(str(tmp_path))
        prefix = bridge._build_results_prefix(str(tmp_path), "MyScenario")
        assert prefix.startswith(str(tmp_path))

    def test_prefix_contains_sanitized_name(self, tmp_path):
        bridge = _make_bridge(str(tmp_path))
        prefix = bridge._build_results_prefix(str(tmp_path), "My Scenario!")
        assert "My_Scenario_" in prefix

    def test_two_calls_produce_different_prefixes(self, tmp_path):
        bridge = _make_bridge(str(tmp_path))
        a = bridge._build_results_prefix(str(tmp_path), "S")
        b = bridge._build_results_prefix(str(tmp_path), "S")
        assert a != b


# ─── _cleanup_outputs ─────────────────────────────────────────────────────────


class TestCleanupOutputs:
    def test_deletes_existing_files(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.log"
        f1.write_text("x")
        f2.write_text("y")
        bridge = _make_bridge(str(tmp_path))
        bridge._cleanup_outputs(str(f1), str(f2))
        assert not f1.exists()
        assert not f2.exists()

    def test_silently_skips_missing_files(self, tmp_path):
        bridge = _make_bridge(str(tmp_path))
        bridge._cleanup_outputs(str(tmp_path / "ghost.csv"))  # must not raise

    def test_handles_empty_path_string(self, tmp_path):
        bridge = _make_bridge(str(tmp_path))
        bridge._cleanup_outputs("")  # must not raise

    def test_cleanup_failure_does_not_raise(self, tmp_path, monkeypatch):
        f = tmp_path / "locked.csv"
        f.write_text("x")
        bridge = _make_bridge(str(tmp_path))
        original_remove = os.remove

        def bad_remove(path):
            if str(path).endswith("locked.csv"):
                raise OSError("locked")
            original_remove(path)

        monkeypatch.setattr(os, "remove", bad_remove)
        bridge._cleanup_outputs(str(f))  # must not raise


# ─── _find_output_csv ─────────────────────────────────────────────────────────


class TestFindOutputCsv:
    def test_returns_path_when_csv_exists(self, tmp_path):
        prefix = str(tmp_path / "run1")
        csv_path = prefix + "Results.csv"
        open(csv_path, "w").close()
        bridge = _make_bridge(str(tmp_path))
        assert bridge._find_output_csv(prefix) == csv_path

    def test_raises_output_not_found_when_absent(self, tmp_path):
        prefix = str(tmp_path / "ghost")
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputNotFoundError) as exc_info:
            bridge._find_output_csv(prefix)
        assert exc_info.value.expected_path == prefix + "Results.csv"


# ─── _parse_csv ───────────────────────────────────────────────────────────────


GOOD_CSV = textwrap.dedent("""\
    Time(s),CTSresistance,HeartRate(1/min),OxygenSaturation
    5.02,0.0,72.5,0.98
    10.02,0.0,74.1,0.97
    15.02,0.0,75.0,0.96
""")


class TestParseCsvGoodPath:
    def test_row_count(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        assert result.row_count == 3

    def test_time_series(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        assert result.time_s == [5.02, 10.02, 15.02]

    def test_time_range(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        assert result.time_range_s == (5.02, 15.02)

    def test_columns_keyed_by_raw_header(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        assert set(result.columns.keys()) == {
            "CTSresistance",
            "HeartRate(1/min)",
            "OxygenSaturation",
        }

    def test_time_column_excluded_from_columns(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        assert "Time(s)" not in result.columns

    def test_final_state_last_row_values(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        assert result.final_state["HeartRate(1/min)"] == pytest.approx(75.0)
        assert result.final_state["OxygenSaturation"] == pytest.approx(0.96)
        assert result.final_state["CTSresistance"] == pytest.approx(0.0)

    def test_bare_column_name_has_empty_unit(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        col = result.columns["OxygenSaturation"].column
        assert col.unit == ""
        assert col.name == "OxygenSaturation"

    def test_ctsresistance_included_verbatim(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "Test", returncode=0)
        assert "CTSresistance" in result.columns
        assert result.columns["CTSresistance"].values == [0.0, 0.0, 0.0]

    def test_scenario_name_preserved(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "MySepsis", returncode=0)
        assert result.scenario_name == "MySepsis"

    def test_returncode_preserved(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "T", returncode=0)
        assert result.returncode == 0

    def test_timeseries_values_parallel_to_time(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), GOOD_CSV)
        bridge = _make_bridge(str(tmp_path))
        result = bridge._parse_csv(csv_file, "T", returncode=0)
        ts = result.columns["HeartRate(1/min)"]
        assert ts.times_s == result.time_s
        assert ts.values == [72.5, 74.1, 75.0]


class TestParseCsvErrorCases:
    def test_empty_file_raises(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), "")
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputParseError) as exc_info:
            bridge._parse_csv(csv_file, "T", returncode=0)
        assert "empty" in exc_info.value.reason

    def test_header_only_no_data_rows_raises(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), "Time(s),HeartRate(1/min)\n")
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputParseError) as exc_info:
            bridge._parse_csv(csv_file, "T", returncode=0)
        assert "zero data rows" in exc_info.value.reason

    def test_wrong_first_column_raises(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), "Seconds,HeartRate\n1.0,72.0\n")
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputParseError) as exc_info:
            bridge._parse_csv(csv_file, "T", returncode=0)
        assert "Seconds" in exc_info.value.reason

    def test_column_count_mismatch_raises(self, tmp_path):
        content = "Time(s),HeartRate(1/min)\n5.0,72.0,extra\n"
        csv_file = _make_csv_file(str(tmp_path), content)
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputParseError) as exc_info:
            bridge._parse_csv(csv_file, "T", returncode=0)
        assert "column count mismatch" in exc_info.value.reason

    def test_non_numeric_cell_raises(self, tmp_path):
        content = "Time(s),HeartRate(1/min)\n5.0,NaN_text\n"
        csv_file = _make_csv_file(str(tmp_path), content)
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputParseError) as exc_info:
            bridge._parse_csv(csv_file, "T", returncode=0)
        assert "non-numeric" in exc_info.value.reason

    def test_non_numeric_in_time_column_raises(self, tmp_path):
        content = "Time(s),HeartRate(1/min)\nbad,72.0\n"
        csv_file = _make_csv_file(str(tmp_path), content)
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputParseError) as exc_info:
            bridge._parse_csv(csv_file, "T", returncode=0)
        assert "non-numeric" in exc_info.value.reason

    def test_csv_path_stored_in_exception(self, tmp_path):
        csv_file = _make_csv_file(str(tmp_path), "")
        bridge = _make_bridge(str(tmp_path))
        with pytest.raises(BioGearsOutputParseError) as exc_info:
            bridge._parse_csv(csv_file, "T", returncode=0)
        assert exc_info.value.csv_path == csv_file


# ─── _invoke_biogears (mocked subprocess) ─────────────────────────────────────


class TestInvokeBioGears:
    def _make_proc(self, returncode=0, stdout="", stderr=""):
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = stderr
        return proc

    def test_successful_run_returns_tuple(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        with patch("subprocess.run", return_value=self._make_proc(returncode=0, stdout="ok")):
            rc, stdout, stderr = bridge._invoke_biogears("sc.xml", "prefix")
        assert rc == 0
        assert stdout == "ok"

    def test_non_zero_exit_raises_process_error(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        with patch("subprocess.run", return_value=self._make_proc(returncode=1, stdout="fail")):
            with pytest.raises(BioGearsProcessError) as exc_info:
                bridge._invoke_biogears("sc.xml", "prefix")
        assert exc_info.value.returncode == 1
        assert "fail" in exc_info.value.stdout

    def test_timeout_raises_timeout_error(self, tmp_path):
        import subprocess as sp

        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="x", timeout=60)):
            with pytest.raises(BioGearsTimeoutError) as exc_info:
                bridge._invoke_biogears("sc.xml", "prefix")
        assert exc_info.value.timeout_seconds == 60

    def test_command_uses_absolute_paths(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        with patch("subprocess.run", return_value=self._make_proc()) as mock_run:
            bridge._invoke_biogears("relative.xml", "prefix")
        cmd = mock_run.call_args[0][0]
        assert os.path.isabs(cmd[2])  # abs_xml
        assert os.path.isabs(cmd[4])  # abs_prefix

    def test_command_structure(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        with patch("subprocess.run", return_value=self._make_proc()) as mock_run:
            bridge._invoke_biogears("sc.xml", "prefix")
        cmd = mock_run.call_args[0][0]
        # [exe, "scenario", xml, "results", prefix]
        assert cmd[1] == "scenario"
        assert cmd[3] == "results"


# ─── Typed exception attributes ───────────────────────────────────────────────


class TestTypedExceptions:
    def test_process_error_attributes(self):
        exc = BioGearsProcessError("msg", returncode=2, stdout="out", stderr="err")
        assert exc.returncode == 2
        assert exc.stdout == "out"
        assert exc.stderr == "err"
        assert str(exc) == "msg"

    def test_process_error_defaults(self):
        exc = BioGearsProcessError("oops", returncode=1)
        assert exc.stdout == ""
        assert exc.stderr == ""

    def test_timeout_error_attribute(self):
        exc = BioGearsTimeoutError("timed out", timeout_seconds=120)
        assert exc.timeout_seconds == 120

    def test_output_not_found_attribute(self):
        exc = BioGearsOutputNotFoundError("missing", expected_path="C:\\foo.csv")
        assert exc.expected_path == "C:\\foo.csv"

    def test_parse_error_attributes(self):
        exc = BioGearsOutputParseError("bad", csv_path="C:\\x.csv", reason="empty")
        assert exc.csv_path == "C:\\x.csv"
        assert exc.reason == "empty"

    def test_all_exceptions_inherit_from_base(self):
        assert issubclass(BioGearsNotFoundError, BioGearsBridgeError)
        assert issubclass(BioGearsProcessError, BioGearsBridgeError)
        assert issubclass(BioGearsTimeoutError, BioGearsBridgeError)
        assert issubclass(BioGearsOutputNotFoundError, BioGearsBridgeError)
        assert issubclass(BioGearsOutputParseError, BioGearsBridgeError)


# ─── run_scenario_file — mocked end-to-end ────────────────────────────────────


class TestRunScenarioFileMocked:
    """Exercise run_scenario_file() with subprocess mocked and a real temp CSV."""

    def _make_fake_invoke(self, tmp_dir, csv_content=None):
        """Return a fake _invoke_biogears that writes a CSV next to the prefix."""
        content = csv_content if csv_content is not None else GOOD_CSV

        def fake_invoke(scenario_xml_path, results_prefix):
            csv_path = results_prefix + "Results.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(content)
            return 0, "", ""

        return fake_invoke

    def test_returns_biogears_result(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        bridge._invoke_biogears = self._make_fake_invoke(str(tmp_path))
        result = bridge.run_scenario_file("dummy.xml", scenario_name="UnitTest")
        assert isinstance(result, BioGearsResult)
        assert result.row_count == 3

    def test_cleanup_called_on_success(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        bridge._invoke_biogears = self._make_fake_invoke(str(tmp_path))

        cleaned = []
        original_cleanup = bridge._cleanup_outputs

        def tracking_cleanup(*paths):
            cleaned.extend(paths)
            original_cleanup(*paths)

        bridge._cleanup_outputs = tracking_cleanup
        bridge.run_scenario_file("dummy.xml")
        assert len(cleaned) == 2  # csv + log

    def test_keep_output_skips_cleanup(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        bridge._invoke_biogears = self._make_fake_invoke(str(tmp_path))

        cleaned = []
        bridge._cleanup_outputs = lambda *paths: cleaned.extend(paths)
        bridge.run_scenario_file("dummy.xml", keep_output=True)
        assert cleaned == []

    def test_raises_not_found_before_invoke(self, tmp_path):
        cfg = BioGearsConfig(
            executable=str(tmp_path / "missing.exe"),
            tmp_dir=str(tmp_path),
        )
        bridge = BioGearsBridge(config=cfg)
        with pytest.raises(BioGearsNotFoundError):
            bridge.run_scenario_file("dummy.xml")

    def test_cleanup_called_even_when_parse_fails(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        # Fake invoke writes an empty (malformed) CSV
        bridge._invoke_biogears = self._make_fake_invoke(str(tmp_path), csv_content="")

        cleaned = []
        original_cleanup = bridge._cleanup_outputs

        def tracking_cleanup(*paths):
            cleaned.extend(paths)
            original_cleanup(*paths)

        bridge._cleanup_outputs = tracking_cleanup

        with pytest.raises(BioGearsOutputParseError):
            bridge.run_scenario_file("dummy.xml")

        assert len(cleaned) == 2  # cleanup still runs in finally


# ─── run_scenario_xml — mocked ────────────────────────────────────────────────


class TestRunScenarioXmlMocked:
    """run_scenario_xml() writes a temp XML and delegates to run_scenario_file."""

    def test_delegates_to_run_scenario_file(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        dummy_result = MagicMock(spec=BioGearsResult)
        with patch.object(bridge, "run_scenario_file", return_value=dummy_result) as mock_rsf:
            result = bridge.run_scenario_xml("<xml/>", scenario_name="Test")
        assert result is dummy_result
        mock_rsf.assert_called_once()

    def test_temp_xml_cleaned_up_on_success(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        created_paths = []

        def capturing_rsf(path, **kwargs):
            created_paths.append(path)
            return MagicMock(spec=BioGearsResult)

        with patch.object(bridge, "run_scenario_file", side_effect=capturing_rsf):
            bridge.run_scenario_xml("<xml/>")

        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

    def test_temp_xml_cleaned_up_on_failure(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        created_paths = []

        def capturing_rsf(path, **kwargs):
            created_paths.append(path)
            raise BioGearsProcessError("fail", returncode=1)

        with patch.object(bridge, "run_scenario_file", side_effect=capturing_rsf):
            with pytest.raises(BioGearsProcessError):
                bridge.run_scenario_xml("<xml/>")

        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

    def test_keep_output_passed_through(self, tmp_path):
        fake_exe = tmp_path / "bg-scenario.exe"
        fake_exe.write_bytes(b"")
        bridge = _make_bridge(str(tmp_path), exe=str(fake_exe))
        dummy_result = MagicMock(spec=BioGearsResult)
        with patch.object(bridge, "run_scenario_file", return_value=dummy_result) as mock_rsf:
            bridge.run_scenario_xml("<xml/>", keep_output=True)
        _, kwargs = mock_rsf.call_args
        assert kwargs.get("keep_output") is True
