"""
test_validate_mimic.py — Phase 2 section 12: exercises the real-data
validation layer (backend/mimic_adapter/validate_mimic.py) against the SAME
MIMIC-STRUCTURE TEST FIXTURES as test_mimic_adapter.py (imported, not
rebuilt) since real MIMIC-IV is unavailable in this environment.
"""
import pandas as pd

from backend.mimic_adapter import validate_mimic
from tests.test_mimic_adapter import build_mimic_structure_fixture, _fixture_concepts
from backend.mimic_adapter import adapter


def test_required_tables_check_passes_on_complete_fixture():
    raw = build_mimic_structure_fixture()
    report = validate_mimic.check_required_tables(raw)
    assert report.ok, str(report)


def test_required_tables_check_fails_when_a_required_table_missing():
    raw = build_mimic_structure_fixture()
    del raw["labevents"]
    report = validate_mimic.check_required_tables(raw)
    assert not report.ok
    assert any("labevents" in e for e in report.errors)


def test_required_columns_check_fails_when_a_column_missing():
    raw = build_mimic_structure_fixture()
    raw["chartevents"] = raw["chartevents"].drop(columns=["valueuom"])
    report = validate_mimic.check_required_columns(raw)
    assert not report.ok
    assert any("chartevents" in e and "valueuom" in e for e in report.errors)


def test_concept_mapping_check_fails_by_default_shipped_state():
    """The shipped concept_map.py intentionally ships unverified (empty
    itemids) — this must be caught as a real-data-blocking error."""
    report = validate_mimic.check_concept_mapping()
    assert not report.ok
    assert len(report.stats["unverified_concepts"]) > 0


def test_stay_boundaries_check_flags_inverted_times():
    raw = build_mimic_structure_fixture()
    bad = raw["icustays"].copy()
    bad.loc[0, ["intime", "outtime"]] = [bad.loc[0, "outtime"], bad.loc[0, "intime"]]
    report = validate_mimic.check_stay_boundaries(bad)
    assert not report.ok
    assert any("before intime" in e for e in report.errors)


def test_stay_boundaries_check_counts_multi_stay_admissions():
    raw = build_mimic_structure_fixture()
    report = validate_mimic.check_stay_boundaries(raw["icustays"])
    assert report.stats["n_admissions_with_multiple_icu_stays"] == 1


def test_full_validation_reports_cohort_and_canonical_stats():
    raw = build_mimic_structure_fixture()
    canonical = adapter.adapt(
        raw, concepts=_fixture_concepts(), apply_cohort_filter=False,
        skip_verification_check=True, antibiotic_drug_names={"Vancomycin"},
    )
    report = validate_mimic.run_full_validation(raw, canonical_df=canonical)
    assert "cohort" in report.stats
    assert "canonical_schema_validation" in report.stats
    # concept-mapping error is still present (shipped map is unverified) —
    # run_full_validation must not hide that even though the fixture-local
    # adapt() call above bypassed it deliberately.
    assert not report.ok
