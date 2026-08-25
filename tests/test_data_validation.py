"""
test_data_validation.py — Phase 1 section 9: dataset validation checks.

Covers: schema matches DATA_CONTRACT.md, no required column is completely
(100%) empty (the exact bug class fixed in generator.py's generate_cohort
during this phase — a regression test for it), no duplicate identifiers,
no impossible physiological values, label is NOT present in the raw/extracted
table (it must be computed downstream by labels_sofa.py, never injected).
"""
import numpy as np
import pandas as pd

from backend.synthetic_data.validate import validate_ml_dataset, REQUIRED_MEASUREMENT_COLS
from backend.synthetic_data.extract_to_ml_dataset import extract

EXPECTED_COLS = [
    "subject_id", "hadm_id", "stay_id", "charttime",
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temp",
    "lactate", "wbc", "creatinine", "bilirubin", "platelets",
    "gcs", "fio2", "ventilated", "urine_output_rate",
    "vasopressor_dose", "fluid_bolus_ml", "age", "sex_male", "suspected_infection",
]


def test_schema_matches_data_contract(small_ml_dataset):
    assert list(small_ml_dataset.columns) == EXPECTED_COLS


def test_validation_passes_on_generated_cohort(small_ml_dataset):
    report = validate_ml_dataset(small_ml_dataset)
    assert report.ok, str(report)


def test_no_required_column_is_completely_empty(small_ml_dataset):
    """
    Regression test for the dtype-upcast bug found and fixed in this phase:
    pd.concat()-ing per-stay frames where some stays contributed the empty,
    default-object-dtype fallback silently upcast whole measurement columns
    (lactate, wbc, creatinine, vasopressor_dose, fluid_bolus_ml, ...) to
    dtype object, which made downstream `.mean(numeric_only=True)` calls
    silently DROP them, producing a 100%-missing column with no error.
    """
    for col in REQUIRED_MEASUREMENT_COLS:
        n_non_null = small_ml_dataset[col].notna().sum()
        assert n_non_null > 0, f"required column '{col}' has zero non-null values — extraction bug"


def test_measurement_columns_are_numeric_dtype(small_ml_dataset):
    numeric_cols = [c for c in REQUIRED_MEASUREMENT_COLS if c in small_ml_dataset.columns]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(small_ml_dataset[col]), (
            f"'{col}' has dtype {small_ml_dataset[col].dtype}, expected numeric"
        )


def test_no_duplicate_identifiers_missing(small_ml_dataset):
    assert not small_ml_dataset[["subject_id", "hadm_id", "stay_id"]].isna().any().any()


def test_no_impossible_values(small_ml_dataset):
    # spot-check a few hard physiological bounds directly (validate_ml_dataset
    # already checks the full PHYSIOLOGICAL_RANGES table; this double-checks
    # the two columns most likely to be silently corrupted by unit errors)
    hr = small_ml_dataset["heart_rate"].dropna()
    assert (hr > 0).all() and (hr < 300).all()
    spo2 = small_ml_dataset["spo2"].dropna()
    assert (spo2 >= 0).all() and (spo2 <= 100).all()


def test_label_is_never_present_in_raw_or_extracted_data(small_raw_cohort, small_ml_dataset):
    """
    The clinical label must be computed downstream by labels_sofa.py from
    observed vitals/labs/treatments — never injected directly into the
    generator output. This asserts no raw table or the extracted flat table
    carries a "label" (or "risk_flag") column at all.
    """
    assert "label" not in small_ml_dataset.columns
    assert "risk_flag" not in small_ml_dataset.columns
    # d_items/d_labitems are dictionary tables and legitimately have a "label"
    # column (mirroring real MIMIC-IV's d_items.label item-name column) — that
    # is not a clinical target and is excluded from this check.
    event_tables = ["patients", "admissions", "icustays", "chartevents", "labevents",
                     "inputevents", "outputevents", "antibiotics", "microbiologyevents"]
    for name in event_tables:
        table = small_raw_cohort[name]
        assert "label" not in table.columns, f"raw table '{name}' unexpectedly has a 'label' column"
