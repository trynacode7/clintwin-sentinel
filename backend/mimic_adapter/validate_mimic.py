"""
validate_mimic.py — Real-data validation layer for MIMIC-IV (Phase 2, section 12).

This layer is what runs ONCE real MIMIC-IV is supplied. It validates:
  - required files/tables are present (loader.py's job, re-exposed here)
  - required columns exist per table (structural, not itemid-dependent)
  - item/concept mappings are verified (concept_map.require_verified)
  - units are recognized before conversion (units.py, fail-loud)
  - timestamps are well-formed and ICU stay boundaries make sense
  - duplicate observations, missingness, impossible physiological values,
    cohort counts, patient-level split integrity (all REUSED from
    backend.synthetic_data.validate, which is explicitly schema-generic —
    see that module's docstring — rather than reimplemented here)
  - temporal consistency / no leakage (the SAME tests in
    tests/test_temporal_leakage.py apply unchanged to MIMIC-adapter output,
    since labels_sofa.py and dataset.py are source-agnostic — this module
    does not duplicate that logic, it documents the reuse)

FAILS LOUDLY. Never silently repairs or drops questionable clinical data —
every check either passes or is reported as an error/warning in the
returned ValidationReport (reused dataclass from backend.synthetic_data.validate).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from backend.synthetic_data.validate import ValidationReport, validate_ml_dataset, validate_patient_split_disjoint
from . import concept_map, cohort as cohort_mod

REQUIRED_RAW_TABLES = [
    "patients", "admissions", "icustays", "chartevents", "labevents",
    "inputevents", "outputevents", "prescriptions", "microbiologyevents",
]
# procedureevents is needed only if the (currently-ambiguous, see
# concept_map.py) ventilated concept resolves to that table — not required
# for the adapter to run, degrades gracefully (ventilated stays 0/unmapped).
OPTIONAL_RAW_TABLES = ["procedureevents", "d_items", "d_labitems"]

# Minimum structural columns each raw table must carry — these are public,
# stable MIMIC-IV table structure facts (column names), not patient data or
# item IDs, and do not require dictionary verification.
REQUIRED_COLUMNS = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime"],
    "icustays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    "chartevents": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"],
    "labevents": ["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"],
    "inputevents": ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate"],
    "outputevents": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"],
    "prescriptions": ["subject_id", "hadm_id", "starttime", "drug"],
    "microbiologyevents": ["subject_id", "hadm_id", "charttime"],
}


def check_required_tables(raw: Dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()
    for name in REQUIRED_RAW_TABLES:
        if name not in raw:
            report.errors.append(f"required MIMIC-IV table '{name}' is missing from the loaded raw tables")
        elif raw[name] is None:
            report.errors.append(f"required MIMIC-IV table '{name}' loaded as None")
    for name in OPTIONAL_RAW_TABLES:
        if name not in raw or raw.get(name) is None:
            report.warnings.append(f"optional MIMIC-IV table '{name}' not provided — features depending on it degrade gracefully")
    return report


def check_required_columns(raw: Dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()
    for table, cols in REQUIRED_COLUMNS.items():
        if table not in raw or raw[table] is None:
            continue  # already reported by check_required_tables
        missing = [c for c in cols if c not in raw[table].columns]
        if missing:
            report.errors.append(f"table '{table}' is missing required column(s): {missing}")
    return report


def check_concept_mapping() -> ValidationReport:
    report = ValidationReport()
    unverified = concept_map.unverified_concepts(required_only=True)
    if unverified:
        report.errors.append(
            f"the following concept mappings are unverified against a real MIMIC-IV "
            f"dictionary and MUST be populated before real-data extraction: {unverified}. "
            f"See backend/mimic_adapter/concept_map.py and docs/MIMIC_IV_SETUP.md."
        )
    report.stats["unverified_concepts"] = unverified
    report.stats["verified_concepts"] = [
        n for n in concept_map.REQUIRED_FOR_V1_SCHEMA if n not in unverified
    ]
    return report


def check_stay_boundaries(icustays: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if icustays.empty:
        report.warnings.append("icustays is empty — nothing to check")
        return report
    missing = icustays["intime"].isna() | icustays["outtime"].isna()
    if missing.any():
        report.errors.append(f"{int(missing.sum())} ICU stay(s) have a missing intime/outtime")
    valid = icustays.dropna(subset=["intime", "outtime"])
    inverted = pd.to_datetime(valid["outtime"]) < pd.to_datetime(valid["intime"])
    if inverted.any():
        report.errors.append(f"{int(inverted.sum())} ICU stay(s) have outtime before intime — impossible stay boundary")
    dup_stay = icustays["stay_id"].duplicated()
    if dup_stay.any():
        report.errors.append(f"{int(dup_stay.sum())} duplicate stay_id value(s) in icustays")
    n_multi_stay_hadm = icustays.groupby("hadm_id")["stay_id"].nunique()
    report.stats["n_admissions_with_multiple_icu_stays"] = int((n_multi_stay_hadm > 1).sum())
    return report


def run_full_validation(
    raw: Dict[str, pd.DataFrame],
    canonical_df: pd.DataFrame = None,
    train_ids=None, val_ids=None, test_ids=None,
) -> ValidationReport:
    """
    Composes every check into one report. `canonical_df` (if supplied) is
    the adapter's output, validated with the SAME schema-generic
    `validate_ml_dataset` the synthetic pipeline uses. `train/val/test_ids`
    (if supplied) get the same `validate_patient_split_disjoint` check the
    synthetic pipeline's split undergoes.
    """
    combined = ValidationReport()

    for sub in [check_required_tables(raw), check_required_columns(raw), check_concept_mapping()]:
        combined.errors += sub.errors
        combined.warnings += sub.warnings
        combined.stats.update(sub.stats)

    if "icustays" in raw and raw["icustays"] is not None:
        sub = check_stay_boundaries(raw["icustays"])
        combined.errors += sub.errors
        combined.warnings += sub.warnings
        combined.stats.update(sub.stats)

        _, cohort_stats = cohort_mod.select_cohort(
            raw["icustays"], raw.get("admissions", pd.DataFrame()), raw.get("patients", pd.DataFrame())
        )
        combined.stats["cohort"] = cohort_stats.as_dict()

    if canonical_df is not None:
        sub = validate_ml_dataset(canonical_df)
        combined.errors += sub.errors
        combined.warnings += sub.warnings
        combined.stats["canonical_schema_validation"] = sub.stats

    if train_ids is not None and val_ids is not None and test_ids is not None:
        sub = validate_patient_split_disjoint(train_ids, val_ids, test_ids)
        combined.errors += sub.errors
        combined.warnings += sub.warnings
        combined.stats["split_validation"] = sub.stats

    combined.stats["note_on_leakage_and_label_compatibility"] = (
        "Temporal leakage (tests/test_temporal_leakage.py) and SOFA label compatibility "
        "(labels_sofa.py) are exercised generically against MIMIC-STRUCTURE TEST FIXTURE "
        "canonical output in tests/test_mimic_adapter.py, since labels_sofa.py/dataset.py "
        "are source-agnostic by design — not duplicated in this module."
    )
    return combined


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MIMIC-IV mapping/validation check (no real data touched by default).")
    p.add_argument("--check-mapping", action="store_true", help="report unverified concept mappings and exit")
    args = p.parse_args()
    if args.check_mapping:
        report = check_concept_mapping()
        print(report)
        raise SystemExit(0 if report.ok else 1)
    print("Run with --check-mapping, or import run_full_validation() against loaded raw MIMIC-IV tables.")
