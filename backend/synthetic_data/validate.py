"""
validate.py — Data validation checks (Phase 1, section 9) and dataset summary
reporting for the synthetic ICU development data layer. These checks are
schema-generic where possible so they also apply once real MIMIC-IV-derived
data flows through the same `ml_dataset.csv` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PHYSIOLOGICAL_RANGES = {
    "heart_rate": (10, 300), "sbp": (20, 300), "dbp": (5, 220), "map": (10, 250),
    "resp_rate": (0, 80), "spo2": (0, 100), "temp": (25, 45),
    "lactate": (0, 30), "wbc": (0, 100), "creatinine": (0, 25),
    "bilirubin": (0, 50), "platelets": (0, 1200), "gcs": (3, 15),
    "fio2": (0.21, 1.0), "urine_output_rate": (0, 2000), "age": (18, 110),
}

ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Columns DATA_CONTRACT.md section 3 marks as having a clear architectural role
# (model input or label-construction input). sbp/dbp are explicitly optional/
# unused-in-v1 per DATA_CONTRACT.md and are excluded from this check.
REQUIRED_MEASUREMENT_COLS = [
    "heart_rate", "map", "resp_rate", "spo2", "temp",
    "lactate", "wbc", "creatinine", "bilirubin", "platelets",
    "gcs", "fio2", "ventilated", "urine_output_rate",
    "vasopressor_dose", "fluid_bolus_ml",
]


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        lines = []
        lines.append(f"ValidationReport: {'PASS' if self.ok else 'FAIL'}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            lines += [f"    - {e}" for e in self.errors]
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_ml_dataset(df: pd.DataFrame, time_col: str = "charttime") -> ValidationReport:
    report = ValidationReport()

    for col in ID_COLS + [time_col]:
        if col not in df.columns:
            report.errors.append(f"missing required identifier/time column: {col}")
    if not report.errors:
        n_missing_ids = df[ID_COLS].isna().any(axis=1).sum()
        if n_missing_ids:
            report.errors.append(f"{n_missing_ids} rows have a missing identifier (subject_id/hadm_id/stay_id)")

    # duplicate timestamps per stay
    dup_mask = df.duplicated(subset=["stay_id", time_col], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup:
        report.warnings.append(f"{n_dup} rows share a duplicate (stay_id, {time_col}) pair "
                                f"(expected in small numbers — rapid-repeat measurements collapse on resample)")

    # impossible / out-of-range values
    range_violations = {}
    for col, (lo, hi) in PHYSIOLOGICAL_RANGES.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
            if bad.any():
                range_violations[col] = int(bad.sum())
    if range_violations:
        report.errors.append(f"physiologically impossible values found: {range_violations}")

    # timestamp ordering per stay (not required to be pre-sorted, but must be sortable/consistent)
    if df[time_col].isna().any():
        report.errors.append(f"{int(df[time_col].isna().sum())} rows have a null {time_col}")

    # malformed records: every measurement column null
    measurement_cols = [c for c in df.columns if c not in ID_COLS + [time_col, "age", "sex_male", "suspected_infection"]]
    if measurement_cols:
        all_null = df[measurement_cols].isna().all(axis=1)
        if all_null.any():
            report.warnings.append(f"{int(all_null.sum())} rows have every measurement column null "
                                    f"(candidates for the drop_all_vitals_missing step downstream)")

    # missingness by column
    missingness = {c: round(float(df[c].isna().mean() * 100), 2) for c in measurement_cols}
    report.stats["missingness_pct"] = missingness

    # a required column that is 100% missing across the ENTIRE dataset is not
    # realistic clinical sparsity — every one of these variables is charted at
    # least occasionally in DATA_CONTRACT.md's design (sparsity varies by
    # variable, but never total). Total absence almost certainly means an
    # extraction/generation bug (e.g. a dtype-upcast that silently drops a
    # column during aggregation) rather than genuine missingness, so this is
    # an error, not an informational missingness-pct entry.
    fully_missing = [c for c in REQUIRED_MEASUREMENT_COLS
                     if c in df.columns and missingness.get(c) == 100.0]
    if fully_missing:
        report.errors.append(
            f"required column(s) 100% missing across entire dataset (likely a bug, "
            f"not realistic sparsity): {fully_missing}"
        )

    # extreme outliers (informational, z-score > 6 within-column)
    outlier_counts = {}
    for col in measurement_cols:
        if col in PHYSIOLOGICAL_RANGES and df[col].notna().sum() > 10:
            vals = df[col].dropna()
            z = (vals - vals.mean()) / (vals.std() + 1e-9)
            n_extreme = int((z.abs() > 6).sum())
            if n_extreme:
                outlier_counts[col] = n_extreme
    report.stats["extreme_outlier_counts"] = outlier_counts

    report.stats["n_rows"] = len(df)
    report.stats["n_patients"] = int(df["subject_id"].nunique())
    report.stats["n_stays"] = int(df["stay_id"].nunique())

    return report


def validate_patient_split_disjoint(train_ids, val_ids, test_ids) -> ValidationReport:
    report = ValidationReport()
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    if tv:
        report.errors.append(f"train ∩ val is non-empty ({len(tv)} shared subject_ids)")
    if tt:
        report.errors.append(f"train ∩ test is non-empty ({len(tt)} shared subject_ids)")
    if vt:
        report.errors.append(f"val ∩ test is non-empty ({len(vt)} shared subject_ids)")
    report.stats.update({
        "train_patients": len(train_ids), "val_patients": len(val_ids), "test_patients": len(test_ids),
        "train_val_intersection": len(tv), "train_test_intersection": len(tt), "val_test_intersection": len(vt),
    })
    return report


def summarize_labels(labels_df: pd.DataFrame, label_col: str = "label") -> Dict[str, object]:
    n_pos = int((labels_df[label_col] == 1).sum())
    n_neg = int((labels_df[label_col] == 0).sum())
    return {
        "n_labeled_windows": n_pos + n_neg,
        "positive": n_pos,
        "negative": n_neg,
        "positive_rate": round(n_pos / max(n_pos + n_neg, 1), 4),
    }


def dataset_summary(df: pd.DataFrame, icustays: Optional[pd.DataFrame] = None,
                     labels_df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "n_patients": int(df["subject_id"].nunique()),
        "n_stays": int(df["stay_id"].nunique()),
        "n_observations": len(df),
    }
    if icustays is not None and len(icustays):
        hours = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600.0
        summary["total_icu_hours"] = round(float(hours.sum()), 1)
        summary["mean_stay_hours"] = round(float(hours.mean()), 1)
        summary["archetype_counts"] = icustays["archetype"].value_counts().to_dict()
    measurement_cols = [c for c in df.columns if c not in ID_COLS + ["charttime", "age", "sex_male", "suspected_infection"]]
    summary["missingness_pct"] = {c: round(float(df[c].isna().mean() * 100), 2) for c in measurement_cols}
    summary["suspected_infection_positive_stays"] = int(
        df.groupby("stay_id")["suspected_infection"].max().sum()
    )
    if labels_df is not None and len(labels_df):
        summary["labels"] = summarize_labels(labels_df)
    return summary
