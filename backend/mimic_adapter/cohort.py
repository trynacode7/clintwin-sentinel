"""
cohort.py — MIMIC-IV cohort extraction (Phase 2, section 8).

MIMIC-IV RAW DATA -> COHORT EXTRACTION -> CANONICAL ICU DATA -> PREPROCESSING -> MODEL DATASET
                      ^^^^^^^^^^^^^^^^^ this module

Cohort selection (which ICU stays are even eligible to enter the pipeline)
is kept STRUCTURALLY SEPARATE from both the canonical-schema construction
(adapter.py) and model preprocessing (backend/preprocessing.py) — this
module only decides "in cohort or not," never touches vitals/labs values,
and never resamples/scales/windows anything.

Criteria are taken directly from the project's own already-agreed cohort
definition (TECHNICAL_BLUEPRINT.md section 2 / ARCHITECTURE_ASSESSMENT.md
section C.2) — nothing invented here:
  - Adults (>= 18) at ICU admission.
  - First ICU stay per hospital admission (avoids the "recovering from a
    prior stay within the same admission" confound).
  - Minimum ICU length of stay so a full 6h input window can exist before
    the first prediction point (TECHNICAL_BLUEPRINT.md recommends >= 8h;
    used as the default here, overridable).

Per-`careunit` missingness auditing (also called for in the blueprint) is
NOT implemented here — it requires real chartevents missingness statistics
this session cannot compute (see the Phase 2 final report's "requires real
MIMIC-IV" list) and is explicitly deferred, not silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

DEFAULT_MIN_LOS_HOURS = 8.0
DEFAULT_MIN_AGE_YEARS = 18.0

INCLUSION_CRITERIA = [
    "age >= 18 years at ICU admission",
    "ICU length of stay >= min_los_hours (default 8h) — enough history for one full input window",
]
EXCLUSION_CRITERIA = [
    "ICU stays that are not the first ICU stay within their hospital admission "
    "(avoids the 'recovering from a prior stay' confound)",
    "stays with a missing/unresolvable age or a missing intime/outtime",
]


@dataclass
class CohortStats:
    n_stays_total: int = 0
    n_admissions_total: int = 0
    n_patients_total: int = 0
    n_excluded_missing_times: int = 0
    n_excluded_not_first_stay: int = 0
    n_excluded_age: int = 0
    n_excluded_short_los: int = 0
    n_stays_final: int = 0
    n_patients_final: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_stays_total": self.n_stays_total,
            "n_admissions_total": self.n_admissions_total,
            "n_patients_total": self.n_patients_total,
            "n_excluded_missing_times": self.n_excluded_missing_times,
            "n_excluded_not_first_stay": self.n_excluded_not_first_stay,
            "n_excluded_age": self.n_excluded_age,
            "n_excluded_short_los": self.n_excluded_short_los,
            "n_stays_final": self.n_stays_final,
            "n_patients_final": self.n_patients_final,
            "inclusion_criteria": INCLUSION_CRITERIA,
            "exclusion_criteria": EXCLUSION_CRITERIA,
            "notes": self.notes,
        }


def compute_age_at_admission(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.Series:
    """
    age = patients.anchor_age + (admissions.admittime.year - patients.anchor_year)

    This is the standard MIMIC-IV de-identified-age computation. REQUIRES
    VERIFICATION against real data for one documented edge case: MIMIC-IV
    codes every patient whose true age at any point exceeded 89 as
    anchor_age=91 — ages computed this way for such patients are a ceiling,
    not an exact value. This function does not attempt to detect or special-
    case that (it would require assumptions about the real anchor_year_group
    field this session cannot verify) — flagged here, not silently handled.
    """
    merged = admissions[["subject_id", "hadm_id", "admittime"]].merge(
        patients[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left"
    )
    admit_year = pd.to_datetime(merged["admittime"]).dt.year
    age = merged["anchor_age"] + (admit_year - merged["anchor_year"])
    age.index = merged["hadm_id"].values
    return age  # indexed by hadm_id


def select_cohort(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    min_los_hours: float = DEFAULT_MIN_LOS_HOURS,
    min_age_years: float = DEFAULT_MIN_AGE_YEARS,
) -> "tuple[list, CohortStats]":
    """
    Returns (eligible_stay_ids, CohortStats). Pure selection — does not
    modify or return any observation data.
    """
    stats = CohortStats()
    df = icustays.copy()
    stats.n_stays_total = len(df)
    stats.n_admissions_total = int(df["hadm_id"].nunique()) if len(df) else 0
    stats.n_patients_total = int(df["subject_id"].nunique()) if len(df) else 0

    if df.empty:
        stats.notes.append("icustays is empty — nothing to select.")
        return [], stats

    # 1. missing intime/outtime
    missing_times = df["intime"].isna() | df["outtime"].isna()
    stats.n_excluded_missing_times = int(missing_times.sum())
    df = df[~missing_times]

    # 2. first ICU stay per hospital admission
    df = df.sort_values(["hadm_id", "intime"])
    is_first = ~df.duplicated(subset=["hadm_id"], keep="first")
    stats.n_excluded_not_first_stay = int((~is_first).sum())
    df = df[is_first]

    # 3. age >= min_age_years
    age_by_hadm = compute_age_at_admission(patients, admissions)
    df_age = df["hadm_id"].map(age_by_hadm)
    age_ok = df_age.notna() & (df_age >= min_age_years)
    stats.n_excluded_age = int((~age_ok).sum())
    df = df[age_ok.values]

    # 4. minimum ICU LOS
    los_hours = (pd.to_datetime(df["outtime"]) - pd.to_datetime(df["intime"])).dt.total_seconds() / 3600.0
    los_ok = los_hours >= min_los_hours
    stats.n_excluded_short_los = int((~los_ok).sum())
    df = df[los_ok.values]

    stats.n_stays_final = len(df)
    stats.n_patients_final = int(df["subject_id"].nunique()) if len(df) else 0
    stats.notes.append(
        "Per-careunit missingness auditing (TECHNICAL_BLUEPRINT.md section 2) is NOT "
        "implemented — requires real chartevents missingness statistics this session "
        "cannot compute. Deferred, not silently skipped."
    )
    return df["stay_id"].tolist(), stats
