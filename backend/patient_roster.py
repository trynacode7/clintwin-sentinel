"""
patient_roster.py — Canonical Patient Roster (Phase B)
=========================================================
Defines the SINGLE canonical list of patients ("the roster") that the
backend API serves and that the frontend must consume (Phase K) instead
of generating its own patient list.

WHY THIS EXISTS
----------------
Before Phase B, `frontend/src/hooks/usePatientData.js` defined its own
hardcoded 9-entry `PATIENT_REGISTRY` (ids like "patient_003") with
invented `baseRisk`/`archetype`/`trend` fields, entirely disconnected
from any data the backend held. Meanwhile the backend's own `/patients`
and `/patient/{id}` endpoints (api.py `_generate_demo_data()`) ALSO
invented a separate, different, random 10-patient list on every process
start. Two independent fabricated registries, neither derived from real
data, neither matching the other.

This module is the fix: ONE roster, derived from the real synthetic
ICU stays (`stay_id` / `subject_id` / `hadm_id`) in the model's
held-out test split — the same rows `inference_service.py` uses to run
real model inference. Both the backend API and (from Phase K) the
frontend read this same roster; nothing downstream invents patient IDs.

DESIGN NOTE — no duplicated pipeline logic
--------------------------------------------
This module does NOT call `preprocessing.preprocess()` (or run any part
of the feature pipeline) itself. It is a pure transform over a
DataFrame that `inference_service.py` has already produced via that
single authoritative pipeline entrypoint. Building the roster here is
just "which stay_ids exist in this dataframe, and what's their
identity/demographic info" — never a second, independent data-loading
or feature-generation path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("icu_digital_twin")


@dataclass
class RosterPatient:
    """
    One canonical roster entry. Every field is derived from real
    synthetic ml_dataset.csv rows for this stay_id — nothing here is
    invented. `latest_risk_score` / `status` start as None and are
    filled in by inference_service.py once real predictions have been
    computed (never fabricated, never defaulted to a random value).
    """

    stay_id: int
    subject_id: int
    hadm_id: int
    age_years: Optional[float]
    sex_male: Optional[bool]
    n_observed_timesteps: int
    first_charttime: str
    last_charttime: str
    split: str = "test"
    latest_risk_score: Optional[float] = None
    latest_status: Optional[str] = None
    n_windows: int = 0

    def to_dict(self) -> dict:
        return {
            "stay_id": self.stay_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "age_years": self.age_years,
            "sex_male": self.sex_male,
            "n_observed_timesteps": self.n_observed_timesteps,
            "first_charttime": self.first_charttime,
            "last_charttime": self.last_charttime,
            "split": self.split,
            "latest_risk_score": self.latest_risk_score,
            "latest_status": self.latest_status,
            "n_windows": self.n_windows,
        }


def build_roster_from_unscaled_df(df: pd.DataFrame, split: str = "test") -> List[RosterPatient]:
    """
    Build the canonical roster from an UNSCALED (original-units)
    per-timestep dataframe — i.e. `stay_id`/`subject_id`/`hadm_id`/
    `charttime`/`age`/`sex_male` in their real units (recovered via
    `scaler.inverse_transform`, see inference_service.py), grouped by
    stay_id. One roster entry per stay_id present in `df`.
    """
    required = {"stay_id", "subject_id", "hadm_id", "charttime", "age", "sex_male"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_roster_from_unscaled_df: input dataframe is missing required "
            f"columns {sorted(missing)} — this must be called with an unscaled "
            f"engineered dataframe from inference_service.py, not raw ml_dataset.csv."
        )

    entries: List[RosterPatient] = []
    for stay_id, group in df.groupby("stay_id"):
        group = group.sort_values("charttime")
        age_vals = group["age"].dropna()
        sex_vals = group["sex_male"].dropna()

        entries.append(
            RosterPatient(
                stay_id=int(stay_id),
                subject_id=int(group["subject_id"].iloc[0]),
                hadm_id=int(group["hadm_id"].iloc[0]),
                age_years=float(age_vals.iloc[0]) if len(age_vals) else None,
                sex_male=bool(sex_vals.iloc[0] >= 0.5) if len(sex_vals) else None,
                n_observed_timesteps=int(len(group)),
                first_charttime=str(group["charttime"].iloc[0]),
                last_charttime=str(group["charttime"].iloc[-1]),
                split=split,
            )
        )

    entries.sort(key=lambda e: e.stay_id)
    logger.info(f"Built canonical roster: {len(entries)} patients from the '{split}' split.")
    return entries
