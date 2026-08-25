"""
extract_to_ml_dataset.py — Flattens the raw, MIMIC-IV-shaped synthetic tables
(patients, admissions, icustays, chartevents, labevents, inputevents,
outputevents, antibiotics, microbiologyevents) into the single flat modeling
table `ml_dataset.csv` that `preprocessing.load_data()` reads.

THIS IS THE ONLY MODULE THAT CHANGES WHEN REAL MIMIC-IV REPLACES THE
SYNTHETIC GENERATOR. Everything downstream of `ml_dataset.csv`
(preprocessing.py onward) is schema-stable — a real-data extraction script
producing the same columns (DATA_CONTRACT.md section 3) is a drop-in
replacement for this module. A real extraction would run equivalent SQL
against BigQuery/PhysioNet-hosted MIMIC-IV instead of pivoting in-memory
DataFrames, but the output contract is identical.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import schema
from .generator import _wide_range_datetime

REVERSE_CHART_ITEMS = {v: k for k, v in schema.CHART_ITEMS.items()}
REVERSE_LAB_ITEMS = {v: k for k, v in schema.LAB_ITEMS.items()}

# Sanity clip ranges applied at extraction — represents the kind of hard
# physiological bounds checking a real extraction pipeline performs, without
# removing the plausible-but-noisy outliers the generator injects.
HARD_BOUNDS = {
    "heart_rate": (10, 300), "sbp": (20, 300), "dbp": (5, 220), "map": (10, 250),
    "resp_rate": (0, 80), "spo2": (0, 100), "temp": (25, 45),
    "lactate": (0, 30), "wbc": (0, 100), "creatinine": (0, 25),
    "bilirubin": (0, 50), "platelets": (0, 1200), "gcs": (3, 15),
    "fio2": (0.21, 1.0), "urine_output_rate": (0, 2000),
}


def _pivot_chartevents(chartevents: pd.DataFrame) -> pd.DataFrame:
    if chartevents.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime"])
    df = chartevents.copy()
    df["var"] = df["itemid"].map(REVERSE_CHART_ITEMS)
    wide = df.pivot_table(index=["subject_id", "hadm_id", "stay_id", "charttime"],
                           columns="var", values="valuenum", aggfunc="mean")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def _pivot_labevents(labevents: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 7 (post Phase-1-audit remediation): real MIMIC-IV `labevents` has no
    `stay_id` (labs are drawn hospital-wide, not per-ICU-stay) — the previous
    implementation attached `stay_id` with a blind `merge(..., on="hadm_id")`,
    which silently DUPLICATES every lab row once per ICU stay under that
    admission whenever a `hadm_id` has more than one `stay_id` (e.g. a
    patient transferred back to the ICU during the same hospitalization).
    The synthetic generator currently only ever produces one stay per
    admission, so this bug was latent and undetected — see
    tests/test_multi_stay.py for a hand-built fixture with a real multi-stay
    admission that would have caught it.

    Fixed by attaching `stay_id` via an INTERVAL match instead: a lab is
    attributed to the stay whose [intime, outtime] window contains its
    charttime. A lab drawn outside every stay's window for that admission
    (e.g. pre-ICU) is dropped — this pipeline only models ICU-time-series
    inputs. This is also exactly what a real MIMIC-IV extraction needs to do
    (see DATA_CONTRACT.md section 4a).
    """
    if labevents.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime"])
    df = labevents.copy()
    df["var"] = df["itemid"].map(REVERSE_LAB_ITEMS)
    wide = df.pivot_table(index=["subject_id", "hadm_id", "charttime"],
                           columns="var", values="valuenum", aggfunc="mean").reset_index()
    wide.columns.name = None

    merged = wide.merge(icustays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id", how="inner")
    in_window = (merged["charttime"] >= merged["intime"]) & (merged["charttime"] <= merged["outtime"])
    merged = merged[in_window].drop(columns=["intime", "outtime"])
    # Defensive de-dup: only fires if a hadm_id's ICU stays overlap in time
    # (shouldn't happen for real, non-overlapping stays), which would
    # otherwise let one lab match more than one stay's window.
    merged = merged.drop_duplicates(subset=["subject_id", "hadm_id", "stay_id", "charttime"])
    return merged


def _vasopressor_series(inputevents: pd.DataFrame) -> pd.DataFrame:
    vaso = inputevents[inputevents["itemid"] == schema.INPUT_ITEMS["vasopressor"]]
    if vaso.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "vasopressor_dose"])
    starts = vaso[["subject_id", "hadm_id", "stay_id", "starttime", "rate"]].rename(
        columns={"starttime": "charttime", "rate": "vasopressor_dose"})
    # a stop event (dose -> 0) at the end of the last segment per stay
    stops = (vaso.sort_values("endtime").groupby("stay_id").tail(1)
             [["subject_id", "hadm_id", "stay_id", "endtime"]]
             .rename(columns={"endtime": "charttime"}))
    stops["vasopressor_dose"] = 0.0
    return pd.concat([starts, stops], ignore_index=True)


def _fluid_bolus_series(inputevents: pd.DataFrame) -> pd.DataFrame:
    fluid = inputevents[inputevents["itemid"] == schema.INPUT_ITEMS["fluid_bolus"]]
    if fluid.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "fluid_bolus_ml"])
    return fluid[["subject_id", "hadm_id", "stay_id", "starttime", "amount"]].rename(
        columns={"starttime": "charttime", "amount": "fluid_bolus_ml"})


def _urine_series(outputevents: pd.DataFrame) -> pd.DataFrame:
    if outputevents.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "urine_output_rate"])
    return outputevents.rename(columns={"value": "urine_output_rate"})[
        ["subject_id", "hadm_id", "stay_id", "charttime", "urine_output_rate"]]


def compute_suspected_infection_windows(antibiotics: pd.DataFrame, microbiology: pd.DataFrame) -> pd.DataFrame:
    """
    Sepsis-3 suspected-infection window: for a hadm_id, if a culture was
    obtained within [-24h, +72h] of an antibiotic administration, suspected
    infection is considered active from min(antibiotic_time, culture_time)
    onward for the remainder of the admission.

    Returns one row per qualifying hadm_id: {hadm_id, suspected_infection_start}.
    """
    if antibiotics.empty or microbiology.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_start"])
    merged = antibiotics.merge(microbiology, on=["subject_id", "hadm_id"], suffixes=("_abx", "_cx"))
    if merged.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_start"])
    delta_h = (merged["charttime"] - merged["starttime"]).dt.total_seconds() / 3600.0
    qualifies = (delta_h >= -24.0) & (delta_h <= 72.0)
    merged = merged[qualifies].copy()
    if merged.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_start"])
    merged["onset"] = merged[["starttime", "charttime"]].min(axis=1)
    onset = merged.groupby("hadm_id")["onset"].min().reset_index().rename(
        columns={"onset": "suspected_infection_start"})
    return onset


def extract(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten raw synthetic MIMIC-shaped tables into ml_dataset.csv rows."""
    icustays = raw["icustays"]
    chart_wide = _pivot_chartevents(raw["chartevents"])
    lab_wide = _pivot_labevents(raw["labevents"], icustays)
    vaso = _vasopressor_series(raw["inputevents"])
    fluid = _fluid_bolus_series(raw["inputevents"])
    urine = _urine_series(raw["outputevents"])

    frames = []
    for df in [chart_wide, lab_wide, vaso, fluid, urine]:
        if len(df):
            frames.append(df)
    if not frames:
        raise ValueError("No observations generated — check generator configuration.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.groupby(["subject_id", "hadm_id", "stay_id", "charttime"], as_index=False).mean(numeric_only=True)

    combined = combined.merge(raw["patients"][["subject_id", "age", "sex_male"]], on="subject_id", how="left")

    onset = compute_suspected_infection_windows(raw["antibiotics"], raw["microbiologyevents"])
    combined = combined.merge(onset, on="hadm_id", how="left")
    combined["suspected_infection"] = (
        combined["suspected_infection_start"].notna()
        & (combined["charttime"] >= combined["suspected_infection_start"])
    ).astype(int)
    combined = combined.drop(columns=["suspected_infection_start"])

    # Stays that never had a vasopressor event have a TRUE zero dose throughout
    # (absence of treatment, not an unknown value) — distinct from labs/GCS,
    # where an unobserved value before the first measurement really is unknown.
    # LOCF alone would otherwise leave these stays entirely NaN on this column.
    if "vasopressor_dose" in combined.columns:
        stays_with_vaso = set(raw.get("inputevents", pd.DataFrame(columns=["stay_id", "itemid"]))
                               .loc[lambda d: d.get("itemid") == schema.INPUT_ITEMS["vasopressor"], "stay_id"]) \
            if "itemid" in raw.get("inputevents", pd.DataFrame()).columns else set()
        no_vaso_mask = ~combined["stay_id"].isin(stays_with_vaso)
        combined.loc[no_vaso_mask, "vasopressor_dose"] = combined.loc[no_vaso_mask, "vasopressor_dose"].fillna(0.0)

    for col, (lo, hi) in HARD_BOUNDS.items():
        if col in combined.columns:
            combined[col] = combined[col].clip(lower=lo, upper=hi)

    expected_cols = [
        "subject_id", "hadm_id", "stay_id", "charttime",
        "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temp",
        "lactate", "wbc", "creatinine", "bilirubin", "platelets",
        "gcs", "fio2", "ventilated", "urine_output_rate",
        "vasopressor_dose", "fluid_bolus_ml", "age", "sex_male", "suspected_infection",
    ]
    for col in expected_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    # Coerce to a wide-range-safe datetime resolution regardless of the input
    # timestamps' magnitude. This project's real generator output uses
    # far-future de-identified dates (~year 2446) that overflow pandas'
    # default nanosecond datetime64[ns] resolution; hand-built test fixtures
    # (e.g. tests/test_multi_stay.py) use ordinary present-day dates. Both
    # must end up on the SAME resolution so extract()'s output can safely be
    # concatenated with other extract() output (as the test suite does) or
    # written to CSV without ever silently corrupting to NaT or overflowing.
    # See backend/synthetic_data/generator.py::_wide_range_datetime for the
    # full rationale.
    if "charttime" in combined.columns:
        combined["charttime"] = _wide_range_datetime(combined["charttime"])

    combined = combined.sort_values(["subject_id", "stay_id", "charttime"]).reset_index(drop=True)
    return combined[expected_cols]
