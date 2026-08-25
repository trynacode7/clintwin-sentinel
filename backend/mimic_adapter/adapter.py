"""
adapter.py — MIMIC-IV -> canonical ClinTwin ICU schema adapter (Phase 2).

This is the ONLY module (together with concept_map.py and units.py, its two
collaborators in this package) that knows MIMIC-IV table names, item IDs, or
MIMIC-specific quirks. Its output is BYTE-FOR-BYTE the same schema as
`backend/synthetic_data/extract_to_ml_dataset.py`'s output (DATA_CONTRACT.md
section 3) — built from the SAME config.ID_COLS/TIME_COL/ALL_FEATURE_COLS/
LABEL_ONLY_COLS definitions, not a hand-copied column list, so the two
sources cannot silently drift apart.

    MIMIC-IV raw tables
            |
            v
      cohort.select_cohort()      <- separate stage, no observation values touched
            |
            v
      adapt()                     <- THIS MODULE: pivots + unit-harmonizes +
            |                         interval-joins into the canonical schema
            v
    canonical ml_dataset.csv-shaped DataFrame
            |
            v
    existing preprocessing.py / feature_engineering.py / labels_sofa.py /
    dataset.py / train.py — unchanged, source-agnostic

Multi-stay discipline: `_pivot_labevents` below reuses the EXACT interval
(time-window) join pattern introduced in Phase 1 for the synthetic pipeline's
`extract_to_ml_dataset._pivot_labevents` (Fix 7) — labevents has no native
stay_id in real MIMIC-IV either, so the same bug class (a blind hadm_id join
duplicating every lab row once per stay) applies here just as much, and is
prevented the same way.

Verification-gating: `adapt()` refuses to run (raises via
`concept_map.require_verified()`) unless every concept the v1 canonical
schema needs is marked `verified=True` in concept_map.py — UNLESS
`skip_verification_check=True` is passed, which must only ever be used for
MIMIC-STRUCTURE TEST FIXTURES (see tests/test_mimic_adapter.py), never for
data claiming to be real MIMIC-IV.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

import config
from . import concept_map, cohort as cohort_mod, units

# Re-derived from config so the MIMIC canonical output and the synthetic
# canonical output (extract_to_ml_dataset.extract()) are provably the same
# schema, not two independently-maintained column lists.
_EXPECTED_COLS = (
    list(config.ID_COLS) + [config.TIME_COL] + list(config.ALL_FEATURE_COLS) + list(config.LABEL_ONLY_COLS)
)

# Same sanity-bounds contract as the synthetic extractor (extract_to_ml_dataset.HARD_BOUNDS).
from backend.synthetic_data.extract_to_ml_dataset import HARD_BOUNDS  # noqa: E402


def _itemid_to_var_map(concepts: Dict[str, concept_map.ConceptMapping]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, cm in concepts.items():
        for itemid in cm.itemids:
            out[itemid] = name
    return out


def pivot_chartevents(chartevents: pd.DataFrame, concepts: Dict[str, concept_map.ConceptMapping]) -> pd.DataFrame:
    """
    mimiciv_icu.chartevents (subject_id, hadm_id, stay_id, charttime, itemid,
    value, valuenum, valueuom) -> wide per-(stay_id, charttime) canonical
    columns, with per-row unit harmonization for temperature and FiO2.
    """
    if chartevents.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime"])

    itemid_to_var = _itemid_to_var_map(concepts)
    df = chartevents[chartevents["itemid"].isin(itemid_to_var.keys())].copy()
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime"])
    df["var"] = df["itemid"].map(itemid_to_var)

    # Temperature: row-wise C/F harmonization BEFORE aggregation.
    temp_mask = df["var"] == "temp"
    if temp_mask.any():
        df.loc[temp_mask, "valuenum"] = units.convert_temperature(
            df.loc[temp_mask, "valuenum"], df.loc[temp_mask, "valueuom"]
        )

    # FiO2: percent-vs-fraction normalization.
    fio2_mask = df["var"] == "fio2"
    if fio2_mask.any():
        normalized, _pct_mask = units.validate_and_normalize_fio2(df.loc[fio2_mask, "valuenum"])
        df.loc[fio2_mask, "valuenum"] = normalized

    wide = df.pivot_table(
        index=["subject_id", "hadm_id", "stay_id", "charttime"],
        columns="var", values="valuenum", aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    # GCS total = sum of the three verified components, only where all
    # three concepts are actually present as columns (they may not be, if a
    # caller passed a partial concept dict, e.g. in a fixture test).
    gcs_parts = [c for c in ("gcs_eye", "gcs_verbal", "gcs_motor") if c in wide.columns]
    if len(gcs_parts) == 3:
        wide["gcs"] = wide[gcs_parts].sum(axis=1, skipna=False)
        wide = wide.drop(columns=gcs_parts)
    elif gcs_parts:
        wide = wide.drop(columns=gcs_parts)  # partial components with no total is not usable

    return wide


def pivot_labevents(labevents: pd.DataFrame, icustays: pd.DataFrame,
                     concepts: Dict[str, concept_map.ConceptMapping]) -> pd.DataFrame:
    """
    mimiciv_hosp.labevents has NO native stay_id (labs are drawn hospital-
    wide). Attributes each lab to the ICU stay whose [intime, outtime) window
    contains its charttime — the SAME interval-join pattern as
    extract_to_ml_dataset._pivot_labevents (Phase 1, Fix 7) — never a blind
    hadm_id merge, which would duplicate every lab row once per stay for any
    admission with more than one ICU stay.
    """
    if labevents.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime"])

    itemid_to_var = _itemid_to_var_map(concepts)
    df = labevents[labevents["itemid"].isin(itemid_to_var.keys())].copy()
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime"])
    df["var"] = df["itemid"].map(itemid_to_var)

    wide = df.pivot_table(
        index=["subject_id", "hadm_id", "charttime"], columns="var", values="valuenum", aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    merged = wide.merge(icustays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id", how="inner")
    in_window = (merged["charttime"] >= merged["intime"]) & (merged["charttime"] <= merged["outtime"])
    merged = merged[in_window].drop(columns=["intime", "outtime"])
    merged = merged.drop_duplicates(subset=["subject_id", "hadm_id", "stay_id", "charttime"])
    return merged


def vasopressor_series(inputevents: pd.DataFrame, concepts: Dict[str, concept_map.ConceptMapping]) -> pd.DataFrame:
    itemids = concepts["vasopressor_dose"].itemids
    vaso = inputevents[inputevents["itemid"].isin(itemids)]
    if vaso.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "vasopressor_dose"])
    starts = vaso[["subject_id", "hadm_id", "stay_id", "starttime", "rate"]].rename(
        columns={"starttime": "charttime", "rate": "vasopressor_dose"})
    stops = (vaso.sort_values("endtime").groupby("stay_id").tail(1)
             [["subject_id", "hadm_id", "stay_id", "endtime"]]
             .rename(columns={"endtime": "charttime"}))
    stops["vasopressor_dose"] = 0.0
    return pd.concat([starts, stops], ignore_index=True)


def fluid_bolus_series(inputevents: pd.DataFrame, concepts: Dict[str, concept_map.ConceptMapping]) -> pd.DataFrame:
    itemids = concepts["fluid_bolus_ml"].itemids
    fluid = inputevents[inputevents["itemid"].isin(itemids)]
    if fluid.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "fluid_bolus_ml"])
    return fluid[["subject_id", "hadm_id", "stay_id", "starttime", "amount"]].rename(
        columns={"starttime": "charttime", "amount": "fluid_bolus_ml"})


def urine_series(outputevents: pd.DataFrame, concepts: Dict[str, concept_map.ConceptMapping]) -> pd.DataFrame:
    itemids = concepts["urine_output_rate"].itemids
    out = outputevents[outputevents["itemid"].isin(itemids)]
    if out.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "urine_output_rate"])
    return out.rename(columns={"value": "urine_output_rate"})[
        ["subject_id", "hadm_id", "stay_id", "charttime", "urine_output_rate"]]


def ventilation_indicator(procedureevents: pd.DataFrame, chartevents_wide: pd.DataFrame,
                           concepts: Dict[str, concept_map.ConceptMapping]) -> pd.DataFrame:
    """
    ventilated = 1 for any observation row whose charttime falls inside a
    ventilation episode's [starttime, endtime); 0 otherwise (true absence,
    not missing — same convention as vasopressor_dose). concept_map.py flags
    this concept's source table as ambiguous (procedureevents vs.
    chartevents) — this function takes procedureevents episodes as given
    (the caller resolves which source to use once verified) and interval-
    joins against the already-pivoted chartevents_wide frame's timestamps so
    a ventilated flag exists at every timestamp that already has other
    observations, without inventing new rows.
    """
    if procedureevents.empty or chartevents_wide.empty:
        result = chartevents_wide[["subject_id", "hadm_id", "stay_id", "charttime"]].copy()
        result["ventilated"] = 0.0
        return result

    itemids = concepts["ventilated"].itemids
    vent = procedureevents[procedureevents["itemid"].isin(itemids)]
    result = chartevents_wide[["subject_id", "hadm_id", "stay_id", "charttime"]].copy()
    result["ventilated"] = 0.0
    if vent.empty:
        return result

    merged = result.merge(vent[["stay_id", "starttime", "endtime"]], on="stay_id", how="left")
    overlapping = (merged["charttime"] >= merged["starttime"]) & (merged["charttime"] < merged["endtime"])
    overlapping = overlapping.fillna(False)
    hit_index = merged.loc[overlapping, ["subject_id", "hadm_id", "stay_id", "charttime"]].drop_duplicates()
    result = result.merge(hit_index.assign(_hit=1.0), on=["subject_id", "hadm_id", "stay_id", "charttime"], how="left")
    result["ventilated"] = result["_hit"].fillna(0.0)
    return result.drop(columns=["_hit"])


def compute_suspected_infection_windows(prescriptions: pd.DataFrame, microbiologyevents: pd.DataFrame,
                                         antibiotic_drug_names: Optional[set] = None) -> pd.DataFrame:
    """
    Sepsis-3 suspected-infection window: for a hadm_id, if a culture was
    obtained within [-24h, +72h] of a QUALIFYING antibiotic administration,
    suspected infection is active from min(antibiotic_time, culture_time)
    onward. Same window logic as
    extract_to_ml_dataset.compute_suspected_infection_windows.

    `antibiotic_drug_names`: real prescriptions.drug is free text — this
    function does NOT invent an antibiotic name list (see concept_map.py's
    antibiotic_administration entry). If not supplied, every prescriptions
    row is treated as non-qualifying and this returns empty — callers must
    supply a citable antibiotic name/class list once real data is available.
    """
    if prescriptions.empty or microbiologyevents.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_start"])
    abx = prescriptions
    if antibiotic_drug_names is not None:
        abx = prescriptions[prescriptions["drug"].isin(antibiotic_drug_names)]
    if abx.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_start"])

    micro = microbiologyevents.rename(columns={"charttime": "_culture_charttime"})
    merged = abx.merge(micro, on=["subject_id", "hadm_id"])
    if merged.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_start"])
    delta_h = (merged["_culture_charttime"] - merged["starttime"]).dt.total_seconds() / 3600.0
    qualifies = (delta_h >= -24.0) & (delta_h <= 72.0)
    merged = merged[qualifies].copy()
    if merged.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_infection_start"])
    merged["onset"] = merged[["starttime", "_culture_charttime"]].min(axis=1)
    onset = merged.groupby("hadm_id")["onset"].min().reset_index().rename(columns={"onset": "suspected_infection_start"})
    return onset


def adapt(
    raw: Dict[str, pd.DataFrame],
    concepts: Optional[Dict[str, concept_map.ConceptMapping]] = None,
    apply_cohort_filter: bool = True,
    skip_verification_check: bool = False,
    antibiotic_drug_names: Optional[set] = None,
) -> pd.DataFrame:
    """
    Full MIMIC-IV raw tables -> canonical ml_dataset.csv-shaped DataFrame.

    `raw` keys expected: patients, admissions, icustays, chartevents,
    labevents, inputevents, outputevents, procedureevents, prescriptions,
    microbiologyevents (any absent/empty table degrades gracefully to no
    contribution from that source, matching extract_to_ml_dataset.py's
    behavior for the synthetic side).

    `skip_verification_check=True` must ONLY be used for MIMIC-STRUCTURE
    TEST FIXTURES (see tests/test_mimic_adapter.py) — never for data
    claiming to be real MIMIC-IV.
    """
    concepts = concepts or concept_map.ALL_CONCEPTS
    if not skip_verification_check:
        concept_map.require_verified()

    icustays = raw["icustays"]
    if apply_cohort_filter:
        eligible_stay_ids, cohort_stats = cohort_mod.select_cohort(
            icustays, raw["admissions"], raw["patients"]
        )
        icustays = icustays[icustays["stay_id"].isin(eligible_stay_ids)]
    else:
        cohort_stats = None

    chart_wide = pivot_chartevents(raw.get("chartevents", pd.DataFrame()), concepts)
    if len(chart_wide):
        chart_wide = chart_wide[chart_wide["stay_id"].isin(icustays["stay_id"])]

    lab_wide = pivot_labevents(raw.get("labevents", pd.DataFrame()), icustays, concepts)
    vaso = vasopressor_series(raw.get("inputevents", pd.DataFrame()), concepts)
    fluid = fluid_bolus_series(raw.get("inputevents", pd.DataFrame()), concepts)
    urine = urine_series(raw.get("outputevents", pd.DataFrame()), concepts)
    for frame in (vaso, fluid, urine):
        if len(frame):
            frame.drop(frame[~frame["stay_id"].isin(icustays["stay_id"])].index, inplace=True)

    frames = [f for f in [chart_wide, lab_wide, vaso, fluid, urine] if len(f)]
    if not frames:
        raise ValueError("No observations produced by the MIMIC-IV adapter — check concept_map.py "
                          "itemid mappings and cohort filtering.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.groupby(["subject_id", "hadm_id", "stay_id", "charttime"], as_index=False).mean(numeric_only=True)

    if "procedureevents" in raw and len(raw["procedureevents"]) and "ventilated" in concepts and concepts["ventilated"].itemids:
        vent_flags = ventilation_indicator(raw["procedureevents"], chart_wide, concepts)
        combined = combined.merge(
            vent_flags[["subject_id", "hadm_id", "stay_id", "charttime", "ventilated"]],
            on=["subject_id", "hadm_id", "stay_id", "charttime"], how="left",
        )
        combined["ventilated"] = combined["ventilated"].fillna(0.0)

    combined = combined.merge(raw["patients"][["subject_id"]], on="subject_id", how="left")
    age_by_hadm = cohort_mod.compute_age_at_admission(raw["patients"], raw["admissions"])
    combined["age"] = combined["hadm_id"].map(age_by_hadm)
    sex_map = raw["patients"].set_index("subject_id")["gender"].map({"M": 1.0, "F": 0.0})
    combined["sex_male"] = combined["subject_id"].map(sex_map)

    onset = compute_suspected_infection_windows(
        raw.get("prescriptions", pd.DataFrame()), raw.get("microbiologyevents", pd.DataFrame()),
        antibiotic_drug_names=antibiotic_drug_names,
    )
    combined = combined.merge(onset, on="hadm_id", how="left")
    combined["suspected_infection"] = (
        combined["suspected_infection_start"].notna()
        & (combined["charttime"] >= combined["suspected_infection_start"])
    ).astype(int)
    combined = combined.drop(columns=["suspected_infection_start"])

    if "vasopressor_dose" in combined.columns:
        stays_with_vaso = set(vaso["stay_id"].unique()) if len(vaso) else set()
        no_vaso_mask = ~combined["stay_id"].isin(stays_with_vaso)
        combined.loc[no_vaso_mask, "vasopressor_dose"] = combined.loc[no_vaso_mask, "vasopressor_dose"].fillna(0.0)

    for col, (lo, hi) in HARD_BOUNDS.items():
        if col in combined.columns:
            combined[col] = combined[col].clip(lower=lo, upper=hi)

    for col in _EXPECTED_COLS:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined.sort_values(["subject_id", "stay_id", "charttime"]).reset_index(drop=True)
    combined.attrs["cohort_stats"] = cohort_stats.as_dict() if cohort_stats else None
    return combined[_EXPECTED_COLS]
