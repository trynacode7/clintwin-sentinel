"""
labels_sofa.py — SOFA-based, Sepsis-3-consistent future-window label creation.

Implements the clinical endpoint fixed in TECHNICAL_BLUEPRINT.md section 3:

    INPUT WINDOW       6 hours of observed vitals/labs/treatment state
    PREDICTION HORIZON  next 2-6 hours
    LABEL               binary: SOFA total score increases by >= 2 points
                         within the horizon, restricted to stays meeting the
                         Sepsis-3 "suspected infection" criterion
    MODEL OUTPUT         calibrated probability (calibration happens in
                         evaluate.py, not here)

This module NEVER receives a precomputed label — it computes SOFA subscores
and the future-window label from observed vitals/labs/treatment columns only,
exactly as labels.py's original future-window discipline required ("the
label is derived purely from future physiological measurements", never from
a precomputed flag). `suspected_infection` is read here (it is excluded from
config.ALL_FEATURE_COLS — see config.LABEL_ONLY_COLS — so it can define label
eligibility without being visible to the model as an input feature).

SOFA subscore thresholds below are the standard published Sepsis-3/SOFA
criteria (Vincent et al. 1996; Singer et al. 2016 Sepsis-3), with one
documented substitution: PaO2/FiO2 is not available in this schema (no
arterial blood gas is synthesized — see DATA_CONTRACT.md section 7), so the
respiratory subscore uses the SpO2/FiO2 (SF) ratio, a real, published
surrogate for PaO2/FiO2 when blood gases are unavailable (Pandharipande et
al. 2009; Rice et al. 2007), and the cardiovascular subscore is simplified to
a single norepinephrine-equivalent dose (this schema does not distinguish
dopamine/dobutamine, so the SOFA=2 "low-dose dopamine" tier is not
representable and is intentionally skipped — documented, not silently
dropped).

labels.py (the original ad hoc threshold rule) is left completely untouched
and unused by default — see TECHNICAL_BLUEPRINT.md's REPLACE list. train.py
imports from this module instead.

============================================================================
FIX 1 (post Phase-1-audit remediation) — cardiovascular MAP threshold
============================================================================
The cardiovascular subscore's MAP threshold previously reused the legacy
labels.py rule's ad hoc value (65 mmHg) via a shared config constant. The
published SOFA cardiovascular criterion (Vincent et al. 1996) awards 1 point
for MAP < 70 mmHg, not < 65. This has been corrected: `config.
SOFA_CARDIOVASCULAR_MAP_THRESHOLD` is now an independent constant fixed at
70.0 and is never aliased to `config.MAP_THRESHOLD` (the legacy, unused-by-
default labels.py rule) again. See tests/test_sofa_scoring.py for a direct
regression test of this threshold.

============================================================================
FIX 9 (post Phase-1-audit remediation) — baseline definition, documented
============================================================================
The "baseline" SOFA used below (`baseline_sofa[i] = sofa_vals[i]`, i.e. the
CURRENT row's own SOFA at each rolling timestep) is a ROLLING, per-timestep
baseline, not the official Sepsis-3 protocol's baseline (Singer et al. 2016
anchors baseline SOFA at ICU/hospital admission, or assumes 0 absent known
preexisting organ dysfunction, and evaluates the delta from that single
fixed point for the whole encounter). This project deliberately uses the
rolling-baseline reframing because it is built as a CONTINUOUS early-warning
scorer producing a fresh risk estimate every hour, not a one-time admission-
diagnosis tool — this is how several published ICU early-warning-score
papers adapt SOFA-style deltas for continuous monitoring, but it is a
reinterpretation of Sepsis-3, not a literal implementation of it, and must
not be described as the latter.

If a strictly admission-anchored baseline were wanted instead, the required
change would be: compute `admission_sofa = sofa_vals[0]` (or the first
non-NaN SOFA in the stay) ONCE per stay outside the per-row loop, and use
that fixed value as `baseline_sofa[i]` for every `i` in that stay, rather
than `sofa_vals[i]`. This is a substantive change to what the label MEANS
(every future row would be compared against admission state instead of
"how much worse than an hour ago") and is not implemented here — flagging it
for an explicit decision before any future change, per the audit response,
rather than silently switching definitions.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")


# ─── SOFA subscores (0-4 each), vectorized over a preprocessed hourly frame ──

def _respiratory_subscore(spo2: pd.Series, fio2: pd.Series, ventilated: pd.Series) -> pd.Series:
    fio2 = fio2.fillna(0.21).clip(lower=0.21)
    sf_ratio = (spo2 / fio2).clip(upper=1000)
    vent = ventilated.fillna(0).astype(float) >= 0.5

    score = pd.Series(0, index=spo2.index, dtype=float)
    score.loc[sf_ratio < 400] = 1
    score.loc[sf_ratio < 315] = 2
    score.loc[(sf_ratio < 235) & vent] = 3
    score.loc[(sf_ratio < 150) & vent] = 4
    score.loc[spo2.isna()] = np.nan
    return score


def _coagulation_subscore(platelets: pd.Series) -> pd.Series:
    score = pd.Series(0, index=platelets.index, dtype=float)
    score.loc[platelets < 150] = 1
    score.loc[platelets < 100] = 2
    score.loc[platelets < 50] = 3
    score.loc[platelets < 20] = 4
    score.loc[platelets.isna()] = np.nan
    return score


def _liver_subscore(bilirubin: pd.Series) -> pd.Series:
    score = pd.Series(0, index=bilirubin.index, dtype=float)
    score.loc[bilirubin >= 1.2] = 1
    score.loc[bilirubin >= 2.0] = 2
    score.loc[bilirubin >= 6.0] = 3
    score.loc[bilirubin >= 12.0] = 4
    score.loc[bilirubin.isna()] = np.nan
    return score


def _cardiovascular_subscore(map_: pd.Series, vasopressor_dose: pd.Series) -> pd.Series:
    dose = vasopressor_dose.fillna(0)
    score = pd.Series(0, index=map_.index, dtype=float)
    score.loc[map_ < config.SOFA_CARDIOVASCULAR_MAP_THRESHOLD] = 1
    score.loc[(dose > 0) & (dose <= 0.1)] = 3
    score.loc[dose > 0.1] = 4
    score.loc[map_.isna() & vasopressor_dose.isna()] = np.nan
    return score


def _cns_subscore(gcs: pd.Series) -> pd.Series:
    score = pd.Series(0, index=gcs.index, dtype=float)
    score.loc[gcs <= 14] = 1
    score.loc[gcs <= 12] = 2
    score.loc[gcs <= 9] = 3
    score.loc[gcs <= 5] = 4
    score.loc[gcs.isna()] = np.nan
    return score


def _renal_subscore(creatinine: pd.Series, urine_output_rate: pd.Series) -> pd.Series:
    cr_score = pd.Series(0, index=creatinine.index, dtype=float)
    cr_score.loc[creatinine >= 1.2] = 1
    cr_score.loc[creatinine >= 2.0] = 2
    cr_score.loc[creatinine >= 3.5] = 3
    cr_score.loc[creatinine >= 5.0] = 4
    cr_score.loc[creatinine.isna()] = np.nan

    uo_score = pd.Series(0, index=urine_output_rate.index, dtype=float)
    uo_score.loc[urine_output_rate < 30] = 3   # oliguria, roughly < 500 mL/day
    uo_score.loc[urine_output_rate < 10] = 4   # roughly < 200 mL/day
    uo_score.loc[urine_output_rate.isna()] = np.nan

    combined = pd.concat([cr_score, uo_score], axis=1).max(axis=1, skipna=True)
    return combined


def compute_sofa(df: pd.DataFrame) -> pd.Series:
    """
    Compute the total SOFA score per row from observed columns only.

    Missing subscores (no observation of that organ system yet) are treated
    as 0 (i.e. "no evidence of dysfunction observed") rather than NaN, which
    is the standard convention used by published SOFA-computation pipelines
    (e.g. the MIT-LCP mimic-code SOFA queries) — documented here, not silent.
    """
    resp = _respiratory_subscore(df["spo2"], df.get("fio2", pd.Series(index=df.index, dtype=float)),
                                  df.get("ventilated", pd.Series(index=df.index, dtype=float)))
    coag = _coagulation_subscore(df["platelets"])
    liver = _liver_subscore(df["bilirubin"])
    cardio = _cardiovascular_subscore(df["map"], df.get("vasopressor_dose", pd.Series(index=df.index, dtype=float)))
    cns = _cns_subscore(df["gcs"])
    renal = _renal_subscore(df["creatinine"], df.get("urine_output_rate", pd.Series(index=df.index, dtype=float)))

    subscores = pd.concat(
        [resp.fillna(0), coag.fillna(0), liver.fillna(0), cardio.fillna(0), cns.fillna(0), renal.fillna(0)],
        axis=1,
    )
    subscores.columns = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal"]
    total = subscores.sum(axis=1)
    return total, subscores


def create_labels_for_stay(stay_data: pd.DataFrame) -> pd.DataFrame:
    """
    Future-window SOFA-increase label for a single stay, sorted by time.

    Reuses the exact future-window mechanics of the original labels.py:
    for input position i, look at rows [i + FUTURE_WINDOW_START, i +
    FUTURE_WINDOW_END); if there isn't enough future data, the label is NaN
    (dropped downstream, same as the original pipeline).
    """
    n = len(stay_data)
    total_sofa, subscores = compute_sofa(stay_data)
    stay_data = stay_data.copy()
    stay_data["sofa_total"] = total_sofa.values
    for col in subscores.columns:
        stay_data[col] = subscores[col].values

    labels = np.full(n, np.nan)
    baseline_sofa = np.full(n, np.nan)
    future_max_sofa = np.full(n, np.nan)
    suspected_infection_in_horizon = np.zeros(n, dtype=bool)

    sofa_vals = stay_data["sofa_total"].values
    infection_vals = stay_data["suspected_infection"].values if "suspected_infection" in stay_data.columns else np.zeros(n)

    for i in range(n):
        future_start = i + config.FUTURE_WINDOW_START
        future_end = i + config.FUTURE_WINDOW_END
        if future_end > n:
            continue  # not enough future data — excluded, never a manufactured negative

        future_sofa = sofa_vals[future_start:future_end]
        future_infection = infection_vals[future_start:future_end]

        baseline_sofa[i] = sofa_vals[i]
        future_max_sofa[i] = np.nanmax(future_sofa) if len(future_sofa) else np.nan
        eligible = bool(np.any(future_infection >= 1))
        suspected_infection_in_horizon[i] = eligible

        if np.isnan(baseline_sofa[i]) or np.isnan(future_max_sofa[i]):
            continue

        deteriorated = (future_max_sofa[i] - baseline_sofa[i]) >= config.SOFA_INCREASE_THRESHOLD
        labels[i] = float(eligible and deteriorated)

    stay_data["baseline_sofa"] = baseline_sofa
    stay_data["future_max_sofa"] = future_max_sofa
    stay_data["suspected_infection_in_horizon"] = suspected_infection_in_horizon
    stay_data["label"] = labels
    return stay_data


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create SOFA/Sepsis-3-consistent future-window labels for all stays.

    Same external contract as the original labels.py:create_labels() —
    returns a DataFrame with an added "label" column and drops rows without
    enough future data — so dataset.py and train.py need no changes.
    """
    logger.info("Creating SOFA/Sepsis-3-consistent future-window labels...")

    labeled_groups = []
    for stay_id, group in df.groupby("stay_id"):
        group = group.sort_values(config.TIME_COL).reset_index(drop=True)
        labeled_groups.append(create_labels_for_stay(group))

    df_labeled = pd.concat(labeled_groups, ignore_index=True)

    n_before = len(df_labeled)
    df_labeled = df_labeled.dropna(subset=["label"]).reset_index(drop=True)
    n_after = len(df_labeled)
    logger.info(f"Labels created. Dropped {n_before - n_after} rows without sufficient future data.")

    n_pos = (df_labeled["label"] == 1).sum()
    n_neg = (df_labeled["label"] == 0).sum()
    ratio = n_pos / max(n_neg, 1)
    n_eligible = (df_labeled["suspected_infection_in_horizon"]).sum()
    logger.info(
        f"Label distribution: pos={n_pos}, neg={n_neg}, ratio={ratio:.3f}, "
        f"suspected-infection-eligible windows={n_eligible}"
    )

    return df_labeled


if __name__ == "__main__":
    from backend.preprocessing import preprocess, scale_splits
    from backend.feature_engineering import engineer_features

    result = preprocess()
    for split in ["train", "val", "test"]:
        result[split] = engineer_features(result[split])
        result[split] = create_labels(result[split])  # MUST run on raw units — before scale_splits()
    result = scale_splits(result)
    for split in ["train", "val", "test"]:
        n_pos = (result[split]["label"] == 1).sum()
        n_neg = (result[split]["label"] == 0).sum()
        print(f"{split}: pos={n_pos}, neg={n_neg}, total={len(result[split])}")
