"""
test_multi_stay.py — Fix 7 (post Phase-1-audit remediation).

The synthetic generator only ever produces one stay per admission (1
subject_id : 1 hadm_id : 1 stay_id), so the patient-level split logic and
the lab-to-stay attribution join in extract_to_ml_dataset.py had NEVER been
exercised against the one scenario they exist to handle: a real patient
with multiple ICU stays. This hand-builds that scenario directly (not via
the generator) and verifies:

  1. extract_to_ml_dataset._pivot_labevents attributes each lab to the
     correct stay by TIME WINDOW, not by a blind hadm_id join (which would
     duplicate every lab row once per stay under that admission).
  2. extract() end-to-end keeps the two stays separate with no duplication.
  3. preprocessing.split_by_subject never splits one patient's stays across
     train/val/test — genuinely patient-level, not stay-level, isolation.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from backend.synthetic_data import schema
from backend.synthetic_data.extract_to_ml_dataset import extract, _pivot_labevents
from backend.preprocessing import split_by_subject


MULTI_SUBJECT_ID = 999001
MULTI_HADM_ID = 9990011
MULTI_STAY_1 = 99900101
MULTI_STAY_2 = 99900102


def _build_multi_stay_raw():
    """One subject_id, one hadm_id, TWO non-overlapping stay_ids (e.g. an
    ICU readmission during the same hospitalization)."""
    t0 = datetime(2024, 1, 1)
    intime1, outtime1 = t0, t0 + timedelta(hours=30)
    intime2, outtime2 = t0 + timedelta(hours=40), t0 + timedelta(hours=70)

    patients = pd.DataFrame([{"subject_id": MULTI_SUBJECT_ID, "age": 55.0, "sex_male": 1}])
    admissions = pd.DataFrame([{
        "subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID,
        "admittime": t0 - timedelta(hours=2), "dischtime": outtime2 + timedelta(hours=5),
    }])
    icustays = pd.DataFrame([
        {"subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID, "stay_id": MULTI_STAY_1,
         "intime": intime1, "outtime": outtime1, "archetype": "stable"},
        {"subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID, "stay_id": MULTI_STAY_2,
         "intime": intime2, "outtime": outtime2, "archetype": "deteriorating"},
    ])

    chart_rows = []
    for h in [2, 10, 20]:
        chart_rows.append({"subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID, "stay_id": MULTI_STAY_1,
                            "charttime": intime1 + timedelta(hours=h),
                            "itemid": schema.CHART_ITEMS["heart_rate"], "valuenum": 80.0, "valueuom": "bpm"})
    for h in [5, 15, 25]:
        chart_rows.append({"subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID, "stay_id": MULTI_STAY_2,
                            "charttime": intime2 + timedelta(hours=h),
                            "itemid": schema.CHART_ITEMS["heart_rate"], "valuenum": 95.0, "valueuom": "bpm"})
    chartevents = pd.DataFrame(chart_rows)

    # labevents (real MIMIC shape: no stay_id) — one lab in each stay's window,
    # plus one lab drawn in the GAP between the two stays (must be attributed
    # to neither, not duplicated into both).
    lab_rows = [
        {"subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID, "charttime": intime1 + timedelta(hours=5),
         "itemid": schema.LAB_ITEMS["lactate"], "valuenum": 1.5, "valueuom": "mmol/L"},
        {"subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID, "charttime": intime2 + timedelta(hours=10),
         "itemid": schema.LAB_ITEMS["lactate"], "valuenum": 3.0, "valueuom": "mmol/L"},
        {"subject_id": MULTI_SUBJECT_ID, "hadm_id": MULTI_HADM_ID, "charttime": outtime1 + timedelta(hours=5),
         "itemid": schema.LAB_ITEMS["lactate"], "valuenum": 9.9, "valueuom": "mmol/L"},
    ]
    labevents = pd.DataFrame(lab_rows)

    empty_out = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"])
    empty_in = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate"])
    empty_abx = pd.DataFrame(columns=["subject_id", "hadm_id", "starttime"])
    empty_micro = pd.DataFrame(columns=["subject_id", "hadm_id", "charttime"])

    return {
        "patients": patients, "admissions": admissions, "icustays": icustays,
        "chartevents": chartevents, "labevents": labevents,
        "outputevents": empty_out, "inputevents": empty_in,
        "antibiotics": empty_abx, "microbiologyevents": empty_micro,
    }


@pytest.fixture(scope="module")
def multi_stay_raw():
    return _build_multi_stay_raw()


def test_pivot_labevents_attributes_by_time_window_not_hadm_id(multi_stay_raw):
    lab_wide = _pivot_labevents(multi_stay_raw["labevents"], multi_stay_raw["icustays"])
    # the gap-lab (drawn between the two stays) must be dropped, not duplicated into both
    assert len(lab_wide) == 2
    assert not lab_wide.duplicated(subset=["stay_id", "charttime"]).any()

    stay1_rows = lab_wide[lab_wide["stay_id"] == MULTI_STAY_1]
    stay2_rows = lab_wide[lab_wide["stay_id"] == MULTI_STAY_2]
    assert len(stay1_rows) == 1
    assert abs(stay1_rows["lactate"].iloc[0] - 1.5) < 1e-9
    assert len(stay2_rows) == 1
    assert abs(stay2_rows["lactate"].iloc[0] - 3.0) < 1e-9


def test_extract_keeps_stays_separate_and_no_duplication(multi_stay_raw):
    ml = extract(multi_stay_raw)
    assert set(ml["stay_id"].unique()) == {MULTI_STAY_1, MULTI_STAY_2}
    assert (ml["subject_id"] == MULTI_SUBJECT_ID).all()
    assert not ml.duplicated(subset=["stay_id", "charttime"]).any()
    # each stay only carries rows from its own time window: 3 distinct
    # heart_rate timestamps + 1 distinct lactate timestamp (none coincide) = 4
    n_rows_stay1 = (ml["stay_id"] == MULTI_STAY_1).sum()
    n_rows_stay2 = (ml["stay_id"] == MULTI_STAY_2).sum()
    assert n_rows_stay1 == 4
    assert n_rows_stay2 == 4


def test_multi_stay_patient_never_split_across_train_val_test(multi_stay_raw, small_ml_dataset):
    ml_multi = extract(multi_stay_raw)
    combined = pd.concat([small_ml_dataset, ml_multi], ignore_index=True)
    train_df, val_df, test_df = split_by_subject(combined)

    splits_containing_subject = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        subj_rows = split_df[split_df["subject_id"] == MULTI_SUBJECT_ID]
        if len(subj_rows):
            splits_containing_subject.append(name)
            stay_ids_here = set(subj_rows["stay_id"].unique())
            assert stay_ids_here == {MULTI_STAY_1, MULTI_STAY_2}, (
                f"multi-stay patient's stays were split across different dataset "
                f"splits within '{name}': only found {stay_ids_here}, expected both stays together"
            )

    assert len(splits_containing_subject) == 1, (
        f"multi-stay patient's rows appeared in more than one split "
        f"({splits_containing_subject}) — this is exactly the patient-level "
        f"leakage the split-by-subject_id logic is supposed to prevent"
    )
