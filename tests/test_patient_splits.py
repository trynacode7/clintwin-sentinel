"""
test_patient_splits.py — Phase 1 section 7: patient-level split disjointness.

Proves train/val/test never share a subject_id, using validate.py's
validate_patient_split_disjoint() against the ACTUAL split produced by
preprocessing.split_by_subject() (not a synthetic re-implementation of it).
"""
import os
import tempfile

import pandas as pd

from backend.synthetic_data.validate import validate_patient_split_disjoint
from backend.preprocessing import split_by_subject


def test_split_by_subject_is_disjoint(small_ml_dataset):
    train_df, val_df, test_df = split_by_subject(small_ml_dataset)

    train_ids = set(train_df["subject_id"].unique())
    val_ids = set(val_df["subject_id"].unique())
    test_ids = set(test_df["subject_id"].unique())

    report = validate_patient_split_disjoint(train_ids, val_ids, test_ids)
    assert report.ok, str(report)

    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()


def test_split_covers_every_patient_exactly_once(small_ml_dataset):
    train_df, val_df, test_df = split_by_subject(small_ml_dataset)
    all_ids = set(small_ml_dataset["subject_id"].unique())
    split_ids = (
        set(train_df["subject_id"].unique())
        | set(val_df["subject_id"].unique())
        | set(test_df["subject_id"].unique())
    )
    assert split_ids == all_ids


def test_validate_patient_split_disjoint_detects_a_real_violation():
    """Sanity-check the checker itself: it must actually flag an overlap."""
    report = validate_patient_split_disjoint(
        train_ids=[1, 2, 3], val_ids=[3, 4], test_ids=[5, 6]
    )
    assert not report.ok
    assert any("train" in e and "val" in e for e in report.errors)


def test_split_is_reproducible_given_same_seed(small_ml_dataset):
    train1, val1, test1 = split_by_subject(small_ml_dataset, seed=42)
    train2, val2, test2 = split_by_subject(small_ml_dataset, seed=42)
    assert set(train1["subject_id"]) == set(train2["subject_id"])
    assert set(val1["subject_id"]) == set(val2["subject_id"])
    assert set(test1["subject_id"]) == set(test2["subject_id"])
