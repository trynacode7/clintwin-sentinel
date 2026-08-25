"""
test_temporal_leakage.py — Phase 1 section 8: explicit leakage tests.

Two independent properties are proven here, matching the user's spec
("prediction at time T uses only data <= T, never T+1/T+2/future labels/
future interventions/future labs"):

1. labels_sofa.create_labels_for_stay()'s label at row i is a function ONLY
   of rows in [i, i + FUTURE_WINDOW_END) (row i itself as the baseline, plus
   the [i+FUTURE_WINDOW_START, i+FUTURE_WINDOW_END) future window) — in
   particular row i+1 (immediately after the input window) is proven to be
   EXCLUDED, and nothing beyond i+FUTURE_WINDOW_END is used.

2. dataset.ICUDataset's input window X ending at row i is a pure function of
   rows [i-window_size+1, i] — proven by mutating only rows strictly AFTER a
   cutoff and showing every window ending at-or-before the cutoff is
   byte-identical to the unmutated version.
"""
import numpy as np
import pandas as pd

import config
from backend.labels_sofa import create_labels_for_stay
from backend.dataset import ICUDataset


def _baseline_stay(n: int, stay_id: int = 1) -> pd.DataFrame:
    """A stable, non-infected, non-deteriorating stay of length n (SOFA ~ 0)."""
    times = pd.date_range("2024-01-01", periods=n, freq=config.RESAMPLE_FREQ)
    return pd.DataFrame({
        "subject_id": 1, "hadm_id": 1, "stay_id": stay_id, config.TIME_COL: times,
        "spo2": 98.0, "fio2": 0.21, "ventilated": 0.0,
        "platelets": 250.0, "bilirubin": 0.5,
        "map": 80.0, "vasopressor_dose": 0.0,
        "gcs": 15.0, "creatinine": 0.8, "urine_output_rate": 60.0,
        "suspected_infection": 0,
    })


def _inject_severe_event(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Set row k to a severe, SOFA-total>=16, actively-infected state."""
    df = df.copy()
    df.loc[k, ["spo2", "fio2", "ventilated", "platelets", "bilirubin",
               "map", "vasopressor_dose", "gcs", "creatinine",
               "urine_output_rate", "suspected_infection"]] = [
        80.0, 0.9, 1.0, 10.0, 15.0, 50.0, 0.2, 3.0, 5.0, 5.0, 1
    ]
    return df


def test_label_future_window_bounds_are_exact():
    """
    A single severe+infected event at row k must trigger label[i] == 1 if
    and only if k falls in [i + FUTURE_WINDOW_START, i + FUTURE_WINDOW_END)
    AND the full future window [i, i+FUTURE_WINDOW_END) exists in the stay.
    In particular this proves row i+1 (T+1) is never sufficient on its own
    (FUTURE_WINDOW_START=2 excludes it) and rows at/after i+FUTURE_WINDOW_END
    are never used.
    """
    n = 20
    k = 12
    df = _inject_severe_event(_baseline_stay(n), k)

    result = create_labels_for_stay(df)
    labels = result["label"].values

    start, end = config.FUTURE_WINDOW_START, config.FUTURE_WINDOW_END
    for i in range(n):
        if i + end > n:
            assert np.isnan(labels[i]), f"i={i}: expected NaN (insufficient future data)"
            continue
        expected_eligible = start <= (k - i) < end
        if expected_eligible:
            assert labels[i] == 1.0, f"i={i}: event at k={k} is within [{i+start},{i+end}) — expected label=1"
        else:
            assert labels[i] == 0.0, (
                f"i={i}: event at k={k} is OUTSIDE [{i+start},{i+end}) — "
                f"expected label=0, this would indicate leakage from outside the future window"
            )


def test_label_does_not_use_row_immediately_after_input_window():
    """
    Direct T+1-exclusion check: an event placed exactly one row after the
    input window's last row (k = i+1) must NOT trigger label[i]=1, since
    FUTURE_WINDOW_START=2 excludes T+1 from the eligible future window.
    """
    n = 10
    i = 3
    k = i + 1
    df = _inject_severe_event(_baseline_stay(n), k)
    result = create_labels_for_stay(df)
    assert result["label"].values[i] == 0.0, (
        "label used data at T+1, which is outside the defined future window "
        "[T+FUTURE_WINDOW_START, T+FUTURE_WINDOW_END) — this is a leakage bug"
    )


def test_dataset_input_window_never_reflects_future_rows():
    """
    Mutate a stay's feature values ONLY at rows strictly after a cutoff, and
    confirm every ICUDataset window whose LAST observed row is at-or-before
    the cutoff is byte-identical between the original and mutated versions —
    i.e. the model's input at time T is provably independent of data at T+1
    and beyond.
    """
    n = 20
    cutoff = 10
    window_size = config.INPUT_WINDOW
    feature_cols = list(config.ALL_FEATURE_COLS)

    times = pd.date_range("2024-01-01", periods=n, freq=config.RESAMPLE_FREQ)
    base = {"subject_id": 1, "hadm_id": 1, "stay_id": 1, config.TIME_COL: times, "label": 0.0}
    for col in feature_cols:
        base[col] = np.arange(n, dtype=np.float32)  # value == row index, easy to detect leakage
    df_v1 = pd.DataFrame(base)

    df_v2 = df_v1.copy()
    future_mask = df_v2.index > cutoff
    for col in feature_cols:
        df_v2.loc[future_mask, col] = 9999.0

    ds1 = ICUDataset(df_v1, feature_cols, window_size=window_size, step_size=1)
    ds2 = ICUDataset(df_v2, feature_cols, window_size=window_size, step_size=1)

    assert len(ds1) == len(ds2) and len(ds1) > 0

    n_checked = 0
    for idx in range(len(ds1)):
        # window idx covers rows [idx, idx + window_size - 1] (start=idx, step=1)
        last_row = idx + window_size - 1
        if last_row > cutoff:
            continue
        X1, y1, mask1 = ds1[idx]
        X2, y2, mask2 = ds2[idx]
        assert np.array_equal(X1.numpy(), X2.numpy()), (
            f"window ending at row {last_row} (<= cutoff {cutoff}) differs after mutating "
            f"only rows > {cutoff} — the model input is leaking future data"
        )
        n_checked += 1

    assert n_checked > 0, "test did not actually check any windows — cutoff/window_size mismatch"
