"""
test_sofa_scoring.py — direct unit tests of labels_sofa.py's SOFA subscore
functions, in particular the Fix 1 cardiovascular MAP-threshold correction
(65 mmHg -> 70 mmHg, see labels_sofa.py and config.py for the audit finding).
"""
import numpy as np
import pandas as pd

import config
from backend.labels_sofa import (
    _cardiovascular_subscore, _respiratory_subscore, _coagulation_subscore,
    _liver_subscore, _cns_subscore, _renal_subscore,
)


def test_sofa_cardiovascular_threshold_is_70_not_65():
    """Fix 1 regression guard: the config constant itself must be 70, and must
    never be aliased back to the legacy 65 mmHg labels.py threshold."""
    assert config.SOFA_CARDIOVASCULAR_MAP_THRESHOLD == 70.0
    assert config.SOFA_CARDIOVASCULAR_MAP_THRESHOLD != config.MAP_THRESHOLD


def test_map_at_or_above_70_gets_no_map_point():
    """MAP >= 70 (no vasopressor) -> cardiovascular subscore 0."""
    map_ = pd.Series([70.0, 75.0, 90.0])
    dose = pd.Series([0.0, 0.0, 0.0])
    score = _cardiovascular_subscore(map_, dose)
    assert (score == 0).all()


def test_map_just_below_70_gets_the_map_point():
    """MAP < 70 (no vasopressor) -> cardiovascular subscore 1. This is exactly
    the boundary the audit flagged: MAP=68 must score, and previously (with
    the old 65 threshold) it would NOT have."""
    map_ = pd.Series([69.9, 68.0, 65.0, 50.0])
    dose = pd.Series([0.0, 0.0, 0.0, 0.0])
    score = _cardiovascular_subscore(map_, dose)
    assert (score == 1).all()


def test_map_65_to_70_no_longer_silently_scores_zero():
    """The specific band [65, 70) that the pre-fix (threshold=65) implementation
    would have scored 0 for must now score 1 — this is the concrete behavioral
    change Fix 1 makes."""
    map_ = pd.Series([65.0, 66.5, 69.0])
    dose = pd.Series([0.0, 0.0, 0.0])
    score = _cardiovascular_subscore(map_, dose)
    assert (score == 1).all(), "MAP in [65,70) must score 1 under the corrected SOFA threshold"


def test_vasopressor_dose_still_overrides_map_scoring():
    """Vasopressor rules (score 3/4) are independent of the MAP fix and unchanged."""
    map_ = pd.Series([80.0, 80.0])   # MAP normal, would otherwise score 0
    dose = pd.Series([0.05, 0.2])    # low-dose -> 3, higher-dose -> 4
    score = _cardiovascular_subscore(map_, dose)
    assert list(score) == [3.0, 4.0]


def test_cardiovascular_nan_when_both_inputs_missing():
    map_ = pd.Series([np.nan])
    dose = pd.Series([np.nan])
    score = _cardiovascular_subscore(map_, dose)
    assert np.isnan(score.iloc[0])


# ── spot-checks of the other subscores (already-correct thresholds; guard
#    against accidental regressions while editing this module) ─────────────

def test_respiratory_subscore_thresholds():
    spo2 = pd.Series([99.0, 95.0, 90.0, 80.0])
    fio2 = pd.Series([0.21, 0.30, 0.40, 0.60])
    vent = pd.Series([0.0, 0.0, 1.0, 1.0])
    score = _respiratory_subscore(spo2, fio2, vent)
    # sf ratios: 99/0.21=471(>400->0), 95/0.30=317(<400,>=315 boundary->1),
    # 90/0.40=225(<235,vent->3), 80/0.60=133(<150,vent->4)
    assert score.iloc[0] == 0
    assert score.iloc[2] == 3
    assert score.iloc[3] == 4


def test_coagulation_subscore_thresholds():
    platelets = pd.Series([200.0, 120.0, 75.0, 30.0, 15.0])
    score = _coagulation_subscore(platelets)
    assert list(score) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_liver_subscore_thresholds():
    bilirubin = pd.Series([0.8, 1.5, 3.0, 8.0, 13.0])
    score = _liver_subscore(bilirubin)
    assert list(score) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_cns_subscore_thresholds():
    gcs = pd.Series([15.0, 13.0, 11.0, 7.0, 4.0])
    score = _cns_subscore(gcs)
    assert list(score) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_renal_subscore_uses_worse_of_creatinine_and_urine():
    creatinine = pd.Series([1.0, 1.5])   # score 0, 1
    urine = pd.Series([5.0, 60.0])       # score 4 (<10), 0
    score = _renal_subscore(creatinine, urine)
    assert list(score) == [4.0, 1.0]  # elementwise max
