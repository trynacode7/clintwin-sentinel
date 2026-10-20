"""
test_biogears_feature_transformer.py — Unit Tests for BioGears Feature Pipeline
================================================================================
Verifies:
  1. Mandatory SpO2 fraction-to-percentage conversion (0.979 -> 97.9).
  2. Missing lab modes: 'strict_missing' vs 'calibrated_prior'.
  3. Correct output tensor dimensions (72, 46) and zero NaNs.
  4. Dynamic patient_id assignment without hardcoding.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backend.biogears_adapter import BioGearsFeatureTransformer
from backend.biogears_inference import run_lstm_inference

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "biogears_runs",
    "stable_6hour_results.csvResults.csv",
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(CSV_PATH),
    reason="Requires generated BioGears CSV stable_6hour_results.csvResults.csv",
)


@pytest.fixture
def checkpoint_feature_cols():
    ckpt = torch.load(
        getattr(config, "BIOGEARS_LSTM_MODEL_PATH", config.LSTM_MODEL_PATH),
        map_location="cpu",
        weights_only=False,
    )
    return ckpt["feature_cols"]


def test_spo2_unit_scaling():
    """Verify that unitless fractional SpO2 is scaled to percentage [0, 100]."""
    transformer = BioGearsFeatureTransformer()
    df = transformer.load_and_standardize_csv(CSV_PATH)
    assert df["spo2"].max() > 1.0, f"Expected SpO2 > 1.0 (percentage), got {df['spo2'].max()}"
    assert 90.0 <= df["spo2"].mean() <= 100.0


def test_dynamic_patient_id():
    """Verify dynamic patient and stay IDs are assigned properly."""
    custom_id = "patient_custom_42"
    transformer = BioGearsFeatureTransformer()
    df = transformer.load_and_standardize_csv(CSV_PATH, patient_id=custom_id)
    assert (df["subject_id"] == custom_id).all()
    assert (df["stay_id"] == f"{custom_id}_stay").all()


def test_calibrated_prior_vs_strict_missing_modes(checkpoint_feature_cols):
    """Verify both lab handling modes produce valid (72, 46) tensors."""
    transformer = BioGearsFeatureTransformer(feature_cols=checkpoint_feature_cols)

    # Mode 1: Calibrated Prior (normal values at t=0 with LOCF)
    X_prior, mask_prior, _ = transformer.transform_csv(
        CSV_PATH, patient_id="patient_prior", lab_mode="calibrated_prior"
    )
    assert X_prior.shape == (72, 46)
    assert not np.isnan(X_prior).any()

    # Mode 2: Strict Missing (unmeasured throughout)
    X_miss, mask_miss, _ = transformer.transform_csv(
        CSV_PATH, patient_id="patient_miss", lab_mode="strict_missing"
    )
    assert X_miss.shape == (72, 46)
    assert not np.isnan(X_miss).any()

    # Ensure representations differ meaningfully
    assert not np.allclose(X_prior, X_miss)


def test_lstm_inference_raw_outputs(checkpoint_feature_cols):
    """Verify that ICUBiLSTM produces valid raw risk and 256-D latent states."""
    transformer = BioGearsFeatureTransformer(feature_cols=checkpoint_feature_cols)
    X, mask, _ = transformer.transform_csv(CSV_PATH, lab_mode="calibrated_prior")

    result = run_lstm_inference(X)
    assert "risk_probability" in result
    assert "latent_state" in result
    assert result["latent_dimension"] == 256
    assert len(result["latent_state"]) == 256
    assert 0.0 <= result["risk_probability"] <= 1.0
