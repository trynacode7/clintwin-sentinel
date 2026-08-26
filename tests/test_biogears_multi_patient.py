"""
test_biogears_multi_patient.py — Multi-Patient End-to-End Integration Tests
===========================================================================
Proves that multiple distinct patient configurations can independently
proceed through:
  patient configuration -> BioGears CSV -> BioGearsFeatureTransformer ->
  (1, 72, 46) tensor -> BiLSTM inference -> Risk & Latent State extraction.
"""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backend.biogears_adapter import BioGearsFeatureTransformer
from backend.biogears_inference import run_lstm_inference

RUN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "biogears_runs",
)
PATIENT_1_CSV = os.path.join(RUN_DIR, "stable_6hour_results.csvResults.csv")
PATIENT_2_CSV = os.path.join(RUN_DIR, "stable_6hour_saline_interventionResults.csv")


@pytest.fixture
def checkpoint_feature_cols():
    ckpt = torch.load(
        getattr(config, "BIOGEARS_LSTM_MODEL_PATH", config.LSTM_MODEL_PATH),
        map_location="cpu",
        weights_only=False,
    )
    return ckpt["feature_cols"]


def test_multi_patient_independent_transformation(checkpoint_feature_cols):
    """
    Verify that two distinct simulation runs (baseline female vs saline-intervened female)
    independently transform into compatible (72, 46) model input tensors.
    """
    assert os.path.exists(PATIENT_1_CSV), f"Patient 1 CSV not found: {PATIENT_1_CSV}"
    assert os.path.exists(PATIENT_2_CSV), f"Patient 2 CSV not found: {PATIENT_2_CSV}"

    transformer = BioGearsFeatureTransformer(feature_cols=checkpoint_feature_cols)

    # Patient 1: Baseline run
    X1, mask1, df1 = transformer.transform_csv(
        PATIENT_1_CSV, patient_id="patient_female_baseline"
    )

    # Patient 2: Saline intervention run
    X2, mask2, df2 = transformer.transform_csv(
        PATIENT_2_CSV, patient_id="patient_female_saline"
    )

    # Shape and validity checks
    assert X1.shape == (72, 46)
    assert X2.shape == (72, 46)
    assert not np.isnan(X1).any(), "Patient 1 tensor contains NaNs"
    assert not np.isnan(X2).any(), "Patient 2 tensor contains NaNs"

    # Verify that the two feature vectors are distinct due to saline intervention
    assert not np.allclose(X1, X2), "Feature vectors should reflect physiological differences"

    # Patient ID isolation
    assert (df1["subject_id"] == "patient_female_baseline").all()
    assert (df2["subject_id"] == "patient_female_saline").all()


def test_multi_patient_independent_bilstm_inference(checkpoint_feature_cols):
    """
    Verify that two distinct patient tensors pass through the trained BiLSTM
    and yield distinct 256-D latent states.
    """
    transformer = BioGearsFeatureTransformer(feature_cols=checkpoint_feature_cols)

    X1, _, _ = transformer.transform_csv(
        PATIENT_1_CSV, patient_id="patient_female_baseline"
    )
    X2, _, _ = transformer.transform_csv(
        PATIENT_2_CSV, patient_id="patient_female_saline"
    )

    pred1 = run_lstm_inference(X1)
    pred2 = run_lstm_inference(X2)

    assert "risk_probability" in pred1
    assert "risk_probability" in pred2
    assert "latent_state" in pred1
    assert "latent_state" in pred2
    assert len(pred1["latent_state"]) == 256
    assert len(pred2["latent_state"]) == 256

    z1 = np.array(pred1["latent_state"])
    z2 = np.array(pred2["latent_state"])

    # Latent representations must show measurable Euclidean distance
    latent_distance = np.linalg.norm(z1 - z2)
    assert latent_distance > 0.0, "Latent states should differ between baseline and intervened patient"
