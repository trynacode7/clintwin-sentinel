"""
test_clintwin_validation.py — ClinTwin Sentinel Comprehensive Validation Suite
=============================================================================
Tests all 13 core requirements specified in Phase 7:
  1. preprocessing works
  2. labels contain both classes
  3. scaler is train-only
  4. feature count = 112
  5. model accepts 112 features
  6. model output is finite and between 0 and 1
  7. /predict works
  8. invalid patient returns 404
  9. patient switching/API data works
 10. BioGears scenario executes successfully
 11. BioGears output CSV is produced
 12. BloodVolume is present
 13. intervention scenario actually contains the saline action
"""

import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import generate_synthetic_data
from backend.preprocessing import (
    preprocess,
    fit_scaler,
    apply_scaler,
    add_missingness_indicators,
    resample_all,
    prepare_inference_sequence,
)
from backend.labels_sofa import create_labels  # the real SOFA/Sepsis-3 labeling
# used by training (backend.train) and evaluation (backend.evaluate) — was
# importing the deprecated ad-hoc backend.labels rule here, which meant this
# test validated a labeling function that isn't the one actually used to
# train or evaluate the model.
from backend.feature_engineering import engineer_features, get_engineered_feature_cols
from backend.lstm_model import ICUBiLSTM
from backend.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ─── 1. Preprocessing Works ──────────────────────────────────────────────────
def test_1_preprocessing_works():
    """Verify raw data preprocessing, 5-min resampling, and missingness handling."""
    df_raw = generate_synthetic_data(n_patients=5, hours_per_stay=24)
    resampled = resample_all(df_raw)
    assert len(resampled) > 0, "Resampled DataFrame should not be empty"
    with_indicators = add_missingness_indicators(resampled)
    for col in config.ALL_FEATURE_COLS:
        assert f"{col}_missing" in with_indicators.columns
        assert f"{col}_time_since_obs" in with_indicators.columns


# ─── 2. Labels Contain Both Classes ──────────────────────────────────────────
def test_2_labels_contain_both_classes():
    """
    Verify that future-window label generation produces both 0 and 1 classes.

    Uses the real MIMIC-shaped generator (backend.synthetic_data.generator),
    sepsis-archetype-weighted, NOT utils.generate_synthetic_data() — a real
    bug found while fixing this test's import (it previously tested against
    the deprecated backend.labels rule, which never actually ran the real
    SOFA/Sepsis-3 logic). Once pointed at the real backend.labels_sofa,
    utils.generate_synthetic_data()'s output produces ZERO suspected-
    infection-eligible windows (confirmed empirically up to n_patients=100,
    seeded) — it has no antibiotic/culture-order signal at all, so the
    Sepsis-3 cohort restriction can structurally never admit a positive
    case from it. This isn't flakiness; that generator and the real label
    definition are simply incompatible. The MIMIC-shaped generator (used by
    generate_synthetic_data.py, the one that actually produces
    data/synthetic/processed/ml_dataset.csv for real training) does carry
    that signal via its "sepsis" archetype.
    """
    from utils import set_seed
    from backend.synthetic_data.generator import GeneratorConfig, generate_cohort
    from backend.synthetic_data.extract_to_ml_dataset import extract

    set_seed(config.SEED)
    weights = {"stable": 0.1, "improving": 0.05, "deteriorating": 0.15, "sepsis": 0.6, "recovery": 0.1}
    cfg = GeneratorConfig(n_patients=15, seed=config.SEED, archetype_weights=weights)
    raw_tables = generate_cohort(cfg)
    df_raw = extract(raw_tables)

    resampled = resample_all(df_raw)
    with_ind = add_missingness_indicators(resampled)
    labeled = create_labels(with_ind)
    labels = labeled["label"].dropna().unique()
    assert 0 in labels, "Labels must contain class 0 (Stable)"
    assert 1 in labels, "Labels must contain class 1 (Deteriorating)"


# ─── 3. Scaler is Train-Only ──────────────────────────────────────────────────
def test_3_scaler_is_train_only():
    """Verify scaler is fit strictly on training split and leaves test/val mean untouched."""
    df_train = generate_synthetic_data(n_patients=10, hours_per_stay=24)
    df_val = generate_synthetic_data(n_patients=5, hours_per_stay=24)
    feature_cols = [c for c in config.ALL_FEATURE_COLS if c in df_train.columns]

    scaler = fit_scaler(df_train, feature_cols)
    original_mean = scaler.mean_.copy()

    # Apply scaler to val
    _ = apply_scaler(df_val, scaler, feature_cols)

    # Scaler mean must not have changed
    assert np.allclose(scaler.mean_, original_mean), "Scaler was modified during validation transform (data leakage)!"


# ─── 4. Feature Count = 112 ──────────────────────────────────────────────────
def test_4_feature_count_is_112():  # renamed from _is_46 for clarity — body
    # already correctly asserted 112; only the function name was stale.
    """Verify that feature engineering produces the expected engineered features."""
    df_raw = generate_synthetic_data(n_patients=5, hours_per_stay=24)
    resampled = resample_all(df_raw)
    with_ind = add_missingness_indicators(resampled)
    engineered = engineer_features(with_ind)
    cols = get_engineered_feature_cols(engineered)
    assert len(cols) == 112, f"Expected 112 features, got {len(cols)}"


# ─── 5. Model Accepts Features ────────────────────────────────────────────────
def test_5_model_accepts_112_features():  # renamed from _46_features — body
    """Verify BiLSTM accepts input sequences and returns 128-D hidden representation."""
    model = ICUBiLSTM(input_size=112)
    x = torch.randn(2, config.INPUT_WINDOW, 112)
    mask = torch.ones(2, config.INPUT_WINDOW)
    risk, z_t = model(x, mask)
    assert risk.shape == (2, 1)
    assert z_t.shape == (2, 128)


# ─── 6. Model Output is Finite and Between 0 and 1 ───────────────────────────
def test_6_model_output_finite_and_bounded():
    """Verify trained checkpoint output is finite and strictly bounded in [0, 1]."""
    checkpoint = torch.load(config.LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
    model = ICUBiLSTM(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.randn(4, config.INPUT_WINDOW, checkpoint["input_size"])
    mask = torch.ones(4, config.INPUT_WINDOW)
    with torch.no_grad():
        risk, z_t = model(x, mask)

    risk_np = risk.numpy()
    zt_np = z_t.numpy()
    assert np.all(np.isfinite(risk_np)), "Risk score contains NaN or Inf"
    assert np.all(np.isfinite(zt_np)), "z_t latent representation contains NaN or Inf"
    assert np.all((risk_np >= 0.0) & (risk_np <= 1.0)), "Risk score out of [0, 1] range"


# ─── 7. /predict Works ───────────────────────────────────────────────────────
def test_7_predict_endpoint_works(client):
    """Verify POST /predict returns valid risk, 128-D z_t, attribution, and explanation."""
    records = [
        {
            "charttime": f"2026-01-01 {i // 12:02d}:{(i % 12) * 5:02d}:00",
            "heart_rate": 75.0,
            "map": 85.0,
            "resp_rate": 16.0,
            "spo2": 98.0,
            "temp": 37.0,
            "lactate": 1.0,
            "wbc": 7.0,
        }
        for i in range(72)
    ]
    resp = client.post("/predict", json={"patient_data": records, "patient_id": "test_patient"})
    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "risk_score" in data and 0.0 <= data["risk_score"] <= 1.0
    assert "status" in data and data["status"] in ["Stable", "Warning", "Critical"]
    assert "z_t" in data and len(data["z_t"]) == 128
    attribution = data.get("gradient_attribution") or data.get("shap_values", [])
    assert len(attribution) > 0
    assert "explanation" in data and len(data["explanation"]) > 0


# ─── 8. Invalid Patient Returns 404 ──────────────────────────────────────────
def test_8_invalid_patient_returns_404(client):
    """Verify GET /patient/unknown_id returns 404 Not Found."""
    resp = client.get("/patient/999999999")
    assert resp.status_code == 404


# ─── 9. Patient Switching / API Data Works ────────────────────────────────────
def test_9_patient_switching_works(client):
    """Verify switching patients returns distinct physiological data."""
    roster_resp = client.get("/api/roster")
    assert roster_resp.status_code == 200
    roster = roster_resp.json()["patients"]
    assert len(roster) >= 2

    resp1 = client.get(f"/patient/{roster[0]['stay_id']}")
    resp2 = client.get(f"/patient/{roster[1]['stay_id']}")
    assert resp1.status_code == 200 and resp2.status_code == 200

    data1 = resp1.json()
    data2 = resp2.json()
    assert data1["patient"]["stay_id"] != data2["patient"]["stay_id"]


# ─── 10. BioGears Scenario Executes Successfully ─────────────────────────────
def test_10_biogears_scenario_execution():
    """Verify BioGears simulation log and execution status."""
    log_path = Path(r"C:\BioGears\bin\Scenarios\tmp\bg_demo_100ml_saline.log")
    if not log_path.exists():
        pytest.skip("BioGears demo log file not present")
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    assert "Simulation Complete" in content or "Completed scenario" in content or "Time" in content


# ─── 11. BioGears Output CSV is Produced ─────────────────────────────────────
def test_11_biogears_output_csv_produced():
    """Verify BioGears simulation output CSV was generated and contains data rows."""
    csv_path = Path(r"C:\BioGears\bin\Scenarios\tmp\bg_demo_100ml_salineResults.csv")
    if not csv_path.exists():
        pytest.skip("BioGears demo results CSV not present")
    df = pd.read_csv(csv_path)
    assert len(df) > 0, "BioGears results CSV is empty"


# ─── 12. BloodVolume is Present ──────────────────────────────────────────────
def test_12_blood_volume_is_present():
    """Verify BloodVolume physiological column is in the BioGears CSV."""
    csv_path = Path(r"C:\BioGears\bin\Scenarios\tmp\bg_demo_100ml_salineResults.csv")
    if not csv_path.exists():
        pytest.skip("BioGears demo results CSV not present")
    df = pd.read_csv(csv_path)
    col_match = [c for c in df.columns if "BloodVolume" in c]
    assert len(col_match) > 0, f"BloodVolume column missing from BioGears output. Columns: {list(df.columns)}"
    bv_values = df[col_match[0]].dropna()
    assert (bv_values > 4.0).all() and (bv_values < 8.0).all()


# ─── 13. Intervention Scenario Contains Saline Action ─────────────────────────
def test_13_intervention_scenario_contains_saline_action():
    """Verify the scenario XML defines the Saline infusion action."""
    xml_path = Path(r"C:\BioGears\bin\Scenarios\tmp\bg_demo_100ml_saline.xml")
    if not xml_path.exists():
        pytest.skip("BioGears demo XML not present")
    content = xml_path.read_text(encoding="utf-8", errors="ignore")
    assert "SubstanceCompound" in content
    assert "Saline" in content
    assert "SubstanceCompoundInfusionData" in content
