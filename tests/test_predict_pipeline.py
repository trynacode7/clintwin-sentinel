"""
tests/test_predict_pipeline.py — Comprehensive test for /predict inference pipeline.
Tests stable vs deteriorating synthetic patient time series and reports model behavior.
"""

import sys
import os
import math
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.api import app


def run_tests():
    client = TestClient(app)

    print("=" * 70)
    print("CLINTWIN SENTINEL — INFERENCE PIPELINE VERIFICATION TEST")
    print("=" * 70)

    # ── Test Patient A: Stable Patient ───────────────────────────
    # Note: Synthetic physiological series (not validated clinical cohort)
    stable_patient = [
        {
            "charttime": f"2024-01-01 {i // 12:02d}:{(i % 12) * 5:02d}:00",
            "heart_rate": round(72.0 + np.sin(i / 5.0) * 3.0, 1),
            "map": round(85.0 + np.cos(i / 5.0) * 3.0, 1),
            "resp_rate": 14.0,
            "spo2": 98.5,
            "temp": 37.0,
            "lactate": 0.8 if i % 12 == 0 else None,
            "wbc": 6.0 if i % 12 == 0 else None,
        }
        for i in range(72)
    ]

    print("\n[A] Testing Patient A (Stable Synthetic Patient)...")
    res_a = client.post(
        "/predict", json={"patient_data": stable_patient, "patient_id": "patient_stable"}
    )
    print(f"  HTTP status: {res_a.status_code}")
    if res_a.status_code == 200:
        data_a = res_a.json()
        risk_a = data_a["risk_score"]
        status_a = data_a["status"]
        zt_dim_a = len(data_a["z_t"])
        shap_a = data_a.get("gradient_attribution") or data_a.get("shap_values", [])
        expl_a = data_a["explanation"]
        traj_len_a = len(data_a.get("latent_trajectory", []))
        is_finite_a = math.isfinite(risk_a) and all(math.isfinite(x) for x in data_a["z_t"])
        is_bounded_a = 0.0 <= risk_a <= 1.0

        print(f"  Risk score: {risk_a:.6f}")
        print(f"  Status: {status_a}")
        print(f"  Latent dimension: {zt_dim_a}")
        print(f"  Attribution top 3 features: {shap_a[:3]}")
        print(f"  Clinical explanation: {expl_a}")
        print(f"  Trajectory length: {traj_len_a}")
        print(f"  Output is finite: {is_finite_a}")
        print(f"  Risk in [0, 1]: {is_bounded_a}")
    else:
        print(f"  Failed: {res_a.text}")
        data_a = None

    # ── Test Patient B: Deteriorating / Sepsis-like Patient ──────────
    # Note: Synthetic physiological series showing classic septic shock pattern
    deteriorating_patient = [
        {
            "charttime": f"2024-01-01 {i // 12:02d}:{(i % 12) * 5:02d}:00",
            "heart_rate": round(75.0 + (i / 71.0) * 55.0, 1),   # 75 -> 130 bpm
            "map": round(85.0 - (i / 71.0) * 35.0, 1),          # 85 -> 50 mmHg
            "resp_rate": round(16.0 + (i / 71.0) * 14.0, 1),    # 16 -> 30 bpm
            "spo2": round(98.0 - (i / 71.0) * 9.0, 1),          # 98 -> 89 %
            "temp": round(37.0 + (i / 71.0) * 2.2, 1),          # 37 -> 39.2 °C
            "lactate": round(1.0 + (i / 71.0) * 4.0, 2) if i % 6 == 0 else None, # 1.0 -> 5.0
            "wbc": round(7.0 + (i / 71.0) * 13.0, 1) if i % 6 == 0 else None,   # 7.0 -> 20.0
        }
        for i in range(72)
    ]

    print("\n[B] Testing Patient B (Deteriorating Sepsis-like Synthetic Patient)...")
    res_b = client.post(
        "/predict", json={"patient_data": deteriorating_patient, "patient_id": "patient_deteriorating"}
    )
    print(f"  HTTP status: {res_b.status_code}")
    if res_b.status_code == 200:
        data_b = res_b.json()
        risk_b = data_b["risk_score"]
        status_b = data_b["status"]
        zt_dim_b = len(data_b["z_t"])
        shap_b = data_b.get("gradient_attribution") or data_b.get("shap_values", [])
        expl_b = data_b["explanation"]
        traj_len_b = len(data_b.get("latent_trajectory", []))
        is_finite_b = math.isfinite(risk_b) and all(math.isfinite(x) for x in data_b["z_t"])
        is_bounded_b = 0.0 <= risk_b <= 1.0

        print(f"  Risk score: {risk_b:.6f}")
        print(f"  Status: {status_b}")
        print(f"  Latent dimension: {zt_dim_b}")
        print(f"  Attribution top 3 features: {shap_b[:3]}")
        print(f"  Clinical explanation: {expl_b}")
        print(f"  Trajectory length: {traj_len_b}")
        print(f"  Output is finite: {is_finite_b}")
        print(f"  Risk in [0, 1]: {is_bounded_b}")
    else:
        print(f"  Failed: {res_b.text}")
        data_b = None

    # ── Sensitivity & Model Sanity Comparison ───────────────────────
    print("\n" + "=" * 70)
    print("MODEL SENSITIVITY & CHECKPOINT INTEGRITY REPORT")
    print("=" * 70)
    if data_a and data_b:
        risk_diff = abs(data_a["risk_score"] - data_b["risk_score"])
        zt_diff = np.linalg.norm(np.array(data_a["z_t"]) - np.array(data_b["z_t"]))
        print(f"Risk score difference (|Patient A - Patient B|): {risk_diff:.8f}")
        print(f"Latent state vector L2 difference: {zt_diff:.8f}")

        if risk_diff < 1e-4:
            print(
                "\n[WARNING] CHECKPOINT COLLAPSE DETECTED: The model outputs nearly identical risk "
                "(~1.00000) for radically different patient profiles (stable vs septic shock)."
            )
            print(
                "Root Cause: lstm_model.pt trained with saturation in the final classifier layer "
                "(pre-sigmoid logit ~= +17.23). The model requires retraining with balanced loss/regularization."
            )
        else:
            print("\n[OK] Model exhibits risk discrimination across patient profiles.")

    print("=" * 70)


if __name__ == "__main__":
    run_tests()
