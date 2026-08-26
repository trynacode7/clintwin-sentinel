"""
biogears_inference.py — BioGears Inference Pipeline
===================================================
Converts BioGears physiological output into the exact 46-feature
representation and executes real BiLSTM inference to extract deterioration
risk and the 256-dimensional learned latent state trajectory.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch

import config
from backend.biogears_adapter import BioGearsFeatureTransformer, BIOGEARS_INPUT_WINDOW
from backend.lstm_model import ICUBiLSTM

logger = logging.getLogger("icu_digital_twin")

BIOGEARS_DEFAULT_INPUT = Path(config.PROJECT_ROOT) / "biogears_runs" / "stable_6hour_results.csvResults.csv"


def load_biogears_csv(
    path: str | Path,
    patient_id: str = "biogears_patient_001",
    lab_mode: str = "calibrated_prior",
) -> pd.DataFrame:
    """Load and standardize BioGears CSV via BioGearsFeatureTransformer."""
    transformer = BioGearsFeatureTransformer()
    return transformer.load_and_standardize_csv(
        path, patient_id=patient_id, lab_mode=lab_mode
    )


def resample_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Apply resampling and missingness indicators."""
    transformer = BioGearsFeatureTransformer()
    return transformer.resample_time_series(df)


def prepare_features(
    df: pd.DataFrame,
    scaler_path: str | Path = config.SCALER_PATH,
) -> pd.DataFrame:
    """Build standardized and engineered feature representation."""
    transformer = BioGearsFeatureTransformer(scaler_path=scaler_path)
    return transformer.engineer_and_scale(df)


def build_model_input(
    df: pd.DataFrame,
    feature_cols: List[str],
    input_window: int = BIOGEARS_INPUT_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the model input tensor of shape (input_window, 46)."""
    transformer = BioGearsFeatureTransformer(feature_cols=feature_cols)
    return transformer.extract_model_tensor(df, input_window=input_window)


def prepare_biogears_input(
    input_path: str | Path = BIOGEARS_DEFAULT_INPUT,
    scaler_path: str | Path = config.SCALER_PATH,
    feature_cols: Optional[List[str]] = None,
    patient_id: str = "biogears_patient_001",
    lab_mode: str = "calibrated_prior",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Complete BioGears -> Model Input pipeline.
    """
    if feature_cols is None:
        checkpoint = torch.load(
            getattr(config, "BIOGEARS_LSTM_MODEL_PATH", config.LSTM_MODEL_PATH),
            map_location="cpu",
            weights_only=False,
        )
        feature_cols = checkpoint["feature_cols"]

    transformer = BioGearsFeatureTransformer(
        scaler_path=scaler_path,
        feature_cols=feature_cols,
    )
    return transformer.transform_csv(
        csv_path=input_path,
        patient_id=patient_id,
        lab_mode=lab_mode,
    )


def _instantiate_lstm_from_checkpoint(checkpoint: dict) -> ICUBiLSTM:
    """Instantiate and load ICUBiLSTM matching checkpoint tensor dimensions."""
    input_size = checkpoint["input_size"]
    # Infer hidden_size and num_layers from state_dict shapes
    state_dict = checkpoint["model_state_dict"]
    if "input_proj.0.weight" in state_dict:
        hidden_size = state_dict["input_proj.0.weight"].shape[0]
    else:
        hidden_size = 128

    # Count LSTM layers
    layer_indices = [
        int(k.split("_l")[1].split("_")[0])
        for k in state_dict.keys()
        if "lstm.weight_ih_l" in k
    ]
    num_layers = max(layer_indices) + 1 if layer_indices else 2

    model = ICUBiLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_lstm_inference(
    X: np.ndarray,
    model_path: str | Path = getattr(config, "BIOGEARS_LSTM_MODEL_PATH", config.LSTM_MODEL_PATH),
) -> Dict[str, Any]:
    """
    Run the trained BiLSTM on a prepared (72, 46) BioGears feature window.

    Returns the raw unmanipulated model risk and 256-D latent state.
    """
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    model = _instantiate_lstm_from_checkpoint(checkpoint)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.ones((1, X.shape[0]), dtype=torch.float32)

    with torch.no_grad():
        risk, z_t = model(X_tensor, mask_tensor)

    raw_risk = float(risk.item())

    return {
        "risk_probability": raw_risk,
        "risk_percentage": raw_risk * 100.0,
        "risk_class": "HIGH" if raw_risk >= 0.5 else "LOW",
        "latent_state": z_t.squeeze(0).cpu().numpy().tolist(),
        "latent_dimension": int(z_t.shape[-1]),
        "input_shape": list(X.shape),
    }


def run_lstm_trajectory(
    X: np.ndarray,
    model_path: str | Path = getattr(config, "BIOGEARS_LSTM_MODEL_PATH", config.LSTM_MODEL_PATH),
) -> Dict[str, Any]:
    """
    Extract the step-by-step 256-D latent state evolution over time.
    """
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    model = _instantiate_lstm_from_checkpoint(checkpoint)

    trajectory = []

    with torch.no_grad():
        for t in range(X.shape[0]):
            prefix = X[: t + 1]
            X_tensor = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.ones((1, prefix.shape[0]), dtype=torch.float32)

            risk_t, z_t = model(X_tensor, mask_tensor)

            trajectory.append(
                {
                    "t_idx": t,
                    "risk_score": float(risk_t.item()),
                    "latent_state": z_t.squeeze(0).cpu().numpy().tolist(),
                }
            )

    return {
        "latent_trajectory": trajectory,
        "latent_dimension": (
            len(trajectory[-1]["latent_state"]) if trajectory else 0
        ),
        "timesteps": len(trajectory),
    }


if __name__ == "__main__":
    checkpoint = torch.load(
        getattr(config, "BIOGEARS_LSTM_MODEL_PATH", config.LSTM_MODEL_PATH),
        map_location="cpu",
        weights_only=False,
    )

    X, mask, df = prepare_biogears_input(
        input_path=BIOGEARS_DEFAULT_INPUT,
        feature_cols=checkpoint["feature_cols"],
    )

    print("=" * 45)
    print("BioGears Feature Preparation Output")
    print("=" * 45)
    print(f"Input shape: {X.shape} (Expected: (72, 46))")
    print(f"Mask shape:  {mask.shape}")
    print(f"NaN count:   {np.isnan(X).sum()}")

    result = run_lstm_inference(X)
    print("\n" * 1 + "=" * 45)
    print("BiLSTM Raw Inference Output")
    print("=" * 45)
    print(f"Raw risk probability: {result['risk_probability']:.6f}")
    print(f"Risk percentage:      {result['risk_percentage']:.2f}%")
    print(f"Risk classification:  {result['risk_class']}")
    print(f"Latent dimension:     {result['latent_dimension']}")
