"""
biogears_adapter.py — BioGears Feature Transformation Pipeline
==============================================================
Converts raw high-frequency BioGears scenario CSV output into the exact
46-feature representation expected by the trained ClinTwin models.

Key Behaviors:
  1. Dynamic patient and hospital identifier attribution (no hardcoding).
  2. Mandatory SpO2 unit scaling (fraction [0, 1] -> percentage [0, 100]).
  3. 5-minute interval downsampling and clinical LOCF/interpolation.
  4. Missingness indicators and observation timing feature generation.
  5. Documented 'calibrated_prior' vs 'strict_missing' mode option for unmeasured labs.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import joblib
import numpy as np
import pandas as pd

import config
from backend.preprocessing import (
    apply_scaler,
    resample_stay,
)

logger = logging.getLogger("icu_digital_twin")

BIOGEARS_COLUMNS = {
    "Time(s)": "time_seconds",
    "HeartRate(1/min)": "heart_rate",
    "MeanArterialPressure(mmHg)": "map",
    "SystolicArterialPressure(mmHg)": "systolic_bp",
    "DiastolicArterialPressure(mmHg)": "diastolic_bp",
    "RespirationRate(1/min)": "resp_rate",
    "OxygenSaturation": "spo2",
    "CoreTemperature(degC)": "temp",
    "CardiacOutput(L/min)": "cardiac_output",
    "CentralVenousPressure(mmHg)": "central_venous_pressure",
    "TidalVolume(mL)": "tidal_volume",
}

BASE_FEATURES = [
    "heart_rate",
    "map",
    "resp_rate",
    "spo2",
    "temp",
    "lactate",
    "wbc",
]

# Standard 6-hour observation window at 5-minute intervals
BIOGEARS_INPUT_WINDOW = 72
BIOGEARS_ROLLING_WINDOW = 12  # 12 * 5min = 1 hour


def add_biogears_missingness_indicators(
    df: pd.DataFrame, feature_cols: List[str] = BASE_FEATURES
) -> pd.DataFrame:
    """Add binary missingness indicators and time-since-last-observation."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(float)
            is_obs = ~df[col].isna()
            obs_times = df["charttime"].where(is_obs)
            last_obs_time = obs_times.groupby(df["stay_id"]).ffill()
            time_diff = (df["charttime"] - last_obs_time).dt.total_seconds() / 60.0
            df[f"{col}_time_since_obs"] = time_diff.fillna(0.0)
    return df


def engineer_biogears_features(
    df: pd.DataFrame,
    feature_cols: List[str] = BASE_FEATURES,
    rolling_window: int = BIOGEARS_ROLLING_WINDOW,
) -> pd.DataFrame:
    """Compute temporal, clinical, and time features for the 46-feature contract."""
    df = df.copy()

    # 1. Delta features
    for col in feature_cols:
        if col in df.columns:
            df[f"{col}_delta"] = df.groupby("stay_id")[col].diff().fillna(0.0)

    # 2. Rolling statistics (mean & std)
    for col in feature_cols:
        if col in df.columns:
            df[f"{col}_rolling_mean"] = (
                df.groupby("stay_id")[col]
                .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
            )
            df[f"{col}_rolling_std"] = (
                df.groupby("stay_id")[col]
                .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).std())
                .fillna(0.0)
            )

    # 3. Clinical feature: Shock Index = heart_rate / map
    if "heart_rate" in df.columns and "map" in df.columns:
        df["shock_index"] = df["heart_rate"] / df["map"].replace(0, np.nan)
        df["shock_index"] = df["shock_index"].fillna(0.0)

    # 4. Time features
    if "charttime" in df.columns:
        first_time = df.groupby("stay_id")["charttime"].transform("min")
        df["time_since_admission"] = (
            (df["charttime"] - first_time).dt.total_seconds() / 60.0
        )
        hour = df["charttime"].dt.hour + df["charttime"].dt.minute / 60.0
        df["hour_of_day_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_of_day_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return df


class BioGearsFeatureTransformer:
    """
    Transforms BioGears simulation time-series into the standardized
    46-feature ICU model representation.
    """

    def __init__(
        self,
        scaler_path: str | Path = getattr(config, "BIOGEARS_SCALER_PATH", config.SCALER_PATH),
        feature_cols: Optional[List[str]] = None,
    ):
        self.scaler_path = Path(scaler_path)
        self.feature_cols = feature_cols

    def load_and_standardize_csv(
        self,
        csv_path: str | Path,
        patient_id: str = "biogears_patient_001",
        start_time: Optional[pd.Timestamp] = None,
        lab_mode: str = "calibrated_prior",
        baseline_lactate: float = 1.0,
        baseline_wbc: float = 7.0,
    ) -> pd.DataFrame:
        """
        Load raw BioGears CSV and standardize columns and units.
        """
        df = pd.read_csv(csv_path)

        required = [
            "Time(s)",
            "HeartRate(1/min)",
            "MeanArterialPressure(mmHg)",
            "RespirationRate(1/min)",
            "OxygenSaturation",
            "CoreTemperature(degC)",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"BioGears CSV missing required columns: {missing}")

        # Rename to ClinTwin internal schema
        rename_dict = {k: v for k, v in BIOGEARS_COLUMNS.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # Mandatory SpO2 unit scaling: BioGears outputs fraction 0.979 -> scale to 97.9%
        if "spo2" in df.columns and df["spo2"].max() <= 1.0:
            df["spo2"] = df["spo2"] * 100.0

        # Construct synthetic charttime from elapsed simulation seconds
        base_dt = start_time or pd.Timestamp("2026-01-01 00:00:00")
        df["charttime"] = base_dt + pd.to_timedelta(df["time_seconds"], unit="s")

        # Dynamic patient identification
        df["subject_id"] = patient_id
        df["hadm_id"] = f"{patient_id}_hadm"
        df["stay_id"] = f"{patient_id}_stay"

        # Lab handling per mode specification
        if lab_mode == "calibrated_prior":
            # Normal physiological prior at t=0; forward-fill will preserve it
            df["lactate"] = np.nan
            df["wbc"] = np.nan
            df.loc[0, "lactate"] = baseline_lactate
            df.loc[0, "wbc"] = baseline_wbc
        else:
            # Strict missing mode: unmeasured labs
            df["lactate"] = np.nan
            df["wbc"] = np.nan

        return df.sort_values("charttime").reset_index(drop=True)

    def resample_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to 5-minute intervals and compute missingness indicators."""
        resampled = resample_stay(df, freq="5min")
        resampled = add_biogears_missingness_indicators(resampled, feature_cols=BASE_FEATURES)
        resampled = resampled.dropna(subset=config.VITAL_COLS, how="all")
        return resampled

    def engineer_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted StandardScaler and compute engineered features."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

        scaler = joblib.load(self.scaler_path)
        df_scaled = apply_scaler(df, scaler, BASE_FEATURES)
        df_featured = engineer_biogears_features(df_scaled, feature_cols=BASE_FEATURES)
        return df_featured

    def extract_model_tensor(
        self,
        df: pd.DataFrame,
        input_window: int = BIOGEARS_INPUT_WINDOW,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the (input_window, 46) feature tensor.
        """
        if self.feature_cols is None:
            raise ValueError("feature_cols must be set before extracting model tensor.")

        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_cols}")

        n_steps = len(df)
        features = df[self.feature_cols].to_numpy(dtype=np.float32)

        if n_steps < input_window:
            pad_len = input_window - n_steps
            X = np.zeros((input_window, len(self.feature_cols)), dtype=np.float32)
            X[pad_len:] = features
            mask = np.zeros(input_window, dtype=np.float32)
            mask[pad_len:] = 1.0
        else:
            X = features[-input_window:].copy()
            mask = np.ones(input_window, dtype=np.float32)

        X = np.nan_to_num(X, nan=0.0)

        return X, mask

    def transform_csv(
        self,
        csv_path: str | Path,
        patient_id: str = "biogears_patient_001",
        start_time: Optional[pd.Timestamp] = None,
        lab_mode: str = "calibrated_prior",
        input_window: int = BIOGEARS_INPUT_WINDOW,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Full end-to-end transformation: CSV -> (72, 46) tensor + metadata DataFrame.
        """
        raw = self.load_and_standardize_csv(
            csv_path,
            patient_id=patient_id,
            start_time=start_time,
            lab_mode=lab_mode,
        )
        resampled = self.resample_time_series(raw)
        featured = self.engineer_and_scale(resampled)
        X, mask = self.extract_model_tensor(featured, input_window=input_window)
        return X, mask, featured
