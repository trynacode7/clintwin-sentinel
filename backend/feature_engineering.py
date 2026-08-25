"""
feature_engineering.py — Temporal, Clinical & Time Feature Engineering
======================================================================
Adds derived features to the preprocessed ICU data:

  Temporal Features:
    - Delta (first difference) for each measurement
    - Rolling mean (1-hour window at 5-min intervals)
    - Rolling standard deviation

  Clinical Features:
    - Shock Index = heart_rate / MAP (key clinical deterioration indicator)

  Time Features:
    - time_since_admission: minutes from first charttime per stay
    - hour_of_day: cyclical encoding using sin/cos for periodicity
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute first differences (deltas) for each measurement feature.
    
    Delta captures the *rate of change* — clinically, a rapidly falling
    MAP or rising lactate is more alarming than a static abnormal value.
    
    Computed per stay_id to avoid cross-stay contamination.
    """
    df = df.copy()
    
    for col in config.ALL_FEATURE_COLS:
        delta_col = f"{col}_delta"
        df[delta_col] = df.groupby("stay_id")[col].diff()
        # First row of each stay has NaN delta — fill with 0 (no change)
        df[delta_col] = df[delta_col].fillna(0)
    
    return df


def add_rolling_features(
    df: pd.DataFrame, window: int = config.ROLLING_WINDOW
) -> pd.DataFrame:
    """
    Compute rolling mean and standard deviation for each measurement.
    
    Window = 12 steps × 5 min = 1-hour lookback.
    
    Rolling statistics smooth out noise and capture trends:
    - Rolling mean shows the trend direction
    - Rolling std captures volatility (instability = risk)
    
    Computed per stay_id to avoid cross-stay contamination.
    """
    df = df.copy()
    
    for col in config.ALL_FEATURE_COLS:
        mean_col = f"{col}_rolling_mean"
        std_col = f"{col}_rolling_std"
        
        df[mean_col] = (
            df.groupby("stay_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[std_col] = (
            df.groupby("stay_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
        # Fill NaN std (when only 1 value in window) with 0
        df[std_col] = df[std_col].fillna(0)
    
    return df


def add_shock_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Shock Index = heart_rate / MAP.
    
    Shock Index > 0.9 is associated with hemodynamic instability.
    SI > 1.0 indicates likely shock. This is a widely used clinical
    marker that combines two vitals into a more predictive signal.
    
    Handles division by zero (MAP = 0 or NaN) gracefully.
    """
    df = df.copy()
    
    # Avoid division by zero or NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        df["shock_index"] = np.where(
            (df["map"].notna()) & (df["map"] != 0),
            df["heart_rate"] / df["map"],
            np.nan,
        )
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features:
    
    1. time_since_admission: Minutes from the first charttime in each stay.
       Captures how long the patient has been in the ICU — longer stays
       may indicate higher acuity or complications.
    
    2. hour_of_day_sin/cos: Cyclical encoding of the hour.
       ICU events follow circadian patterns (e.g., more interventions
       during day shifts, different staffing at night).
       Using sin/cos ensures 23:00 is close to 00:00.
    """
    df = df.copy()
    
    # Time since admission (minutes)
    df["time_since_admission"] = df.groupby("stay_id")[config.TIME_COL].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60.0
    )
    
    # Hour of day — cyclical encoding
    hour = df[config.TIME_COL].dt.hour + df[config.TIME_COL].dt.minute / 60.0
    df["hour_of_day_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_of_day_cos"] = np.cos(2 * np.pi * hour / 24.0)
    
    return df


def get_engineered_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the full list of feature columns after engineering.
    
    Includes: original features, missingness, time_since_obs,
    deltas, rolling stats, shock_index, time features.
    """
    feature_cols = []
    
    # Original measurement features
    feature_cols.extend(config.ALL_FEATURE_COLS)
    
    # Missingness indicators
    for col in config.ALL_FEATURE_COLS:
        if f"{col}_missing" in df.columns:
            feature_cols.append(f"{col}_missing")
        if f"{col}_time_since_obs" in df.columns:
            feature_cols.append(f"{col}_time_since_obs")
    
    # Delta features
    for col in config.ALL_FEATURE_COLS:
        delta_col = f"{col}_delta"
        if delta_col in df.columns:
            feature_cols.append(delta_col)
    
    # Rolling features
    for col in config.ALL_FEATURE_COLS:
        mean_col = f"{col}_rolling_mean"
        std_col = f"{col}_rolling_std"
        if mean_col in df.columns:
            feature_cols.append(mean_col)
        if std_col in df.columns:
            feature_cols.append(std_col)
    
    # Clinical features
    if "shock_index" in df.columns:
        feature_cols.append("shock_index")
    
    # Time features
    for tc in ["time_since_admission", "hour_of_day_sin", "hour_of_day_cos"]:
        if tc in df.columns:
            feature_cols.append(tc)
    
    return feature_cols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Order matters:
    1. Deltas (need raw values)
    2. Rolling stats (need raw values)
    3. Shock index (needs heart_rate and map)
    4. Time features (needs charttime)
    """
    logger.info("Engineering features...")
    
    df = add_delta_features(df)
    df = add_rolling_features(df)
    df = add_shock_index(df)
    df = add_time_features(df)
    
    # Fill any remaining NaN in engineered features with 0
    eng_cols = [c for c in df.columns if any(
        c.endswith(s) for s in ["_delta", "_rolling_mean", "_rolling_std"]
    )]
    eng_cols += ["shock_index", "time_since_admission", "hour_of_day_sin", "hour_of_day_cos"]
    for col in eng_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    n_features = len(get_engineered_feature_cols(df))
    logger.info(f"Feature engineering complete. Total features: {n_features}")
    
    return df


if __name__ == "__main__":
    from preprocessing import load_data, resample_all, add_missingness_indicators
    
    df = load_data()
    df = resample_all(df)
    df = add_missingness_indicators(df)
    df = engineer_features(df)
    
    feature_cols = get_engineered_feature_cols(df)
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Columns: {feature_cols}")
    print(f"\nSample:\n{df[feature_cols].head()}")
