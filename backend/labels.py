"""
labels.py — Future-Window Label Creation
==========================================
Creates clinical deterioration labels using a FUTURE window approach.

CRITICAL DESIGN DECISIONS:
  - The original risk_flag column is NEVER used
  - Labels are computed from future data (24-72 timesteps ahead)
  - This simulates real clinical prediction: "will the patient deteriorate?"
  
Label Logic:
  Binary label = 1 if ANY of the following occur in the future window:
    1. lactate > 2.0 (ignoring NaN — don't flag missing labs)
    2. MAP < 65 mmHg (hypotension)
    3. heart_rate > 100 for ≥ 3 consecutive timesteps (sustained tachycardia)
  
  Continuous risk score:
    Weighted combination of condition severities
    
  Delta risk:
    Change in risk score over time
    
FUTURE LABEL ALIGNMENT:
  For input window ending at time t, the label is computed from
  data in [t + FUTURE_WINDOW_START, t + FUTURE_WINDOW_END].
  This ensures the model predicts what happens AFTER the input window.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")


def check_lactate_criterion(future_data: pd.DataFrame) -> bool:
    """
    Check if lactate exceeds threshold in future window.
    
    IMPORTANT: NaN values are IGNORED — we only flag when we have
    an actual measurement showing elevated lactate. Missing labs
    should not trigger a positive label.
    """
    lactate_values = future_data["lactate"].dropna()
    if len(lactate_values) == 0:
        return False
    return (lactate_values > config.LACTATE_THRESHOLD).any()


def check_map_criterion(future_data: pd.DataFrame) -> bool:
    """
    Check if MAP drops below threshold in future window.
    MAP < 65 indicates hypotension and potential organ hypoperfusion.
    """
    map_values = future_data["map"].dropna()
    if len(map_values) == 0:
        return False
    return (map_values < config.MAP_THRESHOLD).any()


def check_hr_criterion(future_data: pd.DataFrame) -> bool:
    """
    Check if heart rate exceeds threshold for consecutive timesteps.
    
    Sustained tachycardia (≥3 consecutive steps of HR > 100) is more
    clinically significant than a brief spike. This captures patients
    who are genuinely deteriorating vs. momentary artifacts.
    """
    hr_values = future_data["heart_rate"].values
    
    consecutive = 0
    for val in hr_values:
        if not np.isnan(val) and val > config.HR_THRESHOLD:
            consecutive += 1
            if consecutive >= config.HR_CONSECUTIVE_STEPS:
                return True
        else:
            consecutive = 0
    
    return False


def compute_continuous_risk(future_data: pd.DataFrame) -> float:
    """
    Compute a continuous risk score from the future window.
    
    Combines multiple deterioration signals with clinical weighting:
    - Lactate severity (strongly weighted — direct tissue hypoxia marker)
    - MAP depression (moderate weight — perfusion pressure)
    - Heart rate elevation (moderate weight — compensatory mechanism)
    
    Returns:
        Float in [0, 1] representing composite risk severity.
    """
    risk_components = []
    
    # Lactate component: normalized excess above threshold
    lactate_vals = future_data["lactate"].dropna()
    if len(lactate_vals) > 0:
        max_lactate = lactate_vals.max()
        if max_lactate > config.LACTATE_THRESHOLD:
            # Normalize: 2->0, 4->0.5, 6->0.75, 10->0.9
            lactate_risk = 1.0 - 1.0 / (1.0 + (max_lactate - config.LACTATE_THRESHOLD) / 2.0)
            risk_components.append(("lactate", 0.4, lactate_risk))
    
    # MAP component: normalized deficit below threshold
    map_vals = future_data["map"].dropna()
    if len(map_vals) > 0:
        min_map = map_vals.min()
        if min_map < config.MAP_THRESHOLD:
            # Normalize: 65->0, 55->0.5, 45->0.75
            map_risk = 1.0 - 1.0 / (1.0 + (config.MAP_THRESHOLD - min_map) / 10.0)
            risk_components.append(("map", 0.35, map_risk))
    
    # Heart rate component: sustained elevation
    hr_vals = future_data["heart_rate"].dropna()
    if len(hr_vals) > 0:
        hr_above = (hr_vals > config.HR_THRESHOLD).sum()
        hr_fraction = hr_above / max(len(hr_vals), 1)
        if hr_fraction > 0:
            risk_components.append(("hr", 0.25, hr_fraction))
    
    if not risk_components:
        return 0.0
    
    # Weighted sum, normalized by total weight
    total_weight = sum(w for _, w, _ in risk_components)
    risk = sum(w * r for _, w, r in risk_components) / total_weight
    
    return float(np.clip(risk, 0, 1))


def create_labels_for_stay(stay_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create labels for all valid positions in a single stay.
    
    For each timestep t, looks ahead at data in the future window
    [t + FUTURE_WINDOW_START, t + FUTURE_WINDOW_END] to compute labels.
    
    This is the core of the future-label alignment — we predict what
    happens AFTER the current observation window.
    
    Args:
        stay_data: DataFrame for a single stay, sorted by time
        
    Returns:
        DataFrame with added label columns
    """
    n = len(stay_data)
    labels = np.zeros(n, dtype=np.float32)
    risk_scores = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        future_start = i + config.FUTURE_WINDOW_START
        future_end = i + config.FUTURE_WINDOW_END
        
        if future_end > n:
            # Not enough future data — mark as NaN (will be excluded)
            labels[i] = np.nan
            risk_scores[i] = np.nan
            continue
        
        future_data = stay_data.iloc[future_start:future_end]
        
        # Binary label: ANY deterioration criterion met
        label = 0
        if check_lactate_criterion(future_data):
            label = 1
        elif check_map_criterion(future_data):
            label = 1
        elif check_hr_criterion(future_data):
            label = 1
        
        labels[i] = label
        risk_scores[i] = compute_continuous_risk(future_data)
    
    stay_data = stay_data.copy()
    stay_data["label"] = labels
    stay_data["risk_score"] = risk_scores
    
    return stay_data


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create future-window labels for all stays.
    
    Also computes delta_risk (change in risk score over time),
    which captures the trajectory of deterioration.
    
    Returns:
        DataFrame with added: label, risk_score, delta_risk columns
    """
    logger.info("Creating future-window labels...")
    
    labeled_groups = []
    for stay_id, group in df.groupby("stay_id"):
        group = group.sort_values(config.TIME_COL).reset_index(drop=True)
        labeled = create_labels_for_stay(group)
        labeled_groups.append(labeled)
    
    df_labeled = pd.concat(labeled_groups, ignore_index=True)
    
    # Drop rows without valid labels (not enough future data)
    n_before = len(df_labeled)
    df_labeled = df_labeled.dropna(subset=["label"]).reset_index(drop=True)
    n_after = len(df_labeled)
    logger.info(f"Labels created. Dropped {n_before - n_after} rows without future data.")
    
    # Compute delta risk (change in risk score)
    df_labeled["delta_risk"] = df_labeled.groupby("stay_id")["risk_score"].diff().fillna(0)
    
    # Log class distribution
    n_pos = (df_labeled["label"] == 1).sum()
    n_neg = (df_labeled["label"] == 0).sum()
    ratio = n_pos / max(n_neg, 1)
    logger.info(f"Label distribution: pos={n_pos}, neg={n_neg}, ratio={ratio:.3f}")
    
    return df_labeled


if __name__ == "__main__":
    from preprocessing import preprocess
    from feature_engineering import engineer_features
    
    result = preprocess()
    
    for split in ["train", "val", "test"]:
        result[split] = engineer_features(result[split])
        result[split] = create_labels(result[split])
        
        n_pos = (result[split]["label"] == 1).sum()
        n_neg = (result[split]["label"] == 0).sum()
        print(f"{split}: pos={n_pos}, neg={n_neg}, total={len(result[split])}")
