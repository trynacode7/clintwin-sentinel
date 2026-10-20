"""
utils.py — Utility Functions for ICU Digital Twin System
========================================================
Logging, seeding, device detection, and synthetic data generation.
"""

import os
import random
import logging
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta


def setup_logging(level=logging.INFO):
    """Configure structured logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("icu_digital_twin")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.getLogger("icu_digital_twin").info(
            f"Using CUDA: {torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device("cpu")
        logging.getLogger("icu_digital_twin").info("Using CPU")
    return device


def generate_synthetic_data(
    n_patients: int = 50,
    stays_per_patient: int = 1,
    hours_per_stay: int = 48,
    freq_minutes: int = 5,
    missing_rate: float = 0.3,
    lab_missing_rate: float = 0.7,
) -> pd.DataFrame:
    """
    Generate realistic synthetic ICU time-series data for demo/testing.

    Simulates:
    - Multiple patients with unique subject_ids
    - Irregularly sampled vitals and labs
    - Realistic value ranges
    - Sparse lab values (higher missing rate)
    - Some patients with deterioration patterns

    Returns:
        pd.DataFrame with columns matching MIMIC-IV schema
    """
    np.random.seed(42)
    records = []
    
    for pid in range(1, n_patients + 1):
        for sid in range(1, stays_per_patient + 1):
            stay_id = pid * 100 + sid
            hadm_id = pid * 1000 + sid
            
            # Base time
            base_time = datetime(2024, 1, 1) + timedelta(days=pid)
            n_steps = (hours_per_stay * 60) // freq_minutes
            
            # Decide if this patient will deteriorate
            deteriorates = np.random.random() < 0.3
            deterioration_start = int(n_steps * 0.6) if deteriorates else n_steps + 1
            
            for t in range(n_steps):
                # Time with slight jitter for irregular sampling
                jitter = np.random.randint(-2, 3)
                charttime = base_time + timedelta(minutes=t * freq_minutes + jitter)
                
                # Base vitals with physiological noise
                if t < deterioration_start:
                    hr = np.random.normal(80, 10)
                    map_val = np.random.normal(80, 8)
                    rr = np.random.normal(16, 3)
                    spo2 = np.random.normal(97, 1.5)
                    temp = np.random.normal(37.0, 0.3)
                    lactate = np.random.normal(1.0, 0.3)
                    wbc = np.random.normal(8.0, 2.0)
                else:
                    # Deterioration: rising HR, falling MAP, rising lactate
                    severity = (t - deterioration_start) / (n_steps - deterioration_start + 1)
                    hr = np.random.normal(80 + 40 * severity, 8)
                    map_val = np.random.normal(80 - 25 * severity, 6)
                    rr = np.random.normal(16 + 10 * severity, 3)
                    spo2 = np.random.normal(97 - 5 * severity, 2)
                    temp = np.random.normal(37.0 + 1.5 * severity, 0.4)
                    lactate = np.random.normal(1.0 + 3.0 * severity, 0.5)
                    wbc = np.random.normal(8.0 + 6.0 * severity, 2.5)
                
                # Clamp to physiological ranges
                hr = np.clip(hr, 40, 180)
                map_val = np.clip(map_val, 30, 140)
                rr = np.clip(rr, 6, 40)
                spo2 = np.clip(spo2, 70, 100)
                temp = np.clip(temp, 34, 42)
                lactate = np.clip(lactate, 0.3, 15)
                wbc = np.clip(wbc, 1, 30)
                
                row = {
                    "subject_id": pid,
                    "hadm_id": hadm_id,
                    "stay_id": stay_id,
                    "charttime": charttime,
                    "heart_rate": hr if np.random.random() > missing_rate else np.nan,
                    "map": map_val if np.random.random() > missing_rate else np.nan,
                    "resp_rate": rr if np.random.random() > missing_rate else np.nan,
                    "spo2": spo2 if np.random.random() > missing_rate else np.nan,
                    "temp": temp if np.random.random() > (missing_rate + 0.2) else np.nan,
                    "lactate": lactate if np.random.random() > lab_missing_rate else np.nan,
                    "wbc": wbc if np.random.random() > lab_missing_rate else np.nan,
                    "risk_flag": int(deteriorates and t >= deterioration_start),
                }
                records.append(row)
    
    df = pd.DataFrame(records)
    return df


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced binary classification.
    
    Uses inverse frequency weighting:
        weight_class_i = total_samples / (n_classes * count_class_i)
    
    Args:
        labels: Binary label array
        
    Returns:
        Tensor with [weight_class_0, weight_class_1]
    """
    n_total = len(labels)
    n_pos = labels.sum()
    n_neg = n_total - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0])
    
    weight_neg = n_total / (2.0 * n_neg)
    weight_pos = n_total / (2.0 * n_pos)
    
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
