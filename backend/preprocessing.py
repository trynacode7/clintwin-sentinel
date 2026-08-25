"""
preprocessing.py — ICU Data Preprocessing Pipeline
====================================================
Handles:
  1. CSV loading and datetime parsing
  2. Per-stay sorting and 5-minute resampling
  3. Missing value handling (LOCF + interpolation for vitals ONLY)
  4. Missingness indicators and time-since-last-observation
  5. Z-score normalization (fit on training set only)
  6. Train/val/test split by subject_id (no data leakage)

IMPORTANT DESIGN DECISIONS:
  - Missing values are NEVER treated as zero
  - Labs (lactate, wbc) use forward-fill ONLY — no interpolation
  - Vitals get forward-fill THEN linear interpolation
  - Normalization scaler is fit exclusively on training data
  - All splits are done by subject_id to prevent leakage
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import setup_logging

logger = setup_logging()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the ICU dataset from CSV.

    DATA_MODE-aware, with NO silent fallback (Phase 1 requirement — see
    config.py and DATA_CONTRACT.md): if the expected file is missing, this
    raises with an explicit, actionable message rather than substituting
    synthetic data for a requested real-data run, or vice versa.

    Args:
        path: Path to CSV file. Defaults to config.DATA_PATH (which itself
              depends on config.DATA_MODE).

    Returns:
        Raw DataFrame with parsed datetime column.
    """
    path = path or config.DATA_PATH

    if not os.path.exists(path):
        if config.DATA_MODE == "real":
            raise FileNotFoundError(
                f"DATA_MODE='real' but no extracted dataset exists at {path}.\n"
                f"What is missing: the flat, canonical-schema ml_dataset.csv (DATA_CONTRACT.md "
                f"section 3) that the MIMIC-IV adapter produces.\n"
                f"Expected location: {path} (configure via the CLINTWIN_REAL_DATA_PATH "
                f"environment variable).\n"
                f"How to produce it: place your own PhysioNet-credentialed, locally-extracted "
                f"raw MIMIC-IV table CSVs at {config.REAL_DATA_RAW_DIR} (configure via "
                f"CLINTWIN_REAL_DATA_RAW_DIR) and run "
                f"`python extract_real_mimic_to_ml_dataset.py` (see docs/MIMIC_IV_SETUP.md). "
                f"Refusing to silently fall back to synthetic data."
            )
        raise FileNotFoundError(
            f"DATA_MODE='synthetic' but no dataset exists at {path}. Generate it first, e.g.:\n"
            f"    python generate_synthetic_data.py --patients 1000 --seed 42"
        )

    logger.info(f"Loading data from {path} (DATA_MODE={config.DATA_MODE})")
    df = pd.read_csv(path, parse_dates=[config.TIME_COL])
    logger.info(f"Loaded {len(df)} rows, {df['subject_id'].nunique()} patients")

    # Ensure charttime is datetime
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL])
    
    # Sort by stay_id and time
    df = df.sort_values([*config.ID_COLS, config.TIME_COL]).reset_index(drop=True)
    
    # Log missingness statistics
    for col in config.ALL_FEATURE_COLS:
        miss_pct = df[col].isna().mean() * 100
        logger.info(f"  {col}: {miss_pct:.1f}% missing")
    
    return df


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary missingness indicators and time-since-last-observation.
    
    For each feature column, creates:
      - {feature}_missing: 1 if value is NaN, 0 otherwise
      - {feature}_time_since_obs: minutes since the last non-NaN observation
    
    These are critical for the model to distinguish between "value is normal"
    and "value was not measured" — a key distinction in clinical data.
    """
    df = df.copy()
    
    for col in config.ALL_FEATURE_COLS:
        # Binary missingness indicator
        df[f"{col}_missing"] = df[col].isna().astype(float)
        
        # Time since last observation (per stay)
        # For each row, compute minutes since the last non-NaN value
        df[f"{col}_time_since_obs"] = np.nan
        
        for stay_id, group in df.groupby("stay_id"):
            mask = group[col].notna()
            if mask.any():
                last_obs_time = group.loc[mask, config.TIME_COL].reindex(
                    group.index, method="ffill"
                )
                time_diff = (
                    group[config.TIME_COL] - last_obs_time
                ).dt.total_seconds() / 60.0
                df.loc[group.index, f"{col}_time_since_obs"] = time_diff
    
    # Fill remaining NaN in time_since_obs with a large value (no prior observation)
    for col in config.ALL_FEATURE_COLS:
        df[f"{col}_time_since_obs"] = df[f"{col}_time_since_obs"].fillna(999.0)
    
    return df


def resample_stay(group: pd.DataFrame, freq: str = config.RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Resample a single stay's data to regular intervals.

    Generalized (Phase 1) to use config.COLUMN_FILL_STRATEGY rather than a
    hardcoded two-way vitals/labs split, so new columns (SOFA labs, GCS,
    treatment/output variables, and the label-only suspected_infection flag)
    each get a clinically appropriate fill strategy without further code
    changes when the feature set grows again (e.g. for real MIMIC-IV):
      - "interpolate": forward-fill then linear-interpolate (continuous vitals)
      - "locf":         forward-fill only, no interpolation (sparse labs,
                        infusion/ventilation status that holds until changed)
      - "sum":          resample-sum per bin (event amounts — e.g. fluid
                        bolus mL — that must never be carried forward)

    Args:
        group: DataFrame for a single stay_id
        freq: Resampling frequency string

    Returns:
        Resampled DataFrame
    """
    group = group.set_index(config.TIME_COL)

    cols_by_strategy: Dict[str, List[str]] = {}
    for col, strategy in config.COLUMN_FILL_STRATEGY.items():
        if col in group.columns:
            cols_by_strategy.setdefault(strategy, []).append(col)

    parts = []
    if cols_by_strategy.get("interpolate"):
        part = group[cols_by_strategy["interpolate"]].resample(freq).mean()
        part = part.ffill()
        part = part.interpolate(method="linear", limit_direction="forward")
        parts.append(part)
    if cols_by_strategy.get("locf"):
        part = group[cols_by_strategy["locf"]].resample(freq).mean()
        part = part.ffill()
        parts.append(part)
    if cols_by_strategy.get("sum"):
        part = group[cols_by_strategy["sum"]].resample(freq).sum(min_count=0).fillna(0.0)
        parts.append(part)

    resampled = pd.concat(parts, axis=1) if parts else group.resample(freq).mean()

    # Carry forward ID columns
    for col in config.ID_COLS:
        resampled[col] = group[col].iloc[0]

    resampled = resampled.reset_index()

    return resampled


def resample_all(df: pd.DataFrame) -> pd.DataFrame:
    """Resample all stays to regular intervals (config.RESAMPLE_FREQ)."""
    logger.info(f"Resampling all stays to {config.RESAMPLE_FREQ} intervals...")
    
    resampled_groups = []
    for stay_id, group in df.groupby("stay_id"):
        if len(group) < 2:
            continue
        resampled = resample_stay(group)
        resampled_groups.append(resampled)
    
    df_resampled = pd.concat(resampled_groups, ignore_index=True)
    logger.info(f"After resampling: {len(df_resampled)} rows")
    
    return df_resampled


def drop_all_vitals_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where ALL vital signs are missing.
    
    Rows with at least one vital present are kept — the missingness
    indicators will inform the model about which values are imputed.
    """
    all_missing = df[config.VITAL_COLS].isna().all(axis=1)
    n_dropped = all_missing.sum()
    df = df[~all_missing].reset_index(drop=True)
    logger.info(f"Dropped {n_dropped} rows with all vitals missing")
    return df


def split_by_subject(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by subject_id to prevent data leakage.
    
    All stays from a single patient go into the same split.
    This is critical — splitting by rows or stays could leak
    temporal patterns from the same patient across splits.
    
    Args:
        df: Full DataFrame
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_df, val_df, test_df)
    """
    np.random.seed(seed)
    
    subjects = df["subject_id"].unique()
    np.random.shuffle(subjects)
    
    n = len(subjects)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_subjects = set(subjects[:train_end])
    val_subjects = set(subjects[train_end:val_end])
    test_subjects = set(subjects[val_end:])
    
    train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjects)].reset_index(drop=True)
    
    logger.info(
        f"Split: train={len(train_df)} ({len(train_subjects)} patients), "
        f"val={len(val_df)} ({len(val_subjects)} patients), "
        f"test={len(test_df)} ({len(test_subjects)} patients)"
    )
    
    return train_df, val_df, test_df


def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> StandardScaler:
    """
    Fit Z-score normalization scaler on TRAINING data only.
    
    This prevents data leakage — the scaler learns statistics
    only from the training set, then transforms all splits.
    
    Args:
        train_df: Training DataFrame
        feature_cols: Columns to normalize
        
    Returns:
        Fitted StandardScaler
    """
    scaler = StandardScaler()
    
    # Only fit on non-NaN values
    train_values = train_df[feature_cols].values
    # Replace NaN with column means for fitting (they'll be handled separately)
    col_means = np.nanmean(train_values, axis=0)
    train_filled = np.where(np.isnan(train_values), col_means, train_values)
    
    scaler.fit(train_filled)
    
    # Save scaler
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"Scaler saved to {config.SCALER_PATH}")
    
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Apply Z-score normalization using a pre-fitted scaler.
    
    Preserves NaN positions — normalization is only applied
    to non-missing values.
    """
    df = df.copy()
    values = df[feature_cols].values
    
    # Track NaN positions
    nan_mask = np.isnan(values)
    
    # Fill NaN temporarily for transform
    col_means = np.nanmean(values, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    values_filled = np.where(nan_mask, col_means, values)
    
    # Transform
    values_scaled = scaler.transform(values_filled)
    
    # Restore NaN positions
    values_scaled[nan_mask] = np.nan
    
    df[feature_cols] = values_scaled
    return df


def get_all_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get the full list of feature columns after preprocessing.
    Includes original features, missingness indicators, and time-since-obs.
    """
    feature_cols = list(config.ALL_FEATURE_COLS)
    
    for col in config.ALL_FEATURE_COLS:
        miss_col = f"{col}_missing"
        time_col = f"{col}_time_since_obs"
        if miss_col in df.columns:
            feature_cols.append(miss_col)
        if time_col in df.columns:
            feature_cols.append(time_col)
    
    return feature_cols


def preprocess(
    data_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    RAW preprocessing pipeline: load, resample, add missingness
    indicators, drop all-missing rows, split by subject_id.

    Deliberately does NOT scale/normalize anything. Scaling is a
    separate step (`scale_splits`, below) that must run AFTER
    feature_engineering.engineer_features() and
    labels_sofa.create_labels() — not before.

    Why (bug found and fixed during the local-machine reconciliation
    phase): labels_sofa.py's SOFA subscores use absolute published
    clinical thresholds (e.g. platelets < 150, bilirubin >= 1.2, MAP <
    70 mmHg, creatinine >= 1.2). feature_engineering.py's shock_index
    (heart_rate / MAP) is a clinical ratio with a real-world meaning.
    Both require real clinical units. If scaling ran first (the
    previous order — inside this function), every one of those
    thresholds was being compared against a z-scored value (mean 0,
    std 1) instead of a real mmHg/mg-dL/count value, which silently
    produced a wrong, near-degenerate SOFA distribution (verified: the
    old order produced SOFA totals clustered at 11-23 out of a
    possible 24, when a real synthetic ICU cohort should show a much
    lower, more varied distribution) — i.e. the model's supervision
    labels were not actually the documented SOFA/Sepsis-3 labels.
    Moving scaling to run LAST (after engineering + labeling, right
    before the data is turned into model input tensors) fixes this
    while leaving the model's final input feature *names* and count
    unchanged — see `scale_splits()`.

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames, all in
        real clinical units (not yet scaled).
    """
    # 1. Load data
    df = load_data(data_path)

    # 2. Resample to regular intervals
    df = resample_all(df)

    # 3. Add missingness indicators (before filling)
    df = add_missingness_indicators(df)

    # 4. Drop rows where all vitals are missing
    df = drop_all_vitals_missing(df)

    # 5. Split by subject_id
    train_df, val_df, test_df = split_by_subject(df)

    logger.info(
        f"Raw preprocessing complete (not yet scaled). "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} rows."
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def scale_splits(
    data_dict: Dict[str, pd.DataFrame],
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fit Z-score normalization on TRAIN ONLY and apply it to train/val/test.

    MUST be called AFTER feature_engineering.engineer_features() and
    labels_sofa.create_labels() have already run on real, unscaled
    clinical values — see preprocess()'s docstring for why. Calling
    this before either of those silently corrupts SOFA labeling and
    the clinical shock_index feature.

    Only the raw measurement columns (config.ALL_FEATURE_COLS) are
    normalized by default — matching the original design ("only
    normalize the actual measurement columns, not indicators").
    Engineered columns (deltas, rolling stats, shock_index, time
    features) and missingness/time-since-obs indicators are left as
    computed. Pass an explicit `feature_cols` to scale a different set.

    Returns a NEW dict: the input splits plus 'scaler' (the fitted
    StandardScaler, also saved to config.SCALER_PATH).
    """
    feature_cols = list(feature_cols) if feature_cols is not None else list(config.ALL_FEATURE_COLS)

    train_df = data_dict["train"]
    scaler = fit_scaler(train_df, feature_cols)

    out = dict(data_dict)
    out["train"] = apply_scaler(data_dict["train"], scaler, feature_cols)
    if "val" in data_dict:
        out["val"] = apply_scaler(data_dict["val"], scaler, feature_cols)
    if "test" in data_dict:
        out["test"] = apply_scaler(data_dict["test"], scaler, feature_cols)
    out["scaler"] = scaler
    out["scaled_feature_cols"] = feature_cols

    logger.info(f"Scaling complete. {len(feature_cols)} columns normalized on train-fit statistics.")

    return out


if __name__ == "__main__":
    from feature_engineering import engineer_features, get_engineered_feature_cols
    from labels_sofa import create_labels

    result = preprocess()
    for split in ["train", "val", "test"]:
        result[split] = engineer_features(result[split])
        result[split] = create_labels(result[split])
    result = scale_splits(result)

    feature_cols = get_engineered_feature_cols(result["train"])
    print(f"\nTrain shape: {result['train'].shape}")
    print(f"Val shape:   {result['val'].shape}")
    print(f"Test shape:  {result['test'].shape}")
    print(f"Features:    {len(feature_cols)}")
    print(f"\nFeature columns:\n{feature_cols}")
