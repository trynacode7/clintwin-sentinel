"""
dataset.py — PyTorch Dataset for ICU Time-Series
==================================================
Generates sliding windows from per-stay time-series data.

Key Design:
  - Windows are generated per stay_id (no cross-stay windows)
  - Returns (X, y, mask) where mask indicates non-missing values
  - Split by subject_id is enforced upstream in preprocessing
  - No data leakage: future data is never in the input window

Tensors:
  X: [batch, seq_len, n_features] — input features
  y: [batch] — binary label (0 or 1)
  mask: [batch, seq_len] — 1 where data exists, 0 for padding
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
from typing import Tuple, List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")


class ICUDataset(Dataset):
    """
    PyTorch Dataset for ICU time-series sliding windows.
    
    For each stay, generates overlapping windows of length INPUT_WINDOW.
    Each window is paired with the label computed from the future window.
    
    The mask tensor indicates which timesteps have valid (non-padded) data.
    For stays shorter than INPUT_WINDOW, data is left-padded with zeros
    and the mask is set to 0 for padded positions.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        window_size: int = config.INPUT_WINDOW,
        step_size: int = config.SLIDING_STEP,
        track_stay_ids: bool = False,
    ):
        """
        Args:
            df: Preprocessed DataFrame with features and labels
            feature_cols: List of feature column names to use as input
            window_size: Number of timesteps per input window
            step_size: Stride for sliding window
            track_stay_ids: if True, each window also carries its source
                stay_id as a 4th element, so predictions can later be
                joined back to subject-level metadata (age, sex) for
                subgroup evaluation — see evaluate.py's
                subgroup_evaluation() and create_eval_loader_with_stay_ids()
                below. Additive and opt-in: default False preserves the
                exact (X, y, mask) 3-tuple every existing caller (train.py,
                evaluate_model(), get_all_labels()) already expects.
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.track_stay_ids = track_stay_ids
        self.windows: List[Tuple] = []

        self._generate_windows(df, step_size)
        logger.info(f"Created dataset with {len(self.windows)} windows")
    
    def _generate_windows(self, df: pd.DataFrame, step_size: int):
        """
        Generate sliding windows grouped by stay_id.
        
        For each stay:
          1. Extract feature matrix and label vector
          2. Slide a window of size INPUT_WINDOW across the stay
          3. For each position, create (X, y, mask) tuple
          4. Short stays are left-padded to window_size
        """
        for stay_id, group in df.groupby("stay_id"):
            group = group.sort_values(config.TIME_COL).reset_index(drop=True)
            
            # Extract features and labels
            features = group[self.feature_cols].values.astype(np.float32)
            labels = group["label"].values.astype(np.float32)
            
            n_steps = len(group)
            
            if n_steps < self.window_size:
                # Stay is shorter than window — pad and create single sample
                if np.isnan(labels[-1]):
                    continue
                
                pad_len = self.window_size - n_steps
                X = np.zeros((self.window_size, features.shape[1]), dtype=np.float32)
                X[pad_len:] = features
                
                mask = np.zeros(self.window_size, dtype=np.float32)
                mask[pad_len:] = 1.0
                
                # Replace NaN in features with 0 (model uses mask)
                X = np.nan_to_num(X, nan=0.0)

                if self.track_stay_ids:
                    self.windows.append((X, labels[-1], mask, stay_id))
                else:
                    self.windows.append((X, labels[-1], mask))
            else:
                # Generate sliding windows
                for start in range(0, n_steps - self.window_size + 1, step_size):
                    end = start + self.window_size
                    
                    # Use the label at the END of the input window
                    # (this label was computed from FUTURE data beyond this point)
                    y = labels[end - 1]
                    
                    if np.isnan(y):
                        continue
                    
                    X = features[start:end].copy()
                    mask = np.ones(self.window_size, dtype=np.float32)
                    
                    # Mark rows with all-NaN features as padding
                    all_nan_rows = np.isnan(X).all(axis=1)
                    mask[all_nan_rows] = 0.0
                    
                    # Replace NaN with 0 (model uses mask + missingness indicators)
                    X = np.nan_to_num(X, nan=0.0)

                    if self.track_stay_ids:
                        self.windows.append((X, y, mask, stay_id))
                    else:
                        self.windows.append((X, y, mask))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        if self.track_stay_ids:
            X, y, mask, stay_id = self.windows[idx]
            return (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                stay_id,
            )
        X, y, mask = self.windows[idx]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


def create_dataloaders(
    data_dict: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    batch_size: int = config.BATCH_SIZE,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, val, and test splits.
    
    Args:
        data_dict: Dictionary with 'train', 'val', 'test' DataFrames
        feature_cols: Feature column names
        batch_size: Batch size for training
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    
    for split in ["train", "val", "test"]:
        if split not in data_dict:
            continue
        
        # Validation and test use 1-hour step size (12 timesteps) for representative coverage
        step = config.SLIDING_STEP if split == "train" else max(1, config.INPUT_WINDOW // 6)
        
        dataset = ICUDataset(
            df=data_dict[split],
            feature_cols=feature_cols,
            window_size=config.INPUT_WINDOW,
            step_size=step,
        )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            num_workers=0,  # Windows compatibility
            pin_memory=torch.cuda.is_available(),
        )
        
        logger.info(f"{split} loader: {len(dataset)} samples, {len(loaders[split])} batches")
    
    return loaders


def get_all_labels(loader: DataLoader) -> np.ndarray:
    """Extract all labels from a DataLoader for class weight computation."""
    all_labels = []
    for _, y, _ in loader:
        all_labels.append(y.numpy())
    return np.concatenate(all_labels)


def create_eval_loader_with_stay_ids(
    df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int = config.BATCH_SIZE,
) -> DataLoader:
    """
    Build an evaluation-only DataLoader whose batches are
    (X, y, mask, stay_ids) instead of the usual (X, y, mask) — so a
    caller can join predictions back to subject-level metadata (age,
    sex, admission type) for subgroup evaluation. See evaluate.py's
    subgroup_evaluation().

    Not used anywhere in the training path — this exists purely for
    evaluation and does not change ICUDataset's default behavior for
    any existing caller. shuffle=False and drop_last=False
    deliberately, so every test-split window is evaluated exactly
    once, in a stable order.
    """
    step = max(1, config.INPUT_WINDOW // 6)
    dataset = ICUDataset(
        df=df,
        feature_cols=feature_cols,
        window_size=config.INPUT_WINDOW,
        step_size=step,
        track_stay_ids=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


if __name__ == "__main__":
    # Smoke-test block only — not used by training or evaluation.
    # Was importing the old ad-hoc backend.labels (lactate/MAP/HR rule)
    # instead of the real backend.labels_sofa, and ran engineer_features
    # before create_labels — both fixed here to match the documented
    # required order (see preprocessing.preprocess()'s docstring and
    # backend/train.py's Step 1-4 ordering) so this smoke test actually
    # reflects the real pipeline.
    from preprocessing import preprocess
    from feature_engineering import engineer_features, get_engineered_feature_cols
    from labels_sofa import create_labels

    result = preprocess()

    for split in ["train", "val", "test"]:
        result[split] = create_labels(result[split])
        result[split] = engineer_features(result[split])
    
    feature_cols = get_engineered_feature_cols(result["train"])
    loaders = create_dataloaders(result, feature_cols)
    
    # Inspect a batch
    for X, y, mask in loaders["train"]:
        print(f"X shape: {X.shape}")       # [batch, 72, n_features]
        print(f"y shape: {y.shape}")       # [batch]
        print(f"mask shape: {mask.shape}") # [batch, 72]
        print(f"y distribution: pos={y.sum().item()}, neg={(1-y).sum().item()}")
        break
