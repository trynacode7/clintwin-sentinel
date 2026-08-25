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
    ):
        """
        Args:
            df: Preprocessed DataFrame with features and labels
            feature_cols: List of feature column names to use as input
            window_size: Number of timesteps per input window
            step_size: Stride for sliding window
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.windows: List[Tuple[np.ndarray, float, np.ndarray]] = []
        
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
                    
                    self.windows.append((X, y, mask))
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        dataset = ICUDataset(
            df=data_dict[split],
            feature_cols=feature_cols,
            window_size=config.INPUT_WINDOW,
            step_size=config.SLIDING_STEP if split == "train" else config.INPUT_WINDOW,
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


if __name__ == "__main__":
    from preprocessing import preprocess, scale_splits
    from feature_engineering import engineer_features, get_engineered_feature_cols
    from labels_sofa import create_labels  # SOFA/Sepsis-3-consistent — NOT labels.py's legacy rule

    result = preprocess()

    for split in ["train", "val", "test"]:
        result[split] = engineer_features(result[split])
        result[split] = create_labels(result[split])
    result = scale_splits(result)  # scale AFTER engineering + labeling (raw clinical units needed for both)
    
    feature_cols = get_engineered_feature_cols(result["train"])
    loaders = create_dataloaders(result, feature_cols)
    
    # Inspect a batch
    for X, y, mask in loaders["train"]:
        print(f"X shape: {X.shape}")       # [batch, 72, n_features]
        print(f"y shape: {y.shape}")       # [batch]
        print(f"mask shape: {mask.shape}") # [batch, 72]
        print(f"y distribution: pos={y.sum().item()}, neg={(1-y).sum().item()}")
        break
