"""
train.py — Training Loop for ICU Risk Prediction Models
=========================================================
Supports both LSTM and Transformer models with:
  - BCE loss with automatic class weighting (handles imbalanced labels)
  - Adam optimizer
  - ReduceLROnPlateau scheduler
  - Early stopping with patience
  - Gradient clipping
  - Model checkpointing (saves best model)
  - Training metrics logging

Usage:
  python -m backend.train --model lstm
  python -m backend.train --model transformer
  python -m backend.train --model lstm --demo  (uses synthetic data)
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import setup_logging, set_seed, get_device, compute_class_weights
from backend.preprocessing import preprocess, scale_splits
from backend.feature_engineering import engineer_features, get_engineered_feature_cols
from backend.labels_sofa import create_labels  # Phase 1: SOFA/Sepsis-3-consistent label
                                                 # (labels.py's original ad hoc threshold rule
                                                 # is preserved, unmodified, and unused by default)
from backend.dataset import create_dataloaders, get_all_labels
from backend.lstm_model import ICUBiLSTM
from backend.transformer_model import ICUTransformer

logger = setup_logging()


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "lstm",
    grad_clip: float = config.GRAD_CLIP_MAX_NORM,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with 'loss' and 'accuracy' metrics.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (X, y, mask) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == "transformer":
            risk, z_t, attn = model(X, mask)
        else:
            risk, z_t = model(X, mask)
        
        risk = risk.squeeze(-1)  # [batch]
        
        # Compute loss
        loss = criterion(risk, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * len(y)
        preds = (risk > 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)
    
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "lstm",
) -> Dict[str, float]:
    """
    Validate the model.
    
    Returns:
        Dictionary with 'loss', 'accuracy', and 'predictions'.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for X, y, mask in loader:
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)
        
        if model_type == "transformer":
            risk, z_t, attn = model(X, mask)
        else:
            risk, z_t = model(X, mask)
        
        risk = risk.squeeze(-1)
        loss = criterion(risk, y)
        
        total_loss += loss.item() * len(y)
        preds = (risk > 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)
        
        all_preds.extend(risk.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training if it hasn't
    improved for `patience` consecutive epochs.
    """
    
    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_model(
    model_type: str = "lstm",
    data_path: Optional[str] = None,
    num_epochs: int = config.NUM_EPOCHS,
) -> Dict:
    """
    Full training pipeline.
    
    Steps:
      1. Preprocess data
      2. Engineer features
      3. Create labels
      4. Create DataLoaders
      5. Initialize model
      6. Train with early stopping
      7. Save best model
    
    Args:
        model_type: "lstm" or "transformer"
        data_path: Path to CSV (None = use config default)
        num_epochs: Maximum training epochs
        
    Returns:
        Dictionary with trained model, metrics, and feature columns.
    """
    set_seed(config.SEED)
    device = get_device()
    
    # ─── Data Pipeline ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Training {model_type.upper()} model")
    logger.info("=" * 60)
    
    # Step 1: Preprocess (RAW clinical units — no scaling yet)
    logger.info("Step 1/6: Preprocessing...")
    result = preprocess(data_path)

    # Step 2: Feature Engineering (on RAW values — deltas/rolling/shock_index
    # must be computed in real clinical units)
    logger.info("Step 2/6: Feature engineering...")
    for split in ["train", "val", "test"]:
        result[split] = engineer_features(result[split])

    # Step 3: Create Labels (SOFA subscores need RAW clinical units — must
    # run BEFORE scaling; see preprocessing.preprocess()'s docstring)
    logger.info("Step 3/6: Creating labels...")
    for split in ["train", "val", "test"]:
        result[split] = create_labels(result[split])

    # Step 3b: Scale (Z-score normalize) the raw measurement columns —
    # ONLY now, after engineering and labeling are done with real units.
    logger.info("Step 3b/6: Scaling...")
    result = scale_splits(result)

    # Step 4: Create DataLoaders
    logger.info("Step 4/6: Creating DataLoaders...")
    feature_cols = get_engineered_feature_cols(result["train"])
    loaders = create_dataloaders(result, feature_cols)
    
    if "train" not in loaders or len(loaders["train"]) == 0:
        logger.error("No training data available!")
        return None
    
    input_size = len(feature_cols)
    logger.info(f"Input features: {input_size}")
    
    # ─── Model Setup ─────────────────────────────────────────────────
    logger.info("Step 5/6: Initializing model...")
    
    if model_type == "transformer":
        model = ICUTransformer(input_size=input_size).to(device)
        model_path = config.TRANSFORMER_MODEL_PATH
    else:
        model = ICUBiLSTM(input_size=input_size).to(device)
        model_path = config.LSTM_MODEL_PATH
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Class weighting for imbalanced labels
    train_labels = get_all_labels(loaders["train"])
    class_weights = compute_class_weights(train_labels)
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Note: Since our model already applies sigmoid, we use BCELoss instead
    criterion = nn.BCELoss(weight=None)
    # Actually, let's use a custom approach: weighted BCE
    pos_w = class_weights[1].item()
    
    def weighted_bce(pred, target):
        """BCE with class weighting to handle label imbalance."""
        weights = torch.where(target == 1, pos_w, 1.0)
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        return (bce * weights).mean()
    
    criterion = weighted_bce
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.LR_SCHEDULER_PATIENCE,
        factor=config.LR_SCHEDULER_FACTOR,
        # Phase 1 bugfix: `verbose` was removed from ReduceLROnPlateau in the
        # installed torch version (2.13) — pre-existing incompatibility
        # surfaced only by actually running training, not by this phase's
        # data-layer changes. LR changes are still visible via the per-epoch
        # `LR:` field already logged below.
    )
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # ─── Training Loop ───────────────────────────────────────────────
    logger.info("Step 6/6: Training...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, loaders["train"], optimizer, criterion, device, model_type
        )
        
        # Validate
        val_metrics = validate(
            model, loaders["val"], criterion, device, model_type
        )
        
        # Update scheduler
        scheduler.step(val_metrics["loss"])
        
        # Log
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"{elapsed:.1f}s"
        )
        
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "input_size": input_size,
                "feature_cols": feature_cols,
                "model_type": model_type,
            }, model_path)
            logger.info(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("=" * 60)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)
    
    return {
        "model": model,
        "history": history,
        "feature_cols": feature_cols,
        "loaders": loaders,
        "device": device,
        "model_type": model_type,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICU Risk Prediction Model")
    parser.add_argument(
        "--model", type=str, default="lstm", choices=["lstm", "transformer"],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic data for demo"
    )
    args = parser.parse_args()
    
    data_path = None
    if args.demo:
        data_path = "NONEXISTENT_PATH"  # Forces synthetic data generation
    
    result = train_model(
        model_type=args.model,
        data_path=data_path,
        num_epochs=args.epochs,
    )
