"""
baselines.py — Non-deep baselines required before BiLSTM/Transformer
performance means anything (TECHNICAL_BLUEPRINT.md section 4):

  1. Penalized logistic regression on last-observed feature values + deltas
     — approximates "what a clinician reading the chart would infer."
  2. Gradient-boosted trees (XGBoost) on per-window summary statistics
     (last, mean, std, approximate slope) — the standard strong tabular
     baseline in the clinical-deterioration literature.

Both consume the SAME (X, y, mask) windows dataset.py already produces —
no new windowing logic, no change to dataset.py. This module only adds a
window -> tabular-feature-vector reduction step used solely by these two
baselines; the BiLSTM/Transformer continue to consume the full sequence.

No performance numbers are hardcoded or assumed anywhere in this file —
whatever these models score is whatever they score on whatever data is
passed in (synthetic, dev-only, until Phase 2+).
"""

import os
import sys
import numpy as np
import logging
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")


def window_to_tabular(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Reduce a [n, seq_len, n_features] window batch to a
    [n, n_features * 4] tabular matrix of (last, mean, std, approx_slope)
    per feature, computed only over valid (mask==1) timesteps.

    approx_slope is (last_valid - first_valid) / (n_valid - 1) — a fast
    finite-difference approximation of a linear trend, not a fitted OLS
    slope (kept simple deliberately; this is a baseline, not the final model).
    """
    n, seq_len, n_features = X.shape
    last = np.zeros((n, n_features), dtype=np.float32)
    mean = np.zeros((n, n_features), dtype=np.float32)
    std = np.zeros((n, n_features), dtype=np.float32)
    slope = np.zeros((n, n_features), dtype=np.float32)

    for i in range(n):
        valid = np.where(mask[i] > 0)[0]
        if len(valid) == 0:
            continue
        xi = X[i, valid, :]
        last[i] = xi[-1]
        mean[i] = xi.mean(axis=0)
        std[i] = xi.std(axis=0) if len(valid) > 1 else 0.0
        slope[i] = (xi[-1] - xi[0]) / max(len(valid) - 1, 1)

    return np.concatenate([last, mean, std, slope], axis=1)


def loaders_to_tabular(loader) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize a DataLoader of (X, y, mask) batches into tabular (X_tab, y)."""
    X_parts, y_parts = [], []
    for X, y, mask in loader:
        X_np, mask_np = X.numpy(), mask.numpy()
        X_parts.append(window_to_tabular(X_np, mask_np))
        y_parts.append(y.numpy())
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def train_logreg(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # Logistic-regression baseline uses only the "last value" + "slope" blocks
    # (first half of last/mean, second half of std/slope) — i.e. last-observed
    # values and their trend, per TECHNICAL_BLUEPRINT.md section 4's
    # "last-observed feature values + deltas" description. mean/std blocks
    # are dropped for this baseline to keep it close to that description;
    # GBT (below) uses the full tabular representation.
    n_features = X_train.shape[1] // 4
    idx = np.concatenate([np.arange(0, n_features), np.arange(3 * n_features, 4 * n_features)])
    X_train_sub = X_train[:, idx]

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])
    pipe.fit(X_train_sub, y_train)
    return pipe, idx


def predict_logreg(model_and_idx, X):
    pipe, idx = model_and_idx
    return pipe.predict_proba(X[:, idx])[:, 1]


def train_gbt(X_train, y_train):
    import xgboost as xgb

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        missing=np.nan,
        random_state=config.SEED,
    )
    model.fit(X_train, y_train)
    return model


def predict_gbt(model, X):
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    from utils import setup_logging, set_seed
    from backend.preprocessing import preprocess, scale_splits
    from backend.feature_engineering import engineer_features, get_engineered_feature_cols
    from backend.labels_sofa import create_labels
    from backend.dataset import create_dataloaders
    from backend.evaluate import compute_metrics

    setup_logging()
    set_seed(config.SEED)

    result = preprocess()
    for split in ["train", "val", "test"]:
        result[split] = engineer_features(result[split])
        result[split] = create_labels(result[split])
    result = scale_splits(result)  # scale AFTER engineering + SOFA labeling — see preprocess() docstring

    feature_cols = get_engineered_feature_cols(result["train"])
    loaders = create_dataloaders(result, feature_cols)

    X_train_tab, y_train = loaders_to_tabular(loaders["train"])
    X_val_tab, y_val = loaders_to_tabular(loaders["val"])
    X_test_tab, y_test = loaders_to_tabular(loaders["test"])

    print(f"Tabular baseline features: {X_train_tab.shape[1]} (from {len(feature_cols)} sequence features)")

    print("\n=== Logistic Regression baseline ===")
    logreg = train_logreg(X_train_tab, y_train)
    val_pred = predict_logreg(logreg, X_val_tab)
    test_pred = predict_logreg(logreg, X_test_tab)
    print("val:", compute_metrics(y_val, val_pred))
    print("test:", compute_metrics(y_test, test_pred))

    print("\n=== Gradient-Boosted Trees baseline ===")
    gbt = train_gbt(X_train_tab, y_train)
    val_pred = predict_gbt(gbt, X_val_tab)
    test_pred = predict_gbt(gbt, X_test_tab)
    print("val:", compute_metrics(y_val, val_pred))
    print("test:", compute_metrics(y_test, test_pred))
