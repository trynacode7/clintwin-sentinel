"""
evaluate.py — Model Evaluation Metrics
========================================
Computes comprehensive evaluation metrics for ICU risk prediction:

  Standard Metrics:
    - Accuracy, Precision, Recall, F1-Score
    - ROC-AUC (Area Under ROC Curve)
    - False Positive Rate
    
  Clinical Metrics:
    - Early Prediction Gain: How early (in timesteps) the model
      correctly identifies deterioration before it occurs
    - Per-threshold analysis: Metrics at multiple probability thresholds
"""

import os
import sys
import numpy as np
import torch
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_pred_proba: Predicted probabilities (0-1)
        threshold: Classification threshold
        
    Returns:
        Dictionary of metric name → value
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
    
    # ROC-AUC (requires both classes present)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["roc_auc"] = 0.0
    
    # False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["false_positive_rate"] = fp / max(fp + tn, 1)
    metrics["true_positive_rate"] = tp / max(tp + fn, 1)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    
    return metrics


def compute_early_prediction_gain(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Early Prediction Gain (EPG).
    
    Measures how many timesteps BEFORE the actual deterioration event
    the model first raises an alarm. Higher EPG means earlier warning.
    
    For each true positive sequence:
    - Find the first timestep where prediction > threshold
    - Find the corresponding deterioration onset
    - Gain = onset_time - first_alarm_time
    
    Args:
        y_true: Binary labels
        y_pred_proba: Predicted probabilities
        threshold: Alert threshold
        
    Returns:
        Dictionary with EPG metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Find transitions: 0 → 1 in true labels (deterioration onset)
    onset_indices = []
    for i in range(1, len(y_true)):
        if y_true[i] == 1 and y_true[i - 1] == 0:
            onset_indices.append(i)
    
    if not onset_indices:
        return {"early_prediction_gain_mean": 0.0, "early_prediction_gain_std": 0.0}
    
    gains = []
    for onset in onset_indices:
        # Look backward from onset for first alarm
        lookback = min(onset, config.INPUT_WINDOW)
        for dt in range(lookback):
            idx = onset - dt - 1
            if idx >= 0 and y_pred[idx] == 1:
                gains.append(dt + 1)  # +1 because dt=0 means 1 step early
                break
    
    if not gains:
        return {"early_prediction_gain_mean": 0.0, "early_prediction_gain_std": 0.0}
    
    return {
        "early_prediction_gain_mean": float(np.mean(gains)),
        "early_prediction_gain_std": float(np.std(gains)),
        "early_prediction_gain_max": float(np.max(gains)),
        "early_prediction_count": len(gains),
        "total_onsets": len(onset_indices),
    }


def threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[list] = None,
) -> list:
    """
    Compute metrics at multiple probability thresholds.
    
    Useful for finding the optimal operating point that balances
    sensitivity (catching deteriorations) vs. specificity (avoiding
    false alarms in the ICU).
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    for t in thresholds:
        metrics = compute_metrics(y_true, y_pred_proba, threshold=t)
        results.append(metrics)
    
    return results


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    model_type: str = "lstm",
) -> Dict:
    """
    Full model evaluation on a DataLoader.
    
    Returns:
        Dictionary with all metrics, predictions, and analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    for X, y, mask in loader:
        X = X.to(device)
        mask = mask.to(device)
        
        if model_type == "transformer":
            risk, z_t, attn = model(X, mask)
        else:
            risk, z_t = model(X, mask)
        
        all_preds.extend(risk.squeeze(-1).cpu().numpy())
        all_labels.extend(y.numpy())
    
    y_true = np.array(all_labels)
    y_pred_proba = np.array(all_preds)
    
    # Standard metrics
    metrics = compute_metrics(y_true, y_pred_proba)
    
    # Early prediction gain
    epg = compute_early_prediction_gain(y_true, y_pred_proba)
    metrics.update(epg)
    
    # Threshold analysis
    threshold_results = threshold_analysis(y_true, y_pred_proba)
    
    # Classification report
    y_pred = (y_pred_proba >= 0.5).astype(int)
    report = classification_report(
        y_true, y_pred, target_names=["Stable", "Deteriorating"], zero_division=0
    )
    
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info(f"FPR:         {metrics['false_positive_rate']:.4f}")
    logger.info(f"EPG (mean):  {metrics.get('early_prediction_gain_mean', 0):.1f} steps")
    logger.info("\n" + report)
    
    return {
        "metrics": metrics,
        "threshold_analysis": threshold_results,
        "predictions": y_pred_proba,
        "labels": y_true,
        "report": report,
    }


if __name__ == "__main__":
    from backend.train import train_model
    
    result = train_model(model_type="lstm", num_epochs=5)
    if result:
        eval_result = evaluate_model(
            result["model"],
            result["loaders"]["test"],
            result["device"],
            result["model_type"],
        )
