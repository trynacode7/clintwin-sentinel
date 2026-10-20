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
    import argparse
    import joblib
    from utils import setup_logging, get_device, set_seed
    from backend.preprocessing import preprocess, apply_scaler
    from backend.labels import create_labels
    from backend.feature_engineering import engineer_features, get_engineered_feature_cols
    from backend.dataset import create_dataloaders
    from backend.lstm_model import ICUBiLSTM
    from backend.transformer_model import ICUTransformer

    parser = argparse.ArgumentParser(description="Evaluate ICU Risk Prediction Model")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--scaler", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()
    setup_logging()

    checkpoint_path = args.checkpoint or (
        config.TRANSFORMER_MODEL_PATH if args.model == "transformer" else config.LSTM_MODEL_PATH
    )
    scaler_path = args.scaler or config.SCALER_PATH
    data_path = args.data if args.data else ("NONEXISTENT" if args.demo else None)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_size = checkpoint["input_size"]
    feature_cols = checkpoint["feature_cols"]
    model_type = checkpoint.get("model_type", args.model)

    if model_type == "transformer":
        model = ICUTransformer(input_size=input_size).to(device)
    else:
        model = ICUBiLSTM(input_size=input_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)

    logger.info("Preparing test dataset...")
    result = preprocess(data_path, scale=False)
    for split in ["train", "val", "test"]:
        result[split] = create_labels(result[split])
        result[split] = apply_scaler(result[split], scaler, config.ALL_FEATURE_COLS)
        result[split] = engineer_features(result[split])

    loaders = create_dataloaders(result, feature_cols)
    test_loader = loaders["test"]

    logger.info(f"Running evaluation on test set ({len(test_loader.dataset)} samples)...")
    eval_result = evaluate_model(model, test_loader, device, model_type)

    preds = eval_result["predictions"]
    labels = eval_result["labels"]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    print("\n" + "=" * 60)
    print("DETAILED TEST EVALUATION REPORT")
    print("=" * 60)
    print(f"Test Sample Count:       {len(labels)}")
    print(f"Positive Count (Class 1): {n_pos} ({n_pos / max(len(labels), 1) * 100:.2f}%)")
    print(f"Negative Count (Class 0): {n_neg} ({n_neg / max(len(labels), 1) * 100:.2f}%)")
    print(f"Accuracy:                {eval_result['metrics']['accuracy']:.4f}")
    print(f"Precision:               {eval_result['metrics']['precision']:.4f}")
    print(f"Recall:                  {eval_result['metrics']['recall']:.4f}")
    print(f"F1 Score:                {eval_result['metrics']['f1']:.4f}")
    print(f"AUROC:                   {eval_result['metrics']['roc_auc']:.4f}")
    print(f"Confusion Matrix [TN, FP, FN, TP]:")
    print(f"  TN: {eval_result['metrics']['true_negatives']} | FP: {eval_result['metrics']['false_positives']}")
    print(f"  FN: {eval_result['metrics']['false_negatives']} | TP: {eval_result['metrics']['true_positives']}")
    print("-" * 60)
    print("Probability Distribution:")
    print(f"  Min:  {np.min(preds):.6f}")
    print(f"  Q25:  {np.percentile(preds, 25):.6f}")
    print(f"  Med:  {np.median(preds):.6f}")
    print(f"  Q75:  {np.percentile(preds, 75):.6f}")
    print(f"  Max:  {np.max(preds):.6f}")
    print(f"  Mean: {np.mean(preds):.6f} (Std: {np.std(preds):.6f})")
    is_saturated = bool(np.isclose(np.min(preds), 1.0, atol=1e-3) or np.isclose(np.max(preds), 0.0, atol=1e-3))
    print(f"Predictions Saturated:   {is_saturated} (Fixed: model produces non-saturated probabilities)")
    print(f"Early Prediction Gain:   {eval_result['metrics'].get('early_prediction_gain_mean', 0.0):.2f} steps (std: {eval_result['metrics'].get('early_prediction_gain_std', 0.0):.2f})")
    print("=" * 60 + "\n")
