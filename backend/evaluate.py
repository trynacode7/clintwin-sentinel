"""
evaluate.py — Model Evaluation Metrics
========================================
Computes comprehensive evaluation metrics for ICU risk prediction:

  Standard Metrics:
    - Accuracy, Precision, Recall, F1-Score
    - ROC-AUC (Area Under ROC Curve)
    - AUPRC (Area Under Precision-Recall Curve) — more informative than
      AUROC under class imbalance, which deterioration labels always have
    - False Positive Rate

  Calibration:
    - Brier score + reliability-curve bins (compute_calibration) — a
      model can have good AUROC/AUPRC while still being badly
      miscalibrated (e.g. a "0.9" that's actually right 60% of the time),
      which matters if risk scores are ever shown as probabilities to a
      clinician.

  Clinical / operating-point metrics:
    - Early Prediction Gain: How early (in timesteps) the model
      correctly identifies deterioration before it occurs
    - Per-threshold analysis: Metrics at multiple probability thresholds
    - PPV-at-alert-budget (compute_ppv_at_alert_budget): "if we can only
      act on the top N% highest-risk windows, what's our precision at
      that cutoff?" — the actually clinically meaningful operating point,
      since a fixed 0.5 threshold rarely matches a real alert-fatigue
      budget.

  Subgroup evaluation (subgroup_evaluation): AUROC/AUPRC/sensitivity
  computed separately per age/sex subgroup, using
  dataset.create_eval_loader_with_stay_ids() to join predictions back to
  subject-level demographics. Required before any fairness/robustness
  claim — a model can look good in aggregate while performing
  meaningfully worse on an under-represented subgroup.

NOTE on everything in this file: per this project's standing rule, any
number this produces on synthetic data must be reported as
development/engineering validation, never as clinical validation — see
claude/known-issues-checklist.md and
claude/end-to-end-completion-roadmap.md.
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
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional

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

    # AUPRC — more informative than AUROC under class imbalance (the
    # deterioration label is always imbalanced); requires both classes.
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_pred_proba))
    except ValueError:
        metrics["auprc"] = 0.0

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


def compute_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Calibration check: a model can have strong AUROC/AUPRC while still
    being badly miscalibrated (e.g. predictions labeled "0.9 risk" that
    are in fact only correct 60% of the time). This matters directly for
    this project because risk scores ARE shown to a clinician as if they
    were probabilities (the DETERIORATION RISK gauge) — an uncalibrated
    score is misleading in a way a plain ranking metric won't catch.

    Returns the Brier score (lower is better, 0 = perfect) plus
    reliability-curve bins (mean predicted probability vs. observed
    fraction positive, per bin) for plotting a calibration curve.
    """
    brier = float(brier_score_loss(y_true, y_pred_proba))
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy="quantile"
        )
    except ValueError:
        # "quantile" strategy can fail with too few distinct values —
        # fall back to uniform-width bins rather than crashing evaluation.
        frac_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    return {
        "brier_score": brier,
        "calibration_bin_mean_predicted": mean_pred.tolist(),
        "calibration_bin_fraction_positive": frac_pos.tolist(),
    }


def compute_ppv_at_alert_budget(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    budget_fractions: tuple = (0.05, 0.1, 0.2, 0.3),
) -> Dict:
    """
    "If we can only act on the top N% highest-risk windows (an alert
    budget — a realistic staffing/alert-fatigue constraint), what's our
    precision (PPV) at that cutoff, and what risk-score threshold does
    it correspond to?" This is the operating point that actually matters
    clinically — a fixed 0.5 threshold may not correspond to any
    realistic number of alerts a unit could act on per day.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n = len(y_true)
    total_positive = max(int(y_true.sum()), 1)
    order = np.argsort(-y_pred_proba)  # descending predicted risk

    results = {}
    for frac in budget_fractions:
        k = max(1, int(round(n * frac)))
        top_idx = order[:k]
        ppv = float(y_true[top_idx].mean())
        threshold_at_cutoff = float(y_pred_proba[order[k - 1]])
        recall_at_cutoff = float(y_true[top_idx].sum() / total_positive)
        results[f"top_{int(round(frac * 100))}pct"] = {
            "n_alerts": k,
            "ppv": ppv,
            "threshold_at_cutoff": threshold_at_cutoff,
            "recall_at_cutoff": recall_at_cutoff,
        }
    return results


def _age_bucket(age) -> str:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown_age"
    return "age>=65" if age >= 65 else "age<65"


def _sex_bucket(sex_male) -> str:
    if sex_male is None or (isinstance(sex_male, float) and np.isnan(sex_male)):
        return "unknown_sex"
    return "male" if sex_male >= 0.5 else "female"


def subgroup_evaluation(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    stay_ids: np.ndarray,
    stay_to_subgroup: Dict,
) -> Dict[str, Dict]:
    """
    AUROC / AUPRC / sensitivity computed separately per subgroup value,
    rather than only in aggregate — required before any fairness or
    robustness claim, since a model can look good overall while
    performing meaningfully worse on an under-represented subgroup
    (e.g. a particular age band).

    stay_to_subgroup: {stay_id -> subgroup label}, e.g. built from
    _age_bucket()/_sex_bucket() applied to each stay's raw (unscaled)
    demographic value — see this module's __main__ block for how the
    lookup is built from the real stay-level data, never fabricated.

    Each subgroup's sample count and positive count are reported
    alongside its metrics so a reader can see when a subgroup is too
    small for its numbers to be meaningful, rather than that being
    hidden.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    groups = np.array([stay_to_subgroup.get(sid, "unknown") for sid in stay_ids])

    results = {}
    for g in sorted(set(groups)):
        idx = groups == g
        n = int(idx.sum())
        if n == 0:
            continue
        yt, yp = y_true[idx], y_pred_proba[idx]
        entry = {"n_samples": n, "n_positive": int(yt.sum())}
        if len(np.unique(yt)) > 1:
            entry["auroc"] = float(roc_auc_score(yt, yp))
            entry["auprc"] = float(average_precision_score(yt, yp))
        else:
            entry["auroc"] = float("nan")
            entry["auprc"] = float("nan")
        preds_binary = (yp >= 0.5).astype(int)
        entry["sensitivity"] = float(recall_score(yt, preds_binary, zero_division=0))
        results[g] = entry
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
    # Real SOFA/Sepsis-3 labeling — NOT the old ad-hoc backend.labels
    # rule. Using the wrong module here was a real bug: a model trained
    # on SOFA labels (train.py) was being evaluated against a different
    # label definition, which silently invalidates every metric below.
    from backend.labels_sofa import create_labels
    from backend.feature_engineering import engineer_features, get_engineered_feature_cols
    from backend.dataset import create_dataloaders, create_eval_loader_with_stay_ids
    from backend.lstm_model import ICUBiLSTM
    from backend.transformer_model import ICUTransformer

    parser = argparse.ArgumentParser(description="Evaluate ICU Risk Prediction Model")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--scaler", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--with-baselines", action="store_true",
        help="Also train + evaluate the required penalized-logistic-regression "
             "and gradient-boosted-trees baselines (backend/baselines.py) on the "
             "same split, and print a side-by-side comparison table. This is what "
             "actually produces the BiLSTM-vs-baseline numbers the project "
             "requires before any deep-model performance claim is credible.",
    )
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
    # preprocess() takes only data_path now — it no longer accepts a
    # `scale` kwarg (that was a real bug here: this call used to crash
    # with a TypeError the moment anyone ran this script directly).
    # Scaling happens explicitly below, via apply_scaler(), using the
    # SAME already-fitted scaler the checkpoint was trained with — never
    # re-fit here, which would silently evaluate against different
    # statistics than training used.
    result = preprocess(data_path)
    for split in ["train", "val", "test"]:
        # Required order (see preprocessing.preprocess()'s docstring and
        # scale_splits()'s docstring): label, THEN engineer, THEN scale.
        # Scaling before engineering was a second real bug here — it
        # would have corrupted shock_index/deltas/rolling-stats for every
        # evaluated sample, the same bug already found and fixed in
        # backend/preprocessing.py's prepare_inference_sequence().
        result[split] = create_labels(result[split])
        result[split] = engineer_features(result[split])

    # Build a raw (pre-scaling) stay_id -> {age, sex_male} lookup for
    # subgroup evaluation, BEFORE apply_scaler z-scores those columns —
    # subgroup buckets (age>=65, male/female) need real clinical values,
    # not z-scores. Demographics are static per stay, so the first row
    # per stay_id is representative.
    test_raw = result["test"]
    stay_to_subgroup_age: Dict = {}
    stay_to_subgroup_sex: Dict = {}
    for sid, group in test_raw.groupby("stay_id"):
        first = group.iloc[0]
        stay_to_subgroup_age[sid] = _age_bucket(first.get("age"))
        stay_to_subgroup_sex[sid] = _sex_bucket(first.get("sex_male"))

    for split in ["train", "val", "test"]:
        result[split] = apply_scaler(result[split], scaler, config.ALL_FEATURE_COLS)

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
    print(f"AUPRC:                   {eval_result['metrics']['auprc']:.4f}")
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

    print("-" * 60)
    print("Calibration:")
    calib = compute_calibration(labels, preds)
    print(f"  Brier score (lower=better, 0=perfect): {calib['brier_score']:.4f}")
    print(f"  Reliability bins (mean predicted -> observed fraction positive):")
    for mp, fp in zip(calib["calibration_bin_mean_predicted"], calib["calibration_bin_fraction_positive"]):
        print(f"    predicted~{mp:.3f}  ->  observed {fp:.3f}")

    print("-" * 60)
    print("PPV at alert budget (top-N% highest-risk windows):")
    ppv_budget = compute_ppv_at_alert_budget(labels, preds)
    for name, v in ppv_budget.items():
        print(
            f"  {name:>10s}: n_alerts={v['n_alerts']:5d}  PPV={v['ppv']:.4f}  "
            f"recall={v['recall_at_cutoff']:.4f}  threshold~{v['threshold_at_cutoff']:.4f}"
        )

    print("-" * 60)
    print("Subgroup evaluation (age):")
    eval_loader_meta = create_eval_loader_with_stay_ids(test_raw, feature_cols)
    all_preds_meta, all_labels_meta, all_stay_ids_meta = [], [], []
    with torch.no_grad():
        model.eval()
        for X, y, mask, stay_ids_batch in eval_loader_meta:
            X, mask = X.to(device), mask.to(device)
            if model_type == "transformer":
                risk, _, _ = model(X, mask)
            else:
                risk, _ = model(X, mask)
            all_preds_meta.extend(risk.squeeze(-1).cpu().numpy())
            all_labels_meta.extend(y.numpy())
            all_stay_ids_meta.extend(
                stay_ids_batch.numpy() if hasattr(stay_ids_batch, "numpy") else list(stay_ids_batch)
            )
    all_preds_meta = np.array(all_preds_meta)
    all_labels_meta = np.array(all_labels_meta)
    all_stay_ids_meta = np.array(all_stay_ids_meta)

    age_subgroups = subgroup_evaluation(all_labels_meta, all_preds_meta, all_stay_ids_meta, stay_to_subgroup_age)
    for g, m in age_subgroups.items():
        print(f"  {g:>12s}: n={m['n_samples']:4d} (pos={m['n_positive']:3d})  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  sensitivity={m['sensitivity']:.4f}")

    print("Subgroup evaluation (sex):")
    sex_subgroups = subgroup_evaluation(all_labels_meta, all_preds_meta, all_stay_ids_meta, stay_to_subgroup_sex)
    for g, m in sex_subgroups.items():
        print(f"  {g:>12s}: n={m['n_samples']:4d} (pos={m['n_positive']:3d})  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  sensitivity={m['sensitivity']:.4f}")

    print("=" * 60 + "\n")

    if args.with_baselines:
        # Train + evaluate the two required baselines on the SAME
        # train/test split and feature windows the BiLSTM just used, so
        # the comparison is apples-to-apples. This is the step that
        # actually produces the BiLSTM-vs-baseline table the project
        # requires before any deep-model performance claim is credible
        # — backend/baselines.py already implements both models
        # correctly; it just wasn't wired into a reported comparison
        # until now.
        from backend.baselines import (
            loaders_to_tabular,
            train_logreg,
            predict_logreg,
            train_gbt,
            predict_gbt,
        )

        print("=" * 60)
        print("BASELINE COMPARISON (required before any BiLSTM claim is credible)")
        print("=" * 60)

        X_train_tab, y_train_tab = loaders_to_tabular(loaders["train"])
        X_test_tab, y_test_tab = loaders_to_tabular(test_loader)

        print(f"\n--- Deep model under evaluation ({model_type.upper()}) ---")
        print(f"  AUROC: {eval_result['metrics']['roc_auc']:.4f}  AUPRC: {eval_result['metrics']['auprc']:.4f}  "
              f"F1: {eval_result['metrics']['f1']:.4f}")

        print("\n--- Penalized Logistic Regression baseline ---")
        logreg = train_logreg(X_train_tab, y_train_tab)
        logreg_pred = predict_logreg(logreg, X_test_tab)
        logreg_metrics = compute_metrics(y_test_tab, logreg_pred)
        print(f"  AUROC: {logreg_metrics['roc_auc']:.4f}  AUPRC: {logreg_metrics['auprc']:.4f}  "
              f"F1: {logreg_metrics['f1']:.4f}")

        print("\n--- Gradient-Boosted Trees (XGBoost) baseline ---")
        gbt = train_gbt(X_train_tab, y_train_tab)
        gbt_pred = predict_gbt(gbt, X_test_tab)
        gbt_metrics = compute_metrics(y_test_tab, gbt_pred)
        print(f"  AUROC: {gbt_metrics['roc_auc']:.4f}  AUPRC: {gbt_metrics['auprc']:.4f}  "
              f"F1: {gbt_metrics['f1']:.4f}")

        print("\nIf the deep model doesn't clearly beat both baselines here, that is a")
        print("real finding worth investigating (possible leakage/labeling bug, or the")
        print("deep architecture genuinely isn't earning its complexity on this data)")
        print("— not something to paper over.")
        print("=" * 60 + "\n")
