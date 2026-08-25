"""
explainability.py — Feature Attribution + Natural Language Explanations
==========================================================================
Provides model-agnostic and model-specific explanations.

Phase D naming fix (see PHASE_D_REPORT.md): this module previously
called everything "SHAP" regardless of which method actually ran.
`compute_shap_values()` genuinely attempts real SHAP (`shap.
KernelExplainer`) but silently fell back to a gradient×input
approximation on ANY exception — including the `shap` package not
being importable — while still returning the result as if it were
"SHAP values", with no way for the caller to tell which one they
actually got. That is exactly the misleading terminology this project
now explicitly forbids. Fixed as follows:

  - `compute_shap_values()` now returns `(values, method)`, where
    `method` is the literal string `"shap_kernel_explainer"` (real
    Shapley values) or `"gradient_x_input_fallback"` (not SHAP) — the
    caller always knows which one it received; nothing downstream may
    silently assume "SHAP" from this function's name alone.
  - `_gradient_shap_approximation()` — the function whose own NAME
    contained "shap" despite explicitly not being SHAP — is renamed to
    `gradient_x_input_attribution()`. This is the function actually
    used on every live prediction path (backend/inference_service.py,
    and the legacy POST /predict endpoint) — SHAP itself (KernelExplainer)
    is never invoked on any live path; it is exercised only by
    `compute_shap_values()`, which is currently dead code (no live
    caller — see PHASE_D_REPORT.md).
  - Every downstream field/parameter that used to say "shap_value(s)"
    generically (regardless of which method produced it) is renamed to
    "attribution_value(s)" — see compute_global_importance(),
    compute_local_explanation(), generate_clinical_explanation(),
    explain_prediction().

  Local/global feature attribution:
    - `gradient_x_input_attribution()`: real input×gradient saliency —
      the actual method behind every live explanation in this system.
    - `compute_shap_values()`: attempts real Shapley-value SHAP via
      `shap.KernelExplainer`; falls back to `gradient_x_input_attribution()`
      on failure, and always reports which one ran. Not used on any live
      path currently (dead code, kept for future use — see report).
    - Global feature importance = mean |attribution value| across all
      samples and timesteps.
    - Local per-sample explanations = signed attribution values.

  Natural Language Explanations:
    Generates human-readable clinical explanations like:
    "Risk increased due to rising lactate, declining MAP,
     and sustained tachycardia."

    Maps feature importance back to clinical context.
"""

import os
import sys
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")

# Feature name to clinical description mapping
CLINICAL_DESCRIPTIONS = {
    "heart_rate": "heart rate",
    "map": "mean arterial pressure (MAP)",
    "resp_rate": "respiratory rate",
    "spo2": "oxygen saturation (SpO2)",
    "temp": "body temperature",
    "lactate": "lactate level",
    "wbc": "white blood cell count",
    "heart_rate_delta": "change in heart rate",
    "map_delta": "change in MAP",
    "resp_rate_delta": "change in respiratory rate",
    "spo2_delta": "change in SpO2",
    "temp_delta": "change in temperature",
    "lactate_delta": "change in lactate",
    "wbc_delta": "change in WBC",
    "shock_index": "shock index (HR/MAP)",
    "heart_rate_rolling_mean": "trending heart rate",
    "map_rolling_mean": "trending MAP",
    "lactate_rolling_mean": "trending lactate",
    "heart_rate_rolling_std": "heart rate variability",
    "map_rolling_std": "MAP instability",
    "time_since_admission": "time in ICU",
    "heart_rate_missing": "missing heart rate data",
    "lactate_missing": "missing lactate data",
    "lactate_time_since_obs": "time since last lactate measurement",
}

# Directional descriptions for increasing/decreasing trends
DIRECTION_TEMPLATES = {
    "heart_rate": {"high": "elevated heart rate (tachycardia)", "low": "low heart rate (bradycardia)"},
    "map": {"high": "elevated MAP", "low": "declining MAP (hypotension)"},
    "resp_rate": {"high": "elevated respiratory rate (tachypnea)", "low": "low respiratory rate"},
    "spo2": {"high": "normal oxygen saturation", "low": "declining oxygen saturation (hypoxemia)"},
    "temp": {"high": "elevated temperature (fever)", "low": "hypothermia"},
    "lactate": {"high": "rising lactate (tissue hypoxia)", "low": "improving lactate"},
    "wbc": {"high": "elevated white blood cells (leukocytosis)", "low": "low white blood cells"},
    "shock_index": {"high": "worsening shock index", "low": "improving shock index"},
}


def compute_shap_values(
    model: torch.nn.Module,
    X: torch.Tensor,
    background: torch.Tensor,
    mask: torch.Tensor = None,
    model_type: str = "lstm",
) -> Tuple[np.ndarray, str]:
    """
    Attempt REAL Shapley-value SHAP (shap.KernelExplainer) for a batch of
    inputs; fall back to gradient×input attribution on any failure.

    Phase D fix: this function used to return just an array and let the
    caller assume it was always "SHAP" — even on the fallback path. It
    now returns `(values, method)` so the caller always knows which one
    it actually got and can label it honestly downstream. Never
    silently relabel a `"gradient_x_input_fallback"` result as SHAP.

    Args:
        model: Trained model
        X: Input samples [n_samples, seq_len, n_features]
        background: Background samples for SHAP [n_bg, seq_len, n_features]
        mask: Padding mask [n_samples, seq_len]
        model_type: "lstm" or "transformer"

    Returns:
        (attribution_values [n_samples, seq_len, n_features], method)
        where method is "shap_kernel_explainer" or "gradient_x_input_fallback".
    """
    model.eval()
    device = next(model.parameters()).device

    try:
        import shap

        # Wrapper function for SHAP
        def model_fn(x_np):
            x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
            m = torch.ones(x_t.shape[0], x_t.shape[1]).to(device)
            with torch.no_grad():
                if model_type == "transformer":
                    risk, _, _ = model(x_t, m)
                else:
                    risk, _ = model(x_t, m)
            return risk.squeeze(-1).cpu().numpy()

        # Use KernelExplainer (model-agnostic, genuine Shapley values)
        bg_np = background.cpu().numpy()
        x_np = X.cpu().numpy()

        # Reshape for SHAP: [samples, seq_len * features]
        bg_flat = bg_np.reshape(bg_np.shape[0], -1)
        x_flat = x_np.reshape(x_np.shape[0], -1)

        def model_fn_flat(x_flat):
            x_3d = x_flat.reshape(-1, X.shape[1], X.shape[2])
            return model_fn(x_3d)

        explainer = shap.KernelExplainer(model_fn_flat, bg_flat[:50])
        shap_values_flat = explainer.shap_values(x_flat, nsamples=100)

        # Reshape back to [samples, seq_len, features]
        attribution_values = shap_values_flat.reshape(x_np.shape)

        return attribution_values, "shap_kernel_explainer"

    except (ImportError, Exception) as e:
        logger.warning(f"Real SHAP computation failed ({e}). Falling back to gradient×input attribution — NOT SHAP.")
        return gradient_x_input_attribution(model, X, mask, model_type, device), "gradient_x_input_fallback"


def gradient_x_input_attribution(
    model: torch.nn.Module,
    X: torch.Tensor,
    mask: torch.Tensor,
    model_type: str,
    device: torch.device,
) -> np.ndarray:
    """
    Real input×gradient saliency — the actual attribution method behind
    every live explanation in this system (backend/inference_service.py
    and the legacy POST /predict endpoint both call this directly).

    Phase D rename: this function's own previous name,
    `_gradient_shap_approximation`, itself said "SHAP" despite this
    explicitly NOT being a Shapley value — exactly the kind of
    terminology this project now forbids. This is a real, useful,
    genuinely-computed attribution method; it just isn't SHAP, and its
    name should never have implied it was. It is equivalent to
    first-order SHAP for linear models and provides a reasonable
    approximation for neural networks, but is not itself SHAP.
    """
    model.eval()
    X = X.to(device).requires_grad_(True)
    
    if mask is not None:
        mask = mask.to(device)
    else:
        mask = torch.ones(X.shape[0], X.shape[1]).to(device)
    
    if model_type == "transformer":
        risk, _, _ = model(X, mask)
    else:
        risk, _ = model(X, mask)
    
    risk_sum = risk.sum()
    risk_sum.backward()

    # Input × gradient attribution
    attribution = (X * X.grad).detach().cpu().numpy()

    return attribution


def compute_global_importance(
    attribution_values: np.ndarray,
    feature_cols: List[str],
    top_k: int = 15,
) -> List[Dict]:
    """
    Compute global feature importance from an attribution-values array.

    Works identically regardless of which method produced
    `attribution_values` (real SHAP or gradient×input) — this function
    itself makes no claim about which one it is; the caller is
    responsible for labeling that honestly (see inference_service.py's
    `explanation_method` field).

    Global importance = mean |attribution value| across all samples and
    timesteps.

    Args:
        attribution_values: [n_samples, seq_len, n_features]
        feature_cols: Feature names
        top_k: Number of top features to return

    Returns:
        List of {feature, importance} dicts, sorted by importance
    """
    # Mean absolute attribution across samples and timesteps
    mean_abs_attribution = np.abs(attribution_values).mean(axis=(0, 1))  # [n_features]

    # Sort by importance
    importance = []
    for i, col in enumerate(feature_cols):
        if i < len(mean_abs_attribution):
            importance.append({
                "feature": col,
                "importance": float(mean_abs_attribution[i]),
                "clinical_name": CLINICAL_DESCRIPTIONS.get(col, col),
            })

    importance.sort(key=lambda x: x["importance"], reverse=True)
    return importance[:top_k]


def compute_local_explanation(
    attribution_values: np.ndarray,
    feature_cols: List[str],
    sample_idx: int = 0,
    top_k: int = 10,
) -> List[Dict]:
    """
    Compute local explanation for a single sample.

    Returns per-feature attribution values (signed — positive pushes
    toward risk=1, negative pushes toward risk=0). See
    compute_global_importance() docstring re: method-agnosticism.
    """
    # Mean attribution across timesteps for this sample
    sample_attribution = attribution_values[sample_idx].mean(axis=0)  # [n_features]

    explanation = []
    for i, col in enumerate(feature_cols):
        if i < len(sample_attribution):
            explanation.append({
                "feature": col,
                "attribution_value": float(sample_attribution[i]),
                "direction": "risk_increasing" if sample_attribution[i] > 0 else "risk_decreasing",
                "clinical_name": CLINICAL_DESCRIPTIONS.get(col, col),
            })

    # Sort by absolute value
    explanation.sort(key=lambda x: abs(x["attribution_value"]), reverse=True)
    return explanation[:top_k]


def generate_clinical_explanation(
    attribution_values: np.ndarray,
    feature_cols: List[str],
    risk_score: float,
    sample_idx: int = 0,
    input_data: np.ndarray = None,
    top_k: int = 3,
) -> str:
    """
    Generate a natural language clinical explanation.

    Translates feature attribution values into human-readable clinical
    text. See compute_global_importance() docstring re: method-agnosticism
    — this function does not claim SHAP either.

    Example output:
    "Risk increased to 0.82 (Critical) due to rising lactate (tissue hypoxia),
     declining MAP (hypotension), and sustained tachycardia."

    Args:
        attribution_values: attribution values array
        feature_cols: Feature names
        risk_score: Predicted risk (0-1)
        sample_idx: Which sample to explain
        input_data: Raw input values for context
        top_k: Number of top factors to mention

    Returns:
        Natural language explanation string
    """
    local = compute_local_explanation(attribution_values, feature_cols, sample_idx, top_k=top_k + 5)

    # Get top risk-increasing factors
    risk_factors = [x for x in local if x["attribution_value"] > 0][:top_k]
    protective_factors = [x for x in local if x["attribution_value"] < 0][:2]
    
    # Determine status
    if risk_score >= 0.7:
        status = "Critical"
    elif risk_score >= 0.4:
        status = "Warning"
    else:
        status = "Stable"
    
    # Build explanation
    parts = []
    
    # Risk factors
    if risk_factors:
        factor_descriptions = []
        for f in risk_factors:
            base_feature = f["feature"].replace("_delta", "").replace("_rolling_mean", "").replace("_rolling_std", "")
            
            if base_feature in DIRECTION_TEMPLATES:
                if f["attribution_value"] > 0:
                    desc = DIRECTION_TEMPLATES[base_feature].get("high", f["clinical_name"])
                else:
                    desc = DIRECTION_TEMPLATES[base_feature].get("low", f["clinical_name"])
            else:
                desc = f["clinical_name"]
            
            factor_descriptions.append(desc)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_factors = []
        for fd in factor_descriptions:
            if fd not in seen:
                seen.add(fd)
                unique_factors.append(fd)
        
        if len(unique_factors) == 1:
            factors_text = unique_factors[0]
        elif len(unique_factors) == 2:
            factors_text = f"{unique_factors[0]} and {unique_factors[1]}"
        else:
            factors_text = ", ".join(unique_factors[:-1]) + f", and {unique_factors[-1]}"
        
        parts.append(
            f"Risk {'increased' if risk_score > 0.5 else 'assessed'} to {risk_score:.2f} "
            f"({status}) due to {factors_text}."
        )
    else:
        parts.append(f"Risk level: {risk_score:.2f} ({status}).")
    
    # Protective factors
    if protective_factors and risk_score < 0.7:
        pf_desc = [CLINICAL_DESCRIPTIONS.get(f["feature"], f["feature"]) for f in protective_factors[:2]]
        if len(pf_desc) == 1:
            parts.append(f"Stable {pf_desc[0]} is a positive indicator.")
        else:
            parts.append(f"Stable {pf_desc[0]} and {pf_desc[1]} are positive indicators.")
    
    return " ".join(parts)


def explain_prediction(
    model: torch.nn.Module,
    X: torch.Tensor,
    mask: torch.Tensor,
    feature_cols: List[str],
    background: torch.Tensor = None,
    model_type: str = "lstm",
) -> Dict:
    """
    Full explanation pipeline for a batch of predictions.

    Returns:
        Dictionary with attribution values (+ which method produced
        them), global importance, local explanations, and text
        explanations.
    """
    device = next(model.parameters()).device

    # Generate background if not provided
    if background is None:
        background = torch.randn_like(X[:min(50, len(X))])

    # Attempt real SHAP; falls back to gradient×input attribution and
    # discloses which one actually ran (see compute_shap_values docstring).
    attribution_values, attribution_method = compute_shap_values(model, X, background, mask, model_type)

    # Get risk predictions
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        mask_dev = mask.to(device) if mask is not None else None

        if model_type == "transformer":
            risk, z_t, attn = model(X_dev, mask_dev)
        else:
            risk, z_t = model(X_dev, mask_dev)
            attn = None

    risk_scores = risk.squeeze(-1).cpu().numpy()

    # Global importance
    global_importance = compute_global_importance(attribution_values, feature_cols)

    # Per-sample explanations
    explanations = []
    for i in range(min(len(X), 10)):  # Explain first 10 samples
        local = compute_local_explanation(attribution_values, feature_cols, i)
        text = generate_clinical_explanation(
            attribution_values, feature_cols, risk_scores[i], i, X[i].numpy()
        )
        explanations.append({
            "sample_idx": i,
            "risk_score": float(risk_scores[i]),
            "local_explanation": local,
            "text_explanation": text,
        })

    return {
        "attribution_values": attribution_values,
        "attribution_method": attribution_method,
        "global_importance": global_importance,
        "explanations": explanations,
        "attention_weights": [a.cpu().numpy() for a in attn] if attn else None,
    }


if __name__ == "__main__":
    # Quick test with random data
    from backend.lstm_model import ICUBiLSTM
    
    input_size = 50
    model = ICUBiLSTM(input_size=input_size)
    
    X = torch.randn(4, config.INPUT_WINDOW, input_size)
    mask = torch.ones(4, config.INPUT_WINDOW)
    feature_cols = [f"feature_{i}" for i in range(input_size)]
    
    result = explain_prediction(model, X, mask, feature_cols, model_type="lstm")
    
    print("Global importance:")
    for item in result["global_importance"][:5]:
        print(f"  {item['feature']}: {item['importance']:.4f}")
    
    print(f"\nText explanation:")
    print(f"  {result['explanations'][0]['text_explanation']}")
