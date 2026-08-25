"""
test_explainability_naming.py — Phase D: verify the SHAP-vs-gradient
naming fix actually behaves as documented, not just that the code
imports.

compute_shap_values() is dead code on every live path (see
PHASE_D_REPORT.md) but is kept for future use, so it's tested directly
here rather than only indirectly through the live endpoints.
"""
import numpy as np
import torch

import config
from backend.lstm_model import ICUBiLSTM
from backend.explainability import (
    compute_shap_values,
    gradient_x_input_attribution,
    compute_global_importance,
    compute_local_explanation,
)


def _small_model_and_input():
    input_size = 20
    model = ICUBiLSTM(input_size=input_size)
    model.eval()
    X = torch.randn(2, config.INPUT_WINDOW, input_size)
    mask = torch.ones(2, config.INPUT_WINDOW)
    feature_cols = [f"feature_{i}" for i in range(input_size)]
    return model, X, mask, feature_cols


def test_gradient_x_input_attribution_is_real_and_finite():
    model, X, mask, feature_cols = _small_model_and_input()
    device = torch.device("cpu")
    attribution = gradient_x_input_attribution(model, X, mask, "lstm", device)
    assert attribution.shape == (X.shape[0], X.shape[1], X.shape[2])
    assert np.all(np.isfinite(attribution))
    # Not all-zero — a real gradient signal, not a placeholder.
    assert np.abs(attribution).sum() > 0


def test_compute_shap_values_discloses_which_method_actually_ran():
    """
    Whichever branch runs (real KernelExplainer or the gradient fallback),
    the caller must be told which one it got — never a bare array that
    silently implies "SHAP" regardless of what happened internally.
    """
    model, X, mask, feature_cols = _small_model_and_input()
    background = torch.randn(5, config.INPUT_WINDOW, X.shape[2])

    attribution_values, method = compute_shap_values(model, X, background, mask, "lstm")

    assert method in ("shap_kernel_explainer", "gradient_x_input_fallback")
    assert attribution_values.shape == (X.shape[0], X.shape[1], X.shape[2])
    assert np.all(np.isfinite(attribution_values))


def test_global_and_local_attribution_fields_are_honestly_named():
    model, X, mask, feature_cols = _small_model_and_input()
    device = torch.device("cpu")
    attribution = gradient_x_input_attribution(model, X, mask, "lstm", device)

    global_importance = compute_global_importance(attribution, feature_cols, top_k=5)
    assert len(global_importance) > 0
    for item in global_importance:
        assert "shap" not in str(item.keys()).lower()

    local = compute_local_explanation(attribution, feature_cols, sample_idx=0, top_k=5)
    assert len(local) > 0
    for item in local:
        assert "attribution_value" in item
        assert "shap_value" not in item
