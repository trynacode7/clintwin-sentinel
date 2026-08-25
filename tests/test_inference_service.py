"""
test_inference_service.py — Phase B: execute (not just import) the real
inference path — feature-contract validation, canonical roster,
real per-window prediction/explanation/trajectory/twin-state — against
the small, fast synthetic cohort fixture (see conftest.py), and confirm
the fail-loud contracts (unknown stay_id, missing checkpoint, mismatched
feature contract) actually fail loudly rather than silently degrading.

Uses the REAL trained checkpoint at config.LSTM_MODEL_PATH — feature
column NAMES depend only on config.py's column definitions, not on
dataset size, so the checkpoint (trained on the full synthetic dataset)
is expected to satisfy the feature contract against this small fixture
too; that expectation is itself asserted below (test_feature_contract_
passes_against_small_fixture), not just assumed.
"""
import os

import numpy as np
import pytest
import torch

import config
from backend.inference_service import (
    InferenceService,
    FeatureContractError,
    ModelOutputCorrectnessError,
    NoTrainedModelError,
    UnknownStayIdError,
)


@pytest.fixture(scope="module")
def small_dataset_csv(tmp_path_factory, small_ml_dataset):
    d = tmp_path_factory.mktemp("inference_service_small")
    path = str(d / "ml_dataset.csv")
    small_ml_dataset.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def service(small_dataset_csv):
    return InferenceService(model_type="lstm", split="test", data_path=small_dataset_csv)


def test_feature_contract_passes_against_small_fixture(service):
    # Constructing `service` itself already ran the fail-loud contract
    # check inside _build_pipeline_and_roster(); reaching this line means
    # it passed. Assert the contract explicitly too, for a clear failure
    # message if this ever regresses.
    assert len(service.feature_cols) == 112
    assert service.feature_cols[:10] == [
        "heart_rate", "map", "resp_rate", "spo2", "temp",
        "lactate", "wbc", "creatinine", "bilirubin", "platelets",
    ]


def test_roster_is_real_and_nonempty(service):
    roster = service.get_roster()
    assert len(roster) > 0
    for p in roster:
        assert isinstance(p["stay_id"], int)
        assert isinstance(p["subject_id"], int)
        assert isinstance(p["hadm_id"], int)
        assert p["split"] == "test"
        # Every roster patient must have a real prediction attached —
        # never left as None (that would mean fabrication was needed to
        # fill a gap).
        assert p["latest_risk_score"] is not None
        assert 0.0 <= p["latest_risk_score"] <= 1.0
        assert p["latest_status"] in ("Stable", "Warning", "Critical")
        assert p["n_windows"] > 0


def test_prediction_is_real_forward_pass_output(service):
    sid = service.get_roster()[0]["stay_id"]
    pred = service.get_prediction(sid)

    assert 0.0 <= pred["risk_score"] <= 1.0
    assert pred["status"] in ("Stable", "Warning", "Critical")
    assert pred["model_type"] == "lstm"
    assert pred["model_checkpoint"] == os.path.basename(config.LSTM_MODEL_PATH)
    # z_t must be a real 128-dim vector (LSTM_HIDDEN_SIZE * 2, bidirectional)
    assert len(pred["z_t"]) == config.LSTM_HIDDEN_SIZE * 2
    # Field name discipline — never "shap_values" on the new real path.
    assert "gradient_attribution" in pred
    assert "shap_values" not in pred
    assert "gradient_x_input" in pred["explanation_method"]


def test_explanation_reuses_real_gradient_attribution(service):
    sid = service.get_roster()[0]["stay_id"]
    exp = service.get_explanation(sid)
    assert len(exp["gradient_attribution"]) > 0
    assert all("feature" in item and "importance" in item for item in exp["gradient_attribution"])
    assert isinstance(exp["explanation_text"], str) and len(exp["explanation_text"]) > 0
    # Phase D: local (signed, directional) attribution uses "attribution_value",
    # never "shap_value" — the underlying computation is gradient×input, not SHAP.
    assert len(exp["local_gradient_attribution"]) > 0
    for item in exp["local_gradient_attribution"]:
        assert "attribution_value" in item
        assert "shap_value" not in item


def test_trajectory_has_one_real_point_per_window(service):
    sid = service.get_roster()[0]["stay_id"]
    n_windows = service.roster[sid].n_windows
    traj = service.get_trajectory(sid)
    assert len(traj) == n_windows
    # chronological, 0-indexed, contiguous
    assert [p["window_idx"] for p in traj] == list(range(n_windows))
    # every trajectory point must carry a real (non-placeholder) 2D
    # projection now that PCA is warmed on the whole roster up front —
    # guards against the [0.0, 0.0] cold-start artifact silently
    # reappearing for small cohorts.
    for p in traj:
        assert p["z_2d"] != [0.0, 0.0]


def test_twin_state_carries_real_disclaimer(service):
    sid = service.get_roster()[0]["stay_id"]
    twin_state = service.get_twin_state(sid)
    assert "not a physiological simulation" in twin_state["disclaimer"].lower()


def test_twin_state_separates_observed_predicted_simulated(service):
    """Pre-BioGears redesign: the twin state must never blur real vitals,
    the model's real prediction, and (not-yet-available) simulated
    physiology together — see digital_twin.py's module docstring."""
    sid = service.get_roster()[0]["stay_id"]
    twin_state = service.get_twin_state(sid)

    # OBSERVED: real vitals, keyed by real feature names, not model output.
    assert isinstance(twin_state["observed"]["vitals"], dict)
    assert len(twin_state["observed"]["vitals"]) > 0

    # PREDICTED: real z_t from this trained model, correct dimensionality.
    latent = twin_state["predicted"]["latent_state"]
    assert len(latent["z_t"]) == config.LSTM_HIDDEN_SIZE * 2
    assert len(latent["z_2d"]) == 2
    assert isinstance(twin_state["predicted"]["risk_score"], float)

    # SIMULATED: honestly reports unavailable, never a fabricated value.
    assert twin_state["simulated"]["available"] is False
    assert "not yet available" in twin_state["simulated"]["description"].lower()


def test_unknown_stay_id_raises_not_silently_fabricates(service):
    with pytest.raises(UnknownStayIdError):
        service.get_prediction(stay_id=-1)
    with pytest.raises(UnknownStayIdError):
        service.get_trajectory(stay_id=-1)


def test_unknown_window_idx_raises(service):
    sid = service.get_roster()[0]["stay_id"]
    n_windows = service.roster[sid].n_windows
    with pytest.raises(UnknownStayIdError):
        service.get_prediction(sid, window_idx=n_windows + 100)


def test_missing_checkpoint_raises_not_silently_falls_back(monkeypatch, small_dataset_csv):
    monkeypatch.setattr(config, "LSTM_MODEL_PATH", "/nonexistent/path/lstm_model.pt")
    with pytest.raises(NoTrainedModelError):
        InferenceService(model_type="lstm", split="test", data_path=small_dataset_csv)


def test_nonfinite_model_output_raises_not_silently_served(monkeypatch, small_dataset_csv):
    """
    Phase C output-correctness guard: if a real forward pass ever
    produces a NaN/Inf risk score (corrupted checkpoint, numerical
    blow-up, etc.), the service must fail loudly at the source rather
    than let a non-JSON-safe value reach an API response.
    """
    import backend.lstm_model as lstm_model

    real_forward = lstm_model.ICUBiLSTM.forward

    def _nan_forward(self, X, mask):
        risk, z_t = real_forward(self, X, mask)
        return risk * float("nan"), z_t

    monkeypatch.setattr(lstm_model.ICUBiLSTM, "forward", _nan_forward)

    with pytest.raises(ModelOutputCorrectnessError):
        InferenceService(model_type="lstm", split="test", data_path=small_dataset_csv)


def test_mismatched_feature_contract_raises(monkeypatch, tmp_path, small_dataset_csv):
    """
    A checkpoint whose saved feature_cols don't match the current
    pipeline output must be rejected loudly, never silently used with a
    misaligned input vector.
    """
    real_ckpt = torch.load(config.LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
    bad_ckpt = dict(real_ckpt)
    bad_ckpt["feature_cols"] = list(real_ckpt["feature_cols"])[:-1] + ["not_a_real_feature"]

    bad_path = str(tmp_path / "bad_lstm_model.pt")
    torch.save(bad_ckpt, bad_path)
    monkeypatch.setattr(config, "LSTM_MODEL_PATH", bad_path)

    with pytest.raises(FeatureContractError):
        InferenceService(model_type="lstm", split="test", data_path=small_dataset_csv)
