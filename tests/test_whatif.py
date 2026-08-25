"""
test_whatif.py — pre-BioGears What-If architecture (backend/whatif.py,
plus the /api/whatif/interventions and /api/patients/{id}/whatif routes
in backend/api.py).

Covers what IS real right now: request validation against real patient
data and the real intervenable-feature vocabulary, and a real baseline
attached to a scenario. Also covers the one thing this architecture
must NEVER do: fabricate a simulated result before BioGears exists.
"""
import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi not installed in this environment"
)

from backend.whatif import (
    INTERVENABLE_FEATURES,
    SUPPORTED_ENGINES,
    InterventionValidationError,
    validate_scenario_request,
    run_scenario,
    list_intervenable_features,
)


# ─── Unit-level: validate_scenario_request / run_scenario, no HTTP ─────────

KNOWN_IDS = {10001401, 10002401}


def test_valid_scenario_passes_validation():
    req = validate_scenario_request(
        stay_id=10001401,
        window_idx=None,
        interventions=[{"feature": "fio2", "value": 0.5}],
        engine="biogears",
        known_stay_ids=KNOWN_IDS,
    )
    assert req.stay_id == 10001401
    assert req.interventions[0].feature == "fio2"
    assert req.interventions[0].value == 0.5


def test_unknown_stay_id_rejected():
    with pytest.raises(InterventionValidationError, match="not in the canonical roster"):
        validate_scenario_request(
            stay_id=999999999,
            window_idx=None,
            interventions=[{"feature": "fio2", "value": 0.5}],
            engine="biogears",
            known_stay_ids=KNOWN_IDS,
        )


def test_non_intervenable_feature_rejected():
    """Measured vitals/labs are OBSERVED, not settable — e.g. heart_rate
    must never be accepted as an intervention target."""
    with pytest.raises(InterventionValidationError, match="not an intervenable feature"):
        validate_scenario_request(
            stay_id=10001401,
            window_idx=None,
            interventions=[{"feature": "heart_rate", "value": 90}],
            engine="biogears",
            known_stay_ids=KNOWN_IDS,
        )


def test_out_of_range_value_rejected():
    with pytest.raises(InterventionValidationError, match="outside the valid range"):
        validate_scenario_request(
            stay_id=10001401,
            window_idx=None,
            interventions=[{"feature": "fio2", "value": 5.0}],
            engine="biogears",
            known_stay_ids=KNOWN_IDS,
        )


def test_non_numeric_value_rejected():
    with pytest.raises(InterventionValidationError, match="must be numeric"):
        validate_scenario_request(
            stay_id=10001401,
            window_idx=None,
            interventions=[{"feature": "fio2", "value": "high"}],
            engine="biogears",
            known_stay_ids=KNOWN_IDS,
        )


def test_empty_interventions_rejected():
    with pytest.raises(InterventionValidationError, match="At least one intervention"):
        validate_scenario_request(
            stay_id=10001401,
            window_idx=None,
            interventions=[],
            engine="biogears",
            known_stay_ids=KNOWN_IDS,
        )


def test_duplicate_feature_rejected():
    with pytest.raises(InterventionValidationError, match="more than once"):
        validate_scenario_request(
            stay_id=10001401,
            window_idx=None,
            interventions=[
                {"feature": "fio2", "value": 0.5},
                {"feature": "fio2", "value": 0.6},
            ],
            engine="biogears",
            known_stay_ids=KNOWN_IDS,
        )


def test_unsupported_engine_rejected():
    with pytest.raises(InterventionValidationError, match="not supported"):
        validate_scenario_request(
            stay_id=10001401,
            window_idx=None,
            interventions=[{"feature": "fio2", "value": 0.5}],
            engine="model_reassessment",
            known_stay_ids=KNOWN_IDS,
        )


def test_intervenable_features_are_real_model_features():
    """INTERVENABLE_FEATURES must only reference real feature names the
    trained model actually uses — never an invented variable."""
    import config
    assert set(INTERVENABLE_FEATURES.keys()) <= set(config.ALL_FEATURE_COLS)
    assert set(INTERVENABLE_FEATURES.keys()) == set(
        config.TREATMENT_COLS + config.RESP_SUPPORT_COLS + config.VENT_COLS
    )


def test_run_scenario_never_fabricates_a_simulated_result():
    """The one property this whole module exists to guarantee: the
    PHYSIOLOGICAL SIMULATION section never returns a fake outcome — even
    though the ML COUNTERFACTUAL section now returns a real one."""
    req = validate_scenario_request(
        stay_id=10001401,
        window_idx=None,
        interventions=[{"feature": "vasopressor_dose", "value": 1.0}],
        engine="biogears",
        known_stay_ids=KNOWN_IDS,
    )
    baseline_prediction = {
        "window_idx": 3,
        "risk_score": 0.42,
        "status": "Warning",
        "observed_vitals": {"heart_rate": 90.0},
    }

    class _FakeCounterfactualService:
        """Stand-in that proves run_scenario() delegates the actual
        computation rather than computing (or fabricating) it itself."""
        def compute_counterfactual(self, stay_id, window_idx, interventions):
            assert stay_id == 10001401
            assert window_idx == 3
            assert interventions == [{"feature": "vasopressor_dose", "value": 1.0}]
            return {
                "risk_score": 0.77,
                "status": "Critical",
                "z_t": [0.1, 0.2],
                "z_2d": [0.1, 0.2],
                "gradient_attribution": [],
                "explanation_method": "gradient_x_input saliency (real)",
            }

    result = run_scenario(baseline_prediction, req, _FakeCounterfactualService())

    # Simulation (BioGears) section: never available, never fabricated.
    assert result["simulation"]["available"] is False
    assert result["simulation"]["message"] == "Simulation requires BioGears integration."
    # Baseline must be the REAL prediction passed in, not recomputed/invented.
    assert result["baseline"]["risk_score"] == 0.42
    assert result["baseline"]["status"] == "Warning"
    # Counterfactual section: exactly what compute_counterfactual() returned,
    # plus the derived risk_change — run_scenario() must not alter the number.
    assert result["counterfactual"]["risk_score"] == 0.77
    assert result["counterfactual"]["status"] == "Critical"
    assert result["counterfactual"]["risk_change"] == pytest.approx(0.77 - 0.42)


def test_list_intervenable_features_matches_dict():
    listed = list_intervenable_features()
    assert {f["feature"] for f in listed} == set(INTERVENABLE_FEATURES.keys())


# ─── HTTP-level: real FastAPI app, real InferenceService, real checkpoint ──

@pytest.fixture(scope="module")
def client():
    from backend.api import app
    with fastapi_testclient.TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_stay_id(client):
    r = client.get("/api/roster")
    assert r.status_code == 200
    return r.json()["patients"][0]["stay_id"]


def test_list_interventions_endpoint(client):
    r = client.get("/api/whatif/interventions")
    assert r.status_code == 200
    body = r.json()
    features = {f["feature"] for f in body["interventions"]}
    assert features == set(INTERVENABLE_FEATURES.keys())


def test_whatif_endpoint_valid_scenario(client, real_stay_id):
    r = client.post(
        f"/api/patients/{real_stay_id}/whatif",
        json={"interventions": [{"feature": "fio2", "value": 0.6}], "engine": "biogears"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stay_id"] == real_stay_id
    assert body["simulation"]["available"] is False

    # The ML counterfactual is REAL now — finite, present, and honestly
    # labeled (never claiming to be a physiological simulation).
    cf = body["counterfactual"]
    assert isinstance(cf["risk_score"], float) and 0.0 <= cf["risk_score"] <= 1.0
    assert cf["risk_score"] == pytest.approx(body["baseline"]["risk_score"] + cf["risk_change"])
    assert len(cf["gradient_attribution"]) > 0
    assert "not a shapley" in cf["explanation_method"].lower() or "gradient_x_input" in cf["explanation_method"].lower()
    # Must be honestly labeled as an ML counterfactual, never presented as
    # (or confused with) a BioGears/physiological simulation.
    assert "ml counterfactual" in cf["description"].lower()
    assert "not a physiological simulation" in cf["description"].lower()

    # Baseline risk score must match the real prediction endpoint for the
    # same stay/window — not a second, independently-computed number.
    pred = client.get(f"/api/patients/{real_stay_id}/predict").json()
    assert body["baseline"]["risk_score"] == pytest.approx(pred["risk_score"])


def test_whatif_counterfactual_is_reproducible(client, real_stay_id):
    """Same scenario, run twice -> byte-identical counterfactual. A real
    deterministic forward pass is reproducible; a fabricated/random result
    would not reliably be."""
    payload = {"interventions": [{"feature": "vasopressor_dose", "value": 2.0}], "engine": "biogears"}
    r1 = client.post(f"/api/patients/{real_stay_id}/whatif", json=payload)
    r2 = client.post(f"/api/patients/{real_stay_id}/whatif", json=payload)
    assert r1.json()["counterfactual"]["risk_score"] == r2.json()["counterfactual"]["risk_score"]
    assert r1.json()["counterfactual"]["gradient_attribution"] == r2.json()["counterfactual"]["gradient_attribution"]


def test_whatif_counterfactual_changes_with_intervention_value(client, real_stay_id):
    """Two materially different intervention values must not produce the
    identical counterfactual risk score — proves the result is a real
    function of the modified input, not a constant/no-op placeholder."""
    low = client.post(
        f"/api/patients/{real_stay_id}/whatif",
        json={"interventions": [{"feature": "fio2", "value": 0.21}], "engine": "biogears"},
    ).json()
    high = client.post(
        f"/api/patients/{real_stay_id}/whatif",
        json={"interventions": [{"feature": "fio2", "value": 1.0}], "engine": "biogears"},
    ).json()
    assert low["counterfactual"]["risk_score"] != high["counterfactual"]["risk_score"]


def test_whatif_counterfactual_matches_independent_manual_recomputation(client, real_stay_id):
    """The strongest available proof this is a REAL counterfactual and not
    fabricated/interpolated/edited: independently re-derive the expected
    result in this test (raw window -> apply intervention -> re-engineer
    features -> re-scale -> real model forward pass) using only the public
    building blocks, and assert it matches the API's answer exactly. If
    the endpoint ever started returning an edited number, an interpolated
    guess, or a hardcoded value instead of running the real pipeline, this
    test would fail."""
    import numpy as np
    import torch
    import config
    from backend.inference_service import get_inference_service
    from backend.feature_engineering import engineer_features
    from backend.preprocessing import apply_scaler
    from backend.explainability import gradient_x_input_attribution

    svc = get_inference_service()
    baseline = svc.get_prediction(real_stay_id)
    window_idx = baseline["window_idx"]
    feature, value = "vasopressor_dose", 3.0

    # Independent re-derivation, using only the same public pipeline
    # functions train.py itself uses — no call into compute_counterfactual().
    raw_stay = svc._raw_by_stay[real_stay_id].copy()
    n_steps = len(raw_stay)
    window_size = config.INPUT_WINDOW
    if n_steps < window_size:
        start, end = 0, n_steps
    else:
        start, end = window_idx * window_size, window_idx * window_size + window_size
    idx_labels = raw_stay.index[start:end]
    raw_stay.loc[idx_labels, feature] = value
    raw_stay.loc[idx_labels, f"{feature}_missing"] = 0.0
    raw_stay.loc[idx_labels, f"{feature}_time_since_obs"] = 0.0

    engineered = engineer_features(raw_stay)
    scaled = apply_scaler(engineered, svc.scaler, list(config.ALL_FEATURE_COLS))
    feats = scaled[svc.feature_cols].values.astype(np.float32)
    if n_steps < window_size:
        X = np.zeros((window_size, feats.shape[1]), dtype=np.float32)
        X[window_size - n_steps:] = feats
    else:
        X = feats[start:end].copy()
    X = np.nan_to_num(X, nan=0.0)

    X_t = torch.tensor(X).unsqueeze(0)
    mask = torch.ones(1, X_t.shape[1])
    svc.model.eval()
    with torch.no_grad():
        if svc.checkpoint_meta["model_type"] == "transformer":
            risk, _, _ = svc.model(X_t, mask)
        else:
            risk, _ = svc.model(X_t, mask)
    expected_risk = float(risk.squeeze().cpu().item())

    # Now hit the real API and assert it matches this independent derivation.
    r = client.post(
        f"/api/patients/{real_stay_id}/whatif",
        json={"interventions": [{"feature": feature, "value": value}], "engine": "biogears"},
    )
    assert r.json()["counterfactual"]["risk_score"] == pytest.approx(expected_risk, abs=1e-5)


def test_whatif_endpoint_rejects_bad_feature(client, real_stay_id):
    r = client.post(
        f"/api/patients/{real_stay_id}/whatif",
        json={"interventions": [{"feature": "spo2", "value": 95}]},
    )
    assert r.status_code == 400
    assert "not an intervenable feature" in r.json()["detail"]


def test_whatif_endpoint_rejects_unknown_stay_id(client):
    r = client.post(
        "/api/patients/999999999/whatif",
        json={"interventions": [{"feature": "fio2", "value": 0.5}]},
    )
    assert r.status_code == 400
    assert "not in the canonical roster" in r.json()["detail"]


def test_whatif_endpoint_rejects_out_of_range(client, real_stay_id):
    r = client.post(
        f"/api/patients/{real_stay_id}/whatif",
        json={"interventions": [{"feature": "ventilated", "value": 7}]},
    )
    assert r.status_code == 400
    assert "outside the valid range" in r.json()["detail"]
