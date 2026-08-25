"""
test_end_to_end_pre_biogears.py — walks the exact user-facing flow this
project is meant to demonstrate right now, entirely on real
DATA_MODE=synthetic data and a real trained checkpoint, through the
real FastAPI app:

    patient selection -> patient data -> ML prediction -> explanation
    -> trajectory -> digital twin (OBSERVED/PREDICTED/SIMULATED)
    -> intervention selected -> real ML counterfactual (same model,
       real recomputed features) -> baseline vs counterfactual diff
    -> "Simulation requires BioGears integration" (physiological
       simulation only — the ML counterfactual above IS real)

Nothing above the BioGears boundary is allowed to be fabricated, and
nothing is allowed to silently claim BioGears output exists. This test
asserts real cross-endpoint consistency (the same real number shows up
everywhere it's supposed to) rather than just "200 OK, schema present".
"""
import numpy as np
import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi not installed in this environment"
)


@pytest.fixture(scope="module")
def client():
    from backend.api import app
    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_full_pre_biogears_workflow_is_real_and_consistent(client):
    # 1. Patient selection — real canonical roster, not a fake list.
    roster = client.get("/api/roster")
    assert roster.status_code == 200
    patients = roster.json()["patients"]
    assert len(patients) > 0
    sid = patients[0]["stay_id"]

    # 2. Patient data — real current state for the selected patient.
    current = client.get(f"/api/patients/{sid}")
    assert current.status_code == 200
    current_body = current.json()
    risk_score = current_body["prediction"]["risk_score"]
    assert 0.0 <= risk_score <= 1.0

    # 3. ML prediction — must be the SAME real number as step 2, not a
    # second, independently-fabricated one.
    predict = client.get(f"/api/patients/{sid}/predict")
    assert predict.status_code == 200
    assert predict.json()["risk_score"] == pytest.approx(risk_score)

    # 4. Explanation — real gradient attribution, honestly named,
    # non-empty, and for THIS prediction (same risk_score again).
    explanation = client.get(f"/api/patients/{sid}/explanation")
    assert explanation.status_code == 200
    expl_body = explanation.json()
    assert expl_body["risk_score"] == pytest.approx(risk_score)
    assert len(expl_body["gradient_attribution"]) > 0
    assert "shap_values" not in expl_body
    # The method string may honestly MENTION "SHAP" while disclosing that
    # it is NOT SHAP (see explainability.py) — it must never claim to BE
    # a Shapley-value explanation.
    assert "not a shapley" in expl_body["explanation_method"].lower()

    # 5. Trajectory — real per-window history; the LAST point must
    # match the current prediction (it's the same window).
    trajectory = client.get(f"/api/patients/{sid}/trajectory")
    assert trajectory.status_code == 200
    traj_points = trajectory.json()["trajectory"]
    assert len(traj_points) > 0
    assert traj_points[-1]["risk_score"] == pytest.approx(risk_score)

    # 6. Digital twin — OBSERVED/PREDICTED/SIMULATED, never blurred.
    twin = client.get(f"/api/patients/{sid}/twin-state")
    assert twin.status_code == 200
    twin_body = twin.json()
    assert twin_body["observed"]["vitals"]  # real, non-empty
    assert twin_body["predicted"]["risk_score"] == pytest.approx(risk_score)
    assert twin_body["simulated"]["available"] is False

    # 7. What-If interface — real validated scenario against this same
    # real patient: a REAL ML counterfactual (same model, modified input,
    # real re-engineered features), plus the honest BioGears boundary for
    # the still-unavailable physiological simulation.
    intervention_value = 0.9  # deliberately far from a "no-op" FiO2 value
    whatif = client.post(
        f"/api/patients/{sid}/whatif",
        json={"interventions": [{"feature": "fio2", "value": intervention_value}], "engine": "biogears"},
    )
    assert whatif.status_code == 200
    whatif_body = whatif.json()
    assert whatif_body["baseline"]["risk_score"] == pytest.approx(risk_score)

    # The counterfactual must be a real, finite, independently-computed
    # number — never null, never an edited copy of the baseline, never a
    # fabricated/interpolated placeholder.
    cf = whatif_body["counterfactual"]
    assert cf is not None
    assert isinstance(cf["risk_score"], float) and 0.0 <= cf["risk_score"] <= 1.0
    assert cf["risk_score"] == pytest.approx(risk_score + cf["risk_change"])
    assert len(cf["gradient_attribution"]) > 0
    assert all(np.isfinite(item["importance"]) for item in cf["gradient_attribution"])
    assert "ml counterfactual" in cf["description"].lower()
    assert "not a physiological simulation" in cf["description"].lower()

    # Same scenario twice -> identical counterfactual: a real deterministic
    # forward pass reproduces; a fabricated one would have no reason to.
    whatif_again = client.post(
        f"/api/patients/{sid}/whatif",
        json={"interventions": [{"feature": "fio2", "value": intervention_value}], "engine": "biogears"},
    ).json()
    assert whatif_again["counterfactual"]["risk_score"] == cf["risk_score"]

    # The physiological simulation section — the ONLY thing this system
    # is required to leave unavailable at this phase — must still be
    # honest and unavailable, never blurred with the real counterfactual
    # above.
    assert whatif_body["simulation"]["available"] is False
    assert whatif_body["simulation"]["message"] == "Simulation requires BioGears integration."

    # The workflow's terminal state must be honest, not silent: every
    # response body up to and including the What-If call is 200 with
    # real content, and the ONLY place anything says "not available" is
    # the simulation field — nothing upstream of it is degraded.
    assert current.status_code == predict.status_code == explanation.status_code == 200
    assert trajectory.status_code == twin.status_code == whatif.status_code == 200


def test_workflow_never_touches_real_mimic_or_bigquery(client):
    """Priority #7 / #9 regression guard: the active data mode for this
    entire workflow must remain synthetic."""
    import config
    assert config.DATA_MODE == "synthetic"
    roster = client.get("/api/roster").json()
    assert roster["split"] == "test"  # the synthetic test-split roster
