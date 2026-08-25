"""
test_api_phase_b.py — execute (not just import) the real FastAPI app and
its Phase B endpoints end-to-end via HTTP, using the full trained
checkpoint + full synthetic dataset (config.DATA_PATH) — the exact
model/data pair the actual running API serves.

Slower than the rest of the suite (builds the real InferenceService once
per test session — real preprocessing + real forward passes across the
whole test-split roster) so it's isolated in its own module-scoped
fixture rather than re-built per test.
"""
import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi not installed in this environment"
)


@pytest.fixture(scope="module")
def client():
    from backend.api import app
    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_root_health_check(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_roster_endpoint_is_real(client):
    r = client.get("/api/roster")
    assert r.status_code == 200
    body = r.json()
    assert body["split"] == "test"
    assert len(body["patients"]) > 0
    for p in body["patients"]:
        assert isinstance(p["stay_id"], int)
        assert p["latest_risk_score"] is not None


def test_full_real_pipeline_for_one_patient(client):
    roster = client.get("/api/roster").json()["patients"]
    sid = roster[0]["stay_id"]

    current = client.get(f"/api/patients/{sid}")
    assert current.status_code == 200
    assert 0.0 <= current.json()["prediction"]["risk_score"] <= 1.0

    predict = client.get(f"/api/patients/{sid}/predict")
    assert predict.status_code == 200
    assert predict.json()["risk_score"] == current.json()["prediction"]["risk_score"]

    explanation = client.get(f"/api/patients/{sid}/explanation")
    assert explanation.status_code == 200
    assert "shap_values" not in explanation.json()
    assert len(explanation.json()["gradient_attribution"]) > 0

    trajectory = client.get(f"/api/patients/{sid}/trajectory")
    assert trajectory.status_code == 200
    assert len(trajectory.json()["trajectory"]) > 0

    twin_state = client.get(f"/api/patients/{sid}/twin-state")
    assert twin_state.status_code == 200
    assert "disclaimer" in twin_state.json()


def test_unknown_stay_id_returns_404_not_fabricated_data(client):
    r = client.get("/api/patients/999999999")
    assert r.status_code == 404


def test_legacy_patients_endpoint_returns_real_roster(client):
    r = client.get("/patients")
    assert r.status_code == 200
    roster_via_legacy = {p["stay_id"] for p in r.json()["patients"]}
    roster_via_new = {p["stay_id"] for p in client.get("/api/roster").json()["patients"]}
    assert roster_via_legacy == roster_via_new


def test_legacy_patient_by_id_endpoint_uses_real_stay_id(client):
    sid = client.get("/api/roster").json()["patients"][0]["stay_id"]
    r = client.get(f"/patient/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert "trajectory" in body and len(body["trajectory"]) > 0

    r = client.get("/patient/999999999")
    assert r.status_code == 404


def test_demo_endpoint_is_marked_deprecated(client):
    r = client.get("/demo")
    assert r.status_code == 200
    assert r.json()["deprecated"] is True


def test_legacy_predict_endpoint_still_runs_real_inference(client):
    payload = {
        "patient_data": [{"heart_rate": 90.0, "map": 70.0} for _ in range(6)],
        "patient_id": "phase_b_test_patient",
        "model_type": "lstm",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["status"] in ("Stable", "Warning", "Critical")
