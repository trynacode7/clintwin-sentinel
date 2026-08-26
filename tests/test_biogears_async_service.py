"""
test_biogears_async_service.py — Unit Tests for Async Job Execution & API
=========================================================================
Tests:
  1. ThreadPoolExecutor background execution lifecycle (queued -> running -> completed).
  2. Async API endpoints: POST /api/v1/biogears/simulate and GET /api/v1/biogears/jobs/{job_id}.
"""

import asyncio
import os
import sys
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api import app
from backend.biogears_service import job_manager

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "biogears_runs",
    "stable_6hour_results.csvResults.csv",
)


@pytest.mark.anyio
async def test_async_job_manager_lifecycle():
    """Verify background submission and synchronous execution in worker pool."""
    job_id = await job_manager.submit_simulation_job(
        csv_path=CSV_PATH,
        patient_id="test_async_patient_99",
        scenario_type="baseline",
        lab_mode="calibrated_prior",
    )
    assert job_id.startswith("bg_")

    # Poll until completed (timeout 10s)
    job = None
    for _ in range(20):
        await asyncio.sleep(0.5)
        job = job_manager.get_job(job_id)
        if job and job.status == "completed":
            break

    assert job is not None
    assert job.status == "completed"
    assert job.result is not None
    assert job.result["patient_id"] == "test_async_patient_99"
    assert job.result["latent_dimension"] == 256
    assert 0.0 <= job.result["risk_probability"] <= 1.0


def test_api_async_simulation_endpoints():
    """Verify FastAPI async simulation submission and polling endpoints."""
    client = TestClient(app)

    # 1. Submit simulation job
    resp = client.post(
        "/api/v1/biogears/simulate",
        json={
            "patient_id": "api_test_patient_01",
            "scenario_type": "baseline",
            "csv_path": CSV_PATH,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    job_id = data["job_id"]

    # 2. Poll job status
    poll_resp = client.get(f"/api/v1/biogears/jobs/{job_id}")
    assert poll_resp.status_code == 200
    poll_data = poll_resp.json()
    assert poll_data["job_id"] == job_id
    assert poll_data["status"] in ["queued", "running", "completed"]

    # 3. List jobs
    list_resp = client.get("/api/v1/biogears/jobs")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert any(j["job_id"] == job_id for j in list_data["jobs"])
