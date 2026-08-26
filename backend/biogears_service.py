"""
biogears_service.py — Asynchronous BioGears Simulation & Inference Service
==========================================================================
Provides non-blocking simulation lifecycle management, job queuing, and
worker pool execution. Enables background BioGears execution and polling.
"""

import asyncio
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch

import config
from backend.biogears_adapter import BioGearsFeatureTransformer
from backend.biogears_inference import _instantiate_lstm_from_checkpoint
from backend.digital_twin import DigitalTwin

logger = logging.getLogger("icu_digital_twin")

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="BioGearsWorker")
_JOB_REGISTRY: Dict[str, "BioGearsJob"] = {}
_REGISTRY_LOCK = threading.Lock()


@dataclass
class BioGearsJob:
    job_id: str
    patient_id: str
    scenario_type: str
    status: str  # "queued", "running", "completed", "failed"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "patient_id": self.patient_id,
            "scenario_type": self.scenario_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "result": self.result,
        }


class BioGearsJobManager:
    """Manages asynchronous lifecycle of BioGears simulation and inference jobs."""

    def __init__(self, run_dir: str = "biogears_runs"):
        self.run_dir = Path(config.PROJECT_ROOT) / run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, patient_id: str, scenario_type: str = "baseline") -> str:
        job_id = f"bg_{uuid.uuid4().hex[:12]}"
        job = BioGearsJob(
            job_id=job_id,
            patient_id=patient_id,
            scenario_type=scenario_type,
            status="queued",
        )
        with _REGISTRY_LOCK:
            _JOB_REGISTRY[job_id] = job
        logger.info(f"Created BioGears job {job_id} for patient {patient_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[BioGearsJob]:
        with _REGISTRY_LOCK:
            return _JOB_REGISTRY.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with _REGISTRY_LOCK:
            jobs = list(_JOB_REGISTRY.values())
        return [j.to_dict() for j in jobs[-limit:]]

    def _execute_sync_pipeline(
        self,
        job_id: str,
        csv_path: Path,
        patient_id: str,
        lab_mode: str = "calibrated_prior",
    ) -> Dict[str, Any]:
        """Synchronous worker function: Feature Extraction -> BiLSTM -> Trajectory."""
        try:
            with _REGISTRY_LOCK:
                if job_id in _JOB_REGISTRY:
                    _JOB_REGISTRY[job_id].status = "running"
                    _JOB_REGISTRY[job_id].started_at = datetime.utcnow().isoformat()

            # 1. Load trained checkpoint for feature columns & weights
            checkpoint = torch.load(
                getattr(config, "BIOGEARS_LSTM_MODEL_PATH", config.LSTM_MODEL_PATH),
                map_location="cpu",
                weights_only=False,
            )
            feature_cols = checkpoint["feature_cols"]

            # 2. Transform CSV
            transformer = BioGearsFeatureTransformer(feature_cols=feature_cols)
            X, mask, featured_df = transformer.transform_csv(
                csv_path=csv_path,
                patient_id=patient_id,
                lab_mode=lab_mode,
            )

            # 3. Instantiate and run BiLSTM
            model = _instantiate_lstm_from_checkpoint(checkpoint)

            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                risk, z_t = model(X_tensor, mask_tensor)

            raw_risk = float(risk.item())

            # 4. Extract step-by-step trajectory
            trajectory = []
            with torch.no_grad():
                for t in range(X.shape[0]):
                    prefix = X[: t + 1]
                    px = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0)
                    pm = torch.ones((1, prefix.shape[0]), dtype=torch.float32)
                    r_t, z_step = model(px, pm)
                    trajectory.append(
                        {
                            "t_idx": t,
                            "risk_score": float(r_t.item()),
                            "latent_state": z_step.squeeze(0).cpu().numpy().tolist(),
                        }
                    )

            # 5. Project trajectory to 2D PCA space
            all_z = [np.array(pt["latent_state"], dtype=np.float32) for pt in trajectory]
            if len(all_z) >= 2:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2)
                pca_coords = pca.fit_transform(np.array(all_z))
                for idx, pt in enumerate(trajectory):
                    pt["z_2d"] = [float(pca_coords[idx, 0]), float(pca_coords[idx, 1])]
            else:
                for pt in trajectory:
                    pt["z_2d"] = [0.0, 0.0]

            # 6. Extract latest raw vitals & timeline
            raw_df = transformer.load_and_standardize_csv(
                csv_path, patient_id=patient_id, lab_mode=lab_mode
            )
            latest_raw = raw_df.iloc[-1]

            timeline = (
                raw_df.set_index("charttime")[
                    ["heart_rate", "map", "resp_rate", "spo2", "temp"]
                ]
                .resample("5min")
                .mean()
                .ffill()
                .tail(72)
                .reset_index()
            )

            vitals_timeline = {
                "timestamps": [str(t) for t in timeline["charttime"]],
                "heart_rate": [float(v) for v in timeline["heart_rate"]],
                "map": [float(v) for v in timeline["map"]],
                "resp_rate": [float(v) for v in timeline["resp_rate"]],
                "spo2": [float(v) for v in timeline["spo2"]],
                "temp": [float(v) for v in timeline["temp"]],
            }

            result_payload = {
                "source": "BioGears",
                "patient_id": patient_id,
                "csv_file": str(csv_path.name),
                "risk_probability": raw_risk,
                "risk_percentage": raw_risk * 100.0,
                "risk_class": "HIGH" if raw_risk >= 0.5 else "LOW",
                "input_shape": list(X.shape),
                "timesteps": int(X.shape[0]),
                "features": int(X.shape[1]),
                "latent_state": z_t.squeeze(0).cpu().numpy().tolist(),
                "latent_dimension": int(z_t.shape[-1]),
                "latent_trajectory": trajectory,
                "latest_vitals": {
                    "heart_rate": float(latest_raw["heart_rate"]),
                    "map": float(latest_raw["map"]),
                    "resp_rate": float(latest_raw["resp_rate"]),
                    "spo2": float(latest_raw["spo2"]),
                    "temp": float(latest_raw["temp"]),
                },
                "vitals_timeline": vitals_timeline,
                "simulation_time_seconds": float(latest_raw["time_seconds"]),
                "charttime": str(latest_raw["charttime"]),
            }

            with _REGISTRY_LOCK:
                if job_id in _JOB_REGISTRY:
                    _JOB_REGISTRY[job_id].status = "completed"
                    _JOB_REGISTRY[job_id].completed_at = datetime.utcnow().isoformat()
                    _JOB_REGISTRY[job_id].result = result_payload

            logger.info(f"BioGears job {job_id} successfully completed.")
            return result_payload

        except Exception as e:
            logger.error(f"BioGears job {job_id} failed: {e}", exc_info=True)
            with _REGISTRY_LOCK:
                if job_id in _JOB_REGISTRY:
                    _JOB_REGISTRY[job_id].status = "failed"
                    _JOB_REGISTRY[job_id].completed_at = datetime.utcnow().isoformat()
                    _JOB_REGISTRY[job_id].error_message = str(e)
            raise

    async def submit_simulation_job(
        self,
        csv_path: str | Path,
        patient_id: str = "biogears_patient_001",
        scenario_type: str = "baseline",
        lab_mode: str = "calibrated_prior",
    ) -> str:
        """Submit a simulation task asynchronously to the worker pool."""
        job_id = self.create_job(patient_id=patient_id, scenario_type=scenario_type)
        path_obj = Path(csv_path)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            _EXECUTOR,
            self._execute_sync_pipeline,
            job_id,
            path_obj,
            patient_id,
            lab_mode,
        )
        return job_id


# Global singleton manager instance
job_manager = BioGearsJobManager()
