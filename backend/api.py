"""
api.py — FastAPI Backend for ICU Digital Twin
===============================================
Phase B (real synthetic-data endpoints — see PHASE_B_REPORT.md):
  GET  /api/roster                          — real canonical patient roster
  GET  /api/patients/{stay_id}               — real current state (latest window)
  GET  /api/patients/{stay_id}/predict        — real model prediction
  GET  /api/patients/{stay_id}/explanation    — real gradient-attribution explanation
  GET  /api/patients/{stay_id}/trajectory     — real per-window trajectory
  GET  /api/patients/{stay_id}/twin-state     — real digital-twin state,
                                                 split into OBSERVED /
                                                 PREDICTED / SIMULATED
                                                 (see digital_twin.py)

  All of the above are backed by backend/inference_service.py — real
  model inference on real synthetic test-split ICU stays. No endpoint
  under /api/* fabricates a risk score, SHAP/gradient value, z_t, or
  trajectory point.

Pre-BioGears What-If / true ML counterfactual architecture (see
backend/whatif.py and docs/BIOGEARS_INTEGRATION_BOUNDARY.md):
  GET  /api/whatif/interventions              — real intervenable-feature
                                                 vocabulary
  POST /api/patients/{stay_id}/whatif          — validates a scenario
                                                 against real patient data,
                                                 returns: (1) the real
                                                 baseline prediction, (2) a
                                                 REAL ML counterfactual —
                                                 the same trained model
                                                 re-run on the modified
                                                 input through the real
                                                 feature-engineering
                                                 pipeline (see
                                                 InferenceService.
                                                 compute_counterfactual) —
                                                 and (3) an honest
                                                 "physiological simulation
                                                 not yet available"
                                                 boundary. The ML
                                                 counterfactual (2) is NOT
                                                 the same thing as (3) — see
                                                 backend/whatif.py's module
                                                 docstring. Never fabricates
                                                 a simulated outcome.

Legacy endpoints (pre-Phase-B):
  POST /predict       — real inference on client-supplied raw feature
                         vectors (unchanged — still the only endpoint
                         that accepts arbitrary patient_data directly)
  GET  /patients       — REPLACED in Phase B: now returns the real
                         roster (same data as /api/roster), not the old
                         fabricated 10-patient list
  GET  /patient/{id}   — REPLACED in Phase B: {id} is now interpreted as
                         a real stay_id from the roster; returns real
                         current state + explanation + trajectory
  GET  /demo           — DEPRECATED, NOT REMOVED: still returns the old
                         fully-fabricated dashboard payload (random
                         vitals, invented feature-importance numbers,
                         invented patients) for backward compatibility
                         only. Response is stamped `"deprecated": true`.
                         The production frontend must not call this
                         endpoint — see PHASE_B_REPORT.md section on
                         fake-endpoint cleanup.

Field-naming note (Phase D — see PHASE_D_REPORT.md): every endpoint in
this file, old and new, now uses `gradient_attribution`, never
`shap_values` — the underlying computation everywhere is real
input×gradient saliency (explainability.gradient_x_input_attribution),
not a Shapley value. Real SHAP (shap.KernelExplainer) exists in
explainability.compute_shap_values() but is not invoked by any live
endpoint in this file; see that module's docstring for why and how the
distinction is preserved if it's ever wired in.
"""

import os
import sys
import json
import numpy as np
import torch
import logging
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import setup_logging, set_seed

# Initialize logging
logger = setup_logging()

# Lazy imports to avoid slow startup
_model = None
_feature_cols = None
_digital_twin = None
_demo_data = None


def _load_model():
    """Load trained model or create a demo model."""
    global _model, _feature_cols
    
    from backend.lstm_model import ICUBiLSTM
    from backend.transformer_model import ICUTransformer
    
    # Try loading LSTM first, then Transformer
    for model_path, ModelClass, model_type in [
        (config.LSTM_MODEL_PATH, ICUBiLSTM, "lstm"),
        (config.TRANSFORMER_MODEL_PATH, ICUTransformer, "transformer"),
    ]:
        if os.path.exists(model_path):
            logger.info(f"Loading trained {model_type} model from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            input_size = checkpoint["input_size"]
            _feature_cols = checkpoint["feature_cols"]
            state_dict = checkpoint["model_state_dict"]

            if model_type == "lstm":
                hidden_size = state_dict["input_proj.0.weight"].shape[0] if "input_proj.0.weight" in state_dict else config.LSTM_HIDDEN_SIZE
                layer_indices = [
                    int(k.split("_l")[1].split("_")[0])
                    for k in state_dict.keys()
                    if "lstm.weight_ih_l" in k
                ]
                num_layers = max(layer_indices) + 1 if layer_indices else config.LSTM_NUM_LAYERS
                bidirectional = any("reverse" in k for k in state_dict.keys())
                model = ICUBiLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                )
            else:
                d_model = state_dict["embedding.0.weight"].shape[0] if "embedding.0.weight" in state_dict else config.TRANSFORMER_D_MODEL
                layer_indices = [
                    int(k.split("encoder_layers.")[1].split(".")[0])
                    for k in state_dict.keys()
                    if "encoder_layers." in k
                ]
                num_layers = max(layer_indices) + 1 if layer_indices else config.TRANSFORMER_NUM_LAYERS
                dim_feedforward = state_dict["encoder_layers.0.linear1.weight"].shape[0] if "encoder_layers.0.linear1.weight" in state_dict else config.TRANSFORMER_DIM_FEEDFORWARD
                model = ICUTransformer(
                    input_size=input_size,
                    d_model=d_model,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                )

            model.load_state_dict(state_dict)
            model.eval()
            _model = (model, model_type)
            return
    
    # No trained model — create demo model
    logger.warning("No trained model found. Creating demo model.")
    input_size = 50
    _feature_cols = [f"feature_{i}" for i in range(input_size)]
    model = ICUBiLSTM(input_size=input_size)
    model.eval()
    _model = (model, "lstm")


def _get_digital_twin():
    """
    Get or create the digital twin instance used by the LEGACY POST
    /predict endpoint below.

    Phase B fix: this used to call `generate_demo_trajectories()` on
    startup, seeding the twin with 10 entirely fabricated patient
    trajectories BEFORE any real prediction ever happened — meaning a
    real /predict call and nine invented ones lived in the same object,
    indistinguishable to any caller. That fabricated seeding is removed.
    A fresh, real POST /predict call now starts this twin's trajectory
    from nothing (correct, if PCA-cold-start-affected for the first ~10
    calls — see digital_twin.py). The NEW /api/* roster-driven endpoints
    do not use this global twin at all; they use inference_service.py's
    own DigitalTwin instance, which IS pre-warmed with real z_t vectors
    from real inference across the whole roster (see
    InferenceService._warm_pca).
    """
    global _digital_twin

    if _digital_twin is None:
        from backend.digital_twin import DigitalTwin
        _digital_twin = DigitalTwin(latent_dim=128)

    return _digital_twin


def _generate_demo_data():
    """Generate comprehensive demo data for the dashboard."""
    global _demo_data
    
    if _demo_data is not None:
        return _demo_data
    
    np.random.seed(config.SEED)
    set_seed(config.SEED)
    
    twin = _get_digital_twin()
    
    # Generate vital signs timeline
    n_steps = 72
    time_labels = [f"T-{n_steps - i}" for i in range(n_steps)]
    
    # Simulate a deteriorating patient
    vitals = {
        "timestamps": time_labels,
        "heart_rate": [],
        "map": [],
        "spo2": [],
        "resp_rate": [],
        "lactate": [],
    }
    
    for t in range(n_steps):
        severity = max(0, (t - 40) / 32) if t > 40 else 0
        vitals["heart_rate"].append(round(78 + 35 * severity + np.random.normal(0, 3), 1))
        vitals["map"].append(round(82 - 22 * severity + np.random.normal(0, 4), 1))
        vitals["spo2"].append(round(97 - 4 * severity + np.random.normal(0, 1), 1))
        vitals["resp_rate"].append(round(16 + 8 * severity + np.random.normal(0, 2), 1))
        vitals["lactate"].append(
            round(1.0 + 2.5 * severity + np.random.normal(0, 0.3), 2) if np.random.random() > 0.6 else None
        )
    
    # Current risk score (last timestep)
    current_risk = float(np.clip(0.15 + 0.65 * max(0, (71 - 40) / 32), 0, 1))
    
    # Fabricated feature importance (this whole function is the deprecated
    # /demo path — see module docstring). Phase D: never call this "SHAP" —
    # it's an invented list of numbers, not a computed attribution of any
    # kind, real or approximate.
    feature_names = [
        "heart_rate", "MAP", "lactate", "resp_rate", "SpO2",
        "shock_index", "HR_delta", "MAP_delta", "lactate_delta",
        "HR_rolling_std", "temp", "WBC", "time_in_ICU",
        "HR_missing", "lactate_time_since_obs"
    ]
    fabricated_importance_values = [
        0.23, -0.19, 0.31, 0.12, -0.08,
        0.18, 0.15, -0.14, 0.22,
        0.09, 0.03, 0.05, 0.07,
        -0.02, 0.06,
    ]

    feature_importance_data = [
        {"feature": name, "importance": abs(val), "value": val, "direction": "risk" if val > 0 else "protective"}
        for name, val in zip(feature_names, fabricated_importance_values)
    ]
    feature_importance_data.sort(key=lambda x: x["importance"], reverse=True)
    
    # Attention weights (simulated 4 heads × 72 × 72)
    attn = np.random.dirichlet(np.ones(n_steps) * 0.5, size=(4, n_steps))
    # Make recent timesteps get more attention
    for h in range(4):
        for i in range(n_steps):
            attn[h, i, max(0, i-10):i+1] *= 3
        attn[h] = attn[h] / attn[h].sum(axis=1, keepdims=True)
    
    # Subsample attention for visualization (every 4th timestep)
    step = max(1, n_steps // 18)
    attn_sub = attn[0, ::step, ::step]
    attn_labels = [time_labels[i] for i in range(0, n_steps, step)]
    
    # Latent state trajectory (2D projection)
    trajectory = twin.get_trajectory("patient_001")
    if not trajectory:
        trajectory = []
        for t in range(50):
            severity = max(0, (t - 25) / 25)
            trajectory.append({
                "t_idx": t,
                "z_2d": [
                    float(-2 + 3 * severity + np.random.normal(0, 0.2)),
                    float(1 - 2 * severity + np.random.normal(0, 0.2)),
                ],
                "risk_score": float(np.clip(0.1 + 0.7 * severity, 0, 1)),
            })
    
    # Clinical explanation
    explanation = (
        "Risk increased to {:.2f} (Critical) due to rising lactate (tissue hypoxia), "
        "declining MAP (hypotension), and sustained tachycardia. "
        "Elevated shock index indicates hemodynamic instability."
    ).format(current_risk)
    
    # Patient list
    patients = []
    statuses = ["Stable", "Warning", "Critical", "Stable", "Warning",
                "Stable", "Critical", "Stable", "Warning", "Stable"]
    for i in range(10):
        risk = np.random.uniform(0.05, 0.95)
        patients.append({
            "id": f"patient_{i+1:03d}",
            "name": f"Patient {i+1:03d}",
            "stay_id": f"S{1000 + i}",
            "risk_score": round(float(risk), 3),
            "status": statuses[i],
            "hours_in_icu": round(float(np.random.uniform(4, 120)), 1),
        })
    
    _demo_data = {
        "current_patient": {
            "id": "patient_001",
            "risk_score": round(current_risk, 3),
            "status": "Critical" if current_risk > 0.7 else ("Warning" if current_risk > 0.4 else "Stable"),
            "risk_trend": "increasing",
        },
        "vitals": vitals,
        "feature_importance": feature_importance_data,
        "attention": {
            "matrix": attn_sub.tolist(),
            "labels": attn_labels,
        },
        "latent_trajectory": trajectory,
        "explanation": explanation,
        "patients": patients,
    }
    
    return _demo_data


# ═══════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize models and demo data on startup.

    Phase B: also builds the real InferenceService (real model + real
    roster + real per-window predictions across the whole test split)
    eagerly at startup, not lazily on first request. This is deliberate:
    if the checkpoint's feature contract doesn't match the current
    pipeline (FeatureContractError) or no trained checkpoint exists
    (NoTrainedModelError), the API should fail loudly at startup and
    refuse to come up — not serve broken/fabricated responses on the
    first real request.
    """
    logger.info("Starting ICU Digital Twin API...")
    _load_model()
    _get_digital_twin()
    _generate_demo_data()

    from backend.inference_service import get_inference_service
    svc = get_inference_service(model_type="lstm")
    logger.info(
        f"Real InferenceService ready: {len(svc.roster)} roster patients, "
        f"checkpoint={svc.checkpoint_meta.get('checkpoint_path')}"
    )

    logger.info("API ready!")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="ICU Digital Twin API",
    description="Real-time ICU patient risk prediction with explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    patient_data: List[Dict]  # List of timestep dicts with feature values
    patient_id: Optional[str] = None
    model_type: Optional[str] = "lstm"


class PredictResponse(BaseModel):
    """
    Response from /predict endpoint.

    Phase D: `gradient_attribution` (was `shap_values`) — this endpoint
    calls explainability.gradient_x_input_attribution() directly, never
    real SHAP, so it is never honestly labeled "shap_values". This is a
    breaking change to this field's name; nothing in this repository
    called it under the old name (grepped before renaming — the
    frontend fabricates its own data independently and never called
    this endpoint at all), so there was no live consumer to migrate.
    """
    risk_score: float
    status: str
    z_t: List[float]
    gradient_attribution: List[Dict]
    attention_weights: Optional[List] = None
    explanation: str
    latent_trajectory: Optional[List[Dict]] = None


# ═══════════════════════════════════════════════════════════════════
# Phase C — explicit response models for the real /api/* endpoints.
#
# WHY: without these, every /api/* endpoint above just returned a raw
# dict — FastAPI would happily serialize whatever inference_service.py
# handed back, with no check that the shape matches what callers (soon:
# the frontend) actually expect. Declaring `response_model=...` makes
# FastAPI VALIDATE the real return value against this contract on every
# request — if inference_service.py's output shape ever silently drifts
# (a renamed field, a dict where a list was expected, a missing
# required field), the endpoint now fails loudly with a clear
# validation error instead of quietly serving a differently-shaped
# payload the frontend would fail to parse in some confusing way.
# This is schema-drift prevention, not a new source of data — every
# field here already existed in the dicts inference_service.py returns
# (see backend/inference_service.py), just now with an explicit,
# checked contract.
# ═══════════════════════════════════════════════════════════════════


class RosterPatientOut(BaseModel):
    stay_id: int
    subject_id: int
    hadm_id: int
    age_years: Optional[float] = None
    sex_male: Optional[bool] = None
    n_observed_timesteps: int
    first_charttime: str
    last_charttime: str
    split: str
    latest_risk_score: Optional[float] = None
    latest_status: Optional[str] = None
    n_windows: int


class RosterResponse(BaseModel):
    patients: List[RosterPatientOut]
    split: str
    model_type: str


class CurrentStatePrediction(BaseModel):
    risk_score: float
    status: str
    model_type: str
    model_epoch: Optional[int] = None
    model_checkpoint: str


class CurrentStateResponse(BaseModel):
    patient: RosterPatientOut
    observed_vitals: Dict[str, Optional[float]]
    window_end_time: str
    prediction: CurrentStatePrediction


class GradientAttributionItem(BaseModel):
    feature: str
    importance: float
    clinical_name: str


class LocalGradientAttributionItem(BaseModel):
    feature: str
    attribution_value: float
    direction: str
    clinical_name: str


class PredictionOut(BaseModel):
    stay_id: int
    subject_id: int
    hadm_id: int
    window_idx: int
    window_start_time: str
    window_end_time: str
    risk_score: float
    status: str
    z_t: List[float]
    z_2d: List[float]
    gradient_attribution: List[GradientAttributionItem]
    local_gradient_attribution: List[LocalGradientAttributionItem]
    attention_weights: Optional[List] = None
    explanation: str
    explanation_method: str
    ground_truth_label: Optional[float] = None
    observed_vitals: Dict[str, Optional[float]]
    model_type: str
    model_epoch: Optional[int] = None
    model_checkpoint: str


class ExplanationResponse(BaseModel):
    stay_id: int
    window_idx: int
    risk_score: float
    status: str
    gradient_attribution: List[GradientAttributionItem]
    local_gradient_attribution: List[LocalGradientAttributionItem]
    attention_weights: Optional[List] = None
    explanation_text: str
    explanation_method: str


class TrajectoryPoint(BaseModel):
    window_idx: int
    window_start_time: str
    window_end_time: str
    risk_score: float
    status: str
    z_2d: List[float]
    ground_truth_label: Optional[float] = None
    observed_vitals: Dict[str, Optional[float]]


class TrajectoryResponse(BaseModel):
    stay_id: int
    trajectory: List[TrajectoryPoint]


class ObservedState(BaseModel):
    """Real measured vitals for this window — ground truth data, never model output."""
    vitals: Dict[str, Optional[float]]
    description: str


class PredictedLatentState(BaseModel):
    z_t: List[float]
    z_2d: List[float]
    latent_dim: int


class PredictedState(BaseModel):
    """The trained model's real output from one forward pass on OBSERVED data above."""
    risk_score: float
    status: str
    latent_state: PredictedLatentState
    description: str


class SimulatedState(BaseModel):
    """Forward physiological simulation under a hypothetical intervention — not yet available."""
    available: bool
    description: str


class TwinStateResponse(BaseModel):
    stay_id: int
    window_idx: int
    window_end_time: str
    observed: ObservedState
    predicted: PredictedState
    simulated: SimulatedState
    disclaimer: str


# ─── What-If / counterfactual architecture (see backend/whatif.py and
# WHATIF_ARCHITECTURE.md) — pre-BioGears: every scenario request is
# really validated against real patient data and a real intervenable-
# feature vocabulary, and gets a real baseline, but the actual forward
# physiological simulation stops at an explicit, honest boundary until
# BioGears is connected. Nothing below fabricates a simulated result.

class InterventionIn(BaseModel):
    feature: str
    value: float


class WhatIfScenarioIn(BaseModel):
    window_idx: Optional[int] = None
    interventions: List[InterventionIn]
    engine: str = "biogears"


class IntervenableFeatureOut(BaseModel):
    feature: str
    display_name: str
    unit: str
    min: float
    max: float
    clinical_note: str
    counterfactual_note: str


class InterventionsListResponse(BaseModel):
    interventions: List[IntervenableFeatureOut]


class WhatIfBaseline(BaseModel):
    risk_score: float
    status: str
    observed_vitals: Dict[str, Optional[float]]
    description: str


class WhatIfCounterfactual(BaseModel):
    """The REAL ML counterfactual — the same trained model re-run on a
    hypothetically modified input. NOT a physiological simulation."""
    risk_score: float
    status: str
    risk_change: float
    gradient_attribution: List[GradientAttributionItem]
    explanation_method: str
    description: str


class WhatIfSimulation(BaseModel):
    available: bool
    boundary: str
    message: str
    description: str


class WhatIfScenarioResult(BaseModel):
    stay_id: int
    window_idx: int
    engine: str
    requested_interventions: List[InterventionIn]
    baseline: WhatIfBaseline
    counterfactual: WhatIfCounterfactual
    simulation: WhatIfSimulation


class LegacyRosterResponse(BaseModel):
    patients: List[RosterPatientOut]


class LegacyPatientDetailResponse(BaseModel):
    patient: RosterPatientOut
    observed_vitals: Dict[str, Optional[float]]
    prediction: CurrentStatePrediction
    gradient_attribution: List[GradientAttributionItem]
    explanation: str
    explanation_method: str
    trajectory: List[TrajectoryPoint]


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "ICU Digital Twin API"}


@app.get("/demo")
async def get_demo_data():
    """
    DEPRECATED (Phase B) — fully fabricated dashboard payload.

    Kept only for backward compatibility with anything still calling it
    directly; not used by any endpoint the production/demo frontend
    should call from Phase K onward. Use GET /api/roster and
    GET /api/patients/{stay_id}/* instead — those are backed by real
    model inference on real synthetic ICU stays.
    """
    data = _generate_demo_data()
    return {**data, "deprecated": True, "replacement": "/api/roster, /api/patients/{stay_id}/*"}


@app.get("/patients", response_model=LegacyRosterResponse)
async def list_patients():
    """
    List real patients from the canonical roster (Phase B).

    REPLACED (not just deprecated) — this used to return an independently
    fabricated 10-patient list regenerated randomly on every process
    start. It now returns backend/patient_roster.py's real roster: real
    stay_id/subject_id/hadm_id from the synthetic test-split ICU stays,
    with latest_risk_score/latest_status computed by real model
    inference (inference_service.py). Identical to GET /api/roster;
    kept at this path too since it's the more discoverable legacy name.
    """
    from backend.inference_service import get_inference_service
    svc = get_inference_service()
    return {"patients": svc.get_roster()}


@app.get("/patient/{patient_id}", response_model=LegacyPatientDetailResponse)
async def get_patient(patient_id: int):
    """
    Get real current state + explanation + trajectory for one patient.

    REPLACED (Phase B) — `patient_id` is now a real `stay_id` from the
    canonical roster (GET /patients or GET /api/roster), not an
    invented string like "patient_003". All fields below come from a
    real forward pass of the trained model on this stay's real
    synthetic data; see backend/inference_service.py.
    """
    from backend.inference_service import get_inference_service, UnknownStayIdError

    svc = get_inference_service()
    try:
        current = svc.get_current_state(patient_id)
        explanation = svc.get_explanation(patient_id)
        trajectory = svc.get_trajectory(patient_id)
    except UnknownStayIdError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "patient": current["patient"],
        "observed_vitals": current["observed_vitals"],
        "prediction": current["prediction"],
        "gradient_attribution": explanation["gradient_attribution"],
        "explanation": explanation["explanation_text"],
        "explanation_method": explanation["explanation_method"],
        "trajectory": trajectory,
    }


# ═══════════════════════════════════════════════════════════════════
# Phase B — real roster-driven endpoints (backend/inference_service.py)
# ═══════════════════════════════════════════════════════════════════


@app.get("/api/roster", response_model=RosterResponse)
async def get_roster():
    """The single canonical patient roster — real stay_ids only."""
    from backend.inference_service import get_inference_service
    svc = get_inference_service()
    return {"patients": svc.get_roster(), "split": svc.split, "model_type": svc.model_type}


@app.get("/api/patients/{stay_id}", response_model=CurrentStateResponse)
async def get_patient_current_state(stay_id: int):
    """Real latest observed vitals + real latest model prediction for this stay."""
    from backend.inference_service import get_inference_service, UnknownStayIdError
    svc = get_inference_service()
    try:
        return svc.get_current_state(stay_id)
    except UnknownStayIdError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/patients/{stay_id}/predict", response_model=PredictionOut)
async def get_patient_prediction(stay_id: int, window_idx: Optional[int] = None):
    """
    Real model prediction for one window of this stay.

    window_idx: optional; defaults to the latest available window
    (i.e. the patient's current state). Pass an explicit window_idx to
    inspect any earlier point in the stay (0-indexed, chronological).
    """
    from backend.inference_service import get_inference_service, UnknownStayIdError
    svc = get_inference_service()
    try:
        return svc.get_prediction(stay_id, window_idx)
    except UnknownStayIdError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/patients/{stay_id}/explanation", response_model=ExplanationResponse)
async def get_patient_explanation(stay_id: int, window_idx: Optional[int] = None):
    """Real gradient-attribution explanation (global + local + NL text) for one window."""
    from backend.inference_service import get_inference_service, UnknownStayIdError
    svc = get_inference_service()
    try:
        return svc.get_explanation(stay_id, window_idx)
    except UnknownStayIdError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/patients/{stay_id}/trajectory", response_model=TrajectoryResponse)
async def get_patient_trajectory(stay_id: int):
    """Real per-window trajectory (observed vitals + predicted risk + z_2d) for this stay."""
    from backend.inference_service import get_inference_service, UnknownStayIdError
    svc = get_inference_service()
    try:
        return {"stay_id": stay_id, "trajectory": svc.get_trajectory(stay_id)}
    except UnknownStayIdError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/patients/{stay_id}/twin-state", response_model=TwinStateResponse)
async def get_patient_twin_state(stay_id: int, window_idx: Optional[int] = None):
    """Real latent digital-twin state (z_t / z_2d) for one window of this stay."""
    from backend.inference_service import get_inference_service, UnknownStayIdError
    svc = get_inference_service()
    try:
        return svc.get_twin_state(stay_id, window_idx)
    except UnknownStayIdError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/whatif/interventions", response_model=InterventionsListResponse)
async def list_whatif_interventions():
    """
    Real, single source of truth for what a What-If scenario can ask
    about — see backend/whatif.py's INTERVENABLE_FEATURES. The
    frontend fetches this rather than hardcoding a second copy.
    """
    from backend.whatif import list_intervenable_features
    return {"interventions": list_intervenable_features()}


@app.post("/api/patients/{stay_id}/whatif", response_model=WhatIfScenarioResult)
async def run_whatif_scenario(stay_id: int, scenario: WhatIfScenarioIn):
    """
    Pre-BioGears What-If / true ML counterfactual endpoint (see
    backend/whatif.py and docs/BIOGEARS_INTEGRATION_BOUNDARY.md).
    Validates the request against real patient data and the real
    intervenable-feature vocabulary, computes a REAL ML counterfactual
    (the same trained model re-run on the modified input through the
    real feature-engineering pipeline — see
    InferenceService.compute_counterfactual), and returns a result that
    stops at the honest BioGears integration boundary for the
    physiological-simulation portion — never a fabricated simulated
    outcome.
    """
    from backend.inference_service import get_inference_service, UnknownStayIdError
    from backend.whatif import validate_scenario_request, run_scenario, InterventionValidationError

    svc = get_inference_service()
    known_stay_ids = {p["stay_id"] for p in svc.get_roster()}

    try:
        request = validate_scenario_request(
            stay_id=stay_id,
            window_idx=scenario.window_idx,
            interventions=[i.model_dump() for i in scenario.interventions],
            engine=scenario.engine,
            known_stay_ids=known_stay_ids,
        )
    except InterventionValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        baseline_prediction = svc.get_prediction(stay_id, scenario.window_idx)
    except UnknownStayIdError as e:
        raise HTTPException(status_code=404, detail=str(e))

    from backend.inference_service import ModelOutputCorrectnessError
    try:
        return run_scenario(baseline_prediction, request, svc)
    except ModelOutputCorrectnessError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Predict risk from patient time-series data.
    
    Input: List of timestep dictionaries with feature values
    Output: Risk score, latent state, real gradient×input attribution
    (never SHAP — see explainability.py's module docstring), and explanation.
    """
    global _model, _feature_cols
    
    if _model is None:
        _load_model()
    
    model, model_type = _model
    device = next(model.parameters()).device
    
    try:
        # Convert input to tensor
        n_features = len(_feature_cols)
        n_steps = len(request.patient_data)
        
        X = np.zeros((1, max(n_steps, config.INPUT_WINDOW), n_features), dtype=np.float32)
        
        for t, step_data in enumerate(request.patient_data):
            for j, col in enumerate(_feature_cols):
                if col in step_data:
                    val = step_data[col]
                    if val is not None and not np.isnan(float(val)):
                        X[0, t, j] = float(val)
        
        X_tensor = torch.tensor(X).to(device)
        mask = torch.ones(1, X.shape[1]).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            if model_type == "transformer":
                risk, z_t, attn = model(X_tensor, mask)
                attn_list = [a.cpu().numpy().tolist() for a in attn]
            else:
                risk, z_t = model(X_tensor, mask)
                attn_list = None
        
        risk_score = float(risk.squeeze().cpu().item())
        z_t_list = z_t.squeeze().cpu().numpy().tolist()
        
        # Status
        if risk_score >= 0.7:
            status = "Critical"
        elif risk_score >= 0.4:
            status = "Warning"
        else:
            status = "Stable"
        
        # Update digital twin
        twin = _get_digital_twin()
        patient_id = request.patient_id or "realtime_patient"
        twin_result = twin.extract_trajectory(
            model, X_tensor, mask, patient_id, model_type
        )
        
        # Generate explanation — real gradient×input attribution, never SHAP
        # (see explainability.py's Phase D docstring; this endpoint has never
        # called real SHAP, only ever this approximation).
        from backend.explainability import generate_clinical_explanation, gradient_x_input_attribution

        attribution_vals = gradient_x_input_attribution(model, X_tensor, mask, model_type, device)

        from backend.explainability import compute_global_importance
        global_imp = compute_global_importance(attribution_vals, _feature_cols, top_k=10)

        explanation = generate_clinical_explanation(
            attribution_vals, _feature_cols, risk_score, sample_idx=0
        )

        return PredictResponse(
            risk_score=risk_score,
            status=status,
            z_t=z_t_list,
            gradient_attribution=global_imp,
            attention_weights=attn_list,
            explanation=explanation,
            latent_trajectory=twin.get_trajectory(patient_id),
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Phase 3 - Real BioGears Physiology & Inference Integration
# ============================================================

@app.post("/biogears/predict")
@app.post("/api/biogears/predict")
async def predict_biogears(request: Optional[dict] = None):
    """
    Run real ICUBiLSTM inference on a standardized 6-hour BioGears
    simulation output, returning raw risk and 256-D latent state.
    """
    try:
        from backend.biogears_adapter import BioGearsFeatureTransformer
        from backend.biogears_inference import (
            prepare_biogears_input,
            run_lstm_inference,
            run_lstm_trajectory,
            load_biogears_csv,
        )

        request = request or {}

        if not os.path.exists(config.LSTM_MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Trained LSTM model not found at models/lstm_model.pt.",
            )

        checkpoint = torch.load(
            config.LSTM_MODEL_PATH,
            map_location="cpu",
            weights_only=False,
        )
        feature_cols = checkpoint["feature_cols"]

        biogears_csv = request.get(
            "csv_path",
            os.path.join(
                config.PROJECT_ROOT,
                "biogears_runs",
                "stable_6hour_results.csvResults.csv",
            ),
        )

        if not os.path.exists(biogears_csv):
            raise HTTPException(
                status_code=404,
                detail=f"BioGears results file not found: {biogears_csv}",
            )

        patient_id = request.get("patient_id", "biogears_patient_001")
        lab_mode = request.get("lab_mode", "calibrated_prior")

        # Prepare the exact 72 x 46 input expected by the trained LSTM.
        X, mask, df = prepare_biogears_input(
            input_path=biogears_csv,
            feature_cols=feature_cols,
            patient_id=patient_id,
            lab_mode=lab_mode,
        )

        # Run the real trained model.
        result = run_lstm_inference(X)

        # Extract the learned latent state at each timestep.
        trajectory_result = run_lstm_trajectory(X)
        latent_trajectory = trajectory_result["latent_trajectory"]

        # Project latent states to 2D PCA space
        all_z = [np.array(pt["latent_state"], dtype=np.float32) for pt in latent_trajectory]
        if len(all_z) >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(np.array(all_z))
            for idx, pt in enumerate(latent_trajectory):
                pt["z_2d"] = [float(pca_coords[idx, 0]), float(pca_coords[idx, 1])]
        else:
            for pt in latent_trajectory:
                pt["z_2d"] = [0.0, 0.0]

        raw_df = load_biogears_csv(biogears_csv, patient_id=patient_id, lab_mode=lab_mode)
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

        return {
            "source": "BioGears",
            "simulation_file": os.path.basename(biogears_csv),
            "patient_id": patient_id,
            "risk_probability": result["risk_probability"],
            "risk_percentage": result["risk_percentage"],
            "risk_class": result["risk_class"],
            "input_shape": result["input_shape"],
            "timesteps": int(X.shape[0]),
            "features": int(X.shape[1]),
            "latent_state": result["latent_state"],
            "latent_dimension": result["latent_dimension"],
            "latent_trajectory": latent_trajectory,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BioGears prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/biogears/intervene")
@app.post("/api/biogears/intervene")
async def intervene_biogears(request: Optional[dict] = None):
    """
    Run a real BioGears intervention scenario and compare
    baseline vs post-intervention LSTM predictions.
    """
    try:
        from backend.biogears_intervention import run_biogears_intervention

        request = request or {}
        intervention = str(request.get("intervention", "saline")).lower()
        patient_id = str(request.get("patient_id", "biogears_intervention_patient"))

        result = run_biogears_intervention(
            intervention=intervention,
            patient_id=patient_id,
        )
        return result

    except Exception as e:
        logger.error(f"BioGears intervention error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/biogears/simulate")
async def submit_simulation(request: Optional[dict] = None):
    """
    Submit an asynchronous simulation job to the background ThreadPoolExecutor.
    Returns immediately with HTTP 200 and the job_id for polling.
    """
    try:
        from backend.biogears_service import job_manager

        request = request or {}
        patient_id = str(request.get("patient_id", "biogears_patient_001"))
        scenario_type = str(request.get("scenario_type", "baseline"))
        lab_mode = str(request.get("lab_mode", "calibrated_prior"))

        csv_path = request.get(
            "csv_path",
            os.path.join(
                config.PROJECT_ROOT,
                "biogears_runs",
                "stable_6hour_results.csvResults.csv",
            ),
        )

        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail=f"Simulation source file not found: {csv_path}",
            )

        job_id = await job_manager.submit_simulation_job(
            csv_path=csv_path,
            patient_id=patient_id,
            scenario_type=scenario_type,
            lab_mode=lab_mode,
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "BioGears simulation job submitted successfully.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async simulation submission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/biogears/jobs/{job_id}")
async def get_simulation_job(job_id: str):
    """Poll status and retrieve result payload for a background BioGears job."""
    from backend.biogears_service import job_manager

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"BioGears job '{job_id}' not found.",
        )
    return job.to_dict()


@app.get("/api/v1/biogears/jobs")
async def list_simulation_jobs():
    """List recent BioGears background execution jobs."""
    from backend.biogears_service import job_manager
    return {"jobs": job_manager.list_jobs()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )

