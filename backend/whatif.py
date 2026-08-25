"""
whatif.py — Pre-BioGears What-If / True ML Counterfactual Architecture
=========================================================================
Defines the SHAPE of a "what if we changed this patient's treatment"
scenario request, a real validation layer, and now a REAL ML
counterfactual computation — all WITHOUT performing any physiological
simulation and WITHOUT assuming anything about BioGears' actual output
schema.

THREE DISTINCT CONCEPTS, kept separate everywhere in this module, the
API, and the frontend (never blurred):

  1. MODEL PREDICTION  — "What does our trained ML model predict for
     this patient's real, observed data?" Real, already implemented
     (backend/inference_service.py's cached baseline prediction).

  2. ML COUNTERFACTUAL  — "What does the SAME trained ML model predict
     if we hypothetically change an input?" Real, implemented in this
     phase (see `run_scenario()` below and
     `InferenceService.compute_counterfactual()`): the modified raw
     input is run back through the real feature-engineering pipeline
     and the real trained model. This is still just the statistical
     model — it has no notion of physiology, drug pharmacokinetics, or
     organ systems beyond what the model learned from the training
     data's correlations.

  3. PHYSIOLOGICAL SIMULATION  — "What would a mechanistic physiology
     engine (BioGears) predict actually happens to this patient's
     body?" NOT implemented, NOT available, NOT guessed at anywhere in
     this codebase. This requires BioGears, built separately by a
     teammate. See docs/BIOGEARS_INTEGRATION_BOUNDARY.md.

    ClinTwin prediction (1, real)
              |
              v
      WhatIfScenarioRequest  (real: validated against real patient data)
              |
              v
      ML Counterfactual (2, real — see compute_ml_counterfactual below)
              |
              v
      ---- [BIOGEARS INTEGRATION BOUNDARY] ----   <- nothing crosses this yet
              |
              v
      Physiological Simulation Result (3, NOT YET AVAILABLE)
              |
              v
      ClinTwin visualization (renders 1 and 2 now; ready for 3 once it exists)

Intervenable feature vocabulary:
  Deliberately limited to the subset of config.ALL_FEATURE_COLS that
  represents something a clinician actually ORDERS or TITRATES
  (config.TREATMENT_COLS + RESP_SUPPORT_COLS + VENT_COLS) — not
  measured vitals/labs (those are OBSERVED, not set) and not
  demographics (not an intervention at all). This keeps the "what can
  I ask a what-if about" list grounded in the same real feature
  definitions the trained model already uses, rather than inventing a
  parallel vocabulary. Every one of these four features participates
  in feature_engineering.py's per-column delta/rolling-stat loop (it
  loops over ALL of config.ALL_FEATURE_COLS uniformly), so a modified
  raw value can always be correctly re-engineered — see each entry's
  `counterfactual_note` below for exactly how.
"""

import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─── Intervenable feature vocabulary (real, grounded in config.py) ──────────

INTERVENABLE_FEATURES: Dict[str, dict] = {
    "vasopressor_dose": {
        "display_name": "Vasopressor dose",
        "unit": "normalized dose (matches training data units)",
        "min": 0.0,
        "max": 5.0,
        "clinical_note": "e.g. norepinephrine-equivalent infusion rate.",
        "counterfactual_note": (
            "Raw column config.TREATMENT_COLS[0]. Setting it re-derives "
            "vasopressor_dose_delta/_rolling_mean/_rolling_std (feature_"
            "engineering.py) from the modified raw series, and changes the "
            "cardiovascular SOFA subscore's input if labels were "
            "recomputed. Fully representable by the current model — "
            "evaluable."
        ),
    },
    "fluid_bolus_ml": {
        "display_name": "Fluid bolus",
        "unit": "mL",
        "min": 0.0,
        "max": 2000.0,
        "clinical_note": "Volume of a single crystalloid/colloid bolus.",
        "counterfactual_note": (
            "Raw column config.TREATMENT_COLS[1]. Setting it re-derives "
            "fluid_bolus_ml_delta/_rolling_mean/_rolling_std. Fully "
            "representable by the current model — evaluable."
        ),
    },
    "fio2": {
        "display_name": "FiO2",
        "unit": "fraction (0.21-1.0)",
        "min": 0.21,
        "max": 1.0,
        "clinical_note": "Fraction of inspired oxygen.",
        "counterfactual_note": (
            "Raw column config.RESP_SUPPORT_COLS[0]. Setting it re-derives "
            "fio2_delta/_rolling_mean/_rolling_std, and (if labels were "
            "recomputed) the SF-ratio respiratory SOFA subscore. Does NOT "
            "change spo2 (a separate, OBSERVED column) — the model sees "
            "'FiO2 changed, SpO2 as originally observed', which is a real "
            "limitation of a 4-variable intervention vocabulary, not a bug; "
            "disclosed here rather than silently assumed away. Fully "
            "evaluable by the current model."
        ),
    },
    "ventilated": {
        "display_name": "Mechanical ventilation",
        "unit": "boolean (0 or 1)",
        "min": 0.0,
        "max": 1.0,
        "clinical_note": "Whether the patient is on invasive mechanical ventilation.",
        "counterfactual_note": (
            "Raw column config.VENT_COLS[0]. Setting it re-derives "
            "ventilated_delta/_rolling_mean/_rolling_std and (if labels "
            "were recomputed) the ventilation-gated tiers of the "
            "respiratory SOFA subscore. Fully evaluable by the current "
            "model."
        ),
    },
}

assert set(INTERVENABLE_FEATURES.keys()) <= set(config.ALL_FEATURE_COLS), (
    "INTERVENABLE_FEATURES must only reference real model feature names "
    "from config.ALL_FEATURE_COLS — never an invented variable."
)

SUPPORTED_ENGINES = {"biogears"}
# The only engine this architecture currently targets. Deliberately a
# single-entry set, not an enum with placeholder future members —
# adding an engine (e.g. a real model-reassessment counterfactual) is
# a legitimate future option (see WHATIF_ARCHITECTURE.md
# "Recommendations") but is NOT implemented here, so it is not listed
# as "supported" until it actually exists.


class InterventionValidationError(Exception):
    """A what-if scenario request failed validation against real patient
    data or the real intervenable-feature vocabulary. Callers turn this
    into an honest 4xx response — never silently coerced/ignored."""


@dataclass
class Intervention:
    feature: str
    value: float


@dataclass
class WhatIfScenarioRequest:
    stay_id: int
    window_idx: Optional[int]
    interventions: List[Intervention]
    engine: str = "biogears"


def validate_scenario_request(
    stay_id: int,
    window_idx: Optional[int],
    interventions: List[dict],
    engine: str,
    known_stay_ids: set,
) -> WhatIfScenarioRequest:
    """
    Real validation against real data — no placeholder/no-op checks.

    Raises InterventionValidationError with a specific, actionable
    message on any failure. Never silently clamps or drops an invalid
    value.

    Note: window_idx range validation is intentionally NOT duplicated
    here — InferenceService.get_prediction() is the single source of
    truth for which window_idx values exist for a stay_id (it already
    raises a clear UnknownStayIdError), and the caller re-uses that
    check when it looks up the baseline prediction. Duplicating a
    second range check here would risk the two drifting out of sync.
    """
    if stay_id not in known_stay_ids:
        raise InterventionValidationError(
            f"stay_id {stay_id} is not in the canonical roster. "
            f"Use GET /api/roster to list valid stay_ids."
        )

    if engine not in SUPPORTED_ENGINES:
        raise InterventionValidationError(
            f"engine {engine!r} is not supported. Supported engines: "
            f"{sorted(SUPPORTED_ENGINES)}. (No real simulation engine is "
            f"connected yet — see WHATIF_ARCHITECTURE.md.)"
        )

    if not interventions:
        raise InterventionValidationError(
            "At least one intervention is required. "
            f"Valid features: {sorted(INTERVENABLE_FEATURES)}."
        )

    parsed: List[Intervention] = []
    seen_features = set()
    for item in interventions:
        feature = item.get("feature") if isinstance(item, dict) else getattr(item, "feature", None)
        value = item.get("value") if isinstance(item, dict) else getattr(item, "value", None)

        if feature not in INTERVENABLE_FEATURES:
            raise InterventionValidationError(
                f"{feature!r} is not an intervenable feature. Valid features: "
                f"{sorted(INTERVENABLE_FEATURES)}. (Measured vitals/labs are "
                f"OBSERVED, not something a scenario can set — see "
                f"digital_twin.py's OBSERVED/PREDICTED/SIMULATED distinction.)"
            )
        if feature in seen_features:
            raise InterventionValidationError(
                f"Feature {feature!r} was specified more than once in the same scenario."
            )
        seen_features.add(feature)

        bounds = INTERVENABLE_FEATURES[feature]
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise InterventionValidationError(
                f"Value for {feature!r} must be numeric, got {value!r}."
            )
        if not (bounds["min"] <= value <= bounds["max"]):
            raise InterventionValidationError(
                f"Value {value} for {feature!r} is outside the valid range "
                f"[{bounds['min']}, {bounds['max']}] {bounds['unit']}."
            )
        parsed.append(Intervention(feature=feature, value=value))

    return WhatIfScenarioRequest(
        stay_id=stay_id, window_idx=window_idx, interventions=parsed, engine=engine
    )


def run_scenario(baseline_prediction: dict, request: WhatIfScenarioRequest, inference_service) -> dict:
    """
    Compute the REAL ML counterfactual for a validated scenario and
    attach it to its real baseline prediction. The physiological
    simulation section stops cleanly at the BioGears integration
    boundary — no BioGears result is ever fabricated.

    `baseline_prediction` is a real prediction dict from
    InferenceService.get_prediction() — the same real model output
    used everywhere else in this app. `inference_service` is the live
    InferenceService singleton; its `compute_counterfactual()` does the
    actual re-engineer-and-re-infer work (see inference_service.py) —
    this function only orchestrates and assembles the response, it
    never computes a risk number itself.
    """
    cf = inference_service.compute_counterfactual(
        stay_id=request.stay_id,
        window_idx=baseline_prediction["window_idx"],
        interventions=[asdict(i) for i in request.interventions],
    )

    baseline_risk = baseline_prediction["risk_score"]
    counterfactual_risk = cf["risk_score"]

    return {
        "stay_id": request.stay_id,
        "window_idx": baseline_prediction["window_idx"],
        "engine": request.engine,
        "requested_interventions": [asdict(i) for i in request.interventions],
        "baseline": {
            "risk_score": baseline_risk,
            "status": baseline_prediction["status"],
            "observed_vitals": baseline_prediction["observed_vitals"],
            "description": (
                "Real current state for this patient/window — the same "
                "OBSERVED vitals and PREDICTED risk score shown elsewhere "
                "in this app, not recomputed here."
            ),
        },
        "counterfactual": {
            "risk_score": counterfactual_risk,
            "status": cf["status"],
            "risk_change": counterfactual_risk - baseline_risk,
            "gradient_attribution": cf["gradient_attribution"],
            "explanation_method": cf["explanation_method"],
            "description": (
                "ML COUNTERFACTUAL — the SAME trained model, re-run on this "
                "patient's real window with the requested intervention(s) "
                "applied to the raw input and the real feature-engineering "
                "pipeline re-executed. This is the model's statistical "
                "response to a changed input, not a physiological "
                "simulation of the patient's body — see backend/whatif.py "
                "module docstring and docs/BIOGEARS_INTEGRATION_BOUNDARY.md."
            ),
        },
        "simulation": {
            "available": False,
            "boundary": "BIOGEARS_INTEGRATION_BOUNDARY",
            "message": "Simulation requires BioGears integration.",
            "description": (
                "Forward PHYSIOLOGICAL simulation of this scenario (what a "
                "mechanistic physiology engine would predict actually "
                "happens to this patient's body) is not yet implemented. "
                "This is a different question from the ML counterfactual "
                "above, which only reflects the trained model's learned "
                "correlations. This project intentionally does not "
                "fabricate a result here — see "
                "docs/BIOGEARS_INTEGRATION_BOUNDARY.md for the integration "
                "boundary and what BioGears needs to provide before this "
                "can return a real simulated outcome."
            ),
        },
    }


def list_intervenable_features() -> List[dict]:
    """Real, single source of truth for what a scenario can ask about —
    consumed by GET /api/whatif/interventions so the frontend never
    hardcodes a second copy of this vocabulary."""
    return [
        {"feature": name, **bounds} for name, bounds in INTERVENABLE_FEATURES.items()
    ]
