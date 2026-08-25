# ClinTwin Sentinel — Architecture at the Pre-BioGears Boundary

Current as of this session. This document is the single entry point
for "what actually exists right now, and where does BioGears plug
in." `WHATIF_ARCHITECTURE.md` and `docs/BIOGEARS_INTEGRATION_BOUNDARY.md`
go deeper on the What-If / BioGears boundary specifically; this
document is the one to read first for everything else.

## 1. What this system does

Given an ICU patient's recent vitals/labs/treatment history
(synthetic, MIMIC-IV-shaped data), a trained sequence model (BiLSTM,
also has a Transformer variant) predicts the probability of clinical
deterioration in the next 2–6 hours, defined as a future-window SOFA
score increase ≥2 (Sepsis-3-consistent — see `DATA_CONTRACT.md` and
`labels_sofa.py`). The system explains that prediction with real
gradient-based feature attribution, visualizes the model's internal
patient-state trajectory, and — as of this phase — lets a user propose
a hypothetical intervention and see how far the system can honestly
take that question today (real baseline, explicit "physiological
simulation not yet available" boundary).

## 2. What is REAL right now (implemented and verified)

- **Data**: `DATA_MODE=synthetic` (see `config.py`). A real synthetic,
  MIMIC-IV-shaped dataset — 300 patients, 45 in the canonical test-split
  roster this app serves — generated entirely outside any
  PhysioNet-credentialed source. No real MIMIC-IV/BigQuery data has
  been used anywhere in this project. Switching to real data requires
  the explicit "APPROVED — SWITCH TO REAL MIMIC" phrase from the
  project owner and has not happened.
- **Pipeline ordering fix (this phase)**: found and fixed a real bug
  where `preprocessing.preprocess()` used to Z-score raw vitals/labs
  BEFORE `labels_sofa.py` computed SOFA subscores and before
  `feature_engineering.py` computed `shock_index` — both need real
  clinical units (SOFA's thresholds are absolute, e.g. "MAP < 70
  mmHg"; a z-scored MAP compared against 70 is meaningless). This
  silently produced a corrupted, near-saturated SOFA distribution
  (verified: totals clustered at 11-23 out of a possible 24). Fixed by
  splitting `preprocess()` into a RAW step and a separate
  `scale_splits()` step that now runs LAST, after engineering and
  labeling. Verified fix: SOFA totals are now a realistic right-skewed
  distribution (mean ~3.9, median 3, max 15). Both models were
  retrained on the corrected pipeline.
- **Model**: a trained BiLSTM checkpoint (`models/lstm_model.pt`) with
  a verified exact 112/112 feature-contract match against the current
  feature-engineering pipeline, enforced at `InferenceService` startup
  (see `backend/inference_service.py::_build_pipeline_and_roster`). A Transformer
  variant also exists (`transformer_model.py`) but is not the model the
  live API currently serves. **Retrained this phase** after fixing a
  real preprocessing-order bug (see "Pipeline ordering fix" below) —
  the checkpoint in the repo reflects the corrected pipeline; any
  older checkpoint trained before this fix is invalid against it and
  must be retrained (`python -m backend.train --model lstm`).
- **Inference**: `backend/inference_service.py` — one process-wide
  `InferenceService` runs real forward passes across the whole roster
  at startup and caches per-window results; every endpoint reads from
  that same real cache. No endpoint runs a separate, potentially
  inconsistent computation.
- **Explainability**: real gradient×input saliency
  (`explainability.gradient_x_input_attribution`), computed from the
  actual model gradients on the actual input. Never called "SHAP" —
  every response carries an explicit `attribution_method`/
  `explanation_method` string naming the real method used, and no
  endpoint's field is named `shap_values` unless it genuinely is one.
  A genuine `shap.KernelExplainer` path exists (`compute_shap_values`) but is
  dead code on every live endpoint.
- **Digital Twin**: `backend/digital_twin.py` — real `z_t` extracted
  from the model's hidden state on a real forward pass, projected to
  2D via a PCA fit once on a representative real batch
  (`warm_pca()`, fixing an earlier `[0,0]` cold-start bug). As of this
  phase, twin state is explicitly split into OBSERVED / PREDICTED /
  SIMULATED (see section 5) rather than being one undifferentiated
  blob. The dead fabricated-trajectory generator
  (`generate_demo_trajectories()`) has been removed.
- **Frontend**: a React dashboard (`frontend/`) fetches every value it
  shows from the real backend (`usePatientData.js`) — no seeded
  pseudo-random generator, no hardcoded patient registry. See section
  7 for the panel-to-endpoint mapping.
- **What-If / ML counterfactual**: `backend/whatif.py` +
  `InferenceService.compute_counterfactual()` — real request
  validation against real patient data and a real, grounded
  intervenable-feature vocabulary; a real baseline; and now a REAL ML
  counterfactual: the modified raw input is re-run through the real
  feature-engineering pipeline and the same trained model checkpoint
  (never an edited number, an interpolation, or a fabricated
  trajectory — verified in `tests/test_whatif.py` by independently
  re-deriving the expected result and asserting an exact match). The
  physiological-simulation portion remains an explicit, honest
  "simulation requires BioGears integration" — that part isn't built.
  See section 6 and `WHATIF_ARCHITECTURE.md`.
- **Tests**: 102 passing (`pytest tests/`), including a full
  cross-endpoint consistency test
  (`test_end_to_end_pre_biogears.py`) that asserts the SAME real risk
  score appears at every step of patient selection → prediction →
  explanation → trajectory → twin state → What-If baseline → ML
  counterfactual, not just that each endpoint independently returns
  200, plus a dedicated test
  (`test_whatif_counterfactual_matches_independent_manual_recomputation`)
  that independently re-derives the expected counterfactual number
  from first principles and asserts an exact match — proof the
  counterfactual is a real computation, not a placeholder.

## 3. What remains MOCKED or NOT YET BUILT

- **BioGears physiological simulation**: not installed, not run,
  not fabricated anywhere. See section 6.
- **Transformer attention visualization**: `AttentionHeatmap` honestly
  shows "not applicable for this model" for the BiLSTM. If the live
  model is ever switched to the Transformer, a real `/attention`
  endpoint would need to be added — not built yet, out of scope for
  this phase.
- **`GET /demo`**: intentionally left in place, still returns the
  original fully-fabricated payload, clearly stamped
  `"deprecated": true` in its response, for backward-compatibility
  only. The production frontend never calls it.
- **Cosmetic**: the synthetic dataset's timestamps are shifted into a
  far-future range (MIMIC-style de-identification convention) — real
  data, not a bug, but may look odd in a live demo.

## 4. API contract (current)

All real endpoints are under `/api/*`, backed by one
`InferenceService` instance:

| Method | Path | Real, defined response |
|---|---|---|
| GET | `/api/roster` | Canonical 45-patient synthetic test-split roster |
| GET | `/api/patients/{stay_id}` | Latest observed vitals + latest prediction |
| GET | `/api/patients/{stay_id}/predict` | Full real prediction for the latest (or a specific) window |
| GET | `/api/patients/{stay_id}/explanation` | Real gradient attribution + generated explanation text |
| GET | `/api/patients/{stay_id}/trajectory` | Real per-window history (vitals, risk, latent state) |
| GET | `/api/patients/{stay_id}/twin-state` | OBSERVED / PREDICTED / SIMULATED digital-twin state |
| GET | `/api/whatif/interventions` | Real intervenable-feature vocabulary |
| POST | `/api/patients/{stay_id}/whatif` | Validated scenario → real baseline + BioGears boundary |

Legacy, pre-Phase-B endpoints (`POST /predict`, `GET /patients`,
`GET /patient/{id}`, `GET /demo`) remain for backward compatibility —
see the module docstring in `backend/api.py` for exactly what changed
and why each was kept or deprecated.

Every response is validated against an explicit Pydantic
`response_model` (schema-drift prevention), and `inference_service.py`
raises `ModelOutputCorrectnessError` rather than silently serving a
NaN/Inf value from a broken forward pass.

## 5. Digital Twin: OBSERVED / PREDICTED / COUNTERFACTUAL / SIMULATED

This phase's redesign (see `backend/digital_twin.py`'s module
docstring for the full rationale). Every twin-state response now
separates:

- **OBSERVED** — real measured vitals for this window. Ground truth,
  not produced by the twin.
- **PREDICTED** — this trained model's real output on that OBSERVED
  data: risk score, latent state `z_t`, and its 2D PCA projection. A
  learned representation, never described as a physiological
  simulation.
- **COUNTERFACTUAL** — what the SAME trained model predicts if inputs
  are hypothetically changed and the real pipeline is re-run. REAL as
  of this phase (`InferenceService.compute_counterfactual()`), but
  only appears in a What-If response (`POST /api/patients/{id}/whatif`)
  for a scenario the user actually asked about — never in the
  persistent `GET /twin-state` response, since it doesn't describe
  this window's actual state.
- **SIMULATED** — forward physiological simulation under a
  hypothetical intervention. `available: false` everywhere in this
  codebase right now. This is BioGears' eventual slot. Never to be
  confused with COUNTERFACTUAL above — one is physics, the other is
  the same statistical model on different input.

The frontend (`LatentStateViz.jsx`) renders this as three labeled
badges so the distinction is visible, not just documented.

## 6. The BioGears / What-If boundary

Full detail in `WHATIF_ARCHITECTURE.md`. Summary: `backend/whatif.py`
validates a scenario (patient, window, one or more interventions drawn
from a real 4-variable vocabulary grounded in `config.py`'s existing
treatment/respiratory-support/ventilation feature columns) against
real patient data, attaches it to a real baseline prediction, and
returns:

```json
"simulation": {
  "available": false,
  "message": "Simulation requires BioGears integration.",
  ...
}
```

**Nothing about BioGears' actual scenario format, variable names,
units, or output schema is assumed anywhere in this codebase.**
`WHATIF_ARCHITECTURE.md` lists exactly what information is needed from
the teammate building the BioGears bridge before real integration can
start, and exactly which four files/functions will change once that
information exists (`backend/whatif.py::run_scenario`,
`SUPPORTED_ENGINES`, the relevant Pydantic models in `backend/api.py`,
and `WhatIfPanel.jsx`'s rendering of a real `simulation` payload).

## 7. Frontend architecture

`frontend/src/hooks/usePatientData.js` is the single data-fetching
layer — every component receives real, already-fetched data as props
and fabricates nothing itself. Panel-to-endpoint mapping:

| Component | Backend source |
|---|---|
| `RiskPanel` | `/api/patients/{id}` (current prediction) |
| `VitalsTimeline` | `/api/patients/{id}/trajectory` (observed_vitals per window) |
| `AttributionPanel` | `/api/patients/{id}/explanation` (local_gradient_attribution) |
| `LatentStateViz` | `/api/patients/{id}/trajectory` (z_2d) + `/twin-state` (OBSERVED/PREDICTED/SIMULATED badges + disclaimer) |
| `AttentionHeatmap` | N/A for BiLSTM — honest empty state |
| `ClinicalExplanation` | `/api/patients/{id}/explanation` (explanation_text) |
| `WhatIfPanel` | `/api/whatif/interventions` + `POST /api/patients/{id}/whatif` |

## 8. How to run the whole thing end-to-end right now

```bash
# backend
pip install -r backend/requirements.txt
python -m uvicorn backend.api:app --reload --port 8000

# frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`. Every panel shows real data. The
What-If panel will always end in "Simulation requires BioGears
integration" until the BioGears bridge is wired in per section 6.

## 9. What's still required from the BioGears implementation

See `WHATIF_ARCHITECTURE.md` section 3 for the full list. Short
version: BioGears' real input format/variables/units, its real output
schema (single end-state vs. time series, variable names, units), and
whether a run completes fast enough to answer synchronously or needs a
job/polling model. None of these are guessed at in this codebase.
