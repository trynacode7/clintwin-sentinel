# ClinTwin Sentinel

An ICU deterioration-prediction digital twin: given a patient's recent
vitals/labs/treatment history, a trained sequence model predicts the
probability of clinical deterioration in the next 2-6 hours, explains
that prediction with real gradient-based feature attribution,
visualizes the model's internal patient-state trajectory, and lets a
clinician ask a real "what if we changed this intervention" ML
counterfactual question.

**Data status:** this project runs entirely on synthetic,
MIMIC-IV-shaped data (`DATA_MODE=synthetic`). No real
PhysioNet-credentialed MIMIC-IV data has been used anywhere. A real
MIMIC-IV extraction path exists in the codebase (`backend/mimic_adapter/`,
`extract_real_mimic_to_ml_dataset.py`) but is unexercised against real
data — see [`docs/MIMIC_IV_SETUP.md`](docs/MIMIC_IV_SETUP.md).

**Simulation status:** the What-If panel provides a real **ML
counterfactual** (the same trained model, re-run on a modified input)
but does **not** include a physiological simulation (e.g. BioGears).
That integration is intentionally not yet built — see
[`docs/BIOGEARS_INTEGRATION_BOUNDARY.md`](docs/BIOGEARS_INTEGRATION_BOUNDARY.md).

## Where to start

| Question | Read this |
|---|---|
| What is this, architecturally, right now? | [`PRE_BIOGEARS_ARCHITECTURE.md`](PRE_BIOGEARS_ARCHITECTURE.md) — the single entry point |
| How do I run the whole thing from scratch? | [`docs/RUN_EVERYTHING.md`](docs/RUN_EVERYTHING.md) |
| What does the model actually consume as input? | [`DATA_CONTRACT.md`](DATA_CONTRACT.md) |
| How does the synthetic dataset work / how do I regenerate it? | [`docs/SYNTHETIC_DATA.md`](docs/SYNTHETIC_DATA.md) |
| How would I plug in real MIMIC-IV data? | [`docs/MIMIC_IV_SETUP.md`](docs/MIMIC_IV_SETUP.md) |
| How does the What-If / ML counterfactual engine work? | [`WHATIF_ARCHITECTURE.md`](WHATIF_ARCHITECTURE.md) |
| Where does BioGears plug in, and what's the boundary? | [`docs/BIOGEARS_INTEGRATION_BOUNDARY.md`](docs/BIOGEARS_INTEGRATION_BOUNDARY.md) |

## Quickstart

```bash
python -m venv venv
source venv/bin/activate       # venv\Scripts\activate on Windows
pip install -r backend/requirements.txt

python generate_synthetic_data.py --patients 300 --seed 42

python -m backend.train --model lstm
python -m backend.train --model transformer

python -m pytest tests/ -q     # expect all tests passing

python -m uvicorn backend.api:app --reload --port 8000   # terminal 1

cd frontend && npm install && npm run dev                 # terminal 2
```

Open `http://localhost:5173`. Full detail, platform-specific notes,
and troubleshooting are in [`docs/RUN_EVERYTHING.md`](docs/RUN_EVERYTHING.md).

**Trained model checkpoints are not committed to this repository.**
`backend/inference_service.py` requires real checkpoints at
`models/lstm_model.pt` / `models/transformer_model.pt` and
`models/scaler.joblib` to start, and fails loudly (not silently, and
never with random/untrained weights) if they're missing or don't match
the current feature contract. Train them yourself with the commands
above — this is deliberate: a checkpoint is a large binary artifact
tied to one specific run of a stochastic training process, and
committing one invites silent drift between the repo's code and the
weights actually being served.

## Architecture at a glance

```
Synthetic MIMIC-IV-shaped data (backend/synthetic_data/)
        |
        v
Preprocessing (backend/preprocessing.py) — resample, missingness, split
        |
        v
Feature engineering (backend/feature_engineering.py) — deltas, rolling stats, shock index
        |
        v
SOFA/Sepsis-3-consistent labeling (backend/labels_sofa.py)
        |
        v
Scaling — fit+applied LAST, after labeling (backend/preprocessing.py::scale_splits)
        |
        v
Trained BiLSTM / Transformer (backend/lstm_model.py, backend/transformer_model.py)
        |
        v
InferenceService (backend/inference_service.py) — real forward passes, cached per window
        |
        +--> Explainability (backend/explainability.py) — real gradient x input saliency
        +--> Digital Twin (backend/digital_twin.py) — real z_t, PCA-projected
        +--> What-If / ML Counterfactual (backend/whatif.py) — same model, modified input
        |
        v
FastAPI backend (backend/api.py)  <-->  React frontend (frontend/)
```

## Repository layout

```
backend/                Python backend: pipeline, model, inference, API
  synthetic_data/        Synthetic MIMIC-IV-shaped data generator + ETL
  mimic_adapter/          Real-MIMIC-IV extraction path (unexercised against real data)
frontend/                React + Vite dashboard
tests/                    pytest suite (backend, pipeline, API, end-to-end)
docs/                     Setup and reference documentation
data/synthetic/           Generated locally by generate_synthetic_data.py — not committed
data/real/                Where a real MIMIC-IV extraction would be placed — never committed
models/                   Trained checkpoints, produced locally by backend/train.py — not committed
config.py                 Central configuration (paths, feature lists, thresholds)
generate_synthetic_data.py         Entry point for synthetic data generation
extract_real_mimic_to_ml_dataset.py   Real-MIMIC-IV extraction (local CSV path)
extract_real_mimic_via_bigquery.py    Real-MIMIC-IV extraction (BigQuery path)
```

## Status

Real (implemented and tested): synthetic data generation, preprocessing,
feature engineering, SOFA labeling, model training, real inference,
real gradient-based explainability, real digital twin state, real API,
real frontend, real ML counterfactual. Not built: BioGears physiological
simulation, real MIMIC-IV/BigQuery activation. See
[`PRE_BIOGEARS_ARCHITECTURE.md`](PRE_BIOGEARS_ARCHITECTURE.md) for the
full, current, verified breakdown of what's real versus not-yet-built.

## Running tests

```bash
python -m pytest tests/ -q
```

Requires `torch` and a trained checkpoint for the model/API/end-to-end
tests; the data/pipeline tests run without either. See
[`docs/RUN_EVERYTHING.md`](docs/RUN_EVERYTHING.md) for the exact
command set and expected results.
