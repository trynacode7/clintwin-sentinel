# ClinTwin Sentinel — ICU Digital Twin Dashboard

Real-time ICU patient risk prediction with explainability, built with FastAPI + React/Vite.

For the full, verified, step-by-step sequence from a clean environment — including training the model, running the full test suite, and walking the complete demo flow — see **`docs/RUN_EVERYTHING.md`**. This README's Quick Start below is the short version.

---

## Quick Start

### 1 — Install backend dependencies

```bash
cd C:\Users\diyap\Documents\projects\clinTwinSentinel\project
pip install -r backend/requirements.txt
```

### 2 — Start the FastAPI backend

```bash
cd C:\Users\diyap\Documents\projects\clinTwinSentinel\project
python -m uvicorn backend.api:app --reload --port 8000
```

The API will be available at **http://localhost:8000**  
Interactive docs: http://localhost:8000/docs

### 3 — Start the React frontend (separate terminal)

```bash
cd C:\Users\diyap\Documents\projects\clinTwinSentinel\project\frontend
npm install
npm run dev
```

The dashboard will be available at **http://localhost:5173**

---

## API Endpoints

The table below is a legacy subset kept for quick reference — it predates several real endpoints this project now has (the `/api/*` roster-driven endpoints, the BioGears/What-If endpoints). See `backend/api.py`'s own module docstring for the complete, current, authoritative endpoint list — it's actively maintained there, not here, so it won't drift out of date the way this table has before.

| Method | Endpoint          | Description                                   |
|--------|-------------------|-----------------------------------------------|
| GET    | `/` / `/health`   | Health check                                  |
| GET    | `/api/roster`     | Real canonical patient roster (primary)       |
| GET    | `/api/patients/{stay_id}` | Real current state for one patient     |
| GET    | `/api/patients/{stay_id}/predict` | Real model prediction           |
| POST   | `/predict`        | Run model inference on custom time-series     |
| GET    | `/patients`, `/patient/{id}` | Legacy aliases, same real data as `/api/*` above |
| GET    | `/demo`           | **Deprecated** — fabricated demo snapshot, kept only for backward compatibility |

### `/patient/{id}` Response Shape

```json
{
  "patient": {
    "id": "patient_001",
    "name": "Patient 001",
    "risk_score": 0.86,
    "status": "Critical",
    "hours_in_icu": 18.4,
    "risk_trend": "increasing"
  },
  "vitals": {
    "timestamps": ["T-72", ...],
    "heart_rate": [78.2, ...],
    "map": [82.1, ...],
    "spo2": [97.0, ...],
    "resp_rate": [16.3, ...],
    "lactate": [null, 1.2, ...]
  },
  "shap": [
    { "feature": "lactate", "value": 0.31, "importance": 0.31, "direction": "risk" },
    ...
  ],
  "attention": {
    "matrix": [[...], ...],
    "labels": ["T-72", ...]
  },
  "latent_trajectory": [
    { "t_idx": 0, "z_2d": [-1.8, 0.9], "risk_score": 0.02 },
    ...
  ],
  "explanation": "Risk increased to 0.86 (Critical) due to ..."
}
```

---

## Frontend → Backend Connection

- **Patient list**: `GET /patients` → sidebar
- **Patient selection**: `GET /patient/{id}` → all six panels
- **Demo fallback**: If the backend is unreachable, locally generated data is used and a yellow banner is displayed

---

## Training (required — checkpoints are gitignored, not shipped)

`models/lstm_model.pt`, `transformer_model.pt`, and `scaler.joblib` are **not** committed to this repo (gitignored) and must be trained locally before the backend can serve real predictions — a fresh clone does not have working models until you run this.

```bash
# BiLSTM — the primary, reported model (train this one)
python -m backend.train --model lstm

# Transformer — kept only as an optional ablation comparison, not a
# second production model (short 6-timestep input window means
# self-attention's main advantage doesn't apply here — see
# backend/train.py's module docstring for the full reasoning)
python -m backend.train --model transformer

# Train on synthetic data (no real CSV needed)
python -m backend.train --model lstm --demo

# Evaluate a trained model, including the required baseline comparison
python -m backend.evaluate --model lstm --with-baselines
```

---

## Project Structure

This is illustrative, not exhaustive — the frontend in particular grows with active feature work (e.g. the BioGears What-If/Scenario Comparison components), so treat `frontend/src/` and `backend/` directory listings as the source of truth over this diagram.

```
project/
├── backend/
│   ├── api.py               # FastAPI application
│   ├── train.py             # Training pipeline (BiLSTM primary; Transformer = optional ablation only)
│   ├── evaluate.py          # Evaluation metrics (incl. required baseline comparison, --with-baselines)
│   ├── baselines.py         # Required logistic-regression + XGBoost baselines
│   ├── lstm_model.py        # BiLSTM architecture (primary)
│   ├── transformer_model.py # Transformer architecture (ablation-only, not production)
│   ├── preprocessing.py     # Data preprocessing
│   ├── feature_engineering.py
│   ├── labels_sofa.py       # Real label generation (SOFA/Sepsis-3) — the one actually used
│   ├── labels.py            # Deprecated ad-hoc rule — NOT used for training/evaluation, kept for history only
│   ├── dataset.py           # PyTorch DataLoader
│   ├── digital_twin.py      # Latent state management
│   ├── explainability.py    # Gradient-attribution explanations (not real SHAP — see module docstring)
│   ├── whatif.py            # ML counterfactual (same trained model, modified input)
│   ├── biogears_*.py        # Real BioGears physiological simulation integration
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── hooks/
│       │   └── usePatientData.js  # Real API client + graceful demo-mode fallback
│       └── components/            # see frontend/src/components/ for the current full list
├── models/                  # gitignored — train locally, see "Training" below
│   ├── lstm_model.pt
│   ├── transformer_model.pt
│   └── scaler.joblib
├── config.py
└── utils.py
```
