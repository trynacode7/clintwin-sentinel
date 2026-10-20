# ClinTwin Sentinel — ICU Digital Twin Dashboard

Real-time ICU patient risk prediction with explainability, built with FastAPI + React/Vite.

---

## Quick Start

### 1 — Install backend dependencies

```bash
cd C:\CAPSTONE\project
pip install -r backend/requirements.txt
```

### 2 — Start the FastAPI backend

```bash
cd C:\CAPSTONE\project
python -m backend.api
```

The API will be available at **http://localhost:8000**  
Interactive docs: http://localhost:8000/docs

### 3 — Start the React frontend (separate terminal)

```bash
cd C:\CAPSTONE\project\frontend
npm install
npm run dev
```

The dashboard will be available at **http://localhost:5173**

---

## API Endpoints

| Method | Endpoint          | Description                                   |
|--------|-------------------|-----------------------------------------------|
| GET    | `/`               | Health check                                  |
| GET    | `/patients`       | List all monitored patients (id, risk, status)|
| GET    | `/patient/{id}`   | Full dashboard data for one patient           |
| GET    | `/demo`           | Global demo snapshot (patient_001 view)       |
| POST   | `/predict`        | Run model inference on custom time-series     |

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

## Training (optional — models already exist)

```bash
# Train LSTM
python -m backend.train --model lstm

# Train Transformer
python -m backend.train --model transformer

# Train on synthetic data (no real CSV needed)
python -m backend.train --model lstm --demo
```

---

## Project Structure

```
project/
├── backend/
│   ├── api.py               # FastAPI application
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation metrics
│   ├── lstm_model.py        # BiLSTM architecture
│   ├── transformer_model.py # Transformer architecture
│   ├── preprocessing.py     # Data preprocessing
│   ├── feature_engineering.py
│   ├── labels.py            # Risk label generation
│   ├── dataset.py           # PyTorch DataLoader
│   ├── digital_twin.py      # Latent state management
│   ├── explainability.py    # SHAP + NL explanations
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── hooks/
│       │   └── usePatientData.js  # API client + fallback
│       └── components/
│           ├── RiskPanel.jsx
│           ├── VitalsTimeline.jsx
│           ├── ShapPanel.jsx
│           ├── LatentStateViz.jsx
│           ├── AttentionHeatmap.jsx
│           └── ClinicalExplanation.jsx
├── models/
│   ├── lstm_model.pt
│   ├── transformer_model.pt
│   └── scaler.joblib
├── config.py
└── utils.py
```
