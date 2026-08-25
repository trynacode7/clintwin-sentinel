"""
config.py — Global Configuration for ICU Digital Twin System
=============================================================
Centralizes all hyperparameters, paths, feature definitions,
and training settings for reproducibility and easy tuning.

Phase 1 changes (synthetic MIMIC-IV-compatible development layer):
  - Explicit DATA_MODE ("synthetic" | "real") replaces the old silent
    DEMO_MODE fallback. The system must never silently fall back from
    real-data mode to synthetic data — see preprocessing.load_data().
  - Feature set extended with SOFA-relevant labs/vitals/treatment
    variables (DATA_CONTRACT.md section 3).
  - Resampling grid moved from 5-minute to 1-hour
    (TECHNICAL_BLUEPRINT.md section 2 — MIMIC-IV vitals are charted
    roughly hourly; a 5-minute grid is mostly forward-fill, not signal).
  - Windowing changed to a 6-hour input window predicting a 2-6 hour
    horizon (TECHNICAL_BLUEPRINT.md section 3), at 1-hour resolution.
"""

import os

# ─── Data mode (explicit — no silent fallback) ──────────────────────────────
# "synthetic": read from SYNTHETIC_DATA_DIR/processed/ml_dataset.csv — this is
#              engineering/development data only, never clinical evidence.
# "real":      read from REAL_DATA_PATH (PhysioNet-credentialed MIMIC-IV
#              extraction, produced entirely outside this repository/chat).
# If DATA_MODE="real" and REAL_DATA_PATH does not exist, the system raises —
# it does NOT fall back to synthetic data.
DATA_MODE = os.environ.get("CLINTWIN_DATA_MODE", "synthetic").strip().lower()
if DATA_MODE not in ("synthetic", "real"):
    raise ValueError(f"CLINTWIN_DATA_MODE must be 'synthetic' or 'real', got {DATA_MODE!r}")

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DATA_DIR = os.environ.get("CLINTWIN_SYNTHETIC_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "synthetic"))
REAL_DATA_DIR = os.environ.get("CLINTWIN_REAL_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "real"))
REAL_DATA_PATH = os.environ.get("CLINTWIN_REAL_DATA_PATH", os.path.join(REAL_DATA_DIR, "ml_dataset.csv"))
# Phase 2: where the RAW, per-table MIMIC-IV CSVs (patients.csv, admissions.csv,
# icustays.csv, chartevents.csv, labevents.csv, inputevents.csv, outputevents.csv,
# prescriptions.csv, microbiologyevents.csv, optionally procedureevents.csv) are
# expected before extract_real_mimic_to_ml_dataset.py runs the adapter — see
# docs/MIMIC_IV_SETUP.md. Distinct from REAL_DATA_PATH, which is the ALREADY-
# EXTRACTED flat ml_dataset.csv that preprocessing.load_data() reads.
REAL_DATA_RAW_DIR = os.environ.get("CLINTWIN_REAL_DATA_RAW_DIR", os.path.join(REAL_DATA_DIR, "raw"))
SYNTHETIC_ML_DATASET_PATH = os.path.join(SYNTHETIC_DATA_DIR, "processed", "ml_dataset.csv")

DATA_PATH = REAL_DATA_PATH if DATA_MODE == "real" else SYNTHETIC_ML_DATASET_PATH

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_model.pt")
LOGREG_MODEL_PATH = os.path.join(MODEL_DIR, "logreg_baseline.joblib")
GBT_MODEL_PATH = os.path.join(MODEL_DIR, "gbt_baseline.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Feature Definitions ────────────────────────────────────────────────────
ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TIME_COL = "charttime"

# NOTE: feature_engineering.add_shock_index() hardcodes the literal column
# names "heart_rate" and "map" — do not rename these without also updating
# that function (see DATA_CONTRACT.md section 3).
VITAL_COLS = ["heart_rate", "map", "resp_rate", "spo2", "temp"]
LAB_COLS = ["lactate", "wbc", "creatinine", "bilirubin", "platelets"]
GCS_COLS = ["gcs"]
RESP_SUPPORT_COLS = ["fio2"]
VENT_COLS = ["ventilated"]
OUTPUT_COLS = ["urine_output_rate"]
TREATMENT_COLS = ["vasopressor_dose", "fluid_bolus_ml"]
DEMOGRAPHIC_COLS = ["age", "sex_male"]

ALL_FEATURE_COLS = (
    VITAL_COLS + LAB_COLS + GCS_COLS + RESP_SUPPORT_COLS + VENT_COLS
    + OUTPUT_COLS + TREATMENT_COLS + DEMOGRAPHIC_COLS
)

# Columns present in ml_dataset.csv that must NEVER be used as model input.
# `suspected_infection` defines label eligibility (Sepsis-3 cohort criterion)
# and is deliberately withheld from the feature set for the same reason the
# original prototype never fed `risk_flag` to the model — see labels_sofa.py
# and DATA_CONTRACT.md section 6.
EXCLUDE_COLS = ["risk_flag", "suspected_infection"]
LABEL_ONLY_COLS = ["suspected_infection"]

# Column-level resampling/fill strategy, used by preprocessing.resample_stay().
# "interpolate": forward-fill then linear-interpolate (continuously-varying
#                physiological signals — vitals, FiO2).
# "locf":        forward-fill only, no interpolation (sparse labs, GCS,
#                urine rate, infusion status that holds until changed).
# "sum":         resample-sum within each bin, fill empty bins with 0 (event
#                amounts that must never be carried forward — fluid boluses).
COLUMN_FILL_STRATEGY = {
    **{c: "interpolate" for c in VITAL_COLS},
    **{c: "locf" for c in LAB_COLS},
    **{c: "locf" for c in GCS_COLS},
    "fio2": "interpolate",
    "ventilated": "locf",
    "urine_output_rate": "locf",
    "vasopressor_dose": "locf",
    "fluid_bolus_ml": "sum",
    "age": "locf",
    "sex_male": "locf",
    # label-construction-only column (never a model feature — see LABEL_ONLY_COLS)
    "suspected_infection": "locf",
}

# ─── Preprocessing ──────────────────────────────────────────────────────────
RESAMPLE_FREQ = "1h"  # 1-hour interval resampling (see module docstring)

# ─── Windowing ───────────────────────────────────────────────────────────────
INPUT_WINDOW = 6          # 6 hourly timesteps of input (6h observation window)
FUTURE_WINDOW_START = 2   # horizon starts 2h after the input window ends
FUTURE_WINDOW_END = 6     # horizon ends 6h after the input window ends
SLIDING_STEP = 1          # step size for sliding window (hours)

# ─── Legacy label thresholds (labels.py — preserved, unused by default) ─────
# Kept only so the original prototype's labels.py continues to import
# cleanly; the default labeling path is labels_sofa.py (see train.py).
LACTATE_THRESHOLD = 2.0
MAP_THRESHOLD = 65.0
HR_THRESHOLD = 100.0
HR_CONSECUTIVE_STEPS = 3

# ─── SOFA / Sepsis-3-consistent label (labels_sofa.py) ──────────────────────
SOFA_INCREASE_THRESHOLD = 2         # >= this point increase within the horizon => positive label
# FIX 1 (Phase 1 audit remediation): this MUST be independent of the legacy
# MAP_THRESHOLD above (65 mmHg) — that value is the old ad hoc labels.py
# rule's threshold, not the published SOFA cardiovascular criterion. The
# real SOFA cardiovascular component (Vincent et al. 1996) awards 1 point
# for MAP < 70 mmHg, not < 65. Do not point this at MAP_THRESHOLD again.
SOFA_CARDIOVASCULAR_MAP_THRESHOLD = 70.0  # mmHg — correct published SOFA threshold (was 65, fixed post-audit)

# ─── Feature Engineering ─────────────────────────────────────────────────────
ROLLING_WINDOW = 4  # 4 hourly steps — a clinically meaningful short-term trend window at 1h resolution

# ─── Model Architecture ─────────────────────────────────────────────────────
# LSTM
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.2
LSTM_BIDIRECTIONAL = True

# Transformer
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128
TRANSFORMER_DROPOUT = 0.1

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRAD_CLIP_MAX_NORM = 1.0
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# ─── Data Split Ratios (by subject_id) ──────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Random Seed ─────────────────────────────────────────────────────────────
SEED = 42

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
