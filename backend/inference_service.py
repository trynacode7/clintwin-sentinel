"""
inference_service.py — Single Authoritative Inference Path (Phase B)
========================================================================
This module is the ONE place that turns "a stay_id from the roster"
into "a real model prediction, a real explanation, a real trajectory,
and a real digital-twin state." Nothing downstream of this module (API
endpoints, and the frontend) should compute a risk score, gradient
attribution, z_t, or trajectory point on its own — they all come from
here.

NO FABRICATION, ANYWHERE IN THIS MODULE:
  - Feature values: reuses `preprocessing.preprocess()` and
    `feature_engineering.engineer_features()` VERBATIM — the exact same
    functions `train.py` uses to build the data the checkpoint was
    trained on. No second, divergent feature-computation path.
  - Feature CONTRACT: the checkpoint's saved `feature_cols` (see
    models/lstm_model.pt) is treated as an explicit contract. On
    startup this module recomputes the pipeline's feature columns fresh
    and compares them to the checkpoint's list EXACTLY (order-sensitive
    — the model indexes features positionally). Any mismatch raises
    FeatureContractError immediately; it never silently predicts from a
    misaligned vector.
  - Predictions/explanations/z_t: all come from one real forward pass
    of the trained model per window (see `_run_all_predictions`). The
    attribution reuses `explainability.gradient_x_input_attribution()`
    (Phase D: renamed from `_gradient_shap_approximation` — the old name
    itself said "SHAP" despite not being SHAP; see explainability.py's
    docstring) — the same function the legacy POST /predict endpoint
    uses — and is surfaced under the field name `gradient_attribution`,
    NEVER `shap_values`: it is real input×gradient saliency, not a
    Shapley value. Real SHAP is never invoked anywhere in this module.

WHY PREDICTIONS ARE CACHED, NOT RECOMPUTED PER REQUEST (design decision
— documented per the Phase B "document your reasoning" instruction):
  The test-split synthetic cohort is static for the lifetime of this
  process — there is no live incoming vitals stream in Phase B. So this
  service runs each window through the model EXACTLY ONCE, at service
  construction time, and caches the result. This has two benefits
  beyond speed: (1) repeated requests for the same patient/window get
  byte-identical answers, which is what "real, reproducible inference"
  should look like for a fixed cohort; (2) DigitalTwin.extract_trajectory
  APPENDS a new trajectory point every time it's called — if a
  request handler called it again on every API call, the twin's stored
  trajectory for that patient would grow a new (duplicate) point per
  request, corrupting the very trajectory the demo is supposed to show.
  Running each window through the twin exactly once, in chronological
  order, at startup avoids that entirely. When real streaming data
  exists (post real-MIMIC integration), this caching assumption is the
  first thing that should be revisited.

PIPELINE ORDERING FIX (local-machine reconciliation phase): this module
used to call `preprocessing.preprocess()` (which scaled the raw
measurement columns) and only THEN call `engineer_features()` and
`labels_sofa.create_labels()`. That silently fed z-scored values into
SOFA's absolute clinical thresholds (e.g. "platelets < 150", "MAP <
70") and into the clinically-meaningful shock_index ratio, corrupting
both. Fixed by splitting `preprocess()` into a RAW step (load/resample/
missingness/split — no scaling) and a separate `scale_splits()` step
that now runs LAST, after engineering and labeling. This module keeps
a RAW (unscaled) per-stay dataframe (`self._raw_by_stay`) specifically
so the true ML counterfactual (see below) can modify a real clinical
value and correctly recompute engineered features from it — this
would not have been possible correctly with the old, scale-first
order.

TRUE ML COUNTERFACTUAL (`compute_counterfactual`, below): given a
stay_id, a window, and one or more interventions (e.g. "what if FiO2
had been 0.6 instead of whatever was observed"), this module:
  1. takes that stay's REAL raw (unscaled) observed window,
  2. overwrites the intervened raw column(s) at every timestep in that
     window with the requested value (and marks them as "observed", not
     missing),
  3. re-runs `feature_engineering.engineer_features()` on the WHOLE
     modified stay (deltas/rolling stats need the surrounding raw
     series, exactly as at training time),
  4. applies the SAME fitted scaler used for the real baseline,
  5. runs the SAME trained model checkpoint on the modified window.
This is a real forward pass through the real pipeline on modified
input — never an edited number, an interpolation, or a fabricated
trajectory. See `WhatIfScenarioResult` in `backend/api.py` and
`backend/whatif.py` for how this is surfaced, clearly labeled "ML
Counterfactual" and kept distinct from the (still unavailable)
BioGears physiological simulation.

WHY THIS DOES NOT REUSE `dataset.ICUDataset` FOR WINDOWING (documented
deviation): `ICUDataset` requires a `label` column and silently skips
any window whose label is NaN (`if np.isnan(y): continue` — see
dataset.py). That's correct for TRAINING (you can't train on an unknown
label) but wrong for LIVE INFERENCE: `labels_sofa.create_labels()`
marks a window's label NaN whenever there isn't yet enough FUTURE data
to know the outcome — which is *always true* of the most recent window
of any ongoing stay (that's the whole point of predicting on it!). If
Phase B windowing reused ICUDataset as-is, the demo's "current patient
state" would always be one window stale, or unavailable for short
stays. So this module keeps its own thin windowing loop, deliberately
mirroring ICUDataset's window_size / step_size / NaN-fill / mask
conventions exactly, but keyed by (stay_id, window_idx) and never
gated on label availability. Ground-truth labels are still attached
where they exist (`Window.ground_truth_label`), just not used as a
filter.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.preprocessing import preprocess, scale_splits, apply_scaler
from backend.feature_engineering import engineer_features, get_engineered_feature_cols
from backend.labels_sofa import create_labels
from backend.lstm_model import ICUBiLSTM
from backend.transformer_model import ICUTransformer
from backend.digital_twin import DigitalTwin
from backend.explainability import (
    gradient_x_input_attribution,
    compute_global_importance,
    compute_local_explanation,
    generate_clinical_explanation,
)
from backend.patient_roster import RosterPatient, build_roster_from_unscaled_df

logger = logging.getLogger("icu_digital_twin")


class FeatureContractError(RuntimeError):
    """Raised when the checkpoint's saved feature_cols do not exactly
    (order-sensitive) match the freshly-engineered pipeline output.
    Never caught silently — this must surface as a hard failure."""


class NoTrainedModelError(RuntimeError):
    """Raised when no trained checkpoint exists for the requested
    model_type. inference_service.py never falls back to an untrained
    or randomly-initialized model — that would silently serve
    fake-quality predictions labeled as real ones."""


class UnknownStayIdError(KeyError):
    """Raised when a requested stay_id is not in the canonical roster."""


class ModelOutputCorrectnessError(RuntimeError):
    """
    Raised when a real forward pass produces a non-finite (NaN/Inf) risk
    score or latent state (Phase C: live model output correctness).

    This should never happen with valid inputs and a healthy checkpoint
    — if it does, it's a genuine bug (e.g. a corrupted checkpoint, a
    NaN that escaped the nan_to_num window-building step) worth
    surfacing loudly. NaN/Inf are also not valid JSON — silently
    letting one reach an API response would either crash JSON
    serialization or hand the frontend a literal `NaN` token that
    breaks `JSON.parse` in the browser. Fail here, at the source,
    instead.
    """


@dataclass
class Window:
    """One real (stay_id, time-range) inference window. `X_scaled` is
    exactly what the model consumes; `unscaled_row` is the same
    timestep's values in original clinical units (heart rate in bpm,
    not z-scores) for human-facing display."""

    stay_id: int
    subject_id: int
    hadm_id: int
    window_idx: int
    start_time: str
    end_time: str
    X_scaled: np.ndarray
    unscaled_row: dict
    ground_truth_label: Optional[float]


class InferenceService:
    """
    Loads the trained model and the synthetic test-split data ONCE,
    builds the canonical roster and real inference windows, runs every
    window through the model exactly once, and serves the cached real
    results. Constructed once as a process-wide singleton (see
    `get_inference_service()` below).
    """

    def __init__(self, model_type: str = "lstm", split: str = "test", data_path: Optional[str] = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split!r}")
        self.model_type = model_type
        self.split = split
        self.data_path = data_path  # None => preprocess()'s own default (config.DATA_PATH)
        self.model: Optional[torch.nn.Module] = None
        self.checkpoint_meta: dict = {}
        self.feature_cols: List[str] = []
        self.scaler = None
        self.roster: Dict[int, RosterPatient] = {}
        self.digital_twin = DigitalTwin(latent_dim=config.LSTM_HIDDEN_SIZE * 2)

        self._windows_by_stay: Dict[int, List[Window]] = {}
        self._predictions_by_stay: Dict[int, List[dict]] = {}
        # RAW (unscaled, pre-feature-engineering) per-stay dataframes for
        # this service's split — the real clinical-unit basis the true ML
        # counterfactual modifies. See compute_counterfactual().
        self._raw_by_stay: Dict[int, "pd.DataFrame"] = {}

        t0 = time.time()
        self._load_model(model_type)
        self._build_pipeline_and_roster()
        self._run_all_predictions()
        logger.info(
            f"InferenceService ready in {time.time() - t0:.1f}s: "
            f"{len(self.roster)} patients, "
            f"{sum(len(w) for w in self._windows_by_stay.values())} real inference windows."
        )

    # ── model loading (strict — no silent untrained fallback) ───────────

    def _load_model(self, model_type: str):
        if model_type == "lstm":
            path, ModelClass = config.LSTM_MODEL_PATH, ICUBiLSTM
        elif model_type == "transformer":
            path, ModelClass = config.TRANSFORMER_MODEL_PATH, ICUTransformer
        else:
            raise ValueError(f"Unknown model_type {model_type!r} (expected 'lstm' or 'transformer')")

        if not os.path.exists(path):
            raise NoTrainedModelError(
                f"No trained '{model_type}' checkpoint found at {path}. "
                f"inference_service.py refuses to fall back to an untrained/randomly "
                f"initialized model — that would silently produce fake-quality "
                f"predictions labeled as real ones. Train a model first: "
                f"`python -m backend.train` (see backend/train.py)."
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ModelClass(input_size=checkpoint["input_size"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.feature_cols = list(checkpoint["feature_cols"])
        self.checkpoint_meta = {
            "model_type": checkpoint.get("model_type", model_type),
            "input_size": checkpoint["input_size"],
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "checkpoint_path": path,
        }
        logger.info(
            f"InferenceService: loaded REAL trained {model_type} checkpoint from {path} "
            f"(epoch={self.checkpoint_meta['epoch']}, val_loss={self.checkpoint_meta['val_loss']}, "
            f"input_size={self.checkpoint_meta['input_size']})"
        )

    # ── pipeline + roster (single authoritative feature path) ───────────

    def _build_pipeline_and_roster(self):
        # RAW step — no scaling yet. See preprocessing.preprocess()'s
        # docstring for why scaling must not happen before feature
        # engineering / SOFA labeling.
        result = preprocess(data_path=self.data_path)  # the exact function train.py uses — no duplicated logic

        # Keep the RAW (unscaled, pre-engineering) per-stay slices for this
        # service's split — this is the real basis the true ML
        # counterfactual (compute_counterfactual, below) modifies.
        raw_split_df = result[self.split].sort_values(["stay_id", config.TIME_COL]).reset_index(drop=True)
        self._raw_by_stay = {
            int(sid): group.reset_index(drop=True)
            for sid, group in raw_split_df.groupby("stay_id")
        }

        # Engineer + label ALL splits on RAW clinical units (the scaler must
        # be fit on train, and SOFA/shock_index both require real units).
        engineered_all = {s: engineer_features(result[s]) for s in ("train", "val", "test")}
        labeled_all = {s: create_labels(engineered_all[s]) for s in ("train", "val", "test")}

        fresh_feature_cols = get_engineered_feature_cols(labeled_all["train"])
        if fresh_feature_cols != self.feature_cols:
            missing_in_ckpt = sorted(set(fresh_feature_cols) - set(self.feature_cols))
            missing_in_fresh = sorted(set(self.feature_cols) - set(fresh_feature_cols))
            raise FeatureContractError(
                "Feature contract mismatch between the trained checkpoint "
                f"({self.checkpoint_meta.get('checkpoint_path')}) and the current "
                "preprocessing.preprocess() + feature_engineering.engineer_features() "
                "pipeline output.\n"
                f"  checkpoint feature_cols: {len(self.feature_cols)} columns\n"
                f"  freshly engineered feature_cols: {len(fresh_feature_cols)} columns\n"
                f"  in fresh, not in checkpoint: {missing_in_ckpt}\n"
                f"  in checkpoint, not in fresh: {missing_in_fresh}\n"
                "This model must be retrained on the current pipeline output before it "
                "can be used for inference (see backend/train.py). Refusing to predict "
                "from a misaligned feature vector."
            )
        logger.info(
            f"Feature contract verified: checkpoint's {len(self.feature_cols)} feature_cols "
            f"exactly match the current pipeline output, in order."
        )

        # Scale (Z-score) LAST — only now, after engineering + labeling used
        # real clinical units.
        scaled_all = scale_splits(labeled_all)
        self.scaler = scaled_all["scaler"]

        # Real clinical-unit copy (engineered + labeled, NOT scaled) for
        # human-readable display and ground truth — no inverse_transform
        # round-trip needed, this IS the real unscaled data.
        unscaled = labeled_all[self.split]
        label_lookup = {
            (int(sid), t): float(lab)
            for sid, t, lab in zip(unscaled["stay_id"], unscaled[config.TIME_COL], unscaled["label"])
        }

        roster_list = build_roster_from_unscaled_df(unscaled, split=self.split)
        self.roster = {p.stay_id: p for p in roster_list}

        # Model-input (scaled) version of the same split — same rows/order
        # as `unscaled` (apply_scaler only rewrites column values in place).
        engineered = scaled_all[self.split]

        window_size = config.INPUT_WINDOW
        step = config.INPUT_WINDOW  # non-overlapping — matches dataset.py's eval-split convention

        windows: Dict[int, List[Window]] = {}
        for stay_id, group in engineered.groupby("stay_id"):
            stay_id = int(stay_id)
            group = group.sort_values(config.TIME_COL).reset_index(drop=True)
            unscaled_group = (
                unscaled[unscaled["stay_id"] == stay_id]
                .sort_values(config.TIME_COL)
                .reset_index(drop=True)
            )
            feats = group[self.feature_cols].values.astype(np.float32)
            n_steps = len(group)
            stay_windows: List[Window] = []

            if n_steps < window_size:
                # Left-pad, single window — mirrors dataset.py's short-stay handling.
                pad_len = window_size - n_steps
                X = np.zeros((window_size, feats.shape[1]), dtype=np.float32)
                X[pad_len:] = feats
                X = np.nan_to_num(X, nan=0.0)
                start_row, end_row = group.iloc[0], group.iloc[-1]
                gt = label_lookup.get((stay_id, end_row[config.TIME_COL]))
                stay_windows.append(
                    Window(
                        stay_id=stay_id,
                        subject_id=int(end_row["subject_id"]),
                        hadm_id=int(end_row["hadm_id"]),
                        window_idx=0,
                        start_time=str(start_row[config.TIME_COL]),
                        end_time=str(end_row[config.TIME_COL]),
                        X_scaled=X,
                        unscaled_row=unscaled_group.iloc[-1].to_dict(),
                        ground_truth_label=gt,
                    )
                )
            else:
                for widx, start in enumerate(range(0, n_steps - window_size + 1, step)):
                    end = start + window_size
                    X = np.nan_to_num(feats[start:end].copy(), nan=0.0)
                    start_row, end_row = group.iloc[start], group.iloc[end - 1]
                    gt = label_lookup.get((stay_id, end_row[config.TIME_COL]))
                    stay_windows.append(
                        Window(
                            stay_id=stay_id,
                            subject_id=int(end_row["subject_id"]),
                            hadm_id=int(end_row["hadm_id"]),
                            window_idx=widx,
                            start_time=str(start_row[config.TIME_COL]),
                            end_time=str(end_row[config.TIME_COL]),
                            X_scaled=X,
                            unscaled_row=unscaled_group.iloc[end - 1].to_dict(),
                            ground_truth_label=gt,
                        )
                    )

            windows[stay_id] = stay_windows

        self._windows_by_stay = windows

    # ── real inference, run once per window at startup ──────────────────

    def _warm_pca(self):
        """
        Real forward pass across every window, z_t only, purely to give
        DigitalTwin.warm_pca() a representative batch before any trajectory
        point is recorded — avoids the first ~10-processed-patients-get-
        [0,0] cold-start artifact (see digital_twin.py's warm_pca
        docstring). This is a genuine extra real forward pass, not a
        fabricated shortcut — z_t here comes from the same trained model,
        just discarded after collection.
        """
        model_type = self.checkpoint_meta["model_type"]
        all_zt = []
        with torch.no_grad():
            for stay_id in sorted(self._windows_by_stay.keys()):
                for window in self._windows_by_stay[stay_id]:
                    X = torch.tensor(window.X_scaled).unsqueeze(0)
                    mask = torch.ones(1, X.shape[1])
                    if model_type == "transformer":
                        _, z_t, _ = self.model(X, mask)
                    else:
                        _, z_t = self.model(X, mask)
                    all_zt.append(z_t.squeeze(0).cpu().numpy())
        self.digital_twin.warm_pca(all_zt)
        logger.info(f"Digital twin PCA warmed on {len(all_zt)} real z_t vectors from the full roster.")

    def _run_all_predictions(self):
        model_type = self.checkpoint_meta["model_type"]
        device = torch.device("cpu")

        self._warm_pca()

        for stay_id in sorted(self._windows_by_stay.keys()):
            stay_predictions = []
            for window in self._windows_by_stay[stay_id]:
                X = torch.tensor(window.X_scaled).unsqueeze(0)
                mask = torch.ones(1, X.shape[1])

                self.model.eval()
                with torch.no_grad():
                    if model_type == "transformer":
                        risk, z_t, attn = self.model(X, mask)
                        attn_list = [a.cpu().numpy().tolist() for a in attn]
                    else:
                        risk, z_t = self.model(X, mask)
                        attn_list = None

                risk_score = float(risk.squeeze().cpu().item())
                z_t_np = z_t.squeeze(0).detach().cpu().numpy()

                if not np.isfinite(risk_score):
                    raise ModelOutputCorrectnessError(
                        f"Non-finite risk_score ({risk_score!r}) from a real forward pass on "
                        f"stay_id={stay_id}, window_idx={window.window_idx}. Refusing to serve "
                        f"this prediction — this indicates a genuine model/data bug, not a "
                        f"formatting issue."
                    )
                if not np.all(np.isfinite(z_t_np)):
                    raise ModelOutputCorrectnessError(
                        f"Non-finite value(s) in z_t from a real forward pass on "
                        f"stay_id={stay_id}, window_idx={window.window_idx}. Refusing to serve "
                        f"this prediction."
                    )

                status = "Critical" if risk_score >= 0.7 else ("Warning" if risk_score >= 0.4 else "Stable")

                grad_attr = gradient_x_input_attribution(self.model, X, mask, model_type, device)
                if not np.all(np.isfinite(grad_attr)):
                    raise ModelOutputCorrectnessError(
                        f"Non-finite value(s) in gradient attribution for stay_id={stay_id}, "
                        f"window_idx={window.window_idx}. Refusing to serve this explanation."
                    )
                global_importance = compute_global_importance(grad_attr, self.feature_cols, top_k=10)
                local_explanation = compute_local_explanation(grad_attr, self.feature_cols, sample_idx=0, top_k=10)
                explanation_text = generate_clinical_explanation(
                    grad_attr, self.feature_cols, risk_score, sample_idx=0
                )

                # Exactly ONE real trajectory point appended per window,
                # in chronological order — see module docstring on caching.
                twin_result = self.digital_twin.extract_trajectory(
                    self.model, X, mask, patient_id=str(stay_id), model_type=model_type
                )

                stay_predictions.append(
                    {
                        "stay_id": stay_id,
                        "subject_id": window.subject_id,
                        "hadm_id": window.hadm_id,
                        "window_idx": window.window_idx,
                        "window_start_time": window.start_time,
                        "window_end_time": window.end_time,
                        "risk_score": risk_score,
                        "status": status,
                        "z_t": twin_result["z_t"],
                        "z_2d": twin_result["z_2d"],
                        "gradient_attribution": global_importance,
                        "local_gradient_attribution": local_explanation,
                        "attention_weights": attn_list,
                        "explanation": explanation_text,
                        "explanation_method": (
                            "gradient_x_input saliency (real, computed from this model's "
                            "actual gradients) — NOT a Shapley-value SHAP explanation. "
                            "Deliberately not named 'shap_values'; see explainability.py."
                        ),
                        "ground_truth_label": window.ground_truth_label,
                        "observed_vitals": {
                            k: (None if pd.isna(v) else v)
                            for k, v in window.unscaled_row.items()
                            if k in config.ALL_FEATURE_COLS
                        },
                        "model_type": model_type,
                        "model_epoch": self.checkpoint_meta.get("epoch"),
                        "model_checkpoint": os.path.basename(self.checkpoint_meta.get("checkpoint_path", "")),
                    }
                )

            self._predictions_by_stay[stay_id] = stay_predictions

            if stay_predictions:
                latest = stay_predictions[-1]
                roster_entry = self.roster[stay_id]
                roster_entry.latest_risk_score = latest["risk_score"]
                roster_entry.latest_status = latest["status"]
                roster_entry.n_windows = len(stay_predictions)

    # ── true ML counterfactual (real, no BioGears involved) ─────────────

    def compute_counterfactual(
        self, stay_id: int, window_idx: int, interventions: List[Dict[str, float]]
    ) -> dict:
        """
        Real ML counterfactual — NOT a physiological simulation, NOT a
        BioGears result, NOT an edited/interpolated number.

        Pipeline (mirrors training/baseline inference exactly, just on a
        modified input):
          1. Start from this stay's REAL raw (unscaled) observed window
             (`self._raw_by_stay`).
          2. Overwrite the intervened raw column(s) at EVERY timestep
             within that window (the scope is deliberately limited to the
             window being predicted from, not the patient's whole history
             — see backend/whatif.py), and mark those cells as observed
             (not missing).
          3. Re-run `feature_engineering.engineer_features()` on the WHOLE
             modified stay — deltas/rolling stats need the surrounding raw
             series, exactly as at training time. Recomputing only the
             window slice would silently drop that lookback context.
          4. Apply the SAME fitted scaler used for the real baseline (no
             separate statistics).
          5. Run the SAME trained model checkpoint on the modified window.

        Returns a dict with the counterfactual risk_score, status,
        latent state, and gradient attribution — all from one real
        forward pass. Raises ModelOutputCorrectnessError on a non-finite
        output, exactly like the baseline path.
        """
        self._require_stay(stay_id)
        if stay_id not in self._raw_by_stay:
            raise UnknownStayIdError(f"No raw data cached for stay_id {stay_id}.")

        raw_stay = self._raw_by_stay[stay_id].copy()
        n_steps = len(raw_stay)
        window_size = config.INPUT_WINDOW
        step = config.INPUT_WINDOW

        if n_steps < window_size:
            start, end = 0, n_steps
        else:
            start = window_idx * step
            end = start + window_size
            if start < 0 or end > n_steps:
                raise UnknownStayIdError(
                    f"window_idx {window_idx} does not exist for stay_id {stay_id}."
                )

        window_index_labels = raw_stay.index[start:end]

        for item in interventions:
            feature = item["feature"]
            value = float(item["value"])
            if feature not in config.ALL_FEATURE_COLS:
                # Should never happen — whatif.py validates against
                # INTERVENABLE_FEATURES (a subset of ALL_FEATURE_COLS) before
                # this is ever called. Refuse rather than silently no-op.
                raise ValueError(
                    f"{feature!r} is not a real model feature — refusing to "
                    f"fabricate a counterfactual for it."
                )
            raw_stay.loc[window_index_labels, feature] = value
            missing_col, tso_col = f"{feature}_missing", f"{feature}_time_since_obs"
            if missing_col in raw_stay.columns:
                raw_stay.loc[window_index_labels, missing_col] = 0.0
            if tso_col in raw_stay.columns:
                raw_stay.loc[window_index_labels, tso_col] = 0.0

        modified_engineered = engineer_features(raw_stay)
        modified_scaled = apply_scaler(modified_engineered, self.scaler, list(config.ALL_FEATURE_COLS))

        feats = modified_scaled[self.feature_cols].values.astype(np.float32)
        if n_steps < window_size:
            pad_len = window_size - n_steps
            X = np.zeros((window_size, feats.shape[1]), dtype=np.float32)
            X[pad_len:] = feats
        else:
            X = feats[start:end].copy()
        X = np.nan_to_num(X, nan=0.0)

        model_type = self.checkpoint_meta["model_type"]
        device = torch.device("cpu")
        X_t = torch.tensor(X).unsqueeze(0)
        mask = torch.ones(1, X_t.shape[1])

        self.model.eval()
        with torch.no_grad():
            if model_type == "transformer":
                risk, z_t, _ = self.model(X_t, mask)
            else:
                risk, z_t = self.model(X_t, mask)
        risk_score = float(risk.squeeze().cpu().item())
        z_t_np = z_t.squeeze(0).detach().cpu().numpy()

        if not np.isfinite(risk_score):
            raise ModelOutputCorrectnessError(
                f"Non-finite counterfactual risk_score for stay_id={stay_id}, "
                f"window_idx={window_idx}. Refusing to serve this result."
            )
        if not np.all(np.isfinite(z_t_np)):
            raise ModelOutputCorrectnessError(
                f"Non-finite counterfactual latent state for stay_id={stay_id}, "
                f"window_idx={window_idx}."
            )

        status = "Critical" if risk_score >= 0.7 else ("Warning" if risk_score >= 0.4 else "Stable")

        grad_attr = gradient_x_input_attribution(self.model, X_t, mask, model_type, device)
        if not np.all(np.isfinite(grad_attr)):
            raise ModelOutputCorrectnessError(
                f"Non-finite counterfactual gradient attribution for stay_id={stay_id}, "
                f"window_idx={window_idx}."
            )
        global_importance = compute_global_importance(grad_attr, self.feature_cols, top_k=10)

        z_2d = self.digital_twin.project_to_2d(z_t_np)
        z_2d_list = z_2d.tolist() if z_2d is not None else None

        return {
            "risk_score": risk_score,
            "status": status,
            "z_t": z_t_np.tolist(),
            "z_2d": z_2d_list,
            "gradient_attribution": global_importance,
            "explanation_method": (
                "gradient_x_input saliency (real, computed from this model's "
                "actual gradients on the counterfactual input) — NOT a "
                "Shapley-value SHAP explanation."
            ),
        }

    # ── public read API (consumed by backend/api.py) ────────────────────

    def _require_stay(self, stay_id: int):
        if stay_id not in self._predictions_by_stay or not self._predictions_by_stay[stay_id]:
            raise UnknownStayIdError(
                f"stay_id {stay_id} is not in the canonical roster (split={self.split}). "
                f"Use GET /api/roster to list valid stay_ids."
            )

    def get_roster(self) -> List[dict]:
        return [self.roster[sid].to_dict() for sid in sorted(self.roster.keys())]

    def get_prediction(self, stay_id: int, window_idx: Optional[int] = None) -> dict:
        self._require_stay(stay_id)
        preds = self._predictions_by_stay[stay_id]
        if window_idx is None:
            return preds[-1]
        matches = [p for p in preds if p["window_idx"] == window_idx]
        if not matches:
            raise UnknownStayIdError(
                f"window_idx {window_idx} does not exist for stay_id {stay_id} "
                f"(valid range: 0..{len(preds) - 1})"
            )
        return matches[0]

    def get_current_state(self, stay_id: int) -> dict:
        """Real latest observed vitals + real latest prediction for this stay."""
        latest = self.get_prediction(stay_id)
        roster_entry = self.roster[stay_id].to_dict()
        return {
            "patient": roster_entry,
            "observed_vitals": latest["observed_vitals"],
            "window_end_time": latest["window_end_time"],
            "prediction": {
                "risk_score": latest["risk_score"],
                "status": latest["status"],
                "model_type": latest["model_type"],
                "model_epoch": latest["model_epoch"],
                "model_checkpoint": latest["model_checkpoint"],
            },
        }

    def get_explanation(self, stay_id: int, window_idx: Optional[int] = None) -> dict:
        pred = self.get_prediction(stay_id, window_idx)
        return {
            "stay_id": stay_id,
            "window_idx": pred["window_idx"],
            "risk_score": pred["risk_score"],
            "status": pred["status"],
            "gradient_attribution": pred["gradient_attribution"],
            "local_gradient_attribution": pred["local_gradient_attribution"],
            "attention_weights": pred["attention_weights"],
            "explanation_text": pred["explanation"],
            "explanation_method": pred["explanation_method"],
        }

    def get_trajectory(self, stay_id: int) -> List[dict]:
        self._require_stay(stay_id)
        return [
            {
                "window_idx": p["window_idx"],
                "window_start_time": p["window_start_time"],
                "window_end_time": p["window_end_time"],
                "risk_score": p["risk_score"],
                "status": p["status"],
                "z_2d": p["z_2d"],
                "ground_truth_label": p["ground_truth_label"],
                "observed_vitals": p["observed_vitals"],
            }
            for p in self._predictions_by_stay[stay_id]
        ]

    def get_twin_state(self, stay_id: int, window_idx: Optional[int] = None) -> dict:
        """
        Real digital-twin state for one window, explicitly split into
        OBSERVED / PREDICTED / SIMULATED (pre-BioGears redesign — see
        digital_twin.py's module docstring for what each category means
        and why they're never blurred together).
        """
        pred = self.get_prediction(stay_id, window_idx)
        categories = self.digital_twin.describe_state_categories()
        return {
            "stay_id": stay_id,
            "window_idx": pred["window_idx"],
            "window_end_time": pred["window_end_time"],
            "observed": {
                "vitals": pred["observed_vitals"],
                "description": categories["observed"],
            },
            "predicted": {
                "risk_score": pred["risk_score"],
                "status": pred["status"],
                "latent_state": {
                    "z_t": pred["z_t"],
                    "z_2d": pred["z_2d"],
                    "latent_dim": self.digital_twin.latent_dim,
                },
                "description": categories["predicted"],
            },
            "simulated": {
                "available": False,
                "description": categories["simulated"],
            },
            "disclaimer": (
                "This digital twin currently shows OBSERVED (real measured vitals) "
                "and PREDICTED (this model's real risk score and latent "
                "representation) state only. It is not a physiological simulation. "
                "A real ML COUNTERFACTUAL (the same model re-evaluated on a "
                "hypothetically modified input) is available on demand via the "
                "What-If endpoint (POST /api/patients/{stay_id}/whatif) — it is "
                "not shown here since it applies to a scenario, not this "
                "window's actual observed state. SIMULATED state (forward "
                "PHYSIOLOGY under a hypothetical intervention, from BioGears) "
                "requires the BioGears integration and is not yet available — "
                "see docs/BIOGEARS_INTEGRATION_BOUNDARY.md."
            ),
        }


_service: Optional[InferenceService] = None


def get_inference_service(model_type: str = "lstm") -> InferenceService:
    """Process-wide singleton — build once, reuse across requests."""
    global _service
    if _service is None:
        _service = InferenceService(model_type=model_type)
    return _service
