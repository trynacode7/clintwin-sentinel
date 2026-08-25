"""
digital_twin.py — Digital Twin Core: Latent State Management
==============================================================
Pre-BioGears redesign (this phase): the Digital Twin's state is now
explicitly three kinds of thing, never blurred together:

  OBSERVED  — real measured vitals for this patient/window. Ground
              truth data, not produced by this module at all (it comes
              from backend/preprocessing.py's real synthetic chart
              events). This module never fabricates observed data.

  PREDICTED — this module's actual output: z_t, the latent patient
              state extracted from the trained model's hidden
              representation when run on OBSERVED data, plus its 2D
              PCA projection (z_2d) for visualization, plus the risk
              score that came out of the same forward pass. This is a
              real, deterministic function of a real trained model —
              never randomly generated — but it is a *prediction*
              (the model's internal representation and risk estimate),
              not a physiological measurement or a mechanistic
              simulation of organ systems.

  COUNTERFACTUAL — what this SAME trained model predicts (risk score
              + latent state) if one or more inputs are hypothetically
              changed and the real pipeline is re-run on the modified
              input. A real second forward pass of this model — never
              an edited number, an interpolation, or a physiological
              simulation. Computed on demand by
              InferenceService.compute_counterfactual() (see
              backend/whatif.py); NOT part of the persistent twin-state
              response (GET /api/patients/{id}/twin-state never returns
              a "counterfactual" section — that only appears in a
              What-If response for a scenario the user actually asked
              for).

  SIMULATED — forward PHYSIOLOGICAL simulation of a hypothetical
              intervention (e.g. "what if we started norepinephrine"),
              produced by a mechanistic physiology engine (BioGears) —
              NOT this trained ML model. NOT YET AVAILABLE. This module
              has no simulated state and does not fabricate one — see
              backend/whatif.py and docs/BIOGEARS_INTEGRATION_BOUNDARY.md
              for the integration boundary where real BioGears output
              will eventually plug in. Never confuse this with
              COUNTERFACTUAL above — one is a physics-based simulation
              of the body, the other is the same statistical model
              re-evaluated on different input.

z_t Explained:
  z_t is a dense vector (e.g., 256-dim for BiLSTM, 128-dim for
  Transformer) that encodes the patient's current state as learned by
  the neural network from real observed vitals. It is a *learned
  representation*, not a physiological state variable — there is no
  claim that any dimension of z_t corresponds to a specific organ
  system or clinical quantity. Treat it as "what the model currently
  believes about this patient, in the model's own internal language,"
  useful for similarity comparison and trajectory visualization, not
  as a mechanistic model of the patient's physiology.

This module:
  - Extracts real z_t trajectories from real model forward passes
    (PREDICTED state — see above)
  - Projects z_t to 2D (PCA) for visualization
  - Tracks trajectories over time
  - Finds similar patient states for comparison

This module intentionally does NOT:
  - Generate synthetic/fabricated trajectories (a prior version had a
    `generate_demo_trajectories()` method that produced random-walk
    fake patient states for demo purposes; it was dead code — never
    called from any live API path — and has been removed as part of
    this redesign, since a "digital twin" module that can silently
    fabricate physiological-looking trajectories is exactly the kind
    of thing this project's honesty requirements rule out)
  - Simulate physiology going forward in time under an intervention
    (that is SIMULATED state, and belongs to the future BioGears
    integration, not this module)
"""

import os
import sys
import numpy as np
import torch
import logging
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("icu_digital_twin")


class DigitalTwin:
    """
    Manages the digital twin state (z_t) for patient monitoring.
    
    Stores z_t trajectories, performs dimensionality reduction for
    visualization, and enables patient similarity comparison.
    """
    
    def __init__(self, latent_dim: int = config.LSTM_HIDDEN_SIZE * 2):
        """
        Args:
            latent_dim: Dimensionality of z_t vectors
        """
        self.latent_dim = latent_dim
        
        # Storage: patient_id → list of (timestamp_idx, z_t, risk_score)
        self.trajectories: Dict[str, List[Tuple[int, np.ndarray, float]]] = {}
        
        # PCA model for 2D projection
        self.pca = PCA(n_components=2)
        self._pca_fitted = False
        
        # All z_t vectors for PCA fitting
        self._all_zt: List[np.ndarray] = []
    
    @torch.no_grad()
    def extract_trajectory(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        mask: torch.Tensor,
        patient_id: str,
        model_type: str = "lstm",
    ) -> Dict:
        """
        Extract z_t trajectory for a patient's time series.
        
        Processes the input through the model and stores the sequence
        of latent states, forming the patient's trajectory through
        the latent space.
        
        Args:
            model: Trained model
            X: Input time series [seq_len, n_features] or [1, seq_len, n_features]
            mask: Padding mask
            patient_id: Unique patient identifier
            model_type: "lstm" or "transformer"
            
        Returns:
            Dictionary with z_t, risk_score, and 2D projection
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Ensure batch dimension
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        X = X.to(device)
        mask = mask.to(device)
        
        # Forward pass
        if model_type == "transformer":
            risk, z_t, attn = model(X, mask)
        else:
            risk, z_t = model(X, mask)
            attn = None
        
        z_t_np = z_t.squeeze(0).cpu().numpy()
        risk_score = risk.squeeze().cpu().item()
        
        # Store in trajectory
        if patient_id not in self.trajectories:
            self.trajectories[patient_id] = []
        
        t_idx = len(self.trajectories[patient_id])
        self.trajectories[patient_id].append((t_idx, z_t_np, risk_score))
        self._all_zt.append(z_t_np)
        
        # Project to 2D
        z_2d = self.project_to_2d(z_t_np)
        
        return {
            "z_t": z_t_np.tolist(),
            "risk_score": risk_score,
            "z_2d": z_2d.tolist() if z_2d is not None else [0.0, 0.0],
            "attention_weights": [a.cpu().numpy().tolist() for a in attn] if attn else None,
            "t_idx": t_idx,
        }
    
    def project_to_2d(
        self,
        z_t: np.ndarray,
        method: str = "pca",
    ) -> Optional[np.ndarray]:
        """
        Project z_t from high-dimensional latent space to 2D for visualization.
        
        Uses PCA (fast, deterministic) for real-time visualization.
        t-SNE could be used for batch visualization but is too slow for real-time.
        
        Args:
            z_t: Latent state vector [latent_dim]
            method: Projection method ("pca")
            
        Returns:
            2D coordinates [2] or None if PCA not fitted
        """
        if not self._pca_fitted:
            if len(self._all_zt) >= 10:
                self._fit_pca()
            else:
                return np.zeros(2)
        
        if self._pca_fitted:
            return self.pca.transform(z_t.reshape(1, -1)).squeeze()
        
        return np.zeros(2)
    
    def _fit_pca(self):
        """Fit PCA on all collected z_t vectors."""
        if len(self._all_zt) < 2:
            return
        
        all_zt_array = np.array(self._all_zt)
        n_components = min(2, all_zt_array.shape[0], all_zt_array.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(all_zt_array)
        self._pca_fitted = True
        
        logger.info(
            f"PCA fitted on {len(self._all_zt)} z_t vectors. "
            f"Explained variance: {self.pca.explained_variance_ratio_}"
        )
    
    def get_trajectory(self, patient_id: str) -> List[Dict]:
        """
        Get the full z_t trajectory for a patient.
        
        Returns:
            List of {t_idx, z_2d, risk_score} dicts
        """
        if patient_id not in self.trajectories:
            return []
        
        trajectory = []
        for t_idx, z_t, risk_score in self.trajectories[patient_id]:
            z_2d = self.project_to_2d(z_t)
            trajectory.append({
                "t_idx": t_idx,
                "z_2d": z_2d.tolist() if z_2d is not None else [0.0, 0.0],
                "risk_score": risk_score,
            })
        
        return trajectory
    
    def get_all_trajectories_2d(self) -> Dict[str, List[Dict]]:
        """Get 2D projections for all tracked patients."""
        result = {}
        for patient_id in self.trajectories:
            result[patient_id] = self.get_trajectory(patient_id)
        return result
    
    def find_similar_patients(
        self,
        z_t: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find patients with the most similar latent states.
        
        Uses cosine similarity in the latent space to identify
        patients whose physiological states are most similar.
        
        Args:
            z_t: Query latent state
            top_k: Number of similar patients to return
            
        Returns:
            List of {patient_id, similarity, risk_score} dicts
        """
        similarities = []
        
        for patient_id, trajectory in self.trajectories.items():
            if not trajectory:
                continue
            
            # Use the most recent z_t for each patient
            _, latest_zt, risk_score = trajectory[-1]
            
            # Cosine similarity
            cos_sim = np.dot(z_t, latest_zt) / (
                np.linalg.norm(z_t) * np.linalg.norm(latest_zt) + 1e-8
            )
            
            similarities.append({
                "patient_id": patient_id,
                "similarity": float(cos_sim),
                "risk_score": float(risk_score),
            })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def warm_pca(self, z_t_vectors: List[np.ndarray]):
        """
        Pre-fit PCA on a batch of REAL z_t vectors collected up front.

        Phase B addition. Without this, `project_to_2d` fits PCA lazily on
        whatever the first 10 z_t vectors happen to be, in whatever order
        `extract_trajectory()` is called — so the first ~10 processed
        patients get a degenerate `[0.0, 0.0]` projection for their entire
        trajectory (this was true even before Phase B; it just had no real
        data flowing through it to expose the issue). inference_service.py
        now runs a real forward pass across the whole roster up front
        specifically so it can hand this method a representative batch of
        real z_t vectors before any trajectory point is actually recorded.
        This does not change what the PCA fit means — it is still fit once
        and not continuously refit (see class docstring / Phase E) — it
        only fixes WHICH real vectors it is fit on.
        """
        if not z_t_vectors:
            return
        self._all_zt.extend(z_t_vectors)
        self._fit_pca()

    def describe_state_categories(self) -> Dict[str, str]:
        """
        Static, honest description of what this twin can and cannot show
        right now — OBSERVED / PREDICTED / COUNTERFACTUAL / SIMULATED.
        Used by inference_service.get_twin_state() so the distinction
        lives in one place rather than being restated ad hoc at each call
        site. COUNTERFACTUAL is not part of the persistent twin-state
        response (GET /api/patients/{id}/twin-state only ever shows
        OBSERVED/PREDICTED/SIMULATED for the real, actually-occurred
        window) — it appears only in an on-demand What-If response (see
        backend/whatif.py), described here so the full state taxonomy is
        documented in one place.
        """
        return {
            "observed": (
                "Real measured vitals for this patient/window, from the "
                "synthetic test-split ICU stay. Not produced by this class."
            ),
            "predicted": (
                "z_t (this trained model's latent representation), its 2D "
                "PCA projection z_2d, and the risk score — all from one "
                "real forward pass of the trained model on the OBSERVED "
                "vitals above. A learned representation, not a "
                "mechanistic physiological state."
            ),
            "counterfactual": (
                "What the SAME trained model predicts (risk score + latent "
                "state) if one or more inputs are hypothetically changed and "
                "the real feature-engineering + inference pipeline is re-run "
                "on that modified input. A real second forward pass of this "
                "model — never an edited number, an interpolation, or a "
                "physiological simulation. See backend/whatif.py and "
                "inference_service.compute_counterfactual()."
            ),
            "simulated": (
                "Forward PHYSIOLOGICAL simulation under a hypothetical "
                "intervention, produced by a mechanistic physiology engine "
                "(BioGears) — not this trained ML model. Not yet available "
                "— requires the BioGears integration (see "
                "docs/BIOGEARS_INTEGRATION_BOUNDARY.md)."
            ),
        }
