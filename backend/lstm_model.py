"""
lstm_model.py — Bidirectional LSTM for ICU Risk Prediction
===========================================================
Architecture:
  Input → 2-layer Bidirectional LSTM → Mean Pooling → FC → Sigmoid

Digital Twin State (z_t):
  z_t is the mean-pooled hidden state across all timesteps.
  This 256-dim vector (128 × 2 directions) encodes the patient's
  current physiological state as learned by the model. It serves as
  the "digital twin" — a compressed, learned representation of the
  patient's trajectory that can be:
    - Tracked over time to visualize patient trajectories
    - Projected to 2D for clustering and visualization  
    - Compared across patients to find similar cases

Masking:
  The mask tensor indicates which timesteps have valid data (1) vs.
  padding (0). Padded timesteps are zeroed out before mean pooling
  so they don't contaminate the latent state.

Returns:
  risk: Sigmoid probability of deterioration [batch]
  z_t:  Latent patient state [batch, hidden_size * 2]
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ICUBiLSTM(nn.Module):
    """
    Bidirectional LSTM for ICU clinical deterioration prediction.
    
    The model processes variable-length time-series of patient vitals/labs
    and outputs a risk score along with a latent state vector (z_t).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
        bidirectional: bool = config.LSTM_BIDIRECTIONAL,
    ):
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout between LSTM layers
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection with LayerNorm for stability
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Latent dimension: hidden_size * num_directions
        latent_dim = hidden_size * self.num_directions
        
        # Risk prediction head: z_t → risk score
        self.risk_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            mask: Padding mask [batch, seq_len], 1=valid, 0=padding
            
        Returns:
            risk: Deterioration probability [batch, 1]
            z_t:  Digital twin latent state [batch, latent_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_proj(x)  # [batch, seq_len, hidden_size]
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_size * num_directions]
        
        # Compute z_t via masked mean pooling
        # z_t represents the patient's latent physiological state
        if mask is not None:
            # Expand mask for broadcasting: [batch, seq_len, 1]
            mask_expanded = mask.unsqueeze(-1)
            
            # Zero out padded timesteps
            masked_out = lstm_out * mask_expanded
            
            # Mean pool over valid timesteps only
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
            z_t = masked_out.sum(dim=1) / lengths  # [batch, latent_dim]
        else:
            # No mask — simple mean over all timesteps
            z_t = lstm_out.mean(dim=1)  # [batch, latent_dim]
        
        # Predict risk from latent state
        risk = torch.sigmoid(self.risk_head(z_t))  # [batch, 1]
        
        return risk, z_t
    
    def get_latent_dim(self) -> int:
        """Return the dimensionality of z_t."""
        return self.hidden_size * self.num_directions


if __name__ == "__main__":
    # Quick test
    batch_size = 8
    seq_len = config.INPUT_WINDOW
    input_size = 50  # Approximate number of features
    
    model = ICUBiLSTM(input_size=input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    mask = torch.ones(batch_size, seq_len)
    
    risk, z_t = model(x, mask)
    print(f"Risk shape: {risk.shape}")  # [8, 1]
    print(f"z_t shape:  {z_t.shape}")   # [8, 256]
    print(f"Latent dim: {model.get_latent_dim()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
