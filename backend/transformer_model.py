"""
transformer_model.py — Transformer Encoder for ICU Risk Prediction
====================================================================
Architecture:
  Input → Linear Embedding → LayerNorm → Positional Encoding
  → N Transformer Encoder Layers (Multi-Head Self-Attention)
  → Attention Pooling → FC → Sigmoid

Digital Twin State (z_t):
  z_t is computed via attention-weighted pooling over the encoder
  output. Unlike simple mean pooling, attention pooling learns to
  weight timesteps by importance — giving more weight to clinically
  critical moments (e.g., sudden MAP drop, lactate spike).

Masking:
  The padding mask is passed to the Transformer encoder so that
  padded positions (where mask=0) receive zero attention weights.
  This prevents the model from attending to dummy padding data.

Attention Weights:
  The model returns attention weights from all layers and heads.
  These are used for:
    - Explainability: which timesteps does the model focus on?
    - Visualization: attention heatmap on the dashboard
    - Clinical insight: temporal patterns in deterioration

Returns:
  risk: Sigmoid probability [batch, 1]
  z_t:  Latent patient state [batch, d_model]
  attention_weights: List of attention matrices per layer
"""

import math
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for time-series data.
    
    Unlike fixed sinusoidal encoding, learnable positions can adapt
    to the specific temporal patterns in ICU data (e.g., circadian
    rhythms, medication schedules, procedure timing).
    """
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: [batch, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling to create z_t from encoder output.
    
    Instead of simple mean pooling, learns a query vector that
    attends to all timesteps. This allows the model to focus on
    the most clinically relevant moments when constructing the
    patient's latent state.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True
        )
    
    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Encoder output [batch, seq_len, d_model]
            key_padding_mask: True for padded positions [batch, seq_len]
            
        Returns:
            z_t: Pooled representation [batch, d_model]
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        
        z_t, _ = self.attention(
            query, x, x, key_padding_mask=key_padding_mask
        )
        return z_t.squeeze(1)  # [batch, d_model]


class ICUTransformer(nn.Module):
    """
    Transformer Encoder for ICU clinical deterioration prediction.
    
    Uses multi-head self-attention to capture complex temporal
    dependencies in patient trajectories. Returns attention weights
    for explainability.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = config.TRANSFORMER_D_MODEL,
        nhead: int = config.TRANSFORMER_NHEAD,
        num_layers: int = config.TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = config.TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = config.TRANSFORMER_DROPOUT,
    ):
        """
        Args:
            input_size: Number of input features per timestep
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        
        # Linear embedding: project input features to d_model
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Learnable positional encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model)
        
        # Transformer encoder layers
        # We use individual layers (not nn.TransformerEncoder) to
        # extract per-layer attention weights
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-norm for training stability
            )
            for _ in range(num_layers)
        ])
        
        # Final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Attention pooling for z_t
        self.attention_pool = AttentionPooling(d_model)
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        # For extracting attention weights
        self._attention_weights = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        for layer in self.encoder_layers:
            layer.self_attn.register_forward_hook(self._attention_hook)
    
    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights from self-attention layers."""
        # output is (attn_output, attn_weights) when need_weights=True
        # But with batch_first TransformerEncoderLayer, we need a different approach
        pass  # We'll compute attention weights explicitly in forward()
    
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
            z_t:  Digital twin latent state [batch, d_model]
            attention_weights: List of [batch, nhead, seq_len, seq_len] per layer
        """
        # Create key_padding_mask for transformer (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # True where padded
        
        # Embed and add positional encoding
        x = self.embedding(x)      # [batch, seq_len, d_model]
        x = self.pos_encoder(x)    # [batch, seq_len, d_model]
        
        # Pass through encoder layers, collecting attention weights
        attention_weights = []
        for layer in self.encoder_layers:
            # Manually compute attention to extract weights
            # Pre-norm
            x_norm = layer.norm1(x)
            
            # Self-attention with explicit weight extraction
            attn_out, attn_w = layer.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # Get per-head weights
            )
            attention_weights.append(attn_w.detach())
            
            # Residual + dropout
            x = x + layer.dropout1(attn_out)
            
            # FFN with pre-norm
            x_norm2 = layer.norm2(x)
            x = x + layer.dropout2(layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x_norm2)))
            ))
        
        x = self.final_norm(x)
        
        # Attention-weighted pooling → z_t (digital twin state)
        z_t = self.attention_pool(x, key_padding_mask)  # [batch, d_model]
        
        # Risk prediction
        risk = torch.sigmoid(self.risk_head(z_t))  # [batch, 1]
        
        return risk, z_t, attention_weights
    
    def get_latent_dim(self) -> int:
        """Return the dimensionality of z_t."""
        return self.d_model


if __name__ == "__main__":
    # Quick test
    batch_size = 8
    seq_len = config.INPUT_WINDOW
    input_size = 50
    
    model = ICUTransformer(input_size=input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    mask = torch.ones(batch_size, seq_len)
    # Simulate some padding
    mask[:, -10:] = 0
    
    risk, z_t, attn = model(x, mask)
    print(f"Risk shape: {risk.shape}")           # [8, 1]
    print(f"z_t shape:  {z_t.shape}")             # [8, 128]
    print(f"Layers:     {len(attn)}")             # 4
    print(f"Attn shape: {attn[0].shape}")         # [8, 4, 72, 72]
    print(f"Latent dim: {model.get_latent_dim()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
