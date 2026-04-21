"""
time_embedding.py
─────────────────
Learnable time embeddings for TemporalProteoTranslator.

Two embeddings:
  TimeEmbedding      — encodes current timepoint: log(t + 1)
  DeltaTimeEmbedding — encodes prediction horizon: log(Δt + 1)

Both produce a (batch, dim) conditioning vector that is passed to
the FiLMConditioner in temporal_proteotranslator.py.  FiLM then
applies a learned scale+shift to EVERY gene token before Performer
attention, so the time signal reaches every position directly.

Previously this file produced (batch, 1, dim) tokens for prepending.
The change to (batch, dim) removes the need for sequence-length
bookkeeping and makes the conditioning more direct.

Biological rationale:
  - TimeEmbedding tells the model WHERE we are in the response
  - DeltaTimeEmbedding tells the model HOW FAR AHEAD to predict
  - Together: RNA(t) + t + Δt → Protein(t + Δt)
"""

import math
import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """
    Fixed sinusoidal encoding for a scalar input.
    Maps x → dim-vector using sin/cos at multiple frequencies.
    Serves as a stable initialisation before the learned MLP.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal encoding"
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(0, dim, 2, dtype=torch.float32) / dim
        )
        self.register_buffer("freqs", freqs)  # (dim/2,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch,)
        Returns:
            (batch, dim)
        """
        x = x.unsqueeze(-1)                           # (batch, 1)
        args = x * self.freqs.unsqueeze(0)            # (batch, dim/2)
        return torch.cat([torch.sin(args),
                          torch.cos(args)], dim=-1)   # (batch, dim)


class TimeEmbedding(nn.Module):
    """
    Embeds the CURRENT timepoint into a (batch, dim) conditioning vector.

    Architecture:
        log(t+1) scalar
            → SinusoidalEncoding (dim)
            → Linear(dim, dim) → GELU → Linear(dim, dim) → LayerNorm
            → (batch, dim)

    The output is consumed by FiLMConditioner, which uses it to compute
    per-feature scale and shift applied to every RNA token.

    Input:  log_t  (batch,)   log1p-transformed minutes
    Output: emb    (batch, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalEncoding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, log_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_t: (batch,)
        Returns:
            (batch, dim)
        """
        x = self.sinusoidal(log_t)
        return self.mlp(x)             # (batch, dim)


class DeltaTimeEmbedding(nn.Module):
    """
    Embeds the PREDICTION HORIZON Δt into a (batch, dim) conditioning vector.

    Identical architecture to TimeEmbedding but with separate weights.
    When Δt = 0, predicts at the current timepoint (same-tp task).
    When Δt > 0, shifts the prediction into the future.

    Input:  log_delta_t  (batch,)   log1p(Δt in minutes), 0 for same-tp
    Output: emb          (batch, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalEncoding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, log_delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_delta_t: (batch,)
        Returns:
            (batch, dim)
        """
        x = self.sinusoidal(log_delta_t)
        return self.mlp(x)             # (batch, dim)
