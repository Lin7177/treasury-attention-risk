"""Small, explicit variable-token ablations for the frozen boundary study.

Input x is [batch, historical changes, tenor], already scaled with training-only
statistics. A shared temporal encoder maps each tenor history to one token.
Tenor identities are learned, so changing channel order changes the financial
meaning. No parameters or preprocessing are borrowed from reference projects.
"""
from __future__ import annotations

import math
import torch
from torch import nn

MODEL_NAMES = ('B2_fixed', 'T1_attention', 'B3_fixed_state', 'T2_condition',
               'T3_post_state', 'B4_uniform')


class ContentMixer(nn.Module):
    """Content attention, or a uniform-weight diagnostic with the same tensors.

    Uniform mode computes Q and K but multiplies their logits by zero. Thus its
    Q/K weight and bias rows receive identically zero gradients. Equal total
    parameter counts do NOT mean equal active capacity; see parameter_counts.
    """
    def __init__(self, dim: int = 16, heads: int = 2, uniform: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.dim, self.heads, self.uniform = dim, heads, uniform
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = q @ k.transpose(-2, -1) / math.sqrt(dim // self.heads)
        weights = torch.softmax(logits * 0.0 if self.uniform else logits, dim=-1)
        merged = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.output(merged)


class BoundaryModel(nn.Module):
    def __init__(self, name: str, window: int = 20, channels: int = 5):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(f'Unknown neural model {name!r}; choose {MODEL_NAMES}')
        if window <= 0 or channels <= 0:
            raise ValueError('window and channels must be positive')
        self.name, self.window, self.channels = name, window, channels
        self.dim = 16
        self.fixed = name in ('B2_fixed', 'B3_fixed_state')
        self.uses_state = name in ('B3_fixed_state', 'T2_condition', 'T3_post_state')
        self.conditioned = name == 'T2_condition'
        self.encoder = nn.Sequential(nn.Linear(window, self.dim), nn.GELU())
        self.tenor_identity = nn.Parameter(torch.empty(channels, self.dim))
        nn.init.normal_(self.tenor_identity, mean=0., std=.02)
        # Fixed and adaptive branches share the per-token residual FFN, norm,
        # temporal encoder, identity information, and zero dropout convention.
        self.norm1 = nn.LayerNorm(self.dim) if not self.fixed else nn.Identity()
        self.mixer = None if self.fixed else ContentMixer(self.dim, 2, name == 'B4_uniform')
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(nn.Linear(self.dim, 2*self.dim), nn.GELU(),
                                 nn.Linear(2*self.dim, self.dim))
        self.fusion = nn.Sequential(nn.Linear(channels*self.dim, self.dim), nn.GELU()) if self.fixed else None
        self.readout = nn.Linear(self.dim + (3 if self.uses_state else 0), 1)
        # Created last and initialized to the identity modulation. With the same
        # seed, T2 and T3 have identical backbone/readout initialization; only T2
        # can subsequently learn state-dependent token modulation before Q/K.
        self.film = nn.Linear(3, 2*self.dim) if self.conditioned else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.window, self.channels):
            raise ValueError(f'x must have shape [B,{self.window},{self.channels}]')
        return self.encoder(x.transpose(1, 2)) + self.tenor_identity.unsqueeze(0)

    def mix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.mixer is not None:
            tokens = tokens + self.mixer(self.norm1(tokens))
        return tokens + self.ffn(self.norm2(tokens))

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.encode(x)
        if self.uses_state:
            if state is None or state.shape != (x.shape[0], 3):
                raise ValueError('state must have shape [B,3] for this model')
            if self.film is not None:
                gamma, beta = self.film(state).chunk(2, dim=-1)
                tokens = tokens * (1. + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        tokens = self.mix_tokens(tokens)
        representation = (self.fusion(tokens.flatten(1)) if self.fixed else tokens.mean(dim=1))
        if self.uses_state:
            representation = torch.cat([representation, state], dim=-1)
        return self.readout(representation).squeeze(-1)


def build_model(name: str, window: int = 20, channels: int = 5) -> nn.Module:
    """Return a new model with scalar raw logits; caller owns RNG and scaling."""
    return BoundaryModel(name, window, channels)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dead_qk = 0
    if isinstance(model, BoundaryModel) and model.name == 'B4_uniform':
        dead_qk = 2*model.dim*model.dim + 2*model.dim
    return {'total': total, 'trainable_declared': trainable,
            'structurally_zero_gradient_qk': dead_qk,
            'potentially_active_excluding_uniform_qk': trainable-dead_qk}
