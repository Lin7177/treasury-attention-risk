"""V3 time-aggregation models. No data loading, scaling fit, or training.

Input is already training-standardized [batch, 20, raw5+sq5].
Outputs are normalized Q5 predictions; the runner supplies the fold mean Q5.
"""
from __future__ import annotations

import hashlib
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_NAMES = ("Fixed", "Uniform", "Content")
SEEDS = (202611051, 202611052, 202611053)
NORMALIZED_FLOOR = 1e-8
SHARED_COMPONENTS = ("encoder", "position", "norm1", "norm2", "ffn1", "ffn2", "head")


def subseed(seed: int, stream: str) -> int:
    """Fixed protocol SHA256 child stream; independent of global torch RNG."""
    digest = hashlib.sha256((str(seed) + "|" + stream).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


class TimeAggregationModel(nn.Module):
    """Latest-time query over 20 known historical time tokens.

    forward defaults to u [batch]. With return_raw, returns (u, raw_softplus).
    With return_attention, returns (u, weights), weights [B, 2, 1, 20],
    or None for Fixed. With both, returns (u, raw_softplus, weights).
    raw_softplus <= NORMALIZED_FLOOR must be rejected by the runner.
    """

    def __init__(self, name: Literal["Fixed", "Uniform", "Content"], seed: int):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.name = name
        self.seed = seed
        # fork_rng restores the caller's CPU RNG, including when construction fails.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(10, 8)
            self.position = nn.Parameter(torch.empty(20, 8))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            self.norm1 = nn.LayerNorm(8, eps=1e-5)
            self.norm2 = nn.LayerNorm(8, eps=1e-5)
            self.ffn1 = nn.Linear(8, 16)
            self.ffn2 = nn.Linear(16, 8)
            self.head = nn.Linear(8, 1)
        if name == "Fixed":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(subseed(seed, "time_fixed"))
                self.fixed1 = nn.Linear(160, 2)
                self.fixed2 = nn.Linear(2, 8)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(subseed(seed, "time_attention"))
                self.attention = nn.MultiheadAttention(
                    embed_dim=8, num_heads=2, dropout=0.0, bias=True, batch_first=True
                )

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (20, 10) or x.shape[0] < 1:
            raise ValueError("Expected nonempty input [batch, 20, 10]")
        if not torch.isfinite(x).all():
            raise ValueError("Model input must be finite")
        return F.gelu(self.encoder(x)) + self.position

    def aggregate(self, h: Tensor, force_uniform: bool = False):
        a = self.norm1(h)
        if self.name == "Fixed":
            context = self.fixed2(F.gelu(self.fixed1(a.flatten(start_dim=1))))
            return context, None
        weights, biases = self.attention.in_proj_weight.chunk(3), self.attention.in_proj_bias.chunk(3)
        # Q/K are deliberately not evaluated for Uniform: no task dependence.
        value = F.linear(a, weights[2], biases[2])
        batch = a.shape[0]
        value = value.reshape(batch, 20, 2, 4).transpose(1, 2)
        if self.name == "Uniform" or force_uniform:
            attn_weights = a.new_full((batch, 2, 1, 20), 1.0 / 20.0)
        else:
            query = F.linear(a[:, -1:, :], weights[0], biases[0])
            key = F.linear(a, weights[1], biases[1])
            query = query.reshape(batch, 1, 2, 4).transpose(1, 2)
            key = key.reshape(batch, 20, 2, 4).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(4)
            attn_weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attn_weights, value)
        mixed = mixed.transpose(1, 2).reshape(batch, 8)
        context = self.attention.out_proj(mixed)
        return context, attn_weights

    def forward(
        self,
        x: Tensor,
        force_uniform: bool = False,
        return_attention: bool = False,
        return_raw: bool = False,
    ):
        h = self.encode(x)
        context, weights = self.aggregate(h, force_uniform=force_uniform)
        recent_residual = h[:, -1, :] + context
        v = recent_residual + self.ffn2(F.gelu(self.ffn1(self.norm2(recent_residual))))
        raw = F.softplus(self.head(v).squeeze(-1))
        u = raw + NORMALIZED_FLOOR
        if return_raw and return_attention:
            return u, raw, weights
        if return_raw:
            return u, raw
        if return_attention:
            return u, weights
        return u

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "shared": 569,
            "inactive_qk": 144 if self.name == "Uniform" else 0,
        }

    def initialization_hashes(self) -> dict[str, str]:
        """Call before training; raw bytes include all parameters, including Q/K."""
        groups: dict[str, list[tuple[str, Tensor]]] = {"shared": [], "mixer": []}
        for key, tensor in self.state_dict().items():
            group = "shared" if key.split(".")[0] in SHARED_COMPONENTS else "mixer"
            groups[group].append((key, tensor))
        hashes = {}
        for group, entries in groups.items():
            digest = hashlib.sha256()
            for key, tensor in entries:
                digest.update(key.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            hashes[group] = digest.hexdigest()
        return hashes


def build_model(name: str, seed: int) -> TimeAggregationModel:
    return TimeAggregationModel(name, seed)
