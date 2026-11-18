"""V4 bounded local matrix retrieval. No ETL, statistics fitting, or market training.

x is training-RMS normalized [B,20,5], C0 is the saved daily 3x3 prior.
Every output is a normalized five-segment second-moment matrix.
"""
from __future__ import annotations
import hashlib
import math
from typing import Literal
import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_NAMES = ("Static", "Content", "MLP")
SEEDS = (2026110541, 2026110542, 2026110543)
RHO = 0.1
FEATURE_EPSILON = 1e-8
TIME_LAMBDA = 0.94
UPPER_TRIANGLE = tuple((i, j) for i in range(5) for j in range(i, 5))


def subseed(seed: int, stream: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{stream}".encode()).digest()[:8], "little") % (2**31 - 1)


def _validate_x(x: Tensor):
    if x.ndim != 3 or tuple(x.shape[1:]) != (20, 5) or x.shape[0] < 1:
        raise ValueError("Expected nonempty [batch,20,5] normalized input")
    if x.dtype != torch.float64 or x.device.type != "cpu":
        raise ValueError("V4 requires float64 CPU input")
    if not torch.isfinite(x).all():
        raise ValueError("Input contains nonfinite values")


def key_query_features(x: Tensor) -> tuple[Tensor, Tensor]:
    """16 features per known day; latest-five query. No extra rolling history."""
    _validate_x(x)
    energy = x.square().sum(-1)
    shape = torch.stack([x[..., i] * x[..., j] for i, j in UPPER_TRIANGLE], dim=-1)
    shape = shape / (energy.unsqueeze(-1) + FEATURE_EPSILON)
    keys = torch.cat((shape, torch.log1p(energy).unsqueeze(-1)), dim=-1)
    if not torch.isfinite(keys).all():
        raise ValueError("Nonfinite derived key features")
    query = keys[:, -5:, :].mean(dim=1)
    return keys, query


def time_log_prior(*, dtype=torch.float64) -> Tensor:
    exponents = torch.arange(19, -1, -1, dtype=dtype)
    weights = (1 - TIME_LAMBDA) * TIME_LAMBDA**exponents / (1 - TIME_LAMBDA**20)
    return torch.log(weights)


def covariance_from_weights(x: Tensor, c0, weights: Tensor) -> Tensor:
    _validate_x(x)
    c0 = torch.as_tensor(c0, dtype=x.dtype, device=x.device)
    if c0.shape != (3, 3) or c0.requires_grad or not torch.isfinite(c0).all():
        raise ValueError("C0 must be one finite fixed 3x3 daily prior")
    if not torch.allclose(c0, c0.T, atol=1e-12, rtol=1e-12):
        raise ValueError("C0 must be symmetric")
    if int(torch.linalg.cholesky_ex(c0).info) != 0:
        raise ValueError("C0 must be strictly positive definite")
    if weights.shape != x.shape[:2] or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Expected finite nonnegative [batch,20] weights")
    if not torch.allclose(weights.sum(-1), torch.ones(x.shape[0], dtype=x.dtype), atol=1e-12, rtol=1e-12):
        raise ValueError("Weights must sum to one")
    values = x[:, :, 2:5]  # Fixed [5Y,7Y,10Y], including signed cross products.
    weighted = values.transpose(1, 2) @ (values * weights.unsqueeze(-1))
    if not torch.isfinite(weighted).all():
        raise FloatingPointError("Nonfinite weighted Value matrix")
    result = 5.0 * ((1.0 - RHO) * weighted + RHO * c0)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite output matrix")
    with torch.no_grad():
        if (torch.linalg.cholesky_ex(result.detach()).info != 0).any():
            raise FloatingPointError("Output is not numerically positive definite")
    return result


def kernel_bandwidth_statistics(training_x: Tensor) -> dict:
    """One geometry estimate, no labels/optimizer. Cost O(N*20*16), storage N*20.

    h² = median strictly positive squared distances between each training query
    and its 20 training keys. All-zero distances -> h²=1. No bandwidth search.
    torch.quantile(.5) implements the usual midpoint median for even counts.
    """
    with torch.no_grad():
        keys, query = key_query_features(training_x)
        distance2 = (keys - query[:, None, :]).square().sum(-1)
        positive = distance2[distance2 > 0]
        bandwidth = float(torch.quantile(positive, 0.5)) if positive.numel() else 1.0
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Invalid fixed kernel bandwidth")
        return dict(
            bandwidth_squared=bandwidth,
            training_requests=int(training_x.shape[0]),
            query_key_pairs=int(distance2.numel()),
            positive_distance_pairs=int(positive.numel()),
            rule="median_positive_squared_distance_same_training_query_key_geometry",
            fallback_used=not bool(positive.numel()),
            labels_used=False, optimizer_fits=0,
        )


def fixed_matrix(x: Tensor, c0, name: str, bandwidth_squared: float | None = None, return_weights=False):
    """Uniform / EWMA / Kernel baselines with exactly the same Value and C0."""
    _validate_x(x)
    if name == "Uniform":
        weights = x.new_full(x.shape[:2], 1.0 / 20)
    elif name == "EWMA":
        weights = torch.softmax(time_log_prior(dtype=x.dtype), 0).expand(x.shape[0], -1)
    elif name == "Kernel":
        if bandwidth_squared is None or not math.isfinite(bandwidth_squared) or bandwidth_squared <= 0:
            raise ValueError("Kernel requires one positive saved training bandwidth_squared")
        keys, query = key_query_features(x)
        distance2 = (keys - query[:, None, :]).square().sum(-1)
        logits = time_log_prior(dtype=x.dtype)[None, :] - distance2 / (2.0 * bandwidth_squared)
        weights = torch.softmax(logits, dim=-1)
    else:
        raise ValueError(f"Unknown fixed matrix method: {name}")
    result = covariance_from_weights(x, c0, weights)
    return (result, weights) if return_weights else result


class MatrixRetrieval(nn.Module):
    def __init__(self, name: Literal["Static", "Content", "MLP"], seed: int):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown scorer: {name}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.name, self.seed = name, seed
        self.position_delta = nn.Parameter(torch.zeros(19, dtype=torch.float64))
        self.register_buffer("log_time_prior", time_log_prior())
        self.register_buffer("age", torch.arange(19, -1, -1, dtype=torch.float64) / 19.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(subseed(seed, "matrix_" + name.lower()))
            if name == "Content":
                self.query_projection = nn.Linear(16, 8, bias=True, dtype=torch.float64)
                self.key_projection = nn.Linear(16, 8, bias=False, dtype=torch.float64)
                nn.init.zeros_(self.query_projection.weight)
                nn.init.zeros_(self.query_projection.bias)
                nn.init.xavier_uniform_(self.key_projection.weight)
            elif name == "MLP":
                self.hidden = nn.Linear(33, 8, bias=True, dtype=torch.float64)
                self.output = nn.Linear(8, 1, bias=False, dtype=torch.float64)
                nn.init.zeros_(self.output.weight)

    def static_logits(self) -> Tensor:
        delta = torch.cat((self.position_delta, self.position_delta.new_zeros(1)))
        return self.log_time_prior + delta

    def score_logits(self, x: Tensor) -> Tensor:
        keys, query = key_query_features(x)
        base = self.static_logits()[None, :].expand(x.shape[0], -1)
        if self.name == "Static":
            return base
        if self.name == "Content":
            q = self.query_projection(query)
            k = self.key_projection(keys)
            return base + (q[:, None, :] * k).sum(-1) / math.sqrt(8.0)
        q = query[:, None, :].expand(-1, 20, -1)
        age = self.age[None, :, None].expand(x.shape[0], -1, -1)
        feature = torch.cat((q, keys, age), dim=-1)
        return base + self.output(F.gelu(self.hidden(feature))).squeeze(-1)

    def forward(self, x: Tensor, c0, force_uniform=False, force_static=False, return_weights=False):
        if force_uniform and force_static:
            raise ValueError("Choose at most one counterfactual")
        _validate_x(x)
        if force_uniform:
            weights = x.new_full(x.shape[:2], 1.0 / 20.0)
        elif force_static:
            weights = torch.softmax(self.static_logits(), dim=-1).expand(x.shape[0], -1)
        else:
            weights = torch.softmax(self.score_logits(x), dim=-1)
        h = covariance_from_weights(x, c0, weights)
        return (h, weights) if return_weights else h

    def parameter_summary(self) -> dict:
        return dict(total=sum(p.numel() for p in self.parameters()), static_position=19,
                    structurally_inactive=0, name=self.name)

    def initialization_sha256(self) -> str:
        h = hashlib.sha256()
        for name, tensor in self.state_dict().items():
            h.update(name.encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def build_model(name: str, seed: int) -> MatrixRetrieval:
    return MatrixRetrieval(name, seed)
