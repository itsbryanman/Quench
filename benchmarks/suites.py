"""Deterministic synthetic benchmark suites for Quench."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quench.core.config import QuenchConfig


@dataclass(frozen=True)
class TensorBenchmarkCase:
    """One single-tensor benchmark target."""

    benchmark_name: str
    tensor_name: str
    tensor_type: str
    tensor: np.ndarray[Any, np.dtype[Any]]
    config: QuenchConfig


@dataclass(frozen=True)
class BundleBenchmarkCase:
    """One streamed multi-tensor benchmark target."""

    benchmark_name: str
    tensor_type: str
    tensors: dict[str, np.ndarray[Any, np.dtype[Any]]]
    config: QuenchConfig


def build_tensor_suite(config: QuenchConfig | None = None, *, seed: int = 2025) -> list[TensorBenchmarkCase]:
    """Return deterministic tensor benchmark cases."""
    active_config = config or QuenchConfig()
    state_dict = build_transformer_bundle(seed=seed)
    return [
        TensorBenchmarkCase("tensor/weight", "attn.q_proj.weight", "weight", state_dict["attn.q_proj.weight"], active_config),
        TensorBenchmarkCase("tensor/kv_cache", "layer_0.k_cache", "kv_cache", state_dict["layer_0.k_cache"], active_config),
        TensorBenchmarkCase("tensor/embedding", "token_embed.weight", "embedding", state_dict["token_embed.weight"], active_config),
        TensorBenchmarkCase("tensor/optimizer_state", "optimizer.exp_avg", "optimizer_state", state_dict["optimizer.exp_avg"], active_config),
        TensorBenchmarkCase("tensor/bias", "attn.q_proj.bias", "bias", state_dict["attn.q_proj.bias"], active_config),
        TensorBenchmarkCase("tensor/mixed_precision", "mlp.fp16_gate", "mixed_precision", state_dict["mlp.fp16_gate"], active_config),
    ]


def build_bundle_suite(config: QuenchConfig | None = None, *, seed: int = 2025) -> list[BundleBenchmarkCase]:
    """Return deterministic streamed bundle benchmark cases."""
    active_config = config or QuenchConfig()
    bundle = build_transformer_bundle(seed=seed)
    return [
        BundleBenchmarkCase(
            benchmark_name="bundle/streamed_transformer",
            tensor_type="bundle",
            tensors=bundle,
            config=active_config,
        )
    ]


def build_transformer_bundle(*, seed: int = 2025) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Build a representative synthetic transformer-style bundle."""
    rng = np.random.default_rng(seed)

    token_embeddings = rng.normal(loc=0.0, scale=0.08, size=(512, 96)).astype(np.float32)
    token_embeddings[:128] = 0.0
    token_embeddings[rng.random(token_embeddings.shape) < 0.12] = 0.0

    q_proj = rng.normal(loc=0.0, scale=0.12, size=(96, 96)).astype(np.float32)
    q_proj *= rng.lognormal(mean=-0.4, sigma=0.25, size=(96, 1)).astype(np.float32)

    k_proj = rng.normal(loc=0.0, scale=0.11, size=(96, 96)).astype(np.float32)
    k_proj *= rng.lognormal(mean=-0.3, sigma=0.28, size=(96, 1)).astype(np.float32)

    kv_steps = rng.normal(loc=0.0, scale=0.02, size=(2, 6, 96, 16)).astype(np.float32)
    kv_cache = np.cumsum(kv_steps, axis=2, dtype=np.float32)

    optimizer_exp_avg = rng.normal(loc=0.0, scale=0.01, size=(96, 96)).astype(np.float32)
    optimizer_exp_avg = np.cumsum(optimizer_exp_avg, axis=1, dtype=np.float32) * 0.1

    optimizer_exp_avg_sq = np.abs(rng.normal(loc=0.02, scale=0.01, size=(96, 96))).astype(np.float32)

    bias = rng.normal(loc=0.0, scale=0.04, size=(96,)).astype(np.float32)

    fp16_gate = rng.normal(loc=0.0, scale=0.09, size=(192, 96)).astype(np.float16)

    return {
        "attn.q_proj.weight": q_proj,
        "attn.k_proj.weight": k_proj,
        "attn.q_proj.bias": bias,
        "layer_0.k_cache": kv_cache,
        "mlp.fp16_gate": fp16_gate,
        "optimizer.exp_avg": optimizer_exp_avg,
        "optimizer.exp_avg_sq": optimizer_exp_avg_sq,
        "token_embed.weight": token_embeddings,
    }
