"""Tests for Phase 4 strategy coverage."""
from __future__ import annotations

import numpy as np

import quench
from quench.core.types import TensorType


def test_optimizer_state_strategy_roundtrip() -> None:
    rng = np.random.default_rng(77)
    tensor = np.cumsum(rng.normal(scale=0.01, size=(64, 64)).astype(np.float32), axis=1)

    compressed = quench.compress(tensor, name="optimizer.exp_avg")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.OPTIMIZER_STATE
    error = np.abs(restored.astype(np.float32) - tensor.astype(np.float32))
    assert float(np.mean(error)) <= 0.03
    assert float(np.max(error)) <= 0.18


def test_bias_strategy_roundtrip() -> None:
    rng = np.random.default_rng(88)
    tensor = rng.normal(scale=0.02, size=(128,)).astype(np.float32)

    compressed = quench.compress(tensor, name="proj.bias")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.BIAS
    np.testing.assert_allclose(restored, tensor, atol=0.05)


def test_mixed_precision_strategy_preserves_dtype() -> None:
    rng = np.random.default_rng(99)
    tensor = rng.normal(scale=0.1, size=(64, 32)).astype(np.float16)

    compressed = quench.compress(tensor, name="mlp.fp16_gate")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.MIXED_PRECISION
    assert restored.dtype == tensor.dtype
    np.testing.assert_allclose(restored.astype(np.float32), tensor.astype(np.float32), atol=0.08)
