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


def test_mask_strategy_binary_causal_mask() -> None:
    """Binary causal mask should compress much better than RLE."""
    mask = np.tril(np.ones((256, 256), dtype=np.float32))

    compressed = quench.compress(mask, name="h.0.attn.bias")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.MASK
    np.testing.assert_array_equal(restored.astype(np.float32), mask)
    assert compressed.compressed_nbytes < 2000, (
        f"Mask compressed to {compressed.compressed_nbytes} bytes, expected <2000"
    )


def test_mask_strategy_neg_inf_causal_mask() -> None:
    """Mask using 0/-inf values should also compress well."""
    mask = np.where(
        np.tril(np.ones((128, 128), dtype=np.float32)),
        0.0,
        float("-inf"),
    ).astype(np.float32)

    compressed = quench.compress(mask, name="attn_mask")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.MASK
    np.testing.assert_array_equal(restored.astype(np.float32), mask)
    assert compressed.compressed_nbytes < 1500


def test_mask_strategy_constant() -> None:
    """Constant masks should still use the constant path."""
    mask = np.ones((64, 64), dtype=np.float32)
    compressed = quench.compress(mask, name="mask.all_ones")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.MASK
    np.testing.assert_array_equal(restored, mask)
    assert compressed.compressed_nbytes < 300


def test_bias_small_goes_lossless() -> None:
    """Biases <= 4096 bytes should be lossless."""
    rng = np.random.default_rng(100)
    tensor = rng.normal(scale=0.02, size=(384,)).astype(np.float32)

    compressed = quench.compress(tensor, name="layer.bias")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.BIAS
    np.testing.assert_array_equal(restored, tensor)


def test_bias_large_uses_6bit() -> None:
    """Biases > 4096 bytes should use higher bit depth for quality."""
    rng = np.random.default_rng(101)
    tensor = rng.normal(scale=0.1, size=(2048,)).astype(np.float32)

    compressed = quench.compress(tensor, name="big.bias")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.BIAS
    orig = tensor.astype(np.float64)
    rec = restored.astype(np.float64)
    cosine = float(np.dot(orig, rec)) / (
        float(np.linalg.norm(orig)) * float(np.linalg.norm(rec))
    )
    assert cosine > 0.98, f"Bias cosine similarity {cosine} too low, expected >0.98"


def test_mixed_precision_small_goes_lossless() -> None:
    """Small mixed-precision tensors should be lossless."""
    rng = np.random.default_rng(102)
    tensor = rng.normal(scale=0.1, size=(512,)).astype(np.float16)

    compressed = quench.compress(tensor, name="fp16.small")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.MIXED_PRECISION
    np.testing.assert_array_equal(restored, tensor)
