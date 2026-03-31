"""Unit tests for the Phase 3 codec pipeline."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import quench
from quench.codec import QuenchDecoder, QuenchEncoder
from quench.core import (
    MalformedPayloadError,
    MetadataError,
    QuenchConfig,
    UnsupportedStrategyError,
)
from quench.core.types import TensorHeader, TensorType


def _weight_tensor() -> np.ndarray:
    rng = np.random.default_rng(11)
    base = rng.normal(loc=0.0, scale=0.18, size=(96, 64)).astype(np.float32)
    scales = rng.lognormal(mean=-0.4, sigma=0.25, size=(96, 1)).astype(np.float32)
    return base * scales


def _kv_tensor() -> np.ndarray:
    rng = np.random.default_rng(12)
    steps = rng.normal(loc=0.0, scale=0.025, size=(2, 8, 96, 16)).astype(np.float32)
    return np.cumsum(steps, axis=2, dtype=np.float32)


def _embedding_tensor() -> np.ndarray:
    rng = np.random.default_rng(13)
    tensor = rng.normal(loc=0.0, scale=0.1, size=(256, 48)).astype(np.float32)
    tensor[:64] = 0.0
    tensor[rng.random(tensor.shape) < 0.15] = 0.0
    return tensor


def _activation_tensor() -> np.ndarray:
    rng = np.random.default_rng(14)
    tensor = rng.normal(loc=0.0, scale=0.8, size=(4, 32, 64)).astype(np.float32)
    tensor = np.maximum(tensor, 0.0)
    tensor[tensor < 0.25] = 0.0
    return tensor


def _assert_bounded_error(
    restored: np.ndarray,
    original: np.ndarray,
    *,
    mae_limit: float,
    max_limit: float,
) -> None:
    error = np.abs(restored.astype(np.float32) - original.astype(np.float32))
    assert float(np.mean(error)) <= mae_limit
    assert float(np.max(error)) <= max_limit


def test_weight_roundtrip_end_to_end() -> None:
    tensor = _weight_tensor()

    compressed = quench.compress(tensor, name="mlp.weight")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.WEIGHT
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    _assert_bounded_error(restored, tensor, mae_limit=0.02, max_limit=0.08)


def test_kv_roundtrip_end_to_end() -> None:
    tensor = _kv_tensor()

    compressed = quench.compress(tensor, name="layer_0.k_cache")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.KV_CACHE
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    _assert_bounded_error(restored, tensor, mae_limit=0.02, max_limit=0.08)


def test_embedding_roundtrip_end_to_end() -> None:
    tensor = _embedding_tensor()

    compressed = quench.compress(tensor, name="token_embed.weight")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.EMBEDDING
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    _assert_bounded_error(restored, tensor, mae_limit=0.03, max_limit=0.10)


def test_activation_roundtrip_end_to_end() -> None:
    tensor = _activation_tensor()

    compressed = quench.compress(tensor, name="block.activation")
    restored = quench.decompress(compressed)

    assert compressed.header.tensor_type == TensorType.ACTIVATION
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    _assert_bounded_error(restored, tensor, mae_limit=0.04, max_limit=0.15)


def test_auto_detection_picks_expected_tensor_types() -> None:
    assert quench.compress(_weight_tensor(), name="attn.q_proj.weight").header.tensor_type == TensorType.WEIGHT
    assert quench.compress(_kv_tensor(), name="block_0.k_cache").header.tensor_type == TensorType.KV_CACHE
    assert quench.compress(_embedding_tensor(), name="token_embed.weight").header.tensor_type == TensorType.EMBEDDING
    assert quench.compress(_activation_tensor(), name="activation").header.tensor_type == TensorType.ACTIVATION


def test_mask_roundtrip_causal() -> None:
    mask = np.tril(np.ones((1024, 1024), dtype=np.float32))
    compressed = quench.compress(mask, tensor_type=TensorType.MASK)
    restored = quench.decompress(compressed)
    np.testing.assert_array_equal(mask, restored)
    assert compressed.compressed_nbytes < mask.nbytes


def test_mask_roundtrip_neg_inf() -> None:
    mask = np.where(np.tril(np.ones((512, 512))), 0.0, float('-inf')).astype(np.float32)
    compressed = quench.compress(mask, tensor_type=TensorType.MASK)
    restored = quench.decompress(compressed)
    np.testing.assert_array_equal(mask, restored)


def test_constant_tensor_roundtrip() -> None:
    const = np.ones((256, 256), dtype=np.float32)
    compressed = quench.compress(const, tensor_type=TensorType.MASK)
    restored = quench.decompress(compressed)
    np.testing.assert_array_equal(const, restored)
    # Constant tensor payload should be tiny (metadata + header overhead is separate)
    assert len(compressed.payload) < 100


def test_layernorm_weight_is_lossless() -> None:
    """LayerNorm weights should round-trip exactly."""
    values = np.ones(768, dtype=np.float32) + np.random.randn(768).astype(np.float32) * 0.001
    compressed = quench.compress(values, name="model.layers.0.input_layernorm.weight")
    restored = quench.decompress(compressed)
    np.testing.assert_array_equal(values, restored)


def test_tiny_tensor_is_lossless() -> None:
    """Tensors under 1KB should always be lossless regardless of config."""
    values = np.random.randn(64).astype(np.float32)  # 256 bytes
    compressed = quench.compress(values, name="some.small.param")
    restored = quench.decompress(compressed)
    np.testing.assert_array_equal(values, restored)


def test_decode_rejects_malformed_metadata() -> None:
    tensor = _weight_tensor()
    compressed = quench.compress(tensor, name="mlp.weight")
    broken = replace(compressed, metadata=b"{broken json")

    with pytest.raises(MetadataError):
        quench.decompress(broken)


def test_decode_rejects_strategy_mismatch() -> None:
    tensor = _weight_tensor()
    compressed = quench.compress(tensor, name="mlp.weight")
    bad_header = TensorHeader(
        tensor_type=compressed.header.tensor_type,
        dtype=compressed.header.dtype,
        shape=compressed.header.shape,
        codec_mode=compressed.header.codec_mode,
        magic=compressed.header.magic,
        version=compressed.header.version,
        strategy_id=2,
        checksum=compressed.header.checksum,
    )
    broken = replace(compressed, header=bad_header)

    with pytest.raises(UnsupportedStrategyError):
        quench.decompress(broken)


def test_decode_rejects_truncated_payload() -> None:
    tensor = _activation_tensor()
    compressed = quench.compress(tensor, name="activation")
    broken = replace(compressed, payload=compressed.payload[:-3])

    with pytest.raises(MalformedPayloadError):
        quench.decompress(broken)


def test_encode_dict_decode_dict_roundtrip() -> None:
    config = QuenchConfig(target_bits=4)
    encoder = QuenchEncoder(config=config)
    decoder = QuenchDecoder(config=config)
    tensors = {
        "token_embed.weight": _embedding_tensor(),
        "layer_0.k_cache": _kv_tensor(),
        "mlp.weight": _weight_tensor(),
    }

    compressed = encoder.encode_dict(tensors)
    restored = decoder.decode_dict(compressed)

    assert list(compressed) == sorted(tensors)
    assert list(restored) == sorted(tensors)
    for name, original in tensors.items():
        recovered = restored[name]
        assert recovered.shape == original.shape
        assert recovered.dtype == original.dtype
        _assert_bounded_error(recovered, original, mae_limit=0.05, max_limit=0.16)
