"""Unit tests for the Phase 3 codec pipeline."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

import quench
from quench.codec import QuenchDecoder, QuenchEncoder, deserialize_metadata
from quench.core import (
    MalformedPayloadError,
    MetadataError,
    QuenchConfig,
    UnsupportedStrategyError,
)
from quench.core.types import TensorHeader, TensorType
from quench.io.tiny_bundle import try_build_tiny_exact_bundle_entry


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


def _compress_zstd(data: bytes) -> int | None:
    """Return the zstd-compressed size when an implementation is available."""
    try:
        import zstandard
    except Exception:
        if shutil.which("zstd") is None:
            return None
        completed = subprocess.run(
            ["zstd", "-3", "-q", "-c"],
            input=data,
            capture_output=True,
            check=True,
        )
        return len(completed.stdout)
    return len(zstandard.ZstdCompressor(level=3).compress(data))


def _strategy_metadata(compressed: quench.CompressedTensor) -> dict[str, object]:
    """Load strategy metadata from either the legacy or compact metadata schema."""
    metadata = deserialize_metadata(compressed.metadata)
    strategy = metadata.get("strategy")
    if isinstance(strategy, dict):
        strategy_metadata = strategy.get("metadata")
        assert isinstance(strategy_metadata, dict)
        return strategy_metadata
    return metadata


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
    strategy_metadata = _strategy_metadata(compressed)
    np.testing.assert_array_equal(mask, restored)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "mtri"
    assert compressed.compressed_nbytes < mask.nbytes
    zstd_size = _compress_zstd(np.ascontiguousarray(mask).tobytes())
    if zstd_size is not None:
        assert compressed.compressed_nbytes < zstd_size


def test_mask_roundtrip_neg_inf() -> None:
    mask = np.where(np.tril(np.ones((512, 512))), 0.0, float('-inf')).astype(np.float32)
    compressed = quench.compress(mask, tensor_type=TensorType.MASK)
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)
    np.testing.assert_array_equal(mask, restored)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "mtri"
    zstd_size = _compress_zstd(np.ascontiguousarray(mask).tobytes())
    if zstd_size is not None:
        assert compressed.compressed_nbytes < zstd_size


def test_constant_tensor_roundtrip() -> None:
    const = np.ones((256, 256), dtype=np.float32)
    compressed = quench.compress(const, tensor_type=TensorType.MASK)
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)
    np.testing.assert_array_equal(const, restored)
    # Constant tensor payload should be tiny (metadata + header overhead is separate)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "mconst"
    assert len(compressed.payload) == 0
    assert compressed.compressed_nbytes < 128


def test_all_zero_mask_uses_compact_constant_path() -> None:
    const = np.zeros((128, 128), dtype=np.float32)
    compressed = quench.compress(const, tensor_type=TensorType.MASK)
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)

    np.testing.assert_array_equal(const, restored)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "mconst"
    assert len(compressed.payload) == 0


def test_nontrivial_binary_mask_uses_compact_fallback() -> None:
    row = (np.arange(64) % 3 == 0).astype(np.float32)
    mask = np.vstack([np.roll(row, shift) for shift in range(64)]).astype(np.float32)
    compressed = quench.compress(mask, tensor_type=TensorType.MASK)
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)

    np.testing.assert_array_equal(mask, restored)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) in {"mbits", "mrle"}
    assert compressed.compressed_nbytes < mask.nbytes


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


def test_small_bias_uses_compact_exact_raw_path() -> None:
    """Small exact biases should stop entropy-coding raw bytes."""
    rng = np.random.default_rng(303)
    values = rng.normal(scale=0.02, size=(384,)).astype(np.float32)

    compressed = quench.compress(values, name="proj.bias")
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)

    np.testing.assert_array_equal(values, restored)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "raw"
    assert len(compressed.payload) == values.nbytes
    zstd_size = _compress_zstd(np.ascontiguousarray(values).tobytes())
    if zstd_size is not None:
        assert compressed.compressed_nbytes <= int(zstd_size * 1.25)


def test_small_constant_vector_uses_compact_exact_constant_path() -> None:
    values = np.ones((384,), dtype=np.float32)

    compressed = quench.compress(values, name="layer.bias")
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)

    np.testing.assert_array_equal(values, restored)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "const"
    assert len(compressed.payload) == values.dtype.itemsize
    assert compressed.compressed_nbytes < 128


def test_small_layernorm_weight_uses_compact_exact_path() -> None:
    rng = np.random.default_rng(404)
    values = (1.0 + rng.normal(scale=0.001, size=(192,))).astype(np.float32)

    compressed = quench.compress(values, name="encoder.layer.0.output.LayerNorm.weight")
    restored = quench.decompress(compressed)

    np.testing.assert_array_equal(values, restored)
    assert len(compressed.payload) <= values.nbytes


def test_position_ids_use_structural_sequence_path() -> None:
    values = np.arange(512, dtype=np.int64).reshape(1, 512)

    compressed = quench.compress(values, name="embeddings.position_ids")
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)

    np.testing.assert_array_equal(values, restored)
    assert compressed.header.tensor_type == TensorType.MIXED_PRECISION
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "aseq"
    assert len(compressed.payload) == 0
    zstd_size = _compress_zstd(np.ascontiguousarray(values).tobytes())
    if zstd_size is not None:
        assert compressed.compressed_nbytes < zstd_size


def test_repeated_position_ids_use_broadcast_sequence_path() -> None:
    values = np.broadcast_to(np.arange(128, dtype=np.int64), (4, 128)).copy()

    compressed = quench.compress(values, name="position_ids")
    restored = quench.decompress(compressed)
    strategy_metadata = _strategy_metadata(compressed)

    np.testing.assert_array_equal(values, restored)
    assert strategy_metadata.get("k", strategy_metadata.get("path")) == "bseq"
    assert len(compressed.payload) == 0


def test_tiny_int_tensor_roundtrip_exact() -> None:
    values = np.array([7, -3, 5, 12], dtype=np.int64)

    compressed = quench.compress(values, name="small.ints")
    restored = quench.decompress(compressed)

    np.testing.assert_array_equal(values, restored)
    assert len(compressed.payload) <= values.nbytes


def test_tiny_exact_bundle_candidate_selection_is_exact_and_size_limited() -> None:
    sequence = quench.compress(np.arange(128, dtype=np.int64).reshape(1, 128), name="position_ids")
    large = quench.compress(np.arange(1024, dtype=np.int64).reshape(1, 1024), name="large.position_ids")

    sequence_candidate = try_build_tiny_exact_bundle_entry("position_ids", sequence)
    large_candidate = try_build_tiny_exact_bundle_entry("large.position_ids", large)

    assert sequence_candidate is not None
    assert sequence_candidate.kind == "aseq"
    assert large_candidate is None


def test_metadata_rejects_oversized_ndarray() -> None:
    """Crafted metadata with a huge shape must not trigger OOM."""
    import json
    import base64
    from quench.codec.metadata import deserialize_metadata
    from quench.core.exceptions import MetadataError

    # Shape that would require >256 MiB
    malicious = json.dumps({
        "__ndarray__": True,
        "dtype": "<f8",
        "shape": [1, 1073741824],  # 8 GiB at float64
        "data": base64.b64encode(b"").decode("ascii"),
    }).encode("utf-8")
    with pytest.raises(MetadataError, match="too large"):
        deserialize_metadata(malicious)


def test_metadata_rejects_overflowing_shape() -> None:
    """Shape dimensions that overflow int64 when multiplied must not crash."""
    import json
    import base64
    from quench.codec.metadata import deserialize_metadata
    from quench.core.exceptions import MetadataError

    malicious = json.dumps({
        "__ndarray__": True,
        "dtype": "<f4",
        "shape": [2**33, 2**33],  # product overflows int64
        "data": base64.b64encode(b"").decode("ascii"),
    }).encode("utf-8")
    with pytest.raises(MetadataError):
        deserialize_metadata(malicious)


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
