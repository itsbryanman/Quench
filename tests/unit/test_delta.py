"""Unit tests for quench.delta compression."""
from __future__ import annotations

import numpy as np

from quench.core.config import QuenchConfig
from quench.delta.analysis import analyze_delta
from quench.delta.manifest import DeltaManifest, deserialize_manifest, serialize_manifest
from quench.delta.strategy import (
    decode_delta,
    decode_lossless,
    decode_sign_scale,
    decode_zero,
    encode_delta,
    encode_lossless,
    encode_sign_scale,
    encode_zero,
)


def test_analyze_zero_delta() -> None:
    delta = np.zeros((128, 128), dtype=np.float32)
    profile = analyze_delta(delta, "test.weight")
    assert profile.recommended_path == "zero"
    assert profile.recommended_bits == 0
    assert profile.sparsity == 1.0


def test_analyze_tiny_delta_goes_lossless() -> None:
    rng = np.random.default_rng(42)
    delta = rng.normal(scale=0.01, size=(16, 16)).astype(np.float32)
    profile = analyze_delta(delta, "small.weight")
    assert profile.recommended_path == "lossless"


def test_analyze_sparse_delta() -> None:
    rng = np.random.default_rng(42)
    delta = np.zeros((1024,), dtype=np.float32)
    delta[:100] = rng.normal(scale=0.01, size=100).astype(np.float32)
    profile = analyze_delta(delta, "sparse.weight")
    assert profile.recommended_path == "sparse"
    assert profile.sparsity > 0.85


def test_analyze_dense_delta_goes_quantize() -> None:
    rng = np.random.default_rng(42)
    delta = rng.normal(scale=0.01, size=(1024,)).astype(np.float32)
    profile = analyze_delta(delta, "dense.weight")
    assert profile.recommended_path == "quantize"


def test_analyze_sign_scale_eligible() -> None:
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=(1024, 1024)).astype(np.float32)
    delta = signs * 0.005
    profile = analyze_delta(delta, "big.weight")
    assert profile.recommended_path == "sign_scale"


def test_manifest_roundtrip() -> None:
    manifest = DeltaManifest(
        base_model_id="test/model",
        shared_tensors=["w1", "w2"],
        added_tensors=["new_head"],
        removed_tensors=[],
    )
    arr = serialize_manifest(manifest)
    assert arr.dtype == np.uint8
    restored = deserialize_manifest(arr)
    assert restored.base_model_id == "test/model"
    assert restored.shared_tensors == ["w1", "w2"]
    assert restored.added_tensors == ["new_head"]


def test_zero_path_roundtrip() -> None:
    payload, _meta = encode_zero()
    assert payload == b""
    restored = decode_zero((64, 64), "<f4")
    np.testing.assert_array_equal(restored, np.zeros((64, 64), dtype=np.float32))


def test_lossless_path_roundtrip() -> None:
    rng = np.random.default_rng(42)
    delta = rng.normal(scale=0.01, size=(8, 8)).astype(np.float32)
    payload, meta = encode_lossless(delta)
    restored = decode_lossless(payload, meta)
    np.testing.assert_array_equal(restored, delta)


def test_sign_scale_roundtrip() -> None:
    config = QuenchConfig()
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=(256, 256)).astype(np.float32)
    delta = signs * 0.005
    payload, meta = encode_sign_scale(delta, config)
    restored = decode_sign_scale(payload, meta, config)
    cosine = _cosine_similarity(delta, restored)
    assert cosine > 0.95, f"sign_scale cosine {cosine} too low"
    assert len(payload) < delta.nbytes / 5


def test_quantize_path_roundtrip() -> None:
    config = QuenchConfig()
    rng = np.random.default_rng(42)
    delta = rng.normal(scale=0.01, size=(512, 512)).astype(np.float32)
    payload, meta = encode_delta(delta, "quantize", config, bits=4)
    restored = decode_delta(payload, meta, config)
    cosine = _cosine_similarity(delta, restored)
    assert cosine > 0.98, f"quantize cosine {cosine} too low"


def test_sparse_path_roundtrip() -> None:
    config = QuenchConfig()
    rng = np.random.default_rng(42)
    delta = np.zeros((2048,), dtype=np.float32)
    delta[:200] = rng.normal(scale=0.01, size=200).astype(np.float32)
    payload, meta = encode_delta(delta, "sparse", config, bits=4)
    restored = decode_delta(payload, meta, config)
    nz_mask = np.abs(delta) > 1e-7
    cosine = _cosine_similarity(delta[nz_mask], restored[nz_mask])
    assert cosine > 0.95


def test_dispatch_selects_correct_path() -> None:
    config = QuenchConfig()
    rng = np.random.default_rng(42)

    _payload, meta = encode_delta(np.zeros((64, 64), dtype=np.float32), "zero", config)
    assert meta["path"] == "zero"

    delta = rng.normal(scale=0.01, size=(8, 8)).astype(np.float32)
    _payload, meta = encode_delta(delta, "lossless", config)
    assert meta["path"] == "lossless"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.sqrt(np.dot(a_flat, a_flat)))
    norm_b = float(np.sqrt(np.dot(b_flat, b_flat)))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return dot / (norm_a * norm_b)
