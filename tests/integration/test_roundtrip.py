"""Integration tests for representative tensor round-trips."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

import quench
from quench.backends import native_backend_available
from quench.codec import QuenchDecoder, QuenchEncoder
from quench.core.config import QuenchConfig


def _compress_zstd(data: bytes) -> int | None:
    """Return the zstd-compressed size when a baseline implementation is available."""
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


def _representative_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2025)

    weights = rng.normal(loc=0.0, scale=0.14, size=(256, 128)).astype(np.float32)
    weights *= rng.lognormal(mean=-0.4, sigma=0.3, size=(256, 1)).astype(np.float32)
    weights[rng.random(weights.shape) < 0.10] = 0.0

    kv_steps = rng.normal(loc=0.0, scale=0.02, size=(2, 8, 128, 16)).astype(np.float32)
    kv_cache = np.cumsum(kv_steps, axis=2, dtype=np.float32)

    embeddings = rng.normal(loc=0.0, scale=0.09, size=(512, 64)).astype(np.float32)
    embeddings[:128] = 0.0
    embeddings[rng.random(embeddings.shape) < 0.18] = 0.0

    activations = rng.normal(loc=0.0, scale=0.85, size=(8, 64, 64)).astype(np.float32)
    activations = np.maximum(activations, 0.0)
    activations[activations < 0.3] = 0.0

    return {
        "attn.q_proj.weight": weights,
        "layer_0.k_cache": kv_cache,
        "token_embed.weight": embeddings,
        "block.activation": activations,
    }


@pytest.mark.parametrize(
    ("name", "mae_limit", "max_limit"),
    [
        ("attn.q_proj.weight", 0.025, 0.09),
        ("layer_0.k_cache", 0.020, 0.09),
        ("token_embed.weight", 0.030, 0.12),
        ("block.activation", 0.050, 0.18),
    ],
)
def test_representative_roundtrip(
    name: str,
    mae_limit: float,
    max_limit: float,
) -> None:
    tensors = _representative_tensors()
    tensor = tensors[name]

    compressed = quench.compress(tensor, name=name)
    restored = quench.decompress(compressed)

    error = np.abs(restored.astype(np.float32) - tensor.astype(np.float32))
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert float(np.mean(error)) <= mae_limit
    assert float(np.max(error)) <= max_limit
    assert compressed.compressed_nbytes < tensor.nbytes

    zstd_size = _compress_zstd(np.ascontiguousarray(tensor).tobytes())
    if zstd_size is not None:
        assert compressed.compressed_nbytes <= int(zstd_size * 1.15)


def test_total_compression_is_better_than_raw_and_prints_summary() -> None:
    tensors = _representative_tensors()
    rows: list[str] = []
    total_raw = 0
    total_compressed = 0
    total_zstd = 0
    have_zstd = True

    for name, tensor in tensors.items():
        compressed = quench.compress(tensor, name=name)
        restored = quench.decompress(compressed)
        np.testing.assert_equal(restored.shape, tensor.shape)

        raw_size = tensor.nbytes
        compressed_size = compressed.compressed_nbytes
        zstd_size = _compress_zstd(np.ascontiguousarray(tensor).tobytes())
        if zstd_size is None:
            have_zstd = False
            zstd_display = "n/a"
        else:
            total_zstd += zstd_size
            zstd_display = str(zstd_size)

        rows.append(
            f"{name[:24]:24} {raw_size:10d} {compressed_size:12d} {compressed_size / raw_size:8.3f} {zstd_display:>10}"
        )
        total_raw += raw_size
        total_compressed += compressed_size

    print(
        f"{'tensor':24} {'raw':>10} {'compressed':>12} {'ratio':>8} {'zstd':>10}\n"
        + "\n".join(rows)
    )

    assert total_compressed < total_raw * 0.75
    if have_zstd:
        assert total_compressed <= int(total_zstd * 1.15)


@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel", "blockwise"])
def test_roundtrip_all_granularities(granularity: str) -> None:
    config = QuenchConfig(
        target_bits=4,
        quantization_granularity=granularity,
        block_size=16,
    )
    rng = np.random.default_rng(999)
    tensor = rng.normal(size=(64, 32)).astype(np.float32)
    encoder = QuenchEncoder(config=config)
    decoder = QuenchDecoder(config=config)

    compressed = encoder.encode(tensor, name="test.weight")
    restored = decoder.decode(compressed)

    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    error = np.abs(restored - tensor)
    assert float(np.mean(error)) < 0.1


@pytest.mark.skipif(not native_backend_available(), reason="Rust backend not built")
@pytest.mark.parametrize(
    ("name", "mae_limit", "max_limit"),
    [
        ("attn.q_proj.weight", 0.025, 0.09),
        ("layer_0.k_cache", 0.020, 0.09),
        ("token_embed.weight", 0.030, 0.12),
        ("block.activation", 0.050, 0.18),
    ],
)
def test_representative_roundtrip_with_rust_entropy_backend(
    name: str,
    mae_limit: float,
    max_limit: float,
) -> None:
    tensors = _representative_tensors()
    tensor = tensors[name]
    config = QuenchConfig(entropy_backend="rust")
    encoder = QuenchEncoder(config=config)
    decoder = QuenchDecoder(config=config)

    compressed = encoder.encode(tensor, name=name)
    restored = decoder.decode(compressed)

    error = np.abs(restored.astype(np.float32) - tensor.astype(np.float32))
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert float(np.mean(error)) <= mae_limit
    assert float(np.max(error)) <= max_limit
    assert compressed.compressed_nbytes < tensor.nbytes
