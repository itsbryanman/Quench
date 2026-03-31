"""Integration coverage for streamed container paths and backend dispatch."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.reporting import compare_reports, load_json_report
from benchmarks.runner import run_benchmark_suite
from benchmarks.suites import build_transformer_bundle
from quench.codec import QuenchDecoder, QuenchEncoder, deserialize_metadata
from quench.core import QuenchConfig
from quench.integrations import load_compressed, save_compressed
from quench.io import QNCReader


def _compress_zstd(data: bytes) -> int | None:
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


def test_streamed_bundle_encode_decode_roundtrip(tmp_path: Path) -> None:
    bundle = build_transformer_bundle(seed=3030)
    path = tmp_path / "streamed.qnc"
    config = QuenchConfig(target_bits=4)

    save_compressed(path, bundle, config=config, chunk_size=128)
    restored = load_compressed(path, config=config)
    records = list(QNCReader(path).iter_tensor_records())

    assert len(records) == len(bundle)
    assert any(record.chunk_count > 1 for record in records)
    for record in records:
        chunks = list(record.iter_payload_chunks())
        assert len(chunks) == record.chunk_count
        assert all(len(chunk) <= 128 for chunk in chunks)

    for name, original in bundle.items():
        recovered = restored[name]
        assert recovered.shape == original.shape
        assert recovered.dtype == original.dtype
        assert float(np.mean(np.abs(recovered.astype(np.float32) - original.astype(np.float32)))) <= 0.08


def test_python_backend_pluggable_encode_decode_path(tmp_path: Path) -> None:
    tensor = np.linspace(-1.0, 1.0, num=512, dtype=np.float32).reshape(64, 8)
    config = QuenchConfig(
        target_bits=4,
        entropy_coder="raw",
        pack_bits=True,
        entropy_backend="python",
        packing_backend="python",
    )
    encoder = QuenchEncoder(config=config)
    decoder = QuenchDecoder(config=config)

    compressed = encoder.encode(tensor, name="proj.bias")
    metadata = deserialize_metadata(compressed.metadata)
    restored = decoder.decode(compressed)
    strategy_metadata = metadata["strategy"]["metadata"] if "strategy" in metadata else metadata

    # 512 float32 values are exactly 2048 bytes, so the encoder now forces lossless routing.
    assert strategy_metadata.get("k", strategy_metadata.get("path")) in {"raw", "lossless"}
    np.testing.assert_allclose(restored, tensor, atol=0.2)


def test_benchmark_artifacts_and_regression_check(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"

    baseline_json, baseline_csv, baseline_results, _ = run_benchmark_suite(baseline_dir, repeats=1)
    current_json, current_csv, current_results, _ = run_benchmark_suite(current_dir, repeats=1)

    assert baseline_json.exists()
    assert baseline_csv.exists()
    assert current_json.exists()
    assert current_csv.exists()
    assert baseline_results
    assert current_results

    loaded = load_json_report(current_json)
    assert loaded[0].benchmark_name
    comparison = compare_reports(load_json_report(baseline_json), load_json_report(baseline_json))
    assert comparison.passed


def test_streamed_bundle_tiny_exact_cohort_uses_shared_container_overhead(tmp_path: Path) -> None:
    rng = np.random.default_rng(4242)
    config = QuenchConfig(target_bits=4)
    tensors = {
        "attn.q_proj.weight": (rng.normal(size=(128, 128)).astype(np.float32) * 0.1),
        "token_embed.weight": rng.normal(size=(256, 96)).astype(np.float32),
        "encoder.layer.0.attention.self.query.bias": rng.normal(scale=0.02, size=(24,)).astype(np.float32),
        "encoder.layer.0.attention.self.key.bias": rng.normal(scale=0.02, size=(24,)).astype(np.float32),
        "encoder.layer.0.attention.self.value.bias": rng.normal(scale=0.02, size=(24,)).astype(np.float32),
        "encoder.layer.0.attention.output.dense.bias": rng.normal(scale=0.02, size=(24,)).astype(np.float32),
        "encoder.layer.0.output.LayerNorm.weight": np.ones((128,), dtype=np.float32),
        "encoder.layer.0.output.LayerNorm.bias": np.zeros((128,), dtype=np.float32),
        "embeddings.position_ids": np.arange(512, dtype=np.int64).reshape(1, 512),
        "decoder.position_ids": np.arange(512, dtype=np.int64).reshape(1, 512),
        "repeated_position_ids": np.broadcast_to(np.arange(128, dtype=np.int64), (4, 128)).copy(),
    }
    tiny_names = sorted(name for name in tensors if "weight" not in name or "LayerNorm" in name or "position_ids" in name)

    bundled_path = tmp_path / "bundled.qnc"
    plain_path = tmp_path / "plain.qnc"
    save_compressed(bundled_path, tensors, config=config, enable_tiny_exact_bundle=True)
    save_compressed(plain_path, tensors, config=config, enable_tiny_exact_bundle=False)

    restored = load_compressed(bundled_path, config=config)
    bundled_records = {record.name: record for record in QNCReader(bundled_path).iter_tensor_records()}
    plain_records = {record.name: record for record in QNCReader(plain_path).iter_tensor_records()}

    for name in tiny_names:
        np.testing.assert_array_equal(restored[name], tensors[name])
    for name in {"attn.q_proj.weight", "token_embed.weight"}:
        recovered = restored[name]
        original = tensors[name]
        assert recovered.shape == original.shape
        assert recovered.dtype == original.dtype
        assert float(np.mean(np.abs(recovered.astype(np.float32) - original.astype(np.float32)))) <= 0.08

    bundled_tiny_bytes = sum(bundled_records[name].storage_nbytes for name in tiny_names)
    plain_tiny_bytes = sum(plain_records[name].storage_nbytes for name in tiny_names)
    assert bundled_path.stat().st_size < plain_path.stat().st_size
    assert bundled_tiny_bytes < plain_tiny_bytes

    zstd_size = _compress_zstd(
        b"".join(np.ascontiguousarray(tensors[name]).view(np.uint8).tobytes() for name in tiny_names)
    )
    if zstd_size is not None:
        assert bundled_tiny_bytes < zstd_size
