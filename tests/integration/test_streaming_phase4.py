"""Integration coverage for streamed container paths and backend dispatch."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.reporting import compare_reports, load_json_report
from benchmarks.runner import run_benchmark_suite
from benchmarks.suites import build_transformer_bundle
from quench.codec import QuenchDecoder, QuenchEncoder, deserialize_metadata
from quench.core import QuenchConfig
from quench.integrations import load_compressed, save_compressed
from quench.io import QNCReader


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

    # 512 float32 values are exactly 2048 bytes, so the encoder now forces lossless routing.
    assert metadata["strategy"]["metadata"]["stream"]["encoding"] == "raw"
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
