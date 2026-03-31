"""Tests for benchmark artifact reporting and regression checks."""
from __future__ import annotations

from pathlib import Path

from benchmarks.reporting import (
    BenchmarkResult,
    RegressionThresholds,
    build_aggregates,
    compare_reports,
    load_json_report,
    write_csv_report,
    write_json_report,
)


def _result(name: str, *, ratio: float = 2.0, mse: float = 0.01, throughput: float = 100.0) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_name=name,
        tensor_name=name,
        tensor_type="weight",
        tensor_shape="8x8",
        dtype="float32",
        config="{}",
        raw_bytes=256,
        compressed_bytes=128,
        compression_ratio=ratio,
        mse=mse,
        max_abs_error=mse,
        relative_error=mse,
        encode_throughput_bytes_per_sec=throughput,
        decode_throughput_bytes_per_sec=throughput,
        backend_name="python",
    )


def test_reporting_writes_json_and_csv(tmp_path: Path) -> None:
    results = [_result("tensor/weight")]
    json_path = tmp_path / "bench.json"
    csv_path = tmp_path / "bench.csv"

    write_json_report(
        results,
        json_path,
        summary={"synthetic": {"rows": 1}},
        aggregates={"by_run": {"rows": 1}},
    )
    write_csv_report(results, csv_path)

    assert json_path.exists()
    assert csv_path.exists()
    loaded = load_json_report(json_path)
    assert loaded == results
    payload = json_path.read_text(encoding="utf-8")
    assert '"summary"' in payload
    assert '"aggregates"' in payload


def test_compare_reports_fails_on_regression() -> None:
    baseline = [_result("tensor/weight", ratio=2.0, mse=0.01, throughput=100.0)]
    current = [_result("tensor/weight", ratio=1.7, mse=0.02, throughput=80.0)]

    report = compare_reports(
        baseline,
        current,
        thresholds=RegressionThresholds(
            max_compression_ratio_drop=0.05,
            max_error_increase=0.05,
            max_throughput_drop=0.05,
        ),
    )

    assert not report.passed
    assert report.messages


def test_build_aggregates_tracks_cosine_similarity() -> None:
    results = [
        BenchmarkResult(**{**_result("tensor/a").__dict__, "cosine_similarity": 0.99}),
        BenchmarkResult(**{**_result("tensor/b").__dict__, "cosine_similarity": 0.95}),
    ]

    aggregates = build_aggregates(results)

    assert aggregates["by_run"]["mean_cosine_similarity"] == 0.97
    assert aggregates["by_run"]["median_cosine_similarity"] == 0.97
    assert aggregates["by_run"]["p5_cosine_similarity"] == 0.952


def test_build_aggregates_prefers_container_bytes_when_present() -> None:
    result = BenchmarkResult(
        **{
            **_result("tensor/container").__dict__,
            "quench_container_bytes": 96,
            "quench_container_payload_bytes": 72,
            "quench_container_overhead_bytes": 24,
        }
    )

    aggregates = build_aggregates([result])

    assert aggregates["by_run"]["compressed_bytes_total"] == 96
    assert aggregates["by_run"]["aggregate_compression_ratio"] == (256 / 96)
    assert aggregates["by_run"]["mean_metadata_plus_header_bytes"] == 24
