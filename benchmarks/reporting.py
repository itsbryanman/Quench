"""Stable JSON/CSV reporting and regression comparison for Quench benchmarks."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
CSV_FIELDS = [
    "benchmark_name",
    "tensor_name",
    "tensor_type",
    "tensor_shape",
    "dtype",
    "config",
    "raw_bytes",
    "compressed_bytes",
    "compression_ratio",
    "mse",
    "max_abs_error",
    "relative_error",
    "encode_throughput_bytes_per_sec",
    "decode_throughput_bytes_per_sec",
    "backend_name",
]


@dataclass(frozen=True)
class BenchmarkResult:
    """Machine-readable benchmark result schema used for JSON and CSV artifacts."""

    benchmark_name: str
    tensor_name: str
    tensor_type: str
    tensor_shape: str
    dtype: str
    config: str
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    mse: float
    max_abs_error: float
    relative_error: float
    encode_throughput_bytes_per_sec: float
    decode_throughput_bytes_per_sec: float
    backend_name: str


@dataclass(frozen=True)
class RegressionThresholds:
    """Configurable thresholds for CI regression checks."""

    max_compression_ratio_drop: float = 0.05
    max_error_increase: float = 0.10
    max_throughput_drop: float = 0.10


@dataclass(frozen=True)
class RegressionReport:
    """Outcome of comparing two benchmark artifacts."""

    passed: bool
    messages: tuple[str, ...]


def write_json_report(
    results: Iterable[BenchmarkResult],
    path: str | Path,
    *,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    """Write benchmark results as a stable JSON artifact."""
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_metadata": run_metadata or {},
        "schema_version": SCHEMA_VERSION,
        "results": [asdict(result) for result in results],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv_report(results: Iterable[BenchmarkResult], path: str | Path) -> None:
    """Write benchmark results as a stable CSV artifact."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def load_json_report(path: str | Path) -> list[BenchmarkResult]:
    """Load benchmark results from a JSON artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark schema version: {payload['schema_version']}")
    return [BenchmarkResult(**item) for item in payload["results"]]


def compare_reports(
    baseline: Iterable[BenchmarkResult],
    current: Iterable[BenchmarkResult],
    *,
    thresholds: RegressionThresholds | None = None,
) -> RegressionReport:
    """Compare two benchmark result sets and report threshold violations."""
    active_thresholds = thresholds or RegressionThresholds()
    baseline_map = {result.benchmark_name: result for result in baseline}
    current_map = {result.benchmark_name: result for result in current}
    failures: list[str] = []

    for name, current_result in sorted(current_map.items()):
        baseline_result = baseline_map.get(name)
        if baseline_result is None:
            failures.append(f"{name}: missing baseline result")
            continue

        min_ratio = baseline_result.compression_ratio * (1.0 - active_thresholds.max_compression_ratio_drop)
        if current_result.compression_ratio < min_ratio:
            failures.append(
                f"{name}: compression ratio regressed from {baseline_result.compression_ratio:.4f} "
                f"to {current_result.compression_ratio:.4f}"
            )

        max_mse = baseline_result.mse * (1.0 + active_thresholds.max_error_increase)
        if current_result.mse > max_mse:
            failures.append(f"{name}: MSE regressed from {baseline_result.mse:.6g} to {current_result.mse:.6g}")

        max_abs = baseline_result.max_abs_error * (1.0 + active_thresholds.max_error_increase)
        if current_result.max_abs_error > max_abs:
            failures.append(
                f"{name}: max abs error regressed from {baseline_result.max_abs_error:.6g} "
                f"to {current_result.max_abs_error:.6g}"
            )

        min_encode = baseline_result.encode_throughput_bytes_per_sec * (1.0 - active_thresholds.max_throughput_drop)
        if current_result.encode_throughput_bytes_per_sec < min_encode:
            failures.append(
                f"{name}: encode throughput regressed from "
                f"{baseline_result.encode_throughput_bytes_per_sec:.2f} to "
                f"{current_result.encode_throughput_bytes_per_sec:.2f}"
            )

        min_decode = baseline_result.decode_throughput_bytes_per_sec * (1.0 - active_thresholds.max_throughput_drop)
        if current_result.decode_throughput_bytes_per_sec < min_decode:
            failures.append(
                f"{name}: decode throughput regressed from "
                f"{baseline_result.decode_throughput_bytes_per_sec:.2f} to "
                f"{current_result.decode_throughput_bytes_per_sec:.2f}"
            )

    return RegressionReport(passed=not failures, messages=tuple(failures))
