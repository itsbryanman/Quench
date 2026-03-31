"""Stable JSON/CSV reporting and regression comparison for Quench benchmarks."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 3
CSV_FIELDS = [
    "benchmark_name",
    "model_id",
    "model_revision",
    "model_local_path",
    "source_file",
    "source_file_sha256",
    "benchmark_mode",
    "sample_policy",
    "was_sampled",
    "tensor_role",
    "tensor_name",
    "tensor_type",
    "tensor_shape",
    "dtype",
    "config",
    "raw_bytes",
    "compressed_bytes",
    "zstd_raw_bytes",
    "zstd_quantized_bytes",
    "quantized_bytes",
    "quench_payload_bytes",
    "quench_metadata_bytes",
    "quench_header_bytes",
    "quench_container_bytes",
    "quench_container_payload_bytes",
    "quench_container_overhead_bytes",
    "quench_exact_kind",
    "quench_bundle_entries",
    "compression_ratio",
    "zstd_raw_ratio",
    "zstd_quantized_ratio",
    "mse",
    "max_abs_error",
    "relative_error",
    "cosine_similarity",
    "encode_throughput_bytes_per_sec",
    "decode_throughput_bytes_per_sec",
    "backend_name",
    "source_dtype",
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
    model_id: str = ""
    model_revision: str = ""
    model_local_path: str = ""
    source_file: str = ""
    source_file_sha256: str = ""
    benchmark_mode: str = ""
    sample_policy: str = ""
    was_sampled: bool = False
    tensor_role: str = ""
    zstd_raw_bytes: int | None = None
    zstd_quantized_bytes: int | None = None
    quantized_bytes: int | None = None
    quench_payload_bytes: int | None = None
    quench_metadata_bytes: int | None = None
    quench_header_bytes: int | None = None
    quench_container_bytes: int | None = None
    quench_container_payload_bytes: int | None = None
    quench_container_overhead_bytes: int | None = None
    quench_exact_kind: str | None = None
    quench_bundle_entries: int | None = None
    zstd_raw_ratio: float | None = None
    zstd_quantized_ratio: float | None = None
    cosine_similarity: float | None = None
    source_dtype: str | None = None


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
    summary: dict[str, Any] | None = None,
    aggregates: dict[str, Any] | None = None,
) -> None:
    """Write benchmark results as a stable JSON artifact."""
    rows = [asdict(result) for result in results]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_metadata": run_metadata or {},
        "schema_version": SCHEMA_VERSION,
        "summary": summary or {},
        "aggregates": aggregates or {},
        "results": rows,
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
    schema_version = int(payload["schema_version"])
    if schema_version not in {1, 2, 3}:
        raise ValueError(f"Unsupported benchmark schema version: {payload['schema_version']}")
    valid_fields = {f.name for f in __import__("dataclasses").fields(BenchmarkResult)}
    results = []
    for item in payload["results"]:
        filtered = {k: v for k, v in item.items() if k in valid_fields}
        results.append(BenchmarkResult(**filtered))
    return results


def build_aggregates(results: Sequence[BenchmarkResult]) -> dict[str, Any]:
    """Build stable aggregate summaries for benchmark artifacts."""
    rows = list(results)
    return {
        "by_model": _aggregate_groups(
            rows,
            key_fn=lambda item: (item.model_id or "synthetic", item.model_revision or ""),
            labels=("model_id", "model_revision"),
        ),
        "by_tensor_type": _aggregate_groups(
            rows,
            key_fn=lambda item: (item.tensor_type,),
            labels=("tensor_type",),
        ),
        "by_run": _aggregate_rows(rows),
    }


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


def _aggregate_groups(
    results: Sequence[BenchmarkResult],
    *,
    key_fn: Any,
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[BenchmarkResult]] = {}
    for result in results:
        key = tuple(str(part) for part in key_fn(result))
        groups.setdefault(key, []).append(result)

    aggregates: list[dict[str, Any]] = []
    for key in sorted(groups):
        aggregate = _aggregate_rows(groups[key])
        for label, value in zip(labels, key):
            aggregate[label] = value
        aggregates.append(aggregate)
    return aggregates


def _aggregate_rows(results: Sequence[BenchmarkResult]) -> dict[str, Any]:
    rows = list(results)
    raw_total = sum(item.raw_bytes for item in rows)
    compressed_total = sum(_effective_compressed_bytes(item) for item in rows)
    zstd_raw_values = [item.zstd_raw_bytes for item in rows if item.zstd_raw_bytes is not None]
    zstd_quantized_values = [item.zstd_quantized_bytes for item in rows if item.zstd_quantized_bytes is not None]
    cosine_values = [item.cosine_similarity for item in rows if item.cosine_similarity is not None]
    encoded = [item.encode_throughput_bytes_per_sec for item in rows]
    decoded = [item.decode_throughput_bytes_per_sec for item in rows]
    mse_values = [item.mse for item in rows]
    max_abs_values = [item.max_abs_error for item in rows]
    relative_values = [item.relative_error for item in rows]
    metadata_overheads = [_effective_overhead_bytes(item) for item in rows]
    savings_vs_zstd_raw = [
        item.zstd_raw_bytes - _effective_compressed_bytes(item)
        for item in rows
        if item.zstd_raw_bytes is not None
    ]
    savings_vs_zstd_quantized = [
        item.zstd_quantized_bytes - _effective_compressed_bytes(item)
        for item in rows
        if item.zstd_quantized_bytes is not None
    ]
    pct_savings_vs_zstd_raw = [
        (item.zstd_raw_bytes - _effective_compressed_bytes(item)) / item.zstd_raw_bytes
        for item in rows
        if item.zstd_raw_bytes not in (None, 0)
    ]
    pct_savings_vs_zstd_quantized = [
        (item.zstd_quantized_bytes - _effective_compressed_bytes(item)) / item.zstd_quantized_bytes
        for item in rows
        if item.zstd_quantized_bytes not in (None, 0)
    ]

    return {
        "rows": len(rows),
        "raw_bytes_total": raw_total,
        "compressed_bytes_total": compressed_total,
        "zstd_raw_bytes_total": sum(zstd_raw_values) if zstd_raw_values else None,
        "zstd_quantized_bytes_total": sum(zstd_quantized_values) if zstd_quantized_values else None,
        "aggregate_compression_ratio": (raw_total / compressed_total if compressed_total else 0.0),
        "aggregate_zstd_raw_ratio": (
            raw_total / sum(zstd_raw_values) if zstd_raw_values and sum(zstd_raw_values) else None
        ),
        "aggregate_zstd_quantized_ratio": (
            raw_total / sum(zstd_quantized_values)
            if zstd_quantized_values and sum(zstd_quantized_values)
            else None
        ),
        "mean_compression_ratio": mean(_effective_compression_ratio(item) for item in rows) if rows else 0.0,
        "median_compression_ratio": median(_effective_compression_ratio(item) for item in rows) if rows else 0.0,
        "mean_mse": mean(mse_values) if mse_values else 0.0,
        "median_mse": median(mse_values) if mse_values else 0.0,
        "max_abs_error_max": max(max_abs_values) if max_abs_values else 0.0,
        "relative_error_max": max(relative_values) if relative_values else 0.0,
        "mean_cosine_similarity": mean(cosine_values) if cosine_values else None,
        "median_cosine_similarity": median(cosine_values) if cosine_values else None,
        "p5_cosine_similarity": _percentile(cosine_values, 5.0) if cosine_values else None,
        "mean_encode_throughput_bytes_per_sec": mean(encoded) if encoded else 0.0,
        "median_encode_throughput_bytes_per_sec": median(encoded) if encoded else 0.0,
        "mean_decode_throughput_bytes_per_sec": mean(decoded) if decoded else 0.0,
        "median_decode_throughput_bytes_per_sec": median(decoded) if decoded else 0.0,
        "mean_metadata_plus_header_bytes": mean(metadata_overheads) if metadata_overheads else 0.0,
        "median_metadata_plus_header_bytes": median(metadata_overheads) if metadata_overheads else 0.0,
        "wins_vs_zstd_raw": sum(1 for item in savings_vs_zstd_raw if item > 0),
        "wins_vs_zstd_quantized": sum(1 for item in savings_vs_zstd_quantized if item > 0),
        "mean_bytes_saved_vs_zstd_raw": mean(savings_vs_zstd_raw) if savings_vs_zstd_raw else None,
        "median_bytes_saved_vs_zstd_raw": median(savings_vs_zstd_raw) if savings_vs_zstd_raw else None,
        "mean_bytes_saved_vs_zstd_quantized": (
            mean(savings_vs_zstd_quantized) if savings_vs_zstd_quantized else None
        ),
        "median_bytes_saved_vs_zstd_quantized": (
            median(savings_vs_zstd_quantized) if savings_vs_zstd_quantized else None
        ),
        "mean_pct_saved_vs_zstd_raw": mean(pct_savings_vs_zstd_raw) if pct_savings_vs_zstd_raw else None,
        "median_pct_saved_vs_zstd_raw": (
            median(pct_savings_vs_zstd_raw) if pct_savings_vs_zstd_raw else None
        ),
        "mean_pct_saved_vs_zstd_quantized": (
            mean(pct_savings_vs_zstd_quantized) if pct_savings_vs_zstd_quantized else None
        ),
        "median_pct_saved_vs_zstd_quantized": (
            median(pct_savings_vs_zstd_quantized) if pct_savings_vs_zstd_quantized else None
        ),
    }


def _effective_compressed_bytes(result: BenchmarkResult) -> int:
    return int(result.quench_container_bytes or result.compressed_bytes)


def _effective_overhead_bytes(result: BenchmarkResult) -> int:
    if result.quench_container_overhead_bytes is not None:
        return int(result.quench_container_overhead_bytes)
    return int((result.quench_metadata_bytes or 0) + (result.quench_header_bytes or 0))


def _effective_compression_ratio(result: BenchmarkResult) -> float:
    compressed = _effective_compressed_bytes(result)
    return result.raw_bytes / compressed if compressed else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Compute a deterministic percentile using linear interpolation."""
    if not values:
        raise ValueError("percentile requires at least one value")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = rank - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * weight
