"""Benchmark runner for Quench compression ratios, errors, and throughput."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

from benchmarks.reporting import BenchmarkResult, RegressionThresholds, compare_reports, write_csv_report, write_json_report
from benchmarks.suites import build_bundle_suite, build_tensor_suite
from quench.codec import QuenchDecoder, QuenchEncoder
from quench.core.config import QuenchConfig
from quench.integrations import load_compressed, save_compressed


def run_benchmark_suite(
    output_dir: str | Path,
    *,
    config: QuenchConfig | None = None,
    repeats: int = 3,
    seed: int = 2025,
    compare_against: str | Path | None = None,
    thresholds: RegressionThresholds | None = None,
) -> tuple[Path, Path, list[BenchmarkResult], tuple[str, ...]]:
    """Run the synthetic benchmark suite and emit JSON/CSV artifacts."""
    active_config = config or QuenchConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = [
        *_run_tensor_cases(active_config, repeats=repeats, seed=seed),
        *_run_bundle_cases(active_config, repeats=repeats, seed=seed),
    ]
    json_path = output_path / "quench-benchmarks.json"
    csv_path = output_path / "quench-benchmarks.csv"
    config_json = json.dumps(active_config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    write_json_report(
        results,
        json_path,
        run_metadata={"config": config_json, "repeats": repeats, "seed": seed},
    )
    write_csv_report(results, csv_path)

    comparison_messages: tuple[str, ...] = ()
    if compare_against is not None:
        baseline = _load_results(compare_against)
        comparison = compare_reports(baseline, results, thresholds=thresholds)
        comparison_messages = comparison.messages
        if not comparison.passed:
            raise RuntimeError("\n".join(comparison.messages))

    return json_path, csv_path, results, comparison_messages


def _run_tensor_cases(config: QuenchConfig, *, repeats: int, seed: int) -> list[BenchmarkResult]:
    encoder = QuenchEncoder(config=config)
    decoder = QuenchDecoder(config=config)
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    results: list[BenchmarkResult] = []

    for case in build_tensor_suite(config, seed=seed):
        compressed = encoder.encode(case.tensor, name=case.tensor_name)
        restored = decoder.decode(compressed)

        encode_seconds = _measure_seconds(lambda: encoder.encode(case.tensor, name=case.tensor_name), repeats=repeats)
        decode_seconds = _measure_seconds(lambda: decoder.decode(compressed), repeats=repeats)

        raw_bytes = int(case.tensor.nbytes)
        compressed_bytes = int(compressed.compressed_nbytes)
        mse, max_abs, relative_error = _error_metrics(case.tensor, restored)

        results.append(
            BenchmarkResult(
                benchmark_name=case.benchmark_name,
                tensor_name=case.tensor_name,
                tensor_type=case.tensor_type,
                tensor_shape="x".join(str(dim) for dim in case.tensor.shape),
                dtype=str(case.tensor.dtype),
                config=config_json,
                raw_bytes=raw_bytes,
                compressed_bytes=compressed_bytes,
                compression_ratio=(raw_bytes / compressed_bytes if compressed_bytes else 0.0),
                mse=mse,
                max_abs_error=max_abs,
                relative_error=relative_error,
                encode_throughput_bytes_per_sec=(raw_bytes / encode_seconds if encode_seconds else 0.0),
                decode_throughput_bytes_per_sec=(raw_bytes / decode_seconds if decode_seconds else 0.0),
                backend_name=config.entropy_backend,
            )
        )

    return results


def _run_bundle_cases(config: QuenchConfig, *, repeats: int, seed: int) -> list[BenchmarkResult]:
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    results: list[BenchmarkResult] = []
    for case in build_bundle_suite(config, seed=seed):
        raw_bytes = sum(int(tensor.nbytes) for tensor in case.tensors.values())
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle.qnc"
            save_compressed(bundle_path, case.tensors, config=config)
            restored = load_compressed(bundle_path, config=config)
            compressed_bytes = int(bundle_path.stat().st_size)
            encode_seconds = _measure_seconds(
                lambda: save_compressed(bundle_path, case.tensors, config=config),
                repeats=repeats,
            )
            decode_seconds = _measure_seconds(
                lambda: load_compressed(bundle_path, config=config),
                repeats=repeats,
            )
        mse, max_abs, relative_error = _bundle_error_metrics(case.tensors, restored)
        results.append(
            BenchmarkResult(
                benchmark_name=case.benchmark_name,
                tensor_name="bundle",
                tensor_type=case.tensor_type,
                tensor_shape=f"{len(case.tensors)} tensors",
                dtype="mixed",
                config=config_json,
                raw_bytes=raw_bytes,
                compressed_bytes=compressed_bytes,
                compression_ratio=(raw_bytes / compressed_bytes if compressed_bytes else 0.0),
                mse=mse,
                max_abs_error=max_abs,
                relative_error=relative_error,
                encode_throughput_bytes_per_sec=(raw_bytes / encode_seconds if encode_seconds else 0.0),
                decode_throughput_bytes_per_sec=(raw_bytes / decode_seconds if decode_seconds else 0.0),
                backend_name=config.entropy_backend,
            )
        )
    return results


def _measure_seconds(fn: Callable[[], Any], *, repeats: int) -> float:
    durations: list[float] = []
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return median(durations)


def _error_metrics(
    original: np.ndarray[Any, np.dtype[Any]],
    restored: np.ndarray[Any, np.dtype[Any]],
) -> tuple[float, float, float]:
    orig = np.asarray(original, dtype=np.float64)
    rec = np.asarray(restored, dtype=np.float64)
    diff = rec - orig
    mse = float(np.mean(diff ** 2))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.max(np.abs(orig))) or 1.0
    rel = max_abs / denom
    return mse, max_abs, rel


def _bundle_error_metrics(
    original: dict[str, np.ndarray[Any, np.dtype[Any]]],
    restored: dict[str, np.ndarray[Any, np.dtype[Any]]],
) -> tuple[float, float, float]:
    mses: list[float] = []
    maxes: list[float] = []
    relatives: list[float] = []
    for name, tensor in original.items():
        mse, max_abs, rel = _error_metrics(tensor, restored[name])
        mses.append(mse)
        maxes.append(max_abs)
        relatives.append(rel)
    return float(np.mean(mses)), float(np.max(maxes)), float(np.max(relatives))


def _load_results(path: str | Path) -> list[BenchmarkResult]:
    from benchmarks.reporting import load_json_report

    return load_json_report(path)
