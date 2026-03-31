"""Summarize Quench benchmark artifacts with bucket spotlights and comparisons."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reporting import BenchmarkResult, build_aggregates, load_json_report
from quench.codec.metadata import serialize_metadata

_SPOTLIGHT_TYPES = ("mask", "bias", "mixed_precision")
_EXACT_KINDS = {"raw", "const", "aseq", "bseq", "lossless"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Path to a quench-benchmarks.json artifact")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional baseline artifact to compare against",
    )
    parser.add_argument(
        "--worst-limit",
        type=int,
        default=8,
        help="Number of worst rows to print for zstd(quantized) deltas",
    )
    parser.add_argument(
        "--spotlight-limit",
        type=int,
        default=8,
        help="Number of spotlight rows to print per tensor type",
    )
    parser.add_argument(
        "--tiny-threshold",
        type=int,
        default=4096,
        help="Raw-byte threshold for the tiny-tensor overhead cohort",
    )
    args = parser.parse_args()

    current = load_json_report(args.artifact)
    print(f"Artifact: {args.artifact}")
    summarize_results(
        current,
        worst_limit=args.worst_limit,
        spotlight_limit=args.spotlight_limit,
        tiny_threshold=args.tiny_threshold,
    )

    if args.compare is not None:
        baseline = load_json_report(args.compare)
        print()
        print(f"Compare Against: {args.compare}")
        summarize_comparison(
            baseline,
            current,
            spotlight_limit=args.spotlight_limit,
            tiny_threshold=args.tiny_threshold,
        )


def summarize_results(
    results: list[BenchmarkResult],
    *,
    worst_limit: int,
    spotlight_limit: int,
    tiny_threshold: int,
) -> None:
    """Print high-signal summaries for one artifact."""
    aggregates = build_aggregates(results)
    by_run = aggregates["by_run"]

    print("Overall:")
    print(_format_aggregate_line("run", by_run))
    print(
        "  metadata+header mean/median:"
        f" {by_run['mean_metadata_plus_header_bytes']:.1f} / {by_run['median_metadata_plus_header_bytes']:.1f}"
    )

    print()
    print("Per-model aggregates:")
    for aggregate in aggregates["by_model"]:
        label = aggregate["model_id"]
        revision = aggregate.get("model_revision") or ""
        if revision:
            label = f"{label} @ {revision[:12]}"
        print(_format_aggregate_line(label, aggregate, indent="  "))

    print()
    print("Per-tensor-type aggregates:")
    for aggregate in aggregates["by_tensor_type"]:
        print(_format_aggregate_line(aggregate["tensor_type"], aggregate, indent="  "))

    print()
    print("Worst rows vs zstd(quantized):")
    for row in _worst_rows_vs_quantized(results, worst_limit):
        print(_format_row_summary(row, indent="  "))

    print()
    print(f"Tiny exact tensor cohort (raw_bytes <= {tiny_threshold}):")
    _print_tiny_overhead(_tiny_exact_rows(results, tiny_threshold))

    for tensor_type in _SPOTLIGHT_TYPES:
        print()
        print(f"Spotlight: {tensor_type}")
        spotlight_rows = [row for row in results if row.tensor_type == tensor_type]
        if not spotlight_rows:
            print("  <none>")
            continue
        for row in _sort_spotlight_rows(spotlight_rows)[:spotlight_limit]:
            print(_format_row_summary(row, indent="  "))


def summarize_comparison(
    baseline: list[BenchmarkResult],
    current: list[BenchmarkResult],
    *,
    spotlight_limit: int,
    tiny_threshold: int,
) -> None:
    """Print aggregate and spotlight deltas for two artifacts."""
    baseline_aggregates = _aggregate_map(baseline)
    current_aggregates = _aggregate_map(current)

    print("Aggregate deltas:")
    for key in ("run", "mask", "bias", "mixed_precision"):
        before = baseline_aggregates.get(key)
        after = current_aggregates.get(key)
        if before is None or after is None:
            continue
        print(_format_delta_line(key, before, after))

    print()
    print(f"Tiny exact tensor cohort delta (raw_bytes <= {tiny_threshold}):")
    _print_tiny_overhead_delta(baseline, current, tiny_threshold=tiny_threshold)

    print()
    print("Spotlight deltas:")
    baseline_rows = {row.benchmark_name: row for row in baseline}
    current_rows = {row.benchmark_name: row for row in current}
    spotlight = [
        row
        for row in current
        if row.tensor_type in _SPOTLIGHT_TYPES and row.benchmark_name in baseline_rows
    ]
    spotlight.sort(
        key=lambda row: (
            baseline_rows[row.benchmark_name].compressed_bytes - row.compressed_bytes,
            row.tensor_type,
            row.benchmark_name,
        ),
        reverse=True,
    )
    for row in spotlight[:spotlight_limit]:
        before = baseline_rows[row.benchmark_name]
        print(_format_row_delta(before, row, indent="  "))


def _aggregate_map(results: Iterable[BenchmarkResult]) -> dict[str, dict[str, float | int | None | str]]:
    """Flatten the run and tensor-type aggregates into one lookup."""
    aggregates = build_aggregates(list(results))
    mapped: dict[str, dict[str, float | int | None | str]] = {"run": aggregates["by_run"]}
    for aggregate in aggregates["by_tensor_type"]:
        mapped[str(aggregate["tensor_type"])] = aggregate
    return mapped


def _worst_rows_vs_quantized(results: list[BenchmarkResult], limit: int) -> list[BenchmarkResult]:
    """Rows where Quench loses most badly to zstd(quantized)."""
    rows = [row for row in results if row.zstd_quantized_bytes is not None]
    rows.sort(
        key=lambda row: (
            _effective_compressed_bytes(row) - (row.zstd_quantized_bytes or 0),
            _effective_compressed_bytes(row) - (row.zstd_raw_bytes or 0),
        ),
        reverse=True,
    )
    return rows[:limit]


def _sort_spotlight_rows(rows: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Sort spotlight rows by the most relevant loss against baselines."""
    rows = list(rows)
    rows.sort(
        key=lambda row: (
            _effective_compressed_bytes(row) - (row.zstd_raw_bytes or math.inf),
            _effective_compressed_bytes(row) - (row.zstd_quantized_bytes or math.inf),
            _effective_compressed_bytes(row),
        ),
        reverse=True,
    )
    return rows


def _metadata_ratio(row: BenchmarkResult) -> float:
    """Return metadata+header share of the compressed tensor."""
    metadata = _effective_overhead_bytes(row)
    compressed = _effective_compressed_bytes(row)
    return metadata / compressed if compressed else 0.0


def _print_tiny_overhead(rows: list[BenchmarkResult]) -> None:
    """Print tiny-tensor metadata/header ratios."""
    if not rows:
        print("  <none>")
        return
    ratios = sorted(_metadata_ratio(row) for row in rows)
    mean_ratio = sum(ratios) / len(ratios)
    median_ratio = ratios[len(ratios) // 2]
    mean_bytes = sum(_effective_overhead_bytes(row) for row in rows) / len(rows)
    mean_saved_vs_zstd = [
        (row.zstd_raw_bytes - _effective_compressed_bytes(row))
        for row in rows
        if row.zstd_raw_bytes is not None
    ]
    print(
        "  rows:"
        f" {len(rows)}, mean metadata+header bytes {mean_bytes:.1f},"
        f" mean ratio {mean_ratio:.3f}, median ratio {median_ratio:.3f}"
    )
    if mean_saved_vs_zstd:
        print(
            "  mean bytes saved vs zstd(raw): "
            f"{sum(mean_saved_vs_zstd) / len(mean_saved_vs_zstd):+.1f}"
        )


def _print_tiny_overhead_delta(
    baseline: list[BenchmarkResult],
    current: list[BenchmarkResult],
    *,
    tiny_threshold: int,
) -> None:
    """Compare tiny-tensor metadata/header ratios between two artifacts."""
    baseline_rows = _tiny_exact_rows(baseline, tiny_threshold)
    current_rows = _tiny_exact_rows(current, tiny_threshold)
    if not baseline_rows or not current_rows:
        print("  <insufficient data>")
        return
    baseline_mean = sum(_metadata_ratio(row) for row in baseline_rows) / len(baseline_rows)
    current_mean = sum(_metadata_ratio(row) for row in current_rows) / len(current_rows)
    baseline_bytes = sum(_effective_overhead_bytes(row) for row in baseline_rows) / len(baseline_rows)
    current_bytes = sum(_effective_overhead_bytes(row) for row in current_rows) / len(current_rows)
    baseline_total = sum(_effective_compressed_bytes(row) for row in baseline_rows)
    current_total = sum(_effective_compressed_bytes(row) for row in current_rows)
    print(
        "  mean metadata+header bytes:"
        f" {baseline_bytes:.1f} -> {current_bytes:.1f} (delta {current_bytes - baseline_bytes:+.1f})"
    )
    print(
        "  mean metadata/header ratio:"
        f" {baseline_mean:.3f} -> {current_mean:.3f} (delta {current_mean - baseline_mean:+.3f})"
    )
    print(
        "  aggregate tiny exact bytes:"
        f" {baseline_total} -> {current_total} (delta {current_total - baseline_total:+d})"
    )


def _format_aggregate_line(
    label: str,
    aggregate: dict[str, float | int | None | str],
    *,
    indent: str = "",
) -> str:
    """Format one aggregate line."""
    return (
        f"{indent}{label}: rows={aggregate['rows']} "
        f"agg_ratio={_fmt_optional(aggregate.get('aggregate_compression_ratio'))} "
        f"vs_zstd_raw={_fmt_optional(aggregate.get('mean_bytes_saved_vs_zstd_raw'))}B "
        f"vs_zstd_q={_fmt_optional(aggregate.get('mean_bytes_saved_vs_zstd_quantized'))}B "
        f"wins_raw={aggregate.get('wins_vs_zstd_raw')} "
        f"wins_q={aggregate.get('wins_vs_zstd_quantized')}"
    )


def _format_delta_line(
    label: str,
    before: dict[str, float | int | None | str],
    after: dict[str, float | int | None | str],
) -> str:
    """Format one aggregate delta line."""
    return (
        f"  {label}: "
        f"agg_ratio {before['aggregate_compression_ratio']:.4f} -> {after['aggregate_compression_ratio']:.4f}; "
        f"mean_vs_zstd_raw {before.get('mean_bytes_saved_vs_zstd_raw')} -> {after.get('mean_bytes_saved_vs_zstd_raw')}; "
        f"mean_vs_zstd_q {before.get('mean_bytes_saved_vs_zstd_quantized')} -> {after.get('mean_bytes_saved_vs_zstd_quantized')}; "
        f"mean_meta+hdr {before['mean_metadata_plus_header_bytes']:.1f} -> {after['mean_metadata_plus_header_bytes']:.1f}"
    )


def _format_row_summary(row: BenchmarkResult, *, indent: str = "") -> str:
    """Format one spotlight row."""
    return (
        f"{indent}{row.tensor_type:>15}  {row.model_id}  {row.tensor_name}  "
        f"raw={row.raw_bytes} q={_effective_compressed_bytes(row)} "
        f"zraw={_fmt_int(row.zstd_raw_bytes)} zq={_fmt_int(row.zstd_quantized_bytes)} "
        f"payload={_fmt_int(_effective_payload_bytes(row))} meta={_fmt_int(_effective_overhead_bytes(row))} "
        f"kind={row.quench_exact_kind or 'n/a'} meta_ratio={_metadata_ratio(row):.3f}"
    )


def _format_row_delta(before: BenchmarkResult, after: BenchmarkResult, *, indent: str = "") -> str:
    """Format one per-row comparison line."""
    return (
        f"{indent}{after.tensor_type:>15}  {after.model_id}  {after.tensor_name}  "
        f"compressed {_effective_compressed_bytes(before)} -> {_effective_compressed_bytes(after)} "
        f"(delta {_effective_compressed_bytes(after) - _effective_compressed_bytes(before):+d}); "
        f"payload {_effective_payload_bytes(before)} -> {_effective_payload_bytes(after)}; "
        f"meta {_effective_overhead_bytes(before)} -> {_effective_overhead_bytes(after)}"
    )


def _tiny_exact_rows(results: list[BenchmarkResult], tiny_threshold: int) -> list[BenchmarkResult]:
    return [
        row
        for row in results
        if row.raw_bytes <= tiny_threshold and _is_exact_row(row)
    ]


def _is_exact_row(row: BenchmarkResult) -> bool:
    if row.quench_exact_kind is not None:
        return row.quench_exact_kind in _EXACT_KINDS
    return row.mse == 0.0 and row.max_abs_error == 0.0 and row.relative_error == 0.0


def _effective_compressed_bytes(row: BenchmarkResult) -> int:
    if row.quench_container_bytes is not None:
        return int(row.quench_container_bytes)
    if _is_exact_row(row) and row.raw_bytes <= 4096:
        return _estimated_v2_storage(row)[0]
    return int(row.compressed_bytes)


def _effective_payload_bytes(row: BenchmarkResult) -> int:
    if row.quench_container_payload_bytes is not None:
        return int(row.quench_container_payload_bytes)
    return int(row.quench_payload_bytes or 0)


def _effective_overhead_bytes(row: BenchmarkResult) -> int:
    if row.quench_container_overhead_bytes is not None:
        return int(row.quench_container_overhead_bytes)
    if _is_exact_row(row) and row.raw_bytes <= 4096:
        return _estimated_v2_storage(row)[2]
    return int((row.quench_metadata_bytes or 0) + (row.quench_header_bytes or 0))


def _estimated_v2_storage(row: BenchmarkResult) -> tuple[int, int, int]:
    payload = int(row.quench_payload_bytes or 0)
    metadata_len = int(row.quench_metadata_bytes or 0)
    record_metadata = serialize_metadata(
        {
            "chunk_count": 1,
            "chunk_lengths": [payload],
            "header": b"\x00" * 64,
            "metadata": b"\x00" * metadata_len,
            "name": row.tensor_name,
            "original_nbytes": row.raw_bytes,
            "record_id": 0,
        }
    )
    chunk_metadata = serialize_metadata(
        {
            "index": 0,
            "length": payload,
            "record_id": 0,
        }
    )
    total = 24 + len(record_metadata) + 24 + len(chunk_metadata) + payload
    return total, payload, total - payload


def _fmt_optional(value: float | int | None) -> str:
    """Format optional numeric values compactly."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def _fmt_int(value: int | None) -> str:
    """Format optional integer values."""
    return "n/a" if value is None else str(value)


if __name__ == "__main__":
    main()
