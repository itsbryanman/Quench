"""CLI wrapper for running Quench benchmarks and optional regression checks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.real_models import load_model_manifest
from benchmarks.reporting import RegressionThresholds
from benchmarks.runner import run_benchmark_suite
from quench.core.config import QuenchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Quench synthetic or real-model benchmarks.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV artifacts")
    parser.add_argument(
        "--suite",
        choices=("synthetic", "real", "all"),
        default="synthetic",
        help="Benchmark suite to run",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timing repetitions per benchmark")
    parser.add_argument("--seed", type=int, default=2025, help="Deterministic benchmark seed")
    parser.add_argument(
        "--model-manifest",
        help="Path to a JSON manifest produced by tools/download_models.py",
    )
    parser.add_argument(
        "--real-model-mode",
        choices=("full", "sampled"),
        default="full",
        help="Whether to benchmark all discovered tensors or a deterministic subset",
    )
    parser.add_argument(
        "--sampled-extra-tensors",
        type=int,
        default=16,
        help="Additional non-core tensors to include when --real-model-mode=sampled",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=3,
        help="Compression level for zstd baseline measurements",
    )
    parser.add_argument("--compare-against", help="Prior JSON artifact used for regression checks")
    parser.add_argument("--max-compression-ratio-drop", type=float, default=0.05)
    parser.add_argument("--max-error-increase", type=float, default=0.10)
    parser.add_argument("--max-throughput-drop", type=float, default=0.10)
    parser.add_argument("--entropy-backend", default="python", help="Entropy backend to benchmark")
    parser.add_argument("--packing-backend", default="python", help="Packing backend to benchmark")
    parser.add_argument(
        "--pack-bits",
        action="store_true",
        help="Enable backend bit-packing for eligible integer symbol streams",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_specs = None
    if args.suite in {"real", "all"}:
        if not args.model_manifest:
            print("--model-manifest is required when --suite is real or all", file=sys.stderr)
            sys.exit(1)
        model_specs = load_model_manifest(args.model_manifest)
        if not model_specs:
            print(f"No models found in manifest: {args.model_manifest}", file=sys.stderr)
            sys.exit(1)

    thresholds = RegressionThresholds(
        max_compression_ratio_drop=args.max_compression_ratio_drop,
        max_error_increase=args.max_error_increase,
        max_throughput_drop=args.max_throughput_drop,
    )
    try:
        json_path, csv_path, results, comparison_messages = run_benchmark_suite(
            args.output_dir,
            config=QuenchConfig(
                entropy_backend=args.entropy_backend,
                packing_backend=args.packing_backend,
                pack_bits=args.pack_bits,
            ),
            repeats=args.repeats,
            seed=args.seed,
            suite=args.suite,
            real_model_specs=model_specs,
            real_model_mode=args.real_model_mode,
            sampled_extra_tensors=args.sampled_extra_tensors,
            zstd_level=args.zstd_level,
            compare_against=args.compare_against,
            thresholds=thresholds,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(results)} benchmark rows to {json_path} and {csv_path}")
    for message in comparison_messages:
        print(message)


if __name__ == "__main__":
    main()
