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

from benchmarks.reporting import RegressionThresholds
from benchmarks.runner import run_benchmark_suite
from quench.core.config import QuenchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Quench synthetic benchmarks.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV artifacts")
    parser.add_argument("--repeats", type=int, default=3, help="Timing repetitions per benchmark")
    parser.add_argument("--seed", type=int, default=2025, help="Deterministic benchmark seed")
    parser.add_argument("--compare-against", help="Prior JSON artifact used for regression checks")
    parser.add_argument("--max-compression-ratio-drop", type=float, default=0.05)
    parser.add_argument("--max-error-increase", type=float, default=0.10)
    parser.add_argument("--max-throughput-drop", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    thresholds = RegressionThresholds(
        max_compression_ratio_drop=args.max_compression_ratio_drop,
        max_error_increase=args.max_error_increase,
        max_throughput_drop=args.max_throughput_drop,
    )
    try:
        json_path, csv_path, results, comparison_messages = run_benchmark_suite(
            args.output_dir,
            config=QuenchConfig(),
            repeats=args.repeats,
            seed=args.seed,
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
