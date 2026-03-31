"""Profile tensor mappings and report the recommended codec strategy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quench.analyze import TensorProfiler, TensorTypeDetector
from quench.codec import get_strategy
from quench.core.exceptions import QuenchError
from quench.integrations import load_tensor_mapping


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for tensor profiling."""
    parser = argparse.ArgumentParser(
        description="Profile tensors and show the strategy Quench would choose."
    )
    parser.add_argument("--input", required=True, help="Tensor file, directory, or .qnc bundle")
    return parser


def main() -> None:
    """Run the tensor profiler CLI."""
    parser = build_parser()
    args = parser.parse_args()

    profiler = TensorProfiler()
    detector = TensorTypeDetector()

    try:
        tensors = load_tensor_mapping(args.input)
    except (OSError, QuenchError, ValueError, TypeError) as exc:
        print(f"profile_tensor failed: {exc}", file=sys.stderr)
        sys.exit(1)

    rows: list[str] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        stats = profiler.profile(tensor)
        tensor_type = detector.detect(tensor, name=name)
        strategy = get_strategy(tensor_type)
        rows.append(
            f"{name[:24]:24} {str(tuple(tensor.shape))[:18]:18} {tensor.dtype.name:8} "
            f"{stats.mean:9.4f} {stats.std:9.4f} {stats.sparsity:8.3f} "
            f"{stats.entropy_bits:9.3f} {tensor_type.name:12} {strategy.strategy_name:12}"
        )

    print(
        f"{'name':24} {'shape':18} {'dtype':8} {'mean':>9} {'std':>9} "
        f"{'sparsity':>8} {'entropy':>9} {'tensor_type':12} {'strategy':12}"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
