"""CLI entry point for delta compression and loading."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from quench.core.config import QuenchConfig
from quench.core.types import CodecMode


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``quench-delta`` CLI."""
    parser = argparse.ArgumentParser(
        prog="quench-delta",
        description="Compress or load fine-tuned models as deltas against a base model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("compress", help="Compress a fine-tune as a delta against a base model.")
    comp.add_argument("--base", required=True, help="Base model: local path or HuggingFace repo ID")
    comp.add_argument("--finetune", required=True, help="Fine-tuned model: local path or HuggingFace repo ID")
    comp.add_argument("--output", required=True, help="Output .qnc delta file path")
    comp.add_argument("--bits", type=int, default=2, help="Default quantization bit width (default: 2)")
    comp.add_argument("--entropy-backend", default="python", help="Entropy backend name")

    load_parser = sub.add_parser("load", help="Reconstruct a fine-tuned model from a delta file.")
    load_parser.add_argument("--base", required=True, help="Base model: local path or HuggingFace repo ID")
    load_parser.add_argument("--delta", required=True, help="Delta .qnc file path")
    load_parser.add_argument("--output", required=True, help="Output path for restored tensors")

    inspect_parser = sub.add_parser("inspect", help="Print the manifest of a delta file.")
    inspect_parser.add_argument("--delta", required=True, help="Delta .qnc file path")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Run the ``quench-delta`` CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "compress":
            return _run_compress(args)
        if args.command == "load":
            return _run_load(args)
        if args.command == "inspect":
            return _run_inspect(args)
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"quench-delta failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """Entry point for the ``quench-delta`` command."""
    sys.exit(run())


def _run_compress(args: argparse.Namespace) -> int:
    from quench.delta.engine import compress

    config = QuenchConfig(
        target_bits=args.bits,
        codec_mode=CodecMode.LOSSY,
        entropy_backend=args.entropy_backend,
    )
    compress(
        base=args.base,
        finetune=args.finetune,
        output=args.output,
        config=config,
        bits=args.bits,
        verbose=True,
    )
    return 0


def _run_load(args: argparse.Namespace) -> int:
    from quench.delta.engine import load
    from quench.integrations import save_tensor_mapping

    state_dict = load(base=args.base, delta=args.delta, verbose=True)
    save_tensor_mapping(args.output, state_dict)
    print(f"Restored {len(state_dict)} tensors to {args.output}")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    from quench.delta.engine import inspect

    info = inspect(args.delta)
    print(json.dumps(info, indent=2, default=str))
    print(
        f"\nSummary:"
        f"\n  Base model:      {info['base_model_id']}"
        f"\n  Shared tensors:  {len(info['shared_tensors'])}"
        f"\n  Added tensors:   {len(info['added_tensors'])}"
        f"\n  Removed tensors: {len(info['removed_tensors'])}"
    )
    return 0
