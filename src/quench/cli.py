"""CLI entry point for model compression and decompression."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from quench.core.config import QuenchConfig
from quench.core.exceptions import QuenchError
from quench.core.types import CodecMode
from quench.integrations import (
    load_compressed,
    load_compressed_bundle,
    load_tensor_mapping,
    save_compressed_bundle,
    save_tensor_mapping,
)
from quench.codec import QuenchEncoder


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the public compression CLI."""
    parser = argparse.ArgumentParser(
        description="Compress or decompress tensor mappings with the Quench codec."
    )
    parser.add_argument("--input", required=True, help="Input tensor file, directory, or .qnc bundle")
    parser.add_argument("--output", required=True, help="Output path for the converted data")
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Target quantization bit width for lossy compression",
    )
    parser.add_argument(
        "--mode",
        choices=("lossy", "lossless"),
        default="lossy",
        help="Codec fidelity mode used during compression",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Interpret the input as a .qnc bundle and restore the tensor mapping",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Run the compression CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.decompress:
            compressed = load_compressed_bundle(args.input)
            restored = load_compressed(args.input)
            save_tensor_mapping(args.output, restored)
            print(_render_decompression_table(compressed))
            return 0

        config = QuenchConfig(
            target_bits=args.bits,
            codec_mode=CodecMode.LOSSLESS if args.mode == "lossless" else CodecMode.LOSSY,
        )
        tensors = load_tensor_mapping(args.input)
        encoder = QuenchEncoder(config=config)
        compressed = encoder.encode_dict(tensors, config=config)
        save_compressed_bundle(args.output, compressed)
        print(_render_compression_table(tensors, compressed))
        return 0
    except (OSError, QuenchError, ValueError, TypeError) as exc:
        print(f"quench-compress failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """Entry point for the ``quench-compress`` command."""
    sys.exit(run())


def _render_compression_table(
    tensors: dict[str, object],
    compressed: dict[str, object],
) -> str:
    """Render a readable per-tensor compression summary."""
    rows: list[str] = []
    total_raw = 0
    total_compressed = 0

    for name in sorted(tensors):
        raw = int(getattr(tensors[name], "nbytes"))
        blob = compressed[name]
        compressed_nbytes = int(getattr(blob, "compressed_nbytes"))
        ratio = raw / compressed_nbytes if compressed_nbytes else 0.0
        tensor_type = getattr(blob, "header").tensor_type.name
        rows.append(
            f"{name[:28]:28} {tensor_type:12} {raw:10d} {compressed_nbytes:12d} {ratio:8.3f}"
        )
        total_raw += raw
        total_compressed += compressed_nbytes

    total_ratio = total_raw / total_compressed if total_compressed else 0.0
    header = (
        f"{'tensor':28} {'type':12} {'raw_bytes':>10} {'compressed':>12} {'ratio':>8}\n"
        + "\n".join(rows)
    )
    footer = (
        f"\n{'TOTAL':28} {'':12} {total_raw:10d} {total_compressed:12d} {total_ratio:8.3f}"
    )
    return header + footer


def _render_decompression_table(compressed: dict[str, object]) -> str:
    """Render a readable per-tensor decompression summary."""
    rows: list[str] = []
    total_raw = 0
    total_compressed = 0

    for name in sorted(compressed):
        blob = compressed[name]
        raw = int(getattr(blob, "original_nbytes"))
        compressed_nbytes = int(getattr(blob, "compressed_nbytes"))
        ratio = raw / compressed_nbytes if compressed_nbytes else 0.0
        rows.append(f"{name[:28]:28} {compressed_nbytes:12d} {raw:10d} {ratio:8.3f}")
        total_raw += raw
        total_compressed += compressed_nbytes

    total_ratio = total_raw / total_compressed if total_compressed else 0.0
    header = f"{'tensor':28} {'compressed':>12} {'restored':>10} {'ratio':>8}\n" + "\n".join(rows)
    footer = f"\n{'TOTAL':28} {total_compressed:12d} {total_raw:10d} {total_ratio:8.3f}"
    return header + footer


if __name__ == "__main__":
    main()
