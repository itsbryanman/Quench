"""CLI entry point for model compression and decompression."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from quench.codec import QuenchDecoder, QuenchEncoder
from quench.core.config import CalibrationPolicyKind, QuantizationGranularity, QuenchConfig
from quench.core.exceptions import QuenchError
from quench.core.types import CodecMode
from quench.integrations import iter_tensor_mapping, save_tensor_mapping
from quench.io import QNCReader, QNCWriter


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the public compression CLI."""
    parser = argparse.ArgumentParser(
        description="Compress or decompress tensor mappings with the Quench codec."
    )
    parser.add_argument("--input", required=True, help="Input tensor file, directory, or .qnc bundle")
    parser.add_argument("--output", required=True, help="Output path for the converted data")
    parser.add_argument("--bits", type=int, default=4, help="Target quantization bit width")
    parser.add_argument(
        "--mode",
        choices=("lossy", "lossless"),
        default="lossy",
        help="Codec fidelity mode used during compression",
    )
    parser.add_argument(
        "--granularity",
        choices=tuple(item.value for item in QuantizationGranularity),
        default=QuantizationGranularity.PER_CHANNEL.value,
        help="Quantization granularity used for lossy paths",
    )
    parser.add_argument(
        "--calibration-policy",
        choices=tuple(item.value for item in CalibrationPolicyKind),
        default=CalibrationPolicyKind.MINMAX.value,
        help="Calibration policy used to derive quantization ranges",
    )
    parser.add_argument("--block-size", type=int, default=128, help="Block size for blockwise quantization")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Percentile value used by percentile calibration",
    )
    parser.add_argument(
        "--quant-axis",
        type=int,
        default=None,
        help="Optional quantization axis override",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1 << 20,
        help="Payload chunk size for streamed QNC records",
    )
    parser.add_argument(
        "--pack-bits",
        action="store_true",
        help="Use backend bit-packing for quantized symbol streams when possible",
    )
    parser.add_argument(
        "--entropy-backend",
        default="python",
        help="Entropy backend name registered in Quench",
    )
    parser.add_argument(
        "--packing-backend",
        default="python",
        help="Packing backend name registered in Quench",
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
            return _run_decompression(args.input, args.output)

        config = QuenchConfig(
            target_bits=args.bits,
            codec_mode=CodecMode.LOSSLESS if args.mode == "lossless" else CodecMode.LOSSY,
            quantization_granularity=QuantizationGranularity(args.granularity),
            calibration_policy=CalibrationPolicyKind(args.calibration_policy),
            block_size=args.block_size,
            percentile_value=args.percentile,
            quantization_axis=args.quant_axis,
            pack_bits=args.pack_bits,
            entropy_backend=args.entropy_backend,
            packing_backend=args.packing_backend,
        )
        return _run_compression(args.input, args.output, config=config, chunk_size=args.chunk_size)
    except (OSError, QuenchError, ValueError, TypeError) as exc:
        print(f"quench-compress failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """Entry point for the ``quench-compress`` command."""
    sys.exit(run())


def _run_compression(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: QuenchConfig,
    chunk_size: int,
) -> int:
    encoder = QuenchEncoder(config=config)
    total_count = _count_tensor_inputs(input_path)
    total_raw = 0
    total_compressed = 0
    rows: list[str] = []

    with QNCWriter(output_path, tensor_count=total_count, chunk_size=chunk_size) as writer:
        for index, (name, tensor) in enumerate(iter_tensor_mapping(input_path), start=1):
            compressed = encoder.encode(tensor, name=name)
            writer.write_compressed_tensor(name, compressed, chunk_size=chunk_size)
            raw = int(np.asarray(tensor).nbytes)
            compressed_nbytes = compressed.compressed_nbytes
            ratio = raw / compressed_nbytes if compressed_nbytes else 0.0
            total_raw += raw
            total_compressed += compressed_nbytes
            rows.append(
                f"{name[:28]:28} {compressed.header.tensor_type.name:16} {raw:10d} {compressed_nbytes:12d} {ratio:8.3f}"
            )
            print(_progress_line("compress", index, total_count, name, raw, compressed_nbytes))

    total_ratio = total_raw / total_compressed if total_compressed else 0.0
    print(
        f"{'tensor':28} {'type':16} {'raw_bytes':>10} {'compressed':>12} {'ratio':>8}\n"
        + "\n".join(rows)
        + f"\n{'TOTAL':28} {'':16} {total_raw:10d} {total_compressed:12d} {total_ratio:8.3f}"
    )
    return 0


def _run_decompression(input_path: str | Path, output_path: str | Path) -> int:
    reader = QNCReader(input_path)
    decoder = QuenchDecoder()
    output = Path(output_path)
    total_raw = 0
    total_compressed = 0
    rows: list[str] = []

    if _can_stream_directory_output(output):
        output.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(reader.iter_tensor_records(), start=1):
            restored = decoder.decode(record.to_compressed_tensor())
            _save_tensor_to_directory(output, record.name, restored)
            compressed_nbytes = record.payload_nbytes + len(record.metadata) + 64
            raw = int(restored.nbytes)
            total_raw += raw
            total_compressed += compressed_nbytes
            rows.append(f"{record.name[:28]:28} {compressed_nbytes:12d} {raw:10d} {raw / compressed_nbytes:8.3f}")
            print(_progress_line("decompress", index, None, record.name, compressed_nbytes, raw))
    else:
        restored_tensors: dict[str, np.ndarray] = {}
        for index, record in enumerate(reader.iter_tensor_records(), start=1):
            restored = decoder.decode(record.to_compressed_tensor())
            restored_tensors[record.name] = restored
            compressed_nbytes = record.payload_nbytes + len(record.metadata) + 64
            raw = int(restored.nbytes)
            total_raw += raw
            total_compressed += compressed_nbytes
            rows.append(f"{record.name[:28]:28} {compressed_nbytes:12d} {raw:10d} {raw / compressed_nbytes:8.3f}")
            print(_progress_line("decompress", index, None, record.name, compressed_nbytes, raw))
        save_tensor_mapping(output, restored_tensors)

    total_ratio = total_raw / total_compressed if total_compressed else 0.0
    print(
        f"{'tensor':28} {'compressed':>12} {'restored':>10} {'ratio':>8}\n"
        + "\n".join(rows)
        + f"\n{'TOTAL':28} {total_compressed:12d} {total_raw:10d} {total_ratio:8.3f}"
    )
    return 0


def _count_tensor_inputs(path: str | Path) -> int | None:
    source = Path(path)
    if source.is_dir():
        return len(list(source.rglob("*.npy")))
    if source.suffix == ".npy":
        return 1
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as loaded:
            return len(loaded.files)
    return None


def _progress_line(
    action: str,
    index: int,
    total: int | None,
    name: str,
    left_bytes: int,
    right_bytes: int,
) -> str:
    prefix = f"[{index}/{total}]" if total is not None else f"[{index}]"
    return f"{prefix} {action:10} {name} {left_bytes}B -> {right_bytes}B"


def _can_stream_directory_output(path: Path) -> bool:
    return path.suffix == "" or path.is_dir()


def _save_tensor_to_directory(
    destination: Path,
    name: str,
    tensor: np.ndarray[Any, np.dtype[Any]],
) -> None:
    file_path = destination / f"{name}.npy"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, tensor)


if __name__ == "__main__":
    main()
