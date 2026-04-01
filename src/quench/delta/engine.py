"""Delta compression engine: compress fine-tunes against a base model."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from quench.codec import QuenchDecoder, QuenchEncoder, deserialize_metadata, serialize_metadata
from quench.core.config import QuenchConfig
from quench.core.types import CodecMode, CompressedTensor, TensorHeader, TensorType
from quench.delta.analysis import analyze_delta
from quench.delta.manifest import (
    MANIFEST_TENSOR_NAME,
    DeltaManifest,
    deserialize_manifest,
    serialize_manifest,
)
from quench.delta.resolve import build_tensor_index, load_tensor, resolve_safetensors
from quench.delta.strategy import decode_delta, encode_delta
from quench.io import QNCReader, QNCWriter


def compress(
    base: str,
    finetune: str,
    output: str | Path,
    *,
    config: QuenchConfig | None = None,
    bits: int = 2,
    verbose: bool = True,
) -> Path:
    """Compress a fine-tuned model as a delta against *base*."""
    active_config = config or QuenchConfig(target_bits=bits, codec_mode=CodecMode.LOSSY)
    output_path = Path(output)

    if verbose:
        print(f"Resolving base: {base}")
    base_index = build_tensor_index(resolve_safetensors(base))
    if verbose:
        print(f"Resolving finetune: {finetune}")
    ft_index = build_tensor_index(resolve_safetensors(finetune))

    base_names = set(base_index)
    ft_names = set(ft_index)
    shared = sorted(base_names & ft_names)
    added = sorted(ft_names - base_names)
    removed = sorted(base_names - ft_names)

    total_count = len(shared) + len(added) + 1
    total_raw = 0
    total_compressed = 0
    profiles: dict[str, dict[str, Any]] = {}
    encoder = QuenchEncoder(config=active_config)

    with QNCWriter(output_path, tensor_count=total_count) as writer:
        for index, name in enumerate(shared, start=1):
            base_tensor = load_tensor(name, base_index)
            ft_tensor = load_tensor(name, ft_index)
            original_dtype = ft_tensor.dtype

            base_values = base_tensor.astype(np.float32, copy=False)
            ft_values = ft_tensor.astype(np.float32, copy=False)
            delta = ft_values - base_values
            profile = analyze_delta(delta, name, default_bits=bits)
            profiles[name] = asdict(profile)

            raw_bytes = int(ft_tensor.nbytes)
            total_raw += raw_bytes
            payload, metadata = encode_delta(
                delta,
                profile.recommended_path,
                active_config,
                bits=profile.recommended_bits,
            )
            metadata["original_dtype"] = original_dtype.str

            compressed = CompressedTensor(
                header=TensorHeader(
                    tensor_type=TensorType.WEIGHT,
                    dtype=np.dtype(original_dtype).name,
                    shape=tuple(int(dim) for dim in delta.shape),
                    codec_mode=(
                        CodecMode.LOSSLESS
                        if profile.recommended_path in {"zero", "lossless"}
                        else CodecMode.LOSSY
                    ),
                    strategy_id=0,
                    checksum=0,
                ),
                payload=payload,
                metadata=serialize_metadata(metadata),
                original_nbytes=raw_bytes,
            )
            writer.write_compressed_tensor(name, compressed)
            total_compressed += compressed.compressed_nbytes

            if verbose:
                ratio = raw_bytes / compressed.compressed_nbytes if compressed.compressed_nbytes else 0.0
                shape_str = "x".join(str(dim) for dim in delta.shape)
                print(
                    f"[{index}/{len(shared)}] {name[:50]:50} [{shape_str:>16}] "
                    f"{profile.recommended_path:12} "
                    f"{raw_bytes / 1024 / 1024:8.1f}MB -> "
                    f"{compressed.compressed_nbytes / 1024 / 1024:8.2f}MB "
                    f"({ratio:6.1f}x)"
                )

        for name in added:
            ft_tensor = load_tensor(name, ft_index)
            compressed = encoder.encode(ft_tensor, name=name)
            writer.write_compressed_tensor(name, compressed)
            total_raw += int(ft_tensor.nbytes)
            total_compressed += compressed.compressed_nbytes
            profiles[name] = {"path": "added", "shape": list(ft_tensor.shape)}
            if verbose:
                print(f"  [added] {name}")

        manifest = DeltaManifest(
            base_model_id=base,
            shared_tensors=shared,
            added_tensors=added,
            removed_tensors=removed,
            tensor_profiles=profiles,
            config=active_config.model_dump(mode="json"),
        )
        manifest_compressed = encoder.encode(serialize_manifest(manifest), name=MANIFEST_TENSOR_NAME)
        writer.write_compressed_tensor(MANIFEST_TENSOR_NAME, manifest_compressed)

    if verbose:
        overall_ratio = total_raw / total_compressed if total_compressed else 0.0
        file_size = output_path.stat().st_size
        print(
            f"\nDelta compression complete:"
            f"\n  Shared tensors:  {len(shared)}"
            f"\n  Added tensors:   {len(added)}"
            f"\n  Removed tensors: {len(removed)}"
            f"\n  Total raw:       {total_raw / 1024 / 1024:.1f} MB"
            f"\n  File size:       {file_size / 1024 / 1024:.1f} MB"
            f"\n  Overall ratio:   {overall_ratio:.1f}x"
            f"\n  Output:          {output_path}"
        )

    return output_path


def load(
    base: str,
    delta: str | Path,
    *,
    config: QuenchConfig | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Load a delta-compressed model by reconstructing against *base*."""
    active_config = config or QuenchConfig()
    manifest = _read_manifest(delta, active_config)
    base_index = build_tensor_index(resolve_safetensors(base))

    decoder = QuenchDecoder(config=active_config)
    delta_records: dict[str, CompressedTensor] = {}
    for record in QNCReader(delta).iter_tensor_records():
        delta_records[record.name] = record.to_compressed_tensor()

    result: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for name in manifest.shared_tensors:
        base_values = load_tensor(name, base_index).astype(np.float32, copy=False)
        compressed = delta_records.get(name)
        if compressed is None:
            raise ValueError(f"Missing delta record for shared tensor: {name}")

        metadata = deserialize_metadata(compressed.metadata)
        strategy_block = metadata.get("strategy")
        if isinstance(strategy_block, dict):
            delta_metadata = strategy_block.get("metadata", metadata)
        else:
            delta_metadata = metadata

        if str(delta_metadata.get("path", "")) == "zero":
            restored = base_values
        else:
            restored = base_values + decode_delta(compressed.payload, delta_metadata, active_config)

        original_dtype = delta_metadata.get("original_dtype")
        if original_dtype is not None:
            restored = restored.astype(np.dtype(str(original_dtype)), copy=False)

        result[name] = restored
        if verbose:
            print(f"  Restored: {name}")

    for name in manifest.added_tensors:
        compressed = delta_records.get(name)
        if compressed is None:
            raise ValueError(f"Missing record for added tensor: {name}")
        result[name] = decoder.decode(compressed)
        if verbose:
            print(f"  Added: {name}")

    return result


def inspect(delta: str | Path) -> dict[str, Any]:
    """Read the manifest from a delta QNC file without loading any model."""
    return asdict(_read_manifest(delta, QuenchConfig()))


def _read_manifest(delta: str | Path, config: QuenchConfig) -> DeltaManifest:
    """Extract and deserialize the manifest from a delta QNC file."""
    decoder = QuenchDecoder(config=config)
    for record in QNCReader(delta).iter_tensor_records():
        if record.name == MANIFEST_TENSOR_NAME:
            manifest_array = decoder.decode(record.to_compressed_tensor())
            return deserialize_manifest(manifest_array)
    raise ValueError(f"No delta manifest found in {delta}")
