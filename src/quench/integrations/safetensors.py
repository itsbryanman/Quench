"""Bundle I/O helpers for Quench compressed tensors.

The `.qnc` container format is intentionally simple and deterministic:

File header:
    - 4 bytes magic: ``QNCB``
    - 2 bytes bundle version (uint16, little-endian)
    - 4 bytes tensor count (uint32, little-endian)

For each tensor, in sorted name order:
    - 4 bytes tensor name length (uint32)
    - tensor name bytes (UTF-8)
    - 4 bytes serialized header length (uint32)
    - serialized header bytes
    - 8 bytes metadata length (uint64)
    - metadata bytes
    - 8 bytes payload length (uint64)
    - payload bytes

The per-tensor `original_nbytes` field is derived from the stored header dtype
and shape when loading the bundle.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from quench.codec.decoder import QuenchDecoder
from quench.codec.encoder import QuenchEncoder
from quench.codec.metadata import deserialize_metadata as _deserialize_metadata
from quench.codec.metadata import serialize_metadata as _serialize_metadata
from quench.core.config import QuenchConfig
from quench.core.exceptions import CodecError
from quench.core.header import decode_header, encode_header
from quench.core.types import CompressedTensor

try:  # pragma: no cover - optional dependency
    from safetensors.numpy import load_file as _load_safetensors
    from safetensors.numpy import save_file as _save_safetensors
except Exception:  # pragma: no cover - optional dependency
    _load_safetensors = None
    _save_safetensors = None


_BUNDLE_MAGIC = b"QNCB"
_BUNDLE_VERSION = 1
_BUNDLE_HEADER = struct.Struct("<4sHI")
_NAME_LENGTH = struct.Struct("<I")
_HEADER_LENGTH = struct.Struct("<I")
_BLOB_LENGTH = struct.Struct("<Q")


def serialize_metadata(metadata: dict[str, Any]) -> bytes:
    """Serialize metadata using the codec's deterministic JSON encoding."""
    return _serialize_metadata(metadata)


def deserialize_metadata(data: bytes) -> dict[str, Any]:
    """Deserialize metadata produced by :func:`serialize_metadata`."""
    return _deserialize_metadata(data)


def save_compressed(
    path: str | Path,
    state_dict: dict[str, np.ndarray[Any, np.dtype[Any]] | Any],
    config: QuenchConfig | None = None,
) -> None:
    """Compress a tensor mapping and write it to a `.qnc` bundle."""
    encoder = QuenchEncoder(config=config)
    save_compressed_bundle(path, encoder.encode_dict(state_dict, config=config))


def load_compressed(
    path: str | Path,
    config: QuenchConfig | None = None,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Load and decompress a `.qnc` bundle into numpy tensors."""
    decoder = QuenchDecoder(config=config)
    return decoder.decode_dict(load_compressed_bundle(path))


def save_compressed_bundle(
    path: str | Path,
    tensors: dict[str, CompressedTensor],
) -> None:
    """Write an already-compressed tensor mapping to a `.qnc` bundle."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as handle:
        handle.write(_BUNDLE_HEADER.pack(_BUNDLE_MAGIC, _BUNDLE_VERSION, len(tensors)))
        for name in sorted(tensors):
            compressed = tensors[name]
            name_bytes = name.encode("utf-8")
            header_bytes = encode_header(compressed.header)

            handle.write(_NAME_LENGTH.pack(len(name_bytes)))
            handle.write(name_bytes)
            handle.write(_HEADER_LENGTH.pack(len(header_bytes)))
            handle.write(header_bytes)
            handle.write(_BLOB_LENGTH.pack(len(compressed.metadata)))
            handle.write(compressed.metadata)
            handle.write(_BLOB_LENGTH.pack(len(compressed.payload)))
            handle.write(compressed.payload)


def load_compressed_bundle(path: str | Path) -> dict[str, CompressedTensor]:
    """Load a `.qnc` bundle without decoding its payloads."""
    source = Path(path)
    with source.open("rb") as handle:
        magic, version, tensor_count = _read_struct(handle, _BUNDLE_HEADER, "bundle header")
        if magic != _BUNDLE_MAGIC:
            raise CodecError(f"QNC bundle magic mismatch: expected {_BUNDLE_MAGIC!r}, got {magic!r}")
        if version != _BUNDLE_VERSION:
            raise CodecError(
                f"Unsupported QNC bundle version: expected {_BUNDLE_VERSION}, got {version}"
            )

        tensors: dict[str, CompressedTensor] = {}
        for _ in range(tensor_count):
            (name_len,) = _read_struct(handle, _NAME_LENGTH, "tensor name length")
            name = _read_exact(handle, name_len, "tensor name").decode("utf-8")

            (header_len,) = _read_struct(handle, _HEADER_LENGTH, "tensor header length")
            header = decode_header(_read_exact(handle, header_len, "tensor header"))

            (metadata_len,) = _read_struct(handle, _BLOB_LENGTH, "metadata length")
            metadata = _read_exact(handle, metadata_len, "tensor metadata")

            (payload_len,) = _read_struct(handle, _BLOB_LENGTH, "payload length")
            payload = _read_exact(handle, payload_len, "tensor payload")

            if name in tensors:
                raise CodecError(f"Duplicate tensor name in QNC bundle: {name}")

            original_nbytes = _header_nbytes(header.dtype, header.shape)
            tensors[name] = CompressedTensor(
                header=header,
                payload=payload,
                metadata=metadata,
                original_nbytes=original_nbytes,
            )

        trailing = handle.read(1)
        if trailing:
            raise CodecError("QNC bundle has trailing bytes after the last tensor")

    return tensors


def load_tensor_mapping(path: str | Path) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Load a tensor mapping from `.qnc`, `.npz`, `.npy`, `.safetensors`, or a directory."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_dir():
        tensors: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
        for file_path in sorted(source.rglob("*.npy")):
            name = str(file_path.relative_to(source).with_suffix("")).replace("\\", "/")
            tensors[name] = np.load(file_path, allow_pickle=False)
        if not tensors:
            raise CodecError(f"No .npy files found in directory: {source}")
        return tensors

    if source.suffix == ".qnc":
        return load_compressed(source)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as loaded:
            return {name: np.asarray(loaded[name]) for name in sorted(loaded.files)}
    if source.suffix == ".npy":
        return {source.stem: np.load(source, allow_pickle=False)}
    if source.suffix == ".safetensors":
        if _load_safetensors is None:
            raise CodecError(
                "Loading .safetensors requires the optional safetensors package; "
                "use .npz or a directory of .npy files instead."
            )
        return {name: np.asarray(value) for name, value in _load_safetensors(str(source)).items()}

    raise CodecError(f"Unsupported tensor input format: {source.suffix or str(source)}")


def save_tensor_mapping(
    path: str | Path,
    tensors: dict[str, np.ndarray[Any, np.dtype[Any]]],
) -> None:
    """Save a tensor mapping as `.npz`, `.npy`, `.safetensors`, or a directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.suffix == ".npz":
        np.savez(destination, **{name: np.asarray(value) for name, value in tensors.items()})
        return
    if destination.suffix == ".npy":
        if len(tensors) != 1:
            raise CodecError("Saving to a single .npy file requires exactly one tensor")
        np.save(destination, next(iter(tensors.values())))
        return
    if destination.suffix == ".safetensors":
        if _save_safetensors is None:
            raise CodecError(
                "Saving .safetensors requires the optional safetensors package; "
                "use .npz or a directory of .npy files instead."
            )
        _save_safetensors({name: np.asarray(value) for name, value in tensors.items()}, str(destination))
        return

    destination.mkdir(parents=True, exist_ok=True)
    for name, value in sorted(tensors.items()):
        file_path = destination / f"{name}.npy"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, np.asarray(value))


def _read_struct(handle: Any, fmt: struct.Struct, label: str) -> tuple[Any, ...]:
    """Read and unpack a fixed-size struct from *handle*."""
    return fmt.unpack(_read_exact(handle, fmt.size, label))


def _read_exact(handle: Any, length: int, label: str) -> bytes:
    """Read exactly *length* bytes or raise a clear bundle error."""
    data = handle.read(length)
    if len(data) != length:
        raise CodecError(f"Unexpected EOF while reading {label}")
    return data


def _header_nbytes(dtype_name: str, shape: tuple[int, ...]) -> int:
    """Derive the original tensor size from header dtype and shape."""
    dtype = np.dtype(dtype_name)
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
