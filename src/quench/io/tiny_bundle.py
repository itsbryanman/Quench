"""Compact shared-container encoding for tiny exact tensors.

The tiny exact bundle stores a group of logical tensors inside one QNC segment.
Each entry contributes:

* name length + UTF-8 name bytes
* exact kind byte (`raw`, `const`, `aseq`, or `bseq`)
* tensor type / strategy id / checksum
* compact dtype token
* ndim and varint-encoded shape dims
* structural parameters for sequence kinds

The segment payload stores only the concatenated per-entry payload bytes for the
`raw` and `const` kinds in descriptor order. Sequence kinds carry no payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quench.codec.metadata import deserialize_metadata, serialize_metadata
from quench.core.types import CodecMode, CompressedTensor, TensorHeader, TensorType

TINY_BUNDLE_MAGIC = b"QNTB"
TINY_BUNDLE_VERSION = 1

_TINY_KIND_TO_CODE = {
    "raw": 0,
    "const": 1,
    "aseq": 2,
    "bseq": 3,
}
_TINY_CODE_TO_KIND = {code: kind for kind, code in _TINY_KIND_TO_CODE.items()}

_DTYPE_CODES = (
    np.dtype(np.bool_).str,
    np.dtype(np.uint8).str,
    np.dtype(np.int8).str,
    np.dtype(np.uint16).str,
    np.dtype(np.int16).str,
    np.dtype(np.uint32).str,
    np.dtype(np.int32).str,
    np.dtype(np.uint64).str,
    np.dtype(np.int64).str,
    np.dtype(np.float16).str,
    np.dtype(np.float32).str,
    np.dtype(np.float64).str,
)
_DTYPE_TO_CODE = {dtype: index for index, dtype in enumerate(_DTYPE_CODES)}
_INLINE_DTYPE_CODE = 0xFF
_RAW_TINY_NBYTES_LIMIT = 2048
_STRUCTURED_TINY_NBYTES_LIMIT = 4096


@dataclass(frozen=True)
class TinyExactBundleEntry:
    """One tiny exact tensor that can share a bundle envelope."""

    name: str
    tensor_type: TensorType
    strategy_id: int
    checksum: int
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    kind: str
    payload: bytes
    start: int = 0
    step: int = 0
    axis: int = 0

    @property
    def header(self) -> TensorHeader:
        return TensorHeader(
            tensor_type=self.tensor_type,
            dtype=self.dtype.name,
            shape=self.shape,
            codec_mode=CodecMode.LOSSLESS,
            strategy_id=self.strategy_id,
            checksum=self.checksum,
        )

    @property
    def original_nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def payload_nbytes(self) -> int:
        if self.kind == "raw":
            return self.original_nbytes
        if self.kind == "const":
            return self.dtype.itemsize
        return 0

    @property
    def metadata_bytes(self) -> bytes:
        metadata: dict[str, Any] = {"l": 1, "k": self.kind}
        if self.kind == "aseq":
            metadata["v"] = self.start
            metadata["p"] = self.step
        elif self.kind == "bseq":
            metadata["a"] = self.axis
            metadata["v"] = self.start
            metadata["p"] = self.step
        return serialize_metadata(metadata)


@dataclass(frozen=True)
class TinyExactBundleStoredEntry:
    """Decoded tiny-bundle entry metadata plus its payload slice description."""

    name: str
    tensor_type: TensorType
    strategy_id: int
    checksum: int
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    kind: str
    payload_length: int
    descriptor_nbytes: int
    start: int = 0
    step: int = 0
    axis: int = 0

    @property
    def header(self) -> TensorHeader:
        return TensorHeader(
            tensor_type=self.tensor_type,
            dtype=self.dtype.name,
            shape=self.shape,
            codec_mode=CodecMode.LOSSLESS,
            strategy_id=self.strategy_id,
            checksum=self.checksum,
        )

    @property
    def original_nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def metadata_bytes(self) -> bytes:
        metadata: dict[str, Any] = {"l": 1, "k": self.kind}
        if self.kind == "aseq":
            metadata["v"] = self.start
            metadata["p"] = self.step
        elif self.kind == "bseq":
            metadata["a"] = self.axis
            metadata["v"] = self.start
            metadata["p"] = self.step
        return serialize_metadata(metadata)


def try_build_tiny_exact_bundle_entry(
    name: str,
    compressed: CompressedTensor,
) -> TinyExactBundleEntry | None:
    """Return a bundle entry when *compressed* matches the tiny exact fast path."""
    header = compressed.header
    if header.codec_mode != CodecMode.LOSSLESS:
        return None

    metadata = deserialize_metadata(compressed.metadata)
    strategy_metadata = _resolve_strategy_metadata(metadata)
    if not _is_lossless_strategy_metadata(strategy_metadata):
        return None

    kind = str(strategy_metadata.get("k", strategy_metadata.get("path", "")))
    if kind not in _TINY_KIND_TO_CODE:
        return None

    raw_limit = _RAW_TINY_NBYTES_LIMIT
    if kind != "raw" or _looks_like_tiny_exact_role(name):
        raw_limit = _STRUCTURED_TINY_NBYTES_LIMIT
    if compressed.original_nbytes > raw_limit:
        return None

    dtype = np.dtype(header.dtype)
    expected_nbytes = int(np.prod(header.shape, dtype=np.int64)) * dtype.itemsize
    if compressed.original_nbytes != expected_nbytes:
        return None

    payload = bytes(compressed.payload)
    start = 0
    step = 0
    axis = 0
    if kind == "raw":
        if len(payload) != expected_nbytes:
            return None
    elif kind == "const":
        if len(payload) != dtype.itemsize:
            return None
    elif kind == "aseq":
        if payload:
            return None
        start = int(strategy_metadata["v"])
        step = int(strategy_metadata["p"])
    elif kind == "bseq":
        if payload:
            return None
        axis = int(strategy_metadata["a"])
        start = int(strategy_metadata["v"])
        step = int(strategy_metadata["p"])

    return TinyExactBundleEntry(
        name=name,
        tensor_type=header.tensor_type,
        strategy_id=header.strategy_id,
        checksum=header.checksum,
        dtype=dtype,
        shape=tuple(int(dim) for dim in header.shape),
        kind=kind,
        payload=payload,
        start=start,
        step=step,
        axis=axis,
    )


def encode_tiny_exact_bundle(entries: list[TinyExactBundleEntry]) -> tuple[bytes, bytes]:
    """Encode *entries* into bundle metadata bytes plus a shared payload blob."""
    if not entries:
        raise ValueError("tiny exact bundle requires at least one entry")

    metadata = bytearray(TINY_BUNDLE_MAGIC)
    metadata.append(TINY_BUNDLE_VERSION)
    metadata.extend(_encode_uvarint(len(entries)))

    payload = bytearray()
    for entry in entries:
        metadata.extend(_encode_entry(entry))
        payload.extend(entry.payload)
    return bytes(metadata), bytes(payload)


def decode_tiny_exact_bundle(
    metadata: bytes,
    *,
    payload_length: int,
) -> tuple[tuple[TinyExactBundleStoredEntry, ...], int]:
    """Decode bundle metadata into stored-entry descriptors.

    Returns the entries plus the fixed shared metadata bytes that should be
    amortized across them.
    """
    if not metadata.startswith(TINY_BUNDLE_MAGIC):
        raise ValueError("tiny exact bundle metadata magic mismatch")
    cursor = len(TINY_BUNDLE_MAGIC)
    if cursor >= len(metadata):
        raise ValueError("tiny exact bundle metadata is truncated")
    version = metadata[cursor]
    cursor += 1
    if version != TINY_BUNDLE_VERSION:
        raise ValueError(f"unsupported tiny exact bundle version: {version}")
    count, cursor = _decode_uvarint(metadata, cursor)
    shared_metadata_nbytes = cursor

    entries: list[TinyExactBundleStoredEntry] = []
    payload_used = 0
    for _ in range(count):
        descriptor_start = cursor
        name_len, cursor = _decode_uvarint(metadata, cursor)
        name = metadata[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len

        kind_code = metadata[cursor]
        cursor += 1
        kind = _TINY_CODE_TO_KIND.get(kind_code)
        if kind is None:
            raise ValueError(f"unsupported tiny exact entry kind code: {kind_code}")

        tensor_type = TensorType(metadata[cursor])
        cursor += 1
        strategy_id = int.from_bytes(metadata[cursor : cursor + 2], "little")
        cursor += 2
        checksum = int.from_bytes(metadata[cursor : cursor + 4], "little")
        cursor += 4
        dtype, cursor = _decode_dtype_token(metadata, cursor)

        ndim = metadata[cursor]
        cursor += 1
        shape: list[int] = []
        for _ in range(ndim):
            dim, cursor = _decode_uvarint(metadata, cursor)
            shape.append(dim)

        axis = 0
        start = 0
        step = 0
        if kind == "aseq":
            start, cursor = _decode_svarint(metadata, cursor)
            step, cursor = _decode_svarint(metadata, cursor)
        elif kind == "bseq":
            axis = metadata[cursor]
            cursor += 1
            start, cursor = _decode_svarint(metadata, cursor)
            step, cursor = _decode_svarint(metadata, cursor)

        payload_nbytes = _payload_nbytes(kind, dtype, tuple(shape))
        payload_used += payload_nbytes
        entries.append(
            TinyExactBundleStoredEntry(
                name=name,
                tensor_type=tensor_type,
                strategy_id=strategy_id,
                checksum=checksum,
                dtype=dtype,
                shape=tuple(shape),
                kind=kind,
                payload_length=payload_nbytes,
                descriptor_nbytes=(cursor - descriptor_start),
                start=start,
                step=step,
                axis=axis,
            )
        )

    if cursor != len(metadata):
        raise ValueError("tiny exact bundle metadata has trailing bytes")
    if payload_used != payload_length:
        raise ValueError(
            "tiny exact bundle payload size mismatch: "
            f"expected {payload_used}, got {payload_length}"
        )
    return tuple(entries), shared_metadata_nbytes


def distribute_shared_bytes(total: int, count: int) -> tuple[int, ...]:
    """Distribute *total* shared bytes deterministically across *count* entries."""
    if count <= 0:
        return ()
    base = total // count
    remainder = total % count
    return tuple(base + (1 if index < remainder else 0) for index in range(count))


def _encode_entry(entry: TinyExactBundleEntry) -> bytes:
    payload = bytearray()
    name_bytes = entry.name.encode("utf-8")
    payload.extend(_encode_uvarint(len(name_bytes)))
    payload.extend(name_bytes)
    payload.append(_TINY_KIND_TO_CODE[entry.kind])
    payload.append(int(entry.tensor_type))
    payload.extend(int(entry.strategy_id).to_bytes(2, "little"))
    payload.extend(int(entry.checksum).to_bytes(4, "little"))
    payload.extend(_encode_dtype_token(entry.dtype))
    payload.append(len(entry.shape))
    for dim in entry.shape:
        payload.extend(_encode_uvarint(int(dim)))
    if entry.kind == "aseq":
        payload.extend(_encode_svarint(entry.start))
        payload.extend(_encode_svarint(entry.step))
    elif entry.kind == "bseq":
        payload.append(int(entry.axis))
        payload.extend(_encode_svarint(entry.start))
        payload.extend(_encode_svarint(entry.step))
    return bytes(payload)


def _resolve_strategy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    strategy = metadata.get("strategy")
    if isinstance(strategy, dict):
        nested = strategy.get("metadata")
        if isinstance(nested, dict):
            return nested
    return metadata


def _is_lossless_strategy_metadata(metadata: dict[str, Any]) -> bool:
    if metadata.get("lossless") is True:
        return True
    return int(metadata.get("l", 0)) == 1


def _looks_like_tiny_exact_role(name: str) -> bool:
    lower = name.lower()
    return any(
        token in lower
        for token in (
            "bias",
            "norm",
            "position_ids",
            "token_type_ids",
            "type_ids",
        )
    )


def _payload_nbytes(kind: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> int:
    if kind == "raw":
        return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if kind == "const":
        return dtype.itemsize
    return 0


def _encode_dtype_token(dtype: np.dtype[Any]) -> bytes:
    token = str(np.dtype(dtype).str)
    code = _DTYPE_TO_CODE.get(token)
    if code is not None:
        return bytes([code])
    inline: bytes = token.encode("ascii")
    if len(inline) > 255:
        raise ValueError(f"dtype token is too long for tiny exact bundle: {token!r}")
    encoded: bytes = bytes([_INLINE_DTYPE_CODE, len(inline)]) + inline
    return encoded


def _decode_dtype_token(data: bytes, cursor: int) -> tuple[np.dtype[Any], int]:
    code = data[cursor]
    cursor += 1
    if code != _INLINE_DTYPE_CODE:
        return np.dtype(_DTYPE_CODES[code]), cursor
    length = data[cursor]
    cursor += 1
    token = data[cursor : cursor + length].decode("ascii")
    cursor += length
    return np.dtype(token), cursor


def _encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint requires a non-negative integer")
    payload = bytearray()
    remaining = int(value)
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            payload.append(byte | 0x80)
            continue
        payload.append(byte)
        return bytes(payload)


def _decode_uvarint(data: bytes, cursor: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise ValueError("malformed uvarint in tiny exact bundle")


def _encode_svarint(value: int) -> bytes:
    encoded = (abs(int(value)) << 1) - (1 if value < 0 else 0)
    return _encode_uvarint(encoded)


def _decode_svarint(data: bytes, cursor: int) -> tuple[int, int]:
    encoded, cursor = _decode_uvarint(data, cursor)
    value = (encoded >> 1) if (encoded & 1) == 0 else -((encoded + 1) >> 1)
    return value, cursor
