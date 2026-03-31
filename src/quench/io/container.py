"""Deterministic streamed QNC container support."""
from __future__ import annotations

import io
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO, Iterable, Iterator, Sequence

from quench.codec.metadata import deserialize_metadata, serialize_metadata
from quench.core.exceptions import CodecError
from quench.core.header import decode_header, encode_header
from quench.core.types import CompressedTensor, TensorHeader
from quench.io.tiny_bundle import (
    TinyExactBundleEntry,
    decode_tiny_exact_bundle,
    distribute_shared_bytes,
    encode_tiny_exact_bundle,
    try_build_tiny_exact_bundle_entry,
)

QNC_MAGIC = b"QNCB"
QNC_VERSION_V1 = 1
QNC_VERSION_V2 = 2
QNC_VERSION_V3 = 3
QNC_FLAG_STREAMED = 0x01
QNC_FLAG_COUNT_KNOWN = 0x02
QNC_UNKNOWN_TENSOR_COUNT = 0xFFFFFFFF

SEGMENT_MAGIC = b"QNCS"
SEGMENT_VERSION = 1
SEGMENT_TYPE_TENSOR = 1
SEGMENT_TYPE_CHUNK = 2
SEGMENT_TYPE_TINY_EXACT_BUNDLE = 3

_HEADER_PREFIX = struct.Struct("<4sH")
_HEADER_V1_REST = struct.Struct("<I")
_HEADER_V2_REST = struct.Struct("<HI")
_SEGMENT_HEADER = struct.Struct("<4sBBHQQ")


@dataclass(frozen=True)
class PayloadChunkRef:
    """Seekable reference to a payload chunk inside a QNC file."""

    offset: int
    length: int


@dataclass(frozen=True)
class TensorRecord:
    """Descriptor for a streamed tensor record inside a QNC container."""

    source_path: Path
    name: str
    header: TensorHeader
    metadata: bytes
    original_nbytes: int
    chunk_refs: tuple[PayloadChunkRef, ...]
    version: int
    stream_flags: int
    storage_nbytes: int = 0
    storage_payload_nbytes: int = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_refs)

    @property
    def payload_nbytes(self) -> int:
        return sum(chunk.length for chunk in self.chunk_refs)

    @property
    def storage_overhead_nbytes(self) -> int:
        return self.storage_nbytes - self.storage_payload_nbytes

    def iter_payload_chunks(self) -> Iterator[bytes]:
        """Yield payload chunks without loading the full tensor payload at once."""
        with self.source_path.open("rb") as handle:
            for chunk in self.chunk_refs:
                handle.seek(chunk.offset)
                yield _read_exact(handle, chunk.length, f"payload chunk for {self.name}")

    def to_compressed_tensor(self) -> CompressedTensor:
        """Materialize the streamed record as a :class:`CompressedTensor`."""
        return CompressedTensor(
            header=self.header,
            payload=b"".join(self.iter_payload_chunks()),
            metadata=self.metadata,
            original_nbytes=self.original_nbytes,
        )


class QNCWriter:
    """Stream tensor records into deterministic QNC version-2 or version-3 containers."""

    def __init__(
        self,
        path: str | Path,
        *,
        tensor_count: int | None = None,
        chunk_size: int = 1 << 20,
        version: int = QNC_VERSION_V2,
        enable_tiny_exact_bundle: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if version not in {QNC_VERSION_V2, QNC_VERSION_V3}:
            raise ValueError(f"Unsupported QNC writer version: {version}")
        if enable_tiny_exact_bundle and version != QNC_VERSION_V3:
            raise ValueError("Tiny exact bundle support requires QNC version 3")
        self._path = Path(path)
        self._tensor_count_hint = tensor_count
        self._chunk_size = chunk_size
        self._version = version
        self._enable_tiny_exact_bundle = enable_tiny_exact_bundle
        self._handle: BinaryIO | None = None
        self._written = 0
        self._pending_tiny_exact_bundle: list[TinyExactBundleEntry] = []

    def __enter__(self) -> QNCWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("wb")
        flags = QNC_FLAG_STREAMED
        count = QNC_UNKNOWN_TENSOR_COUNT
        if self._tensor_count_hint is not None:
            flags |= QNC_FLAG_COUNT_KNOWN
            count = self._tensor_count_hint
        self._handle.write(_HEADER_PREFIX.pack(QNC_MAGIC, self._version))
        self._handle.write(_HEADER_V2_REST.pack(flags, count))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._handle is not None:
            if exc_type is None:
                self._flush_pending_tiny_exact_bundle()
                self._finalize_header()
            self._handle.close()
            self._handle = None

    def write_tensor_record(
        self,
        name: str,
        header: TensorHeader,
        metadata: bytes,
        payload_chunks: Iterable[bytes],
        *,
        original_nbytes: int,
    ) -> None:
        """Write one tensor record and its payload chunks."""
        self._flush_pending_tiny_exact_bundle()
        handle = self._require_handle()
        record_id = self._written
        chunk_lengths, spool = _spool_chunks(payload_chunks)
        try:
            record_metadata = serialize_metadata(
                {
                    "chunk_count": len(chunk_lengths),
                    "chunk_lengths": chunk_lengths,
                    "header": encode_header(header),
                    "metadata": metadata,
                    "name": name,
                    "original_nbytes": original_nbytes,
                    "record_id": record_id,
                }
            )
            self._write_segment(
                segment_type=SEGMENT_TYPE_TENSOR,
                metadata=record_metadata,
                payload=b"",
            )
            spool.seek(0)
            for index, length in enumerate(chunk_lengths):
                payload = spool.read(length)
                if len(payload) != length:
                    raise CodecError(f"Failed to read spooled payload chunk {index} for {name}")
                chunk_metadata = serialize_metadata(
                    {
                        "index": index,
                        "length": length,
                        "record_id": record_id,
                    }
                )
                self._write_segment(
                    segment_type=SEGMENT_TYPE_CHUNK,
                    metadata=chunk_metadata,
                    payload=payload,
                )
        finally:
            spool.close()

        self._written += 1
        if self._tensor_count_hint is not None and self._written > self._tensor_count_hint:
            raise CodecError("Wrote more tensor records than the declared tensor_count")

    def write_compressed_tensor(
        self,
        name: str,
        compressed: CompressedTensor,
        *,
        chunk_size: int | None = None,
    ) -> None:
        """Write a materialized :class:`CompressedTensor` as a chunked record."""
        if self._enable_tiny_exact_bundle:
            candidate = try_build_tiny_exact_bundle_entry(name, compressed)
            if candidate is not None:
                self._pending_tiny_exact_bundle.append(candidate)
                return
            self._flush_pending_tiny_exact_bundle()

        active_chunk_size = self._chunk_size if chunk_size is None else chunk_size
        self.write_tensor_record(
            name,
            compressed.header,
            compressed.metadata,
            _chunk_bytes(compressed.payload, active_chunk_size),
            original_nbytes=compressed.original_nbytes,
        )

    def _flush_pending_tiny_exact_bundle(self) -> None:
        if not self._pending_tiny_exact_bundle:
            return
        if len(self._pending_tiny_exact_bundle) == 1:
            entry = self._pending_tiny_exact_bundle.pop()
            payload_chunks: Iterable[bytes] = ()
            if entry.payload:
                payload_chunks = (entry.payload,)
            self.write_tensor_record(
                entry.name,
                entry.header,
                entry.metadata_bytes,
                payload_chunks,
                original_nbytes=entry.original_nbytes,
            )
            return
        metadata, payload = encode_tiny_exact_bundle(self._pending_tiny_exact_bundle)
        self._write_segment(
            segment_type=SEGMENT_TYPE_TINY_EXACT_BUNDLE,
            metadata=metadata,
            payload=payload,
        )
        self._written += len(self._pending_tiny_exact_bundle)
        if self._tensor_count_hint is not None and self._written > self._tensor_count_hint:
            raise CodecError("Wrote more tensor records than the declared tensor_count")
        self._pending_tiny_exact_bundle.clear()

    def _write_segment(self, *, segment_type: int, metadata: bytes, payload: bytes) -> None:
        handle = self._require_handle()
        handle.write(
            _SEGMENT_HEADER.pack(
                SEGMENT_MAGIC,
                SEGMENT_VERSION,
                segment_type,
                0,
                len(metadata),
                len(payload),
            )
        )
        handle.write(metadata)
        handle.write(payload)

    def _finalize_header(self) -> None:
        handle = self._require_handle()
        if self._tensor_count_hint is not None and self._written != self._tensor_count_hint:
            raise CodecError(
                f"Expected to write {self._tensor_count_hint} tensors, wrote {self._written}"
            )
        flags = QNC_FLAG_STREAMED | QNC_FLAG_COUNT_KNOWN
        handle.seek(_HEADER_PREFIX.size)
        handle.write(_HEADER_V2_REST.pack(flags, self._written))
        handle.seek(0, io.SEEK_END)

    def _require_handle(self) -> BinaryIO:
        if self._handle is None:
            raise CodecError("QNCWriter must be used as a context manager")
        return self._handle


class QNCReader:
    """Read streamed QNC containers while preserving incremental payload access."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def iter_tensor_records(self) -> Iterator[TensorRecord]:
        """Yield tensor records in on-disk order without loading all payloads eagerly."""
        with self._path.open("rb") as handle:
            magic, version = _HEADER_PREFIX.unpack(_read_exact(handle, _HEADER_PREFIX.size, "QNC prefix"))
            if magic != QNC_MAGIC:
                raise CodecError(f"QNC bundle magic mismatch: expected {QNC_MAGIC!r}, got {magic!r}")

            if version == QNC_VERSION_V1:
                yield from self._iter_v1_records(handle)
                return
            if version not in {QNC_VERSION_V2, QNC_VERSION_V3}:
                raise CodecError(
                    "Unsupported QNC bundle version: expected "
                    f"{QNC_VERSION_V1}, {QNC_VERSION_V2}, or {QNC_VERSION_V3}, got {version}"
                )

            flags, tensor_count = _HEADER_V2_REST.unpack(
                _read_exact(handle, _HEADER_V2_REST.size, "QNC v2 header")
            )
            yielded = 0
            while True:
                header_bytes = handle.read(_SEGMENT_HEADER.size)
                if not header_bytes:
                    break
                if len(header_bytes) != _SEGMENT_HEADER.size:
                    raise CodecError("Truncated QNC segment header")
                segment = _SEGMENT_HEADER.unpack(header_bytes)
                if segment[0] != SEGMENT_MAGIC:
                    raise CodecError("QNC segment magic mismatch")
                if segment[1] != SEGMENT_VERSION:
                    raise CodecError("Unsupported QNC segment version")
                segment_type = int(segment[2])
                metadata_len = int(segment[4])
                payload_len = int(segment[5])
                segment_nbytes = _SEGMENT_HEADER.size + metadata_len + payload_len

                if segment_type == SEGMENT_TYPE_TENSOR:
                    record_metadata = deserialize_metadata(
                        _read_exact(handle, metadata_len, "tensor record metadata")
                    )
                    if payload_len:
                        handle.seek(payload_len, io.SEEK_CUR)
                    yield self._read_v2_record(handle, record_metadata, flags, segment_nbytes, version)
                    yielded += 1
                    continue

                if version == QNC_VERSION_V3 and segment_type == SEGMENT_TYPE_TINY_EXACT_BUNDLE:
                    bundle_metadata = _read_exact(handle, metadata_len, "tiny exact bundle metadata")
                    payload_offset = handle.tell()
                    if payload_len:
                        handle.seek(payload_len, io.SEEK_CUR)
                    for record in self._read_v3_tiny_exact_bundle_records(
                        bundle_metadata,
                        payload_offset=payload_offset,
                        payload_len=payload_len,
                        flags=flags,
                        segment_nbytes=segment_nbytes,
                    ):
                        yield record
                        yielded += 1
                    continue

                raise CodecError(f"Expected tensor record segment, found type {segment_type}")
            if tensor_count != QNC_UNKNOWN_TENSOR_COUNT and yielded != tensor_count:
                raise CodecError(f"QNC tensor count mismatch: expected {tensor_count}, found {yielded}")

    def _iter_v1_records(self, handle: BinaryIO) -> Iterator[TensorRecord]:
        """Read Phase-3 version-1 bundles with eager per-record payload blobs."""
        (tensor_count,) = _HEADER_V1_REST.unpack(_read_exact(handle, _HEADER_V1_REST.size, "v1 tensor count"))
        name_length = struct.Struct("<I")
        header_length = struct.Struct("<I")
        blob_length = struct.Struct("<Q")

        for _ in range(tensor_count):
            (name_len,) = name_length.unpack(_read_exact(handle, name_length.size, "tensor name length"))
            name = _read_exact(handle, name_len, "tensor name").decode("utf-8")

            (header_len,) = header_length.unpack(_read_exact(handle, header_length.size, "tensor header length"))
            header = decode_header(_read_exact(handle, header_len, "tensor header"))

            (metadata_len,) = blob_length.unpack(_read_exact(handle, blob_length.size, "metadata length"))
            metadata = _read_exact(handle, metadata_len, "tensor metadata")

            (payload_len,) = blob_length.unpack(_read_exact(handle, blob_length.size, "payload length"))
            payload_offset = handle.tell()
            handle.seek(payload_len, io.SEEK_CUR)

            yield TensorRecord(
                source_path=self._path,
                name=name,
                header=header,
                metadata=metadata,
                original_nbytes=_header_nbytes(header.dtype, header.shape),
                chunk_refs=(PayloadChunkRef(payload_offset, payload_len),),
                version=QNC_VERSION_V1,
                stream_flags=0,
                storage_nbytes=(
                    name_length.size
                    + name_len
                    + header_length.size
                    + header_len
                    + blob_length.size
                    + metadata_len
                    + blob_length.size
                    + payload_len
                ),
                storage_payload_nbytes=payload_len,
            )

        trailing = handle.read(1)
        if trailing:
            raise CodecError("QNC bundle has trailing bytes after the last tensor")

    def _read_v2_record(
        self,
        handle: BinaryIO,
        record_metadata: dict[str, Any],
        flags: int,
        segment_nbytes: int,
        version: int,
    ) -> TensorRecord:
        """Read one v2 tensor record plus its referenced chunk segments."""
        try:
            name = str(record_metadata["name"])
            header = decode_header(bytes(record_metadata["header"]))
            metadata = bytes(record_metadata["metadata"])
            original_nbytes = int(record_metadata["original_nbytes"])
            record_id = int(record_metadata["record_id"])
            chunk_count = int(record_metadata["chunk_count"])
            chunk_lengths = tuple(int(length) for length in record_metadata["chunk_lengths"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CodecError(f"Malformed v2 tensor record metadata: {exc}") from exc

        if len(chunk_lengths) != chunk_count:
            raise CodecError(
                f"Tensor record {name} declared {chunk_count} chunks but stored {len(chunk_lengths)} lengths"
            )

        chunk_refs: list[PayloadChunkRef] = []
        storage_nbytes = segment_nbytes
        for expected_index, expected_length in enumerate(chunk_lengths):
            segment = _SEGMENT_HEADER.unpack(
                _read_exact(handle, _SEGMENT_HEADER.size, f"chunk segment header for {name}")
            )
            if segment[0] != SEGMENT_MAGIC or segment[1] != SEGMENT_VERSION:
                raise CodecError("Malformed QNC chunk segment header")
            if segment[2] != SEGMENT_TYPE_CHUNK:
                raise CodecError(f"Expected chunk segment for {name}, found type {segment[2]}")
            metadata_len = int(segment[4])
            payload_len = int(segment[5])
            storage_nbytes += _SEGMENT_HEADER.size + metadata_len + payload_len
            chunk_metadata = deserialize_metadata(
                _read_exact(handle, metadata_len, f"chunk metadata for {name}")
            )
            try:
                if int(chunk_metadata["record_id"]) != record_id:
                    raise CodecError(f"Chunk record id mismatch for {name}")
                if int(chunk_metadata["index"]) != expected_index:
                    raise CodecError(f"Chunk index mismatch for {name}")
                if int(chunk_metadata["length"]) != expected_length:
                    raise CodecError(f"Chunk length metadata mismatch for {name}")
            except (KeyError, TypeError, ValueError) as exc:
                raise CodecError(f"Malformed chunk metadata for {name}: {exc}") from exc
            if payload_len != expected_length:
                raise CodecError(
                    f"Chunk payload length mismatch for {name}: expected {expected_length}, got {payload_len}"
                )
            payload_offset = handle.tell()
            handle.seek(payload_len, io.SEEK_CUR)
            chunk_refs.append(PayloadChunkRef(payload_offset, payload_len))

        return TensorRecord(
            source_path=self._path,
            name=name,
            header=header,
            metadata=metadata,
            original_nbytes=original_nbytes,
            chunk_refs=tuple(chunk_refs),
            version=version,
            stream_flags=flags,
            storage_nbytes=storage_nbytes,
            storage_payload_nbytes=sum(chunk_lengths),
        )

    def _read_v3_tiny_exact_bundle_records(
        self,
        bundle_metadata: bytes,
        *,
        payload_offset: int,
        payload_len: int,
        flags: int,
        segment_nbytes: int,
    ) -> tuple[TensorRecord, ...]:
        try:
            entries, shared_metadata_nbytes = decode_tiny_exact_bundle(
                bundle_metadata,
                payload_length=payload_len,
            )
        except (ValueError, IndexError) as exc:
            raise CodecError(f"Malformed tiny exact bundle metadata: {exc}") from exc

        shared_bytes = distribute_shared_bytes(
            _SEGMENT_HEADER.size + shared_metadata_nbytes,
            len(entries),
        )

        records: list[TensorRecord] = []
        cursor = payload_offset
        accounted_nbytes = 0
        for index, entry in enumerate(entries):
            chunk_refs: tuple[PayloadChunkRef, ...]
            if entry.payload_length > 0:
                chunk_refs = (PayloadChunkRef(cursor, entry.payload_length),)
                cursor += entry.payload_length
            else:
                chunk_refs = ()
            storage_nbytes = entry.descriptor_nbytes + entry.payload_length + shared_bytes[index]
            accounted_nbytes += storage_nbytes
            records.append(
                TensorRecord(
                    source_path=self._path,
                    name=entry.name,
                    header=entry.header,
                    metadata=entry.metadata_bytes,
                    original_nbytes=entry.original_nbytes,
                    chunk_refs=chunk_refs,
                    version=QNC_VERSION_V3,
                    stream_flags=flags,
                    storage_nbytes=storage_nbytes,
                    storage_payload_nbytes=entry.payload_length,
                )
            )

        if accounted_nbytes != segment_nbytes:
            raise CodecError(
                "Tiny exact bundle accounting mismatch: "
                f"expected {segment_nbytes}, got {accounted_nbytes}"
            )
        return tuple(records)


def write_tensor_record(
    writer: QNCWriter,
    name: str,
    compressed: CompressedTensor,
    *,
    chunk_size: int | None = None,
) -> None:
    """Write a compressed tensor through an existing :class:`QNCWriter`."""
    writer.write_compressed_tensor(name, compressed, chunk_size=chunk_size)


def iter_tensor_records(path: str | Path) -> Iterator[TensorRecord]:
    """Iterate tensor records from a QNC container."""
    yield from QNCReader(path).iter_tensor_records()


def _read_exact(handle: BinaryIO, length: int, label: str) -> bytes:
    data = handle.read(length)
    if len(data) != length:
        raise CodecError(f"Unexpected EOF while reading {label}")
    return data


def _header_nbytes(dtype_name: str, shape: tuple[int, ...]) -> int:
    import numpy as np

    dtype = np.dtype(dtype_name)
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def _chunk_bytes(data: bytes, chunk_size: int) -> Iterator[bytes]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]
    if not data:
        yield b""


def _spool_chunks(payload_chunks: Iterable[bytes]) -> tuple[list[int], SpooledTemporaryFile[bytes]]:
    spool: SpooledTemporaryFile[bytes] = SpooledTemporaryFile(max_size=8 << 20)
    chunk_lengths: list[int] = []
    for chunk in payload_chunks:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise CodecError(f"Payload chunks must be bytes-like, got {type(chunk)!r}")
        data = bytes(chunk)
        chunk_lengths.append(len(data))
        spool.write(data)
    return chunk_lengths, spool
