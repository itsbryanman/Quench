"""High-level streamed encode/decode helpers for QNC bundles."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, TypeAlias

import numpy as np

from quench.codec.decoder import QuenchDecoder
from quench.codec.encoder import QuenchEncoder
from quench.core.config import QuenchConfig
from quench.core.types import CompressedTensor
from quench.io.container import QNCReader, QNCWriter, TensorRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


TensorLike: TypeAlias = np.ndarray[Any, np.dtype[Any]] | Any


def encode_tensor_stream(
    path: str | Path,
    tensors: Iterable[tuple[str, TensorLike]],
    *,
    config: QuenchConfig | None = None,
    tensor_count: int | None = None,
    chunk_size: int = 1 << 20,
) -> None:
    """Encode a tensor stream into a chunked QNC bundle."""
    encoder = QuenchEncoder(config=config)
    with QNCWriter(path, tensor_count=tensor_count, chunk_size=chunk_size) as writer:
        for name, tensor in tensors:
            compressed = encoder.encode(tensor, name=name)
            writer.write_compressed_tensor(name, compressed, chunk_size=chunk_size)


def decode_tensor_stream(
    path: str | Path,
    *,
    config: QuenchConfig | None = None,
) -> Iterator[tuple[str, np.ndarray[Any, np.dtype[Any]]]]:
    """Decode a QNC bundle incrementally without loading all payloads at once."""
    decoder = QuenchDecoder(config=config)
    for record in QNCReader(path).iter_tensor_records():
        yield record.name, decoder.decode(record.to_compressed_tensor())


def iter_compressed_tensors(path: str | Path) -> Iterator[tuple[str, CompressedTensor]]:
    """Yield compressed tensors from a QNC container one record at a time."""
    for record in QNCReader(path).iter_tensor_records():
        yield record.name, record.to_compressed_tensor()


def iter_tensor_records_stream(path: str | Path) -> Iterator[TensorRecord]:
    """Yield streamed tensor record descriptors from *path*."""
    yield from QNCReader(path).iter_tensor_records()
