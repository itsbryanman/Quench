"""Tests for streamed QNC container helpers."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from quench.codec import QuenchEncoder
from quench.core.config import QuenchConfig
from quench.core.header import encode_header
from quench.core.types import CodecMode, QuantMode
from quench.io import QNCReader, QNCWriter


def test_streamed_container_roundtrip_and_chunk_iteration(tmp_path: Path) -> None:
    tensor = np.arange(2048, dtype=np.float32).reshape(128, 16)
    compressed = QuenchEncoder(config=QuenchConfig(target_bits=4)).encode(tensor, name="mlp.weight")
    path = tmp_path / "chunked.qnc"

    with QNCWriter(path, tensor_count=1, chunk_size=64) as writer:
        writer.write_compressed_tensor("mlp.weight", compressed, chunk_size=64)

    records = list(QNCReader(path).iter_tensor_records())
    assert len(records) == 1
    record = records[0]
    assert record.name == "mlp.weight"
    assert record.chunk_count > 1
    chunk_sizes = [len(chunk) for chunk in record.iter_payload_chunks()]
    assert max(chunk_sizes) <= 64
    assert record.to_compressed_tensor().payload == compressed.payload


def test_reader_loads_phase3_v1_bundles(tmp_path: Path) -> None:
    tensor = np.arange(32, dtype=np.float32)
    compressed = QuenchEncoder(
        config=QuenchConfig(codec_mode=CodecMode.LOSSLESS, quant_mode=QuantMode.NONE)
    ).encode(tensor, name="legacy")
    path = tmp_path / "legacy.qnc"

    bundle_header = struct.Struct("<4sHI")
    name_length = struct.Struct("<I")
    header_length = struct.Struct("<I")
    blob_length = struct.Struct("<Q")
    name = b"legacy"
    header_bytes = encode_header(compressed.header)
    with path.open("wb") as handle:
        handle.write(bundle_header.pack(b"QNCB", 1, 1))
        handle.write(name_length.pack(len(name)))
        handle.write(name)
        handle.write(header_length.pack(len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob_length.pack(len(compressed.metadata)))
        handle.write(compressed.metadata)
        handle.write(blob_length.pack(len(compressed.payload)))
        handle.write(compressed.payload)

    records = list(QNCReader(path).iter_tensor_records())
    assert len(records) == 1
    assert records[0].name == "legacy"
    assert records[0].to_compressed_tensor().payload == compressed.payload
