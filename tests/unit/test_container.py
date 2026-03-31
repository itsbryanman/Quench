"""Tests for streamed QNC container helpers."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from quench.codec import QuenchEncoder
from quench.core.config import QuenchConfig
from quench.core.header import encode_header
from quench.core.types import CodecMode, QuantMode
from quench.integrations import load_compressed, save_compressed, save_compressed_bundle
from quench.io import QNCReader, QNCWriter
from quench.io.container import QNC_VERSION_V2, QNC_VERSION_V3, SEGMENT_TYPE_TENSOR, SEGMENT_TYPE_TINY_EXACT_BUNDLE


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


def test_save_compressed_bundles_mixed_tiny_exact_entries(tmp_path: Path) -> None:
    config = QuenchConfig(target_bits=4)
    tensors = {
        "layer.bias": np.arange(16, dtype=np.float32),
        "layer.constant": np.ones((32,), dtype=np.float32),
        "position_ids": np.arange(64, dtype=np.int64).reshape(1, 64),
        "repeated_ids": np.broadcast_to(np.arange(16, dtype=np.int64), (4, 16)).copy(),
    }
    path = tmp_path / "tiny-bundle.qnc"

    save_compressed(path, tensors, config=config)
    restored = load_compressed(path, config=config)
    records = list(QNCReader(path).iter_tensor_records())

    assert _container_version(path) == QNC_VERSION_V3
    assert _segment_types(path) == [SEGMENT_TYPE_TINY_EXACT_BUNDLE]
    assert len(records) == len(tensors)
    assert sum(record.storage_nbytes for record in records) + _container_header_nbytes() == path.stat().st_size
    for name, original in tensors.items():
        np.testing.assert_array_equal(restored[name], original)


def test_save_compressed_bundle_bundles_precompressed_exact_entries(tmp_path: Path) -> None:
    config = QuenchConfig(target_bits=4)
    encoder = QuenchEncoder(config=config)
    tensors = {
        "bias": encoder.encode(np.arange(8, dtype=np.float32), name="bias"),
        "constant": encoder.encode(np.ones((16,), dtype=np.float32), name="constant"),
        "position_ids": encoder.encode(np.arange(32, dtype=np.int64).reshape(1, 32), name="position_ids"),
    }
    path = tmp_path / "precompressed.qnc"

    save_compressed_bundle(path, tensors)

    assert _container_version(path) == QNC_VERSION_V3
    assert _segment_types(path) == [SEGMENT_TYPE_TINY_EXACT_BUNDLE]


def test_save_compressed_without_eligible_tiny_exacts_keeps_regular_v2_path(tmp_path: Path) -> None:
    config = QuenchConfig(target_bits=4)
    rng = np.random.default_rng(909)
    tensors = {
        "mlp.weight": rng.normal(size=(128, 128)).astype(np.float32),
        "token_embed.weight": rng.normal(size=(256, 96)).astype(np.float32),
    }
    path = tmp_path / "regular.qnc"

    save_compressed(path, tensors, config=config)

    assert _container_version(path) == QNC_VERSION_V2
    assert SEGMENT_TYPE_TINY_EXACT_BUNDLE not in _segment_types(path)
    assert SEGMENT_TYPE_TENSOR in _segment_types(path)


def _container_version(path: Path) -> int:
    prefix = struct.Struct("<4sH")
    with path.open("rb") as handle:
        magic, version = prefix.unpack(handle.read(prefix.size))
    assert magic == b"QNCB"
    return version


def _segment_types(path: Path) -> list[int]:
    prefix = struct.Struct("<4sH")
    rest = struct.Struct("<HI")
    segment = struct.Struct("<4sBBHQQ")

    types: list[int] = []
    with path.open("rb") as handle:
        handle.read(prefix.size)
        handle.read(rest.size)
        while True:
            header = handle.read(segment.size)
            if not header:
                break
            magic, version, segment_type, _flags, metadata_len, payload_len = segment.unpack(header)
            assert magic == b"QNCS"
            assert version == 1
            types.append(int(segment_type))
            handle.seek(int(metadata_len) + int(payload_len), 1)
    return types


def _container_header_nbytes() -> int:
    return struct.calcsize("<4sH") + struct.calcsize("<HI")
