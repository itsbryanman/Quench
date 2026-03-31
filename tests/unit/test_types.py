"""Tests for core domain types."""
from __future__ import annotations

import pytest

from quench.core.types import (
    CodecMode,
    CompressedTensor,
    TensorHeader,
    TensorStats,
    TensorType,
)


class TestTensorType:
    def test_enum_values(self) -> None:
        assert TensorType.WEIGHT == 0
        assert TensorType.KV_CACHE == 1
        assert TensorType.EMBEDDING == 2
        assert TensorType.ACTIVATION == 3
        assert TensorType.OPTIMIZER_STATE == 4
        assert TensorType.UNKNOWN == 5
        assert TensorType.BIAS == 6
        assert TensorType.MIXED_PRECISION == 7
        assert TensorType.MASK == 8

    def test_all_members(self) -> None:
        assert len(TensorType) == 9


class TestTensorHeader:
    def test_construction(self) -> None:
        hdr = TensorHeader(
            tensor_type=TensorType.WEIGHT,
            dtype="float16",
            shape=(768, 3072),
            codec_mode=CodecMode.LOSSY,
        )
        assert hdr.magic == b"QNC1"
        assert hdr.version == 1
        assert hdr.tensor_type == TensorType.WEIGHT
        assert hdr.shape == (768, 3072)

    def test_frozen(self) -> None:
        hdr = TensorHeader(
            tensor_type=TensorType.WEIGHT,
            dtype="float16",
            shape=(10,),
            codec_mode=CodecMode.LOSSY,
        )
        with pytest.raises(AttributeError):
            hdr.version = 2  # type: ignore[misc]


class TestCompressedTensor:
    def _make_ct(self) -> CompressedTensor:
        hdr = TensorHeader(
            tensor_type=TensorType.WEIGHT,
            dtype="float16",
            shape=(100,),
            codec_mode=CodecMode.LOSSY,
            checksum=12345,
        )
        return CompressedTensor(
            header=hdr,
            payload=b"\x00" * 50,
            metadata=b"\x01" * 10,
            original_nbytes=200,
        )

    def test_compression_ratio(self) -> None:
        ct = self._make_ct()
        # compressed = 50 + 10 + 64 = 124
        assert ct.compressed_nbytes == 124
        assert ct.compression_ratio == pytest.approx(200 / 124, rel=1e-6)

    def test_to_bytes_from_bytes_roundtrip(self) -> None:
        ct = self._make_ct()
        blob = ct.to_bytes()
        restored = CompressedTensor.from_bytes(blob)
        assert restored.header == ct.header
        assert restored.payload == ct.payload
        assert restored.metadata == ct.metadata
        assert restored.original_nbytes == ct.original_nbytes


class TestTensorStats:
    def test_construction(self) -> None:
        stats = TensorStats(
            mean=0.0,
            std=1.0,
            min_val=-3.0,
            max_val=3.0,
            sparsity=0.05,
            entropy_bits=7.2,
        )
        assert stats.mean == 0.0
        assert stats.entropy_bits == 7.2
