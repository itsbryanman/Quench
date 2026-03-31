"""Tests for binary header serialization."""
from __future__ import annotations

import pytest

from quench.core.exceptions import HeaderError
from quench.core.header import HEADER_SIZE, decode_header, encode_header
from quench.core.types import CodecMode, TensorHeader, TensorType


class TestHeaderRoundtrip:
    def _roundtrip(self, hdr: TensorHeader) -> TensorHeader:
        data = encode_header(hdr)
        assert len(data) == HEADER_SIZE
        return decode_header(data)

    def test_basic(self) -> None:
        hdr = TensorHeader(
            tensor_type=TensorType.WEIGHT,
            dtype="float16",
            shape=(768, 3072),
            codec_mode=CodecMode.LOSSY,
            checksum=0xDEADBEEF,
        )
        restored = self._roundtrip(hdr)
        assert restored == hdr

    def test_1d_shape(self) -> None:
        hdr = TensorHeader(
            tensor_type=TensorType.EMBEDDING,
            dtype="float32",
            shape=(50257,),
            codec_mode=CodecMode.LOSSLESS,
        )
        assert self._roundtrip(hdr) == hdr

    def test_4d_shape(self) -> None:
        hdr = TensorHeader(
            tensor_type=TensorType.ACTIVATION,
            dtype="bfloat16",
            shape=(2, 12, 128, 64),
            codec_mode=CodecMode.LOSSY,
        )
        assert self._roundtrip(hdr) == hdr

    def test_various_dtypes(self) -> None:
        for dt in ("float16", "float32", "int8", "uint8", "bfloat16"):
            hdr = TensorHeader(
                tensor_type=TensorType.WEIGHT,
                dtype=dt,
                shape=(10,),
                codec_mode=CodecMode.LOSSY,
            )
            assert self._roundtrip(hdr).dtype == dt


class TestHeaderErrors:
    def test_magic_mismatch(self) -> None:
        bad_data = b"BAD!" + b"\x00" * 60
        with pytest.raises(HeaderError, match="Magic mismatch"):
            decode_header(bad_data)

    def test_data_too_short(self) -> None:
        with pytest.raises(HeaderError, match="too short"):
            decode_header(b"\x00" * 10)
