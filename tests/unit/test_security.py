"""Security tests for untrusted input handling."""

import struct

import pytest

from quench.codec.decoder import QuenchDecoder
from quench.core.exceptions import CodecError, HeaderError
from quench.core.types import CodecMode, CompressedTensor, TensorHeader, TensorType


def test_container_rejects_oversized_segment_metadata(tmp_path):
    """Segment with huge metadata_len must not attempt allocation."""
    from quench.io.container import (
        QNC_FLAG_COUNT_KNOWN,
        QNC_FLAG_STREAMED,
        QNC_MAGIC,
        QNC_VERSION_V2,
        QNCReader,
        SEGMENT_MAGIC,
        SEGMENT_TYPE_TENSOR,
        SEGMENT_VERSION,
        _HEADER_PREFIX,
        _HEADER_V2_REST,
        _SEGMENT_HEADER,
    )

    path = tmp_path / "evil.qnc"
    with path.open("wb") as handle:
        handle.write(_HEADER_PREFIX.pack(QNC_MAGIC, QNC_VERSION_V2))
        handle.write(_HEADER_V2_REST.pack(QNC_FLAG_STREAMED | QNC_FLAG_COUNT_KNOWN, 1))
        handle.write(
            _SEGMENT_HEADER.pack(
                SEGMENT_MAGIC,
                SEGMENT_VERSION,
                SEGMENT_TYPE_TENSOR,
                0,
                2**60,
                0,
            )
        )

    with pytest.raises(CodecError, match="safety limit"):
        list(QNCReader(path).iter_tensor_records())


def test_freq_model_rejects_oversized_symbol_count():
    """Frequency model with impossibly many symbols must not OOM."""
    from quench.entropy.freq_model import FrequencyModel

    data = (
        struct.pack("<I", 2_000_000_000)
        + struct.pack("<iI", 0, 1)
        + struct.pack("<iI", 1, 1)
    )

    with pytest.raises(ValueError, match="symbols"):
        FrequencyModel.deserialize(data)


def test_decoder_rejects_disallowed_dtype():
    """Headers with exotic dtypes like void must be rejected."""
    header = TensorHeader(
        tensor_type=TensorType.WEIGHT,
        dtype="V1024",
        shape=(10,),
        codec_mode=CodecMode.LOSSLESS,
    )
    compressed = CompressedTensor(
        header=header,
        payload=b"\x00" * 100,
        metadata=b'{"l":1,"k":"raw"}',
        original_nbytes=10240,
    )

    with pytest.raises(HeaderError, match="Disallowed dtype"):
        QuenchDecoder().decode(compressed)
