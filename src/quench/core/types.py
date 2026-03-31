"""Core domain types for Quench."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from quench.core.exceptions import CodecError


class TensorType(IntEnum):
    """Classification of tensor purpose."""

    WEIGHT = 0
    KV_CACHE = 1
    EMBEDDING = 2
    ACTIVATION = 3
    OPTIMIZER_STATE = 4
    UNKNOWN = 5
    BIAS = 6
    MIXED_PRECISION = 7
    MASK = 8


class QuantMode(IntEnum):
    """Quantization mode."""

    SYMMETRIC = 0
    ASYMMETRIC = 1
    NONE = 2


class CodecMode(IntEnum):
    """Compression fidelity mode."""

    LOSSY = 0
    LOSSLESS = 1


@dataclass(frozen=True)
class TensorHeader:
    """Fixed-size header for a compressed tensor blob."""

    tensor_type: TensorType
    dtype: str
    shape: tuple[int, ...]
    codec_mode: CodecMode
    magic: bytes = b"QNC1"
    version: int = 1
    strategy_id: int = 0
    checksum: int = 0


@dataclass
class CompressedTensor:
    """A compressed tensor with header, payload, and metadata."""

    header: TensorHeader
    payload: bytes
    metadata: bytes
    original_nbytes: int

    _HEADER_WIRE_SIZE: ClassVar[int] = 64

    @property
    def compressed_nbytes(self) -> int:
        return len(self.payload) + len(self.metadata) + self._HEADER_WIRE_SIZE

    @property
    def compression_ratio(self) -> float:
        cn = self.compressed_nbytes
        if cn == 0:
            return 0.0
        return self.original_nbytes / cn

    # ---- serialization ----

    def to_bytes(self) -> bytes:
        """Serialize to a self-contained byte blob."""
        from quench.core.header import encode_header

        hdr_bytes = encode_header(self.header)
        meta_len = len(self.metadata)
        payload_len = len(self.payload)
        # 8 bytes: original_nbytes (uint64)
        # 4 bytes: meta_len (uint32)
        # 4 bytes: payload_len (uint32)
        prefix = struct.pack("<QII", self.original_nbytes, meta_len, payload_len)
        return hdr_bytes + prefix + self.metadata + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> CompressedTensor:
        """Deserialize from a byte blob produced by *to_bytes*."""
        from quench.core.header import HEADER_SIZE, decode_header

        if len(data) < HEADER_SIZE + 16:
            raise CodecError("Data too short for CompressedTensor")

        header = decode_header(data[:HEADER_SIZE])
        offset = HEADER_SIZE
        original_nbytes, meta_len, payload_len = struct.unpack_from("<QII", data, offset)
        offset += 16
        expected_len = offset + meta_len + payload_len
        if len(data) != expected_len:
            raise CodecError(
                "CompressedTensor length mismatch: "
                f"expected {expected_len} bytes, got {len(data)}"
            )
        metadata = data[offset : offset + meta_len]
        offset += meta_len
        payload = data[offset : offset + payload_len]
        return cls(
            header=header,
            payload=payload,
            metadata=metadata,
            original_nbytes=original_nbytes,
        )


@dataclass
class TensorStats:
    """Descriptive statistics for a tensor."""

    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float  # fraction of values with abs < 1e-6
    entropy_bits: float  # estimated Shannon entropy per element
    effective_rank: float | None = None  # Shannon effective rank for 2D tensors
