"""Binary header serialization for the QNC format."""
from __future__ import annotations

import struct

from quench.core.exceptions import HeaderError
from quench.core.types import CodecMode, TensorHeader, TensorType

HEADER_SIZE: int = 64
MAGIC: bytes = b"QNC1"
_MAX_DIMS: int = 8
_DTYPE_FIELD_LEN: int = 10

# Layout (64 bytes total):
#   0-3   : magic (4 bytes)
#   4-5   : version (uint16)
#   6     : tensor_type (uint8)
#   7     : codec_mode (uint8)
#   8     : ndim (uint8)
#   9     : dtype_len (uint8)
#  10-19  : dtype string (10 bytes, null-padded)
#  20-21  : strategy_id (uint16)
#  22-25  : checksum (uint32)
#  26-57  : shape — 8 x uint32 (32 bytes, zero-padded)
#  58-63  : reserved (6 bytes)

_HEADER_FMT = "<4s H BB BB 10s H I 8I 6s"
_HEADER_STRUCT = struct.Struct(_HEADER_FMT)

assert _HEADER_STRUCT.size == HEADER_SIZE, (
    f"Header struct size mismatch: {_HEADER_STRUCT.size} != {HEADER_SIZE}"
)


def encode_header(header: TensorHeader) -> bytes:
    """Pack a *TensorHeader* into exactly 64 bytes."""
    if header.magic != MAGIC:
        raise HeaderError(f"Invalid magic: {header.magic!r}")

    ndim = len(header.shape)
    if ndim > _MAX_DIMS:
        raise HeaderError(f"Too many dimensions ({ndim} > {_MAX_DIMS})")

    dtype_bytes = header.dtype.encode("ascii")
    if len(dtype_bytes) > _DTYPE_FIELD_LEN:
        raise HeaderError(f"dtype string too long ({len(dtype_bytes)} > {_DTYPE_FIELD_LEN})")
    dtype_padded = dtype_bytes.ljust(_DTYPE_FIELD_LEN, b"\x00")

    shape_padded = list(header.shape) + [0] * (_MAX_DIMS - ndim)

    return _HEADER_STRUCT.pack(
        header.magic,
        header.version,
        int(header.tensor_type),
        int(header.codec_mode),
        ndim,
        len(dtype_bytes),
        dtype_padded,
        header.strategy_id,
        header.checksum,
        *shape_padded,
        b"\x00" * 6,
    )


def decode_header(data: bytes) -> TensorHeader:
    """Unpack 64 bytes into a *TensorHeader*."""
    if len(data) < HEADER_SIZE:
        raise HeaderError(f"Header data too short ({len(data)} < {HEADER_SIZE})")

    fields = _HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    magic = fields[0]
    if magic != MAGIC:
        raise HeaderError(f"Magic mismatch: expected {MAGIC!r}, got {magic!r}")

    version = fields[1]
    tensor_type = TensorType(fields[2])
    codec_mode = CodecMode(fields[3])
    ndim = fields[4]
    dtype_len = fields[5]
    dtype_padded: bytes = fields[6]
    strategy_id = fields[7]
    checksum = fields[8]
    shape_vals = fields[9:17]
    # fields[17] is reserved

    dtype_str = dtype_padded[:dtype_len].decode("ascii")
    shape = tuple(shape_vals[:ndim])

    return TensorHeader(
        magic=magic,
        version=version,
        tensor_type=tensor_type,
        dtype=dtype_str,
        shape=shape,
        codec_mode=codec_mode,
        strategy_id=strategy_id,
        checksum=checksum,
    )
