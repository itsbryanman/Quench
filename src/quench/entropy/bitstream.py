"""Low-level byte I/O utilities for entropy coding."""
from __future__ import annotations

import struct


class BitstreamWriter:
    """Accumulates bytes for writing compressed data."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def write_byte(self, b: int) -> None:
        self._buf.append(b & 0xFF)

    def write_bytes(self, data: bytes) -> None:
        self._buf.extend(data)

    def write_uint32(self, val: int) -> None:
        self._buf.extend(struct.pack("<I", val & 0xFFFFFFFF))

    def write_uint64(self, val: int) -> None:
        self._buf.extend(struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF))

    def getvalue(self) -> bytes:
        return bytes(self._buf)


class BitstreamReader:
    """Reads bytes sequentially from a buffer."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_byte(self) -> int:
        if self._pos >= len(self._data):
            raise EOFError("No more bytes to read")
        val = self._data[self._pos]
        self._pos += 1
        return val

    def read_bytes(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise EOFError(f"Cannot read {n} bytes, only {len(self._data) - self._pos} remaining")
        val = self._data[self._pos : self._pos + n]
        self._pos += n
        return val

    def read_uint32(self) -> int:
        raw = self.read_bytes(4)
        val: int = struct.unpack("<I", raw)[0]
        return val

    def read_uint64(self) -> int:
        raw = self.read_bytes(8)
        val: int = struct.unpack("<Q", raw)[0]
        return val

    def remaining(self) -> int:
        return len(self._data) - self._pos
