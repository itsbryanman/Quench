"""Pure-Python backend implementations used as the default runtime path."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from quench.core.exceptions import BackendError
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import RANSDecoder, RANSEncoder


class PythonEntropyBackend:
    """Pure-Python entropy backend backed by the project's rANS implementation."""

    name = "python"

    def encode_symbols(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
        freq_model: FrequencyModel,
    ) -> bytes:
        array = np.asarray(symbols)
        return RANSEncoder(freq_model.freq_table).encode(array.reshape(-1).astype(np.int64, copy=False))

    def decode_symbols(
        self,
        data: bytes,
        freq_model: FrequencyModel,
        num_symbols: int,
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        return RANSDecoder(freq_model.freq_table).decode(data, num_symbols)


class PythonPackingBackend:
    """Deterministic pure-Python bit pack/unpack implementation."""

    name = "python"

    def pack_bits(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        signed: bool,
        layout_metadata: Mapping[str, Any] | None = None,
    ) -> bytes:
        if not (1 <= bits <= 32):
            raise BackendError(f"bits must be in [1, 32], got {bits}")

        array = np.asarray(symbols)
        if not np.issubdtype(array.dtype, np.integer):
            raise BackendError("pack_bits expects an integer tensor")

        mask = (1 << bits) - 1
        flat = array.reshape(-1).astype(np.int64, copy=False)
        out = bytearray()
        bit_buffer = 0
        bits_in_buffer = 0

        for value in flat:
            integer = int(value)
            if signed:
                encoded = integer & mask
            else:
                if integer < 0:
                    raise BackendError("Unsigned packing received a negative symbol")
                encoded = integer
            if encoded & ~mask:
                raise BackendError(f"Symbol {integer} does not fit in {bits} bits")

            bit_buffer |= encoded << bits_in_buffer
            bits_in_buffer += bits
            while bits_in_buffer >= 8:
                out.append(bit_buffer & 0xFF)
                bit_buffer >>= 8
                bits_in_buffer -= 8

        if bits_in_buffer:
            out.append(bit_buffer & 0xFF)
        return bytes(out)

    def unpack_bits(
        self,
        data: bytes,
        bits: int,
        signed: bool,
        shape: tuple[int, ...],
        layout_metadata: Mapping[str, Any] | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if not (1 <= bits <= 32):
            raise BackendError(f"bits must be in [1, 32], got {bits}")

        count = int(np.prod(shape, dtype=np.int64))
        if count < 0:
            raise BackendError("shape must describe a non-negative element count")
        mask = (1 << bits) - 1
        sign_bit = 1 << (bits - 1)
        bit_buffer = 0
        bits_in_buffer = 0
        index = 0
        output = np.empty(count, dtype=np.int64)

        for byte in data:
            bit_buffer |= int(byte) << bits_in_buffer
            bits_in_buffer += 8
            while bits_in_buffer >= bits and index < count:
                value = bit_buffer & mask
                bit_buffer >>= bits
                bits_in_buffer -= bits
                if signed and bits < 64 and (value & sign_bit):
                    value -= 1 << bits
                output[index] = value
                index += 1

        if index != count:
            raise BackendError(
                f"Packed payload ended after {index} symbols; expected {count}"
            )

        dtype_name = None if layout_metadata is None else layout_metadata.get("dtype")
        dtype = np.dtype(dtype_name) if dtype_name is not None else np.int64
        return output.astype(dtype, copy=False).reshape(shape)
