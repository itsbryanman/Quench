"""rANS (range Asymmetric Numeral Systems) entropy coder.

This implements a byte-aligned streaming rANS codec with a 64-bit state.
The encoder processes symbols in reverse order and the decoder processes
them forward, yielding an exact inverse.

State invariant: the rANS state *x* is kept in [RANS_L, RANS_L * 256),
i.e. [2^56, 2^64). Renormalization emits or consumes one byte at a time.

The normalised frequency table **must** sum to ``SCALE = 1 << SCALE_BITS``.

References:
    - Duda, "Asymmetric numeral systems" (2009)
    - Giesen, "Interleaved entropy coders" (2014)
    - ryg, "rans_byte.h" reference implementation
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np

from quench.core.exceptions import EntropyError

# ---- constants ---------------------------------------------------------------

SCALE_BITS: int = 16
SCALE: int = 1 << SCALE_BITS  # frequency table must sum to this

# State lives in [RANS_L, RANS_L * 256) = [2^56, 2^64).
# Byte-aligned: each renorm step emits/consumes 8 bits.
RANS_L: int = 1 << 56

# Precomputed shift for x_max: x_max(freq) = freq << X_MAX_SHIFT
# Derived from: need  (x // freq) * SCALE < 2^64
#   ⇒  x < freq * 2^(64 - SCALE_BITS) = freq << 48
_X_MAX_SHIFT: int = 64 - SCALE_BITS  # 48


# ---- helpers -----------------------------------------------------------------


def build_freq_table(symbols: np.ndarray[Any, np.dtype[Any]]) -> dict[int, int]:
    """Count occurrences of each unique symbol in *symbols*."""
    unique, counts = np.unique(symbols, return_counts=True)
    return {int(s): int(c) for s, c in zip(unique, counts)}


def normalize_freq_table(
    freq: dict[int, int], target_total: int = SCALE
) -> dict[int, int]:
    """Scale frequencies so they sum to *target_total*, preserving every symbol (freq >= 1).

    *target_total* **must** be a power of 2 for the rANS codec.
    """
    if not freq:
        return {}

    current_total = sum(freq.values())
    if current_total == 0:
        return {}

    # First pass: proportional scaling with floor, minimum 1
    result: dict[int, int] = {}
    for sym, count in freq.items():
        scaled = max(1, int(count * target_total / current_total))
        result[sym] = scaled

    # Adjust to hit exact target_total by tweaking the most-frequent symbols
    diff = target_total - sum(result.values())
    sorted_syms = sorted(freq.keys(), key=lambda s: freq[s], reverse=True)
    idx = 0
    while diff != 0:
        sym = sorted_syms[idx % len(sorted_syms)]
        if diff > 0:
            result[sym] += 1
            diff -= 1
        elif result[sym] > 1:
            result[sym] -= 1
            diff += 1
        idx += 1
        if idx > len(sorted_syms) * target_total:
            break  # safety valve

    return result


# ---- encoder -----------------------------------------------------------------


class RANSEncoder:
    """rANS entropy encoder (byte-aligned, 64-bit state)."""

    def __init__(self, freq_table: dict[int, int]) -> None:
        if not freq_table:
            raise EntropyError("Frequency table is empty")

        self._freq = dict(freq_table)
        self._total = sum(self._freq.values())

        if self._total != SCALE:
            raise EntropyError(
                f"Frequency table must sum to {SCALE} (got {self._total}). "
                "Use normalize_freq_table() first."
            )

        # Build cumulative frequencies (sorted symbol order)
        self._symbols_sorted = sorted(self._freq.keys())
        self._cumfreq: dict[int, int] = {}
        cum = 0
        for s in self._symbols_sorted:
            self._cumfreq[s] = cum
            cum += self._freq[s]

    def encode(self, symbols: np.ndarray[Any, np.dtype[Any]]) -> bytes:
        """Encode an array of integer symbols into compressed bytes."""
        if len(symbols) == 0:
            return b""

        freq = self._freq
        cumfreq = self._cumfreq
        total = self._total

        # Output stream (bytes emitted during renormalization, reversed later)
        out_bytes: list[int] = []

        # Initial state
        x: int = RANS_L

        # Process symbols in REVERSE order
        for i in range(len(symbols) - 1, -1, -1):
            s = int(symbols[i])
            if s not in freq:
                raise EntropyError(f"Symbol {s} not in frequency table")

            fs = freq[s]
            cs = cumfreq[s]

            # Renormalize: emit low bytes until x < x_max so the encoding
            # step will keep x within [RANS_L, RANS_L * 256).
            # x_max = fs << 48  (derived in module docstring)
            x_max = fs << _X_MAX_SHIFT
            while x >= x_max:
                out_bytes.append(x & 0xFF)
                x >>= 8

            # rANS encoding step
            x = (x // fs) * total + cs + (x % fs)

        # Flush final state as 8 bytes (uint64 LE)
        final_state = struct.pack("<Q", x)

        # Reverse the byte stream so the decoder can read it forward.
        # The last bytes emitted (for symbol 0) need to be consumed first.
        out_bytes.reverse()

        return final_state + bytes(out_bytes)

    def encode_to_bytes(self, symbols: np.ndarray[Any, np.dtype[Any]]) -> bytes:
        """Convenience alias for *encode*."""
        return self.encode(symbols)


# ---- decoder -----------------------------------------------------------------


class RANSDecoder:
    """rANS entropy decoder (byte-aligned, 64-bit state)."""

    def __init__(self, freq_table: dict[int, int]) -> None:
        if not freq_table:
            raise EntropyError("Frequency table is empty")

        self._freq = dict(freq_table)
        self._total = sum(self._freq.values())

        if self._total != SCALE:
            raise EntropyError(
                f"Frequency table must sum to {SCALE} (got {self._total}). "
                "Use normalize_freq_table() first."
            )

        # Build cumulative frequencies
        self._symbols_sorted = sorted(self._freq.keys())
        self._cumfreq: dict[int, int] = {}
        cum = 0
        for s in self._symbols_sorted:
            self._cumfreq[s] = cum
            cum += self._freq[s]

        # Build O(1) symbol lookup table indexed by cumulative slot
        self._slot_to_sym = [0] * self._total
        for s in self._symbols_sorted:
            cs = self._cumfreq[s]
            fs = self._freq[s]
            for j in range(fs):
                self._slot_to_sym[cs + j] = s

    def decode(self, data: bytes, num_symbols: int) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode *num_symbols* from compressed *data*."""
        if num_symbols == 0:
            return np.array([], dtype=np.int64)

        if len(data) < 8:
            raise EntropyError("Data too short to contain rANS state")

        freq = self._freq
        cumfreq = self._cumfreq
        slot_to_sym = self._slot_to_sym
        mask = self._total - 1  # total is a power of 2

        # Read initial state
        x: int = struct.unpack_from("<Q", data, 0)[0]
        pos = 8

        output = np.empty(num_symbols, dtype=np.int64)

        for i in range(num_symbols):
            # Determine symbol from the low SCALE_BITS of state
            slot = x & mask
            s = slot_to_sym[slot]
            fs = freq[s]
            cs = cumfreq[s]

            # rANS decoding step (exact inverse of encoding)
            x = fs * (x >> SCALE_BITS) + slot - cs

            # Renormalize: read bytes to bring state back into [RANS_L, …)
            while x < RANS_L and pos < len(data):
                x = (x << 8) | data[pos]
                pos += 1

            output[i] = s

        return output
