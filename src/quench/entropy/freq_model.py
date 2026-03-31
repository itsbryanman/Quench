"""Frequency model for entropy coding."""
from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np


class FrequencyModel:
    """Manages symbol frequency tables and cumulative frequencies for rANS."""

    __slots__ = ("freq_table", "total", "cumfreq", "_symbols_sorted")

    def __init__(self, freq_table: dict[int, int]) -> None:
        self.freq_table = dict(freq_table)
        self.total = sum(self.freq_table.values())
        # Sort symbols for deterministic cumfreq ordering
        self._symbols_sorted = sorted(self.freq_table.keys())
        # Build cumulative frequency table: symbol -> cumulative freq *before* this symbol
        cum = 0
        self.cumfreq: dict[int, int] = {}
        for s in self._symbols_sorted:
            self.cumfreq[s] = cum
            cum += self.freq_table[s]

    @classmethod
    def from_data(cls, symbols: np.ndarray[Any, np.dtype[Any]]) -> FrequencyModel:
        """Build a frequency model from an array of integer symbols."""
        unique, counts = np.unique(symbols, return_counts=True)
        freq: dict[int, int] = {int(s): int(c) for s, c in zip(unique, counts)}
        return cls(freq)

    @classmethod
    def from_freq_table(cls, freq: dict[int, int]) -> FrequencyModel:
        """Build from an existing frequency table."""
        return cls(freq)

    def entropy_bound(self) -> float:
        """Shannon entropy in bits per symbol."""
        if self.total == 0:
            return 0.0
        h = 0.0
        for s in self._symbols_sorted:
            f = self.freq_table[s]
            if f > 0:
                p = f / self.total
                h -= p * math.log2(p)
        return h

    def total_entropy_bits(self) -> float:
        """Total Shannon entropy for all symbols in bits."""
        return self.entropy_bound() * self.total

    def serialize(self) -> bytes:
        """Pack frequency table into bytes for storage."""
        # Format: 4 bytes num_symbols, then (4 bytes symbol, 4 bytes freq) per entry
        buf = bytearray()
        buf.extend(struct.pack("<I", len(self.freq_table)))
        for s in self._symbols_sorted:
            buf.extend(struct.pack("<iI", s, self.freq_table[s]))
        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes) -> FrequencyModel:
        """Unpack frequency table from bytes."""
        offset = 0
        (num_symbols,) = struct.unpack_from("<I", data, offset)
        offset += 4
        freq: dict[int, int] = {}
        for _ in range(num_symbols):
            sym, count = struct.unpack_from("<iI", data, offset)
            offset += 8
            freq[sym] = count
        return cls(freq)
