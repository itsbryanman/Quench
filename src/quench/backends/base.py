"""Backend interfaces for pluggable entropy and packing kernels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from quench.entropy.freq_model import FrequencyModel


class EntropyBackend(Protocol):
    """Protocol for entropy backends used by the codec pipeline."""

    name: str

    def encode_symbols(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
        freq_model: FrequencyModel,
    ) -> bytes:
        """Encode integer *symbols* using *freq_model*."""

    def decode_symbols(
        self,
        data: bytes,
        freq_model: FrequencyModel,
        num_symbols: int,
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Decode *num_symbols* from *data* using *freq_model*."""


class PackingBackend(Protocol):
    """Protocol for quantized bit-packing kernels."""

    name: str

    def pack_bits(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        signed: bool,
        layout_metadata: Mapping[str, Any] | None = None,
    ) -> bytes:
        """Pack integer *symbols* into a compact byte string."""

    def unpack_bits(
        self,
        data: bytes,
        bits: int,
        signed: bool,
        shape: tuple[int, ...],
        layout_metadata: Mapping[str, Any] | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Unpack bit-packed integers into a tensor of *shape*."""


@dataclass(frozen=True)
class BackendBinding:
    """Named pairing of entropy and packing implementations."""

    name: str
    entropy: EntropyBackend
    packing: PackingBackend
