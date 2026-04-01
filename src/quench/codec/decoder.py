"""High-level tensor decoder orchestration."""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from quench.codec.metadata import deserialize_metadata
from quench.codec.strategies import get_strategy_by_id
from quench.core.config import QuenchConfig
from quench.core.exceptions import (
    ChecksumMismatchError,
    CodecError,
    HeaderError,
    MalformedPayloadError,
    UnsupportedStrategyError,
)
from quench.core.types import CompressedTensor, TensorHeader

_ALLOWED_DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
    }
)


class QuenchDecoder:
    """Decode :class:`CompressedTensor` objects back into numpy arrays."""

    def __init__(self, config: QuenchConfig | None = None) -> None:
        self._config = config or QuenchConfig()

    def decode(self, compressed: CompressedTensor) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode a single compressed tensor."""
        self._validate_compressed(compressed)
        header = compressed.header
        if header.dtype not in _ALLOWED_DTYPES:
            raise HeaderError(f"Disallowed dtype in tensor header: {header.dtype!r}")
        dtype = np.dtype(header.dtype)
        metadata = deserialize_metadata(compressed.metadata)
        strategy_id, strategy_metadata = self._resolve_strategy_metadata(metadata, header)
        strategy = get_strategy_by_id(strategy_id, header.tensor_type)
        prepared_metadata = dict(strategy_metadata)
        prepared_metadata.setdefault("_d", dtype.str)
        prepared_metadata.setdefault("_s", [int(dim) for dim in header.shape])

        decoded = strategy.decode(compressed.payload, prepared_metadata, config=self._config)
        restored = np.asarray(decoded)

        expected_size = int(np.prod(header.shape, dtype=np.int64))
        if restored.size != expected_size:
            raise MalformedPayloadError(
                "Decoded tensor element count does not match header shape: "
                f"expected {expected_size}, got {restored.size}"
            )

        restored = restored.reshape(header.shape).astype(dtype, copy=False)
        expected_nbytes = expected_size * dtype.itemsize
        if compressed.original_nbytes != expected_nbytes:
            raise HeaderError(
                "Original byte size does not match tensor header: "
                f"expected {expected_nbytes}, got {compressed.original_nbytes}"
            )

        if self._is_lossless_metadata(prepared_metadata):
            checksum = int(
                zlib.crc32(np.ascontiguousarray(restored).view(np.uint8).tobytes()) & 0xFFFFFFFF
            )
            if checksum != header.checksum:
                raise ChecksumMismatchError(
                    f"Checksum mismatch after lossless decode: expected {header.checksum}, got {checksum}"
                )

        return restored

    def decode_dict(self, compressed: dict[str, CompressedTensor]) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
        """Decode a mapping of compressed tensors deterministically."""
        return {name: self.decode(compressed[name]) for name in sorted(compressed)}

    def _validate_compressed(self, compressed: CompressedTensor) -> None:
        """Validate the high-level compressed container before decode."""
        if not isinstance(compressed, CompressedTensor):
            raise CodecError("decode expects a CompressedTensor instance")
        if not isinstance(compressed.header, TensorHeader):
            raise HeaderError("Compressed tensor is missing a valid header")
        if not isinstance(compressed.payload, bytes):
            raise MalformedPayloadError("Compressed tensor payload must be bytes")
        if not isinstance(compressed.metadata, bytes):
            raise MalformedPayloadError("Compressed tensor metadata must be bytes")
        if compressed.original_nbytes <= 0:
            raise CodecError("Compressed tensor original_nbytes must be positive")

    def _resolve_strategy_metadata(
        self,
        metadata: dict[str, Any],
        header: TensorHeader,
    ) -> tuple[int, dict[str, Any]]:
        """Support both legacy wrapped metadata and compact direct strategy metadata."""
        strategy_block = metadata.get("strategy")
        if isinstance(strategy_block, dict):
            strategy_id = int(strategy_block.get("id", header.strategy_id))
            if strategy_id != header.strategy_id:
                raise UnsupportedStrategyError(
                    "Strategy id mismatch between header and metadata: "
                    f"header={header.strategy_id}, metadata={strategy_id}"
                )
            strategy_metadata = strategy_block.get("metadata")
            if not isinstance(strategy_metadata, dict):
                raise MalformedPayloadError("Strategy metadata must be a dictionary")
            return strategy_id, strategy_metadata

        return header.strategy_id, metadata

    @staticmethod
    def _is_lossless_metadata(metadata: dict[str, Any]) -> bool:
        """Support compact and legacy lossless metadata flags."""
        if metadata.get("lossless") is True:
            return True
        return int(metadata.get("l", 0)) == 1
