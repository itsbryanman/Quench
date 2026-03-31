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


class QuenchDecoder:
    """Decode :class:`CompressedTensor` objects back into numpy arrays."""

    def __init__(self, config: QuenchConfig | None = None) -> None:
        self._config = config or QuenchConfig()

    def decode(self, compressed: CompressedTensor) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode a single compressed tensor."""
        self._validate_compressed(compressed)
        header = compressed.header
        metadata = deserialize_metadata(compressed.metadata)
        strategy_block = metadata.get("strategy")
        if not isinstance(strategy_block, dict):
            raise MalformedPayloadError("Compressed metadata is missing the strategy block")

        strategy_id = int(strategy_block.get("id", header.strategy_id))
        if strategy_id != header.strategy_id:
            raise UnsupportedStrategyError(
                "Strategy id mismatch between header and metadata: "
                f"header={header.strategy_id}, metadata={strategy_id}"
            )

        strategy = get_strategy_by_id(strategy_id, header.tensor_type)
        strategy_metadata = strategy_block.get("metadata")
        if not isinstance(strategy_metadata, dict):
            raise MalformedPayloadError("Strategy metadata must be a dictionary")

        decoded = strategy.decode(compressed.payload, strategy_metadata, config=self._config)
        restored = np.asarray(decoded)

        expected_size = int(np.prod(header.shape, dtype=np.int64))
        if restored.size != expected_size:
            raise MalformedPayloadError(
                "Decoded tensor element count does not match header shape: "
                f"expected {expected_size}, got {restored.size}"
            )

        try:
            dtype = np.dtype(header.dtype)
        except TypeError as exc:
            raise HeaderError(f"Unsupported dtype in tensor header: {header.dtype!r}") from exc

        restored = restored.reshape(header.shape).astype(dtype, copy=False)
        expected_nbytes = expected_size * dtype.itemsize
        if compressed.original_nbytes != expected_nbytes:
            raise HeaderError(
                "Original byte size does not match tensor header: "
                f"expected {expected_nbytes}, got {compressed.original_nbytes}"
            )

        if strategy_metadata.get("lossless") is True:
            checksum = int(zlib.crc32(np.ascontiguousarray(restored).view(np.uint8)) & 0xFFFFFFFF)
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
