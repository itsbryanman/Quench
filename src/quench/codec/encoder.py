"""High-level tensor encoder orchestration."""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from quench.analyze import TensorProfiler, TensorTypeDetector
from quench.codec.metadata import serialize_metadata
from quench.codec.strategies import get_strategy
from quench.core.config import QuenchConfig
from quench.core.types import CodecMode, CompressedTensor, TensorHeader, TensorType
from quench.quantize import ImportanceAllocator

try:  # pragma: no cover - torch is optional
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None


class QuenchEncoder:
    """Encode numpy or torch tensors into :class:`CompressedTensor` objects."""

    def __init__(self, config: QuenchConfig | None = None) -> None:
        self._config = config or QuenchConfig()
        self._detector = TensorTypeDetector()
        self._profiler = TensorProfiler()
        self._allocator = ImportanceAllocator(self._profiler)

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]] | "torch.Tensor",
        tensor_type: TensorType | None = None,
        name: str | None = None,
    ) -> CompressedTensor:
        """Encode a single tensor using the configured strategy selection."""
        return self._encode_numpy(self._to_numpy(tensor), tensor_type=tensor_type, name=name, config=self._config)

    def encode_dict(
        self,
        tensors: dict[str, np.ndarray[Any, np.dtype[Any]] | "torch.Tensor"],
        config: QuenchConfig | None = None,
    ) -> dict[str, CompressedTensor]:
        """Encode a state-dict-like mapping deterministically."""
        active_config = config or self._config
        numpy_tensors = {name: self._to_numpy(value) for name, value in tensors.items()}
        encoded: dict[str, CompressedTensor] = {}

        if (
            len(numpy_tensors) > 1
            and active_config.codec_mode == CodecMode.LOSSY
            and 2 <= active_config.target_bits <= 8
        ):
            allocation = self._allocator.allocate_bits(
                numpy_tensors,
                total_budget_bits=active_config.target_bits * len(numpy_tensors),
            )
            lower_bound = max(2, active_config.target_bits - 1)
            allocation = {
                name: max(bits, lower_bound)
                for name, bits in allocation.items()
            }
        else:
            allocation = {}

        for name in sorted(numpy_tensors):
            tensor_config = active_config
            if name in allocation:
                tensor_config = active_config.model_copy(update={"target_bits": allocation[name]})
            encoded[name] = self._encode_numpy(
                numpy_tensors[name],
                tensor_type=None,
                name=name,
                config=tensor_config,
            )

        return encoded

    def _encode_numpy(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        tensor_type: TensorType | None,
        name: str | None,
        config: QuenchConfig,
    ) -> CompressedTensor:
        """Encode a normalized numpy tensor with an explicit config."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot encode an empty tensor")

        detected_type = tensor_type or self._detector.detect(values, name=name)
        _ = self._profiler.profile(values)
        strategy = get_strategy(detected_type)
        payload, strategy_metadata = strategy.encode(values, config)

        metadata = {
            "name": name,
            "original_dtype": np.dtype(values.dtype).str,
            "profiled_tensor_type": int(detected_type),
            "shape": list(values.shape),
            "strategy": {
                "id": strategy.strategy_id,
                "name": strategy.strategy_name,
                "metadata": strategy_metadata,
            },
        }
        metadata_bytes = serialize_metadata(metadata)
        checksum = self._checksum(values)

        header = TensorHeader(
            tensor_type=detected_type,
            dtype=np.dtype(values.dtype).name,
            shape=tuple(int(dim) for dim in values.shape),
            codec_mode=config.codec_mode,
            strategy_id=strategy.strategy_id,
            checksum=checksum,
        )
        return CompressedTensor(
            header=header,
            payload=payload,
            metadata=metadata_bytes,
            original_nbytes=int(values.nbytes),
        )

    def _to_numpy(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]] | "torch.Tensor",
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Convert an accepted tensor input into a CPU numpy array."""
        if isinstance(tensor, np.ndarray):
            return np.asarray(tensor)
        if torch is not None and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        raise TypeError("Expected a numpy.ndarray or torch.Tensor input")

    def _checksum(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> int:
        """Compute a stable checksum from the original tensor bytes."""
        raw = np.ascontiguousarray(tensor).view(np.uint8)
        return int(zlib.crc32(raw) & 0xFFFFFFFF)
