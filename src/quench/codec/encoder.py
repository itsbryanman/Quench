"""High-level tensor encoder orchestration."""
from __future__ import annotations

import zlib
from typing import Any, Iterator, cast

import numpy as np

from quench.analyze import TensorTypeDetector
from quench.codec.metadata import serialize_metadata
from quench.codec.strategies import get_strategy
from quench.core.config import QuenchConfig
from quench.core.types import CodecMode, CompressedTensor, TensorHeader, TensorType
from quench.quantize import ImportanceAllocator

torch: Any
try:  # pragma: no cover - torch is optional
    import torch as _torch
except Exception:  # pragma: no cover - torch is optional
    torch = None
else:  # pragma: no cover - torch is optional
    torch = _torch


class QuenchEncoder:
    """Encode numpy or torch tensors into :class:`CompressedTensor` objects."""

    def __init__(self, config: QuenchConfig | None = None) -> None:
        self._config = config or QuenchConfig()
        self._detector = TensorTypeDetector()
        self._allocator = ImportanceAllocator()

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
        return {
            name: compressed
            for name, compressed in self.iter_encode_dict(tensors, config=config)
        }

    def iter_encode_dict(
        self,
        tensors: dict[str, np.ndarray[Any, np.dtype[Any]] | "torch.Tensor"],
        config: QuenchConfig | None = None,
    ) -> Iterator[tuple[str, CompressedTensor]]:
        """Encode a state-dict-like mapping one tensor at a time."""
        active_config = config or self._config
        numpy_tensors = {name: self._to_numpy(value) for name, value in tensors.items()}

        allocation = self._resolve_bit_allocation(numpy_tensors, active_config)

        for name in sorted(numpy_tensors):
            tensor_config = active_config
            if name in allocation:
                tensor_config = active_config.model_copy(update={"target_bits": allocation[name]})
            yield (
                name,
                self._encode_numpy(
                    numpy_tensors[name],
                    tensor_type=None,
                    name=name,
                    config=tensor_config,
                ),
            )

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
        effective_config = config

        if self._should_force_lossless(values, name=name):
            effective_config = config.model_copy(update={"codec_mode": CodecMode.LOSSLESS})

        strategy = get_strategy(detected_type)
        payload, strategy_metadata = strategy.encode(values, effective_config)
        effective_codec_mode = (
            CodecMode.LOSSLESS
            if self._is_lossless_strategy_metadata(strategy_metadata)
            else effective_config.codec_mode
        )
        metadata_bytes = serialize_metadata(strategy_metadata)
        checksum = self._checksum(values)

        header = TensorHeader(
            tensor_type=detected_type,
            dtype=np.dtype(values.dtype).name,
            shape=tuple(int(dim) for dim in values.shape),
            codec_mode=effective_codec_mode,
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
            return cast(np.ndarray[Any, np.dtype[Any]], tensor.detach().cpu().numpy())
        raise TypeError("Expected a numpy.ndarray or torch.Tensor input")

    def _should_force_lossless(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        name: str | None,
    ) -> bool:
        """Force exact routing for tiny tensors and norm parameters."""
        if int(np.asarray(tensor).nbytes) <= 2048:
            return True
        return self._looks_like_norm_tensor_name(name)

    @staticmethod
    def _looks_like_norm_tensor_name(name: str | None) -> bool:
        """Match common LayerNorm/RMSNorm parameter naming schemes."""
        if not name:
            return False

        lower = name.lower()
        if any(
            token in lower
            for token in (
                "layernorm",
                "layer_norm",
                "rmsnorm",
                "rms_norm",
                "input_layernorm",
                "post_attention_layernorm",
            )
        ):
            return True
        if ".ln_" in lower or lower.startswith("ln_"):
            return True
        if ".norm" in lower or lower.endswith(("norm.weight", "norm.bias")):
            return True
        return False

    def _checksum(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> int:
        """Compute a stable checksum from the original tensor bytes."""
        raw = np.ascontiguousarray(tensor).view(np.uint8)
        return int(zlib.crc32(raw.tobytes()) & 0xFFFFFFFF)

    @staticmethod
    def _is_lossless_strategy_metadata(metadata: dict[str, Any]) -> bool:
        """Support compact and legacy exact-path metadata markers."""
        if metadata.get("lossless") is True:
            return True
        return int(metadata.get("l", 0)) == 1

    def _resolve_bit_allocation(
        self,
        tensors: dict[str, np.ndarray[Any, np.dtype[Any]]],
        config: QuenchConfig,
    ) -> dict[str, int]:
        """Allocate bits across a bundle when the active config allows it."""
        if (
            len(tensors) > 1
            and config.codec_mode == CodecMode.LOSSY
            and 2 <= config.target_bits <= 8
        ):
            allocation = self._allocator.allocate_bits(
                tensors,
                total_budget_bits=config.target_bits * len(tensors),
            )
            lower_bound = max(2, config.target_bits - 1)
            return {name: max(bits, lower_bound) for name, bits in allocation.items()}
        return {}
