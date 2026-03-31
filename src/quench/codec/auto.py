"""Public convenience wrappers for the Phase 3 codec pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.codec.decoder import QuenchDecoder
from quench.codec.encoder import QuenchEncoder
from quench.core.config import QuenchConfig
from quench.core.types import CompressedTensor, TensorType

try:  # pragma: no cover - torch is optional
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None


def auto_compress(
    tensor: np.ndarray[Any, np.dtype[Any]] | "torch.Tensor",
    *,
    tensor_type: TensorType | None = None,
    name: str | None = None,
    config: QuenchConfig | None = None,
) -> CompressedTensor:
    """Compress a tensor with the default encoder."""
    return QuenchEncoder(config=config).encode(tensor, tensor_type=tensor_type, name=name)


def auto_decompress(
    compressed: CompressedTensor,
    *,
    config: QuenchConfig | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Decompress a tensor with the default decoder."""
    return QuenchDecoder(config=config).decode(compressed)
