"""Quench — format-aware compression codec for ML tensors."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.analyze import TensorProfiler, TensorTypeDetector
from quench.core.config import QuenchConfig
from quench.core.types import CodecMode, CompressedTensor, QuantMode, TensorHeader, TensorType
from quench.quantize import Calibrator, ImportanceAllocator, QuantParams, UniformQuantizer
from quench.transform import (
    ChannelNormalizer,
    DeltaCoder,
    PCAState,
    PCATransform,
    SparseEncoder,
    SparseRepresentation,
    StepMetadata,
    TransformPipeline,
)

__version__ = "0.1.0"


def compress(tensor: np.ndarray[Any, np.dtype[Any]], **kwargs: Any) -> CompressedTensor:
    """Compress a tensor.

    Full codec pipeline coming in Phase 3.
    Use ``quench.entropy`` for direct rANS encoding.
    """
    raise NotImplementedError(
        "Full codec pipeline coming in Phase 3. Use quench.entropy for direct rANS encoding."
    )


def decompress(data: CompressedTensor, **kwargs: Any) -> np.ndarray[Any, np.dtype[Any]]:
    """Decompress a tensor.

    Full codec pipeline coming in Phase 3.
    Use ``quench.entropy`` for direct rANS decoding.
    """
    raise NotImplementedError(
        "Full codec pipeline coming in Phase 3. Use quench.entropy for direct rANS decoding."
    )


__all__ = [
    "Calibrator",
    "ChannelNormalizer",
    "CodecMode",
    "CompressedTensor",
    "DeltaCoder",
    "ImportanceAllocator",
    "PCAState",
    "PCATransform",
    "QuantParams",
    "QuenchConfig",
    "QuantMode",
    "SparseEncoder",
    "SparseRepresentation",
    "StepMetadata",
    "TensorProfiler",
    "TensorHeader",
    "TensorTypeDetector",
    "TensorType",
    "TransformPipeline",
    "UniformQuantizer",
    "__version__",
    "compress",
    "decompress",
]
