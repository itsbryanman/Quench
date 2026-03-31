"""Quench — format-aware compression codec for ML tensors."""
from __future__ import annotations

from quench.analyze import TensorProfiler, TensorTypeDetector
from quench.codec import (
    QuenchDecoder,
    QuenchEncoder,
    auto_compress,
    auto_decompress,
)
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

compress = auto_compress
decompress = auto_decompress


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
    "QuenchDecoder",
    "QuenchEncoder",
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
