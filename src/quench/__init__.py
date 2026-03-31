"""Quench — format-aware compression codec for ML tensors."""
from __future__ import annotations

from quench.analyze import TensorProfiler, TensorTypeDetector
from quench.backends import (
    get_backend_binding,
    get_entropy_backend,
    get_packing_backend,
    list_backend_names,
)
from quench.codec import (
    QuenchDecoder,
    QuenchEncoder,
    auto_compress,
    auto_decompress,
)
from quench.io import QNCReader, QNCWriter, decode_tensor_stream, encode_tensor_stream, iter_tensor_records
from quench.core.config import QuenchConfig
from quench.core.types import CodecMode, CompressedTensor, QuantMode, TensorHeader, TensorType
from quench.quantize import (
    BlockQuantParams,
    BlockwiseCalibrationPolicy,
    BlockwiseQuantizer,
    Calibrator,
    ChannelQuantParams,
    ImportanceAllocator,
    PerChannelCalibrationPolicy,
    PerChannelQuantizer,
    PerTensorCalibrationPolicy,
    PerTensorQuantizer,
    PercentileCalibrationPolicy,
    QuantParams,
    QuantizationLayout,
    UniformQuantizer,
)
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
    "BlockQuantParams",
    "BlockwiseCalibrationPolicy",
    "BlockwiseQuantizer",
    "ChannelQuantParams",
    "ImportanceAllocator",
    "PCAState",
    "PCATransform",
    "PerChannelCalibrationPolicy",
    "PerChannelQuantizer",
    "PerTensorCalibrationPolicy",
    "PerTensorQuantizer",
    "PercentileCalibrationPolicy",
    "QuantParams",
    "QuantizationLayout",
    "QNCReader",
    "QNCWriter",
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
    "decode_tensor_stream",
    "decompress",
    "encode_tensor_stream",
    "get_backend_binding",
    "get_entropy_backend",
    "get_packing_backend",
    "iter_tensor_records",
    "list_backend_names",
]
