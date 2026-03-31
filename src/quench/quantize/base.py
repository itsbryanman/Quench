"""Reusable quantization layouts, parameters, and protocol definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np

from quench.core.config import QuantizationGranularity
from quench.core.exceptions import QuantizationError
from quench.core.types import QuantMode


@dataclass(frozen=True)
class QuantizationLayout:
    """Describe how quantization parameters are applied across a tensor."""

    granularity: QuantizationGranularity
    axis: int = 0
    block_size: int | None = None

    def normalized_axis(self, ndim: int) -> int:
        """Normalize the configured axis for a tensor of rank *ndim*."""
        if ndim <= 0:
            raise QuantizationError("quantization layout requires a tensor with at least one dimension")
        if not (-ndim <= self.axis < ndim):
            raise QuantizationError(f"axis {self.axis} is out of bounds for rank-{ndim} tensor")
        return self.axis % ndim


@dataclass(frozen=True)
class QuantParams:
    """Quantization parameters for uniform affine quantization."""

    scale: float
    zero_point: int
    bits: int
    mode: QuantMode
    dtype_orig: str
    value_range_min: float
    value_range_max: float


@dataclass(frozen=True)
class ChannelQuantParams:
    """Per-channel affine parameters along a chosen tensor axis."""

    axis: int
    params: tuple[QuantParams, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class BlockQuantParams:
    """Per-block affine parameters for contiguous slices along one axis."""

    axis: int
    block_size: int
    block_lengths: tuple[int, ...]
    params: tuple[QuantParams, ...]
    shape: tuple[int, ...]


QuantizationParameters: TypeAlias = QuantParams | ChannelQuantParams | BlockQuantParams


class Quantizer(Protocol):
    """Protocol for reusable tensor quantizers."""

    layout: QuantizationLayout

    def quantize(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        params: QuantizationParameters,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Quantize *tensor* using explicit *params*."""

    def dequantize(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        params: QuantizationParameters,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Dequantize *quantized* using explicit *params*."""


class CalibrationPolicy(Protocol):
    """Protocol for deterministic quantization calibration policies."""

    name: str

    def calibrate(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        bits: int,
        mode: QuantMode,
        layout: QuantizationLayout,
    ) -> QuantizationParameters:
        """Compute quantization parameters for *tensor* and *layout*."""


def storage_dtype(bits: int, mode: QuantMode) -> type[np.generic]:
    """Choose a compact integer dtype for a quantized representation."""
    if mode == QuantMode.SYMMETRIC:
        if bits <= 8:
            return np.int8
        if bits <= 16:
            return np.int16
        return np.int32
    if mode == QuantMode.ASYMMETRIC:
        if bits <= 8:
            return np.uint8
        if bits <= 16:
            return np.uint16
        return np.uint32
    raise QuantizationError("QuantMode.NONE does not have a quantized storage dtype")


def quantized_bounds(bits: int, mode: QuantMode) -> tuple[int, int]:
    """Return the representable integer bounds for *bits* and *mode*."""
    if not (1 <= bits <= 32):
        raise QuantizationError(f"bits must be in [1, 32], got {bits}")
    if mode == QuantMode.SYMMETRIC:
        qmax = 1 if bits == 1 else (1 << (bits - 1)) - 1
        return -qmax, qmax
    if mode == QuantMode.ASYMMETRIC:
        return 0, (1 << bits) - 1
    raise QuantizationError("QuantMode.NONE cannot be used for integer quantization")


def compute_scalar_params(
    *,
    value_min: float,
    value_max: float,
    bits: int,
    mode: QuantMode,
    dtype_orig: str,
) -> QuantParams:
    """Compute deterministic scalar affine parameters from a value range."""
    qmin, qmax = quantized_bounds(bits, mode)

    if mode == QuantMode.SYMMETRIC:
        max_abs = max(abs(value_min), abs(value_max))
        scale = max_abs / max(abs(qmin), qmax, 1)
        scale = 1.0 if scale == 0.0 or not np.isfinite(scale) else float(scale)
        zero_point = 0
    elif mode == QuantMode.ASYMMETRIC:
        if value_max == value_min:
            if value_max == 0.0:
                scale = 1.0
                zero_point = 0
            else:
                scale = abs(value_max) / max(qmax, 1)
                zero_point = qmax if value_max < 0.0 else 0
        else:
            scale = (value_max - value_min) / max(qmax - qmin, 1)
            scale = 1.0 if scale == 0.0 or not np.isfinite(scale) else float(scale)
            zero_point = int(np.rint(-value_min / scale))
            zero_point = int(np.clip(zero_point, qmin, qmax))
    else:
        raise QuantizationError("QuantMode.NONE cannot be calibrated for lossy quantization")

    return QuantParams(
        scale=float(scale),
        zero_point=int(zero_point),
        bits=bits,
        mode=mode,
        dtype_orig=dtype_orig,
        value_range_min=float(value_min),
        value_range_max=float(value_max),
    )


def serialize_layout(layout: QuantizationLayout) -> dict[str, Any]:
    """Serialize a quantization layout into JSON-safe metadata."""
    metadata: dict[str, Any] = {
        "axis": layout.axis,
        "granularity": layout.granularity.value,
    }
    if layout.block_size is not None:
        metadata["block_size"] = layout.block_size
    return metadata


def deserialize_layout(metadata: dict[str, Any]) -> QuantizationLayout:
    """Reconstruct a :class:`QuantizationLayout` from metadata."""
    return QuantizationLayout(
        granularity=QuantizationGranularity(str(metadata["granularity"])),
        axis=int(metadata.get("axis", 0)),
        block_size=(
            None
            if metadata.get("block_size") is None
            else int(metadata["block_size"])
        ),
    )


def serialize_quant_params(params: QuantizationParameters) -> dict[str, Any]:
    """Serialize any supported quantization parameter structure."""
    if isinstance(params, QuantParams):
        return {
            "kind": "per_tensor",
            "params": _serialize_scalar_params(params),
        }
    if isinstance(params, ChannelQuantParams):
        return {
            "axis": params.axis,
            "kind": "per_channel",
            "params": [_serialize_scalar_params(item) for item in params.params],
            "shape": list(params.shape),
        }
    if isinstance(params, BlockQuantParams):
        return {
            "axis": params.axis,
            "block_lengths": list(params.block_lengths),
            "block_size": params.block_size,
            "kind": "blockwise",
            "params": [_serialize_scalar_params(item) for item in params.params],
            "shape": list(params.shape),
        }
    raise QuantizationError(f"Unsupported quantization parameter type: {type(params)!r}")


def deserialize_quant_params(metadata: dict[str, Any]) -> QuantizationParameters:
    """Deserialize quantization parameter metadata into a typed object."""
    kind = str(metadata["kind"])
    if kind == "per_tensor":
        return _deserialize_scalar_params(metadata["params"])
    if kind == "per_channel":
        return ChannelQuantParams(
            axis=int(metadata["axis"]),
            params=tuple(_deserialize_scalar_params(item) for item in metadata["params"]),
            shape=tuple(int(dim) for dim in metadata["shape"]),
        )
    if kind == "blockwise":
        return BlockQuantParams(
            axis=int(metadata["axis"]),
            block_size=int(metadata["block_size"]),
            block_lengths=tuple(int(length) for length in metadata["block_lengths"]),
            params=tuple(_deserialize_scalar_params(item) for item in metadata["params"]),
            shape=tuple(int(dim) for dim in metadata["shape"]),
        )
    raise QuantizationError(f"Unsupported quantization parameter metadata kind: {kind!r}")


def _serialize_scalar_params(params: QuantParams) -> dict[str, Any]:
    return {
        "bits": params.bits,
        "dtype_orig": params.dtype_orig,
        "mode": int(params.mode),
        "scale": params.scale,
        "value_range_max": params.value_range_max,
        "value_range_min": params.value_range_min,
        "zero_point": params.zero_point,
    }


def _deserialize_scalar_params(metadata: dict[str, Any]) -> QuantParams:
    return QuantParams(
        scale=float(metadata["scale"]),
        zero_point=int(metadata["zero_point"]),
        bits=int(metadata["bits"]),
        mode=QuantMode(int(metadata["mode"])),
        dtype_orig=str(metadata["dtype_orig"]),
        value_range_min=float(metadata["value_range_min"]),
        value_range_max=float(metadata["value_range_max"]),
    )
