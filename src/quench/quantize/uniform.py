"""Uniform tensor quantization."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.core.config import QuantizationGranularity
from quench.core.exceptions import QuantizationError
from quench.core.types import QuantMode
from quench.quantize.base import (
    BlockQuantParams,
    ChannelQuantParams,
    QuantParams,
    QuantizationLayout,
    compute_scalar_params,
    quantized_bounds,
    storage_dtype,
)


class PerTensorQuantizer:
    """Uniform quantizer that applies one parameter set to an entire tensor."""

    def __init__(self) -> None:
        self.layout = QuantizationLayout(QuantizationGranularity.PER_TENSOR)

    def quantize(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        params: QuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Quantize *tensor* using explicit scalar affine parameters."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise QuantizationError("Cannot quantize an empty tensor")
        if params.scale <= 0.0:
            raise QuantizationError("Quantization scale must be positive")

        working = values.astype(np.float64, copy=False)
        qmin, qmax = quantized_bounds(params.bits, params.mode)

        if params.mode == QuantMode.SYMMETRIC:
            scaled = np.rint(working / params.scale)
        elif params.mode == QuantMode.ASYMMETRIC:
            scaled = np.rint(working / params.scale) + params.zero_point
        else:
            raise QuantizationError("PerTensorQuantizer does not support QuantMode.NONE")

        clipped = np.clip(scaled, qmin, qmax)
        return clipped.astype(storage_dtype(params.bits, params.mode), copy=False)

    def dequantize(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        params: QuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Dequantize *quantized* using scalar affine parameters."""
        if params.scale <= 0.0:
            raise QuantizationError("Quantization scale must be positive")

        values = np.asarray(quantized, dtype=np.float64)
        if params.mode == QuantMode.SYMMETRIC:
            restored = values * params.scale
        elif params.mode == QuantMode.ASYMMETRIC:
            restored = (values - params.zero_point) * params.scale
        else:
            raise QuantizationError("PerTensorQuantizer does not support QuantMode.NONE")

        return restored.astype(np.dtype(params.dtype_orig), copy=False)


class PerChannelQuantizer:
    """Uniform quantizer that applies one parameter set per tensor channel."""

    def __init__(self, axis: int = 0) -> None:
        self.layout = QuantizationLayout(QuantizationGranularity.PER_CHANNEL, axis=axis)
        self._per_tensor = PerTensorQuantizer()

    def quantize(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        params: ChannelQuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Quantize *tensor* independently for each channel."""
        values = np.asarray(tensor)
        axis = self.layout.normalized_axis(values.ndim)
        if params.axis != axis:
            raise QuantizationError("Per-channel parameter axis does not match quantizer axis")
        if values.shape != params.shape:
            raise QuantizationError("Per-channel parameter shape does not match tensor shape")

        moved = np.moveaxis(values, axis, 0)
        if moved.shape[0] != len(params.params):
            raise QuantizationError("Per-channel parameter count does not match the tensor axis length")

        bits = params.params[0].bits
        mode = params.params[0].mode
        can_vectorize = all(p.bits == bits and p.mode == mode for p in params.params)
        if not can_vectorize:
            quantized_channels = [
                self._per_tensor.quantize(channel, channel_params)
                for channel, channel_params in zip(moved, params.params)
            ]
            return np.moveaxis(np.stack(quantized_channels, axis=0), 0, axis)

        qmin, qmax = quantized_bounds(bits, mode)
        scales = np.array([p.scale for p in params.params], dtype=np.float64)
        zero_points = np.array([p.zero_point for p in params.params], dtype=np.float64)

        flat = moved.reshape(moved.shape[0], -1).astype(np.float64)
        if mode == QuantMode.SYMMETRIC:
            scaled = np.rint(flat / scales[:, None])
        elif mode == QuantMode.ASYMMETRIC:
            scaled = np.rint(flat / scales[:, None]) + zero_points[:, None]
        else:
            raise QuantizationError("PerChannelQuantizer does not support QuantMode.NONE")

        clipped = np.clip(scaled, qmin, qmax)
        result = clipped.reshape(moved.shape).astype(storage_dtype(bits, mode), copy=False)
        return np.moveaxis(result, 0, axis)

    def dequantize(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        params: ChannelQuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Dequantize *quantized* using one parameter set per channel."""
        values = np.asarray(quantized)
        axis = self.layout.normalized_axis(values.ndim)
        if params.axis != axis:
            raise QuantizationError("Per-channel parameter axis does not match quantizer axis")
        if values.shape != params.shape:
            raise QuantizationError("Per-channel parameter shape does not match tensor shape")

        moved = np.moveaxis(values, axis, 0)
        if moved.shape[0] != len(params.params):
            raise QuantizationError("Per-channel parameter count does not match the tensor axis length")

        bits = params.params[0].bits
        mode = params.params[0].mode
        can_vectorize = all(p.bits == bits and p.mode == mode for p in params.params)
        if not can_vectorize:
            restored_channels = [
                self._per_tensor.dequantize(channel, channel_params)
                for channel, channel_params in zip(moved, params.params)
            ]
            return np.moveaxis(np.stack(restored_channels, axis=0), 0, axis)

        scales = np.array([p.scale for p in params.params], dtype=np.float64)
        zero_points = np.array([p.zero_point for p in params.params], dtype=np.float64)
        dtype_orig = params.params[0].dtype_orig

        flat = moved.reshape(moved.shape[0], -1).astype(np.float64)
        if mode == QuantMode.SYMMETRIC:
            restored = flat * scales[:, None]
        elif mode == QuantMode.ASYMMETRIC:
            restored = (flat - zero_points[:, None]) * scales[:, None]
        else:
            raise QuantizationError("PerChannelQuantizer does not support QuantMode.NONE")

        result = restored.reshape(moved.shape).astype(np.dtype(dtype_orig), copy=False)
        return np.moveaxis(result, 0, axis)


class BlockwiseQuantizer:
    """Uniform quantizer that applies one parameter set per axis-aligned block."""

    def __init__(self, axis: int = 0, block_size: int = 128) -> None:
        if block_size <= 0:
            raise QuantizationError("block_size must be positive")
        self.layout = QuantizationLayout(
            QuantizationGranularity.BLOCKWISE,
            axis=axis,
            block_size=block_size,
        )
        self._per_tensor = PerTensorQuantizer()

    def quantize(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        params: BlockQuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Quantize *tensor* blockwise along the configured axis."""
        values = np.asarray(tensor)
        axis = self.layout.normalized_axis(values.ndim)
        self._validate_params(values, params, axis)

        moved = np.moveaxis(values, axis, 0)
        quantized_blocks = []
        offset = 0
        for length, block_params in zip(params.block_lengths, params.params):
            quantized_blocks.append(
                self._per_tensor.quantize(moved[offset : offset + length], block_params)
            )
            offset += length
        return np.moveaxis(np.concatenate(quantized_blocks, axis=0), 0, axis)

    def dequantize(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        params: BlockQuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Dequantize *quantized* blockwise along the configured axis."""
        values = np.asarray(quantized)
        axis = self.layout.normalized_axis(values.ndim)
        self._validate_params(values, params, axis)

        moved = np.moveaxis(values, axis, 0)
        restored_blocks = []
        offset = 0
        for length, block_params in zip(params.block_lengths, params.params):
            restored_blocks.append(
                self._per_tensor.dequantize(moved[offset : offset + length], block_params)
            )
            offset += length
        return np.moveaxis(np.concatenate(restored_blocks, axis=0), 0, axis)

    def _validate_params(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        params: BlockQuantParams,
        axis: int,
    ) -> None:
        if params.axis != axis:
            raise QuantizationError("Blockwise parameter axis does not match quantizer axis")
        if params.block_size != self.layout.block_size:
            raise QuantizationError("Blockwise parameter size does not match quantizer block size")
        if tensor.shape != params.shape:
            raise QuantizationError("Blockwise parameter shape does not match tensor shape")
        if sum(params.block_lengths) != tensor.shape[axis]:
            raise QuantizationError("Blockwise parameter lengths do not cover the tensor axis")


class UniformQuantizer:
    """Backward-compatible per-tensor uniform quantizer facade."""

    def __init__(self) -> None:
        self._per_tensor = PerTensorQuantizer()

    def quantize(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        mode: QuantMode,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], QuantParams]:
        """Quantize *tensor* with scalar min/max calibration."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise QuantizationError("Cannot quantize an empty tensor")
        working = values.astype(np.float64, copy=False)
        params = self._compute_params(
            value_min=float(np.min(working)),
            value_max=float(np.max(working)),
            bits=bits,
            mode=mode,
            dtype_orig=values.dtype.str,
        )
        return self._per_tensor.quantize(values, params), params

    def dequantize(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        params: QuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Dequantize *quantized* values using scalar params."""
        return self._per_tensor.dequantize(quantized, params)

    @staticmethod
    def _compute_params(
        value_min: float,
        value_max: float,
        bits: int,
        mode: QuantMode,
        dtype_orig: str,
    ) -> QuantParams:
        """Compute quantization parameters from a clipped value range."""
        return compute_scalar_params(
            value_min=value_min,
            value_max=value_max,
            bits=bits,
            mode=mode,
            dtype_orig=dtype_orig,
        )
