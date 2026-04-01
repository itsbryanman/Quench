"""Calibration utilities for uniform quantization."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.core.config import QuantizationGranularity
from quench.core.exceptions import QuantizationError
from quench.core.types import QuantMode
from quench.quantize.base import (
    BlockQuantParams,
    CalibrationPolicy,
    ChannelQuantParams,
    QuantParams,
    QuantizationLayout,
    QuantizationParameters,
    compute_scalar_params,
)
from quench.quantize.uniform import UniformQuantizer


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize a potentially negative axis index."""
    if not (-ndim <= axis < ndim):
        raise QuantizationError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions")
    return axis % ndim


class PerTensorCalibrationPolicy:
    """Min/max calibration across an entire tensor."""

    name = "per_tensor"

    def calibrate(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        bits: int,
        mode: QuantMode,
        layout: QuantizationLayout,
    ) -> QuantParams:
        values = np.asarray(tensor)
        if values.size == 0:
            raise QuantizationError("Cannot calibrate an empty tensor")
        working = values.astype(np.float64, copy=False)
        return compute_scalar_params(
            value_min=float(np.min(working)),
            value_max=float(np.max(working)),
            bits=bits,
            mode=mode,
            dtype_orig=values.dtype.str,
        )


class PerChannelCalibrationPolicy:
    """Min/max calibration independently for each channel."""

    name = "per_channel"

    def calibrate(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        bits: int,
        mode: QuantMode,
        layout: QuantizationLayout,
    ) -> ChannelQuantParams:
        values = np.asarray(tensor)
        if values.size == 0:
            raise QuantizationError("Cannot calibrate an empty tensor")
        axis = _normalize_axis(layout.axis, values.ndim)
        moved = np.moveaxis(values.astype(np.float64, copy=False), axis, 0)
        params = tuple(
            compute_scalar_params(
                value_min=float(np.min(channel)),
                value_max=float(np.max(channel)),
                bits=bits,
                mode=mode,
                dtype_orig=values.dtype.str,
            )
            for channel in moved
        )
        return ChannelQuantParams(axis=axis, params=params, shape=tuple(values.shape))


class BlockwiseCalibrationPolicy:
    """Min/max calibration independently for contiguous blocks along one axis."""

    name = "blockwise"

    def calibrate(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        bits: int,
        mode: QuantMode,
        layout: QuantizationLayout,
    ) -> BlockQuantParams:
        values = np.asarray(tensor)
        if values.size == 0:
            raise QuantizationError("Cannot calibrate an empty tensor")
        if layout.block_size is None or layout.block_size <= 0:
            raise QuantizationError("Blockwise calibration requires a positive block_size")

        axis = _normalize_axis(layout.axis, values.ndim)
        moved = np.moveaxis(values.astype(np.float64, copy=False), axis, 0)

        params: list[QuantParams] = []
        block_lengths: list[int] = []
        for start in range(0, moved.shape[0], layout.block_size):
            stop = min(start + layout.block_size, moved.shape[0])
            block = moved[start:stop]
            params.append(
                compute_scalar_params(
                    value_min=float(np.min(block)),
                    value_max=float(np.max(block)),
                    bits=bits,
                    mode=mode,
                    dtype_orig=values.dtype.str,
                )
            )
            block_lengths.append(stop - start)

        return BlockQuantParams(
            axis=axis,
            block_size=layout.block_size,
            block_lengths=tuple(block_lengths),
            params=tuple(params),
            shape=tuple(values.shape),
        )


class PercentileCalibrationPolicy:
    """Percentile clipping calibration for tensor, channel, or block layouts."""

    name = "percentile"

    def __init__(self, percentile: float = 99.9) -> None:
        if not (0.0 < percentile <= 100.0):
            raise QuantizationError("percentile must be in (0, 100]")
        self._percentile = float(percentile)

    def calibrate(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        bits: int,
        mode: QuantMode,
        layout: QuantizationLayout,
    ) -> QuantizationParameters:
        values = np.asarray(tensor)
        if values.size == 0:
            raise QuantizationError("Cannot calibrate an empty tensor")

        if layout.granularity == QuantizationGranularity.PER_TENSOR:
            return self._calibrate_array(values, bits=bits, mode=mode)
        if layout.granularity == QuantizationGranularity.PER_CHANNEL:
            axis = _normalize_axis(layout.axis, values.ndim)
            moved = np.moveaxis(values, axis, 0)
            return ChannelQuantParams(
                axis=axis,
                params=tuple(
                    self._calibrate_array(channel, bits=bits, mode=mode)
                    for channel in moved
                ),
                shape=tuple(values.shape),
            )
        if layout.granularity == QuantizationGranularity.BLOCKWISE:
            if layout.block_size is None or layout.block_size <= 0:
                raise QuantizationError("Blockwise percentile calibration requires block_size")
            axis = _normalize_axis(layout.axis, values.ndim)
            moved = np.moveaxis(values, axis, 0)
            params: list[QuantParams] = []
            block_lengths: list[int] = []
            for start in range(0, moved.shape[0], layout.block_size):
                stop = min(start + layout.block_size, moved.shape[0])
                params.append(self._calibrate_array(moved[start:stop], bits=bits, mode=mode))
                block_lengths.append(stop - start)
            return BlockQuantParams(
                axis=axis,
                block_size=layout.block_size,
                block_lengths=tuple(block_lengths),
                params=tuple(params),
                shape=tuple(values.shape),
            )
        raise QuantizationError(f"Unsupported percentile layout: {layout.granularity.value}")

    def _calibrate_array(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        bits: int,
        mode: QuantMode,
    ) -> QuantParams:
        values = np.asarray(tensor)
        working = values.astype(np.float64, copy=False).reshape(-1)
        if mode == QuantMode.SYMMETRIC:
            clip_max = float(np.percentile(np.abs(working), self._percentile))
            clip_min = -clip_max
        elif mode == QuantMode.ASYMMETRIC:
            tail = (100.0 - self._percentile) / 2.0
            clip_min = float(np.percentile(working, tail))
            clip_max = float(np.percentile(working, 100.0 - tail))
        else:
            raise QuantizationError("Percentile calibration does not support QuantMode.NONE")

        return compute_scalar_params(
            value_min=float(clip_min),
            value_max=float(clip_max),
            bits=bits,
            mode=mode,
            dtype_orig=values.dtype.str,
        )


class Calibrator:
    """Backward-compatible calibration facade for the Phase 3 API."""

    def __init__(self, quantizer: UniformQuantizer | None = None) -> None:
        self._quantizer = quantizer or UniformQuantizer()
        self._per_tensor_policy = PerTensorCalibrationPolicy()
        self._per_channel_policy = PerChannelCalibrationPolicy()

    def calibrate_per_tensor(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        mode: QuantMode,
    ) -> QuantParams:
        """Calibrate scalar parameters from the full tensor range."""
        params = self._per_tensor_policy.calibrate(
            tensor,
            bits=bits,
            mode=mode,
            layout=QuantizationLayout(QuantizationGranularity.PER_TENSOR),
        )
        assert isinstance(params, QuantParams)
        return params

    def calibrate_per_channel(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        mode: QuantMode,
        axis: int = 0,
    ) -> list[QuantParams]:
        """Calibrate one set of parameters per channel along *axis*."""
        params = self._per_channel_policy.calibrate(
            tensor,
            bits=bits,
            mode=mode,
            layout=QuantizationLayout(QuantizationGranularity.PER_CHANNEL, axis=axis),
        )
        assert isinstance(params, ChannelQuantParams)
        return list(params.params)

    def percentile_calibrate(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        percentile: float = 99.99,
        mode: QuantMode = QuantMode.SYMMETRIC,
    ) -> QuantParams:
        """Calibrate after clipping outliers to a chosen percentile range."""
        params = PercentileCalibrationPolicy(percentile=percentile).calibrate(
            tensor,
            bits=bits,
            mode=mode,
            layout=QuantizationLayout(QuantizationGranularity.PER_TENSOR),
        )
        assert isinstance(params, QuantParams)
        return params
