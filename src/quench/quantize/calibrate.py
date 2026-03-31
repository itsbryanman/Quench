"""Calibration utilities for uniform quantization."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.core.types import QuantMode
from quench.quantize.uniform import QuantParams, UniformQuantizer


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize a potentially negative axis index."""
    if not (-ndim <= axis < ndim):
        raise ValueError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions")
    return axis % ndim


class Calibrator:
    """Calibrate quantization parameters from tensor statistics."""

    def __init__(self, quantizer: UniformQuantizer | None = None) -> None:
        self._quantizer = quantizer or UniformQuantizer()

    def calibrate_per_tensor(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        mode: QuantMode,
    ) -> QuantParams:
        """Calibrate quantization parameters from the full tensor range."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot calibrate an empty tensor")
        working = values.astype(np.float64, copy=False)
        return self._quantizer._compute_params(
            value_min=float(np.min(working)),
            value_max=float(np.max(working)),
            bits=bits,
            mode=mode,
            dtype_orig=values.dtype.str,
        )

    def calibrate_per_channel(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        mode: QuantMode,
        axis: int = 0,
    ) -> list[QuantParams]:
        """Calibrate one set of parameters per channel along *axis*."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot calibrate an empty tensor")

        axis = _normalize_axis(axis, values.ndim)
        moved = np.moveaxis(values.astype(np.float64, copy=False), axis, 0)

        params: list[QuantParams] = []
        for channel in moved:
            params.append(
                self._quantizer._compute_params(
                    value_min=float(np.min(channel)),
                    value_max=float(np.max(channel)),
                    bits=bits,
                    mode=mode,
                    dtype_orig=values.dtype.str,
                )
            )
        return params

    def percentile_calibrate(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        percentile: float = 99.99,
        mode: QuantMode = QuantMode.SYMMETRIC,
    ) -> QuantParams:
        """Calibrate after clipping outliers to a chosen percentile range."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot calibrate an empty tensor")
        if not (0.0 < percentile <= 100.0):
            raise ValueError("percentile must be in (0, 100]")

        working = values.astype(np.float64, copy=False).reshape(-1)
        if mode == QuantMode.SYMMETRIC:
            clip_max = float(np.percentile(np.abs(working), percentile))
            clip_min = -clip_max
        else:
            tail = (100.0 - percentile) / 2.0
            clip_min = float(np.percentile(working, tail))
            clip_max = float(np.percentile(working, 100.0 - tail))

        clipped = np.clip(working, clip_min, clip_max)
        return self._quantizer._compute_params(
            value_min=float(np.min(clipped)),
            value_max=float(np.max(clipped)),
            bits=bits,
            mode=mode,
            dtype_orig=values.dtype.str,
        )
