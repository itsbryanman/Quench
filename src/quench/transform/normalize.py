"""Per-channel affine normalization."""
from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize a potentially negative axis index."""
    if not (-ndim <= axis < ndim):
        raise ValueError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions")
    return axis % ndim


class ChannelNormalizer:
    """Normalize channels with an invertible affine transform.

    Channels that cross zero use symmetric max-absolute-value scaling with
    ``zero_point = 0``. One-sided channels use asymmetric min-range scaling to
    preserve the available dynamic range for downstream quantization.
    """

    def normalize(
        self, tensor: np.ndarray[Any, np.dtype[Any]], axis: int = 0
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
    ]:
        """Normalize *tensor* along *axis* and return values, scales, and zero-points."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot normalize an empty tensor")

        axis = _normalize_axis(axis, values.ndim)
        working = values.astype(np.float32, copy=False)
        moved = np.moveaxis(working, axis, 0)
        flat = moved.reshape(moved.shape[0], -1)

        mins = np.min(flat, axis=1)
        maxs = np.max(flat, axis=1)
        max_abs = np.maximum(np.abs(mins), np.abs(maxs))
        ranges = maxs - mins

        use_symmetric = (mins < 0.0) & (maxs > 0.0)
        scales = np.where(use_symmetric, max_abs, ranges)
        scales = np.where(scales > 0.0, scales, 1.0).astype(np.float32)
        zero_points = np.where(use_symmetric, 0.0, mins).astype(np.float32)

        normalized = (flat - zero_points[:, None]) / scales[:, None]
        normalized = normalized.reshape(moved.shape)
        return np.moveaxis(normalized, 0, axis), scales, zero_points

    def denormalize(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        scales: np.ndarray[Any, np.dtype[Any]],
        zero_points: np.ndarray[Any, np.dtype[Any]],
        axis: int = 0,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Reverse ``normalize`` using per-channel *scales* and *zero_points*."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot denormalize an empty tensor")

        axis = _normalize_axis(axis, values.ndim)
        scales_arr = np.asarray(scales, dtype=np.float32)
        zero_arr = np.asarray(zero_points, dtype=np.float32)

        if scales_arr.ndim != 1 or zero_arr.ndim != 1:
            raise ValueError("scales and zero_points must be 1D arrays")
        if scales_arr.shape != zero_arr.shape:
            raise ValueError("scales and zero_points must have matching shapes")
        if scales_arr.size != values.shape[axis]:
            raise ValueError(
                "Per-channel parameter length does not match the normalized axis"
            )
        if np.any(scales_arr == 0.0):
            raise ValueError("scales must be non-zero")

        working = values.astype(np.float32, copy=False)
        moved = np.moveaxis(working, axis, 0)
        flat = moved.reshape(moved.shape[0], -1)
        restored = flat * scales_arr[:, None] + zero_arr[:, None]
        restored = restored.reshape(moved.shape)
        return np.moveaxis(restored, 0, axis)
