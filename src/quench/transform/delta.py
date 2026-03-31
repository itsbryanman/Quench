"""Lossless delta coding."""
from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize a potentially negative axis index."""
    if not (-ndim <= axis < ndim):
        raise ValueError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions")
    return axis % ndim


class DeltaCoder:
    """Encode tensors as first-order deltas along a chosen axis.

    Float16 and float32 tensors use float64 working space so encode/decode is
    exactly reversible when cast back to the original dtype. Float64 tensors
    fall back to an unsigned integer view to avoid numerical drift.
    """

    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], axis: int = 0
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        """Encode *tensor* as deltas plus the first-slice anchor."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot delta-encode an empty tensor")

        axis = _normalize_axis(axis, values.ndim)
        if values.shape[axis] == 0:
            raise ValueError("Cannot delta-encode an axis with zero length")

        if values.dtype.itemsize >= 8:
            return self._encode_bitwise(values, axis)

        moved = np.moveaxis(values, axis, 0)
        anchor = np.take(values, [0], axis=axis).copy()
        working_dtype = (
            np.float64
            if np.issubdtype(values.dtype, np.floating)
            else np.result_type(values.dtype, np.int64)
        )
        working = moved.astype(working_dtype, copy=False)
        deltas = np.zeros_like(working)
        deltas[1:] = working[1:] - working[:-1]
        return np.moveaxis(deltas, 0, axis), anchor

    def decode(
        self,
        deltas: np.ndarray[Any, np.dtype[Any]],
        anchor: np.ndarray[Any, np.dtype[Any]],
        axis: int = 0,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode delta-coded *deltas* using *anchor*."""
        delta_values = np.asarray(deltas)
        anchor_values = np.asarray(anchor)
        if delta_values.size == 0:
            raise ValueError("Cannot delta-decode an empty tensor")

        axis = _normalize_axis(axis, delta_values.ndim)
        expected_anchor_shape = list(delta_values.shape)
        expected_anchor_shape[axis] = 1
        if tuple(anchor_values.shape) != tuple(expected_anchor_shape):
            raise ValueError("anchor shape must match deltas except for a size-1 encoded axis")

        if (
            anchor_values.dtype.itemsize >= 8
            and np.issubdtype(delta_values.dtype, np.unsignedinteger)
            and delta_values.dtype.itemsize == anchor_values.dtype.itemsize
        ):
            return self._decode_bitwise(delta_values, anchor_values, axis)

        moved = np.moveaxis(delta_values, axis, 0)
        anchor_moved = np.moveaxis(anchor_values.astype(moved.dtype, copy=False), axis, 0)
        restored = np.array(moved, copy=True)
        restored[0] = anchor_moved[0]
        restored = np.cumsum(restored, axis=0, dtype=restored.dtype)
        return np.moveaxis(restored, 0, axis).astype(anchor_values.dtype, copy=False)

    @staticmethod
    def _encode_bitwise(
        tensor: np.ndarray[Any, np.dtype[Any]], axis: int
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        """Encode 8-byte dtypes through an unsigned integer view."""
        uint_dtype = np.uint64
        moved = np.moveaxis(np.ascontiguousarray(tensor).view(uint_dtype), axis, 0)
        anchor = np.take(tensor, [0], axis=axis).copy()
        deltas = np.zeros_like(moved)
        deltas[1:] = moved[1:] - moved[:-1]
        return np.moveaxis(deltas, 0, axis), anchor

    @staticmethod
    def _decode_bitwise(
        deltas: np.ndarray[Any, np.dtype[np.uint64]],
        anchor: np.ndarray[Any, np.dtype[Any]],
        axis: int,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode a bitwise delta stream back into the original dtype."""
        moved = np.moveaxis(deltas, axis, 0)
        anchor_bits = np.moveaxis(np.ascontiguousarray(anchor).view(np.uint64), axis, 0)
        restored = np.array(moved, copy=True)
        restored[0] = anchor_bits[0]
        restored = np.add.accumulate(restored, axis=0, dtype=np.uint64)
        restored = np.moveaxis(restored, 0, axis)
        return np.ascontiguousarray(restored).view(anchor.dtype)
