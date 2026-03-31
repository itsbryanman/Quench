"""Uniform tensor quantization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quench.core.exceptions import QuantizationError
from quench.core.types import QuantMode


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


class UniformQuantizer:
    """Uniform tensor quantizer with auditable scalar parameters."""

    def quantize(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        mode: QuantMode,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], QuantParams]:
        """Quantize *tensor* with *bits* and *mode*."""
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
        qmin, qmax = self._quantized_bounds(bits, mode)

        if mode == QuantMode.SYMMETRIC:
            scaled = np.rint(working / params.scale)
        elif mode == QuantMode.ASYMMETRIC:
            scaled = np.rint(working / params.scale) + params.zero_point
        else:
            raise QuantizationError("UniformQuantizer does not support QuantMode.NONE")

        clipped = np.clip(scaled, qmin, qmax)
        return clipped.astype(self._storage_dtype(bits, mode), copy=False), params

    def dequantize(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        params: QuantParams,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Dequantize *quantized* values using *params*."""
        if params.scale <= 0.0:
            raise QuantizationError("Quantization scale must be positive")

        values = np.asarray(quantized, dtype=np.float64)
        if params.mode == QuantMode.SYMMETRIC:
            restored = values * params.scale
        elif params.mode == QuantMode.ASYMMETRIC:
            restored = (values - params.zero_point) * params.scale
        else:
            raise QuantizationError("UniformQuantizer does not support QuantMode.NONE")

        return restored.astype(np.dtype(params.dtype_orig), copy=False)

    @classmethod
    def _compute_params(
        cls,
        value_min: float,
        value_max: float,
        bits: int,
        mode: QuantMode,
        dtype_orig: str,
    ) -> QuantParams:
        """Compute quantization parameters from a clipped value range."""
        cls._validate_bits(bits)

        if mode == QuantMode.SYMMETRIC:
            qmin, qmax = cls._quantized_bounds(bits, mode)
            max_abs = max(abs(value_min), abs(value_max))
            scale = max_abs / max(abs(qmin), qmax, 1)
            scale = 1.0 if scale == 0.0 or not np.isfinite(scale) else float(scale)
            zero_point = 0
        elif mode == QuantMode.ASYMMETRIC:
            qmin, qmax = cls._quantized_bounds(bits, mode)
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
            raise QuantizationError("UniformQuantizer does not support QuantMode.NONE")

        return QuantParams(
            scale=float(scale),
            zero_point=int(zero_point),
            bits=bits,
            mode=mode,
            dtype_orig=dtype_orig,
            value_range_min=float(value_min),
            value_range_max=float(value_max),
        )

    @staticmethod
    def _validate_bits(bits: int) -> None:
        """Validate the requested bit width."""
        if not (1 <= bits <= 32):
            raise QuantizationError(f"bits must be in [1, 32], got {bits}")

    @staticmethod
    def _quantized_bounds(bits: int, mode: QuantMode) -> tuple[int, int]:
        """Return the representable integer bounds for *bits* and *mode*."""
        if mode == QuantMode.SYMMETRIC:
            qmax = 1 if bits == 1 else (1 << (bits - 1)) - 1
            qmin = -qmax
            return qmin, qmax
        if mode == QuantMode.ASYMMETRIC:
            return 0, (1 << bits) - 1
        raise QuantizationError("UniformQuantizer does not support QuantMode.NONE")

    @staticmethod
    def _storage_dtype(bits: int, mode: QuantMode) -> type[np.generic]:
        """Choose the smallest practical storage dtype for the quantized tensor."""
        if mode == QuantMode.SYMMETRIC:
            if bits <= 8:
                return np.int8
            if bits <= 16:
                return np.int16
            return np.int32

        if bits <= 8:
            return np.uint8
        if bits <= 16:
            return np.uint16
        return np.uint32
