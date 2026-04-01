"""Per-tensor delta analysis and compression path selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeltaProfile:
    """Statistics and routing decision for a single weight delta tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    num_elements: int
    sparsity: float
    max_abs: float
    mean_abs: float
    std: float
    recommended_bits: int
    recommended_path: str


def analyze_delta(
    delta: np.ndarray[Any, np.dtype[Any]],
    name: str,
    *,
    default_bits: int = 2,
) -> DeltaProfile:
    """Compute statistics for *delta* and choose a compression path.

    Paths:
        ``"zero"``
            Delta is effectively all-zero. Store nothing.
        ``"sign_scale"``
            BitDelta-style 1-bit sign mask + per-row FP16 scale.
            Chosen for large 2-D tensors whose element magnitudes
            have low coefficient of variation.
        ``"sparse"``
            Most of the delta is near-zero. Sparse index + value encoding.
        ``"lossless"``
            Tiny tensor where quantization overhead is not worth it.
        ``"quantize"``
            Standard per-channel normalize -> quantize -> entropy-code.
    """
    values = np.asarray(delta)
    flat = np.ravel(values.astype(np.float64, copy=False))
    abs_flat = np.abs(flat)
    num_elements = int(flat.size)
    sparsity = float(np.count_nonzero(abs_flat < 1e-7) / num_elements) if num_elements else 1.0
    max_abs = float(np.max(abs_flat)) if num_elements else 0.0
    mean_abs = float(np.mean(abs_flat)) if num_elements else 0.0
    std = float(np.std(flat)) if num_elements else 0.0

    if max_abs < 1e-8:
        path, bits = "zero", 0
    elif num_elements >= 1024 and values.ndim == 2 and _sign_scale_viable(flat, mean_abs):
        path, bits = "sign_scale", 1
    elif sparsity > 0.85 and num_elements >= 512:
        path, bits = "sparse", min(default_bits + 2, 8)
    elif num_elements < 512:
        path, bits = "lossless", 0
    else:
        path, bits = "quantize", default_bits

    return DeltaProfile(
        name=name,
        shape=tuple(int(dim) for dim in values.shape),
        dtype=values.dtype.str,
        num_elements=num_elements,
        sparsity=sparsity,
        max_abs=max_abs,
        mean_abs=mean_abs,
        std=std,
        recommended_bits=bits,
        recommended_path=path,
    )


def _sign_scale_viable(flat: np.ndarray[Any, np.dtype[Any]], mean_abs: float) -> bool:
    """Return True when 1-bit signs + row scale should preserve most information."""
    if mean_abs < 1e-10:
        return False
    abs_flat = np.abs(flat)
    cv = float(np.std(abs_flat) / mean_abs)
    return cv < 2.0
