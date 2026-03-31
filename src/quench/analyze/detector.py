"""Heuristic tensor type detection."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.core.types import TensorType


class TensorTypeDetector:
    """Classify tensors into coarse semantic buckets.

    The detector uses deterministic substring and shape/statistical heuristics
    instead of model-specific rules so the behaviour remains stable across runs.
    """

    def __init__(
        self,
        gaussian_sample_size: int = 65_536,
        activation_sparsity_threshold: float = 0.5,
    ) -> None:
        if gaussian_sample_size <= 0:
            raise ValueError("gaussian_sample_size must be positive")
        self._gaussian_sample_size = gaussian_sample_size
        self._activation_sparsity_threshold = activation_sparsity_threshold

    def detect(
        self, tensor: np.ndarray[Any, np.dtype[Any]], name: str | None = None
    ) -> TensorType:
        """Detect the most likely tensor type for *tensor* and *name*."""
        values = np.asarray(tensor)
        if values.size == 0:
            return TensorType.UNKNOWN

        lower_name = (name or "").lower()

        if self._contains_any(lower_name, ("embed",)):
            return TensorType.EMBEDDING

        if values.ndim == 4 and self._contains_any(
            lower_name, ("k_cache", "v_cache", "key", "value")
        ):
            return TensorType.KV_CACHE

        if self._contains_any(lower_name, ("weight", "lm_head", "mlp", "attn")):
            return TensorType.WEIGHT

        if self._estimate_sparsity(values) > self._activation_sparsity_threshold:
            return TensorType.ACTIVATION

        if values.ndim == 2 and self._is_near_gaussian(values):
            return TensorType.WEIGHT

        if values.ndim >= 4:
            if self._looks_like_kv_cache(values):
                return TensorType.KV_CACHE
            return TensorType.ACTIVATION

        return TensorType.UNKNOWN

    @staticmethod
    def _contains_any(name: str, needles: tuple[str, ...]) -> bool:
        """Return ``True`` when *name* contains any needle."""
        return any(needle in name for needle in needles)

    @staticmethod
    def _estimate_sparsity(tensor: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Estimate exact tensor sparsity using the project-wide near-zero rule."""
        values = np.asarray(tensor, dtype=np.float64)
        return float(np.count_nonzero(np.abs(values) < 1e-6) / values.size)

    def _is_near_gaussian(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> bool:
        """Check whether a 2D tensor looks approximately Gaussian."""
        sample = self._sample_values(tensor)
        if sample.size < 16:
            return False

        mean = float(np.mean(sample))
        std = float(np.std(sample))
        if std <= 1e-12:
            return False

        centered = (sample - mean) / std
        skew = float(np.mean(centered**3))
        kurtosis = float(np.mean(centered**4))
        return abs(skew) <= 0.5 and 2.0 <= kurtosis <= 4.5

    @staticmethod
    def _looks_like_kv_cache(tensor: np.ndarray[Any, np.dtype[Any]]) -> bool:
        """Use shape heuristics for cache-like tensors with an explicit token axis."""
        shape = tensor.shape
        if len(shape) < 4:
            return False

        non_batch_dims = shape[1:]
        has_token_axis = any(dim >= 64 for dim in non_batch_dims)
        has_head_axis = any(2 <= dim <= 128 for dim in non_batch_dims[:-1] or non_batch_dims)
        feature_axis_small = shape[-1] <= 256
        return has_token_axis and has_head_axis and feature_axis_small

    def _sample_values(
        self, tensor: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Take a deterministic strided sample for moment estimation."""
        flat = np.ravel(np.asarray(tensor, dtype=np.float64))
        if flat.size <= self._gaussian_sample_size:
            return flat

        step = int(np.ceil(flat.size / self._gaussian_sample_size))
        return flat[::step][: self._gaussian_sample_size]
