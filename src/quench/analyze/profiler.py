"""Tensor profiling utilities."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.core.types import TensorStats


class TensorProfiler:
    """Compute descriptive tensor statistics.

    Expensive metrics use deterministic bounded sampling:
    - entropy is estimated from at most ``max_entropy_samples`` strided elements.
    - effective rank uses a strided 2D submatrix capped at
      ``max_svd_rows`` x ``max_svd_cols`` before running SVD.
    """

    def __init__(
        self,
        max_entropy_samples: int = 131_072,
        histogram_bins: int = 256,
        max_svd_rows: int = 256,
        max_svd_cols: int = 256,
    ) -> None:
        if max_entropy_samples <= 0:
            raise ValueError("max_entropy_samples must be positive")
        if histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")
        if max_svd_rows <= 0 or max_svd_cols <= 0:
            raise ValueError("max_svd_rows and max_svd_cols must be positive")

        self._max_entropy_samples = max_entropy_samples
        self._histogram_bins = histogram_bins
        self._max_svd_rows = max_svd_rows
        self._max_svd_cols = max_svd_cols

    def profile(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> TensorStats:
        """Profile *tensor* and return summary statistics."""
        values = np.asarray(tensor)
        if values.size == 0:
            raise ValueError("Cannot profile an empty tensor")

        working = values.astype(np.float64, copy=False)
        mean = float(np.mean(working))
        std = float(np.std(working))
        min_val = float(np.min(working))
        max_val = float(np.max(working))
        sparsity = float(np.count_nonzero(np.abs(working) < 1e-6) / working.size)
        entropy_bits = self._estimate_entropy(values)
        effective_rank = self._estimate_effective_rank(working) if working.ndim == 2 else None

        return TensorStats(
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            sparsity=sparsity,
            entropy_bits=entropy_bits,
            effective_rank=effective_rank,
        )

    def _estimate_entropy(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Estimate Shannon entropy in bits per element."""
        sample = self._sample_flat(tensor)
        if sample.size <= 1:
            return 0.0

        if np.issubdtype(sample.dtype, np.integer) or np.issubdtype(sample.dtype, np.bool_):
            _, counts = np.unique(sample, return_counts=True)
            probs = counts.astype(np.float64) / counts.sum()
        else:
            sample_f = sample.astype(np.float64, copy=False)
            sample_min = float(np.min(sample_f))
            sample_max = float(np.max(sample_f))
            if sample_min == sample_max:
                return 0.0
            hist, _ = np.histogram(
                sample_f,
                bins=min(self._histogram_bins, max(8, int(np.sqrt(sample_f.size)))),
                range=(sample_min, sample_max),
            )
            hist = hist[hist > 0]
            probs = hist.astype(np.float64) / hist.sum()

        return float(-np.sum(probs * np.log2(probs)))

    def _estimate_effective_rank(
        self, matrix: np.ndarray[Any, np.dtype[np.float64]]
    ) -> float:
        """Estimate the Shannon effective rank of a 2D tensor."""
        sampled = self._sample_matrix(matrix)
        centered = sampled - np.mean(sampled, axis=0, keepdims=True)

        if not np.any(centered):
            return 1.0

        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        singular_values = singular_values[singular_values > 0.0]
        if singular_values.size == 0:
            return 1.0

        probs = singular_values / singular_values.sum()
        entropy = -np.sum(probs * np.log(probs))
        return float(np.exp(entropy))

    def _sample_flat(
        self, tensor: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Take a deterministic strided sample from a flattened tensor."""
        flat = np.ravel(np.asarray(tensor))
        if flat.size <= self._max_entropy_samples:
            return flat

        step = int(np.ceil(flat.size / self._max_entropy_samples))
        return flat[::step][: self._max_entropy_samples]

    def _sample_matrix(
        self, matrix: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Take a deterministic 2D submatrix for SVD-based metrics."""
        sampled = matrix
        if sampled.shape[0] > self._max_svd_rows:
            row_step = int(np.ceil(sampled.shape[0] / self._max_svd_rows))
            sampled = sampled[::row_step][: self._max_svd_rows]
        if sampled.shape[1] > self._max_svd_cols:
            col_step = int(np.ceil(sampled.shape[1] / self._max_svd_cols))
            sampled = sampled[:, ::col_step][:, : self._max_svd_cols]
        return sampled
