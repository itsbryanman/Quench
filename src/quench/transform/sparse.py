"""Lossless sparse tensor encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SparseRepresentation:
    """Compact sparse tensor storage using flat indices."""

    indices: np.ndarray[Any, np.dtype[Any]]
    values: np.ndarray[Any, np.dtype[Any]]
    shape: tuple[int, ...]
    nnz: int
    dtype_orig: str


class SparseEncoder:
    """Encode tensors with explicit zeros into a sparse representation."""

    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], threshold: float = 1e-6
    ) -> SparseRepresentation:
        """Encode *tensor* sparsely while preserving exact reconstruction."""
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")

        values = np.asarray(tensor)
        flat = values.reshape(-1)
        mask = np.abs(flat) > threshold

        discarded = flat[~mask]
        if np.any(discarded != 0):
            raise ValueError(
                "Sparse encoding would drop non-zero values; lower the threshold for exactness"
            )

        indices = np.flatnonzero(mask)
        index_dtype = np.uint32 if flat.size <= np.iinfo(np.uint32).max else np.uint64

        return SparseRepresentation(
            indices=indices.astype(index_dtype, copy=False),
            values=flat[indices].copy(),
            shape=tuple(values.shape),
            nnz=int(indices.size),
            dtype_orig=values.dtype.str,
        )

    def decode(self, sparse: SparseRepresentation) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode a :class:`SparseRepresentation` back into a dense tensor."""
        if sparse.nnz != len(sparse.indices) or sparse.nnz != len(sparse.values):
            raise ValueError("SparseRepresentation nnz does not match stored arrays")

        dtype = np.dtype(sparse.dtype_orig)
        size = int(np.prod(sparse.shape, dtype=np.int64))
        dense = np.zeros(size, dtype=dtype)

        if sparse.nnz:
            indices = sparse.indices.astype(np.intp, copy=False)
            if np.any(indices < 0) or np.any(indices >= size):
                raise ValueError("SparseRepresentation indices are out of bounds")
            dense[indices] = sparse.values.astype(dtype, copy=False)

        return dense.reshape(sparse.shape)
