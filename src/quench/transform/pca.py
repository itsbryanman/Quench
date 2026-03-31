"""Principal-component transforms for tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PCAState:
    """Stored PCA state for explicit inversion."""

    components: np.ndarray[Any, np.dtype[np.float64]]
    mean: np.ndarray[Any, np.dtype[np.float64]]
    explained_variance_ratio: np.ndarray[Any, np.dtype[np.float64]]
    n_components: int
    original_shape: tuple[int, ...]
    sample_axis: int
    feature_shape: tuple[int, ...]
    dtype_orig: str


class PCATransform:
    """Fit a deterministic PCA basis and apply it to a tensor.

    The transformed representation is always a 2D matrix of shape
    ``(samples, components)``. For 2D weight tensors, rows are treated as
    samples. For tensors with 4 or more dimensions, axis 1 is treated as the
    head axis and every remaining dimension is flattened into features.
    """

    def fit_transform(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        n_components: int | None = None,
        variance_threshold: float = 0.99,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], PCAState]:
        """Fit PCA on *tensor* and return the transformed tensor plus state."""
        values = np.asarray(tensor)
        if values.ndim < 2:
            raise ValueError("PCA requires at least a 2D tensor")
        if values.size == 0:
            raise ValueError("Cannot fit PCA on an empty tensor")
        if not (0.0 < variance_threshold <= 1.0):
            raise ValueError("variance_threshold must be in (0, 1]")

        matrix, sample_axis, feature_shape = self._reshape_to_matrix(values)
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Tensor does not contain enough data for PCA")

        mean = np.mean(matrix, axis=0)
        centered = matrix - mean[None, :]
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

        if matrix.shape[0] > 1:
            explained_variance = (singular_values**2) / (matrix.shape[0] - 1)
        else:
            explained_variance = singular_values**2

        total_variance = float(np.sum(explained_variance))
        if total_variance > 0.0:
            explained_variance_ratio = explained_variance / total_variance
        else:
            explained_variance_ratio = np.zeros_like(explained_variance)

        max_components = vt.shape[0]
        if n_components is None:
            if max_components == 0:
                selected_components = 0
            elif total_variance == 0.0:
                selected_components = 1
            else:
                cumulative = np.cumsum(explained_variance_ratio)
                selected_components = int(
                    np.searchsorted(cumulative, variance_threshold, side="left") + 1
                )
        else:
            if not (1 <= n_components <= max_components):
                raise ValueError(
                    f"n_components must be in [1, {max_components}], got {n_components}"
                )
            selected_components = n_components

        selected_components = min(selected_components, max_components)
        components = vt[:selected_components].copy()
        transformed = centered @ components.T if selected_components else centered[:, :0]

        state = PCAState(
            components=components,
            mean=mean.copy(),
            explained_variance_ratio=explained_variance_ratio[:selected_components].copy(),
            n_components=selected_components,
            original_shape=tuple(values.shape),
            sample_axis=sample_axis,
            feature_shape=feature_shape,
            dtype_orig=values.dtype.str,
        )
        return transformed, state

    def inverse_transform(
        self,
        transformed: np.ndarray[Any, np.dtype[Any]],
        state: PCAState,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Reconstruct a tensor from a PCA-transformed matrix and saved *state*."""
        values = np.asarray(transformed, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("transformed must be a 2D matrix")
        if values.shape[1] != state.n_components:
            raise ValueError(
                "transformed component count does not match the saved PCA state"
            )
        if values.shape[0] != state.original_shape[state.sample_axis]:
            raise ValueError("transformed sample dimension does not match the PCA state")

        reconstructed = values @ state.components + state.mean[None, :]
        moved_shape = (state.original_shape[state.sample_axis], *state.feature_shape)
        restored_moved = reconstructed.reshape(moved_shape)
        restored = np.moveaxis(restored_moved, 0, state.sample_axis)
        return restored.astype(np.dtype(state.dtype_orig), copy=False)

    @staticmethod
    def _reshape_to_matrix(
        tensor: np.ndarray[Any, np.dtype[Any]]
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int, tuple[int, ...]]:
        """Reshape a tensor into ``(samples, features)`` for PCA."""
        working = np.asarray(tensor, dtype=np.float64)

        if working.ndim == 2:
            sample_axis = 0
            moved = working
        elif working.ndim >= 4:
            sample_axis = 1
            moved = np.moveaxis(working, sample_axis, 0)
        else:
            sample_axis = 0
            moved = working

        feature_shape = tuple(moved.shape[1:])
        matrix = moved.reshape(moved.shape[0], -1)
        return matrix, sample_axis, feature_shape
