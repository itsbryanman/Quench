"""Tests for reversible transform primitives."""
from __future__ import annotations

import numpy as np
import pytest

from quench.transform import (
    ChannelNormalizer,
    DeltaCoder,
    PCATransform,
    SparseEncoder,
    StepMetadata,
    TransformPipeline,
)


class TestChannelNormalizer:
    def test_normalize_denormalize_roundtrip(self) -> None:
        normalizer = ChannelNormalizer()
        tensor = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-2.0, -1.0, 1.0, 2.0],
                [3.0, 3.5, 4.0, 4.5],
            ],
            dtype=np.float32,
        )

        normalized, scales, zero_points = normalizer.normalize(tensor, axis=0)
        restored = normalizer.denormalize(normalized, scales, zero_points, axis=0)

        assert scales[0] == pytest.approx(1.0)
        assert zero_points[0] == pytest.approx(0.0)
        np.testing.assert_allclose(restored, tensor, atol=1e-6)

    def test_handles_tiny_and_constant_tensors(self) -> None:
        normalizer = ChannelNormalizer()
        tiny = np.array([[5.0]], dtype=np.float32)

        normalized, scales, zero_points = normalizer.normalize(tiny, axis=0)
        restored = normalizer.denormalize(normalized, scales, zero_points, axis=0)

        np.testing.assert_allclose(normalized, np.zeros_like(normalized))
        np.testing.assert_allclose(restored, tiny, atol=1e-6)


class TestPCATransform:
    def test_variance_retention_threshold(self) -> None:
        pca = PCATransform()
        rng = np.random.default_rng(123)
        left = rng.normal(size=(128, 3))
        right = rng.normal(size=(3, 64))
        matrix = (left @ right + 0.01 * rng.normal(size=(128, 64))).astype(np.float32)

        transformed, state = pca.fit_transform(matrix, variance_threshold=0.99)

        assert transformed.shape[1] == state.n_components
        assert np.sum(state.explained_variance_ratio) >= 0.99
        assert state.n_components <= 4

    def test_inverse_roundtrip_with_kv_like_tensor(self) -> None:
        pca = PCATransform()
        rng = np.random.default_rng(5)
        tensor = rng.normal(size=(2, 6, 32, 16)).astype(np.float32)

        transformed, state = pca.fit_transform(tensor, n_components=6)
        restored = pca.inverse_transform(transformed, state)

        np.testing.assert_allclose(restored, tensor, atol=1e-5)

    def test_constant_tensor_is_safe(self) -> None:
        pca = PCATransform()
        tensor = np.full((8, 4), 3.0, dtype=np.float32)

        transformed, state = pca.fit_transform(tensor, variance_threshold=0.99)
        restored = pca.inverse_transform(transformed, state)

        assert state.n_components == 1
        np.testing.assert_allclose(restored, tensor, atol=1e-6)


class TestDeltaCoder:
    def test_delta_roundtrip_exact_for_float32(self) -> None:
        coder = DeltaCoder()
        rng = np.random.default_rng(99)
        tensor = rng.normal(size=(16, 32)).astype(np.float32)

        deltas, anchor = coder.encode(tensor, axis=1)
        restored = coder.decode(deltas, anchor, axis=1)

        np.testing.assert_array_equal(restored, tensor)

    def test_delta_roundtrip_exact_for_float64_bitwise_path(self) -> None:
        coder = DeltaCoder()
        rng = np.random.default_rng(11)
        tensor = rng.normal(size=(4, 8)).astype(np.float64)

        deltas, anchor = coder.encode(tensor, axis=0)
        restored = coder.decode(deltas, anchor, axis=0)

        np.testing.assert_array_equal(restored, tensor)


class TestSparseEncoder:
    def test_sparse_roundtrip_exactness(self) -> None:
        encoder = SparseEncoder()
        tensor = np.array(
            [[0.0, 1.5, 0.0], [0.0, 0.0, -2.0], [3.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        sparse = encoder.encode(tensor)
        restored = encoder.decode(sparse)

        assert sparse.nnz == 3
        np.testing.assert_array_equal(restored, tensor)

    def test_sparse_rejects_lossy_threshold(self) -> None:
        encoder = SparseEncoder()
        tensor = np.array([0.0, 1e-7, 0.0], dtype=np.float32)

        with pytest.raises(ValueError):
            encoder.encode(tensor, threshold=1e-6)


class TestTransformPipeline:
    def test_pipeline_composition_and_inversion(self) -> None:
        normalizer = ChannelNormalizer()
        delta = DeltaCoder()
        pipeline = TransformPipeline()

        def normalize_step(
            tensor: np.ndarray,
        ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
            normalized, scales, zero_points = normalizer.normalize(tensor, axis=0)
            return normalized, (scales, zero_points)

        def normalize_inverse(
            tensor: np.ndarray, metadata: tuple[np.ndarray, np.ndarray]
        ) -> np.ndarray:
            scales, zero_points = metadata
            return normalizer.denormalize(tensor, scales, zero_points, axis=0)

        pipeline.add_step("normalize", normalize_step, normalize_inverse)
        pipeline.add_step(
            "delta",
            lambda tensor: delta.encode(tensor, axis=1),
            lambda tensor, anchor: delta.decode(tensor, anchor, axis=1),
        )

        rng = np.random.default_rng(21)
        tensor = rng.normal(size=(12, 24)).astype(np.float32)
        transformed, metadata = pipeline.forward(tensor)
        restored = pipeline.inverse(transformed, metadata)

        np.testing.assert_allclose(restored, tensor, atol=1e-6)

    def test_pipeline_rejects_malformed_metadata(self) -> None:
        pipeline = TransformPipeline()
        pipeline.add_step("identity", lambda tensor: tensor, lambda tensor: tensor)

        with pytest.raises(ValueError):
            pipeline.inverse(
                np.ones((2, 2), dtype=np.float32),
                [StepMetadata(name="wrong", payload=None, input_shape=(2, 2), output_shape=(2, 2))],
            )

    def test_large_tensor_edge_case(self) -> None:
        normalizer = ChannelNormalizer()
        rng = np.random.default_rng(1234)
        tensor = rng.normal(size=(128, 4096)).astype(np.float32)

        normalized, scales, zero_points = normalizer.normalize(tensor, axis=0)
        restored = normalizer.denormalize(normalized, scales, zero_points, axis=0)

        np.testing.assert_allclose(restored, tensor, atol=1e-5)
