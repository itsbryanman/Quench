"""Tests for tensor analysis helpers."""
from __future__ import annotations

import numpy as np

from quench.analyze import TensorProfiler, TensorTypeDetector
from quench.core.types import TensorType


class TestTensorTypeDetector:
    def test_name_heuristics(self) -> None:
        detector = TensorTypeDetector()
        rng = np.random.default_rng(99)

        embedding = rng.normal(size=(128, 64)).astype(np.float32)
        kv_cache = rng.normal(size=(2, 8, 128, 64)).astype(np.float32)
        weight = rng.normal(size=(64, 64)).astype(np.float32)

        assert detector.detect(embedding, "token_embed.weight") == TensorType.EMBEDDING
        assert detector.detect(kv_cache, "layer_0.k_cache") == TensorType.KV_CACHE
        assert detector.detect(weight, "mlp.weight") == TensorType.WEIGHT

    def test_statistical_heuristics(self) -> None:
        detector = TensorTypeDetector()
        rng = np.random.default_rng(42)

        gaussian_weight = rng.normal(size=(256, 128)).astype(np.float32)
        sparse_activation = np.zeros((8, 128, 128), dtype=np.float32)
        sparse_activation[:, :, :8] = rng.uniform(0.1, 2.0, size=(8, 128, 8)).astype(np.float32)

        assert detector.detect(gaussian_weight) == TensorType.WEIGHT
        assert detector.detect(sparse_activation) == TensorType.ACTIVATION

    def test_causal_mask_detected_as_mask(self) -> None:
        """GPT-2 style attn.bias is a lower-triangular causal mask, not a bias vector."""
        detector = TensorTypeDetector()
        mask = np.tril(np.ones((1024, 1024), dtype=np.float32))
        assert detector.detect(mask, name="h.0.attn.bias") == TensorType.MASK

    def test_neg_inf_mask_detected(self) -> None:
        detector = TensorTypeDetector()
        mask = np.where(np.tril(np.ones((512, 512))), 0.0, float('-inf')).astype(np.float32)
        assert detector.detect(mask, name="attn.masked_bias") == TensorType.MASK

    def test_real_bias_still_detected(self) -> None:
        detector = TensorTypeDetector()
        bias = np.random.randn(768).astype(np.float32) * 0.01
        assert detector.detect(bias, name="mlp.bias") == TensorType.BIAS

    def test_small_1d_without_bias_name_not_bias(self) -> None:
        """1D tensors without 'bias' in name should NOT be classified as BIAS."""
        detector = TensorTypeDetector()
        t = np.ones(128, dtype=np.float32)
        result = detector.detect(t, name="model.layers.0.input_layernorm.weight")
        assert result != TensorType.BIAS

    def test_optimizer_bias_and_mixed_precision_heuristics(self) -> None:
        detector = TensorTypeDetector()
        optimizer = np.ones((32, 32), dtype=np.float32)
        bias = np.random.randn(64).astype(np.float32) * 0.01
        mixed = np.random.randn(16, 16).astype(np.float16)

        assert detector.detect(optimizer, "optimizer.exp_avg") == TensorType.OPTIMIZER_STATE
        assert detector.detect(bias, "proj.bias") == TensorType.BIAS
        assert detector.detect(mixed, "mlp.fp16_gate") == TensorType.MIXED_PRECISION


class TestTensorProfiler:
    def test_profile_2d_tensor_includes_effective_rank(self) -> None:
        profiler = TensorProfiler()
        left = np.array([[1.0], [0.5], [-1.5], [2.0]], dtype=np.float32)
        right = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)
        matrix = left @ right

        stats = profiler.profile(matrix)

        assert stats.std > 0.0
        assert stats.entropy_bits >= 0.0
        assert stats.effective_rank is not None
        assert stats.effective_rank < 1.5

    def test_profile_large_tensor_uses_sampling_without_failures(self) -> None:
        profiler = TensorProfiler(max_entropy_samples=4096, max_svd_rows=32, max_svd_cols=32)
        rng = np.random.default_rng(7)
        tensor = rng.normal(size=(512, 512)).astype(np.float32)

        stats = profiler.profile(tensor)

        assert np.isfinite(stats.mean)
        assert np.isfinite(stats.entropy_bits)
        assert stats.effective_rank is not None
        assert stats.effective_rank > 1.0
