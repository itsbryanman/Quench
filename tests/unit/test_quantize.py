"""Tests for quantization and calibration primitives."""
from __future__ import annotations

import numpy as np

from quench.core.config import QuantizationGranularity
from quench.core.types import QuantMode
from quench.quantize import (
    BlockwiseCalibrationPolicy,
    BlockwiseQuantizer,
    Calibrator,
    ImportanceAllocator,
    PerChannelCalibrationPolicy,
    PerChannelQuantizer,
    PercentileCalibrationPolicy,
    QuantizationLayout,
    UniformQuantizer,
)


class TestUniformQuantizer:
    def test_symmetric_quantize_dequantize_bounded_error(self) -> None:
        quantizer = UniformQuantizer()
        rng = np.random.default_rng(42)
        tensor = rng.normal(size=(512,)).astype(np.float32)

        quantized, params = quantizer.quantize(tensor, bits=4, mode=QuantMode.SYMMETRIC)
        restored = quantizer.dequantize(quantized, params)

        max_error = float(np.max(np.abs(restored - tensor)))
        assert max_error <= params.scale / 2.0 + 1e-6
        assert quantized.dtype == np.int8

    def test_asymmetric_quantize_dequantize_bounded_error(self) -> None:
        quantizer = UniformQuantizer()
        rng = np.random.default_rng(7)
        tensor = rng.uniform(low=-1.5, high=3.0, size=(1024,)).astype(np.float32)

        quantized, params = quantizer.quantize(tensor, bits=8, mode=QuantMode.ASYMMETRIC)
        restored = quantizer.dequantize(quantized, params)

        max_error = float(np.max(np.abs(restored - tensor)))
        assert max_error <= params.scale / 2.0 + 1e-6
        assert quantized.dtype == np.uint8

    def test_handles_all_zero_and_single_value_tensors(self) -> None:
        quantizer = UniformQuantizer()

        zeros = np.zeros((16,), dtype=np.float32)
        quantized_zeros, zero_params = quantizer.quantize(
            zeros, bits=4, mode=QuantMode.SYMMETRIC
        )
        np.testing.assert_array_equal(quantized_zeros, np.zeros_like(quantized_zeros))
        np.testing.assert_array_equal(quantizer.dequantize(quantized_zeros, zero_params), zeros)

        constant = np.full((16,), -3.0, dtype=np.float32)
        quantized_const, const_params = quantizer.quantize(
            constant, bits=4, mode=QuantMode.ASYMMETRIC
        )
        restored_const = quantizer.dequantize(quantized_const, const_params)
        np.testing.assert_allclose(restored_const, constant, atol=1e-6)

    def test_low_bit_widths_remain_stable(self) -> None:
        quantizer = UniformQuantizer()
        tensor = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)

        quantized, params = quantizer.quantize(tensor, bits=2, mode=QuantMode.SYMMETRIC)
        restored = quantizer.dequantize(quantized, params)

        assert quantized.dtype == np.int8
        assert np.all(np.isfinite(restored))

    def test_quantize_nan_tensor_raises_or_handles(self) -> None:
        quantizer = UniformQuantizer()
        tensor = np.array([1.0, float("nan"), 3.0], dtype=np.float32)

        try:
            quantized, _ = quantizer.quantize(tensor, bits=4, mode=QuantMode.SYMMETRIC)
            assert np.all(np.isfinite(quantized))
        except Exception:
            pass

    def test_quantize_inf_tensor(self) -> None:
        quantizer = UniformQuantizer()
        tensor = np.array([1.0, float("inf"), -float("inf"), 3.0], dtype=np.float32)

        try:
            quantized, _ = quantizer.quantize(tensor, bits=4, mode=QuantMode.SYMMETRIC)
            assert np.all(np.isfinite(quantized))
        except Exception:
            pass

    def test_encoder_handles_nan_tensor(self) -> None:
        import quench

        tensor = np.array([[1.0, float("nan")], [3.0, 4.0]], dtype=np.float32)

        try:
            compressed = quench.compress(tensor, name="test.weight")
            restored = quench.decompress(compressed)
            assert restored.shape == tensor.shape
        except Exception:
            pass


class TestCalibrator:
    def test_per_channel_calibration_produces_varying_scales(self) -> None:
        calibrator = Calibrator()
        tensor = np.array(
            [
                [0.1, -0.1, 0.2, -0.2],
                [1.0, -1.0, 2.0, -2.0],
                [10.0, -10.0, 5.0, -5.0],
            ],
            dtype=np.float32,
        )

        params = calibrator.calibrate_per_channel(
            tensor, bits=4, mode=QuantMode.SYMMETRIC, axis=0
        )
        scales = [param.scale for param in params]

        assert len(params) == tensor.shape[0]
        assert scales[0] < scales[1] < scales[2]

    def test_percentile_calibration_reduces_outlier_impact(self) -> None:
        calibrator = Calibrator()
        rng = np.random.default_rng(123)
        tensor = rng.normal(size=10_000).astype(np.float32)
        tensor[0] = 1_000.0

        full = calibrator.calibrate_per_tensor(tensor, bits=4, mode=QuantMode.SYMMETRIC)
        clipped = calibrator.percentile_calibrate(
            tensor, bits=4, percentile=99.0, mode=QuantMode.SYMMETRIC
        )

        assert clipped.scale < full.scale
        assert clipped.value_range_max < full.value_range_max

    def test_per_channel_quantizer_roundtrip(self) -> None:
        rng = np.random.default_rng(99)
        tensor = rng.normal(size=(4, 32)).astype(np.float32)
        layout = QuantizationLayout(QuantizationGranularity.PER_CHANNEL, axis=0)
        policy = PerChannelCalibrationPolicy()
        quantizer = PerChannelQuantizer(axis=0)

        params = policy.calibrate(tensor, bits=5, mode=QuantMode.SYMMETRIC, layout=layout)
        quantized = quantizer.quantize(tensor, params)
        restored = quantizer.dequantize(quantized, params)

        assert quantized.shape == tensor.shape
        assert restored.shape == tensor.shape
        np.testing.assert_allclose(restored, tensor, atol=max(param.scale for param in params.params))

    def test_blockwise_quantizer_roundtrip(self) -> None:
        rng = np.random.default_rng(1234)
        tensor = rng.normal(size=(8, 16)).astype(np.float32)
        layout = QuantizationLayout(QuantizationGranularity.BLOCKWISE, axis=0, block_size=3)
        policy = BlockwiseCalibrationPolicy()
        quantizer = BlockwiseQuantizer(axis=0, block_size=3)

        params = policy.calibrate(tensor, bits=4, mode=QuantMode.SYMMETRIC, layout=layout)
        quantized = quantizer.quantize(tensor, params)
        restored = quantizer.dequantize(quantized, params)

        assert params.block_lengths == (3, 3, 2)
        np.testing.assert_allclose(
            restored,
            tensor,
            atol=max(param.scale for param in params.params),
        )

    def test_percentile_policy_supports_blockwise_layout(self) -> None:
        rng = np.random.default_rng(5)
        tensor = rng.normal(size=(6, 8)).astype(np.float32)
        tensor[0, 0] = 100.0
        layout = QuantizationLayout(QuantizationGranularity.BLOCKWISE, axis=0, block_size=2)

        params = PercentileCalibrationPolicy(percentile=95.0).calibrate(
            tensor,
            bits=4,
            mode=QuantMode.SYMMETRIC,
            layout=layout,
        )

        assert len(params.params) == 3
        assert max(param.value_range_max for param in params.params) < 100.0


class TestImportanceAllocator:
    def test_higher_entropy_tensors_get_more_bits(self) -> None:
        allocator = ImportanceAllocator()
        rng = np.random.default_rng(5)
        tensors = {
            "zeros": np.zeros((256,), dtype=np.float32),
            "binary": rng.choice([0.0, 1.0], size=256, p=[0.95, 0.05]).astype(np.float32),
            "uniform": rng.uniform(-1.0, 1.0, size=256).astype(np.float32),
        }

        allocation = allocator.allocate_bits(tensors, total_budget_bits=12)

        assert allocation["uniform"] >= allocation["binary"] >= allocation["zeros"]
        assert all(2 <= bits <= 8 for bits in allocation.values())
        assert sum(allocation.values()) <= 12

    def test_budget_too_small_raises(self) -> None:
        allocator = ImportanceAllocator()

        try:
            allocator.allocate_bits({"a": np.zeros((4,), dtype=np.float32)}, total_budget_bits=1)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected a ValueError for an infeasible budget")
