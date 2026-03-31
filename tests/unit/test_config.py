"""Tests for QuenchConfig."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from quench.core.config import CalibrationPolicyKind, QuantizationGranularity, QuenchConfig
from quench.core.types import CodecMode, QuantMode


class TestQuenchConfig:
    def test_defaults(self) -> None:
        cfg = QuenchConfig()
        assert cfg.codec_mode == CodecMode.LOSSY
        assert cfg.target_bits == 4
        assert cfg.quant_mode == QuantMode.SYMMETRIC
        assert cfg.entropy_coder == "rans"
        assert cfg.pca_variance_threshold == 0.99
        assert cfg.delta_coding is True
        assert cfg.per_channel is True
        assert cfg.quantization_granularity == QuantizationGranularity.PER_CHANNEL
        assert cfg.calibration_policy == CalibrationPolicyKind.MINMAX
        assert cfg.block_size == 128
        assert cfg.pack_bits is False
        assert cfg.entropy_backend == "python"
        assert cfg.packing_backend == "python"

    def test_target_bits_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="target_bits"):
            QuenchConfig(target_bits=0)

    def test_target_bits_33_raises(self) -> None:
        with pytest.raises(ValueError, match="target_bits"):
            QuenchConfig(target_bits=33)

    def test_target_bits_valid_boundaries(self) -> None:
        assert QuenchConfig(target_bits=1).target_bits == 1
        assert QuenchConfig(target_bits=32).target_bits == 32

    def test_pca_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="pca_variance_threshold"):
            QuenchConfig(pca_variance_threshold=0.0)

    def test_pca_threshold_one_passes(self) -> None:
        cfg = QuenchConfig(pca_variance_threshold=1.0)
        assert cfg.pca_variance_threshold == 1.0

    def test_yaml_roundtrip(self) -> None:
        cfg = QuenchConfig(target_bits=8, entropy_coder="rans", per_channel=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            cfg.to_yaml(path)
            loaded = QuenchConfig.from_yaml(path)
        assert loaded.target_bits == cfg.target_bits
        assert loaded.entropy_coder == cfg.entropy_coder
        assert loaded.per_channel == cfg.per_channel
        assert loaded.codec_mode == cfg.codec_mode

    def test_invalid_calibration_combination_raises(self) -> None:
        with pytest.raises(ValueError, match="per_tensor quantization"):
            QuenchConfig(
                quantization_granularity=QuantizationGranularity.PER_TENSOR,
                calibration_policy=CalibrationPolicyKind.BLOCKWISE,
            )

    def test_quant_mode_none_requires_lossless(self) -> None:
        with pytest.raises(ValueError, match="QuantMode.NONE"):
            QuenchConfig(quant_mode=QuantMode.NONE)

    def test_legacy_per_channel_syncs_granularity(self) -> None:
        cfg = QuenchConfig(per_channel=False)
        assert cfg.quantization_granularity == QuantizationGranularity.PER_TENSOR

    def test_blockwise_defaults_validate(self) -> None:
        cfg = QuenchConfig(
            quantization_granularity=QuantizationGranularity.BLOCKWISE,
            calibration_policy=CalibrationPolicyKind.BLOCKWISE,
            block_size=64,
        )
        assert cfg.block_size == 64
