"""Quench configuration via Pydantic v2."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from quench.core.types import CodecMode, QuantMode


class QuantizationGranularity(str, Enum):
    """Supported quantization layouts for lossy tensor coding."""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    BLOCKWISE = "blockwise"


class CalibrationPolicyKind(str, Enum):
    """Supported calibration policies for computing quantization ranges."""

    MINMAX = "minmax"
    PERCENTILE = "percentile"
    PER_CHANNEL = "per_channel"
    BLOCKWISE = "blockwise"


class QuenchConfig(BaseModel):
    """Top-level configuration for the Quench codec.

    Defaults are selected for deterministic CPU-only operation:
    - `quantization_granularity="per_channel"`
    - `block_size=128` for blockwise layouts
    - `calibration_policy="minmax"`
    - `percentile_value=99.9`
    - `entropy_backend="python"`
    - `packing_backend="python"`
    """

    model_config = ConfigDict(extra="forbid")

    codec_mode: CodecMode = CodecMode.LOSSY
    target_bits: int = 4
    quant_mode: QuantMode = QuantMode.SYMMETRIC
    entropy_coder: str = "rans"
    pca_variance_threshold: float = 0.99
    delta_coding: bool = True
    per_channel: bool = True
    quantization_granularity: QuantizationGranularity = QuantizationGranularity.PER_CHANNEL
    quantization_axis: int | None = None
    block_size: int = 128
    calibration_policy: CalibrationPolicyKind = CalibrationPolicyKind.MINMAX
    percentile_value: float = 99.9
    pack_bits: bool = False
    entropy_backend: str = "python"
    packing_backend: str = "python"

    @model_validator(mode="before")
    @classmethod
    def _sync_legacy_per_channel(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        granularity_present = "quantization_granularity" in normalized
        per_channel_present = "per_channel" in normalized

        if granularity_present and not per_channel_present:
            granularity = normalized["quantization_granularity"]
            granularity_value = granularity.value if isinstance(granularity, QuantizationGranularity) else str(granularity)
            normalized["per_channel"] = (
                granularity_value == QuantizationGranularity.PER_CHANNEL.value
            )
        elif per_channel_present and not granularity_present:
            normalized["quantization_granularity"] = (
                QuantizationGranularity.PER_CHANNEL.value
                if bool(normalized["per_channel"])
                else QuantizationGranularity.PER_TENSOR.value
            )
        return normalized

    @field_validator("target_bits")
    @classmethod
    def _validate_target_bits(cls, value: int) -> int:
        if not (1 <= value <= 32):
            raise ValueError(f"target_bits must be in [1, 32], got {value}")
        return value

    @field_validator("pca_variance_threshold")
    @classmethod
    def _validate_pca(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"pca_variance_threshold must be in (0, 1], got {value}")
        return value

    @field_validator("entropy_coder")
    @classmethod
    def _validate_entropy_coder(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"rans", "raw"}:
            raise ValueError("entropy_coder must be one of: rans, raw")
        return normalized

    @field_validator("block_size")
    @classmethod
    def _validate_block_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"block_size must be positive, got {value}")
        return value

    @field_validator("percentile_value")
    @classmethod
    def _validate_percentile(cls, value: float) -> float:
        if not (0.0 < value <= 100.0):
            raise ValueError(f"percentile_value must be in (0, 100], got {value}")
        return value

    @field_validator("entropy_backend", "packing_backend")
    @classmethod
    def _validate_backend_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("backend names must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_combinations(self) -> QuenchConfig:
        if self.quantization_granularity == QuantizationGranularity.PER_TENSOR:
            if self.calibration_policy in {
                CalibrationPolicyKind.PER_CHANNEL,
                CalibrationPolicyKind.BLOCKWISE,
            }:
                raise ValueError(
                    "per_tensor quantization does not support per_channel or blockwise calibration"
                )
        elif self.quantization_granularity == QuantizationGranularity.PER_CHANNEL:
            if self.calibration_policy == CalibrationPolicyKind.BLOCKWISE:
                raise ValueError("per_channel quantization cannot use blockwise calibration")
        elif self.quantization_granularity == QuantizationGranularity.BLOCKWISE:
            if self.calibration_policy == CalibrationPolicyKind.PER_CHANNEL:
                raise ValueError("blockwise quantization cannot use per_channel calibration")

        if self.quant_mode == QuantMode.NONE and self.codec_mode == CodecMode.LOSSY:
            raise ValueError("QuantMode.NONE is only valid with lossless codec mode")

        object.__setattr__(
            self,
            "per_channel",
            self.quantization_granularity == QuantizationGranularity.PER_CHANNEL,
        )
        return self

    def to_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, default_flow_style=False, sort_keys=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> QuenchConfig:
        """Load config from a YAML file."""
        with open(path, encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle)
        return cls(**data)
