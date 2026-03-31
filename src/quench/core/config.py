"""Quench configuration via Pydantic v2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

from quench.core.types import CodecMode, QuantMode


class QuenchConfig(BaseModel):
    """Top-level configuration for the Quench codec."""

    codec_mode: CodecMode = CodecMode.LOSSY
    target_bits: int = 4
    quant_mode: QuantMode = QuantMode.SYMMETRIC
    entropy_coder: str = "rans"
    pca_variance_threshold: float = 0.99
    delta_coding: bool = True
    per_channel: bool = True

    @field_validator("target_bits")
    @classmethod
    def _validate_target_bits(cls, v: int) -> int:
        if not (1 <= v <= 32):
            raise ValueError(f"target_bits must be in [1, 32], got {v}")
        return v

    @field_validator("pca_variance_threshold")
    @classmethod
    def _validate_pca(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"pca_variance_threshold must be in (0, 1], got {v}")
        return v

    def to_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        data = self.model_dump()
        # Convert enums to their integer values for clean YAML
        data["codec_mode"] = int(data["codec_mode"])
        data["quant_mode"] = int(data["quant_mode"])
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> QuenchConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)
