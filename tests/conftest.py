"""Shared test fixtures for Quench."""
from __future__ import annotations

import numpy as np
import pytest

from quench.core.config import QuenchConfig


@pytest.fixture
def sample_config() -> QuenchConfig:
    """Default QuenchConfig."""
    return QuenchConfig()


@pytest.fixture
def random_uint8() -> np.ndarray:  # type: ignore[type-arg]
    """10 000 uniform random uint8 values."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=10_000, dtype=np.uint8)


@pytest.fixture
def gaussian_quantized() -> np.ndarray:  # type: ignore[type-arg]
    """10 000 Gaussian samples quantized to 256 levels (uint8)."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(10_000).astype(np.float32)
    # Clip to [-4, 4] then map to [0, 255]
    clipped = np.clip(raw, -4.0, 4.0)
    quantized = ((clipped + 4.0) / 8.0 * 255.0).astype(np.uint8)
    return quantized


@pytest.fixture
def zipf_symbols() -> np.ndarray:  # type: ignore[type-arg]
    """10 000 Zipf-distributed symbols in [0, 255]."""
    rng = np.random.default_rng(42)
    alpha = 1.5
    vocab = 256
    # Zipf weights: w_k = 1 / k^alpha
    weights = np.array([1.0 / (k**alpha) for k in range(1, vocab + 1)])
    probs = weights / weights.sum()
    return rng.choice(vocab, size=10_000, p=probs).astype(np.int64)


@pytest.fixture
def uniform_symbols() -> np.ndarray:  # type: ignore[type-arg]
    """10 000 uniform symbols in [0, 255]."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=10_000, dtype=np.int64)
