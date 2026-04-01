"""End-to-end integration tests for quench.delta."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quench.delta.engine import compress, inspect, load

pytest.importorskip("safetensors")
from safetensors.numpy import save_file as save_safetensors


def _save_model(path: Path, tensors: dict[str, np.ndarray[Any, np.dtype[Any]]]) -> None:
    """Save a dict of numpy tensors as a single safetensors file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(tensors, str(path))


def _cosine(a: np.ndarray[Any, np.dtype[Any]], b: np.ndarray[Any, np.dtype[Any]]) -> float:
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    dot = float(np.dot(a_f, b_f))
    norm_a = float(np.sqrt(np.dot(a_f, a_f)))
    norm_b = float(np.sqrt(np.dot(b_f, b_f)))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 1.0


def test_full_roundtrip_small_model(tmp_path: Path) -> None:
    """Compress and load a small synthetic model with multiple tensor types."""
    rng = np.random.default_rng(42)

    base_tensors = {
        "model.layer.0.weight": rng.normal(size=(512, 512)).astype(np.float32),
        "model.layer.0.bias": rng.normal(size=(512,)).astype(np.float32),
        "model.layer.1.weight": rng.normal(size=(512, 512)).astype(np.float32),
        "model.embed.weight": rng.normal(size=(1024, 512)).astype(np.float32),
    }
    ft_tensors = {
        name: tensor + rng.normal(scale=0.005, size=tensor.shape).astype(np.float32)
        for name, tensor in base_tensors.items()
    }

    base_path = tmp_path / "base" / "model.safetensors"
    ft_path = tmp_path / "finetune" / "model.safetensors"
    delta_path = tmp_path / "delta.qnc"
    _save_model(base_path, base_tensors)
    _save_model(ft_path, ft_tensors)

    compress(
        base=str(base_path.parent),
        finetune=str(ft_path.parent),
        output=str(delta_path),
        bits=4,
        verbose=False,
    )

    assert delta_path.exists()
    assert delta_path.stat().st_size > 0

    raw_size = sum(tensor.nbytes for tensor in ft_tensors.values())
    delta_size = delta_path.stat().st_size
    assert delta_size < raw_size / 2, f"Delta file ({delta_size}) not smaller enough vs raw ({raw_size})"

    restored = load(base=str(base_path.parent), delta=str(delta_path), verbose=False)
    assert set(restored) == set(ft_tensors)
    for name in ft_tensors:
        cosine = _cosine(ft_tensors[name], restored[name])
        assert cosine > 0.95, f"Cosine for {name} is {cosine}, expected > 0.95"


def test_identical_model_produces_tiny_delta(tmp_path: Path) -> None:
    """When base and finetune are identical, delta should be very small."""
    rng = np.random.default_rng(43)
    tensors = {
        "w0": rng.normal(size=(256, 256)).astype(np.float32),
        "w1": rng.normal(size=(256, 256)).astype(np.float32),
    }

    base_path = tmp_path / "base" / "model.safetensors"
    ft_path = tmp_path / "ft" / "model.safetensors"
    delta_path = tmp_path / "delta.qnc"
    _save_model(base_path, tensors)
    _save_model(ft_path, tensors)

    compress(
        base=str(base_path.parent),
        finetune=str(ft_path.parent),
        output=str(delta_path),
        verbose=False,
    )

    delta_size = delta_path.stat().st_size
    raw_size = sum(tensor.nbytes for tensor in tensors.values())
    assert delta_size < raw_size / 50

    restored = load(base=str(base_path.parent), delta=str(delta_path), verbose=False)
    for name in tensors:
        np.testing.assert_array_equal(restored[name], tensors[name])


def test_added_tensor_survives_roundtrip(tmp_path: Path) -> None:
    """Tensors only in the finetune should appear in the restored output."""
    rng = np.random.default_rng(44)
    base_tensors = {"shared": rng.normal(size=(128, 128)).astype(np.float32)}
    ft_tensors = {
        "shared": base_tensors["shared"] + rng.normal(scale=0.005, size=(128, 128)).astype(np.float32),
        "new_head": rng.normal(size=(64, 128)).astype(np.float32),
    }

    base_path = tmp_path / "base" / "model.safetensors"
    ft_path = tmp_path / "ft" / "model.safetensors"
    delta_path = tmp_path / "delta.qnc"
    _save_model(base_path, base_tensors)
    _save_model(ft_path, ft_tensors)

    compress(
        base=str(base_path.parent),
        finetune=str(ft_path.parent),
        output=str(delta_path),
        verbose=False,
    )

    restored = load(base=str(base_path.parent), delta=str(delta_path), verbose=False)
    assert "new_head" in restored
    assert restored["new_head"].shape == ft_tensors["new_head"].shape


def test_inspect_returns_manifest(tmp_path: Path) -> None:
    """inspect() should return manifest info without loading a model."""
    rng = np.random.default_rng(45)
    tensors = {"w": rng.normal(size=(128, 128)).astype(np.float32)}

    base_path = tmp_path / "base" / "model.safetensors"
    ft_path = tmp_path / "ft" / "model.safetensors"
    delta_path = tmp_path / "delta.qnc"
    _save_model(base_path, tensors)
    _save_model(
        ft_path,
        {"w": tensors["w"] + rng.normal(scale=0.001, size=(128, 128)).astype(np.float32)},
    )

    compress(
        base=str(base_path.parent),
        finetune=str(ft_path.parent),
        output=str(delta_path),
        verbose=False,
    )

    info = inspect(str(delta_path))
    assert info["base_model_id"] == str(base_path.parent)
    assert "w" in info["shared_tensors"]
    assert info["format_version"] == 1


def test_high_bits_improves_quality(tmp_path: Path) -> None:
    """Compressing at 4-bit should produce better quality than 2-bit."""
    rng = np.random.default_rng(46)
    base_tensors = {"w": rng.normal(size=(512, 512)).astype(np.float32)}
    ft_tensors = {"w": base_tensors["w"] + rng.normal(scale=0.01, size=(512, 512)).astype(np.float32)}

    base_path = tmp_path / "base" / "model.safetensors"
    ft_path = tmp_path / "ft" / "model.safetensors"
    _save_model(base_path, base_tensors)
    _save_model(ft_path, ft_tensors)

    results: dict[int, float] = {}
    for bits in (2, 4):
        delta_path = tmp_path / f"delta_{bits}bit.qnc"
        compress(
            base=str(base_path.parent),
            finetune=str(ft_path.parent),
            output=str(delta_path),
            bits=bits,
            verbose=False,
        )
        restored = load(base=str(base_path.parent), delta=str(delta_path), verbose=False)
        results[bits] = _cosine(ft_tensors["w"], restored["w"])

    assert results[4] >= results[2], f"4-bit cosine ({results[4]}) should be >= 2-bit ({results[2]})"
