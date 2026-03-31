"""Tests for backend registry and backend implementations."""
from __future__ import annotations

import numpy as np
import pytest

import quench.backends.registry as backend_registry
from quench.backends import (
    get_entropy_backend,
    get_native_backend_import_error,
    get_packing_backend,
    list_backend_names,
    native_backend_available,
)
from quench.backends.python_backend import PythonEntropyBackend, PythonPackingBackend
from quench.core.exceptions import EntropyError, UnsupportedBackendError
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import build_freq_table, normalize_freq_table


def _normalized_model(symbols: np.ndarray) -> FrequencyModel:
    return FrequencyModel.from_freq_table(
        normalize_freq_table(build_freq_table(symbols.astype(np.int64, copy=False)))
    )


def _backend_symbol_cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    uniform = rng.integers(0, 256, size=32_768, dtype=np.int16)
    skewed = rng.choice([0, 1, 2, 3], size=32_768, p=[0.94, 0.03, 0.02, 0.01]).astype(np.int16)
    zipf = (np.minimum(rng.zipf(a=1.35, size=32_768), 256).astype(np.int16) - 1)
    gaussian = np.clip(
        np.rint(rng.normal(loc=0.0, scale=18.0, size=32_768)).astype(np.int16),
        -63,
        63,
    )
    return {
        "uniform": uniform,
        "skewed": skewed,
        "zipf": zipf,
        "gaussian": gaussian,
    }


def test_backend_registry_exposes_python_backend() -> None:
    assert "python" in list_backend_names()
    assert get_entropy_backend("python").name == "python"
    assert get_packing_backend("python").name == "python"


def test_unknown_backend_raises() -> None:
    with pytest.raises(UnsupportedBackendError):
        get_entropy_backend("missing")


def test_python_entropy_backend_roundtrip() -> None:
    symbols = np.array([0, 1, 1, 2, 2, 2, 3, 3], dtype=np.int16)
    model = _normalized_model(symbols)
    backend = get_entropy_backend("python")

    encoded = backend.encode_symbols(symbols, model)
    decoded = backend.decode_symbols(encoded, model, len(symbols))

    np.testing.assert_array_equal(decoded.astype(symbols.dtype), symbols)


def test_python_packing_backend_roundtrip() -> None:
    symbols = np.array([-7, -3, 0, 2, 7, -1], dtype=np.int8)
    backend = get_packing_backend("python")

    packed = backend.pack_bits(symbols, bits=4, signed=True, layout_metadata={"dtype": symbols.dtype.str})
    unpacked = backend.unpack_bits(
        packed,
        bits=4,
        signed=True,
        shape=symbols.shape,
        layout_metadata={"dtype": symbols.dtype.str},
    )

    np.testing.assert_array_equal(unpacked, symbols)


def test_register_optional_rust_backend_falls_back_when_native_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_import_error() -> tuple[object, object]:
        raise ImportError("native extension unavailable")

    monkeypatch.setattr(backend_registry, "_ENTROPY_BACKENDS", {})
    monkeypatch.setattr(backend_registry, "_PACKING_BACKENDS", {})
    monkeypatch.setattr(backend_registry, "_load_optional_rust_backends", _raise_import_error)

    backend_registry.register_entropy_backend(PythonEntropyBackend())
    backend_registry.register_packing_backend(PythonPackingBackend())

    assert backend_registry.register_optional_rust_backend() is False
    assert backend_registry.get_entropy_backend("python").name == "python"
    assert backend_registry.get_packing_backend("python").name == "python"
    with pytest.raises(UnsupportedBackendError):
        backend_registry.get_entropy_backend("rust")
    assert isinstance(backend_registry.get_native_backend_import_error(), ImportError)


@pytest.mark.skipif(not native_backend_available(), reason="Rust backend not built")
@pytest.mark.parametrize(("case_name", "symbols"), list(_backend_symbol_cases().items()))
def test_rust_entropy_backend_matches_python_backend(
    case_name: str,
    symbols: np.ndarray,
) -> None:
    del case_name

    model = _normalized_model(symbols)
    python_backend = get_entropy_backend("python")
    rust_backend = get_entropy_backend("rust")

    encoded_python = python_backend.encode_symbols(symbols, model)
    encoded_rust = rust_backend.encode_symbols(symbols, model)
    decoded_from_python = rust_backend.decode_symbols(encoded_python, model, len(symbols))
    decoded_from_rust = python_backend.decode_symbols(encoded_rust, model, len(symbols))

    assert encoded_rust == encoded_python
    np.testing.assert_array_equal(decoded_from_python.astype(symbols.dtype, copy=False), symbols)
    np.testing.assert_array_equal(decoded_from_rust.astype(symbols.dtype, copy=False), symbols)


@pytest.mark.skipif(not native_backend_available(), reason="Rust backend not built")
def test_rust_entropy_backend_rejects_unknown_symbol() -> None:
    backend = get_entropy_backend("rust")
    model = _normalized_model(np.array([0, 0, 1, 1], dtype=np.int16))

    with pytest.raises(EntropyError):
        backend.encode_symbols(np.array([0, 2], dtype=np.int16), model)


@pytest.mark.skipif(not native_backend_available(), reason="Rust backend not built")
def test_rust_entropy_backend_rejects_truncated_payload() -> None:
    backend = get_entropy_backend("rust")
    model = _normalized_model(np.array([0, 1, 1, 2], dtype=np.int16))

    with pytest.raises(EntropyError):
        backend.decode_symbols(b"\x00\x01\x02", model, 4)


def test_native_backend_import_error_is_populated_when_unavailable() -> None:
    if native_backend_available():
        backend_registry.register_optional_rust_backend()
        assert get_native_backend_import_error() is None
    else:
        assert isinstance(get_native_backend_import_error(), ImportError)
