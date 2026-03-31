"""Tests for backend registry and backend implementations."""
from __future__ import annotations

import numpy as np
import pytest

from quench.backends import get_entropy_backend, get_packing_backend, list_backend_names
from quench.core.exceptions import UnsupportedBackendError
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import build_freq_table, normalize_freq_table


def test_backend_registry_exposes_python_backend() -> None:
    assert "python" in list_backend_names()
    assert get_entropy_backend("python").name == "python"
    assert get_packing_backend("python").name == "python"


def test_unknown_backend_raises() -> None:
    with pytest.raises(UnsupportedBackendError):
        get_entropy_backend("missing")


def test_python_entropy_backend_roundtrip() -> None:
    symbols = np.array([0, 1, 1, 2, 2, 2, 3, 3], dtype=np.int16)
    model = FrequencyModel.from_freq_table(
        normalize_freq_table(build_freq_table(symbols.astype(np.int64)))
    )
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
