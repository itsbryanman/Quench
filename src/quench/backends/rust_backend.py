"""Rust-backed codec backends with a pure-Python fallback path."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from quench.backends.python_backend import PythonPackingBackend
from quench.core.exceptions import EntropyError
from quench.entropy.freq_model import FrequencyModel


def _native_build_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    target_root = repo_root / "native" / "target"
    candidates: list[Path] = []
    for profile in ("release", "debug"):
        build_dir = target_root / profile
        if not build_dir.exists():
            continue
        for pattern in (
            "quench_native*.so",
            "libquench_native*.so",
            "quench_native*.pyd",
            "quench_native*.dylib",
            "libquench_native*.dylib",
            "quench_native*.dll",
        ):
            candidates.extend(sorted(build_dir.glob(pattern)))
    return candidates


def _load_module_from_build_output() -> ModuleType:
    last_error: ImportError | None = None
    for candidate in _native_build_candidates():
        spec = importlib.util.spec_from_file_location("quench_native", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["quench_native"] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            last_error = ImportError(str(exc))
            sys.modules.pop("quench_native", None)
            continue
        return module
    if last_error is not None:
        raise last_error
    raise ImportError("quench_native is not installed and no local build artifact was found")


@lru_cache(maxsize=1)
def load_native_module() -> ModuleType:
    """Import the compiled Rust extension or load it from a local cargo build."""
    try:
        return importlib.import_module("quench_native")
    except ImportError:
        try:
            return _load_module_from_build_output()
        except ImportError as artifact_exc:
            raise ImportError("quench_native is unavailable") from artifact_exc


def native_backend_available() -> bool:
    """Return ``True`` when the native extension can be imported."""
    try:
        load_native_module()
    except ImportError:
        return False
    return True


class RustEntropyBackend:
    """Rust entropy backend implementing the project's rANS wire format."""

    name = "rust"

    def __init__(self) -> None:
        self._native = load_native_module()

    def encode_symbols(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
        freq_model: FrequencyModel,
    ) -> bytes:
        array = np.ascontiguousarray(np.asarray(symbols).reshape(-1).astype(np.int64, copy=False))
        try:
            return self._native.encode_symbols(array, dict(freq_model.freq_table))
        except Exception as exc:
            raise EntropyError(str(exc)) from exc

    def decode_symbols(
        self,
        data: bytes,
        freq_model: FrequencyModel,
        num_symbols: int,
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        try:
            decoded = self._native.decode_symbols(data, dict(freq_model.freq_table), int(num_symbols))
        except Exception as exc:
            raise EntropyError(str(exc)) from exc
        return np.asarray(decoded, dtype=np.int64)


class RustPackingBackend(PythonPackingBackend):
    """Compatibility binding for the shared backend name while packing stays in Python."""

    name = "rust"

    def pack_bits(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
        bits: int,
        signed: bool,
        layout_metadata: Mapping[str, Any] | None = None,
    ) -> bytes:
        return super().pack_bits(symbols, bits=bits, signed=signed, layout_metadata=layout_metadata)

    def unpack_bits(
        self,
        data: bytes,
        bits: int,
        signed: bool,
        shape: tuple[int, ...],
        layout_metadata: Mapping[str, Any] | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        return super().unpack_bits(
            data,
            bits=bits,
            signed=signed,
            shape=shape,
            layout_metadata=layout_metadata,
        )
