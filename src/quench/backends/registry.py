"""Global registry for pluggable codec backends."""
from __future__ import annotations

from typing import Dict

from quench.backends.base import BackendBinding, EntropyBackend, PackingBackend
from quench.backends.python_backend import PythonEntropyBackend, PythonPackingBackend
from quench.core.exceptions import UnsupportedBackendError

_ENTROPY_BACKENDS: Dict[str, EntropyBackend] = {}
_PACKING_BACKENDS: Dict[str, PackingBackend] = {}
_NATIVE_BACKEND_IMPORT_ERROR: ImportError | None = None


def register_entropy_backend(backend: EntropyBackend) -> None:
    """Register an entropy backend under its declared name."""
    _ENTROPY_BACKENDS[backend.name.lower()] = backend


def register_packing_backend(backend: PackingBackend) -> None:
    """Register a packing backend under its declared name."""
    _PACKING_BACKENDS[backend.name.lower()] = backend


def get_entropy_backend(name: str) -> EntropyBackend:
    """Resolve an entropy backend by *name*."""
    normalized = name.lower()
    if normalized == "rust" and normalized not in _ENTROPY_BACKENDS:
        register_optional_rust_backend()
    backend = _ENTROPY_BACKENDS.get(normalized)
    if backend is None:
        raise UnsupportedBackendError(f"Unsupported entropy backend: {name}")
    return backend


def get_packing_backend(name: str) -> PackingBackend:
    """Resolve a packing backend by *name*."""
    normalized = name.lower()
    if normalized == "rust" and normalized not in _PACKING_BACKENDS:
        register_optional_rust_backend()
    backend = _PACKING_BACKENDS.get(normalized)
    if backend is None:
        raise UnsupportedBackendError(f"Unsupported packing backend: {name}")
    return backend


def get_backend_binding(name: str) -> BackendBinding:
    """Resolve both backend types under a shared *name*."""
    normalized = name.lower()
    return BackendBinding(
        name=normalized,
        entropy=get_entropy_backend(normalized),
        packing=get_packing_backend(normalized),
    )


def list_backend_names() -> list[str]:
    """Return the sorted union of registered backend names."""
    register_optional_rust_backend()
    return sorted(set(_ENTROPY_BACKENDS) | set(_PACKING_BACKENDS))


def _load_optional_rust_backends() -> tuple[EntropyBackend, PackingBackend] | None:
    from quench.backends.rust_backend import RustEntropyBackend, RustPackingBackend

    return RustEntropyBackend(), RustPackingBackend()


def register_optional_rust_backend() -> bool:
    """Register the optional Rust backend when the native extension is available."""
    global _NATIVE_BACKEND_IMPORT_ERROR

    if "rust" in _ENTROPY_BACKENDS and "rust" in _PACKING_BACKENDS:
        _NATIVE_BACKEND_IMPORT_ERROR = None
        return True

    try:
        entropy_backend, packing_backend = _load_optional_rust_backends()
    except ImportError as exc:
        _NATIVE_BACKEND_IMPORT_ERROR = exc
        return False

    register_entropy_backend(entropy_backend)
    register_packing_backend(packing_backend)
    _NATIVE_BACKEND_IMPORT_ERROR = None
    return True


def get_native_backend_import_error() -> ImportError | None:
    """Return the last import error seen while loading the optional Rust backend."""
    return _NATIVE_BACKEND_IMPORT_ERROR


register_entropy_backend(PythonEntropyBackend())
register_packing_backend(PythonPackingBackend())
register_optional_rust_backend()
