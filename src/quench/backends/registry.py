"""Global registry for pluggable codec backends."""
from __future__ import annotations

from typing import Dict

from quench.backends.base import BackendBinding, EntropyBackend, PackingBackend
from quench.backends.python_backend import PythonEntropyBackend, PythonPackingBackend
from quench.core.exceptions import UnsupportedBackendError

_ENTROPY_BACKENDS: Dict[str, EntropyBackend] = {}
_PACKING_BACKENDS: Dict[str, PackingBackend] = {}


def register_entropy_backend(backend: EntropyBackend) -> None:
    """Register an entropy backend under its declared name."""
    _ENTROPY_BACKENDS[backend.name.lower()] = backend


def register_packing_backend(backend: PackingBackend) -> None:
    """Register a packing backend under its declared name."""
    _PACKING_BACKENDS[backend.name.lower()] = backend


def get_entropy_backend(name: str) -> EntropyBackend:
    """Resolve an entropy backend by *name*."""
    backend = _ENTROPY_BACKENDS.get(name.lower())
    if backend is None:
        raise UnsupportedBackendError(f"Unsupported entropy backend: {name}")
    return backend


def get_packing_backend(name: str) -> PackingBackend:
    """Resolve a packing backend by *name*."""
    backend = _PACKING_BACKENDS.get(name.lower())
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
    return sorted(set(_ENTROPY_BACKENDS) | set(_PACKING_BACKENDS))


register_entropy_backend(PythonEntropyBackend())
register_packing_backend(PythonPackingBackend())
