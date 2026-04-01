"""Bundle and tensor mapping I/O helpers for Quench."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Protocol, TypeAlias, cast

import numpy as np

from quench.codec.decoder import QuenchDecoder
from quench.codec.encoder import QuenchEncoder
from quench.codec.metadata import deserialize_metadata as _deserialize_metadata
from quench.codec.metadata import serialize_metadata as _serialize_metadata
from quench.core.config import QuenchConfig
from quench.core.exceptions import CodecError
from quench.core.types import CompressedTensor
from quench.io import QNCReader, QNCWriter
from quench.io.container import QNC_VERSION_V2, QNC_VERSION_V3
from quench.io.tiny_bundle import try_build_tiny_exact_bundle_entry


class _SafeOpenHandle(Protocol):
    def __enter__(self) -> _SafeOpenHandle: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None: ...
    def keys(self) -> Iterable[str]: ...
    def get_tensor(self, name: str) -> np.ndarray[Any, np.dtype[Any]]: ...


class _SafeOpenFn(Protocol):
    def __call__(self, filename: str, *, framework: str) -> _SafeOpenHandle: ...


class _LoadSafetensorsFn(Protocol):
    def __call__(self, filename: str) -> dict[str, np.ndarray[Any, np.dtype[Any]]]: ...


class _SaveSafetensorsFn(Protocol):
    def __call__(
        self,
        tensors: dict[str, np.ndarray[Any, np.dtype[Any]]],
        filename: str,
        metadata: dict[str, str] | None = None,
    ) -> None: ...


_safe_open: _SafeOpenFn | None
_load_safetensors: _LoadSafetensorsFn | None
_save_safetensors: _SaveSafetensorsFn | None

try:  # pragma: no cover - optional dependency
    from safetensors import safe_open as _safe_open_impl
    from safetensors.numpy import load_file as _load_safetensors_impl
    from safetensors.numpy import save_file as _save_safetensors_impl

    _safe_open = cast(_SafeOpenFn, _safe_open_impl)
    _load_safetensors = cast(_LoadSafetensorsFn, _load_safetensors_impl)
    _save_safetensors = cast(_SaveSafetensorsFn, _save_safetensors_impl)
except Exception:  # pragma: no cover - optional dependency
    _safe_open = None
    _load_safetensors = None
    _save_safetensors = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


TensorLike: TypeAlias = np.ndarray[Any, np.dtype[Any]] | Any


def serialize_metadata(metadata: dict[str, Any]) -> bytes:
    """Serialize metadata using the codec's deterministic JSON encoding."""
    return _serialize_metadata(metadata)


def deserialize_metadata(data: bytes) -> dict[str, Any]:
    """Deserialize metadata produced by :func:`serialize_metadata`."""
    return _deserialize_metadata(data)


def save_compressed(
    path: str | Path,
    state_dict: Mapping[str, TensorLike] | Iterable[tuple[str, TensorLike]],
    config: QuenchConfig | None = None,
    *,
    chunk_size: int = 1 << 20,
    enable_tiny_exact_bundle: bool = True,
) -> None:
    """Compress a tensor mapping and write it to a `.qnc` bundle incrementally."""
    encoder = QuenchEncoder(config=config)
    tensor_count = _mapping_length(state_dict)
    use_tiny_exact_bundle = (
        enable_tiny_exact_bundle
        and isinstance(state_dict, Mapping)
        and _mapping_may_contain_tiny_exact_candidates(state_dict)
    )
    with QNCWriter(
        path,
        tensor_count=tensor_count,
        chunk_size=chunk_size,
        version=(QNC_VERSION_V3 if use_tiny_exact_bundle else QNC_VERSION_V2),
        enable_tiny_exact_bundle=use_tiny_exact_bundle,
    ) as writer:
        if isinstance(state_dict, Mapping):
            for name, compressed in encoder.iter_encode_dict(dict(state_dict), config=config):
                writer.write_compressed_tensor(name, compressed, chunk_size=chunk_size)
            return

        for name, tensor in _iter_sorted_items(state_dict):
            compressed = encoder.encode(tensor, name=name)
            writer.write_compressed_tensor(name, compressed, chunk_size=chunk_size)


def load_compressed(
    path: str | Path,
    config: QuenchConfig | None = None,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Load and decompress a `.qnc` bundle into numpy tensors."""
    decoder = QuenchDecoder(config=config)
    restored: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for record in QNCReader(path).iter_tensor_records():
        restored[record.name] = decoder.decode(record.to_compressed_tensor())
    return restored


def save_compressed_bundle(
    path: str | Path,
    tensors: Mapping[str, CompressedTensor],
    *,
    chunk_size: int = 1 << 20,
    enable_tiny_exact_bundle: bool = True,
) -> None:
    """Write an already-compressed tensor mapping to a `.qnc` bundle."""
    ordered_names = sorted(tensors)
    candidate_count = sum(
        1
        for name in ordered_names
        if try_build_tiny_exact_bundle_entry(name, tensors[name]) is not None
    )
    use_tiny_exact_bundle = enable_tiny_exact_bundle and candidate_count >= 2
    with QNCWriter(
        path,
        tensor_count=len(tensors),
        chunk_size=chunk_size,
        version=(QNC_VERSION_V3 if use_tiny_exact_bundle else QNC_VERSION_V2),
        enable_tiny_exact_bundle=use_tiny_exact_bundle,
    ) as writer:
        for name in ordered_names:
            writer.write_compressed_tensor(name, tensors[name], chunk_size=chunk_size)


def load_compressed_bundle(path: str | Path) -> dict[str, CompressedTensor]:
    """Load a `.qnc` bundle without decoding its payloads."""
    tensors: dict[str, CompressedTensor] = {}
    for record in QNCReader(path).iter_tensor_records():
        if record.name in tensors:
            raise CodecError(f"Duplicate tensor name in QNC bundle: {record.name}")
        tensors[record.name] = record.to_compressed_tensor()
    return tensors


def iter_tensor_mapping(
    path: str | Path,
) -> Iterator[tuple[str, np.ndarray[Any, np.dtype[Any]]]]:
    """Iterate tensors from `.qnc`, `.npz`, `.npy`, `.safetensors`, or a directory."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_dir():
        for file_path in sorted(source.rglob("*.npy")):
            name = str(file_path.relative_to(source).with_suffix("")).replace("\\", "/")
            yield name, np.load(file_path, allow_pickle=False)
        return

    if source.suffix == ".qnc":
        yield from load_compressed(source).items()
        return
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as loaded:
            for name in sorted(loaded.files):
                yield name, np.asarray(loaded[name])
        return
    if source.suffix == ".npy":
        yield source.stem, np.load(source, allow_pickle=False)
        return
    if source.suffix == ".safetensors":
        if _safe_open is not None:
            with _safe_open(str(source), framework="numpy") as handle:
                for name in sorted(handle.keys()):
                    yield name, np.asarray(handle.get_tensor(name))
            return
        if _load_safetensors is None:
            raise CodecError(
                "Loading .safetensors requires the optional safetensors package; "
                "use .npz or a directory of .npy files instead."
            )
        for name, value in sorted(_load_safetensors(str(source)).items()):
            yield name, np.asarray(value)
        return

    raise CodecError(f"Unsupported tensor input format: {source.suffix or str(source)}")


def load_tensor_mapping(path: str | Path) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Load a tensor mapping from disk into memory."""
    tensors = dict(iter_tensor_mapping(path))
    if not tensors:
        raise CodecError(f"No tensors found at {path}")
    return tensors


def save_tensor_mapping(
    path: str | Path,
    tensors: Mapping[str, np.ndarray[Any, np.dtype[Any]]],
) -> None:
    """Save a tensor mapping as `.npz`, `.npy`, `.safetensors`, or a directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.suffix == ".npz":
        tensor_arrays: dict[str, Any] = {name: np.asarray(value) for name, value in tensors.items()}
        np.savez(str(destination), **tensor_arrays)
        return
    if destination.suffix == ".npy":
        if len(tensors) != 1:
            raise CodecError("Saving to a single .npy file requires exactly one tensor")
        np.save(destination, next(iter(tensors.values())))
        return
    if destination.suffix == ".safetensors":
        if _save_safetensors is None:
            raise CodecError(
                "Saving .safetensors requires the optional safetensors package; "
                "use .npz or a directory of .npy files instead."
            )
        _save_safetensors({name: np.asarray(value) for name, value in tensors.items()}, str(destination))
        return

    destination.mkdir(parents=True, exist_ok=True)
    for name, value in sorted(tensors.items()):
        file_path = destination / f"{name}.npy"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, np.asarray(value))


def _mapping_length(
    mapping_or_items: Mapping[str, TensorLike] | Iterable[tuple[str, TensorLike]],
) -> int | None:
    if isinstance(mapping_or_items, Mapping):
        return len(mapping_or_items)
    return None


def _iter_sorted_items(
    mapping_or_items: Mapping[str, TensorLike] | Iterable[tuple[str, TensorLike]],
) -> Iterator[tuple[str, TensorLike]]:
    if isinstance(mapping_or_items, Mapping):
        for name in sorted(mapping_or_items):
            yield name, mapping_or_items[name]
        return
    yield from mapping_or_items


def _mapping_may_contain_tiny_exact_candidates(
    mapping: Mapping[str, TensorLike],
) -> bool:
    candidate_count = 0
    for name in sorted(mapping):
        values = np.asarray(mapping[name])
        raw_nbytes = int(values.nbytes)
        if raw_nbytes <= 2048:
            candidate_count += 1
        else:
            lower = name.lower()
            if raw_nbytes <= 4096 and any(
                token in lower for token in ("bias", "norm", "position_ids", "token_type_ids", "type_ids")
            ):
                candidate_count += 1
        if candidate_count >= 2:
            return True
    return False
