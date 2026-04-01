"""Resolve model sources to local safetensors files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def resolve_safetensors(source: str) -> list[Path]:
    """Resolve *source* to a list of local ``.safetensors`` file paths."""
    source_path = Path(source)

    if source_path.is_file() and source_path.suffix == ".safetensors":
        return [source_path]

    if source_path.is_dir():
        files = sorted(source_path.rglob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {source}")
        return files

    if "/" in source and not source_path.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download models from HuggingFace. "
                "Install with: pip install huggingface_hub"
            ) from exc
        local_dir = snapshot_download(source, allow_patterns=["*.safetensors", "*.json"])
        files = sorted(Path(local_dir).rglob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files in downloaded repo: {source}")
        return files

    raise FileNotFoundError(f"Cannot resolve model source: {source}")


def build_tensor_index(
    safetensors_paths: list[Path],
) -> dict[str, Path]:
    """Build a mapping from tensor name to the safetensors file that contains it."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "The safetensors package is required for delta compression. "
            "Install with: pip install safetensors"
        ) from exc

    index: dict[str, Path] = {}
    for path in safetensors_paths:
        with safe_open(str(path), framework="numpy") as handle:
            for name in handle.keys():
                if name in index:
                    raise ValueError(
                        f"Duplicate tensor name {name!r} in {path} "
                        f"(already seen in {index[name]})"
                    )
                index[name] = path
    return index


def load_tensor(
    name: str,
    index: dict[str, Path],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Load a single tensor by name using a pre-built index."""
    path = index.get(name)
    if path is None:
        raise KeyError(f"Tensor {name!r} not found in index")

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("safetensors is required") from exc

    with safe_open(str(path), framework="numpy") as handle:
        return np.asarray(handle.get_tensor(name))


def iter_tensor_names(index: dict[str, Path]) -> list[str]:
    """Return sorted tensor names from an index."""
    return sorted(index.keys())
