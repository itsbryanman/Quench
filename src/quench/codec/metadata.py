"""Deterministic metadata serialization helpers for codec state."""
from __future__ import annotations

import base64
import json
from typing import Any, Mapping

import numpy as np

from quench.core.exceptions import MetadataError


def serialize_metadata(metadata: Mapping[str, Any]) -> bytes:
    """Serialize codec metadata deterministically as UTF-8 JSON bytes."""
    try:
        normalized = _encode_value(dict(metadata))
        return json.dumps(
            normalized,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"Failed to serialize metadata: {exc}") from exc


def deserialize_metadata(data: bytes) -> dict[str, Any]:
    """Deserialize metadata produced by :func:`serialize_metadata`."""
    try:
        loaded = json.loads(data.decode("utf-8"))
        decoded = _decode_value(loaded)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MetadataError(f"Failed to deserialize metadata: {exc}") from exc

    if not isinstance(decoded, dict):
        raise MetadataError("Metadata root must decode to a dictionary")
    return decoded


def _encode_value(value: Any) -> Any:
    """Convert Python and NumPy values into JSON-safe structures."""
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        payload = base64.b64encode(contiguous.tobytes()).decode("ascii")
        return {
            "__ndarray__": True,
            "data": payload,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {
            "__bytes__": True,
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (int, str)):
        return getattr(value, "value")
    raise TypeError(f"Unsupported metadata value type: {type(value)!r}")


def _decode_value(value: Any) -> Any:
    """Reconstruct Python and NumPy values from JSON-safe structures."""
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("__ndarray__") is True:
        try:
            dtype = np.dtype(str(value["dtype"]))
            shape = tuple(int(dim) for dim in value["shape"])
            payload = base64.b64decode(str(value["data"]).encode("ascii"))
        except (KeyError, TypeError, ValueError) as exc:
            raise MetadataError(f"Malformed ndarray metadata: {exc}") from exc

        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(payload) != expected_size:
            raise MetadataError(
                "Decoded ndarray metadata has unexpected byte length: "
                f"expected {expected_size}, got {len(payload)}"
            )
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    if value.get("__bytes__") is True:
        try:
            return base64.b64decode(str(value["data"]).encode("ascii"))
        except (KeyError, TypeError, ValueError) as exc:
            raise MetadataError(f"Malformed bytes metadata: {exc}") from exc
    return {str(key): _decode_value(item) for key, item in value.items()}
