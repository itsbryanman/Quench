"""Delta manifest serialization for QNC containers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

MANIFEST_TENSOR_NAME = "__quench_delta_manifest__"
MANIFEST_FORMAT_VERSION = 1


@dataclass
class DeltaManifest:
    """Metadata stored inside a delta QNC file."""

    base_model_id: str
    format_version: int = MANIFEST_FORMAT_VERSION
    shared_tensors: list[str] = field(default_factory=list)
    added_tensors: list[str] = field(default_factory=list)
    removed_tensors: list[str] = field(default_factory=list)
    tensor_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


def serialize_manifest(manifest: DeltaManifest) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Serialize a manifest into a 1-D uint8 numpy array for QNC storage."""
    payload = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
    return np.frombuffer(payload.encode("utf-8"), dtype=np.uint8).copy()


def deserialize_manifest(array: np.ndarray[Any, np.dtype[Any]]) -> DeltaManifest:
    """Deserialize a manifest from a 1-D uint8 numpy array."""
    raw = np.asarray(array, dtype=np.uint8).tobytes().decode("utf-8")
    data = json.loads(raw)
    return DeltaManifest(
        base_model_id=str(data["base_model_id"]),
        format_version=int(data.get("format_version", 1)),
        shared_tensors=list(data.get("shared_tensors", [])),
        added_tensors=list(data.get("added_tensors", [])),
        removed_tensors=list(data.get("removed_tensors", [])),
        tensor_profiles=dict(data.get("tensor_profiles", {})),
        config=dict(data.get("config", {})),
        created_at=str(data.get("created_at", "")),
    )
