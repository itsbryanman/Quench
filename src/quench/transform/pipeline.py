"""Composable reversible transform pipelines."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StepMetadata:
    """Metadata needed to reverse one pipeline step."""

    name: str
    payload: Any
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class _PipelineStep:
    """Internal pipeline step registration."""

    name: str
    transform_fn: Callable[..., Any]
    inverse_fn: Callable[..., Any]
    metadata_fn: Callable[[Any], Any] | None


class TransformPipeline:
    """Apply reversible transform steps in sequence and invert them later."""

    def __init__(self) -> None:
        self._steps: list[_PipelineStep] = []

    def add_step(
        self,
        name: str,
        transform_fn: Callable[..., Any],
        inverse_fn: Callable[..., Any],
        metadata_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        """Register a reversible transform step."""
        if not name:
            raise ValueError("Step name must be non-empty")
        if any(step.name == name for step in self._steps):
            raise ValueError(f"Duplicate pipeline step name: {name}")
        self._steps.append(
            _PipelineStep(
                name=name,
                transform_fn=transform_fn,
                inverse_fn=inverse_fn,
                metadata_fn=metadata_fn,
            )
        )

    def forward(
        self, tensor: np.ndarray[Any, np.dtype[Any]]
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], list[StepMetadata]]:
        """Run all registered transforms and collect their inversion metadata."""
        current = np.asarray(tensor)
        metadata: list[StepMetadata] = []

        for step in self._steps:
            input_shape = tuple(current.shape)
            result = step.transform_fn(current)
            if isinstance(result, tuple):
                if len(result) != 2:
                    raise ValueError(
                        f"Transform step '{step.name}' must return an array or a 2-tuple"
                    )
                transformed, raw_metadata = result
            else:
                transformed, raw_metadata = result, None

            current = np.asarray(transformed)
            payload = step.metadata_fn(raw_metadata) if step.metadata_fn else raw_metadata
            metadata.append(
                StepMetadata(
                    name=step.name,
                    payload=payload,
                    input_shape=input_shape,
                    output_shape=tuple(current.shape),
                )
            )

        return current, metadata

    def inverse(
        self,
        transformed: np.ndarray[Any, np.dtype[Any]],
        metadata_list: list[StepMetadata],
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Reverse a previously executed pipeline."""
        if len(metadata_list) != len(self._steps):
            raise ValueError("metadata_list length does not match the pipeline")

        current = np.asarray(transformed)
        for step, metadata in zip(reversed(self._steps), reversed(metadata_list)):
            if not isinstance(metadata, StepMetadata):
                raise TypeError("metadata_list must contain StepMetadata entries")
            if metadata.name != step.name:
                raise ValueError(
                    f"Malformed metadata: expected step '{step.name}', got '{metadata.name}'"
                )
            try:
                if metadata.payload is None:
                    restored = step.inverse_fn(current)
                else:
                    restored = step.inverse_fn(current, metadata.payload)
            except TypeError as exc:
                raise ValueError(
                    f"Incomplete metadata for step '{step.name}' prevented inversion"
                ) from exc

            current = np.asarray(restored)
            if tuple(current.shape) != metadata.input_shape:
                raise ValueError(
                    "Step "
                    f"'{step.name}' restored shape {current.shape}, "
                    f"expected {metadata.input_shape}"
                )

        return current
