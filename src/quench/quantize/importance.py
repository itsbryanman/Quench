"""Importance-aware bit allocation."""
from __future__ import annotations

from typing import Any

import numpy as np

from quench.analyze.profiler import TensorProfiler


class ImportanceAllocator:
    """Allocate per-tensor bit widths from an entropy-weighted budget.

    Budget model:
        ``total_budget_bits`` is the sum of assigned per-tensor bit widths.
        Each tensor receives a minimum of 2 bits and a maximum of 8 bits.

    Strategy:
        1. Assign every tensor 2 bits.
        2. Distribute the remaining budget proportionally to profiler entropy.
        3. Round down, then distribute leftover bits by largest fractional
           remainder with deterministic name-based tie-breaking.
    """

    def __init__(self, profiler: TensorProfiler | None = None) -> None:
        self._profiler = profiler or TensorProfiler()

    def allocate_bits(
        self,
        tensors: dict[str, np.ndarray[Any, np.dtype[Any]]],
        total_budget_bits: int,
    ) -> dict[str, int]:
        """Allocate deterministic per-tensor bit widths."""
        if not tensors:
            return {}

        names = sorted(tensors)
        min_total = 2 * len(names)
        max_total = 8 * len(names)
        if total_budget_bits < min_total:
            raise ValueError(
                f"Budget {total_budget_bits} is too small for {len(names)} tensors"
            )

        effective_budget = min(total_budget_bits, max_total)
        bits = {name: 2 for name in names}
        remaining = effective_budget - min_total
        if remaining == 0:
            return bits

        entropies = np.array(
            [max(self._profiler.profile(tensors[name]).entropy_bits, 1e-9) for name in names],
            dtype=np.float64,
        )
        shares = entropies / entropies.sum() * remaining
        extras = np.minimum(np.floor(shares).astype(int), 6)

        assigned = 0
        for index, name in enumerate(names):
            bits[name] += int(extras[index])
            assigned += int(extras[index])

        remainders = shares - np.floor(shares)
        order = sorted(
            range(len(names)),
            key=lambda idx: (-remainders[idx], -entropies[idx], names[idx]),
        )
        while assigned < remaining:
            progressed = False
            for index in order:
                name = names[index]
                if bits[name] >= 8:
                    continue
                bits[name] += 1
                assigned += 1
                progressed = True
                if assigned == remaining:
                    break
            if not progressed:
                break

        return bits
