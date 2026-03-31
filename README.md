# Quench

[![License](https://img.shields.io/github/license/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-111111?style=for-the-badge&logo=python)](https://github.com/itsbryanman/quench)
[![GitHub](https://img.shields.io/badge/github-itsbryanman%2Fquench-181717?style=for-the-badge&logo=github)](https://github.com/itsbryanman/quench)
[![Stars](https://img.shields.io/github/stars/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/stargazers)
[![Status](https://img.shields.io/badge/status-phase_1_foundation-B1440E?style=for-the-badge)](https://github.com/itsbryanman/quench)

**Harden your tensors. Compress everything.**

Quench is a format-aware compression codec for machine learning tensors: model weights, KV caches, embeddings, activations, and optimizer state.

Phase 1 ships the foundation: strict core types, a fixed-width QNC header, YAML-backed configuration, and a working byte-aligned rANS entropy coder that round-trips exactly and beats `zstd` on quantized synthetic data in the test suite.

## Phase 1 Foundation

- Typed tensor containers and binary serialization primitives
- Strict `pydantic` v2 configuration with YAML I/O
- Byte-aligned 64-bit rANS entropy coding
- Frequency-model and bitstream utilities
- Test coverage for exact decode fidelity and compression quality

## Quick Start

```python
import numpy as np

from quench.entropy import RANSDecoder, RANSEncoder
from quench.entropy.rans import build_freq_table, normalize_freq_table

symbols = np.array([0, 0, 0, 1, 2, 0, 1], dtype=np.int64)
freq = normalize_freq_table(build_freq_table(symbols))

encoded = RANSEncoder(freq).encode(symbols)
decoded = RANSDecoder(freq).decode(encoded, len(symbols))

assert np.array_equal(decoded, symbols)
```

High-level `quench.compress()` and `quench.decompress()` land in Phase 3. The entropy core is usable now via `quench.entropy`.

## Install

```bash
pip install -e ".[dev]"
```

## Development

```bash
make test
make lint
make typecheck
```

## License

Business Source License 1.1. Apache 2.0 applies after the change date for each release.
