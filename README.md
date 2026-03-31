# Quench

[![License](https://img.shields.io/github/license/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-111111?style=for-the-badge&logo=python)](https://github.com/itsbryanman/quench)
[![GitHub](https://img.shields.io/badge/github-itsbryanman%2Fquench-181717?style=for-the-badge&logo=github)](https://github.com/itsbryanman/quench)
[![Stars](https://img.shields.io/github/stars/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/stargazers)
[![Status](https://img.shields.io/badge/status-phase_4_hardening-B1440E?style=for-the-badge)](https://github.com/itsbryanman/quench)

**Harden your tensors. Compress everything.**

Quench is a format-aware compression codec for machine learning tensors: model weights, KV caches, embeddings, activations, optimizer state, biases, and mixed-precision edge cases.

Phase 4 hardens the architecture for larger bundles and future native acceleration:

- reusable per-tensor, per-channel, and blockwise quantizers with separate calibration policies
- streamed `.qnc` bundle read/write paths with chunked payload records
- pluggable backend interfaces for entropy coding and quantized bit packing
- dedicated strategies for optimizer state, biases, and mixed-precision tensors
- benchmark tooling that emits stable JSON and CSV artifacts for CI regression checks

## Phase 4 Highlights

- `QuenchConfig` now supports `quantization_granularity`, `calibration_policy`, `block_size`, `percentile_value`, `pack_bits`, and backend selection fields.
- `.qnc` version 2 stores tensors as deterministic streamed records with explicit segment headers and payload chunks.
- Python remains the default backend, but entropy and packing hot paths now route through `quench.backends`.
- Benchmarks can be generated with `tools/run_benchmarks.py` and compared in CI without network access.

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

High-level `quench.compress()` and `quench.decompress()` are available for individual tensors, and `quench.integrations.save_compressed()` / `load_compressed()` handle streamed `.qnc` bundles.

## Configuration Notes

- `quantization_granularity`: `per_tensor`, `per_channel`, or `blockwise`
- `calibration_policy`: `minmax`, `percentile`, `per_channel`, or `blockwise`
- `block_size`: required for blockwise layouts and validated by config
- `pack_bits`: enables backend-driven bit packing for quantized symbol streams
- `entropy_backend` / `packing_backend`: select registered backend implementations, with `python` as the default

Rank-1 tensors are collapsed to per-tensor quantization to avoid metadata-heavy pseudo channel layouts.

## Streaming Bundles

- `QNCWriter` and `QNCReader` provide incremental `.qnc` write/read access.
- `encode_tensor_stream()` and `decode_tensor_stream()` let large bundles flow one tensor at a time.
- Version-1 Phase 3 bundles remain readable; new writes use version 2 streamed records.

## Benchmarks

Run the synthetic benchmark suite and emit machine-readable artifacts:

```bash
python tools/run_benchmarks.py --output-dir benchmark-artifacts
```

Download public Hugging Face safetensors snapshots and run the real-model suite:

```bash
HF_TOKEN=... python tools/download_models.py \
  --output-dir benchmarks/models/public \
  --model openai-community/gpt2 \
  --model sentence-transformers/all-MiniLM-L6-v2

python tools/run_benchmarks.py \
  --output-dir benchmarks/artifacts/real-public-models \
  --suite real \
  --model-manifest benchmarks/models/public/model-download-manifest.json \
  --real-model-mode full \
  --repeats 1 \
  --zstd-level 3

python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("benchmarks/artifacts/real-public-models/quench-benchmarks.json").read_text())
for row in payload["aggregates"]["by_model"]:
    print(row["model_id"], row["compressed_bytes_total"], row["zstd_raw_bytes_total"], row["zstd_quantized_bytes_total"])
PY
```

Use `--real-model-mode sampled` to benchmark a deterministic subset instead of every tensor. The sampled suite always includes embeddings, `lm_head`, attention projection weights, MLP projection weights, and a seeded extra sample from the remaining tensors.

Artifacts:

- `quench-benchmarks.json`: schema-versioned benchmark summary
- `quench-benchmarks.csv`: flat rows for CI diffs and spreadsheet inspection

Key fields include benchmark name, tensor type, shape, dtype, config JSON, raw bytes, compressed bytes, compression ratio, error metrics, throughput, and backend name.

## Install

```bash
pip install -e ".[dev,bench]"
```

## Development

```bash
make test
make lint
make typecheck
```

## License

Business Source License 1.1. Apache 2.0 applies after the change date for each release.
