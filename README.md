# Quench

[![License](https://img.shields.io/badge/license-BSL_1.1-0A7BBB?style=for-the-badge)](https://github.com/itsbryanman/quench/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/itsbryanman/quench/blob/master/pyproject.toml)
[![Status](https://img.shields.io/badge/status-phase_5.7-B1440E?style=for-the-badge)](https://github.com/itsbryanman/quench/blob/master/README.md#phase-57-highlights)
[![Stars](https://img.shields.io/github/stars/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/commits/master)
[![Issues](https://img.shields.io/github/issues/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/issues)

**Harden your tensors. Compress everything.**

Quench is a format-aware compression codec for machine learning tensors: model weights, KV caches, embeddings, activations, optimizer state, biases, and mixed-precision edge cases.

Phase 5.7 keeps the normal APIs intact while tightening the exact-path economics for small tensors:

- compact exact encodings for tiny raw, constant, arithmetic-sequence, and broadcast-sequence tensors
- streamed `.qnc` bundle read/write paths with shared tiny-tensor framing in version 3
- pluggable backend interfaces for entropy coding and quantized bit packing
- dedicated strategies for optimizer state, biases, masks, and mixed-precision tensors
- benchmark tooling that emits stable JSON and CSV artifacts for regression checks without checking generated outputs into git

## Phase 5.7 Highlights

- `QuenchConfig` now supports `quantization_granularity`, `calibration_policy`, `block_size`, `percentile_value`, `pack_bits`, and backend selection fields.
- `.qnc` version 3 adds a tiny exact bundle segment so multiple exact small tensors can share one container envelope.
- Python remains the default backend, but entropy and packing hot paths now route through `quench.backends`.
- Benchmarks can be generated with `tools/run_benchmarks.py` and compared locally or in CI without checking manifests or artifacts into the repo.

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
- Version-1 Phase 3 bundles remain readable; new writes use streamed records and may use version 3 when tiny exact bundling is active.

## Benchmarks

Run the synthetic benchmark suite and emit machine-readable artifacts:

```bash
python tools/run_benchmarks.py --output-dir benchmark-artifacts
```

Download public Hugging Face safetensors snapshots and run the real-model suite. The download manifest and benchmark artifacts are generated locally and should stay out of git:

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

A sampled real-model snapshot from a local run (`--real-model-mode sampled`, `--repeats 3`, `--zstd-level 3`) showed:

- exact tiny-tensor container bytes (`raw_bytes <= 2048`): `89,692 -> 80,848`
- mean tiny exact overhead bytes: `404.9 -> 216.8`
- MiniLM `embeddings.position_ids`: estimated old v2 container bytes `398`, new bytes `337`
- many 1.5 KB exact bias and norm rows dropped by about `343` bytes each

Reproduce a sampled comparison locally:

```bash
python tools/run_benchmarks.py \
  --output-dir benchmarks/artifacts/local-sampled \
  --suite real \
  --model-manifest benchmarks/models/public/model-download-manifest.json \
  --real-model-mode sampled \
  --repeats 3 \
  --zstd-level 3

python tools/summarize_benchmarks.py \
  benchmarks/artifacts/local-sampled/quench-benchmarks.json
```

Artifacts:

- `quench-benchmarks.json`: schema-versioned benchmark summary
- `quench-benchmarks.csv`: flat rows for CI diffs and spreadsheet inspection

Key fields include benchmark name, tensor type, shape, dtype, config JSON, raw bytes, compressed bytes, container overhead, error metrics, throughput, and backend name. Generated artifacts intentionally omit absolute local model paths.

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

## Roadmap

- Phase 5.7 tiny-tensor bundling
- Phase 6 native Rust rANS backend
- broader real-model benchmark coverage

## License

Business Source License 1.1. Apache 2.0 applies after the change date for each release.
