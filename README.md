# Quench

[![License](https://img.shields.io/badge/license-BSL_1.1-0A7BBB?style=for-the-badge)](https://github.com/itsbryanman/quench/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/itsbryanman/quench/blob/master/pyproject.toml)
[![Status](https://img.shields.io/badge/status-phase_6-B1440E?style=for-the-badge)](https://github.com/itsbryanman/quench/blob/master/README.md#phase-6-highlights)
[![Stars](https://img.shields.io/github/stars/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/commits/master)
[![Issues](https://img.shields.io/github/issues/itsbryanman/quench?style=for-the-badge)](https://github.com/itsbryanman/quench/issues)

**Harden your tensors. Compress everything.**

Quench is a format-aware compression codec for machine learning tensors: model weights, KV caches, embeddings, activations, optimizer state, biases, and mixed-precision edge cases.

Phase 6 keeps the normal APIs intact while moving the entropy hot path behind a first native backend:

- Rust rANS encode/decode via PyO3, loaded behind the existing backend registry
- Python remains the default fallback backend and reference implementation
- encoded payload format and container compatibility remain unchanged across Python and Rust backends
- streamed `.qnc` bundle read/write paths with shared tiny-tensor framing in version 3
- dedicated strategies for optimizer state, biases, masks, and mixed-precision tensors
- benchmark tooling that emits stable JSON and CSV artifacts for regression checks without checking generated outputs into git

## Phase 6 Highlights

- `QuenchConfig` now supports `quantization_granularity`, `calibration_policy`, `block_size`, `percentile_value`, `pack_bits`, and backend selection fields.
- `.qnc` version 3 adds a tiny exact bundle segment so multiple exact small tensors can share one container envelope.
- `entropy_backend="rust"` now routes rANS encode/decode into `native/` while keeping Python orchestration and strategy selection unchanged.
- Python remains the default backend, and the package still imports cleanly when the Rust extension is unavailable.
- Packing is still implemented in Python in this phase, even when the shared backend binding name is `rust`.
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
- `entropy_backend` / `packing_backend`: select registered backend implementations; `python` is the default, and `rust` is available when the native extension is built

Rank-1 tensors are collapsed to per-tensor quantization to avoid metadata-heavy pseudo channel layouts.

## Streaming Bundles

- `QNCWriter` and `QNCReader` provide incremental `.qnc` write/read access.
- `encode_tensor_stream()` and `decode_tensor_stream()` let large bundles flow one tensor at a time.
- Version-1 Phase 3 bundles remain readable; new writes use streamed records and may use version 3 when tiny exact bundling is active.

## Benchmarks

Run the synthetic benchmark suite and emit machine-readable artifacts:

```bash
python tools/run_benchmarks.py \
  --output-dir benchmark-artifacts \
  --suite synthetic \
  --entropy-backend python
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

Phase 6 benchmark snapshot from the generated artifacts in this repo:

- artifacts: `benchmarks/artifacts/phase-6-python/` and `benchmarks/artifacts/phase-6-rust/`
- rows compared: `170`
- aggregate compression ratio: identical at `5.4471x`
- per-row size differences: `0`
- per-row error metric differences: `0`
- weighted encode throughput: `14.93 MB/s -> 84.42 MB/s` overall (`5.65x`)
- weighted decode throughput: `20.19 MB/s -> 231.29 MB/s` overall (`11.45x`)
- weighted encode throughput on synthetic rows: `5.49 MB/s -> 19.24 MB/s` (`3.50x`)
- weighted decode throughput on synthetic rows: `7.50 MB/s -> 86.50 MB/s` (`11.53x`)
- weighted encode throughput on sampled real-model rows: `15.02 MB/s -> 85.39 MB/s` (`5.69x`)
- weighted decode throughput on sampled real-model rows: `20.31 MB/s -> 232.60 MB/s` (`11.45x`)

The Phase 6 numbers above were measured on Linux against the sampled real-model suite using local snapshots of:

- `openai-community/gpt2`
- `sentence-transformers/all-MiniLM-L6-v2`
- `Qwen/Qwen2.5-0.5B-Instruct`

Reproduce the same style of backend comparison locally:

```bash
python tools/run_benchmarks.py \
  --output-dir benchmarks/artifacts/phase-6-python \
  --suite all \
  --model-manifest benchmarks/models/public/model-download-manifest.json \
  --real-model-mode sampled \
  --sampled-extra-tensors 4 \
  --repeats 3 \
  --zstd-level 3 \
  --entropy-backend python

python tools/run_benchmarks.py \
  --output-dir benchmarks/artifacts/phase-6-rust \
  --suite all \
  --model-manifest benchmarks/models/public/model-download-manifest.json \
  --real-model-mode sampled \
  --sampled-extra-tensors 4 \
  --repeats 3 \
  --zstd-level 3 \
  --entropy-backend rust
```

Artifacts:

- `quench-benchmarks.json`: schema-versioned benchmark summary
- `quench-benchmarks.csv`: flat rows for CI diffs and spreadsheet inspection

Key fields include benchmark name, tensor type, shape, dtype, config JSON, raw bytes, compressed bytes, container overhead, error metrics, throughput, and backend name. Generated artifacts intentionally omit absolute local model paths.

## Install

```bash
pip install -e ".[dev,bench]"
```

Build the optional Rust backend for local development:

```bash
pip install -e ".[native]"
maturin develop --manifest-path native/Cargo.toml
```

If you only want a local build artifact without installing the extension into the environment, Quench can also load a cargo build directly from `native/target`:

```bash
cargo build --manifest-path native/Cargo.toml --release
```

## Development

```bash
make test
make lint
make typecheck
make native-build
```

Current native-backend scope and limits:

- Rust accelerates rANS encode/decode only in Phase 6.
- Bit packing and unpacking still use the Python backend.
- The measured build and benchmark flow in this phase was validated on Linux.
- macOS and Windows build support should be treated as unverified until they are exercised separately.

## Roadmap

- Phase 6 native Rust rANS backend
- native bit pack/unpack if profiling justifies it
- broader real-model benchmark coverage

## License

Business Source License 1.1. Apache 2.0 applies after the change date for each release.
