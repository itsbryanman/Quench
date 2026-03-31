"""Real-model benchmark helpers built on local safetensors checkpoints."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

import numpy as np

from benchmarks.reporting import BenchmarkResult
from quench.analyze import TensorTypeDetector
from quench.codec import QuenchDecoder, QuenchEncoder
from quench.codec.metadata import deserialize_metadata
from quench.core.config import QuenchConfig
from quench.core.types import CompressedTensor, TensorType
from quench.integrations import save_compressed
from quench.io import QNCReader
from quench.io.tiny_bundle import distribute_shared_bytes

try:  # pragma: no cover - optional dependency
    import zstandard
except Exception:  # pragma: no cover - optional dependency
    zstandard = None

try:  # pragma: no cover - optional dependency
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional dependency
    safe_open = None

try:  # pragma: no cover - optional dependency
    import torch as _torch
except Exception:  # pragma: no cover - optional dependency
    _torch = None

_QUENCH_HEADER_BYTES = CompressedTensor._HEADER_WIRE_SIZE
_SAMPLED_IMPORTANT_ROLES = {
    "attn_k_proj",
    "attn_o_proj",
    "attn_q_proj",
    "attn_qkv_proj",
    "attn_v_proj",
    "embedding",
    "lm_head",
    "mlp_down_proj",
    "mlp_gate_proj",
    "mlp_up_proj",
}


@dataclass(frozen=True)
class LocalModelSpec:
    """One locally downloaded model snapshot to benchmark."""

    model_id: str
    model_revision: str
    local_path: Path
    source_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelBenchmarkSummary:
    """Stable per-model benchmark metadata written into JSON artifacts."""

    model_id: str
    model_revision: str
    model_local_path: str
    benchmark_mode: str
    sample_policy: str
    tensors_discovered: int
    tensors_benchmarked: int
    tensors_skipped: int
    skipped_reasons: dict[str, int]
    dtypes_observed: tuple[str, ...]
    source_files: tuple[str, ...]
    failures: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class _TensorEntry:
    source_file: Path
    tensor_name: str


def load_model_manifest(path: str | Path) -> list[LocalModelSpec]:
    """Load locally downloaded model metadata produced by ``tools/download_models.py``."""
    manifest_path = Path(path)
    manifest_root = manifest_path.parent.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    models: list[LocalModelSpec] = []
    for item in payload.get("models", []):
        source_hashes = {
            str(file_record["path"]): str(file_record.get("sha256", ""))
            for file_record in item.get("files", [])
            if str(file_record.get("kind", "")) == "weight"
        }
        local_path = Path(str(item["local_path"]))
        if not local_path.is_absolute():
            local_path = (manifest_root / local_path).resolve()
        models.append(
            LocalModelSpec(
                model_id=str(item["repo_id"]),
                model_revision=str(item["resolved_revision"]),
                local_path=local_path,
                source_hashes=source_hashes,
            )
        )
    return models


def run_real_model_suite(
    specs: Iterable[LocalModelSpec],
    *,
    config: QuenchConfig,
    repeats: int,
    zstd_level: int,
    benchmark_mode: str,
    sample_seed: int,
    sampled_extra_tensors: int,
) -> tuple[list[BenchmarkResult], dict[str, Any]]:
    """Benchmark one or more local model snapshots against Quench baselines."""
    results: list[BenchmarkResult] = []
    summaries: list[ModelBenchmarkSummary] = []
    skipped_totals: dict[str, int] = {}

    for spec in specs:
        model_results, summary = _benchmark_model(
            spec,
            config=config,
            repeats=repeats,
            zstd_level=zstd_level,
            benchmark_mode=benchmark_mode,
            sample_seed=sample_seed,
            sampled_extra_tensors=sampled_extra_tensors,
        )
        results.extend(model_results)
        summaries.append(summary)
        for reason, count in summary.skipped_reasons.items():
            skipped_totals[reason] = skipped_totals.get(reason, 0) + count

    payload = {
        "total_models": len(summaries),
        "total_tensors_discovered": sum(item.tensors_discovered for item in summaries),
        "total_tensors_benchmarked": sum(item.tensors_benchmarked for item in summaries),
        "total_tensors_skipped": sum(item.tensors_skipped for item in summaries),
        "skipped_reasons": dict(sorted(skipped_totals.items())),
        "models": [summary.__dict__ for summary in summaries],
    }
    return results, payload


def _benchmark_model(
    spec: LocalModelSpec,
    *,
    config: QuenchConfig,
    repeats: int,
    zstd_level: int,
    benchmark_mode: str,
    sample_seed: int,
    sampled_extra_tensors: int,
) -> tuple[list[BenchmarkResult], ModelBenchmarkSummary]:
    detector = TensorTypeDetector()
    encoder = QuenchEncoder(config=config)
    decoder = QuenchDecoder(config=config)
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    entries = _discover_tensor_entries(spec.local_path)
    selected_names, sample_policy = _select_tensor_names(
        [entry.tensor_name for entry in entries],
        benchmark_mode=benchmark_mode,
        sample_seed=sample_seed,
        sampled_extra_tensors=sampled_extra_tensors,
    )
    selected = set(selected_names)

    results: list[BenchmarkResult] = []
    skipped_reasons: dict[str, int] = {}
    dtypes_observed: set[str] = set()
    failures: list[dict[str, str]] = []
    benchmarked_entries: list[_TensorEntry] = []

    files = sorted({entry.source_file for entry in entries})
    for file_path in files:
        relative_source = str(file_path.relative_to(spec.local_path))
        source_hash = spec.source_hashes.get(relative_source, "")
        with _open_safetensors(file_path) as handle:
            tensor_names_in_file = sorted(handle.keys())

        with _open_safetensors(file_path) as handle:
            for tensor_name in tensor_names_in_file:
                if tensor_name not in selected:
                    _increment_reason(skipped_reasons, "sample_policy_excluded")
                    continue

                try:
                    values, source_dtype_str = _load_tensor_values(
                        handle,
                        file_path=file_path,
                        tensor_name=tensor_name,
                    )
                except Exception as exc:
                    _increment_reason(skipped_reasons, f"load_failed:{type(exc).__name__}")
                    failures.append(
                        {
                            "tensor_name": tensor_name,
                            "source_file": relative_source,
                            "stage": "load",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                skip_reason = _tensor_skip_reason(values)
                if skip_reason is not None:
                    _increment_reason(skipped_reasons, skip_reason)
                    continue

                dtypes_observed.add(source_dtype_str)
                tensor_type = detector.detect(values, name=tensor_name)
                tensor_role = _tensor_role(tensor_name)
                try:
                    result = _benchmark_tensor(
                        values,
                        tensor_name=tensor_name,
                        tensor_type=tensor_type,
                        tensor_role=tensor_role,
                        source_file=relative_source,
                        source_file_sha256=source_hash,
                        model_id=spec.model_id,
                        model_revision=spec.model_revision,
                        model_local_path="",
                        benchmark_mode=benchmark_mode,
                        sample_policy=sample_policy,
                        was_sampled=(benchmark_mode == "sampled"),
                        encoder=encoder,
                        decoder=decoder,
                        config=config,
                        config_json=config_json,
                        repeats=repeats,
                        zstd_level=zstd_level,
                        source_dtype=source_dtype_str,
                    )
                except Exception as exc:
                    _increment_reason(skipped_reasons, f"benchmark_failed:{type(exc).__name__}")
                    failures.append(
                        {
                            "tensor_name": tensor_name,
                            "source_file": relative_source,
                            "stage": "benchmark",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                results.append(result)
                benchmarked_entries.append(_TensorEntry(source_file=file_path, tensor_name=tensor_name))

    if results:
        results = _attach_container_metrics(
            results,
            benchmarked_entries,
            config=config,
        )

    summary = ModelBenchmarkSummary(
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        model_local_path="",
        benchmark_mode=benchmark_mode,
        sample_policy=sample_policy,
        tensors_discovered=len(entries),
        tensors_benchmarked=len(results),
        tensors_skipped=sum(skipped_reasons.values()),
        skipped_reasons=dict(sorted(skipped_reasons.items())),
        dtypes_observed=tuple(sorted(dtypes_observed)),
        source_files=tuple(str(path.relative_to(spec.local_path)) for path in files),
        failures=tuple(failures),
    )
    return results, summary


def _benchmark_tensor(
    tensor: np.ndarray[Any, np.dtype[Any]],
    *,
    tensor_name: str,
    tensor_type: TensorType,
    tensor_role: str,
    source_file: str,
    source_file_sha256: str,
    model_id: str,
    model_revision: str,
    model_local_path: str,
    benchmark_mode: str,
    sample_policy: str,
    was_sampled: bool,
    encoder: QuenchEncoder,
    decoder: QuenchDecoder,
    config: QuenchConfig,
    config_json: str,
    repeats: int,
    zstd_level: int,
    source_dtype: str | None = None,
) -> BenchmarkResult:
    values = np.asarray(tensor)
    compressed = encoder.encode(values, tensor_type=tensor_type, name=tensor_name)
    restored = decoder.decode(compressed)
    exact_kind = _exact_kind(compressed)

    encode_seconds = _measure_seconds(
        lambda: encoder.encode(values, tensor_type=tensor_type, name=tensor_name),
        repeats=repeats,
    )
    decode_seconds = _measure_seconds(lambda: decoder.decode(compressed), repeats=repeats)

    raw_bytes = int(values.nbytes)
    zstd_raw_bytes = len(_compress_zstd(np.ascontiguousarray(values).view(np.uint8).tobytes(), level=zstd_level))
    quantized_baseline = _build_quantized_baseline(
        values,
        tensor_name=tensor_name,
        tensor_type=tensor_type,
        config=config,
        zstd_level=zstd_level,
    )
    mse, max_abs, relative_error, cosine_sim = _error_metrics(values, restored)

    return BenchmarkResult(
        benchmark_name=f"real/{model_id}@{model_revision}/{benchmark_mode}/{tensor_name}",
        tensor_name=tensor_name,
        tensor_type=tensor_type.name.lower(),
        tensor_shape="x".join(str(dim) for dim in values.shape),
        dtype=str(values.dtype),
        config=config_json,
        raw_bytes=raw_bytes,
        compressed_bytes=int(compressed.compressed_nbytes),
        compression_ratio=(raw_bytes / compressed.compressed_nbytes if compressed.compressed_nbytes else 0.0),
        mse=mse,
        max_abs_error=max_abs,
        relative_error=relative_error,
        encode_throughput_bytes_per_sec=(raw_bytes / encode_seconds if encode_seconds else 0.0),
        decode_throughput_bytes_per_sec=(raw_bytes / decode_seconds if decode_seconds else 0.0),
        backend_name=config.entropy_backend,
        model_id=model_id,
        model_revision=model_revision,
        model_local_path=model_local_path,
        source_file=source_file,
        source_file_sha256=source_file_sha256,
        benchmark_mode=benchmark_mode,
        sample_policy=sample_policy,
        was_sampled=was_sampled,
        tensor_role=tensor_role,
        zstd_raw_bytes=zstd_raw_bytes,
        zstd_quantized_bytes=(None if quantized_baseline is None else quantized_baseline["zstd_bytes"]),
        quantized_bytes=(None if quantized_baseline is None else quantized_baseline["payload_bytes"]),
        quench_payload_bytes=len(compressed.payload),
        quench_metadata_bytes=len(compressed.metadata),
        quench_header_bytes=_QUENCH_HEADER_BYTES,
        quench_exact_kind=exact_kind,
        zstd_raw_ratio=(raw_bytes / zstd_raw_bytes if zstd_raw_bytes else None),
        zstd_quantized_ratio=(
            None
            if quantized_baseline is None or quantized_baseline["zstd_bytes"] == 0
            else raw_bytes / quantized_baseline["zstd_bytes"]
        ),
        cosine_similarity=cosine_sim,
        source_dtype=source_dtype,
    )


def _build_quantized_baseline(
    tensor: np.ndarray[Any, np.dtype[Any]],
    *,
    tensor_name: str,
    tensor_type: TensorType,
    config: QuenchConfig,
    zstd_level: int,
) -> dict[str, int] | None:
    baseline_config = config.model_copy(update={"entropy_coder": "raw", "pack_bits": False})
    baseline_encoder = QuenchEncoder(config=baseline_config)
    baseline = baseline_encoder.encode(tensor, tensor_type=tensor_type, name=tensor_name)
    metadata = _resolve_strategy_metadata(deserialize_metadata(baseline.metadata))
    if not isinstance(metadata, dict):
        return None
    if metadata.get("lossless") is True or int(metadata.get("l", 0)) == 1:
        return None

    return {
        "payload_bytes": len(baseline.payload),
        "zstd_bytes": len(_compress_zstd(baseline.payload, level=zstd_level))
        + len(baseline.metadata)
        + _QUENCH_HEADER_BYTES,
    }


def _discover_tensor_entries(local_path: Path) -> list[_TensorEntry]:
    files = sorted(local_path.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under {local_path}")

    entries: list[_TensorEntry] = []
    for file_path in files:
        with _open_safetensors(file_path) as handle:
            for tensor_name in sorted(handle.keys()):
                entries.append(_TensorEntry(source_file=file_path, tensor_name=tensor_name))
    if not entries:
        raise ValueError(f"No tensors found under {local_path}")
    return entries


def _select_tensor_names(
    tensor_names: list[str],
    *,
    benchmark_mode: str,
    sample_seed: int,
    sampled_extra_tensors: int,
) -> tuple[tuple[str, ...], str]:
    ordered = tuple(sorted(tensor_names))
    if benchmark_mode == "full":
        return ordered, "full-model: benchmark every discovered tensor"
    if benchmark_mode != "sampled":
        raise ValueError(f"Unsupported benchmark mode: {benchmark_mode}")

    important = [name for name in ordered if _tensor_role(name) in _SAMPLED_IMPORTANT_ROLES]
    remaining = [name for name in ordered if name not in set(important)]
    ranked = sorted(
        remaining,
        key=lambda name: hashlib.sha256(f"{sample_seed}:{name}".encode("utf-8")).hexdigest(),
    )
    extras = ranked[: max(sampled_extra_tensors, 0)]
    selected = tuple(sorted(set(important + extras)))
    policy = (
        "sampled-model: include all embeddings, lm_head, attention q/k/v/o (and fused qkv) "
        "projection tensors, all MLP up/down/gate tensors, plus "
        f"{len(extras)} additional tensors chosen by sha256(seed:tensor_name) with seed={sample_seed}"
    )
    return selected, policy


def _tensor_role(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("embed", "wte", "wpe")):
        return "embedding"
    if "lm_head" in lower:
        return "lm_head"
    if any(token in lower for token in ("q_proj", "query", ".q.", "to_q")):
        return "attn_q_proj"
    if any(token in lower for token in ("k_proj", "key", ".k.", "to_k")):
        return "attn_k_proj"
    if any(token in lower for token in ("v_proj", "value", ".v.", "to_v")):
        return "attn_v_proj"
    if any(token in lower for token in ("qkv", "c_attn")):
        return "attn_qkv_proj"
    if any(token in lower for token in ("o_proj", "out_proj", "c_proj", "to_out")):
        return "attn_o_proj"
    if any(
        token in lower
        for token in ("up_proj", "fc1", "dense_h_to_4h", "wi", "intermediate.dense")
    ):
        return "mlp_up_proj"
    if any(
        token in lower
        for token in ("down_proj", "fc2", "dense_4h_to_h", "wo", "output.dense")
    ):
        return "mlp_down_proj"
    if "gate_proj" in lower:
        return "mlp_gate_proj"
    if "bias" in lower:
        return "bias"
    if any(token in lower for token in ("norm", "ln_", "layernorm")):
        return "norm"
    return "other"


def _tensor_skip_reason(tensor: np.ndarray[Any, np.dtype[Any]]) -> str | None:
    values = np.asarray(tensor)
    if values.size == 0:
        return "empty_tensor"
    if np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        return None
    return f"unsupported_dtype:{values.dtype}"


def _measure_seconds(fn: Callable[[], Any], *, repeats: int) -> float:
    import time

    durations: list[float] = []
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return median(durations)


def _error_metrics(
    original: np.ndarray[Any, np.dtype[Any]],
    restored: np.ndarray[Any, np.dtype[Any]],
) -> tuple[float, float, float, float]:
    orig = np.asarray(original, dtype=np.float64).ravel()
    rec = np.asarray(restored, dtype=np.float64).ravel()
    diff = rec - orig
    mse = float(np.mean(diff**2))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.max(np.abs(orig))) or 1.0
    relative_error = max_abs / denom
    # Cosine similarity
    dot = float(np.dot(orig, rec))
    norm_orig = float(np.sqrt(np.dot(orig, orig)))
    norm_rec = float(np.sqrt(np.dot(rec, rec)))
    cosine_sim = dot / (norm_orig * norm_rec) if (norm_orig > 0 and norm_rec > 0) else 1.0
    return mse, max_abs, relative_error, cosine_sim


def _compress_zstd(data: bytes, *, level: int) -> bytes:
    if zstandard is not None:
        return zstandard.ZstdCompressor(level=level).compress(data)
    if shutil.which("zstd") is None:
        raise RuntimeError("zstandard Python package or zstd CLI is required for baseline compression")
    completed = subprocess.run(
        ["zstd", f"-{level}", "-q", "-c"],
        input=data,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _open_safetensors(path: Path) -> Any:
    if safe_open is None:
        raise RuntimeError("Real-model benchmarking requires the optional safetensors package")
    return safe_open(str(path), framework="numpy")


def _load_tensor_values(
    handle: Any,
    *,
    file_path: Path,
    tensor_name: str,
) -> tuple[np.ndarray[Any, np.dtype[Any]], str]:
    """Load one tensor, falling back to torch for dtypes numpy cannot materialize."""
    try:
        values = np.asarray(handle.get_tensor(tensor_name))
        return values, str(values.dtype)
    except Exception:
        if _torch is None or safe_open is None:
            raise

    pt_handle = safe_open(str(file_path), framework="pt")
    pt_tensor = pt_handle.get_tensor(tensor_name)
    source_dtype = str(pt_tensor.dtype).replace("torch.", "")
    return pt_tensor.to(dtype=_torch.float32).cpu().numpy(), source_dtype


def _increment_reason(bucket: dict[str, int], reason: str) -> None:
    bucket[reason] = bucket.get(reason, 0) + 1


def _attach_container_metrics(
    results: list[BenchmarkResult],
    benchmarked_entries: list[_TensorEntry],
    *,
    config: QuenchConfig,
) -> list[BenchmarkResult]:
    ordered_entries = sorted(
        benchmarked_entries,
        key=lambda entry: (str(entry.source_file), entry.tensor_name),
    )
    tensors: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for file_path in sorted({entry.source_file for entry in ordered_entries}):
        names = [entry.tensor_name for entry in ordered_entries if entry.source_file == file_path]
        with _open_safetensors(file_path) as handle:
            for tensor_name in names:
                values, _ = _load_tensor_values(
                    handle,
                    file_path=file_path,
                    tensor_name=tensor_name,
                )
                tensors[tensor_name] = values

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "model.qnc"
        save_compressed(bundle_path, tensors, config=config)
        records = list(QNCReader(bundle_path).iter_tensor_records())
        file_size = bundle_path.stat().st_size

    shared_header = max(file_size - sum(record.storage_nbytes for record in records), 0)
    header_shares = distribute_shared_bytes(shared_header, len(records))
    storage_by_name = {
        record.name: (
            record.storage_nbytes + header_shares[index],
            record.storage_payload_nbytes,
            record.storage_overhead_nbytes + header_shares[index],
        )
        for index, record in enumerate(records)
    }

    updated: list[BenchmarkResult] = []
    for result in results:
        container_metrics = storage_by_name.get(result.tensor_name)
        if container_metrics is None:
            updated.append(result)
            continue
        container_bytes, container_payload, container_overhead = container_metrics
        updated.append(
            replace(
                result,
                quench_container_bytes=container_bytes,
                quench_container_payload_bytes=container_payload,
                quench_container_overhead_bytes=container_overhead,
            )
        )
    return updated


def _exact_kind(compressed: CompressedTensor) -> str | None:
    metadata = _resolve_strategy_metadata(deserialize_metadata(compressed.metadata))
    kind = metadata.get("k", metadata.get("path"))
    if kind is None:
        return None
    return str(kind)


def _resolve_strategy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    strategy = metadata.get("strategy")
    if isinstance(strategy, dict):
        nested = strategy.get("metadata")
        if isinstance(nested, dict):
            return nested
    return metadata
