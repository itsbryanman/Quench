"""Integration benchmark comparing Quench primitives against zstd."""
from __future__ import annotations

import os
import struct
import subprocess
from typing import Any

import numpy as np

from quench.analyze import TensorProfiler
from quench.core.types import QuantMode
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import RANSEncoder, build_freq_table, normalize_freq_table
from quench.quantize import UniformQuantizer
from quench.transform import ChannelNormalizer


def _load_real_model_weights(
) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.float16]]], str] | None:
    """Try to load a tiny public model from the local cache only."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception:
        return None

    model_ids = ("sshleifer/tiny-gpt2", "hf-internal-testing/tiny-random-gpt2")
    for model_id in model_ids:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
        except Exception:
            continue

        tensors: dict[str, np.ndarray[Any, np.dtype[np.float16]]] = {}
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if "weight" not in name and "lm_head" not in name:
                    continue
                if value.ndim < 2 or value.numel() < 65_536:
                    continue
                tensors[name] = value.detach().cpu().numpy().astype(np.float16, copy=False)
                if len(tensors) == 3:
                    break

        if tensors:
            return tensors, model_id

    return None


def _build_synthetic_weights() -> tuple[dict[str, np.ndarray[Any, np.dtype[np.float16]]], str]:
    """Create deterministic synthetic transformer-like weights."""
    rng = np.random.default_rng(2025)
    specs = {
        "mlp.fc1.weight": (1024, 512),
        "attn.q_proj.weight": (768, 768),
        "lm_head.weight": (2048, 256),
    }

    tensors: dict[str, np.ndarray[Any, np.dtype[np.float16]]] = {}
    for name, shape in specs.items():
        base = rng.laplace(loc=0.0, scale=0.10, size=shape).astype(np.float32)
        channel_scale = rng.lognormal(mean=-0.2, sigma=0.55, size=(shape[0], 1)).astype(
            np.float32
        )
        structured = base * channel_scale
        structured += 0.03 * np.sin(
            np.linspace(0.0, 8.0 * np.pi, shape[1], dtype=np.float32)
        )[None, :]
        structured[rng.random(shape) < 0.08] = 0.0
        tensors[name] = structured.astype(np.float16)

    return tensors, "synthetic"


def _metadata_bytes(
    scales: np.ndarray[Any, np.dtype[np.float32]],
    zero_points: np.ndarray[Any, np.dtype[np.float32]],
    quant_scale: float,
    bits: int,
    mode: QuantMode,
) -> bytes:
    """Serialize the metadata needed to reverse normalization and quantization."""
    prefix = struct.pack("<fii", quant_scale, bits, int(mode))
    return prefix + scales.astype(np.float32, copy=False).tobytes() + zero_points.astype(
        np.float32, copy=False
    ).tobytes()


def _format_table(rows: list[dict[str, Any]], source: str) -> str:
    """Render a compact comparison table."""
    header = (
        f"\nQuench benchmark source: {source}\n"
        f"{'tensor':28} {'entropy':>8} {'raw_fp16':>10} {'zstd_fp16':>10} "
        f"{'zstd_q':>10} {'quench':>10}"
    )
    body = [
        (
            f"{row['name'][:28]:28} {row['entropy']:8.3f} {row['raw_fp16']:10d} "
            f"{row['zstd_fp16']:10d} {row['zstd_q']:10d} {row['quench']:10d}"
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def _compress_zstd(data: bytes) -> bytes:
    """Compress bytes with zstd using the Python package or the CLI fallback."""
    try:
        import zstandard
    except Exception:
        completed = subprocess.run(
            ["zstd", "-3", "-q", "-c"],
            input=data,
            capture_output=True,
            check=True,
        )
        return completed.stdout

    return zstandard.ZstdCompressor(level=3).compress(data)


def test_quench_pipeline_beats_zstd_on_quantized_weights() -> None:
    loaded = _load_real_model_weights()
    weights, source = loaded if loaded is not None else _build_synthetic_weights()

    profiler = TensorProfiler()
    normalizer = ChannelNormalizer()
    quantizer = UniformQuantizer()

    rows: list[dict[str, Any]] = []
    total_raw = 0
    total_zstd_quant = 0
    total_quench = 0

    for name, weight in weights.items():
        stats = profiler.profile(weight)
        normalized, scales, zero_points = normalizer.normalize(weight, axis=0)
        quantized, params = quantizer.quantize(normalized, bits=4, mode=QuantMode.SYMMETRIC)

        symbols = quantized.reshape(-1).astype(np.int64)
        normalized_freq = normalize_freq_table(build_freq_table(symbols))
        payload = RANSEncoder(normalized_freq).encode(symbols)
        model_bytes = FrequencyModel.from_freq_table(normalized_freq).serialize()

        raw_bytes = weight.astype(np.float16, copy=False).tobytes()
        quantized_bytes = quantized.tobytes()
        metadata = _metadata_bytes(scales, zero_points, params.scale, params.bits, params.mode)

        raw_size = len(raw_bytes)
        zstd_raw_size = len(_compress_zstd(raw_bytes))
        zstd_quant_size = len(_compress_zstd(quantized_bytes)) + len(metadata)
        quench_size = len(payload) + len(model_bytes) + len(metadata)

        rows.append(
            {
                "name": name,
                "entropy": stats.entropy_bits,
                "raw_fp16": raw_size,
                "zstd_fp16": zstd_raw_size,
                "zstd_q": zstd_quant_size,
                "quench": quench_size,
            }
        )

        total_raw += raw_size
        total_zstd_quant += zstd_quant_size
        total_quench += quench_size

        assert quench_size < zstd_quant_size
        assert quench_size < raw_size * 0.75

    print(_format_table(rows, source))

    assert total_quench < total_zstd_quant
    assert total_quench < total_raw * 0.75
