"""Higher-level state_dict integration coverage."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

from quench.codec import QuenchDecoder, QuenchEncoder
from quench.core import QuenchConfig
from quench.integrations import load_compressed, save_compressed


def _compress_zstd(data: bytes) -> int | None:
    """Return a zstd baseline size when available."""
    try:
        import zstandard
    except Exception:
        if shutil.which("zstd") is None:
            return None
        completed = subprocess.run(
            ["zstd", "-3", "-q", "-c"],
            input=data,
            capture_output=True,
            check=True,
        )
        return len(completed.stdout)

    return len(zstandard.ZstdCompressor(level=3).compress(data))


def _synthetic_state_dict() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4242)

    token_embeddings = rng.normal(loc=0.0, scale=0.08, size=(512, 96)).astype(np.float32)
    token_embeddings[:128] = 0.0

    q_proj = rng.normal(loc=0.0, scale=0.12, size=(96, 96)).astype(np.float32)
    q_proj *= rng.lognormal(mean=-0.4, sigma=0.25, size=(96, 1)).astype(np.float32)

    k_proj = rng.normal(loc=0.0, scale=0.12, size=(96, 96)).astype(np.float32)
    k_proj *= rng.lognormal(mean=-0.3, sigma=0.28, size=(96, 1)).astype(np.float32)

    mlp_fc1 = rng.normal(loc=0.0, scale=0.10, size=(192, 96)).astype(np.float32)
    mlp_fc1[rng.random(mlp_fc1.shape) < 0.12] = 0.0

    lm_head = token_embeddings[:256].copy()

    kv_steps = rng.normal(loc=0.0, scale=0.02, size=(2, 6, 96, 16)).astype(np.float32)
    kv_cache = np.cumsum(kv_steps, axis=2, dtype=np.float32)

    activations = rng.normal(loc=0.0, scale=0.7, size=(4, 48, 64)).astype(np.float32)
    activations = np.maximum(activations, 0.0)
    activations[activations < 0.25] = 0.0

    return {
        "token_embed.weight": token_embeddings,
        "attn.q_proj.weight": q_proj,
        "attn.k_proj.weight": k_proj,
        "mlp.fc1.weight": mlp_fc1,
        "lm_head.weight": lm_head,
        "layer_0.k_cache": kv_cache,
        "block.activation": activations,
    }


def test_state_dict_roundtrip_and_qnc_bundle(tmp_path: object) -> None:
    state_dict = _synthetic_state_dict()
    config = QuenchConfig(target_bits=4)
    encoder = QuenchEncoder(config=config)
    decoder = QuenchDecoder(config=config)

    compressed = encoder.encode_dict(state_dict)
    restored = decoder.decode_dict(compressed)

    total_raw = 0
    total_compressed = 0
    total_zstd = 0
    have_zstd = True
    rows: list[str] = []

    for name, original in state_dict.items():
        recovered = restored[name]
        error = np.abs(recovered.astype(np.float32) - original.astype(np.float32))
        raw_size = original.nbytes
        compressed_size = compressed[name].compressed_nbytes
        zstd_size = _compress_zstd(np.ascontiguousarray(original).tobytes())
        if zstd_size is None:
            have_zstd = False
            zstd_display = "n/a"
        else:
            total_zstd += zstd_size
            zstd_display = str(zstd_size)

        rows.append(
            f"{name[:24]:24} {raw_size:10d} {compressed_size:12d} {raw_size / compressed_size:8.3f} {zstd_display:>10}"
        )

        assert recovered.shape == original.shape
        assert recovered.dtype == original.dtype
        assert float(np.mean(error)) <= 0.06
        assert float(np.max(error)) <= 0.20

        total_raw += raw_size
        total_compressed += compressed_size

    print(
        f"{'tensor':24} {'raw':>10} {'compressed':>12} {'ratio':>8} {'zstd':>10}\n"
        + "\n".join(rows)
    )

    bundle_path = tmp_path / "synthetic_model.qnc"
    save_compressed(bundle_path, state_dict, config=config)
    loaded = load_compressed(bundle_path, config=config)

    for name, original in state_dict.items():
        recovered = loaded[name]
        assert recovered.shape == original.shape
        assert recovered.dtype == original.dtype
        assert float(np.mean(np.abs(recovered.astype(np.float32) - original.astype(np.float32)))) <= 0.06

    assert total_compressed < total_raw * 0.75
    if have_zstd:
        assert total_compressed <= int(total_zstd * 1.15)
