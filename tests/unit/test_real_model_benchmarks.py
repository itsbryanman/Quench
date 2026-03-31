"""Tests for local real-model benchmark helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.real_models import LocalModelSpec, load_model_manifest, run_real_model_suite
from quench.core.config import QuenchConfig


def test_load_model_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "repo_id": "acme/model",
                        "resolved_revision": "123abc",
                        "local_path": str(tmp_path / "snapshot"),
                        "files": [
                            {
                                "path": "model.safetensors",
                                "kind": "weight",
                                "sha256": "deadbeef",
                                "size_bytes": 1024,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    models = load_model_manifest(manifest_path)

    assert models == [
        LocalModelSpec(
            model_id="acme/model",
            model_revision="123abc",
            local_path=tmp_path / "snapshot",
            source_hashes={"model.safetensors": "deadbeef"},
        )
    ]


def test_run_real_model_suite_on_local_safetensors(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    safetensors.save_file(
        {
            "model.embed_tokens.weight": np.arange(64, dtype=np.float32).reshape(8, 8),
            "model.layers.0.self_attn.q_proj.weight": np.arange(64, dtype=np.float32).reshape(8, 8) / 10.0,
            "model.layers.0.mlp.down_proj.weight": np.arange(64, dtype=np.float32).reshape(8, 8) / 20.0,
            "model.layers.0.input_layernorm.weight": np.ones((8,), dtype=np.float32),
        },
        str(snapshot / "model.safetensors"),
    )

    results, summary = run_real_model_suite(
        [
            LocalModelSpec(
                model_id="acme/model",
                model_revision="123abc",
                local_path=snapshot,
                source_hashes={"model.safetensors": "hash"},
            )
        ],
        config=QuenchConfig(target_bits=4),
        repeats=1,
        zstd_level=1,
        benchmark_mode="sampled",
        sample_seed=2025,
        sampled_extra_tensors=0,
    )

    assert len(results) == 3
    assert summary["total_tensors_discovered"] == 4
    assert summary["total_tensors_benchmarked"] == 3
    assert summary["total_tensors_skipped"] == 1
    assert summary["skipped_reasons"] == {"sample_policy_excluded": 1}
    assert summary["models"][0]["dtypes_observed"] == ("float32",)
    assert {item.tensor_role for item in results} == {"attn_q_proj", "embedding", "mlp_down_proj"}
    assert all(item.source_dtype == "float32" for item in results)
    assert all(item.cosine_similarity is not None for item in results)


def test_run_real_model_suite_falls_back_to_torch_for_bfloat16(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    safetensors_torch.save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.bfloat16).reshape(8, 8),
        },
        str(snapshot / "model.safetensors"),
    )

    results, summary = run_real_model_suite(
        [
            LocalModelSpec(
                model_id="acme/model",
                model_revision="123abc",
                local_path=snapshot,
                source_hashes={"model.safetensors": "hash"},
            )
        ],
        config=QuenchConfig(target_bits=4),
        repeats=1,
        zstd_level=1,
        benchmark_mode="full",
        sample_seed=2025,
        sampled_extra_tensors=0,
    )

    assert len(results) == 1
    assert results[0].source_dtype == "bfloat16"
    assert results[0].dtype == "float32"
    assert summary["models"][0]["dtypes_observed"] == ("bfloat16",)
