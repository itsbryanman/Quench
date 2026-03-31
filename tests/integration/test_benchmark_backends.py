"""Integration coverage for benchmark backend selection and reporting."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.reporting import load_json_report
from benchmarks.runner import run_benchmark_suite
from quench.backends import native_backend_available
from quench.core.config import QuenchConfig


@pytest.mark.parametrize(
    "backend_name",
    ["python", pytest.param("rust", marks=pytest.mark.skipif(not native_backend_available(), reason="Rust backend not built"))],
)
def test_benchmark_suite_records_selected_backend_name(tmp_path: Path, backend_name: str) -> None:
    output_dir = tmp_path / backend_name
    config = QuenchConfig(entropy_backend=backend_name)

    json_path, csv_path, results, _ = run_benchmark_suite(
        output_dir,
        config=config,
        repeats=1,
        suite="synthetic",
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert results
    assert all(result.backend_name == backend_name for result in results)
    loaded = load_json_report(json_path)
    assert loaded
    assert all(result.backend_name == backend_name for result in loaded)
