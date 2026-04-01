"""Basic CLI tests for quench-compress."""

import numpy as np

from quench.cli import run


def test_cli_compress_decompress_roundtrip(tmp_path) -> None:
    input_path = tmp_path / "tensors"
    input_path.mkdir()
    tensor = np.random.default_rng(42).normal(size=(32, 16)).astype(np.float32)
    np.save(input_path / "test.weight.npy", tensor)

    output_qnc = tmp_path / "output.qnc"
    result = run(["--input", str(input_path), "--output", str(output_qnc), "--bits", "4"])
    assert result == 0
    assert output_qnc.exists()

    restored_path = tmp_path / "restored"
    result = run(["--input", str(output_qnc), "--output", str(restored_path), "--decompress"])
    assert result == 0
    restored = np.load(restored_path / "test.weight.npy")
    assert restored.shape == tensor.shape


def test_cli_returns_nonzero_on_missing_input() -> None:
    result = run(["--input", "/nonexistent/path", "--output", "/tmp/out.qnc"])
    assert result == 1
