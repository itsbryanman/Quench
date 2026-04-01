"""Tests for the rANS entropy coder — correctness and compression quality."""
from __future__ import annotations

import math
import subprocess

import numpy as np

from quench.entropy.bitstream import BitstreamReader, BitstreamWriter
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import (
    SCALE,
    RANSDecoder,
    RANSEncoder,
    build_freq_table,
    normalize_freq_table,
)


def _compress_zstd(data: bytes, level: int) -> bytes:
    """Compress bytes with zstd using the Python package or CLI fallback."""
    try:
        import zstandard
    except Exception:
        completed = subprocess.run(
            ["zstd", f"-{level}", "-q", "-c"],
            input=data,
            capture_output=True,
            check=True,
        )
        return completed.stdout

    return zstandard.ZstdCompressor(level=level).compress(data)

# ---------------------------------------------------------------------------
# rANS round-trip correctness
# ---------------------------------------------------------------------------


class TestRANSRoundtrip:
    """Verify that encode -> decode is lossless for various distributions."""

    @staticmethod
    def _roundtrip(symbols: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        freq = build_freq_table(symbols)
        norm = normalize_freq_table(freq, target_total=1 << 16)
        enc = RANSEncoder(norm)
        dec = RANSDecoder(norm)
        compressed = enc.encode(symbols)
        return dec.decode(compressed, len(symbols))

    def test_uniform(self, random_uint8: np.ndarray) -> None:  # type: ignore[type-arg]
        symbols = random_uint8.astype(np.int64)
        decoded = self._roundtrip(symbols)
        np.testing.assert_array_equal(decoded, symbols)

    def test_zipf(self, zipf_symbols: np.ndarray) -> None:  # type: ignore[type-arg]
        decoded = self._roundtrip(zipf_symbols)
        np.testing.assert_array_equal(decoded, zipf_symbols)

    def test_gaussian_quantized(
        self, gaussian_quantized: np.ndarray
    ) -> None:  # type: ignore[type-arg]
        symbols = gaussian_quantized.astype(np.int64)
        decoded = self._roundtrip(symbols)
        np.testing.assert_array_equal(decoded, symbols)

    def test_small(self) -> None:
        symbols = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.int64)
        decoded = self._roundtrip(symbols)
        np.testing.assert_array_equal(decoded, symbols)

    def test_single_symbol(self) -> None:
        symbols = np.full(1000, 42, dtype=np.int64)
        decoded = self._roundtrip(symbols)
        np.testing.assert_array_equal(decoded, symbols)

    def test_binary(self) -> None:
        rng = np.random.default_rng(99)
        symbols = rng.choice([0, 1], size=5000, p=[0.9, 0.1]).astype(np.int64)
        decoded = self._roundtrip(symbols)
        np.testing.assert_array_equal(decoded, symbols)

    def test_empty(self) -> None:
        freq = normalize_freq_table({0: 1})
        enc = RANSEncoder(freq)
        dec = RANSDecoder(freq)
        compressed = enc.encode(np.array([], dtype=np.int64))
        result = dec.decode(compressed, 0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Compression quality
# ---------------------------------------------------------------------------


class TestCompressionQuality:
    """Verify rANS approaches entropy bound and beats zstd."""

    @staticmethod
    def _encode_rans(symbols: np.ndarray) -> tuple[bytes, bytes]:  # type: ignore[type-arg]
        freq = build_freq_table(symbols)
        norm = normalize_freq_table(freq, target_total=1 << 16)
        enc = RANSEncoder(norm)
        model = FrequencyModel.from_freq_table(norm)
        payload = enc.encode(symbols)
        return payload, model.serialize()

    @staticmethod
    def _shannon_bits(symbols: np.ndarray) -> float:  # type: ignore[type-arg]
        freq = build_freq_table(symbols)
        total = sum(freq.values())
        h = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                h -= p * math.log2(p)
        return h * total

    def test_near_entropy_bound(self, zipf_symbols: np.ndarray) -> None:  # type: ignore[type-arg]
        payload, model_bytes = self._encode_rans(zipf_symbols)
        rans_bits = len(payload) * 8
        total_bits = (len(payload) + len(model_bytes)) * 8
        shannon_bits = self._shannon_bits(zipf_symbols)

        print(f"\n  rANS payload bits: {rans_bits}")
        print(f"  Model bits:        {len(model_bytes) * 8}")
        print(f"  rANS total bits:   {total_bits}")
        print(f"  Shannon bits: {shannon_bits:.0f}")
        print(f"  Overhead:     {(rans_bits / shannon_bits - 1) * 100:.1f}%")

        # rANS should be within 10% of entropy bound
        assert rans_bits < shannon_bits * 1.10

    def test_beats_zstd_on_zipf(self) -> None:
        rng = np.random.default_rng(42)
        alpha = 1.5
        vocab = 256
        weights = np.array([1.0 / (k**alpha) for k in range(1, vocab + 1)])
        probs = weights / weights.sum()
        symbols = rng.choice(vocab, size=100_000, p=probs).astype(np.int64)

        raw_bytes = symbols.astype(np.uint8).tobytes()

        # rANS
        payload, model_bytes = self._encode_rans(symbols)
        rans_size = len(payload) + len(model_bytes)

        # zstd level 3
        zstd3_size = len(_compress_zstd(raw_bytes, level=3))

        # zstd level 19
        zstd19_size = len(_compress_zstd(raw_bytes, level=19))

        shannon_bits = self._shannon_bits(symbols)
        shannon_bytes = shannon_bits / 8

        print("\n  Zipf(1.5), 100K symbols, 256 vocab:")
        print(f"  Shannon bound: {shannon_bytes:,.0f} bytes")
        print(f"  rANS:          {rans_size:,.0f} bytes")
        print(f"  zstd -3:       {zstd3_size:,.0f} bytes")
        print(f"  zstd -19:      {zstd19_size:,.0f} bytes")

        assert rans_size < zstd3_size, (
            f"rANS ({rans_size}) should beat zstd-3 ({zstd3_size})"
        )

    def test_beats_zstd_on_gaussian(self) -> None:
        """Simulate quantized model weights: Laplacian distribution (peaked at
        zero, heavy tails) quantized to 4-bit (16 levels).  This is what real
        INT4 model weights look like after quantization."""

        rng = np.random.default_rng(42)
        # Laplacian(0, 0.5) → strongly peaked at 0, realistic for weights
        raw = rng.laplace(loc=0.0, scale=0.5, size=100_000).astype(np.float32)
        clipped = np.clip(raw, -4.0, 4.0)
        # Quantize to 16 levels (4-bit) — the actual ML use case
        symbols = ((clipped + 4.0) / 8.0 * 15.0).round().astype(np.int64)
        symbols = np.clip(symbols, 0, 15)

        raw_bytes = symbols.astype(np.uint8).tobytes()

        # rANS
        payload, model_bytes = self._encode_rans(symbols)
        rans_size = len(payload) + len(model_bytes)

        # zstd level 3
        zstd3_size = len(_compress_zstd(raw_bytes, level=3))

        # zstd level 19
        zstd19_size = len(_compress_zstd(raw_bytes, level=19))

        shannon_bits = self._shannon_bits(symbols)
        shannon_bytes = shannon_bits / 8

        print("\n  4-bit Laplacian quantized weights, 100K symbols:")
        print(f"  Shannon bound: {shannon_bytes:,.0f} bytes")
        print(f"  rANS:          {rans_size:,.0f} bytes")
        print(f"  zstd -3:       {zstd3_size:,.0f} bytes")
        print(f"  zstd -19:      {zstd19_size:,.0f} bytes")

        # rANS should beat zstd-3 by at least 10%
        assert rans_size < zstd3_size * 0.90, (
            f"rANS ({rans_size}) should beat zstd-3 ({zstd3_size}) by >= 10%"
        )


# ---------------------------------------------------------------------------
# Frequency model
# ---------------------------------------------------------------------------


class TestFrequencyModel:
    def test_from_data(self) -> None:
        symbols = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
        model = FrequencyModel.from_data(symbols)
        assert model.total == 6
        # Cumfreqs must be monotonically increasing
        prev = -1
        for s in sorted(model.cumfreq.keys()):
            assert model.cumfreq[s] > prev or model.cumfreq[s] == 0
            prev = model.cumfreq[s]

    def test_entropy_binary_50_50(self) -> None:
        model = FrequencyModel({0: 500, 1: 500})
        h = model.entropy_bound()
        assert abs(h - 1.0) < 0.01, f"Expected ~1.0 bit, got {h}"

    def test_serialize_roundtrip(self) -> None:
        freq = {0: 100, 5: 200, 10: 50, 255: 150}
        model = FrequencyModel(freq)
        blob = model.serialize()
        restored = FrequencyModel.deserialize(blob)
        assert restored.freq_table == model.freq_table
        assert restored.total == model.total


# ---------------------------------------------------------------------------
# Bitstream
# ---------------------------------------------------------------------------


class TestBitstream:
    def test_roundtrip(self) -> None:
        w = BitstreamWriter()
        w.write_byte(0xAB)
        w.write_uint32(0xDEADBEEF)
        w.write_uint64(0x0102030405060708)
        w.write_bytes(b"hello")

        r = BitstreamReader(w.getvalue())
        assert r.read_byte() == 0xAB
        assert r.read_uint32() == 0xDEADBEEF
        assert r.read_uint64() == 0x0102030405060708
        assert r.read_bytes(5) == b"hello"
        assert r.remaining() == 0

    def test_uint32_edge_cases(self) -> None:
        w = BitstreamWriter()
        w.write_uint32(0)
        w.write_uint32(0xFFFFFFFF)

        r = BitstreamReader(w.getvalue())
        assert r.read_uint32() == 0
        assert r.read_uint32() == 0xFFFFFFFF


# ---------------------------------------------------------------------------
# normalize_freq_table edge cases
# ---------------------------------------------------------------------------


def test_normalize_freq_table_many_symbols() -> None:
    """Normalization with more symbols than SCALE must not hang."""
    # 65536 symbols each with count 1 — already sums to SCALE
    freq = {i: 1 for i in range(SCALE)}
    result = normalize_freq_table(freq)
    assert sum(result.values()) == SCALE
    assert all(v >= 1 for v in result.values())


def test_normalize_freq_table_large_deficit() -> None:
    """Normalization that needs to add many counts completes quickly."""
    freq = {0: 1, 1: 1, 2: 1}
    result = normalize_freq_table(freq, target_total=SCALE)
    assert sum(result.values()) == SCALE
    assert all(v >= 1 for v in result.values())


def test_normalize_freq_table_large_surplus() -> None:
    """Normalization that needs to remove many counts completes quickly."""
    # 3 symbols with huge counts, target is small
    freq = {0: 100000, 1: 100000, 2: 100000}
    result = normalize_freq_table(freq, target_total=1 << 16)
    assert sum(result.values()) == 1 << 16
    assert all(v >= 1 for v in result.values())
