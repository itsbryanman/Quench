"""Tests for the FrequencyModel."""
from __future__ import annotations

from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import normalize_freq_table


class TestCumfreqOrdering:
    def test_strictly_increasing(self) -> None:
        freq = {0: 10, 1: 20, 2: 5, 3: 15}
        model = FrequencyModel(freq)
        sorted_syms = sorted(model.cumfreq.keys())
        for i in range(1, len(sorted_syms)):
            prev_sym = sorted_syms[i - 1]
            cur_sym = sorted_syms[i]
            assert model.cumfreq[cur_sym] > model.cumfreq[prev_sym]


class TestNormalizePreservesSymbols:
    def test_all_symbols_present(self) -> None:
        freq = {i: i + 1 for i in range(50)}
        norm = normalize_freq_table(freq, target_total=1 << 12)
        for sym in freq:
            assert sym in norm
            assert norm[sym] >= 1


class TestEntropyValues:
    def test_uniform_256(self) -> None:
        freq = {i: 100 for i in range(256)}
        model = FrequencyModel(freq)
        h = model.entropy_bound()
        assert abs(h - 8.0) < 0.01, f"Expected ~8.0 bits, got {h}"

    def test_single_symbol(self) -> None:
        freq = {42: 1000}
        model = FrequencyModel(freq)
        h = model.entropy_bound()
        assert h == 0.0, f"Expected 0.0 bits, got {h}"
