use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

const SCALE_BITS: u32 = 16;
const SCALE: u64 = 1 << SCALE_BITS;
const RANS_L: u64 = 1 << 56;
const X_MAX_SHIFT: u32 = 64 - SCALE_BITS;

#[derive(Clone, Copy)]
struct SymbolInfo {
    freq: u32,
    cumfreq: u32,
}

#[derive(Clone, Copy)]
struct SlotEntry {
    symbol: i64,
    freq: u32,
    cumfreq: u32,
}

struct Model {
    total: u64,
    symbols: HashMap<i64, SymbolInfo>,
    slots: Vec<SlotEntry>,
}

impl Model {
    fn build(freq_table: HashMap<i64, u32>) -> PyResult<Self> {
        if freq_table.is_empty() {
            return Err(PyValueError::new_err("Frequency table is empty"));
        }

        let total = freq_table
            .values()
            .try_fold(0_u64, |acc, value| acc.checked_add(u64::from(*value)))
            .ok_or_else(|| PyValueError::new_err("Frequency table total overflowed"))?;
        if total != SCALE {
            return Err(PyValueError::new_err(format!(
                "Frequency table must sum to {SCALE} (got {total})"
            )));
        }

        let mut symbols_sorted: Vec<i64> = freq_table.keys().copied().collect();
        symbols_sorted.sort_unstable();

        let mut symbols = HashMap::with_capacity(symbols_sorted.len());
        let mut slots = vec![
            SlotEntry {
                symbol: 0,
                freq: 0,
                cumfreq: 0,
            };
            total as usize
        ];
        let mut cumfreq = 0_u32;
        for symbol in symbols_sorted {
            let freq = *freq_table
                .get(&symbol)
                .ok_or_else(|| PyValueError::new_err("Frequency table changed during build"))?;
            if freq == 0 {
                return Err(PyValueError::new_err(format!(
                    "Frequency table entry for symbol {symbol} is zero"
                )));
            }
            let info = SymbolInfo { freq, cumfreq };
            symbols.insert(symbol, info);
            let start = usize::try_from(cumfreq).map_err(|_| {
                PyValueError::new_err("Cumulative frequency index does not fit in usize")
            })?;
            let end = start
                .checked_add(usize::try_from(freq).map_err(|_| {
                    PyValueError::new_err("Frequency does not fit in usize")
                })?)
                .ok_or_else(|| PyValueError::new_err("Frequency slot table overflowed"))?;
            for slot in &mut slots[start..end] {
                *slot = SlotEntry {
                    symbol,
                    freq,
                    cumfreq,
                };
            }
            cumfreq = cumfreq
                .checked_add(freq)
                .ok_or_else(|| PyValueError::new_err("Cumulative frequency overflowed"))?;
        }

        Ok(Self {
            total,
            symbols,
            slots,
        })
    }
}

fn encode_impl(symbols: &[i64], model: &Model) -> PyResult<Vec<u8>> {
    if symbols.is_empty() {
        return Ok(Vec::new());
    }

    let mut emitted = Vec::new();
    let mut state = RANS_L;

    for &symbol in symbols.iter().rev() {
        let info = model
            .symbols
            .get(&symbol)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("Symbol {symbol} not in frequency table")))?;
        let freq = u64::from(info.freq);
        let cumfreq = u64::from(info.cumfreq);
        let x_max = freq << X_MAX_SHIFT;
        while state >= x_max {
            emitted.push((state & 0xFF) as u8);
            state >>= 8;
        }
        state = (state / freq) * model.total + cumfreq + (state % freq);
    }

    let mut output = Vec::with_capacity(8 + emitted.len());
    output.extend_from_slice(&state.to_le_bytes());
    emitted.reverse();
    output.extend_from_slice(&emitted);
    Ok(output)
}

fn decode_impl(data: &[u8], model: &Model, num_symbols: usize) -> PyResult<Vec<i64>> {
    if num_symbols == 0 {
        return Ok(Vec::new());
    }
    if data.len() < 8 {
        return Err(PyValueError::new_err("Data too short to contain rANS state"));
    }

    let mut state_bytes = [0_u8; 8];
    state_bytes.copy_from_slice(&data[..8]);
    let mut state = u64::from_le_bytes(state_bytes);
    let mut position = 8_usize;
    let mask = model.total - 1;
    let mut output = Vec::with_capacity(num_symbols);

    for _ in 0..num_symbols {
        let slot = (state & mask) as usize;
        let entry = model
            .slots
            .get(slot)
            .copied()
            .ok_or_else(|| PyValueError::new_err("Decoded slot index exceeds frequency table"))?;
        let freq = u64::from(entry.freq);
        let cumfreq = u64::from(entry.cumfreq);
        state = freq * (state >> SCALE_BITS) + u64::try_from(slot).unwrap() - cumfreq;
        while state < RANS_L && position < data.len() {
            state = (state << 8) | u64::from(data[position]);
            position += 1;
        }
        output.push(entry.symbol);
    }

    Ok(output)
}

#[pyfunction]
fn encode_symbols<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<'py, i64>,
    freq_table: HashMap<i64, u32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let model = Model::build(freq_table)?;
    let encoded = encode_impl(symbols.as_slice()?, &model)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
fn decode_symbols<'py>(
    py: Python<'py>,
    data: &[u8],
    freq_table: HashMap<i64, u32>,
    num_symbols: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let model = Model::build(freq_table)?;
    let decoded = decode_impl(data, &model, num_symbols)?;
    Ok(decoded.into_pyarray(py))
}

#[pymodule]
fn quench_native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(encode_symbols, module)?)?;
    module.add_function(wrap_pyfunction!(decode_symbols, module)?)?;
    Ok(())
}
