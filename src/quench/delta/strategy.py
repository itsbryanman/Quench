"""Delta-specific compression paths."""
from __future__ import annotations

import struct
from typing import Any

import numpy as np

from quench.backends.registry import get_entropy_backend
from quench.core.config import QuantizationGranularity, QuenchConfig
from quench.core.types import QuantMode
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import SCALE, build_freq_table, normalize_freq_table
from quench.quantize import (
    PerChannelCalibrationPolicy,
    PerChannelQuantizer,
    PerTensorCalibrationPolicy,
    PerTensorQuantizer,
    QuantizationLayout,
    deserialize_quant_params,
    serialize_quant_params,
)
from quench.transform import ChannelNormalizer, SparseEncoder, SparseRepresentation

_SECTION_HEADER = struct.Struct("<II")
_SPARSE_SECTIONS = struct.Struct("<III")


def encode_zero() -> tuple[bytes, dict[str, Any]]:
    """Encode an all-zero delta. Payload is empty."""
    return b"", {"path": "zero"}


def decode_zero(
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Decode an all-zero delta."""
    return np.zeros(shape, dtype=np.dtype(dtype))


def encode_sign_scale(
    delta: np.ndarray[Any, np.dtype[Any]],
    config: QuenchConfig,
) -> tuple[bytes, dict[str, Any]]:
    """Encode using a 1-bit sign mask and per-row float16 scale factors."""
    values = np.asarray(delta, dtype=np.float32)
    work = values.reshape(1, -1) if values.ndim == 1 else values.reshape(values.shape[0], -1)

    scales = np.mean(np.abs(work), axis=1, keepdims=True).astype(np.float16)
    sign_bits = (work >= 0).astype(np.uint8)
    packed_signs = np.packbits(sign_bits.ravel(), bitorder="little")

    sign_payload, sign_stream_meta = _encode_rans_stream(packed_signs.astype(np.int64), config)
    scales_bytes = np.ascontiguousarray(scales).tobytes()
    payload = _SECTION_HEADER.pack(len(scales_bytes), len(sign_payload)) + scales_bytes + sign_payload

    return payload, {
        "path": "sign_scale",
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "num_elements": int(values.size),
        "scale_rows": int(work.shape[0]),
        "sign_stream": sign_stream_meta,
    }


def decode_sign_scale(
    payload: bytes,
    metadata: dict[str, Any],
    config: QuenchConfig,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Decode a sign_scale-encoded delta."""
    shape = tuple(int(dim) for dim in metadata["shape"])
    num_elements = int(metadata["num_elements"])
    scale_rows = int(metadata["scale_rows"])

    scales_len, sign_len = _SECTION_HEADER.unpack_from(payload, 0)
    offset = _SECTION_HEADER.size

    scales = np.frombuffer(payload[offset : offset + scales_len], dtype=np.float16).copy()
    offset += scales_len

    sign_stream = _decode_rans_stream(payload[offset : offset + sign_len], metadata["sign_stream"], config)
    packed_signs = sign_stream.astype(np.uint8, copy=False)
    sign_bits = np.unpackbits(packed_signs, bitorder="little")[:num_elements]
    signs = sign_bits.astype(np.float32) * 2.0 - 1.0

    restored = signs.reshape(scale_rows, -1) * scales.astype(np.float32).reshape(scale_rows, -1)
    return restored.reshape(shape)


def encode_sparse(
    delta: np.ndarray[Any, np.dtype[Any]],
    config: QuenchConfig,
    *,
    bits: int = 4,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a highly sparse delta using index + value streams."""
    values = np.asarray(delta, dtype=np.float32)
    sparsified = np.where(np.abs(values) >= 1e-7, values, 0.0).astype(np.float32, copy=False)

    sparse_encoder = SparseEncoder()
    sparse = sparse_encoder.encode(sparsified, threshold=0.0)
    indices = sparse.indices.astype(np.int64, copy=False)
    nonzero_values = sparse.values.astype(np.float32, copy=False)
    nnz = int(sparse.nnz)

    if nnz > 0:
        index_deltas = np.empty_like(indices)
        index_deltas[0] = indices[0]
        index_deltas[1:] = np.diff(indices)
    else:
        index_deltas = np.empty(0, dtype=np.int64)

    index_payload, index_meta = _encode_rans_stream(index_deltas, config)

    if nnz > 0:
        quantizer = PerTensorQuantizer()
        calibrator = PerTensorCalibrationPolicy()
        layout = QuantizationLayout(granularity=QuantizationGranularity.PER_TENSOR, axis=0)
        params = calibrator.calibrate(
            nonzero_values,
            bits=bits,
            mode=QuantMode.SYMMETRIC,
            layout=layout,
        )
        quantized = quantizer.quantize(nonzero_values, params)
        quant_meta = serialize_quant_params(params)
        value_payload, value_meta = _encode_rans_stream(quantized.astype(np.int64), config)
    else:
        quant_meta = {}
        value_payload = b""
        value_meta = {"encoding": "empty", "dtype": np.dtype(np.int64).str, "shape": [0]}

    payload = _SPARSE_SECTIONS.pack(len(index_payload), len(value_payload), nnz) + index_payload + value_payload
    return payload, {
        "path": "sparse",
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "nnz": nnz,
        "bits": bits,
        "index_stream": index_meta,
        "value_stream": value_meta,
        "quantization": quant_meta,
    }


def decode_sparse(
    payload: bytes,
    metadata: dict[str, Any],
    config: QuenchConfig,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Decode a sparse-encoded delta."""
    shape = tuple(int(dim) for dim in metadata["shape"])
    idx_len, val_len, stored_nnz = _SPARSE_SECTIONS.unpack_from(payload, 0)
    offset = _SPARSE_SECTIONS.size

    nnz = int(metadata["nnz"])
    if nnz != int(stored_nnz):
        raise ValueError(f"Sparse payload nnz mismatch: metadata={nnz}, payload={stored_nnz}")
    if nnz == 0:
        return np.zeros(shape, dtype=np.float32)

    index_deltas = _decode_rans_stream(payload[offset : offset + idx_len], metadata["index_stream"], config)
    offset += idx_len
    indices = np.cumsum(index_deltas.astype(np.int64, copy=False))

    quantized = _decode_rans_stream(payload[offset : offset + val_len], metadata["value_stream"], config)
    params = deserialize_quant_params(metadata["quantization"])
    quantizer = PerTensorQuantizer()
    restored_values = quantizer.dequantize(quantized, params).astype(np.float32, copy=False)

    sparse = SparseRepresentation(
        indices=indices.astype(np.uint32 if np.prod(shape, dtype=np.int64) <= np.iinfo(np.uint32).max else np.uint64),
        values=restored_values,
        shape=shape,
        nnz=nnz,
        dtype_orig=np.dtype(np.float32).str,
    )
    return SparseEncoder().decode(sparse).astype(np.float32, copy=False)


def encode_quantize(
    delta: np.ndarray[Any, np.dtype[Any]],
    config: QuenchConfig,
    *,
    bits: int = 2,
) -> tuple[bytes, dict[str, Any]]:
    """Encode with per-channel normalization, quantization, and entropy coding."""
    values = np.asarray(delta, dtype=np.float32)
    normalizer = ChannelNormalizer()

    if values.ndim >= 2:
        normalized, scales, zero_points = normalizer.normalize(values, axis=0)
        norm_meta: dict[str, Any] = {
            "kind": "channel",
            "axis": 0,
            "scales": scales,
            "zero_points": zero_points,
        }
        quantizer = PerChannelQuantizer(axis=0)
        calibrator = PerChannelCalibrationPolicy()
        layout = QuantizationLayout(granularity=QuantizationGranularity.PER_CHANNEL, axis=0)
    else:
        normalized_2d, scales, zero_points = normalizer.normalize(values.reshape(1, -1), axis=0)
        normalized = normalized_2d.reshape(values.shape)
        norm_meta = {
            "kind": "global",
            "scales": scales,
            "zero_points": zero_points,
        }
        quantizer = PerTensorQuantizer()
        calibrator = PerTensorCalibrationPolicy()
        layout = QuantizationLayout(granularity=QuantizationGranularity.PER_TENSOR, axis=0)

    params = calibrator.calibrate(
        normalized,
        bits=bits,
        mode=QuantMode.SYMMETRIC,
        layout=layout,
    )
    quantized = quantizer.quantize(normalized, params)
    payload, stream_meta = _encode_rans_stream(quantized.ravel().astype(np.int64), config)

    return payload, {
        "path": "quantize",
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "bits": bits,
        "normalization": norm_meta,
        "quantization": serialize_quant_params(params),
        "stream": stream_meta,
    }


def decode_quantize(
    payload: bytes,
    metadata: dict[str, Any],
    config: QuenchConfig,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Decode a quantize-path delta."""
    shape = tuple(int(dim) for dim in metadata["shape"])
    symbols = _decode_rans_stream(payload, metadata["stream"], config)
    params = deserialize_quant_params(metadata["quantization"])

    norm_meta = metadata["normalization"]
    kind = str(norm_meta.get("kind", ""))
    if kind == "channel":
        quantizer = PerChannelQuantizer(axis=int(norm_meta["axis"]))
    else:
        quantizer = PerTensorQuantizer()

    dequantized = quantizer.dequantize(symbols.reshape(shape), params)

    normalizer = ChannelNormalizer()
    scales = np.asarray(norm_meta["scales"], dtype=np.float32)
    zero_points = np.asarray(norm_meta["zero_points"], dtype=np.float32)
    if kind == "channel":
        restored = normalizer.denormalize(
            dequantized,
            scales,
            zero_points,
            axis=int(norm_meta["axis"]),
        )
    else:
        restored_2d = normalizer.denormalize(
            dequantized.astype(np.float32, copy=False).reshape(1, -1),
            scales,
            zero_points,
            axis=0,
        )
        restored = restored_2d.reshape(shape)

    return restored.astype(np.float32, copy=False)


def encode_lossless(
    delta: np.ndarray[Any, np.dtype[Any]],
) -> tuple[bytes, dict[str, Any]]:
    """Store the raw delta bytes without any lossy transformation."""
    arr = np.ascontiguousarray(delta)
    return arr.tobytes(), {
        "path": "lossless",
        "shape": list(arr.shape),
        "dtype": arr.dtype.str,
    }


def decode_lossless(
    payload: bytes,
    metadata: dict[str, Any],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Decode a raw lossless delta."""
    shape = tuple(int(dim) for dim in metadata["shape"])
    dtype = np.dtype(str(metadata["dtype"]))
    return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()


def encode_delta(
    delta: np.ndarray[Any, np.dtype[Any]],
    path: str,
    config: QuenchConfig,
    *,
    bits: int = 2,
) -> tuple[bytes, dict[str, Any]]:
    """Route to the appropriate encode function based on *path*."""
    if path == "zero":
        payload, metadata = encode_zero()
        values = np.asarray(delta)
        metadata.update(
            {
                "shape": list(values.shape),
                "dtype": values.dtype.str,
            }
        )
        return payload, metadata
    if path == "sign_scale":
        return encode_sign_scale(delta, config)
    if path == "sparse":
        return encode_sparse(delta, config, bits=bits)
    if path == "quantize":
        return encode_quantize(delta, config, bits=bits)
    if path == "lossless":
        return encode_lossless(delta)
    raise ValueError(f"Unknown delta compression path: {path!r}")


def decode_delta(
    payload: bytes,
    metadata: dict[str, Any],
    config: QuenchConfig,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Route to the appropriate decode function based on metadata."""
    path = str(metadata.get("path", ""))
    shape = tuple(int(dim) for dim in metadata.get("shape", []))
    dtype_str = str(metadata.get("dtype", "<f4"))

    if path == "zero":
        return decode_zero(shape, dtype_str)
    if path == "sign_scale":
        return decode_sign_scale(payload, metadata, config)
    if path == "sparse":
        return decode_sparse(payload, metadata, config)
    if path == "quantize":
        return decode_quantize(payload, metadata, config)
    if path == "lossless":
        return decode_lossless(payload, metadata)
    raise ValueError(f"Unknown delta compression path: {path!r}")


def _encode_rans_stream(
    symbols: np.ndarray[Any, np.dtype[Any]],
    config: QuenchConfig,
) -> tuple[bytes, dict[str, Any]]:
    """Entropy-code an int64 symbol array with rANS, returning payload + metadata."""
    arr = np.ascontiguousarray(symbols.ravel().astype(np.int64))
    meta: dict[str, Any] = {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }

    if arr.size == 0:
        meta["encoding"] = "empty"
        return b"", meta

    unique = np.unique(arr)
    if (
        unique.size <= SCALE
        and int(np.min(unique)) >= np.iinfo(np.int32).min
        and int(np.max(unique)) <= np.iinfo(np.int32).max
    ):
        freq_table = normalize_freq_table(build_freq_table(arr))
        model = FrequencyModel.from_freq_table(freq_table)
        model_bytes = model.serialize()
        backend = get_entropy_backend(config.entropy_backend)
        encoded = backend.encode_symbols(arr, model)
        meta["encoding"] = "rans"
        return _SECTION_HEADER.pack(len(model_bytes), len(encoded)) + model_bytes + encoded, meta

    meta["encoding"] = "raw"
    return arr.tobytes(), meta


def _decode_rans_stream(
    data: bytes,
    metadata: dict[str, Any],
    config: QuenchConfig,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Decode an rANS stream produced by ``_encode_rans_stream``."""
    encoding = str(metadata.get("encoding", ""))
    dtype = np.dtype(str(metadata.get("dtype", "<i8")))
    shape = tuple(int(dim) for dim in metadata.get("shape", []))
    count = int(np.prod(shape, dtype=np.int64))

    if encoding == "empty":
        return np.empty(shape, dtype=dtype)
    if encoding == "raw":
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if encoding == "rans":
        model_len, encoded_len = _SECTION_HEADER.unpack_from(data, 0)
        model_start = _SECTION_HEADER.size
        model_end = model_start + model_len
        model = FrequencyModel.deserialize(data[model_start:model_end])
        backend = get_entropy_backend(config.entropy_backend)
        decoded = backend.decode_symbols(data[model_end : model_end + encoded_len], model, count)
        return decoded.astype(dtype, copy=False).reshape(shape)
    raise ValueError(f"Unsupported stream encoding: {encoding!r}")
