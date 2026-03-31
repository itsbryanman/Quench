"""Compression strategies for different tensor classes."""
from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np

from quench.core.config import QuenchConfig
from quench.core.exceptions import MalformedPayloadError, UnsupportedStrategyError
from quench.core.types import CodecMode, QuantMode, TensorType
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import (
    RANSEncoder,
    RANSDecoder,
    SCALE,
    build_freq_table,
    normalize_freq_table,
)
from quench.quantize import QuantParams, UniformQuantizer
from quench.transform import ChannelNormalizer, DeltaCoder, SparseEncoder


_SEGMENT_HEADER = struct.Struct("<II")
_CHUNK_COUNT_HEADER = struct.Struct("<I")
_CHUNK_LENGTH_HEADER = struct.Struct("<Q")


class CompressionStrategy(Protocol):
    """Protocol implemented by all tensor compression strategies."""

    tensor_type: TensorType
    strategy_id: int
    strategy_name: str

    def encode(self, tensor: np.ndarray[Any, np.dtype[Any]], config: QuenchConfig) -> tuple[bytes, dict[str, Any]]:
        """Encode a tensor into a payload and serializable metadata."""

    def decode(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode a tensor payload using explicit metadata only."""


class BaseCompressionStrategy(ABC):
    """Common helpers shared by the concrete codec strategies."""

    strategy_name: str

    def __init__(self) -> None:
        self._normalizer = ChannelNormalizer()
        self._quantizer = UniformQuantizer()
        self._delta = DeltaCoder()
        self._sparse = SparseEncoder()

    @abstractmethod
    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], config: QuenchConfig
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode *tensor* according to the strategy."""

    @abstractmethod
    def decode(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode *data* according to the strategy."""

    def _encode_lossless(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> tuple[bytes, dict[str, Any]]:
        """Encode original tensor bytes exactly for the lossless mode."""
        raw_bytes = np.ascontiguousarray(tensor).view(np.uint8)
        payload, stream_metadata = self._encode_symbol_stream(raw_bytes)
        return payload, {
            "dtype": np.dtype(tensor.dtype).str,
            "path": "lossless",
            "lossless": True,
            "shape": list(tensor.shape),
            "stream": stream_metadata,
        }

    def _decode_lossless(
        self,
        data: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Decode a lossless byte stream."""
        stream_metadata = self._require_mapping(metadata, "stream")
        decoded = self._decode_symbol_stream(data, stream_metadata)
        dtype = np.dtype(str(metadata.get("dtype", "")))
        shape = tuple(int(dim) for dim in metadata.get("shape", []))
        raw = np.ascontiguousarray(decoded.astype(np.uint8, copy=False))
        return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape).copy()

    def _normalize_tensor(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        axis: int = 0,
        per_channel: bool = True,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], dict[str, Any]]:
        """Normalize tensor values with either per-channel or global scaling."""
        values = np.asarray(tensor)
        if per_channel and values.ndim > 1:
            normalized, scales, zero_points = self._normalizer.normalize(values, axis=axis)
            return normalized, {
                "axis": axis,
                "kind": "channel",
                "scales": scales,
                "zero_points": zero_points,
            }

        flat = values.reshape(1, -1)
        normalized_flat, scales, zero_points = self._normalizer.normalize(flat, axis=0)
        return normalized_flat.reshape(values.shape), {
            "axis": 0,
            "kind": "global",
            "scales": scales,
            "zero_points": zero_points,
        }

    def _denormalize_tensor(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Reverse tensor normalization using explicit metadata."""
        kind = str(metadata.get("kind", ""))
        scales = np.asarray(metadata.get("scales"), dtype=np.float32)
        zero_points = np.asarray(metadata.get("zero_points"), dtype=np.float32)

        if kind == "channel":
            axis = int(metadata["axis"])
            return self._normalizer.denormalize(tensor, scales, zero_points, axis=axis)
        if kind == "global":
            reshaped = np.asarray(tensor, dtype=np.float32).reshape(1, -1)
            restored = self._normalizer.denormalize(reshaped, scales, zero_points, axis=0)
            return restored.reshape(tensor.shape)
        raise MalformedPayloadError(f"Unsupported normalization metadata kind: {kind!r}")

    def _quantize_tensor(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        bits: int,
        mode: QuantMode,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], dict[str, Any]]:
        """Quantize values and serialize the quantization parameters."""
        quantized, params = self._quantizer.quantize(tensor, bits=bits, mode=mode)
        return quantized, self._quant_params_to_metadata(params)

    def _dequantize_tensor(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Reverse uniform quantization from explicit metadata."""
        params = self._quant_params_from_metadata(metadata)
        return self._quantizer.dequantize(quantized, params)

    def _quant_params_to_metadata(self, params: QuantParams) -> dict[str, Any]:
        """Convert :class:`QuantParams` into JSON-safe metadata."""
        return {
            "bits": params.bits,
            "dtype_orig": params.dtype_orig,
            "mode": int(params.mode),
            "scale": params.scale,
            "value_range_max": params.value_range_max,
            "value_range_min": params.value_range_min,
            "zero_point": params.zero_point,
        }

    def _quant_params_from_metadata(self, metadata: dict[str, Any]) -> QuantParams:
        """Reconstruct :class:`QuantParams` from metadata."""
        try:
            return QuantParams(
                scale=float(metadata["scale"]),
                zero_point=int(metadata["zero_point"]),
                bits=int(metadata["bits"]),
                mode=QuantMode(int(metadata["mode"])),
                dtype_orig=str(metadata["dtype_orig"]),
                value_range_min=float(metadata["value_range_min"]),
                value_range_max=float(metadata["value_range_max"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedPayloadError(f"Malformed quantization metadata: {exc}") from exc

    def _encode_symbol_stream(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode integer symbols using rANS when practical, or raw bytes otherwise."""
        array = np.ascontiguousarray(symbols)
        metadata: dict[str, Any] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

        if array.size == 0:
            metadata["encoding"] = "empty"
            return b"", metadata

        unique = np.unique(array)
        if (
            unique.size <= SCALE
            and np.issubdtype(array.dtype, np.integer)
            and int(np.min(unique)) >= np.iinfo(np.int32).min
            and int(np.max(unique)) <= np.iinfo(np.int32).max
        ):
            flat = array.reshape(-1).astype(np.int64, copy=False)
            freq_table = normalize_freq_table(build_freq_table(flat))
            if sum(freq_table.values()) == SCALE:
                model_bytes = FrequencyModel.from_freq_table(freq_table).serialize()
                encoded = RANSEncoder(freq_table).encode(flat)
                metadata["encoding"] = "rans"
                return (
                    _SEGMENT_HEADER.pack(len(model_bytes), len(encoded))
                    + model_bytes
                    + encoded
                ), metadata

        metadata["encoding"] = "raw"
        return array.tobytes(), metadata

    def _decode_symbol_stream(
        self,
        data: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode a symbol stream encoded by :meth:`_encode_symbol_stream`."""
        encoding = str(metadata.get("encoding", ""))
        dtype = np.dtype(str(metadata.get("dtype", "")))
        shape = tuple(int(dim) for dim in metadata.get("shape", []))
        count = int(np.prod(shape, dtype=np.int64))

        if encoding == "empty":
            return np.empty(shape, dtype=dtype)
        if encoding == "raw":
            expected_size = count * dtype.itemsize
            if len(data) != expected_size:
                raise MalformedPayloadError(
                    "Raw symbol stream size mismatch: "
                    f"expected {expected_size}, got {len(data)}"
                )
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        if encoding == "rans":
            if len(data) < _SEGMENT_HEADER.size:
                raise MalformedPayloadError("Entropy segment is too short")
            model_len, encoded_len = _SEGMENT_HEADER.unpack_from(data, 0)
            model_start = _SEGMENT_HEADER.size
            model_end = model_start + model_len
            if model_end > len(data):
                raise MalformedPayloadError("Entropy model length exceeds payload size")
            encoded_end = model_end + encoded_len
            if encoded_end != len(data):
                raise MalformedPayloadError(
                    "Entropy payload length mismatch: "
                    f"expected {encoded_end} bytes, got {len(data)}"
                )
            model = FrequencyModel.deserialize(data[model_start:model_end])
            decoded = RANSDecoder(model.freq_table).decode(data[model_end:encoded_end], count)
            return decoded.astype(dtype, copy=False).reshape(shape)
        raise MalformedPayloadError(f"Unsupported symbol stream encoding: {encoding!r}")

    def _encode_sparse_indices(
        self, indices: np.ndarray[Any, np.dtype[Any]]
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode monotonically increasing sparse indices as delta symbols."""
        working = np.asarray(indices, dtype=np.int64)
        deltas = np.empty_like(working)
        if working.size:
            deltas[0] = working[0]
            deltas[1:] = np.diff(working)
        payload, stream_metadata = self._encode_symbol_stream(deltas)
        return payload, {
            "count": int(working.size),
            "stream": stream_metadata,
        }

    def _decode_sparse_indices(
        self,
        data: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Decode sparse indices stored as delta symbols."""
        stream_metadata = self._require_mapping(metadata, "stream")
        deltas = self._decode_symbol_stream(data, stream_metadata).astype(np.int64, copy=False)
        if deltas.size == 0:
            return np.empty((0,), dtype=np.int64)
        return np.cumsum(deltas, dtype=np.int64)

    def _pack_chunks(self, chunks: list[bytes]) -> bytes:
        """Pack a list of opaque chunks with explicit lengths."""
        header = bytearray(_CHUNK_COUNT_HEADER.pack(len(chunks)))
        for chunk in chunks:
            header.extend(_CHUNK_LENGTH_HEADER.pack(len(chunk)))
        for chunk in chunks:
            header.extend(chunk)
        return bytes(header)

    def _unpack_chunks(self, data: bytes, *, expected_count: int) -> list[bytes]:
        """Unpack a byte string previously created by :meth:`_pack_chunks`."""
        if len(data) < _CHUNK_COUNT_HEADER.size:
            raise MalformedPayloadError("Chunked payload is too short")
        (count,) = _CHUNK_COUNT_HEADER.unpack_from(data, 0)
        if count != expected_count:
            raise MalformedPayloadError(
                f"Expected {expected_count} payload chunks, found {count}"
            )

        offset = _CHUNK_COUNT_HEADER.size
        lengths: list[int] = []
        for _ in range(count):
            if offset + _CHUNK_LENGTH_HEADER.size > len(data):
                raise MalformedPayloadError("Chunked payload is missing a length entry")
            (length,) = _CHUNK_LENGTH_HEADER.unpack_from(data, offset)
            lengths.append(length)
            offset += _CHUNK_LENGTH_HEADER.size

        chunks: list[bytes] = []
        for length in lengths:
            end = offset + length
            if end > len(data):
                raise MalformedPayloadError("Chunked payload is truncated")
            chunks.append(data[offset:end])
            offset = end

        if offset != len(data):
            raise MalformedPayloadError("Chunked payload has trailing bytes")
        return chunks

    def _require_mapping(self, metadata: dict[str, Any], key: str) -> dict[str, Any]:
        """Fetch a mapping from metadata with type validation."""
        value = metadata.get(key)
        if not isinstance(value, dict):
            raise MalformedPayloadError(f"Metadata field {key!r} must be a mapping")
        return value


class WeightStrategy(BaseCompressionStrategy):
    """Weight tensors: normalize, quantize, and entropy-code."""

    tensor_type = TensorType.WEIGHT
    strategy_id = 1
    strategy_name = "weight"

    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], config: QuenchConfig
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor)

        normalized, normalization_metadata = self._normalize_tensor(
            tensor,
            axis=0,
            per_channel=config.per_channel,
        )
        quantized, quant_metadata = self._quantize_tensor(
            normalized,
            bits=config.target_bits,
            mode=config.quant_mode,
        )
        payload, stream_metadata = self._encode_symbol_stream(quantized)
        return payload, {
            "lossless": False,
            "normalization": normalization_metadata,
            "path": "dense",
            "quantization": quant_metadata,
            "stream": stream_metadata,
        }

    def decode(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if metadata.get("lossless") is True:
            return self._decode_lossless(data, metadata)

        stream_metadata = self._require_mapping(metadata, "stream")
        quantized = self._decode_symbol_stream(data, stream_metadata)
        restored = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        return self._denormalize_tensor(restored, self._require_mapping(metadata, "normalization"))


class KVCacheStrategy(BaseCompressionStrategy):
    """KV-cache tensors: delta-code, normalize, quantize, and entropy-code."""

    tensor_type = TensorType.KV_CACHE
    strategy_id = 2
    strategy_name = "kv_cache"

    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], config: QuenchConfig
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor)

        delta_axis = self._choose_delta_axis(tensor)
        deltas, anchor = self._delta.encode(tensor, axis=delta_axis)
        normalized, normalization_metadata = self._normalize_tensor(
            deltas,
            axis=max(deltas.ndim - 1, 0),
            per_channel=config.per_channel,
        )
        bits = min(config.target_bits + 1, 8)
        quantized, quant_metadata = self._quantize_tensor(
            normalized,
            bits=bits,
            mode=config.quant_mode,
        )
        payload, stream_metadata = self._encode_symbol_stream(quantized)
        return payload, {
            "anchor": anchor,
            "delta": {"axis": delta_axis},
            "lossless": False,
            "normalization": normalization_metadata,
            "path": "dense",
            "quantization": quant_metadata,
            "stream": stream_metadata,
        }

    def decode(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if metadata.get("lossless") is True:
            return self._decode_lossless(data, metadata)

        quantized = self._decode_symbol_stream(data, self._require_mapping(metadata, "stream"))
        normalized = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        deltas = self._denormalize_tensor(normalized, self._require_mapping(metadata, "normalization"))
        delta_metadata = self._require_mapping(metadata, "delta")
        anchor = np.asarray(metadata.get("anchor"))
        return self._delta.decode(deltas, anchor, axis=int(delta_metadata["axis"]))

    def _choose_delta_axis(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> int:
        """Pick the longest non-feature axis for cache delta coding."""
        values = np.asarray(tensor)
        if values.ndim <= 1:
            return 0
        candidate_axes = list(range(max(values.ndim - 1, 1)))
        return max(candidate_axes, key=lambda axis: int(values.shape[axis]))


class EmbeddingStrategy(BaseCompressionStrategy):
    """Embedding tensors with an optional sparse path."""

    tensor_type = TensorType.EMBEDDING
    strategy_id = 3
    strategy_name = "embedding"

    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], config: QuenchConfig
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor)

        if self._should_use_sparse(tensor, threshold=0.30):
            return self._encode_sparse(tensor, config)

        normalized, normalization_metadata = self._normalize_tensor(
            tensor,
            axis=0,
            per_channel=config.per_channel,
        )
        quantized, quant_metadata = self._quantize_tensor(
            normalized,
            bits=config.target_bits,
            mode=config.quant_mode,
        )
        payload, stream_metadata = self._encode_symbol_stream(quantized)
        return payload, {
            "lossless": False,
            "normalization": normalization_metadata,
            "path": "dense",
            "quantization": quant_metadata,
            "stream": stream_metadata,
        }

    def decode(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if metadata.get("lossless") is True:
            return self._decode_lossless(data, metadata)
        if metadata.get("path") == "sparse":
            return self._decode_sparse(data, metadata)

        quantized = self._decode_symbol_stream(data, self._require_mapping(metadata, "stream"))
        normalized = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        return self._denormalize_tensor(normalized, self._require_mapping(metadata, "normalization"))

    def _encode_sparse(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode sparse embeddings by storing indices separately."""
        sparse = self._sparse.encode(tensor, threshold=0.0)
        index_payload, index_metadata = self._encode_sparse_indices(sparse.indices)

        if sparse.nnz == 0:
            value_payload = b""
            value_metadata = {"encoding": "empty", "shape": [0], "dtype": np.dtype(np.int8).str}
            quant_metadata: dict[str, Any] | None = None
            normalization_metadata: dict[str, Any] | None = None
        else:
            values = sparse.values.astype(np.float32, copy=False).reshape(1, -1)
            normalized_values, normalization_metadata = self._normalize_tensor(
                values,
                axis=0,
                per_channel=False,
            )
            quantized_values, quant_metadata = self._quantize_tensor(
                normalized_values.reshape(-1),
                bits=config.target_bits,
                mode=config.quant_mode,
            )
            value_payload, value_metadata = self._encode_symbol_stream(quantized_values)

        payload = self._pack_chunks([index_payload, value_payload])
        metadata = {
            "index_stream": index_metadata,
            "lossless": False,
            "nnz": sparse.nnz,
            "normalization": normalization_metadata,
            "path": "sparse",
            "quantization": quant_metadata,
            "shape": list(tensor.shape),
            "value_stream": value_metadata,
        }
        return payload, metadata

    def _decode_sparse(
        self,
        data: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Decode sparse embedding payloads."""
        shape = tuple(int(dim) for dim in metadata.get("shape", []))
        size = int(np.prod(shape, dtype=np.int64))
        chunks = self._unpack_chunks(data, expected_count=2)
        indices = self._decode_sparse_indices(chunks[0], self._require_mapping(metadata, "index_stream"))

        dense = np.zeros(size, dtype=np.float32)
        if int(metadata.get("nnz", 0)) == 0:
            return dense.reshape(shape)

        quantization_metadata = self._require_mapping(metadata, "quantization")
        normalization_metadata = self._require_mapping(metadata, "normalization")
        value_stream = self._require_mapping(metadata, "value_stream")
        quantized_values = self._decode_symbol_stream(chunks[1], value_stream)
        normalized_values = self._dequantize_tensor(quantized_values, quantization_metadata)
        restored_values = self._denormalize_tensor(
            normalized_values.reshape(1, -1),
            normalization_metadata,
        ).reshape(-1)

        dense[indices.astype(np.intp, copy=False)] = restored_values.astype(np.float32, copy=False)
        return dense.reshape(shape)

    def _should_use_sparse(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        threshold: float,
    ) -> bool:
        """Return ``True`` when exact zeros make sparse storage worthwhile."""
        values = np.asarray(tensor)
        if values.size < 4_096:
            return False
        zero_fraction = float(np.count_nonzero(values == 0) / values.size)
        return zero_fraction >= threshold


class ActivationStrategy(BaseCompressionStrategy):
    """Activation tensors with an optional sparse path."""

    tensor_type = TensorType.ACTIVATION
    strategy_id = 4
    strategy_name = "activation"

    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], config: QuenchConfig
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor)

        if self._should_use_sparse(tensor, threshold=0.75):
            return self._encode_sparse(tensor, config)

        quant_mode = self._activation_quant_mode(tensor, config.quant_mode)
        bits = min(config.target_bits + 1, 8)
        quantized, quant_metadata = self._quantize_tensor(
            tensor,
            bits=bits,
            mode=quant_mode,
        )
        payload, stream_metadata = self._encode_symbol_stream(quantized)
        return payload, {
            "lossless": False,
            "path": "dense",
            "quantization": quant_metadata,
            "stream": stream_metadata,
        }

    def decode(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if metadata.get("lossless") is True:
            return self._decode_lossless(data, metadata)
        if metadata.get("path") == "sparse":
            return self._decode_sparse(data, metadata)

        quantized = self._decode_symbol_stream(data, self._require_mapping(metadata, "stream"))
        return self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))

    def _encode_sparse(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode sparse activations by storing non-zero values and indices."""
        sparse = self._sparse.encode(tensor, threshold=0.0)
        index_payload, index_metadata = self._encode_sparse_indices(sparse.indices)

        if sparse.nnz == 0:
            value_payload = b""
            value_metadata = {"encoding": "empty", "shape": [0], "dtype": np.dtype(np.int8).str}
            quant_metadata: dict[str, Any] | None = None
        else:
            quant_mode = self._activation_quant_mode(sparse.values, config.quant_mode)
            bits = min(config.target_bits + 1, 8)
            quantized_values, quant_metadata = self._quantize_tensor(
                sparse.values.astype(np.float32, copy=False),
                bits=bits,
                mode=quant_mode,
            )
            value_payload, value_metadata = self._encode_symbol_stream(quantized_values)

        payload = self._pack_chunks([index_payload, value_payload])
        return payload, {
            "index_stream": index_metadata,
            "lossless": False,
            "nnz": sparse.nnz,
            "path": "sparse",
            "quantization": quant_metadata,
            "shape": list(tensor.shape),
            "value_stream": value_metadata,
        }

    def _decode_sparse(
        self,
        data: bytes,
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Decode sparse activation payloads."""
        shape = tuple(int(dim) for dim in metadata.get("shape", []))
        size = int(np.prod(shape, dtype=np.int64))
        chunks = self._unpack_chunks(data, expected_count=2)
        indices = self._decode_sparse_indices(chunks[0], self._require_mapping(metadata, "index_stream"))

        dense = np.zeros(size, dtype=np.float32)
        if int(metadata.get("nnz", 0)) == 0:
            return dense.reshape(shape)

        value_stream = self._require_mapping(metadata, "value_stream")
        quantized_values = self._decode_symbol_stream(chunks[1], value_stream)
        values = self._dequantize_tensor(quantized_values, self._require_mapping(metadata, "quantization"))
        dense[indices.astype(np.intp, copy=False)] = values.astype(np.float32, copy=False)
        return dense.reshape(shape)

    def _should_use_sparse(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        *,
        threshold: float,
    ) -> bool:
        """Return ``True`` when exact zero activations dominate the tensor."""
        values = np.asarray(tensor)
        if values.size < 2_048:
            return False
        zero_fraction = float(np.count_nonzero(values == 0) / values.size)
        return zero_fraction >= threshold

    def _activation_quant_mode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        requested_mode: QuantMode,
    ) -> QuantMode:
        """Prefer asymmetric quantization for non-negative activations."""
        values = np.asarray(tensor)
        if requested_mode == QuantMode.SYMMETRIC and values.size and float(np.min(values)) >= 0.0:
            return QuantMode.ASYMMETRIC
        return requested_mode


class DefaultStrategy(BaseCompressionStrategy):
    """Fallback strategy that quantizes directly and entropy-codes the result."""

    tensor_type = TensorType.UNKNOWN
    strategy_id = 255
    strategy_name = "default"

    def encode(
        self, tensor: np.ndarray[Any, np.dtype[Any]], config: QuenchConfig
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor)

        quantized, quant_metadata = self._quantize_tensor(
            tensor,
            bits=config.target_bits,
            mode=config.quant_mode,
        )
        payload, stream_metadata = self._encode_symbol_stream(quantized)
        return payload, {
            "lossless": False,
            "path": "dense",
            "quantization": quant_metadata,
            "stream": stream_metadata,
        }

    def decode(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if metadata.get("lossless") is True:
            return self._decode_lossless(data, metadata)

        quantized = self._decode_symbol_stream(data, self._require_mapping(metadata, "stream"))
        return self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))


WEIGHT_STRATEGY = WeightStrategy()
KV_CACHE_STRATEGY = KVCacheStrategy()
EMBEDDING_STRATEGY = EmbeddingStrategy()
ACTIVATION_STRATEGY = ActivationStrategy()
DEFAULT_STRATEGY = DefaultStrategy()

STRATEGY_REGISTRY: dict[TensorType, CompressionStrategy] = {
    TensorType.WEIGHT: WEIGHT_STRATEGY,
    TensorType.KV_CACHE: KV_CACHE_STRATEGY,
    TensorType.EMBEDDING: EMBEDDING_STRATEGY,
    TensorType.ACTIVATION: ACTIVATION_STRATEGY,
}

STRATEGY_ID_REGISTRY: dict[int, CompressionStrategy] = {
    WEIGHT_STRATEGY.strategy_id: WEIGHT_STRATEGY,
    KV_CACHE_STRATEGY.strategy_id: KV_CACHE_STRATEGY,
    EMBEDDING_STRATEGY.strategy_id: EMBEDDING_STRATEGY,
    ACTIVATION_STRATEGY.strategy_id: ACTIVATION_STRATEGY,
    DEFAULT_STRATEGY.strategy_id: DEFAULT_STRATEGY,
}


def get_strategy(tensor_type: TensorType) -> CompressionStrategy:
    """Return the registered strategy for *tensor_type* with a default fallback."""
    return STRATEGY_REGISTRY.get(tensor_type, DEFAULT_STRATEGY)


def get_strategy_by_id(strategy_id: int, tensor_type: TensorType) -> CompressionStrategy:
    """Resolve a strategy by ID and validate it against the tensor type."""
    if strategy_id == 0:
        return get_strategy(tensor_type)

    strategy = STRATEGY_ID_REGISTRY.get(strategy_id)
    if strategy is None:
        raise UnsupportedStrategyError(f"Unsupported strategy id: {strategy_id}")
    if strategy.tensor_type not in (tensor_type, TensorType.UNKNOWN):
        raise UnsupportedStrategyError(
            "Strategy id does not match tensor type: "
            f"strategy={strategy.strategy_name}, tensor_type={tensor_type.name}"
        )
    return strategy
