"""Compression strategies for different tensor classes."""
from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np

from quench.backends.registry import get_entropy_backend, get_packing_backend
from quench.core.config import (
    CalibrationPolicyKind,
    QuantizationGranularity,
    QuenchConfig,
)
from quench.core.exceptions import MalformedPayloadError, UnsupportedStrategyError
from quench.core.types import CodecMode, QuantMode, TensorType
from quench.entropy.freq_model import FrequencyModel
from quench.entropy.rans import SCALE, build_freq_table, normalize_freq_table
from quench.quantize import (
    BlockwiseCalibrationPolicy,
    BlockwiseQuantizer,
    PerChannelCalibrationPolicy,
    PerChannelQuantizer,
    PerTensorCalibrationPolicy,
    PerTensorQuantizer,
    PercentileCalibrationPolicy,
    QuantizationLayout,
    deserialize_layout,
    deserialize_quant_params,
    serialize_layout,
    serialize_quant_params,
)
from quench.transform import ChannelNormalizer, DeltaCoder, SparseEncoder

_SEGMENT_HEADER = struct.Struct("<II")
_CHUNK_COUNT_HEADER = struct.Struct("<I")
_CHUNK_LENGTH_HEADER = struct.Struct("<Q")


class CompressionStrategy(Protocol):
    """Protocol implemented by all tensor compression strategies."""

    tensor_type: TensorType
    strategy_id: int
    strategy_name: str

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
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
        self._delta = DeltaCoder()
        self._sparse = SparseEncoder()

    @abstractmethod
    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
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

    def _active_config(self, config: QuenchConfig | None) -> QuenchConfig:
        return config or QuenchConfig()

    def _encode_lossless(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode original tensor bytes exactly for the lossless mode."""
        raw_bytes = np.ascontiguousarray(tensor).view(np.uint8)
        payload, stream_metadata = self._encode_symbol_stream(raw_bytes, config=config)
        return payload, {
            "dtype": np.dtype(tensor.dtype).str,
            "lossless": True,
            "path": "lossless",
            "shape": list(tensor.shape),
            "stream": stream_metadata,
        }

    def _decode_lossless(
        self,
        data: bytes,
        metadata: dict[str, Any],
        config: QuenchConfig | None,
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Decode a lossless byte stream."""
        stream_metadata = self._require_mapping(metadata, "stream")
        decoded = self._decode_symbol_stream(data, stream_metadata, config=config)
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
        config: QuenchConfig,
        default_axis: int = 0,
        compact_1d: bool = False,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], dict[str, Any]]:
        """Quantize values using reusable quantizer/calibration components."""
        values = np.asarray(tensor)
        layout = self._resolve_layout(
            config,
            tensor_ndim=values.ndim,
            default_axis=default_axis,
            compact_1d=compact_1d,
        )
        quantizer = self._resolve_quantizer(layout)
        calibrator = self._resolve_calibration_policy(config, layout)
        params = calibrator.calibrate(values, bits=bits, mode=mode, layout=layout)
        quantized = quantizer.quantize(values, params)
        return quantized, {
            "layout": serialize_layout(layout),
            "params": serialize_quant_params(params),
        }

    def _dequantize_tensor(
        self,
        quantized: np.ndarray[Any, np.dtype[Any]],
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Reverse quantization from explicit layout and parameter metadata."""
        layout = deserialize_layout(self._require_mapping(metadata, "layout"))
        params = deserialize_quant_params(self._require_mapping(metadata, "params"))
        quantizer = self._resolve_quantizer(layout)
        return quantizer.dequantize(quantized, params)

    def _resolve_layout(
        self,
        config: QuenchConfig,
        *,
        tensor_ndim: int,
        default_axis: int,
        compact_1d: bool = False,
    ) -> QuantizationLayout:
        granularity = config.quantization_granularity
        if compact_1d and tensor_ndim <= 1:
            granularity = QuantizationGranularity.PER_TENSOR
        axis = default_axis if config.quantization_axis is None else config.quantization_axis
        block_size = config.block_size if granularity == QuantizationGranularity.BLOCKWISE else None
        return QuantizationLayout(granularity=granularity, axis=axis, block_size=block_size)

    def _resolve_quantizer(self, layout: QuantizationLayout) -> Any:
        if layout.granularity == QuantizationGranularity.PER_TENSOR:
            return PerTensorQuantizer()
        if layout.granularity == QuantizationGranularity.PER_CHANNEL:
            return PerChannelQuantizer(axis=layout.axis)
        if layout.granularity == QuantizationGranularity.BLOCKWISE:
            return BlockwiseQuantizer(axis=layout.axis, block_size=int(layout.block_size or 0))
        raise MalformedPayloadError(f"Unsupported quantization granularity: {layout.granularity.value}")

    def _resolve_calibration_policy(
        self,
        config: QuenchConfig,
        layout: QuantizationLayout,
    ) -> Any:
        if config.calibration_policy == CalibrationPolicyKind.MINMAX:
            if layout.granularity == QuantizationGranularity.PER_TENSOR:
                return PerTensorCalibrationPolicy()
            if layout.granularity == QuantizationGranularity.PER_CHANNEL:
                return PerChannelCalibrationPolicy()
            return BlockwiseCalibrationPolicy()
        if config.calibration_policy == CalibrationPolicyKind.PERCENTILE:
            return PercentileCalibrationPolicy(percentile=config.percentile_value)
        if config.calibration_policy == CalibrationPolicyKind.PER_CHANNEL:
            return PerChannelCalibrationPolicy()
        if config.calibration_policy == CalibrationPolicyKind.BLOCKWISE:
            return BlockwiseCalibrationPolicy()
        raise MalformedPayloadError(
            f"Unsupported calibration policy: {config.calibration_policy.value!r}"
        )

    def _encode_symbol_stream(
        self,
        symbols: np.ndarray[Any, np.dtype[Any]],
        *,
        config: QuenchConfig,
        bits: int | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode integer symbols with backend-dispatched entropy or bit packing."""
        array = np.ascontiguousarray(symbols)
        metadata: dict[str, Any] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

        if array.size == 0:
            metadata["encoding"] = "empty"
            return b"", metadata

        if (
            bits is not None
            and config.pack_bits
            and np.issubdtype(array.dtype, np.integer)
            and bits < array.dtype.itemsize * 8
        ):
            backend = get_packing_backend(config.packing_backend)
            packed = backend.pack_bits(
                array,
                bits=bits,
                signed=bool(np.issubdtype(array.dtype, np.signedinteger)),
                layout_metadata={"dtype": array.dtype.str},
            )
            metadata.update(
                {
                    "bits": bits,
                    "encoding": "packed",
                    "signed": bool(np.issubdtype(array.dtype, np.signedinteger)),
                }
            )
            return packed, metadata

        unique = np.unique(array)
        if (
            config.entropy_coder == "rans"
            and unique.size <= SCALE
            and np.issubdtype(array.dtype, np.integer)
            and int(np.min(unique)) >= np.iinfo(np.int32).min
            and int(np.max(unique)) <= np.iinfo(np.int32).max
        ):
            flat = array.reshape(-1).astype(np.int64, copy=False)
            freq_table = normalize_freq_table(build_freq_table(flat))
            if sum(freq_table.values()) == SCALE:
                model = FrequencyModel.from_freq_table(freq_table)
                model_bytes = model.serialize()
                encoded = get_entropy_backend(config.entropy_backend).encode_symbols(flat, model)
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
        *,
        config: QuenchConfig | None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode a symbol stream encoded by :meth:`_encode_symbol_stream`."""
        active_config = self._active_config(config)
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
        if encoding == "packed":
            backend = get_packing_backend(active_config.packing_backend)
            return backend.unpack_bits(
                data,
                bits=int(metadata["bits"]),
                signed=bool(metadata["signed"]),
                shape=shape,
                layout_metadata={"dtype": dtype.str},
            )
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
            decoded = get_entropy_backend(active_config.entropy_backend).decode_symbols(
                data[model_end:encoded_end],
                model,
                count,
            )
            return decoded.astype(dtype, copy=False).reshape(shape)
        raise MalformedPayloadError(f"Unsupported symbol stream encoding: {encoding!r}")

    def _encode_sparse_indices(
        self,
        indices: np.ndarray[Any, np.dtype[Any]],
        *,
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode monotonically increasing sparse indices as delta symbols."""
        working = np.asarray(indices, dtype=np.int64)
        deltas = np.empty_like(working)
        if working.size:
            deltas[0] = working[0]
            deltas[1:] = np.diff(working)
        payload, stream_metadata = self._encode_symbol_stream(deltas, config=config)
        return payload, {
            "count": int(working.size),
            "stream": stream_metadata,
        }

    def _decode_sparse_indices(
        self,
        data: bytes,
        metadata: dict[str, Any],
        *,
        config: QuenchConfig | None,
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Decode sparse indices stored as delta symbols."""
        stream_metadata = self._require_mapping(metadata, "stream")
        deltas = self._decode_symbol_stream(data, stream_metadata, config=config).astype(
            np.int64,
            copy=False,
        )
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

    def _activation_quant_mode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        requested_mode: QuantMode,
    ) -> QuantMode:
        """Prefer asymmetric quantization for non-negative tensors."""
        values = np.asarray(tensor)
        if requested_mode == QuantMode.SYMMETRIC and values.size and float(np.min(values)) >= 0.0:
            return QuantMode.ASYMMETRIC
        return requested_mode


class WeightStrategy(BaseCompressionStrategy):
    """Weight tensors: normalize, quantize, and entropy-code."""

    tensor_type = TensorType.WEIGHT
    strategy_id = 1
    strategy_name = "weight"

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor, config)

        normalized, normalization_metadata = self._normalize_tensor(
            tensor,
            axis=0,
            per_channel=config.per_channel,
        )
        quantized, quant_metadata = self._quantize_tensor(
            normalized,
            bits=config.target_bits,
            mode=config.quant_mode,
            config=config,
            default_axis=0,
        )
        payload, stream_metadata = self._encode_symbol_stream(
            quantized,
            config=config,
            bits=config.target_bits,
        )
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
            return self._decode_lossless(data, metadata, config)

        stream_metadata = self._require_mapping(metadata, "stream")
        quantized = self._decode_symbol_stream(data, stream_metadata, config=config)
        restored = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        return self._denormalize_tensor(restored, self._require_mapping(metadata, "normalization"))


class KVCacheStrategy(BaseCompressionStrategy):
    """KV-cache tensors: delta-code, normalize, quantize, and entropy-code."""

    tensor_type = TensorType.KV_CACHE
    strategy_id = 2
    strategy_name = "kv_cache"

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor, config)

        delta_metadata: dict[str, Any] | None = None
        working = np.asarray(tensor)
        if config.delta_coding and working.ndim > 1:
            delta_axis = self._choose_delta_axis(working)
            working, anchor = self._delta.encode(working, axis=delta_axis)
            delta_metadata = {"anchor": anchor, "axis": delta_axis}

        normalized, normalization_metadata = self._normalize_tensor(
            working,
            axis=max(working.ndim - 1, 0),
            per_channel=config.per_channel,
        )
        bits = min(config.target_bits + 1, 8)
        quantized, quant_metadata = self._quantize_tensor(
            normalized,
            bits=bits,
            mode=config.quant_mode,
            config=config,
            default_axis=max(working.ndim - 1, 0),
        )
        payload, stream_metadata = self._encode_symbol_stream(quantized, config=config, bits=bits)
        return payload, {
            "delta": delta_metadata,
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
            return self._decode_lossless(data, metadata, config)

        quantized = self._decode_symbol_stream(
            data,
            self._require_mapping(metadata, "stream"),
            config=config,
        )
        normalized = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        restored = self._denormalize_tensor(normalized, self._require_mapping(metadata, "normalization"))
        delta_metadata = metadata.get("delta")
        if isinstance(delta_metadata, dict):
            anchor = np.asarray(delta_metadata.get("anchor"))
            return self._delta.decode(restored, anchor, axis=int(delta_metadata["axis"]))
        return restored

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
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor, config)

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
            config=config,
            default_axis=0,
        )
        payload, stream_metadata = self._encode_symbol_stream(
            quantized,
            config=config,
            bits=config.target_bits,
        )
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
            return self._decode_lossless(data, metadata, config)
        if metadata.get("path") == "sparse":
            return self._decode_sparse(data, metadata, config)

        quantized = self._decode_symbol_stream(
            data,
            self._require_mapping(metadata, "stream"),
            config=config,
        )
        normalized = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        return self._denormalize_tensor(normalized, self._require_mapping(metadata, "normalization"))

    def _encode_sparse(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode sparse embeddings by storing indices separately."""
        sparse = self._sparse.encode(tensor, threshold=0.0)
        index_payload, index_metadata = self._encode_sparse_indices(sparse.indices, config=config)

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
                config=config,
                default_axis=0,
                compact_1d=True,
            )
            value_payload, value_metadata = self._encode_symbol_stream(
                quantized_values,
                config=config,
                bits=config.target_bits,
            )

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
        config: QuenchConfig | None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Decode sparse embedding payloads."""
        shape = tuple(int(dim) for dim in metadata.get("shape", []))
        size = int(np.prod(shape, dtype=np.int64))
        chunks = self._unpack_chunks(data, expected_count=2)
        indices = self._decode_sparse_indices(
            chunks[0],
            self._require_mapping(metadata, "index_stream"),
            config=config,
        )

        dense = np.zeros(size, dtype=np.float32)
        if int(metadata.get("nnz", 0)) == 0:
            return dense.reshape(shape)

        quantization_metadata = self._require_mapping(metadata, "quantization")
        normalization_metadata = self._require_mapping(metadata, "normalization")
        value_stream = self._require_mapping(metadata, "value_stream")
        quantized_values = self._decode_symbol_stream(chunks[1], value_stream, config=config)
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
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        if config.codec_mode == CodecMode.LOSSLESS or config.quant_mode == QuantMode.NONE:
            return self._encode_lossless(tensor, config)

        if self._should_use_sparse(tensor, threshold=0.75):
            return self._encode_sparse(tensor, config)

        quant_mode = self._activation_quant_mode(tensor, config.quant_mode)
        bits = min(config.target_bits + 1, 8)
        quantized, quant_metadata = self._quantize_tensor(
            tensor,
            bits=bits,
            mode=quant_mode,
            config=config,
            default_axis=max(np.asarray(tensor).ndim - 1, 0),
        )
        payload, stream_metadata = self._encode_symbol_stream(quantized, config=config, bits=bits)
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
            return self._decode_lossless(data, metadata, config)
        if metadata.get("path") == "sparse":
            return self._decode_sparse(data, metadata, config)

        quantized = self._decode_symbol_stream(
            data,
            self._require_mapping(metadata, "stream"),
            config=config,
        )
        return self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))

    def _encode_sparse(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        """Encode sparse activations by storing non-zero values and indices."""
        sparse = self._sparse.encode(tensor, threshold=0.0)
        index_payload, index_metadata = self._encode_sparse_indices(sparse.indices, config=config)

        if sparse.nnz == 0:
            value_payload = b""
            value_metadata = {"encoding": "empty", "shape": [0], "dtype": np.dtype(np.int8).str}
            quant_metadata: dict[str, Any] | None = None
        else:
            quant_mode = self._activation_quant_mode(sparse.values, config.quant_mode)
            bits = min(config.target_bits + 1, 8)
            quantized_values, quant_metadata = self._quantize_tensor(
                sparse.values,
                bits=bits,
                mode=quant_mode,
                config=config,
                default_axis=0,
                compact_1d=True,
            )
            value_payload, value_metadata = self._encode_symbol_stream(
                quantized_values,
                config=config,
                bits=bits,
            )

        payload = self._pack_chunks([index_payload, value_payload])
        metadata = {
            "index_stream": index_metadata,
            "lossless": False,
            "nnz": sparse.nnz,
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
        config: QuenchConfig | None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Decode sparse activation payloads."""
        shape = tuple(int(dim) for dim in metadata.get("shape", []))
        size = int(np.prod(shape, dtype=np.int64))
        chunks = self._unpack_chunks(data, expected_count=2)
        indices = self._decode_sparse_indices(
            chunks[0],
            self._require_mapping(metadata, "index_stream"),
            config=config,
        )

        dense = np.zeros(size, dtype=np.float32)
        if int(metadata.get("nnz", 0)) == 0:
            return dense.reshape(shape)

        value_stream = self._require_mapping(metadata, "value_stream")
        quantized_values = self._decode_symbol_stream(chunks[1], value_stream, config=config)
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


class OptimizerStateStrategy(BaseCompressionStrategy):
    """Optimizer state tensors favour delta-friendly transforms and robust quantization."""

    tensor_type = TensorType.OPTIMIZER_STATE
    strategy_id = 5
    strategy_name = "optimizer_state"

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        values = np.asarray(tensor)
        if (
            config.codec_mode == CodecMode.LOSSLESS
            or config.quant_mode == QuantMode.NONE
            or np.issubdtype(values.dtype, np.integer)
        ):
            return self._encode_lossless(values, config)

        delta_metadata: dict[str, Any] | None = None
        working = values.astype(np.float32, copy=False)
        if config.delta_coding and working.ndim > 1:
            axis = self._choose_delta_axis(working)
            working, anchor = self._delta.encode(working, axis=axis)
            delta_metadata = {"anchor": anchor, "axis": axis}

        normalized, normalization_metadata = self._normalize_tensor(
            working,
            axis=0,
            per_channel=config.per_channel and working.ndim > 1,
        )
        quant_mode = self._activation_quant_mode(working, config.quant_mode)
        quantized, quant_metadata = self._quantize_tensor(
            normalized,
            bits=config.target_bits,
            mode=quant_mode,
            config=config,
            default_axis=0,
            compact_1d=working.ndim <= 1,
        )
        payload, stream_metadata = self._encode_symbol_stream(
            quantized,
            config=config,
            bits=config.target_bits,
        )
        return payload, {
            "delta": delta_metadata,
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
            return self._decode_lossless(data, metadata, config)

        quantized = self._decode_symbol_stream(
            data,
            self._require_mapping(metadata, "stream"),
            config=config,
        )
        normalized = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        restored = self._denormalize_tensor(normalized, self._require_mapping(metadata, "normalization"))
        delta_metadata = metadata.get("delta")
        if isinstance(delta_metadata, dict):
            anchor = np.asarray(delta_metadata.get("anchor"))
            return self._delta.decode(restored, anchor, axis=int(delta_metadata["axis"]))
        return restored

    def _choose_delta_axis(self, tensor: np.ndarray[Any, np.dtype[Any]]) -> int:
        if tensor.ndim <= 1:
            return 0
        return max(range(tensor.ndim), key=lambda axis: int(tensor.shape[axis]))


class BiasStrategy(BaseCompressionStrategy):
    """Bias tensors use a compact direct quantization path to limit metadata overhead."""

    tensor_type = TensorType.BIAS
    strategy_id = 6
    strategy_name = "bias"

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        values = np.asarray(tensor)
        if (
            config.codec_mode == CodecMode.LOSSLESS
            or config.quant_mode == QuantMode.NONE
            or np.issubdtype(values.dtype, np.integer)
        ):
            return self._encode_lossless(values, config)

        quantized, quant_metadata = self._quantize_tensor(
            values.astype(np.float32, copy=False),
            bits=config.target_bits,
            mode=self._activation_quant_mode(values, config.quant_mode),
            config=config,
            default_axis=0,
            compact_1d=True,
        )
        payload, stream_metadata = self._encode_symbol_stream(
            quantized,
            config=config,
            bits=config.target_bits,
        )
        return payload, {
            "lossless": False,
            "path": "compact",
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
            return self._decode_lossless(data, metadata, config)

        quantized = self._decode_symbol_stream(
            data,
            self._require_mapping(metadata, "stream"),
            config=config,
        )
        return self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))


class MixedPrecisionStrategy(BaseCompressionStrategy):
    """Mixed-precision tensors preserve dtype semantics while using safe quantization paths."""

    tensor_type = TensorType.MIXED_PRECISION
    strategy_id = 7
    strategy_name = "mixed_precision"

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        values = np.asarray(tensor)
        if (
            config.codec_mode == CodecMode.LOSSLESS
            or config.quant_mode == QuantMode.NONE
            or np.issubdtype(values.dtype, np.integer)
        ):
            return self._encode_lossless(values, config)

        working = values.astype(np.float32, copy=False)
        quant_mode = self._activation_quant_mode(working, config.quant_mode)
        quantized, quant_metadata = self._quantize_tensor(
            working,
            bits=config.target_bits,
            mode=quant_mode,
            config=config,
            default_axis=max(working.ndim - 1, 0),
            compact_1d=working.ndim <= 1,
        )
        payload, stream_metadata = self._encode_symbol_stream(
            quantized,
            config=config,
            bits=config.target_bits,
        )
        return payload, {
            "lossless": False,
            "original_dtype": values.dtype.str,
            "path": "mixed_precision",
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
            return self._decode_lossless(data, metadata, config)

        quantized = self._decode_symbol_stream(
            data,
            self._require_mapping(metadata, "stream"),
            config=config,
        )
        restored = self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))
        dtype_str = str(metadata.get("original_dtype", restored.dtype.str))
        return restored.astype(np.dtype(dtype_str), copy=False)


class DefaultStrategy(BaseCompressionStrategy):
    """Fallback strategy that quantizes directly and entropy-codes the result."""

    tensor_type = TensorType.UNKNOWN
    strategy_id = 255
    strategy_name = "default"

    def encode(
        self,
        tensor: np.ndarray[Any, np.dtype[Any]],
        config: QuenchConfig,
    ) -> tuple[bytes, dict[str, Any]]:
        values = np.asarray(tensor)
        if (
            config.codec_mode == CodecMode.LOSSLESS
            or config.quant_mode == QuantMode.NONE
            or np.issubdtype(values.dtype, np.integer)
        ):
            return self._encode_lossless(values, config)

        quantized, quant_metadata = self._quantize_tensor(
            values.astype(np.float32, copy=False),
            bits=config.target_bits,
            mode=self._activation_quant_mode(values, config.quant_mode),
            config=config,
            default_axis=max(values.ndim - 1, 0),
            compact_1d=values.ndim <= 1,
        )
        payload, stream_metadata = self._encode_symbol_stream(
            quantized,
            config=config,
            bits=config.target_bits,
        )
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
            return self._decode_lossless(data, metadata, config)

        quantized = self._decode_symbol_stream(
            data,
            self._require_mapping(metadata, "stream"),
            config=config,
        )
        return self._dequantize_tensor(quantized, self._require_mapping(metadata, "quantization"))


WEIGHT_STRATEGY = WeightStrategy()
KV_CACHE_STRATEGY = KVCacheStrategy()
EMBEDDING_STRATEGY = EmbeddingStrategy()
ACTIVATION_STRATEGY = ActivationStrategy()
OPTIMIZER_STATE_STRATEGY = OptimizerStateStrategy()
BIAS_STRATEGY = BiasStrategy()
MIXED_PRECISION_STRATEGY = MixedPrecisionStrategy()
DEFAULT_STRATEGY = DefaultStrategy()

STRATEGY_REGISTRY: dict[TensorType, CompressionStrategy] = {
    TensorType.WEIGHT: WEIGHT_STRATEGY,
    TensorType.KV_CACHE: KV_CACHE_STRATEGY,
    TensorType.EMBEDDING: EMBEDDING_STRATEGY,
    TensorType.ACTIVATION: ACTIVATION_STRATEGY,
    TensorType.OPTIMIZER_STATE: OPTIMIZER_STATE_STRATEGY,
    TensorType.BIAS: BIAS_STRATEGY,
    TensorType.MIXED_PRECISION: MIXED_PRECISION_STRATEGY,
}

STRATEGY_ID_REGISTRY: dict[int, CompressionStrategy] = {
    WEIGHT_STRATEGY.strategy_id: WEIGHT_STRATEGY,
    KV_CACHE_STRATEGY.strategy_id: KV_CACHE_STRATEGY,
    EMBEDDING_STRATEGY.strategy_id: EMBEDDING_STRATEGY,
    ACTIVATION_STRATEGY.strategy_id: ACTIVATION_STRATEGY,
    OPTIMIZER_STATE_STRATEGY.strategy_id: OPTIMIZER_STATE_STRATEGY,
    BIAS_STRATEGY.strategy_id: BIAS_STRATEGY,
    MIXED_PRECISION_STRATEGY.strategy_id: MIXED_PRECISION_STRATEGY,
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
