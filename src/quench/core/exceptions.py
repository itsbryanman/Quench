"""Quench exception hierarchy."""


class QuenchError(Exception):
    """Base exception for all Quench errors."""


class CodecError(QuenchError):
    """Error during compression or decompression."""


class UnsupportedStrategyError(CodecError):
    """Error raised when a tensor strategy cannot be resolved or applied."""


class MalformedPayloadError(CodecError):
    """Error raised when a compressed payload cannot be decoded safely."""


class MetadataError(CodecError):
    """Error raised when codec metadata cannot be serialized or deserialized."""


class ChecksumMismatchError(CodecError):
    """Error raised when a lossless decode fails checksum verification."""


class HeaderError(QuenchError):
    """Error parsing or writing a QNC header."""


class QuantizationError(QuenchError):
    """Error during quantization or dequantization."""


class EntropyError(QuenchError):
    """Error in entropy coding (rANS encode/decode)."""


class BackendError(QuenchError):
    """Error raised while resolving or invoking a codec backend."""


class UnsupportedBackendError(BackendError):
    """Error raised when a requested backend name is not registered."""
