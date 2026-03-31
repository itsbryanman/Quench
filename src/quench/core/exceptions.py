"""Quench exception hierarchy."""


class QuenchError(Exception):
    """Base exception for all Quench errors."""


class CodecError(QuenchError):
    """Error during compression or decompression."""


class HeaderError(QuenchError):
    """Error parsing or writing a QNC header."""


class QuantizationError(QuenchError):
    """Error during quantization or dequantization."""


class EntropyError(QuenchError):
    """Error in entropy coding (rANS encode/decode)."""
