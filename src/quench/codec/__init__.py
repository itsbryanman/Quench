"""Public codec objects and helpers."""
from quench.codec.auto import auto_compress, auto_decompress
from quench.codec.decoder import QuenchDecoder
from quench.codec.encoder import QuenchEncoder
from quench.codec.metadata import deserialize_metadata, serialize_metadata
from quench.codec.strategies import (
    ACTIVATION_STRATEGY,
    DEFAULT_STRATEGY,
    EMBEDDING_STRATEGY,
    KV_CACHE_STRATEGY,
    STRATEGY_ID_REGISTRY,
    STRATEGY_REGISTRY,
    WEIGHT_STRATEGY,
    ActivationStrategy,
    CompressionStrategy,
    DefaultStrategy,
    EmbeddingStrategy,
    KVCacheStrategy,
    WeightStrategy,
    get_strategy,
    get_strategy_by_id,
)

__all__ = [
    "ACTIVATION_STRATEGY",
    "DEFAULT_STRATEGY",
    "EMBEDDING_STRATEGY",
    "KV_CACHE_STRATEGY",
    "STRATEGY_ID_REGISTRY",
    "STRATEGY_REGISTRY",
    "WEIGHT_STRATEGY",
    "ActivationStrategy",
    "CompressionStrategy",
    "DefaultStrategy",
    "EmbeddingStrategy",
    "KVCacheStrategy",
    "QuenchDecoder",
    "QuenchEncoder",
    "WeightStrategy",
    "auto_compress",
    "auto_decompress",
    "deserialize_metadata",
    "get_strategy",
    "get_strategy_by_id",
    "serialize_metadata",
]
