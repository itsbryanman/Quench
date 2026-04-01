"""Delta compression for fine-tuned model checkpoints.

Compress a fine-tune as a compact diff against its base model::

    import quench.delta

    quench.delta.compress(
        base="meta-llama/Llama-3.1-8B",
        finetune="my-org/my-finetune",
        output="finetune-delta.qnc",
    )

    state_dict = quench.delta.load(
        base="meta-llama/Llama-3.1-8B",
        delta="finetune-delta.qnc",
    )
"""
from quench.delta.engine import compress, inspect, load

__all__ = ["compress", "inspect", "load"]
