"""CLI entry point for quench-compress."""
from __future__ import annotations

import sys

import structlog


def main() -> None:
    """Entry point for the ``quench-compress`` command."""
    logger = structlog.get_logger(__name__)
    logger.info(
        "cli_not_implemented",
        command="quench-compress",
        detail="CLI pipeline not implemented yet. See quench.entropy for direct rANS coding.",
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
