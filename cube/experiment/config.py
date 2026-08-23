"""CUBE configuration used in the COOP² paper."""

from __future__ import annotations


PAPER_BLOCK_SPECS = {1: 3, 2: 3, 3: 3}


def paper_block_specs() -> dict[int, int]:
    """Return three blocks at each cooperative weight."""
    return dict(PAPER_BLOCK_SPECS)
