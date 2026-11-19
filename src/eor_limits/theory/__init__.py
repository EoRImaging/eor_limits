"""Init file for theory directory."""

from pathlib import Path

THEORY_PATH = Path(__file__).parent.resolve()
KNOWN_THEORIES = {p.stem: p for p in sorted(THEORY_PATH.glob("*.yaml")) if p.is_file()}

__all__ = [
    "KNOWN_THEORIES",
    "THEORY_PATH",
]
