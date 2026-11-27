"""Init file for the sensitivities directory."""

from pathlib import Path

SENSITIVITY_PATH = Path(__file__).parent.resolve()
KNOWN_SENSITIVITIES = {
    p.stem: p for p in SENSITIVITY_PATH.glob("*.yaml") if p.is_file()
}

__all__ = [
    "KNOWN_SENSITIVITIES",
    "SENSITIVITY_PATH",
]
