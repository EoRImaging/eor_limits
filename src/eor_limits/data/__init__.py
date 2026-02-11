"""Init file for data directory."""
from pathlib import Path

DATA_PATH = Path(__file__).parent.resolve()
KNOWN_LIMITS = {p.stem: p for p in DATA_PATH.glob("*.yaml") if p.is_file()}