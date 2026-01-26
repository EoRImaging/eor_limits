"""Init file for data directory."""

from pathlib import Path

DATA_PATH = Path(__file__).parent.resolve()
THEORY_PATH = DATA_PATH / "theory"

KNOWN_PAPERS = {p.stem: p for p in DATA_PATH.glob("*.yaml") if p.is_file()}

KNOWN_THEORIES = {p.stem: p for p in THEORY_PATH.glob("*.yaml") if p.is_file()}
