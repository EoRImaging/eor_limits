"""Module for defining constants related to data handling."""

from pathlib import Path

DATA_PATH = Path(__file__).parent.resolve()
KNOWN_PAPERS = {p.stem: p for p in DATA_PATH.glob("*.yaml") if p.is_file()}

THEORY_PATH = DATA_PATH / "theory"
