"""Write the theory models in this folder out as YAML files in ``eor_limits/theory/``.

Each processor in this sub-package reads one model's original data files. This
script loads every model through its processor, writes it to
``eor_limits/theory/<key>.yaml`` in the same layout as the limits in
``eor_limits/data/``, and checks that the file loads back to identical data.

Run with ``python -m eor_limits._old_theory.write_yaml``.
"""

from pathlib import Path

import numpy as np
import yaml

from eor_limits._datatypes import DataSet
from eor_limits._old_theory import __all_theories__
from eor_limits.theory import THEORY_PATH as OUTPUT_PATH


def _flow(values: list[float]) -> str:
    """Format a list of floats as a single-line YAML flow sequence."""
    return yaml.safe_dump(values, default_flow_style=True, width=float("inf")).strip()


def dataset_to_yaml(dataset: DataSet) -> str:
    """Serialise a theory DataSet to YAML text, matching the limits file layout."""
    metadata = {
        "telescope": dataset.telescope,
        "author": dataset.author,
        "year": dataset.year,
        "doi": dataset.doi,
        "key": dataset.key,
    }
    lines = [
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        "data:",
        f"    z: {_flow(dataset.data.z.tolist())}",
    ]
    for name in ("k", "delta_squared"):
        rows = [_flow(arr.tolist()) for arr in getattr(dataset.data, name)]
        lines.extend((f"    {name}:", "        [" + ",\n         ".join(rows) + "]"))
    return "\n".join(lines) + "\n"


def _check_round_trip(original: DataSet, path: Path) -> None:
    """Raise if the YAML file at ``path`` does not reproduce ``original`` exactly."""
    loaded = DataSet.load(path)
    for attr in ("telescope", "author", "year", "doi", "key", "notes"):
        if getattr(loaded, attr) != getattr(original, attr):
            raise ValueError(f"{path.name}: '{attr}' changed on round trip.")
    for attr in ("z_lower", "z_upper", "z_tags", "k_lower", "k_upper"):
        if getattr(original.data, attr) is not None:
            raise ValueError(f"{path.name}: '{attr}' is not written to YAML.")
    np.testing.assert_array_equal(loaded.data.z, original.data.z)
    for attr in ("k", "delta_squared"):
        for new, old in zip(
            getattr(loaded.data, attr), getattr(original.data, attr), strict=True
        ):
            np.testing.assert_array_equal(new, old)


def main() -> None:
    """Write and verify a YAML file for every theory model."""
    for name, processor in __all_theories__.items():
        dataset = processor.load_as_dataset()
        path = OUTPUT_PATH / f"{name}.yaml"
        path.write_text(dataset_to_yaml(dataset), encoding="utf-8")
        _check_round_trip(dataset, path)


if __name__ == "__main__":
    main()
