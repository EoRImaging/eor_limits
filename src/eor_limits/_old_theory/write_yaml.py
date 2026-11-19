"""Write the theory models in this folder out as YAML files in ``eor_limits/theory/``.

Each processor in this sub-package reads one model's original data files. This
script loads every model through its processor, writes it to
``eor_limits/theory/<key>.yaml`` in the same layout as the limits in
``eor_limits/data/``, and checks that the file loads back to identical data.

Run with ``python -m eor_limits._old_theory.write_yaml``.
"""

from pathlib import Path

import attrs
import numpy as np
import yaml

from eor_limits._datatypes import DataSet
from eor_limits._old_theory import __all_theories__
from eor_limits.theory import THEORY_PATH as OUTPUT_PATH

# A note on the original data file(s) of each model, written into its YAML file.
# Keys are model names, or a prefix shared by several models (all the Pagano and
# Liu models come from one file).
SOURCE_NOTES = {
    "Mesinger2016Faint": (
        "Converted to YAML from the original files named "
        "ps_no_halos_z<z>_nf<nf>_useTs1_NX2.0e-01_alphaX1.5_TvirminX2.0e+04_"
        "aveTb<aveTb>_Pop2_1024_1600Mpc_v3, one per redshift."
    ),
    "Mesinger2016Bright": (
        "Converted to YAML from the original files named "
        "ps_no_halos_z<z>_nf<nf>_useTs1_NX2.0e-01_alphaX1.5_TvirminX2.0e+05_"
        "aveTb<aveTb>_Pop2_1024_1600Mpc_v3, one per redshift."
    ),
    "Munoz2018FDM3": "Converted to YAML from the original file munoz_2018_fdm3.yaml.",
    "Munoz2022AllGalaxies": (
        "Converted to YAML from the original files 1pt5Gpc_EOS_coeval_pow_kbins.bin, "
        "1pt5Gpc_EOS_coeval_pow_zlist.bin and 1pt5Gpc_EOS_coeval_pow_P21.bin."
    ),
    "Munoz2022Optimistic": (
        "Converted to YAML from the original files 600Mpc_pt0_coeval_pow_kbins.bin, "
        "600Mpc_pt0_coeval_pow_zlist.bin and 600Mpc_pt0_coeval_pow_P21.bin."
    ),
    "PaganoLiu2020": "Converted to YAML from the original file pagano_liu_2020.npz.",
}


def _source_note(name: str) -> str:
    """Return the note on the original data file(s) of the model ``name``."""
    return next(note for key, note in SOURCE_NOTES.items() if name.startswith(key))


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
        "notes": list(dataset.notes),
    }
    lines = [
        yaml.safe_dump(
            metadata, sort_keys=False, allow_unicode=True, width=float("inf")
        ),
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
        dataset = attrs.evolve(processor.load_as_dataset(), notes=(_source_note(name),))
        path = OUTPUT_PATH / f"{name}.yaml"
        path.write_text(dataset_to_yaml(dataset), encoding="utf-8")
        _check_round_trip(dataset, path)


if __name__ == "__main__":
    main()
