"""Processor for Muñoz 2018 Mini-Charged DM model with f_DM=0.3."""

import numpy as np
import yaml

from eor_limits.data import THEORY_PATH
from eor_limits.datatypes import Data

from ._base import BaseTheoryProcessor


class Munoz2018FDM3(BaseTheoryProcessor):
    """Processor for Muñoz 2018 Mini-Charged DM model with f_DM=0.3."""

    simulator = "(F_DM=0.3)"
    author = "Muñoz"
    year = 2018
    doi = "10.1038/s41586-018-0151-x"

    @classmethod
    def _load_data(cls) -> Data:
        with (THEORY_PATH / "munoz_2018_fdm3.yaml").open("r") as f:
            yaml_data = yaml.safe_load(f)

        k = yaml_data["k"]
        z = yaml_data["redshift"]
        delta_squared = yaml_data["delta_squared"]

        return Data(
            z=z,
            k=tuple(np.asarray(kk) for kk in k),
            delta_squared=tuple(np.asarray(ds) for ds in delta_squared),
        )
