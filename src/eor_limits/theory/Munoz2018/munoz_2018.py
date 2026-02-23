"""Processor for Muñoz 2018 Mini-Charged DM model with f_DM=0.3."""

import numpy as np
import yaml

from eor_limits._datatypes import Data
from eor_limits.theory._base import THEORY_PATH, BaseTheoryProcessor

munoz_fdm3_path = THEORY_PATH / "Munoz2018" / "munoz_2018_fdm3.yaml"


class Munoz2018FDM3(BaseTheoryProcessor):
    """Processor for Muñoz 2018 Mini-Charged DM model with f_DM=0.3."""

    simulator = "f_{DM}=0.3"
    author = "Muñoz"
    year = 2018
    doi = "10.1038/s41586-018-0151-x"
    _datapath = munoz_fdm3_path

    @classmethod
    def _load_data(cls) -> Data:
        with (cls._datapath).open("r") as f:
            yaml_data = yaml.safe_load(f)

        k = yaml_data["k"]
        z = yaml_data["redshift"]
        delta_squared = yaml_data["delta_squared"]

        return Data(
            z=z,
            k=tuple(np.asarray(kk) for kk in k),
            delta_squared=tuple(np.asarray(ds) for ds in delta_squared),
        )
