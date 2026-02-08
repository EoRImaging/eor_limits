"""A module defining the data types used in eor-limits."""

from pathlib import Path
from typing import Any, Self

import attrs
import cattrs
import numpy as np
import pandas as pd
import yaml
from attrs.converters import optional as optconv

converter = cattrs.Converter(use_alias=True)


@converter.register_structure_hook
def _make_array(x: Any, _) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _normalize_dataset_name(path: str | Path) -> Path:
    from . import DATA_PATH, KNOWN_PAPERS

    if isinstance(path, str) and not path.endswith(".yaml"):
        path = DATA_PATH / (path + ".yaml")
    elif isinstance(path, str):
        path = Path(path)

    if not path.exists():
        path = DATA_PATH / path

    if not path.exists():
        raise ValueError(
            f"Dataset file '{path.name}' not found. "
            f"Available datasets: {KNOWN_PAPERS.keys()}"
        )

    return path


def _floatarray(x):
    return np.array(x, dtype=float)


def _tuple_of_floatarrays(x):
    return tuple(np.array(arr, dtype=float) for arr in x)


@attrs.define(frozen=True, kw_only=True)
class Data:
    """A class representing the data for a single dataset."""

    z: np.ndarray = attrs.field(converter=_floatarray)
    z_lower: np.ndarray | None = attrs.field(
        default=None, converter=optconv(_floatarray)
    )
    z_upper: np.ndarray | None = attrs.field(
        default=None, converter=optconv(_floatarray)
    )
    z_tags: tuple[str, ...] | None = attrs.field(default=None)

    k: tuple[np.ndarray, ...] = attrs.field(converter=_tuple_of_floatarrays)
    k_lower: tuple[np.ndarray, ...] | None = attrs.field(
        default=None, converter=optconv(_tuple_of_floatarrays)
    )
    k_upper: tuple[np.ndarray, ...] | None = attrs.field(
        default=None, converter=optconv(_tuple_of_floatarrays)
    )

    _delta: tuple[np.ndarray, ...] | None = attrs.field(default=None)
    _delta_squared: tuple[np.ndarray, ...] | None = attrs.field(default=None)

    @z.validator
    def _check_z(self, attribute, value):
        if len(value) == 0:
            raise ValueError("z cannot be empty.")
        if not (value > 0).all():
            raise ValueError("All z values must be positive.")

    @z_lower.validator
    @z_upper.validator
    def _check_z_bounds(self, attribute, value):
        if value is not None:
            if not (value > 0).all():
                raise ValueError(f"All {attribute.name} values must be positive.")
            if not value.shape == self.z.shape:
                raise ValueError(f"{attribute.name} must be the same shape as z.")

    @z_lower.validator
    def _check_z_lower(self, attribute, value):
        if value is not None and (value > self.z).any():
            raise ValueError("All z_lower values must be less than or equal to z.")

    @z_upper.validator
    def _check_z_upper(self, attribute, value):
        if value is not None and (value < self.z).any():
            raise ValueError("All z_upper values must be greater than or equal to z.")

    @z_tags.validator
    def _check_z_tags(self, attribute, value):
        if value is not None:
            if not all(isinstance(x, str) for x in value):
                raise ValueError("z_tags must be a 1D array of strings.")
            if len(value) != self.z.shape[0]:
                raise ValueError("z_tags must be the same shape as z.")

    @k.validator
    def _check_k(self, attribute, value):
        if len(value) != len(self.z):
            raise ValueError("k must have the same number of entries as z.")
        for arr in value:
            if not (arr > 0).all():
                raise ValueError("All k values must be positive.")

    @k_upper.validator
    @k_lower.validator
    @_delta.validator
    @_delta_squared.validator
    def _check_k_bounds(self, attribute, value):
        if value is not None:
            if len(value) != len(self.z):
                raise ValueError(
                    f"{attribute.name} must have the same number of entries as z."
                )
            for kk, arr in zip(self.k, value, strict=True):
                if not arr.shape == kk.shape:
                    raise ValueError(f"{attribute.name} must have the same shape as k.")

                if arr.dtype != float:
                    raise ValueError(
                        f"{attribute.name} must be an array of floats. "
                        f"Got {arr.dtype} instead -- check the YAML formatting."
                    )
                if not (arr[np.isfinite(arr)] >= 0).all():
                    raise ValueError(f"All {attribute.name} values must be positive.")

    @k_upper.validator
    def _check_k_upper(self, attribute, value):
        if value is not None:
            for kk, arr in zip(self.k, value, strict=True):
                if (arr < kk).any():
                    raise ValueError(
                        "All k_upper values must be greater than or equal to k."
                    )

    @k_lower.validator
    def _check_k_lower(self, attribute, value):
        if value is not None:
            for kk, arr in zip(self.k, value, strict=True):
                if (arr > kk).any():
                    raise ValueError(
                        "All k_lower values must be less than or equal to k."
                    )

    @_delta_squared.validator
    def _check_delta_squared(self, attribute, value):
        if value is None and self._delta is None:
            raise ValueError("Either delta_squared or delta must be provided.")
        if value is not None and self._delta is not None:
            raise ValueError("Only one of delta_squared or delta can be provided.")

    @property
    def delta_squared(self) -> tuple[np.ndarray, ...]:
        """The value of the power spectrum upper limit, in mK^2."""
        if self._delta_squared is not None:
            return tuple(np.asarray(arr) for arr in self._delta_squared)
        if self._delta is not None:
            return tuple(np.asarray(arr) ** 2 for arr in self._delta)
        raise ValueError("Either delta_squared or delta must be provided.")

    def as_pandas_df(self) -> pd.DataFrame:
        """Convert the Data object to a pandas DataFrame."""
        # Create DataFrame row by row for each z value
        rows = []
        for i in range(len(self.z)):
            row = {
                "z": self.z[i],
                "z_lower": self.z_lower[i] if self.z_lower is not None else np.nan,
                "z_upper": self.z_upper[i] if self.z_upper is not None else np.nan,
                "z_tags": self.z_tags[i] if self.z_tags is not None else "",
                "k": np.array(self.k[i]),
                "k_lower": np.array(self.k_lower[i])
                if self.k_lower is not None
                else np.nan,
                "k_upper": np.array(self.k_upper[i])
                if self.k_upper is not None
                else np.nan,
                "delta_squared": np.array(self.delta_squared[i]),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def drop_nan(self) -> Self:
        """Return a new Data object with any rows containing NaN values removed."""

        def mask(dsq):
            return np.isfinite(dsq)

        new_k = [
            kk[mask(dsq)] for kk, dsq in zip(self.k, self.delta_squared, strict=True)
        ]
        new_delta_squared = [dsq[mask(dsq)] for dsq in self.delta_squared]
        if self.k_lower is not None:
            new_k_lower = [
                kl[mask(dsq)]
                for kl, dsq in zip(self.k_lower, self.delta_squared, strict=True)
            ]
        else:
            new_k_lower = None
        if self.k_upper is not None:
            new_k_upper = [
                ku[mask(dsq)]
                for ku, dsq in zip(self.k_upper, self.delta_squared, strict=True)
            ]
        else:
            new_k_upper = None

        return attrs.evolve(
            self,
            k=tuple(new_k),
            k_lower=tuple(new_k_lower) if new_k_lower is not None else None,
            k_upper=tuple(new_k_upper) if new_k_upper is not None else None,
            delta_squared=tuple(new_delta_squared),
            delta=None,
        )


@attrs.define(frozen=True)
class DataSet:
    """A class representing a dataset, including metadata and the data itself.

    The data is stored in a Data object, which contains arrays of z, k and
    delta_squared values.

    The metadata includes the telescope, author, year and doi of the dataset, as well
    as any additional notes.
    """

    telescope: str = attrs.field(converter=str)
    author: str = attrs.field(converter=str)
    year: int = attrs.field(converter=int)
    doi: str = attrs.field(converter=str)
    data: Data = attrs.field(validator=attrs.validators.instance_of(Data))

    notes: tuple[str, ...] = attrs.field(
        default=(), validator=attrs.validators.instance_of(tuple)
    )

    def __str__(self) -> str:
        """Return a string representation of the DataSet, including metadata."""
        text = f"DataSet: telescope={self.telescope}, author={self.author}, "
        text += f"year={self.year}, doi={self.doi}"
        if self.notes:
            text += ",\nnotes=["
            text += ",\n       ".join(self.notes)
            text += "]"
        text += ",\ndata=\n"
        text += str(self.data)
        text += "\n"
        return text

    @classmethod
    def load(cls, path: str | Path, /) -> Self:
        """Load a DataSet from a YAML file or a known dataset name."""
        path = _normalize_dataset_name(path)

        with path.open("r") as fl:
            yaml_data = yaml.safe_load(fl)

        return converter.structure(yaml_data, cls)

    def select_z(self, z_min: float, z_max: float) -> Self:
        """Return a new DataSet with only data in the specified redshift range."""
        mask = (self.data.z >= z_min) & (self.data.z <= z_max)
        if not mask.any():
            raise ValueError(
                f"No data points found in the redshift range {z_min} to {z_max}."
            )

        new_data = Data(
            z=self.data.z[mask],
            z_lower=self.data.z_lower[mask] if self.data.z_lower is not None else None,
            z_upper=self.data.z_upper[mask] if self.data.z_upper is not None else None,
            z_tags=tuple(
                self.data.z_tags[i] for i in range(len(self.data.z)) if mask[i]
            )
            if self.data.z_tags is not None
            else None,
            k=tuple(self.data.k[i] for i in range(len(self.data.z)) if mask[i]),
            k_lower=tuple(
                self.data.k_lower[i] for i in range(len(self.data.z)) if mask[i]
            )
            if self.data.k_lower is not None
            else None,
            k_upper=tuple(
                self.data.k_upper[i] for i in range(len(self.data.z)) if mask[i]
            )
            if self.data.k_upper is not None
            else None,
            delta_squared=tuple(
                self.data.delta_squared[i] for i in range(len(self.data.z)) if mask[i]
            ),
        )
        return attrs.evolve(self, data=new_data)

    def _flush_empty_redshifts(
        self,
        zmask: np.ndarray,
        new_k: list[np.ndarray],
        new_k_lower: list[np.ndarray] | None,
        new_k_upper: list[np.ndarray] | None,
        new_dsq: list[np.ndarray],
        delta_sq_min: float,
        delta_sq_max: float,
    ) -> Self:

        new_k = [kk for kk, zm in zip(new_k, zmask, strict=True) if zm]
        new_k_lower = (
            [kl for kl, zm in zip(new_k_lower, zmask, strict=True) if zm]
            if new_k_lower is not None
            else None
        )
        new_k_upper = (
            [ku for ku, zm in zip(new_k_upper, zmask, strict=True) if zm]
            if new_k_upper is not None
            else None
        )
        new_dsq = [dsq for dsq, zm in zip(new_dsq, zmask, strict=True) if zm]
        new_z = self.data.z[zmask]
        new_z_lower = (
            self.data.z_lower[zmask] if self.data.z_lower is not None else None
        )
        new_z_upper = (
            self.data.z_upper[zmask] if self.data.z_upper is not None else None
        )
        new_z_tags = (
            tuple(ztag for ztag, zm in zip(self.data.z_tags, zmask, strict=True) if zm)
            if self.data.z_tags is not None
            else None
        )

        new_data = attrs.evolve(
            self.data,
            k=tuple(new_k),
            k_lower=tuple(new_k_lower) if new_k_lower is not None else None,
            k_upper=tuple(new_k_upper) if new_k_upper is not None else None,
            delta_squared=tuple(new_dsq),
            z=new_z,
            z_lower=new_z_lower,
            z_upper=new_z_upper,
            z_tags=new_z_tags,
        )

        return attrs.evolve(self, data=new_data)

    def _select_with_k_based_mask(self, mask: callable, field: str = "k") -> Self:
        fld = getattr(self.data, field)
        new_k = [kk[mask(q)] for kk, q in zip(self.data.k, fld, strict=True)]

        new_dsq = [
            dsq[mask(q)] for dsq, q in zip(self.data.delta_squared, fld, strict=True)
        ]
        if self.data.k_lower is not None:
            new_k_lower = [
                kl[mask(q)] for kl, q in zip(self.data.k_lower, fld, strict=True)
            ]
        else:
            new_k_lower = None
        if self.data.k_upper is not None:
            new_k_upper = [
                ku[mask(q)] for ku, q in zip(self.data.k_upper, fld, strict=True)
            ]
        else:
            new_k_upper = None

        zmask = np.array([len(kk) > 0 for kk in new_k])
        if not any(zmask):
            raise ValueError("No data points found with this mask")

        return self._flush_empty_redshifts(
            zmask, new_k, new_k_lower, new_k_upper, new_dsq, None, None
        )

    def select_k(self, k_min: float, k_max: float) -> Self:
        """Return a new DataSet with only data in the specified k range."""

        def mask(kk):
            return (kk >= k_min) & (kk <= k_max)

        try:
            return self._select_with_k_based_mask(mask, "k")
        except ValueError as err:
            raise ValueError(
                f"No data points found in the k range {k_min} to {k_max}."
            ) from err

    def select_delta_sq(self, delta_sq_min: float, delta_sq_max: float) -> Self:
        """Return a new DataSet with only data in the specified delta_squared range."""

        def mask(dsq):
            return (dsq >= delta_sq_min) & (dsq <= delta_sq_max)

        try:
            return self._select_with_k_based_mask(mask, "delta_squared")
        except ValueError as err:
            raise ValueError(
                "No data points found in the delta_squared range "
                f"{delta_sq_min} to {delta_sq_max}."
            ) from err

    def select_closest_z(self, z_target: float) -> Self:
        """Return a new DataSet with only the data point closest to the target z."""
        idx = np.argmin(np.abs(self.data.z - z_target))
        new_data = Data(
            z=np.array([self.data.z[idx]]),
            z_lower=np.array([self.data.z_lower[idx]])
            if self.data.z_lower is not None
            else None,
            z_upper=np.array([self.data.z_upper[idx]])
            if self.data.z_upper is not None
            else None,
            z_tags=(self.data.z_tags[idx],) if self.data.z_tags is not None else None,
            k=(self.data.k[idx],),
            k_lower=(self.data.k_lower[idx],)
            if self.data.k_lower is not None
            else None,
            k_upper=(self.data.k_upper[idx],)
            if self.data.k_upper is not None
            else None,
            delta_squared=(self.data.delta_squared[idx],),
        )
        return attrs.evolve(self, data=new_data)

    @property
    def key(self) -> str:
        """Return a unique key for this dataset based on its metadata."""
        return f"{self.author}{self.year}"

    def drop_nan(self) -> Self:
        """Return a new DataSet with any rows containing NaN values removed."""
        new_data = self.data.drop_nan()
        return attrs.evolve(self, data=new_data)
