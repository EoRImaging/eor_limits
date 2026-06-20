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


def _floatarray(x):
    return np.array(x, dtype=float)


def _tuple_of_floatarrays(x):
    return tuple(np.array(arr, dtype=float) for arr in x)


@attrs.define(frozen=True, kw_only=True)
class Data:
    r"""
    Class representing the data for a single dataset.

    The data consists of arrays of redshift values |z|, |k| values
    for those redshifts, and the corresponding power spectrum upper limits
    |dsq|, along with optional lower and upper bounds
    on the |z| and |k| values.

    Attributes
    ----------
    z : np.ndarray
        A 1D array of redshift values |z| for the dataset.
    z_lower : np.ndarray | None
        A 1D array of lower bounds on the |z|-bins,
        or ``None`` if not available.
    z_upper : np.ndarray | None
        A 1D array of upper bounds on the |z|-bins,
        or ``None`` if not available.
    z_tags : tuple[str, ...] | None
        A 1D array of tags for the |z| values (e.g. ``'Pol E-W'``, ``'Field 1'``),
        or ``None`` if not available. Thus, different polarizations or fields
        are treated as separate redshift entries with the same k value
        but different tags.
    k : tuple[np.ndarray, ...]
        A tuple of 1D arrays of scale values |k|, one for each redshift value,
        in units of |h/Mpc|.
    k_lower : tuple[np.ndarray, ...] | None
        A tuple of 1D arrays of lower bounds on the |k|-bins, one for each
        redshift value, in units of |h/Mpc|, or ``None`` if not available.
    k_upper : tuple[np.ndarray, ...] | None
        A tuple of 1D arrays of upper bounds on the |k|-bins, one for each
        redshift value, in units of |h/Mpc|, or ``None`` if not available.
    delta_squared : tuple[np.ndarray, ...]
        A tuple of 1D arrays of |dsq| values, one for each
        redshift value, in units of |mK^2|.
    """

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

    def __repr__(self) -> str:
        """Return a string representation of Data, as a pandas DataFrame."""
        return self.as_pandas_df().__repr__()

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
        if self._delta_squared is not None:
            return tuple(np.asarray(arr) for arr in self._delta_squared)
        if self._delta is not None:
            return tuple(np.asarray(arr) ** 2 for arr in self._delta)
        raise ValueError("Either delta_squared or delta must be provided.")

    def as_pandas_df(self) -> pd.DataFrame:
        """
        Convert the Data object to a pandas DataFrame.

        This is useful for viewing the data in a tabular format
        or using the slicing/conversion capabilities of pandas.
        """
        # Create DataFrame row by row for each z value
        rows = []
        for i in range(len(self.z)):
            row = {
                "z": self.z[i],
                "z_lower": self.z_lower[i] if self.z_lower is not None else None,
                "z_upper": self.z_upper[i] if self.z_upper is not None else None,
                "z_tags": self.z_tags[i] if self.z_tags is not None else "",
                "k": np.array(self.k[i]),
                "k_lower": np.array(self.k_lower[i])
                if self.k_lower is not None
                else None,
                "k_upper": np.array(self.k_upper[i])
                if self.k_upper is not None
                else None,
                "delta_squared": np.array(self.delta_squared[i]),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def drop_nan(self) -> Self:
        """
        Return a new Data object with any rows containing NaN values removed.

        Returns
        -------
        Data
            A new Data object containing only the data points with no NaN values in
            |z|, |k| or |dsq|.
        """

        def mask(dsq):
            return np.isfinite(dsq)

        return attrs.evolve(
            self,
            k=tuple(
                kk[mask(dsq)]
                for kk, dsq in zip(self.k, self.delta_squared, strict=True)
            ),
            k_lower=tuple(
                kl[mask(dsq)]
                for kl, dsq in zip(self.k_lower, self.delta_squared, strict=True)
            )
            if self.k_lower is not None
            else None,
            k_upper=tuple(
                ku[mask(dsq)]
                for ku, dsq in zip(self.k_upper, self.delta_squared, strict=True)
            )
            if self.k_upper is not None
            else None,
            delta_squared=tuple(dsq[mask(dsq)] for dsq in self.delta_squared),
            delta=None,
        )


@attrs.define(frozen=True)
class DataSet:
    r"""
    Class representing a dataset, including metadata and the data itself.

    The metadata includes the telescope,author, year and DOI of the dataset,
    as well as any additional notes. The data is stored in a :class:`~eor_limits.Data`
    object, which contains arrays of redshifts |z|, scales |k| (in units of
    |h/Mpc|) and |dsq| (in units of |mK^2|).

    Attributes
    ----------
    telescope : str
        The name of the telescope or experiment that produced the dataset.
    author : str
        The name of the first author of the dataset,
        or the collaboration name if applicable.
    year : int
        The year the dataset was published.
    doi : str
        The DOI of the dataset. Only published datasets with DOIs should be
        included in the repository.
    data : Data
        The data for the dataset, stored in a Data object.
    notes : tuple[str, ...]
        Any additional notes about the dataset, such as details of the analysis
        or assumptions made about k-bins.
    key : str
        A unique key for the dataset, automatically generated from the author and
        year (e.g. ``'Paciga2013'``). This is used for referencing the dataset.
    """

    telescope: str = attrs.field(converter=str)
    author: str = attrs.field(converter=str)
    year: int = attrs.field(converter=int)
    doi: str = attrs.field(converter=str)
    data: Data = attrs.field(validator=attrs.validators.instance_of(Data))
    notes: tuple[str, ...] = attrs.field(
        default=(), validator=attrs.validators.instance_of(tuple)
    )
    _key: str = attrs.field(converter=str)

    @_key.default
    def _default_key(self) -> str:
        """Generate a default key based on the author and year."""
        return f"{self.author}{self.year}"

    def __repr__(self) -> str:
        """Return a string representation of the DataSet, including metadata."""
        string = "DataSet(\n"
        string += f"\t telescope='{self.telescope}',\n"
        string += f"\t author='{self.author}',\n"
        string += f"\t year={self.year},\n"
        string += f"\t doi='{self.doi}',\n"
        if self.notes:
            string += f"\t notes={self.notes},\n"
        string += "\t data=\n\t\t"
        string_data = self.data.__repr__().replace("\n", "\n\t\t")
        string += string_data
        string += "\n)"
        return string

    @classmethod
    def load(cls, path: str | Path, /) -> Self:
        """
        Load a DataSet from a YAML file or a known dataset name.

        If a known dataset name is provided (e.g. ``'Paciga2013'``), the corresponding
        YAML file will be loaded from the repository. If a path to a YAML file is
        provided, the dataset will be loaded from that file.

        The YAML file must be formatted correctly with the required fields for metadata
        and data, as shown below.

        .. code-block:: yaml

            telescope: GMRT
            author: Paciga
            year: 2013
            doi: 10.1093/mnras/stt753
            data:
              delta_squared: [[6.15e4]]
              k: [[0.5]]
              z: [8.6]

        The limits can be provided as either ``delta`` (|delta|) or
        ``delta_squared`` (|dsq|), but not both.
        If |delta| is provided, it will be squared to obtain |dsq|.
        The |k| values must be provided in units of |h/Mpc|.

        Parameters
        ----------
        path : str | Path
            The path to the YAML file containing the dataset, or a known dataset
            name (e.g. ``'Paciga2013'``) that is registered in ``KNOWN_LIMITS``.

        Returns
        -------
        DataSet
            The loaded dataset as a DataSet object.
        """
        from ._data_loading import _normalize_dataset_name

        path = _normalize_dataset_name(path)

        with path.open("r") as fl:
            yaml_data = yaml.safe_load(fl)

        return converter.structure(yaml_data, cls)

    def _select_with_z_based_mask(self, mask: callable, field: str = "z") -> Self:
        """
        Return a new DataSet with only the data points that satisfy the given z-mask.

        Parameters
        ----------
        mask : callable
            A callable that takes a single argument (the field value) and returns
            a boolean indicating whether to keep that data point.
            e.g. ``lambda z: (z >= z_min) & (z <= z_max)``.
        field : str, optional
            The field to apply the mask to. Can be "z" (default) or "z_tags".
        """
        fld = np.array(getattr(self.data, field))
        fld_mask = mask(fld)
        if not fld_mask.any():
            raise ValueError("No data points found with this mask")
        if fld_mask.shape != self.data.z.shape:
            raise ValueError("Mask must have the same shape as z.")
        new_data = Data(
            z=self.data.z[fld_mask],
            z_lower=self.data.z_lower[fld_mask]
            if self.data.z_lower is not None
            else None,
            z_upper=self.data.z_upper[fld_mask]
            if self.data.z_upper is not None
            else None,
            z_tags=tuple(
                self.data.z_tags[i] for i in range(len(self.data.z)) if fld_mask[i]
            )
            if self.data.z_tags is not None
            else None,
            k=tuple(self.data.k[i] for i in range(len(self.data.z)) if fld_mask[i]),
            k_lower=tuple(
                self.data.k_lower[i] for i in range(len(self.data.z)) if fld_mask[i]
            )
            if self.data.k_lower is not None
            else None,
            k_upper=tuple(
                self.data.k_upper[i] for i in range(len(self.data.z)) if fld_mask[i]
            )
            if self.data.k_upper is not None
            else None,
            delta_squared=tuple(
                self.data.delta_squared[i]
                for i in range(len(self.data.z))
                if fld_mask[i]
            ),
        )
        return attrs.evolve(self, data=new_data)

    def _select_with_k_based_mask(self, mask: callable, field: str = "k") -> Self:
        """
        Return a new DataSet with only the data points that satisfy the given k-mask.

        Parameters
        ----------
        mask : callable
            A callable that takes a single argument (the field value) and returns
            a boolean array indicating which data points to keep.
            e.g. ``lambda k: (k >= k_min) & (k <= k_max)``.
        field : str, optional
            The field to apply the mask to. Can be "k" (default) or "delta_squared".
        """
        fld = getattr(self.data, field)
        fld_mask = [mask(q) for q in fld]
        new_data = Data(
            z=np.array([
                self.data.z[i] for i in range(len(self.data.z)) if any(fld_mask[i])
            ]),
            z_lower=np.array([
                self.data.z_lower[i]
                for i in range(len(self.data.z))
                if any(fld_mask[i])
            ])
            if self.data.z_lower is not None
            else None,
            z_upper=np.array([
                self.data.z_upper[i]
                for i in range(len(self.data.z))
                if any(fld_mask[i])
            ])
            if self.data.z_upper is not None
            else None,
            z_tags=tuple(
                self.data.z_tags[i] for i in range(len(self.data.z)) if any(fld_mask[i])
            )
            if self.data.z_tags is not None
            else None,
            k=tuple(
                kk[mask]
                for kk, mask in zip(self.data.k, fld_mask, strict=True)
                if any(mask)
            ),
            k_lower=tuple(
                kl[mask]
                for kl, mask in zip(self.data.k_lower, fld_mask, strict=True)
                if any(mask)
            )
            if self.data.k_lower is not None
            else None,
            k_upper=tuple(
                ku[mask]
                for ku, mask in zip(self.data.k_upper, fld_mask, strict=True)
                if any(mask)
            )
            if self.data.k_upper is not None
            else None,
            delta_squared=tuple(
                dsq[mask]
                for dsq, mask in zip(self.data.delta_squared, fld_mask, strict=True)
                if any(mask)
            ),
        )
        return attrs.evolve(self, data=new_data)

    def select_z_range(self, z_min: float, z_max: float) -> Self:
        """
        Return a new DataSet with only the data in the specified |z| range.

        Parameters
        ----------
        z_min : float
            The minimum |z| value to include (inclusive).
        z_max : float
            The maximum |z| value to include (inclusive).

        Returns
        -------
        DataSet
            A new DataSet containing only the data points with |z| values
            between ``z_min`` and ``z_max``.
        """

        def mask(z):
            return (z >= z_min) & (z <= z_max)

        try:
            return self._select_with_z_based_mask(mask, "z")
        except ValueError as err:
            raise ValueError(
                f"No data points found in the z range {z_min} to {z_max}."
            ) from err

    def select_k_range(self, k_min: float, k_max: float) -> Self:
        """
        Return a new DataSet with only the data in the specified |k| range.

        Parameters
        ----------
        k_min : float
            The minimum |k| value to include (inclusive).
        k_max : float
            The maximum |k| value to include (inclusive).

        Returns
        -------
        DataSet
            A new DataSet containing only the data points with |k| values
            between ``k_min`` and ``k_max``.
        """

        def mask(kk):
            return (kk >= k_min) & (kk <= k_max)

        try:
            return self._select_with_k_based_mask(mask, "k")
        except ValueError as err:
            raise ValueError(
                f"No data points found in the k range {k_min} to {k_max}."
            ) from err

    def select_delta_squared_range(
        self, delta_squared_min: float, delta_squared_max: float
    ) -> Self:
        """
        Return a new DataSet with only the data in the specified |dsq| range.

        Parameters
        ----------
        delta_squared_min : float
            The minimum |dsq| value to include (inclusive).
        delta_squared_max : float
            The maximum |dsq| value to include (inclusive).

        Returns
        -------
        DataSet
            A new DataSet containing only the data with |dsq| values
            between ``delta_squared_min`` and ``delta_squared_max``.
        """

        def mask(dsq):
            return (dsq >= delta_squared_min) & (dsq <= delta_squared_max)

        try:
            return self._select_with_k_based_mask(mask, "delta_squared")
        except ValueError as err:
            raise ValueError(
                "No data points found in the delta_squared range "
                f"{delta_squared_min} to {delta_squared_max}."
            ) from err

    def select_closest_z(self, z_target: float) -> Self:
        """
        Return a new DataSet with only the data closest to the target |z|.

        Parameters
        ----------
        z_target : float
            The target redshift value to find the closest data to.

        Returns
        -------
        DataSet
            A new DataSet containing only the data with |z| value closest
            to ``z_target``.

        """

        def mask(z):
            # Note: this mask is different from select_closest_k because
            # some datasets have multiple entries for the same redshift
            # (e.g. different polarizations or fields), so we want to keep all of them.
            closest = z[np.argmin(np.abs(z - z_target))]
            return z == closest

        return self._select_with_z_based_mask(mask, "z")

    def select_closest_k(self, k_target: float) -> Self:
        """
        Return a new DataSet with only the data closest to the target |k|.

        Parameters
        ----------
        k_target : float
            The target |k| value to find the closest data to.

        Returns
        -------
        DataSet
            A new DataSet containing only the data with |k| value closest
            to ``k_target`` for each redshift.
        """

        def mask(kk):
            idx = np.abs(kk - k_target).argmin()
            mask = np.zeros_like(kk, dtype=bool)
            mask[idx] = True
            return mask

        return self._select_with_k_based_mask(mask, "k")

    def select_lowest_delta_squared(
        self, per_z: bool = False, per_tag: bool = False
    ) -> Self:
        """
        Return a new DataSet with only the lowest |dsq| data points.

        Parameters
        ----------
        per_z : bool, optional
            If True, and the dataset has |z| tags (e.g. multiple fields or
            polarizations at the same redshift), collapse across tags by
            keeping only the field with the lowest |dsq|.
            If False (default), maintains them as separate entries.

        per_tag : bool, optional
            If True, and the dataset has |z| tags (e.g. multiple fields or
            polarizations at the same redshift), collapse across |z| by
            keeping only the |z| with the lowest |dsq|.
            If False (default), maintains them as separate entries.

        Returns
        -------
        DataSet
            A new DataSet containing only the data point with the lowest |dsq|
            value for each redshift.
        """

        def k_mask(dsq):
            m = np.zeros_like(dsq, dtype=bool)
            m[np.nanargmin(dsq)] = True
            return m

        def z_mask_gen(fld, dsq):
            m = np.zeros(len(fld), dtype=bool)
            for f in set(fld):
                idx = [i for i, g in enumerate(fld) if g == f]
                m[idx[np.nanargmin([dsq[i][0] for i in idx])]] = True
            return m

        selected = self._select_with_k_based_mask(k_mask, "delta_squared")

        if self.data.z_tags is not None:
            if per_z:
                z_mask = z_mask_gen(selected.data.z, selected.data.delta_squared)
                selected = selected._select_with_z_based_mask(lambda z: z_mask, "z")
            if per_tag:
                z_mask = z_mask_gen(selected.data.z_tags, selected.data.delta_squared)
                selected = selected._select_with_z_based_mask(
                    lambda t: z_mask, "z_tags"
                )
        return selected

    @property
    def key(self) -> str:
        return self._key

    def drop_nan(self) -> Self:
        """
        Return a new DataSet with any rows containing NaN values removed.

        Returns
        -------
        DataSet
            A new DataSet containing only the data points with no NaN values in
            |z|, |k| or |dsq|.
        """
        new_data = self.data.drop_nan()
        return attrs.evolve(self, data=new_data)
