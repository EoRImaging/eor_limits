import os
import pathlib

import attr
import numpy as np
import yaml

##################################################################
# Converter functions                   #####
##################################################################


def to_empty(arr):
    if (
        arr == []
        or arr is None
        or arr is np.array([], dtype=object)
        or arr is np.array(None, dtype=object)
    ):
        return np.array([], dtype=object)
    return np.array(arr, dtype=object)


def to_eval(arr):

    # Eval an item. Allows for "21**2" type expressions
    def eval_item(item):
        # Only allow safe evaluation of math expressions
        if item == "nan":
            return np.nan
        if isinstance(item, str):
            item = eval(str(item), {"__builtins__": None}, {})
            return float(item)
        if isinstance(item, (int, float)):
            return float(item)

    # Process the list recursively (handles nested lists)
    def process_list(lst):
        if isinstance(lst, (list, np.ndarray)):
            return [process_list(x) for x in lst]
        return eval_item(lst)

    processed_arr = process_list(arr)

    return np.array(processed_arr, dtype=object)


##################################################################
# Data class                        #####
##################################################################


@attr.define
class Data:
    z: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=[to_empty, to_eval]
    )

    @z.validator
    def check_z(self, attribute, value):
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("z must be a 1D array of numbers.")

    z_lower: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=[to_empty, to_eval]
    )

    @z_lower.validator
    def check_z_lower(self, attribute, value):
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("z_lower must be a 1D array of numbers.")
        if value.size != 0 and not value.shape == self.z.shape:
            raise ValueError("z_lower must be the same shape as z.")

    z_upper: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=[to_empty, to_eval]
    )

    @z_upper.validator
    def check_z_upper(self, attribute, value):
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("z_upper must be a 1D array of numbers.")
        if value.size != 0 and not value.shape == self.z.shape:
            raise ValueError("z_upper must be the same shape as z.")

    z_tags: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=to_empty
    )

    @z_tags.validator
    def check_z_tags(self, attribute, value):
        if not all(isinstance(x, str) for x in value):
            raise ValueError("z_tags must be a 1D array of strings.")
        if value.size != 0 and not value.shape == self.z.shape:
            raise ValueError("z_tags must be the same shape as z.")

    k: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=[to_empty, to_eval]
    )

    @k.validator
    def check_k(self, attribute, value):
        if not all(isinstance(x, (list, np.ndarray)) for x in value):
            raise ValueError("k must be a 2D array of numbers.")

    k_lower: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=[to_empty, to_eval]
    )

    @k_lower.validator
    def check_k_lower(self, attribute, value):
        if not all(isinstance(x, (list, np.ndarray)) for x in value):
            raise ValueError("k_lower must be a 2D array of numbers.")
        if value.size != 0 and not value.shape == self.k.shape:
            raise ValueError("k_lower must be the same shape as k.")

    k_upper: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=[to_empty, to_eval]
    )

    @k_upper.validator
    def check_k_upper(self, attribute, value):
        if not all(isinstance(x, (list, np.ndarray)) for x in value):
            raise ValueError("k_upper must be a 2D array of numbers.")
        if value.size != 0 and not value.shape == self.k.shape:
            raise ValueError("k_upper must be the same shape as k.")

    delta_squared: np.ndarray = attr.field(
        default=np.array([], dtype=object), converter=[to_empty, to_eval]
    )

    @delta_squared.validator
    def check_delta_squared(self, attribute, value):
        if not all(isinstance(x, (list, np.ndarray)) for x in value):
            raise ValueError("delta_squared must be a 2D array of numbers.")
        if not value.shape == self.k.shape:
            raise ValueError("delta_squared must be the same shape as k.")


##################################################################
# Dataset class                      #####
##################################################################


@attr.define
class DataSet:
    telescope: str = attr.field(default="", validator=attr.validators.instance_of(str))
    author: str = attr.field(default="", validator=attr.validators.instance_of(str))
    year: int = attr.field(default=0, validator=attr.validators.instance_of(int))
    doi: str = attr.field(default="", validator=attr.validators.instance_of(str))
    notes: list = attr.field(default=[], validator=attr.validators.instance_of(list))
    data: Data = attr.field(default=Data(), validator=attr.validators.instance_of(Data))


##################################################################
# Loader function                     #####
##################################################################


def get_available_datasets() -> list[str]:

    files = [
        os.path.basename(f)[:-5] for f in os.listdir("data") if f.endswith(".yaml")
    ]
    return files


def load_dataset(file_path: str) -> DataSet:

    file_path = "data/" + file_path
    file_path = file_path + ".yaml" if not file_path.endswith(".yaml") else file_path
    with pathlib.Path(file_path).open() as file:
        yaml_data = yaml.safe_load(file)

    return DataSet(
        telescope=yaml_data.get("telescope", ""),
        author=yaml_data.get("author", ""),
        year=yaml_data.get("year", 0),
        doi=yaml_data.get("doi", ""),
        notes=yaml_data.get("notes", []),
        data=Data(**yaml_data.get("data", {})),
    )


# WARNING: This might be over-estimating the lowest limit, if the lowest k-bin is erroneously low.
def load_dataset_lowest_limits(filepath: str) -> DataSet:

    # Retrieve dataset
    dataset = load_dataset(filepath)

    # Prepare lists to hold lowest limits
    z_L, k_L, dsq_L, k_lower_L, k_upper_L, z_lower_L, z_upper_L, z_tags_L = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Loop over unique z values
    z_arr = dataset.data.z
    unique_z = np.unique(z_arr)
    for z_val in unique_z:
        # Find indices corresponding to this z value
        indices = np.where(z_arr == z_val)[0]
        min_val = np.inf
        min_idx = None
        # Loop over these indices to find the minimum delta_squared
        for iz in indices:
            ik = np.nanargmin(dataset.data.delta_squared[iz])
            min_dsq = np.nanmin(dataset.data.delta_squared[iz])
            if min_dsq < min_val:
                min_val = min_dsq
                min_idx = (iz, ik)
        iz, ik = min_idx
        # Append this minimum to the new dataset
        z_L.append(z_val)
        k_L.append([dataset.data.k[iz][ik]])
        dsq_L.append([dataset.data.delta_squared[iz][ik]])
        if dataset.data.k_lower.size > 0:
            k_lower_L.append([dataset.data.k_lower[iz][ik]])
        if dataset.data.k_upper.size > 0:
            k_upper_L.append([dataset.data.k_upper[iz][ik]])
        if dataset.data.z_lower.size > 0:
            z_lower_L.append(dataset.data.z_lower[iz])
        if dataset.data.z_upper.size > 0:
            z_upper_L.append(dataset.data.z_upper[iz])
        if dataset.data.z_tags.size > 0:
            z_tags_L.append(dataset.data.z_tags[iz])

    # Create new DataSet with lowest limits
    return DataSet(
        telescope=dataset.telescope,
        author=dataset.author,
        year=dataset.year,
        doi=dataset.doi,
        notes=dataset.notes,
        data=Data(
            z=np.array(z_L, dtype=object),
            z_lower=np.array(z_lower_L, dtype=object)
            if z_lower_L
            else np.array([], dtype=object),
            z_upper=np.array(z_upper_L, dtype=object)
            if z_upper_L
            else np.array([], dtype=object),
            z_tags=np.array(z_tags_L, dtype=object)
            if z_tags_L
            else np.array([], dtype=object),
            k=np.array(k_L, dtype=object),
            k_lower=np.array(k_lower_L, dtype=object)
            if k_lower_L
            else np.array([], dtype=object),
            k_upper=np.array(k_upper_L, dtype=object)
            if k_upper_L
            else np.array([], dtype=object),
            delta_squared=np.array(dsq_L, dtype=object),
        ),
    )
