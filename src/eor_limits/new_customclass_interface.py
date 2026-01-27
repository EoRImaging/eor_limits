import os
import pathlib

import attr
import numpy as np
import pandas as pd
import yaml

##################################################################
# Converter and Validator functions           #####
##################################################################


def process_data(d: dict) -> dict:

    # Checking if empty
    def to_empty(arr):
        return [] if arr == [] or arr is None else arr

    # Evaluating expressions
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
        if isinstance(arr, (list, np.ndarray)):
            return [to_eval(x) for x in arr]
        return eval_item(arr)

    # Process each attribute
    for attr_name in [
        "z",
        "z_lower",
        "z_upper",
        "k",
        "k_lower",
        "k_upper",
        "delta_squared",
        "z_tags",
    ]:
        arr = d.get(attr_name, [])
        arr = to_empty(arr)
        if attr_name != "z_tags":
            arr = to_eval(arr)
        d[attr_name] = arr


def validate_data(d: dict) -> None:

    def is_allowed(arr, allowed_types):
        return all(isinstance(x, allowed_types) for x in arr)

    def is_same_length(arr1, arr2):
        if isinstance(arr1, (list, np.ndarray)) and isinstance(
            arr2, (list, np.ndarray)
        ):
            if len(arr1) != len(arr2):
                return False
            return all(is_same_length(a1, a2) for a1, a2 in zip(arr1, arr2))
        return True

    # Check types
    if not is_allowed(d["z"], (int, float)):
        raise ValueError("z must be a 1D array of numbers.")
    if len(d["z_lower"]) != 0 and not is_allowed(d["z_lower"], (int, float)):
        raise ValueError("z_lower must be a 1D array of numbers.")
    if len(d["z_upper"]) != 0 and not is_allowed(d["z_upper"], (int, float)):
        raise ValueError("z_upper must be a 1D array of numbers.")
    if len(d["z_tags"]) != 0 and not is_allowed(d["z_tags"], str):
        raise ValueError("z_tags must be a 1D array of strings.")
    if not is_allowed(d["k"], (list, np.ndarray)):
        raise ValueError("k must be a 2D array of numbers.")
    if not is_allowed(d["k_lower"], (list, np.ndarray)):
        raise ValueError("k_lower must be a 2D array of numbers.")
    if not is_allowed(d["k_upper"], (list, np.ndarray)):
        raise ValueError("k_upper must be a 2D array of numbers.")
    if not is_allowed(d["delta_squared"], (list, np.ndarray)):
        raise ValueError("delta_squared must be a 2D array of numbers.")

    # Check lengths
    if d["z_lower"] and not is_same_length(d["z_lower"], d["z"]):
        raise ValueError("z_lower must be the same shape as z.")
    if d["z_upper"] and not is_same_length(d["z_upper"], d["z"]):
        raise ValueError("z_upper must be the same shape as z.")
    if d["z_tags"] and not is_same_length(d["z_tags"], d["z"]):
        raise ValueError("z_tags must be the same shape as z.")
    if d["k_lower"] and not is_same_length(d["k_lower"], d["k"]):
        raise ValueError("k_lower must be the same shape as k.")
    if d["k_upper"] and not is_same_length(d["k_upper"], d["k"]):
        raise ValueError("k_upper must be the same shape as k.")
    if not is_same_length(d["delta_squared"], d["k"]):
        raise ValueError("delta_squared must be the same shape as k.")


def to_pandas_df(d: dict) -> pd.DataFrame:
    # Create DataFrame row by row for each z value
    rows = []
    for i in range(len(d["z"])):
        row = {
            "z": d["z"][i],
            "z_lower": d["z_lower"][i] if len(d["z_lower"]) != 0 else np.nan,
            "z_upper": d["z_upper"][i] if len(d["z_upper"]) != 0 else np.nan,
            "z_tags": d["z_tags"][i] if len(d["z_tags"]) != 0 else "",
            "k": np.array(d["k"][i]),
            "k_lower": np.array(d["k_lower"][i]) if len(d["k_lower"]) != 0 else np.nan,
            "k_upper": np.array(d["k_upper"][i]) if len(d["k_upper"]) != 0 else np.nan,
            "delta_squared": np.array(d["delta_squared"][i]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


##################################################################
# DataSet class                      #####
##################################################################


@attr.define
class DataSet:
    telescope: str = attr.field(default="", validator=attr.validators.instance_of(str))
    author: str = attr.field(default="", validator=attr.validators.instance_of(str))
    year: int = attr.field(default=0, validator=attr.validators.instance_of(int))
    doi: str = attr.field(default="", validator=attr.validators.instance_of(str))
    notes: list = attr.field(default=[], validator=attr.validators.instance_of(list))
    data: pd.DataFrame = attr.field(
        default=pd.DataFrame(),
        converter=to_pandas_df,
        validator=attr.validators.instance_of(pd.DataFrame),
    )

    def __str__(self) -> str:
        text = f"DataSet: telescope={self.telescope}, author={self.author}, year={self.year}, doi={self.doi}"
        if self.notes:
            text += ",\nnotes=["
            text += ",\n       ".join(self.notes)
            text += "]"
        text += ",\ndata=\n"
        text += str(self.data)
        text += "\n"
        return text

    def __repr__(self) -> str:
        return self.__str__()


def get_available_datasets() -> list[str]:

    files = [
        os.path.basename(f)[:-5] for f in os.listdir("data") if f.endswith(".yaml")
    ]
    return files


def load_dataset(fname: str) -> DataSet:

    fname = fname[:-5] if fname.endswith(".yaml") else fname
    if fname in get_available_datasets():
        pass
    else:
        raise ValueError(
            f"Dataset '{fname}' not found. Available datasets: {get_available_datasets()}"
        )
    with pathlib.Path("data/" + fname + ".yaml").open("r") as file:
        yaml_data = yaml.safe_load(file)

    # Process and validate data
    data_dict = yaml_data.get("data", {})
    process_data(data_dict)
    validate_data(data_dict)

    return DataSet(
        telescope=yaml_data.get("telescope", ""),
        author=yaml_data.get("author", ""),
        year=yaml_data.get("year", 0),
        doi=yaml_data.get("doi", ""),
        notes=yaml_data.get("notes", []),
        data=to_pandas_df(data_dict),
    )


# WARNING: This might be over-estimating the lowest limit, if the lowest k-bin is erroneously low.
def load_dataset_lowest_limits(fname: str) -> DataSet:

    dataset = load_dataset(fname)

    # For all unique z values, find the lowest limit
    z_L, z_lower_L, z_upper_L, z_tags_L, k_L, k_lower_L, k_upper_L, dsq_L = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    unique_z = dataset.data["z"].unique()
    for z_val in unique_z:
        subset = dataset.data[dataset.data["z"] == z_val]
        min_dsq = np.inf
        min_row = None
        for _, row in subset.iterrows():
            ik = np.nanargmin(row["delta_squared"])
            curr_dsq = np.nanmin(row["delta_squared"])
            if curr_dsq < min_dsq:
                min_dsq = curr_dsq
                min_row = (row, ik)
        if min_row is not None:
            row, ik = min_row
            z_L.append(row["z"])
            z_lower_L.append(row["z_lower"] if not pd.isna(row["z_lower"]) else np.nan)
            z_upper_L.append(row["z_upper"] if not pd.isna(row["z_upper"]) else np.nan)
            z_tags_L.append(row["z_tags"] if row["z_tags"] != "" else "")
            k_L.append([row["k"][ik]])
            k_lower_L.append(
                [row["k_lower"][ik]] if not pd.isna(row["k_lower"]).all() else np.nan
            )
            k_upper_L.append(
                [row["k_upper"][ik]] if not pd.isna(row["k_upper"]).all() else np.nan
            )
            dsq_L.append([row["delta_squared"][ik]])

    # Create new DataFrame
    dataset.data = pd.DataFrame({
        "z": z_L,
        "z_lower": z_lower_L,
        "z_upper": z_upper_L,
        "z_tags": z_tags_L,
        "k": k_L,
        "k_lower": k_lower_L,
        "k_upper": k_upper_L,
        "delta_squared": dsq_L,
    })

    return dataset
