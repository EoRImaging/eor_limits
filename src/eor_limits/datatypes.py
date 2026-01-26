"""A module defining the data types used in eor-limits."""

import yaml

from .data import DATA_PATH, THEORY_PATH


def read_data_yaml(paper_name: str, theory: bool = False):
    """
    Read in the data from a paper yaml file.

    Parameters
    ----------
    paper_name : str
        Short name of paper (usually author_year) which corresponds to a file
        in the data directory named <paper_name>.yaml
    theory : bool
        Flag that this is a theory paper and so is in the theory folder.

    Returns
    -------
    dict
        Dictionary with the parsed yaml for use in the plotting code.

    """
    if theory:
        file_name = THEORY_PATH / (paper_name + ".yaml")
    else:
        file_name = DATA_PATH / (paper_name + ".yaml")

    with file_name.open() as pfile:
        paper_dict = yaml.safe_load(pfile)

    if isinstance(paper_dict["delta_squared"][0], (str,)):
        try:
            paper_dict["delta_squared"] = [
                float(val) for val in paper_dict["delta_squared"]
            ]
        except ValueError:
            val_list = []
            for val in paper_dict["delta_squared"]:
                if "**" in val:
                    val_split = val.split("**")
                    val_list.append(float(val_split[0]) ** float(val_split[1]))
                else:
                    val_list.append(float(val))
            paper_dict["delta_squared"] = val_list
    elif isinstance(paper_dict["delta_squared"][0], (list,)) and isinstance(
        paper_dict["delta_squared"][0][0], (str,)
    ):
        for ind, elem in enumerate(paper_dict["delta_squared"]):
            try:
                paper_dict["delta_squared"][ind] = [float(val) for val in elem]
            except ValueError:
                val_list = []
                for val in paper_dict["delta_squared"][ind]:
                    if "**" in val:
                        val_split = val.split("**")
                        val_list.append(float(val_split[0]) ** float(val_split[1]))
                    else:
                        val_list.append(float(val))
                paper_dict["delta_squared"][ind] = val_list

    return paper_dict
