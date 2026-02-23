# eor-limits

`eor-limits` is a small utility that provides a compendium of published
upper limits on the 21-cm power spectrum from the Epoch of Reionization (EoR)
and Cosmic Dawn. It also provides functionality to plot these limits as a function of scale $k$ and redshift $z$, along with some theoretical predictions from simulations.

The published limits are included in human-readable yaml files in the data folder, and the plotting functionality is highly customizable through keywords. It is possible to read in custom yaml files to add new limits or simulations, and we welcome pull requests to add new data sets or simulations from published papers.

![Example EoR Limit plot](docs/source/_static/eor_limits.png)

## Installation and dependencies

### Dependencies

If you prefer to manage dependencies manually, you will
need to install the following packages:

* cattrs>=25.3.0
* cyclopts>=4.5.1
* h5py>=3.15.1
* matplotlib>=3.10.8
* numpy>=2.4.1
* pandas>=3.0.0
* pyyaml>=6.0.3
* rich>=14.3.1

### User installation

Clone the repository using
```git clone https://github.com/EoRImaging/eor_limits```

For a simple user installation, change directories into the `eor_limits` folder and
run ```pip install .``` (including the dot).

To install without dependencies, run `pip install --no-deps .` (including the dot).

### Developer installation

Developers who would like to contribute to the code can install the package as:

```pip install -e .[dev]```

or with `uv`:

```uv sync --all-extras```

To use pre-commit to prevent committing code that does not follow our style,
you'll need to run `pre-commit install` in the top level `eor_limits` directory.

## Usage and examples

There are three main ways to use the code: through the command line interface (CLI), as a library within a notebook or python script, or through the [online interface](https://eorlimits.streamlit.app/).

### Using the CLI

After installation, you can use the `eor-limits` command from the terminal. For example, to make a plot of all the limits in the data folder, run ```eor-limits -h``` or ```eor-limits plot-vs-k -h``` to see the various options for customizing the plot, and then run:

```bash
eor-limits plot-vs-k --out=eor_limits.png
```

to make the default plot of all the limits as a function of scale $k$.

You can also specify which papers to and customize the plot:

```bash
eor-limits plot-vs-k \
    --limits=Barry2019,Kolopanis2019,Li2019 \
    --bold_limits=Barry2019 \
    --theories=Mesinger2016Bright \
    --out=eor_limits_custom.png
```


### Using the library

To use the plotting functionality within a notebook or python script, you can import the relevant functions from the library. For example, to make the default plot of all the limits as a function of scale $k$, run:

```python
from eor_limits import plot_vs_k
plot_vs_k()
```

For more detailed examples, see the Notebooks.

### Using the online interface

We have also created an online interface to make it easy for users to explore the limits and make plots without needing to install anything. You can access the online interface at [https://eorlimits.streamlit.app/](https://eorlimits.streamlit.app/). The online interface allows you to select which limits to include in the plot, filter them in various ways, and download the data and the plot.

## Community guidelines

### Contributing to the repository

We welcome contributions to this repository from the community. If you would like to contribute, please follow these guidelines:

* If you would like to add new data sets or simulations, ensure that they come from peer-reviewed published papers and make a pull request to add them to the repository (see Developer Installation for how to set up your environment).

* If you would like to add unpublished data or simulations to a plot, please fork this repository and add the data or simulations in your fork. If the data or simulations are subsequently published we welcome a pull request from your fork to add them to this repository.

* If you would like to report a bug or request a feature, please file an issue on the github repository.

### License

This code is fully open-source, available under BSD 2-Clause License. See the LICENSE file for more details.  Please feel free to use and modify the code as needed.
