# eor_limits

eor_limits is a small utility for plotting EoR Limits.

The primary functionality is to plot published limits and simulations as a
function of k and redshift.

We aim to include all published limits in the field and to make it easier for
experimentalists and theorists to add their own data or simulations and
to the plot to compare them to the existing limits.

# Community Guidelines
We welcome pull requests to add new data sets or simulations. We require that
data and simulations come from peer-reviewed published papers. If you would
like to add unpublished data or simulations to a plot, please fork this
repository and add the data or simulations in your fork. If the data or
simulations are subsequently published we welcome a pull request from your
fork to add them to this repository.

We also welcome bug reports or feature requests, please file them as issues
on the github repository.

# Installation
Clone the repository using
```git clone https://github.com/EoRImaging/eor_limits```

Change directories into the `eor_limits` folder and run ```pip install .```

To install without dependencies, run `pip install --no-deps .`

## Dependencies
If you prefer to manage dependencies outside of pip (e.g. via conda), you will
need to install the following dependencies.

* numpy
* matplotlib
* pyyaml

# Making the plots
Call the plotting script as ```plot_eor_limits.py``` to make the default
plot (including all the papers in the data folder). There are a number of
options to customize the plot, use ```plot_eor_limits.py --help```
to see the various options.
