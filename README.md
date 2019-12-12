# eor_limits

eor_limits is a small utility for plotting EoR Limits.

The primary functionality is to plot published limits and simulations as a
function of k and redshift. The data for the plots is included in human-readable
yaml files in the data folder, so adding new data or theory lines is as simple
as putting the data in a yaml file and placing it in the data directory.

We aim to continually update the repository to include all published limits in the field.
The goal is to make it easier for experimentalists and theorists to add their own data or simulations and
quickly compare on a 'standard' plot.

This code snippet is fully open source (BSD). Feel free to use the resulting plots in papers or presentations as you see fit.

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

Change directories into the `eor_limits` folder and run ```pip install .``` (including the dot).

To install without dependencies, run `pip install --no-deps .` (including the dot).

### Dependencies
If you prefer to manage dependencies outside of pip (e.g. via conda), you will
need to install the following dependencies.

* numpy
* matplotlib
* pyyaml

# Making the plots

You can use the software either directly from the UNIX command line (pdf output), or from within a python notebook (inline figure).

### Command Line

After installation, call the plotting script as ```plot_eor_limits.py``` from the command line to make the default
plot (including all the papers in the data folder). There are a number of
options to customize the plot, use ```plot_eor_limits.py --help```
to see the various options.

For example, ```plot_eor_limits.py --bold barry_2019 kolopanis_2019 li_2019``` would bold the references to the papers published in 2019


### Notebook

Alternatively, to make the plot within an interactive environment like a
Jupyter notebook, run:
```
import eor_limits
eor_limits.make_plot()
```
and the plot should appear inline

To see the options type ```eor_limits.make_plot?``` in the environment. For example ```eor_limits.make_plot(bold_papers='barry_2019 kolopanis_2019 li_2019')``` will give also bold the 2019 results.
