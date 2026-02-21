# Using `eor-limits`

You can use ``eor-limits`` via the Python library, or the command-line.
See the following intro tutorials for each:

## How to use the library

Import like so:

```python
from eor_limits import plots
```

Then make a plot like this:

```python
fig = plots.make_plot()
fig.show()  # or savefig, or whatever
```

## How to use the CLI

Run

```bash
eor-limits plot [options]
```
