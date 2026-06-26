## Environment

- Use the `uv` python environment to run code.

## Tests

- Write tests using the pytest framework, in `tests/`
- Run narrow tests based on implemented features first, using `uv run pytest -k "test-identifier"`
- Always write a failing test before fixing a bug.
- Write tests of intended behaviour before implementing a feature.

## Style

- Prefer adding type annotations for new parameters and functions, wherever reasonable.
- Always run `uv run ruff format` and `ruff check --fix` before closing the loop.
