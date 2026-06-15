# convino_jax

A Python/JAX port of [Convino](https://github.com/jkiesele/Convino) — a tool for
combining physics measurements with correlated systematic uncertainties. This
reimplementation needs neither ROOT nor Minuit: the χ² is built with JAX and
minimized with SciPy.

## Install

```bash
pip install -e .
```

This pulls in JAX, NumPy and SciPy. (For GPU/TPU builds of JAX, install the
appropriate `jaxlib` wheel for your platform first, then `pip install -e .`.)

## Usage

```bash
# the only required argument is the combination's config file
convino path/to/rho_config.txt --prefix /tmp/mycombo
# -> writes /tmp/mycombo_result.txt

# or, without installing the console script:
python -m convino_jax.cli path/to/rho_config.txt --prefix /tmp/mycombo
```

Options:

| Flag | Meaning |
|------|---------|
| `--prefix P` | Output file prefix (default: `convino`) — result goes to `P_result.txt`. May include a directory (`out/run1`); missing directories are created, and an unwritable destination is rejected up front (before the fit) rather than after. |
| `--pearson`  | Use the Pearson χ² instead of the default Neyman χ². |
| `--debug`    | Print a full traceback on error. |

Input files use the original Convino text format (measurement files with
`[hessian]` / `[correlation matrix]` / `[not fitted]` / `[estimates]` /
`[systematics]` blocks, and a config file with `[observables]` /
`[correlations]` / `[uncertainty impacts]`).

## Library API

```python
from convino_jax import Combiner, write_result

result = Combiner.from_config("rho_config.txt").combine()
write_result(result, prefix="mycombo")
```

## Tests

The suite has three parts: numeric regression tests that run `ConvinoSetups/`
combinations end-to-end and compare to stored golden snapshots
(`test/golden/*.npz`), a byte-level formatter golden (`test/golden/formatter.txt`),
and fast unit tests for the output-path handling.

```bash
python -m unittest discover -t . -s test       # fast suite (~4 s)
```

The full `corrV2` combination (~6 s; central values validated once against the
original C++ Convino) is skipped by default — enable it with:

```bash
CONVINO_SLOW_TESTS=1 python -m unittest discover -t . -s test
```

If you change the numeric or formatted output *intentionally*, regenerate and
commit the goldens:

```bash
python -m test.regen_golden            # all setups + formatter golden
python -m test.regen_golden statonly   # a single setup
```

## Package layout

| Module | Responsibility |
|--------|----------------|
| `parser.py`      | Read measurement + config files into raw dataclasses. |
| `measurement.py` | Turn one measurement's raw data into the matrices the χ² needs (`LM`, `Lk`, `LD`). |
| `objective.py`   | JAX-jitted χ²(pars). |
| `combiner.py`    | Orchestrator: indices, prior, minimize, errors, impacts. |
| `result.py`      | Format the output `.txt`. |
| `cli.py`         | Argparse entry point. |
