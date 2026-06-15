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
| `--prefix P` | Output file prefix (default: `convino`) — result goes to `P_result.txt`. |
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

## Package layout

| Module | Responsibility |
|--------|----------------|
| `parser.py`      | Read measurement + config files into raw dataclasses. |
| `measurement.py` | Turn one measurement's raw data into the matrices the χ² needs (`LM`, `Lk`, `LD`). |
| `objective.py`   | JAX-jitted χ²(pars). |
| `combiner.py`    | Orchestrator: indices, prior, minimize, errors, impacts. |
| `result.py`      | Format the output `.txt`. |
| `cli.py`         | Argparse entry point. |
