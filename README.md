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
| `--export {npz,json,both}` | Also write a machine-readable result (`P_result.npz`/`.json`) alongside the text output — combined values/errors, full covariance, stat/syst split, and per-systematic impacts, for feeding a downstream fit. |
| `--no-impacts` | Skip uncertainty-impact computation entirely (both user-defined impact groups and the per-systematic/stat-only breakdown) — fastest run when only combined values/covariance are needed. Mutually exclusive with `--impacts-only`. |
| `--impacts-only GROUP1,GROUP2` | Only compute the listed `[uncertainty impacts]` group(s) instead of all of them; the per-systematic/stat-only breakdown is unaffected. Mutually exclusive with `--no-impacts`. |
| `--scan` | Scan each `[correlations]` group that has an actual `(nominal & low : high)` range across `--scan-steps` points, recombining at each one; written to `P_scan_result.txt` (plus `.npz`/`.json` if `--export` is also given). |
| `--scan-steps N` | Number of points per correlation-scan group (default: 6). |
| `--verbose`  | Print per-phase timing checkpoints (parse, chi2 build, minimize, post-fit, impacts, total) to stderr. |
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

`CombinationResult` also carries `ndf`/`chi2_per_ndf`/`p_value`, a stat/syst covariance
split (`stat_only_covariance`, `total_syst_impact_up/down`), and a per-systematic
leave-one-out breakdown (`impact_per_systematic`, `cov_per_systematic`) — exported via
`result.to_dict()` / `export_npz()` / `export_json()` (`convino_jax.result`). For
scanning a correlation assumption instead of a single fit, use
`Combiner.scan_correlations()`.

## Tests

The suite has three parts: numeric regression tests that run `ConvinoSetups/`
combinations end-to-end and compare to stored golden snapshots
(`test/golden/*.npz`), a byte-level formatter golden (`test/golden/formatter.txt`),
and fast unit tests for the output-path handling.

```bash
python -m unittest discover -t . -s test       # fast suite (~4 s)
```

The full `corrV2` combination (~25 s; central values validated once against the
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
