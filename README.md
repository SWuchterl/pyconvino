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
| `--export {npz,json,both}` | Also write a machine-readable result (`P_result.npz`/`.json`) alongside the text output — combined values/errors, full covariance, stat/syst split, per-systematic impacts, signed response matrix (`impact_matrix`), and group-level pulls, for feeding a downstream fit. |
| `--no-impacts` | Skip uncertainty-impact computation entirely (both user-defined impact groups and the per-systematic/stat-only breakdown) — fastest run when only combined values/covariance are needed. Mutually exclusive with `--impacts-only`. |
| `--impacts-only GROUP1,GROUP2` | Only compute the listed `[uncertainty impacts]` group(s) instead of all of them; the per-systematic/stat-only breakdown is unaffected. Mutually exclusive with `--no-impacts`. |
| `--scan` | Scan each `[correlations]` group that has an actual `(nominal & low : high)` range across `--scan-steps` points, recombining at each one; written to `P_scan_result.txt` (plus `.npz`/`.json` if `--export` is also given). |
| `--scan-steps N` | Number of points per correlation-scan group (default: 6). |
| `--verbose`  | Print per-phase timing checkpoints (parse, chi2 build, minimize, post-fit, impacts, total) to stderr. |
| `--pd-reg-method {shift,clip,higham}` | Strategy for regularising a non-positive-definite prior correlation matrix (default: `shift`). A `UserWarning` is always emitted with diagnostics (number of negative eigenvalues, λ_min, max/RMS off-diagonal change). See below. |
| `--debug`    | Print a full traceback on error. |

### Prior correlation matrix regularisation

The `[correlations]` block can specify values that together form a
non-positive-definite matrix (e.g. a cycle of high correlations with a sign
flip). When this happens a `UserWarning` is printed with λ_min, the number of
negative eigenvalues, and the max/RMS off-diagonal change — and the matrix is
fixed automatically using the chosen method:

| `--pd-reg-method` | What it does |
|---|---|
| `shift` **(default)** | Adds the smallest δI that makes the matrix PD, then renormalises to restore unit diagonal. Every off-diagonal entry is scaled by the same factor 1/(1+δ) — easy to communicate to collaborators. |
| `clip` | Reflects negative eigenvalues to ε via eigen-decomposition and renormalises. Change is concentrated along the problem eigenvectors. |
| `higham` | Higham (2002) alternating-projection algorithm: nearest correlation matrix in Frobenius norm. Minimises total perturbation but requires iteration. |

For the standard quadratic χ² (all-absolute systematics, Neyman scaling) the
choice of method has **no effect on combined values or uncertainties** — it
only affects nuisance pulls.

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

`CombinationResult` also carries:

- `ndf` / `chi2_per_ndf` / `p_value` — goodness-of-fit figures.
- `stat_only_covariance`, `total_syst_impact_up/down` — stat/syst covariance split.
- `impact_per_systematic`, `cov_per_systematic` — per-systematic leave-one-out breakdown (unsigned quadrature impacts and frozen covariances).
- `impact_matrix` — signed linear response matrix of shape `(nest, nsys)`: column `i` is the shift in each combined observable for a +1σ variation of nuisance `i` (ordered as `sys_names`). Derived from the post-fit Hessian cross-block `A = −inv(H_xx) @ H_xt`; satisfies `A @ A.T = total_syst_covariance` for independent nuisances. Suitable for a nuisance-parameter downstream fit `χ² = (r − A θ)ᵀ C_stat⁻¹ (r − A θ) + ‖θ‖²`.
- `pull_per_group_mean`, `pull_per_group_norm` — impact-weighted mean nuisance pull per user-defined `[uncertainty impacts]` group. Two weighting conventions: `mean` uses the per-bin average `|impact[b]|` as weight; `norm` uses the Euclidean norm. Only populated when `compute_impacts=True` and at least one group is defined.

All fields are exported via `to_dict()` / `export_npz()` / `export_json()` (`convino_jax.result`). For scanning a correlation assumption instead of a single fit, use `Combiner.scan_correlations()`.

## Tests

The suite has three parts: numeric regression tests that run `ConvinoSetups/`
combinations end-to-end and compare to stored golden snapshots
(`test/golden/*.npz`), a byte-level formatter golden (`test/golden/formatter.txt`),
and fast unit tests for the output-path handling.

```bash
python -m unittest discover -t . -s test       # fast suite (~4 s)
```

Two slow tests are skipped by default — enable them with:

```bash
CONVINO_SLOW_TESTS=1 python -m unittest discover -t . -s test
```

- `corrV2` (~25 s): full ATLAS 8+13 / CMS 13 TeV combination; central values validated once against the original C++ Convino.
- `test_pd_regularization_extreme`: runs the `Combination_ATLAS813CMS13_corrExtremeTest` setup with all three `--pd-reg-method` strategies and prints a side-by-side comparison table.

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
