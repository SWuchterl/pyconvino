# pyconvino — Package & Output Reference

This document is a self-contained reference for what the package does and what
every output it produces contains. It is written so that a new session (or a
downstream script) can understand, parse, and use those outputs without having
to re-read the source.

---

## What the package does

`pyconvino` is a Python/JAX port of
[Convino](https://github.com/jkiesele/Convino): a χ² combination tool for
physics measurements with correlated systematic uncertainties. It takes as
input a set of measurement files (each describing one experiment's estimates,
their Hessian / correlation matrix, and their systematic uncertainties) and a
single config file that ties them together. It then:

1. Parses all measurement and config files into internal dataclasses.
2. Builds the global parameter vector: `pars = [λ₁…λ_nsys, x_comb₁…x_comb_nest]`
   (nuisance pulls first, then combined observable values).
3. Constructs the prior inverse-covariance matrix from the user-specified
   cross-measurement correlation assumptions.
4. Builds and JIT-compiles the JAX χ² objective.
5. Minimises the χ²:
   - **Quadratic fast path** (symmetric responses, all-absolute systematics,
     Neyman χ²): one linear solve — constant Hessian, exact in one step.
   - **General path** (relative systematics or Pearson χ²): two-pass
     L-BFGS-B + a guarded Newton-polish step.
6. Extracts asymmetric errors via profile likelihood (MINOS equivalent).
7. Computes the post-fit covariance from the Hessian at the minimum.
8. Optionally computes uncertainty impacts by refitting with systematic groups
   frozen (using the Hessian submatrix method, not a full refit per group).
9. Returns a `CombinationResult` and writes output files.

**Dependencies:** JAX, NumPy, SciPy. No ROOT, no Minuit.

---

## Input format (summary)

### Measurement files

Plain-text files with named blocks:

| Block | Content |
|-------|---------|
| `[estimates]` | Central values of the observables in this measurement |
| `[systematics]` | One line per systematic: `name  type` where type ∈ `absolute`, `relative`, `lognormal`*, or `free` (a Hessian parameter the input fit did not constrain with a prior, e.g. a floating normalisation) |
| `[hessian]` | Lower-triangular Hessian matrix (optional constraint column) |
| `[correlation matrix]` | Alternative to hessian: lower-triangular correlation matrix + constraint column |
| `[not fitted]` | Externalised (non-fitted) uncertainties table |
| `[nuisance values]` | `name = value`: post-fit central values (pulls) of the Hessian nuisances from the input fit. Absent → 0, which is the original Convino approximation; see `mathematical_extensions.md` §11 |

\* `lognormal` is parsed but raises `NotImplementedError` at chi2-build time —
it was never finished in the C++ reference either.

### Config file (`rho_config.txt`)

| Block | Content |
|-------|---------|
| `[inputs]` | List of measurement-file paths |
| `[observables]` | `estimate_name = combined_name` — maps measurement estimates to combined observable names |
| `[correlations]` | `sys_name  nominal_rho  (& low : high)?` — cross-measurement correlation per systematic, optionally with a scan range |
| `[uncertainty impacts]` | Named groups of systematics for merged impact reporting |
| `[global]` | `isDifferential`, `normalise` flags — when both are `true`, the combined bin values are renormalised to fractions of their sum after the fit (see "Differential normalisation" below) |
| `#!FILE = path` | Include directive inside `[correlations]` |

---

## Running

```bash
# Minimal — writes convino_result.txt
pyconvino path/to/rho_config.txt

# Full run with all outputs
pyconvino path/to/rho_config.txt \
    --prefix out/myrun \
    --export both \
    --scan \
    --scan-steps 6 \
    --verbose
```

CLI flags:

| Flag | Effect |
|------|--------|
| `--prefix P` | All output files are named `P_result.txt`, `P_result.npz`, etc. |
| `--export {npz,json,both}` | Write machine-readable output(s) alongside the text file |
| `--no-impacts` | Skip all impact computation (fastest; only combined values + covariance) |
| `--impacts-only G1,G2` | Only compute listed `[uncertainty impacts]` groups; per-systematic breakdown unaffected |
| `--scan` | Scan each `[correlations]` entry with a `& low : high` range |
| `--scan-steps N` | Points per scan group (default 6) |
| `--verbose` | Per-phase timing to stderr |
| `--pd-reg-method {shift,clip,higham}` | How to fix a non-positive-definite prior correlation matrix (default `shift`; see below) |
| `--nonneg-combined` | Constrain combined observables to be ≥ 0 (off by default; see below) |
| `--pearson` | Pearson χ² instead of Neyman |
| `--debug` | Full traceback on error |
| `--version` | Print the installed version and exit |

### Prior correlation matrix regularisation (`--pd-reg-method`)

When the `[correlations]` block specifies values that together form a
non-positive-definite (NPD) matrix (e.g. `ρ(A,B)=0.9`, `ρ(B,C)=0.9`,
`ρ(A,C)=-0.9` — a cycle of high correlations with a sign flip), the prior
must be regularised before it can be inverted.  A `UserWarning` is always
emitted when this happens, reporting:

- the number of negative eigenvalues and the most negative value (λ_min),
- what regularisation was applied,
- the maximum and RMS off-diagonal change (max |Δρ|, RMS |Δρ|).

Three methods are available:

| `--pd-reg-method` | What it does | Off-diagonal change | Interpretation |
|---|---|---|---|
| `shift` **(default)** | Adds δI (smallest δ making λ_min ≥ eps), then renormalises rows/columns to restore unit diagonal. | Uniform: every ρᵢⱼ scaled by the same factor 1/(1+δ). | Clearest to explain: "all correlations reduced to X% because they were mutually inconsistent." |
| `clip` | Reflects negative eigenvalues to eps via eigen-decomposition, then renormalises. | Concentrated in the problem eigenvector directions; zero for uncorrelated pairs. | Natural from the spectral perspective; changes are smaller on average than `shift` but less uniform. |
| `higham` | Higham (2002) alternating-projection algorithm: nearest correlation matrix in Frobenius norm. | Minimises the total (sum-of-squares) change, but individual pairs can change more than with other methods. | Mathematically optimal perturbation, at the cost of an iterative solve and less predictable per-pair changes. |

**Physical impact:** for the standard quadratic χ² (all-absolute systematics,
Neyman scaling), the regularisation method has *no effect on the combined
values or uncertainties* — it only affects the nuisance pulls.  The combined
observables depend on the data and their covariance, not on the systematic
prior; the prior only constrains how far the nuisance parameters are pulled.
`shift` is the default because its uniform scaling is the easiest to
communicate to collaborators; `higham` is available when a minimum-perturbation
argument is needed.

### Non-negative combined observables (`--nonneg-combined`)

Off by default. When set, the `nest` combined-observable parameters are
floored at 0 in the minimiser (the `nsys` nuisance parameters stay unbounded —
Gaussian priors have no physical floor). Implementation notes:

- The quadratic fast path (an unconstrained one-shot linear solve, see
  "What the package does" above) has no notion of bounds: if its solution
  would violate the floor, it is discarded and the general bounded
  two-pass L-BFGS-B path is used instead for that fit.
- The Newton-polish step layered on top of L-BFGS-B is likewise an
  unconstrained linear solve, so it is only accepted when it still respects
  the bound.
- Has no effect on a setup whose unconstrained minimum is already
  non-negative (the common case).

### Differential normalisation (`[global] isDifferential` / `normalise`)

When a config's `[global]` block sets both `isDifferential = true` and
`normalise = true`, `combine()`'s final step renormalises the combined bin
values to fractions of their sum (Monte Carlo error propagation: draw
`N(combined_vals, cov)`, divide each sample by its own sum, take the empirical
covariance of the deviations from the nominal fraction — ports the C++
`normaliser.cpp` algorithm). After this step, `combined_values`,
`combined_err_up`/`down`, and the combined-observable block of
`cov_full`/`corr_full` describe shape (fractions summing to 1), not the
original absolute bin values. `CombinationResult.normalised` (also exported)
is `True` when this ran. No existing `ConvinoSetups` fixture sets these flags,
so this is a strict no-op for all of them.

---

## Output files

For a run with `--prefix out/myrun --export both --scan`:

```
out/myrun_result.txt          # human-readable combination result
out/myrun_result.npz          # machine-readable result (NumPy compressed)
out/myrun_result.json         # machine-readable result (JSON)
out/myrun_scan_result.txt     # human-readable scan log (one block per step)
out/myrun_scan_result.npz     # machine-readable scan results
out/myrun_scan_result.json    # machine-readable scan results
```

---

## `_result.txt` — human-readable text output

Sections appear in this order. Section delimiters are `[name]` / `[end name]`
with a blank line after each closing tag.

### 1. `[pre-combine systematics correlations]`
nsys × nsys matrix. The input correlation assumptions between systematics as
configured in `[correlations]`. Printed with the C++ `triangularMatrix`
scaler/layout (a `multiplied by: X` header appears when the dynamic range is
extreme).

### 2. `[post-combine systematics correlations]`
nsys × nsys block of `corr_full` (the top-left submatrix) — post-fit
correlations among the nuisance parameters.

### 3. `[pre-combine estimate correlations]`
Always empty in this implementation (pre-fit per-measurement correlations are
not available). Section printed with empty body for C++ compatibility.

### 4. `[post-combine result correlations]`
nest × nest submatrix of `corr_full` (bottom-right) — post-fit correlations
among the combined observables.

### 5. `[post-combine result covariance]`
nest × nest covariance of the combined observables, derived from
`corr_full[nsys:, nsys:]` × outer(`sym_err_up`, `sym_err_down`) where
`sym_err = max(|err_up|, |err_down|)`.

### 6. `[combined (minimum chi^2=X)]`
One line per combined observable:
```
name: value +err_up -err_down
```
Followed immediately (if impact groups exist) by:
```
[impact table: name, impact [%]]
  header row
  one row per group: label | rel_impact_obs1 | rel_impact_obs2 | ...
[end impact table]
```
Relative impact = `|abs_impact / combined_value| × 100`. Only the upward
impact is shown in this table.

### 6b. `[goodness of fit]` *(pyconvino extension — no C++ equivalent)*
```
ndf: <int>
chi2/ndf: <float>
p-value: <float>
```
`ndf = (total number of individual input measurements) − nest`. Nuisance
parameters contribute net zero dof (standard Gaussian-constrained treatment).
`chi2/ndf` and `p-value` are `nan` when `ndf ≤ 0` (all real ConvinoSetups
fixtures have `ndf = 0` because each combined observable has exactly one
measurement).

### 7. `[full correlation matrix]`
`(nsys + nest) × (nsys + nest)` matrix — all parameters (systematics first,
then combined observables). Row/column names: `sys_names + combined_names`.

### 8. `[full covariance matrix]`
Same shape as section 7, in covariance units.

### 9. `[nuisance pulls]`
Table with columns `Name`, `pull`, `constraint`:
- `pull`: post-fit nuisance parameter value (λᵢ; 0 = no pull from prior)
- `constraint`: width of the posterior constraint on λᵢ (< 1 = tighter than
  the prior; 1 = unconstrained by the combination)

### 10. `[simple impacts]` → `[simple impact table: name, impact [%]]`
Table mapping each systematic to each combined observable. For systematic `i`
and observable `j`:
```
impact[i,j]   = corr_full[i, nsys+j] × max(|err_up[j]|, |err_down[j]|)
rel_impact[i,j] = |impact / combined_value[j]| × 100  (%)
```

### 11. `[merged impacts]` *(printed when impact groups exist)*
Same format as the table inside section 6, but as a stand-alone section.
Listed as a superset of the C++ reference (the reference's `printFullInfo`
does not populate `impacttable_` for this run type and so omits this section —
kept deliberately in pyconvino).

### 12. `[covariance matrix for merged impacts]` *(pyconvino extension)*
One nest × nest covariance block per impact group, formatted as:
```
[covariance matrix for merged impacts]
[covariance matrix for impact table]
[GroupLabel]
  <matrix rows>
[end GroupLabel]
...
[end covariance matrix for merged impacts]
```
Each block is the covariance of the combined observables when that group's
systematics are frozen — i.e. the diagonal entries are the "frozen error²"
for each observable. No C++ equivalent.

---

## `_result.npz` / `_result.json` — machine-readable result

Both formats carry the same data. `.npz` uses `np.savez_compressed` with
flat keys joined by `__` (e.g. `impact_groups__GroupA__up`). `.json` is
nested with NumPy arrays serialised as nested lists.

Load `.npz`:
```python
import numpy as np
d = np.load("myrun_result.npz", allow_pickle=False)
# access e.g.: d["combined_values"], d["impact_groups__StatOnly__up"]
```

Load `.json`:
```python
import json, numpy as np
with open("myrun_result.json") as f:
    d = json.load(f)
# d is a nested dict; arrays are Python lists — wrap with np.array() as needed.
```

### Top-level fields

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `combined_names` | list[str] | `(nest,)` | Names of the combined observables |
| `combined_values` | float64 | `(nest,)` | Best-fit combined observable values |
| `combined_err_up` | float64 | `(nest,)` | Upward asymmetric errors (positive) |
| `combined_err_down` | float64 | `(nest,)` | Downward asymmetric errors (positive magnitude) |
| `combined_covariance` | float64 | `(nest, nest)` | Full post-fit covariance of combined observables (stat + all syst) — the `nest×nest` bottom-right block of `cov_full`. Use this as the total covariance matrix in a downstream fit. |
| `combined_correlation` | float64 | `(nest, nest)` | Correlation matrix corresponding to `combined_covariance` |
| `chi2_min` | float | scalar | Minimum χ² value |
| `ndf` | int | scalar | Degrees of freedom (see section 6b above) |
| `chi2_per_ndf` | float | scalar | `chi2_min / ndf`; `nan` when `ndf ≤ 0` |
| `p_value` | float | scalar | p-value of the combination; `nan` when `ndf ≤ 0` |
| `converged` | bool | scalar | Whether the minimiser converged |
| `sys_names` | list[str] | `(nsys,)` | Names of all systematic nuisance parameters |
| `pulls` | float64 | `(nsys,)` | Post-fit nuisance-parameter values (pulls) |
| `constraints` | float64 | `(nsys,)` | Post-fit constraint widths on each nuisance |
| `stat_only_covariance` | float64 | `(nest, nest)` | Covariance with **all** systematics frozen at 0 — pure statistical uncertainty. Filled with `nan` when `--no-impacts` is used. |
| `total_syst_covariance` | float64 | `(nest, nest)` | `combined_covariance − stat_only_covariance`. **Do not** add this to `combined_covariance` — that would double-count. |
| `total_syst_impact_up` | float64 | `(nest,)` | Total systematic impact upward per observable (quadrature: `sqrt(err_up² − stat_err_up²)`) |
| `total_syst_impact_down` | float64 | `(nest,)` | Total systematic impact downward per observable |
| `normalised` | bool | scalar | Whether differential bin-renormalisation ran (see "Differential normalisation" above); `False` unless the config sets both `isDifferential` and `normalise` |

### Nested: `impact_groups`

One entry per named `[uncertainty impacts]` group from the config.

| Key | Shape | Description |
|-----|-------|-------------|
| `impact_groups/<label>/up` | `(nest,)` | Absolute upward impact of this group on each combined observable |
| `impact_groups/<label>/down` | `(nest,)` | Absolute downward impact |

In `.npz`: `impact_groups__<label>__up`, `impact_groups__<label>__down`.

### Nested: `impact_cov_groups`

| Key | Shape | Description |
|-----|-------|-------------|
| `impact_cov_groups/<label>` | `(nest, nest)` | Covariance of combined observables with this group frozen — diagonal is the "frozen error²". |

In `.npz`: `impact_cov_groups__<label>`.

### Nested: `impact_per_systematic`

Leave-one-out breakdown per **individual** systematic (independent of any
user-defined groups). Skipped fields are `nan`-filled at the correct shape
when `--no-impacts` is used.

| Key | Shape | Description |
|-----|-------|-------------|
| `impact_per_systematic/<sys_name>/up` | `(nest,)` | Absolute upward impact of this one systematic |
| `impact_per_systematic/<sys_name>/down` | `(nest,)` | Absolute downward impact |

In `.npz`: `impact_per_systematic__<sys_name>__up`, `…__down`.

### Nested: `cov_per_systematic`

| Key | Shape | Description |
|-----|-------|-------------|
| `cov_per_systematic/<sys_name>` | `(nest, nest)` | Covariance with this one systematic frozen |

In `.npz`: `cov_per_systematic__<sys_name>`.

---

## `_scan_result.txt` — human-readable scan log

Concatenated `format_result()` blocks, one per scan step per group, each
preceded by a header line:
```
=== scan group '<name>': step N/M, correlation=<val> ===
<full result block>
```
No C++ byte-fidelity target exists for this file.

---

## `_scan_result.npz` / `_scan_result.json` — machine-readable scan

Structure: `{group_name: {"scan_values": [...], "steps": [to_dict(), ...]}}`.

Each step's dict has **all the same fields** as a single `_result.*` export
(with `impact_groups`/`cov_per_systematic` etc.). The motivation is to
replace the C++ reference's ROOT `TGraphAsymmErrors` (`-p`) plots: plot
`scan_values` on the x-axis against `steps[i]["combined_values"]` or
`steps[i]["combined_err_up"]` on the y-axis.

In `.npz` flat keys: `<group>__scan_values`, `<group>__step0__combined_values`, etc.

---

## Python library API (quick reference)

```python
from pyconvino import Combiner, to_dict, export_npz, export_json
from pyconvino.result import scan_results_to_dict, export_scan_npz

# Single combination
result = Combiner.from_config("rho_config.txt",
    compute_impacts=True,   # set False for --no-impacts behaviour
    verbose=True,
    pd_reg_method="shift",  # "clip" or "higham" also available
).combine()

d = to_dict(result)                 # nested dict, same schema as JSON above
export_npz(result, "out.npz")       # compressed .npz
export_json(result, "out.json")     # JSON

# Correlation scan
combiner = Combiner.from_config("rho_config.txt")
combiner.combine()                  # must call combine() first
scan = combiner.scan_correlations(n_steps=6)
# scan: {group_name: (scan_values_array, [CombinationResult, ...])}

export_scan_npz(scan, "out_scan.npz")
export_scan_json(scan, "out_scan.json")
sd = scan_results_to_dict(scan)    # nested dict
```

### Key types

- `Combiner.from_config(path, *, use_pearson, prefix, compute_impacts, impacts_only, verbose, pd_reg_method, nonneg_combined)`
- `Combiner.combine() → CombinationResult`
- `Combiner.scan_correlations(n_steps) → {group: (scan_values, [CombinationResult, ...])}`
- `CombinationResult` — dataclass, all fields described above
- `to_dict(result) → dict` — nested; arrays are `np.ndarray`
- `export_npz(result, path)` / `export_json(result, path)`
- `format_result(result) → str` — the full text as a string (no file I/O)
- `write_result(result, prefix)` — writes `<prefix>_result.txt`

---

## Notes for a downstream fit

The intended downstream use case is a χ² fit (e.g. a top-quark-mass fit) that
needs:

1. **Combined central values**: `combined_values`
2. **Total experimental covariance**: `combined_covariance`  
   (already includes stat + all systematics — do *not* add anything to it)
3. **Stat-only covariance**: `stat_only_covariance`
4. **Per-systematic covariance contribution**: `cov_per_systematic[name]`  
   (each entry is the covariance of the combined observables with that
   systematic frozen; `combined_covariance − cov_per_systematic[name]` is the
   contribution of that systematic alone)
5. **User-defined group covariance**: `impact_cov_groups[label]`  
   (same frozen-covariance concept, but for groups of systematics defined in
   `[uncertainty impacts]` of the config)

The per-systematic/per-group numbers do not generally sum in quadrature to the
total if systematics are correlated. They are leave-one-out frozen covariances,
not orthogonal contributions.
