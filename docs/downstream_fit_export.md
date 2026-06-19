# pyconvino export API for downstream fits

This document describes every key written by `--export npz` (or `export_npz()`) that
is relevant to a downstream mass fit, how it is derived, and concretely how to use
it in mtpole-ttj.

---

## How to produce the export file

```bash
convino ConvinoSetups/Combination_ATLAS813CMS13_corrV2/rho_config.txt \
    --prefix out/combo \
    --export npz
# → writes out/combo_result.npz
```

Or from Python:

```python
from convino_jax import Combiner, export_npz

result = Combiner.from_config("ConvinoSetups/Combination_ATLAS813CMS13_corrV2/rho_config.txt").combine()
export_npz(result, "out/combo_result.npz")
```

Load in the downstream fit:

```python
import numpy as np
npz = np.load("out/combo_result.npz", allow_pickle=False)
```

---

## Key inventory

All keys in the flat `.npz` file. Nested dicts are flattened with `__` as separator.

### Combined observables

| Key | Shape | Description |
|-----|-------|-------------|
| `combined_names` | `(nest,)` str | Observable names, in column order for all matrices below |
| `combined_values` | `(nest,)` | Best-fit combined cross-sections |
| `combined_err_up` | `(nest,)` | +1σ profile errors |
| `combined_err_down` | `(nest,)` | −1σ profile errors |
| `combined_covariance` | `(nest, nest)` | Full post-fit covariance (stat + all syst profiled). **This is the number to use as the experimental covariance in the chi2.** |
| `combined_correlation` | `(nest, nest)` | Corresponding correlation matrix |

### Goodness of fit

| Key | Shape | Description |
|-----|-------|-------------|
| `chi2_min` | scalar | Minimum χ² of the combination |
| `ndf` | scalar int | Degrees of freedom (= total input measurements − nest; nuisances excluded per Wilks) |
| `chi2_per_ndf` | scalar | NaN when ndf ≤ 0 |
| `p_value` | scalar | NaN when ndf ≤ 0 |

### Nuisance pulls

| Key | Shape | Description |
|-----|-------|-------------|
| `sys_names` | `(nsys,)` str | Nuisance parameter names, in column order for pulls and `impact_matrix` |
| `pulls` | `(nsys,)` | Best-fit nuisance values λ* (prior mean = 0, prior σ = 1) |
| `constraints` | `(nsys,)` | Post-fit nuisance σ / prior σ (< 1 means the data constrains the nuisance beyond its prior) |

### Stat / syst covariance split

| Key | Shape | Description |
|-----|-------|-------------|
| `stat_only_covariance` | `(nest, nest)` | Covariance with **all** nuisances frozen at 0 — pure statistical / measurement-only uncertainty |
| `total_syst_covariance` | `(nest, nest)` | `combined_covariance − stat_only_covariance` — total systematic contribution |
| `total_syst_impact_up` | `(nest,)` | Per-observable: `sqrt(err_up² − stat_err²)` |
| `total_syst_impact_down` | `(nest,)` | Per-observable: `sqrt(err_down² − stat_err²)` |

### Per-systematic leave-one-out breakdown

For each systematic `<name>` (614 entries for corrV2):

| Key | Shape | Description |
|-----|-------|-------------|
| `impact_per_systematic__<name>__up` | `(nest,)` | Impact of this systematic on the +1σ combined error: `sqrt(err_up² − err_frozen_i_up²)` (unsigned, quadrature) |
| `impact_per_systematic__<name>__down` | `(nest,)` | Same for −1σ |
| `cov_per_systematic__<name>` | `(nest, nest)` | Full covariance with this systematic frozen (the `cov_without[s]` in the leave-one-out recipe) |

**Leave-one-out recipe for the downstream fit:**

```python
sigma_total = fit_mass_uncertainty(combined_covariance)
for name in sys_names:
    cov_frozen = npz[f"cov_per_systematic__{name}"]
    sigma_without_s = fit_mass_uncertainty(cov_frozen)
    sigma_s = np.sqrt(max(sigma_total**2 - sigma_without_s**2, 0))
```

This is the covariance-based approach; it matches the existing PDF/scale leave-one-out
in mtpole-ttj and requires no changes to the fit code.

### User-defined impact groups

For each group `<label>` defined in `[uncertainty impacts]` in the config:

| Key | Shape | Description |
|-----|-------|-------------|
| `impact_groups__<label>__up` | `(nest,)` | Merged quadrature impact (up) |
| `impact_groups__<label>__down` | `(nest,)` | Merged quadrature impact (down) |
| `impact_cov_groups__<label>` | `(nest, nest)` | Covariance with the entire group frozen |

---

## New fields: impact_matrix and pull_per_group

### `impact_matrix` — signed linear response (nest × nsys)

```python
A = npz["impact_matrix"]   # shape (nest, nsys)
```

`A[b, i]` is the **signed shift** in `combined_values[b]` for a +1σ variation of
nuisance `i` (ordered as `sys_names`). Columns are in `sys_names` order.

**Derivation.** The post-fit Hessian `H` of the χ² has blocks

```
H = [ H_tt  H_tx ]    t = nuisances (nsys)
    [ H_xt  H_xx ]    x = combined observables (nest)
```

In the all-absolute, symmetric-response (quadratic) case:

```
H_xx = 2 C_meas_eff⁻¹                         (obs-obs block)
H_xt = −2 C_meas_eff⁻¹ A                       (obs-sys cross block)
```

so the response matrix is recovered exactly as:

```
A = −inv(H_xx) @ H_xt
```

No extra fits are needed; it is computed at negligible cost from the already-available
post-fit Hessian.

**Key identity.** For independent nuisances (unit prior covariance, which is the
standard Convino convention):

```
A @ A.T  ==  total_syst_covariance   (exact, not approximate)
```

This was verified numerically on atlas13test, atlas8, and the full corrV2 (19 × 614)
combination — diagonal and off-diagonal entries match to machine precision.

**Why it matters.** It enables a nuisance-parameter formulation of the downstream χ²
(see item C in `mtpole-ttj/PLAN_uncertainty_improvements.md`):

```
chi2 = (r − A θ)ᵀ C_stat⁻¹ (r − A θ)  +  ‖θ‖²
```

where `r = x_data − x_theory(m_t)`, `C_stat = stat_only_covariance`, and `θ` are
iminuit nuisance parameters. Minimising over θ (and m_t) gives the fully profiled
result. This is equivalent to the covariance-based approach (`C_total = C_stat + A A.T`)
in the Gaussian limit via the Woodbury identity, but makes per-nuisance contributions
and pulls directly accessible.

**Reading the matrix:**

```python
sys_names = list(npz["sys_names"])   # convert from numpy str array
A = npz["impact_matrix"]             # (nest, nsys)

# Impact of a specific nuisance on all bins
i = sys_names.index("ATLAS_13TeV_JES_category_reduction_EffectiveNP_1")
print(A[:, i])   # signed shifts per bin

# Reconstruct total syst covariance (sanity check)
assert np.allclose(A @ A.T, npz["total_syst_covariance"])
```

---

### `pull_per_group_mean` / `pull_per_group_norm` — group-level effective pulls

For each user-defined `[uncertainty impacts]` group `<label>`:

| Key | Shape | Description |
|-----|-------|-------------|
| `pull_per_group_mean__<label>` | scalar | Impact-weighted mean pull; weight = mean over bins of `|impact[b]|` |
| `pull_per_group_norm__<label>` | scalar | Impact-weighted mean pull; weight = Euclidean norm of the impact vector |

**Formula:**

```
w_i  = mean_b(|impact_per_systematic[i][up][b]|)   # "mean" convention
     = norm_b( impact_per_systematic[i][up]    )   # "norm" convention

pull_g = Σ_{i ∈ g} w_i * pull_i  /  Σ_{i ∈ g} w_i
```

where the sum is over individual nuisances belonging to group g.

**Purpose.** This enables the group-level mass-shift estimate (item B in
`mtpole-ttj/PLAN_uncertainty_improvements.md`):

```
delta_m_g = (w_abs · v_g) * pull_g   [MeV]
```

where `v_g` is the group's impact on the combined cross-sections (in physical units,
from `impact_groups__<label>`), and `w_abs` is the mass sensitivity vector (gradient
of m_t w.r.t. the combined observables).

**Note on current setups.** All `ConvinoSetups` have `pull_per_group = 0` for every
group. This is physically expected: in the Convino model each combined observable is a
free parameter that absorbs all measurement tension, so nuisances are always pulled
to their prior mean (λ* = 0). Non-zero group pulls would appear when the same
systematic is shared across measurements with conflicting responses — a configuration
not present in the current ATLAS+CMS combination configs.

---

## Complete usage example for mtpole-ttj

```python
import numpy as np

# Load the combination result
npz = np.load("out/combo_result.npz", allow_pickle=False)

combined_values  = npz["combined_values"]        # (19,)   pb
combined_cov     = npz["combined_covariance"]    # (19,19) pb²
stat_cov         = npz["stat_only_covariance"]   # (19,19)
A                = npz["impact_matrix"]          # (19, 614)
sys_names        = list(npz["sys_names"])        # 614 names

# --- Option 1: covariance-based (existing approach) ---
# Feed combined_cov into chi2 as the experimental covariance.
# For the leave-one-out breakdown of the fitted mass uncertainty:
sigma_total = mass_fit_uncertainty(combined_cov)
for name in sys_names:
    cov_without = npz[f"cov_per_systematic__{name}"]
    sigma_s = np.sqrt(max(sigma_total**2 - mass_fit_uncertainty(cov_without)**2, 0))

# --- Option 2: nuisance-parameter (new, item C) ---
# Add 614 theta parameters to iminuit. chi2 term:
#   (r - A @ theta).T @ inv(stat_cov) @ (r - A @ theta)  +  theta @ theta
# where r = data - theory(m_t, PDF_vars).
# Sanity: should give identical mass central value and total uncertainty.

# --- Group-level mass shifts (item B, once pulls are non-zero) ---
for label in [k.replace("pull_per_group_mean__", "")
              for k in npz.files if k.startswith("pull_per_group_mean__")]:
    pull_g   = float(npz[f"pull_per_group_mean__{label}"])
    impact_g = npz[f"impact_groups__{label}__up"]   # (19,) pb
    # delta_m_g = mass_sensitivity_vector @ impact_g * pull_g
```
