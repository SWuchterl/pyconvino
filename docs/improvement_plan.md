# Convino_jax: assumptions, gaps, CLI, and downstream-fit support

_Written 2026-06-16. Analysis/design only — no code changes applied yet. See
`[[improvement-roadmap]]` / `[[downstream-fit-export-design]]` in Claude's memory
for status tracking across sessions._

## Context

`pyconvino` is validated against the original C++ Convino (exact central values,
~0.05% on post-fit errors) and has a regression-test safety net (see
`test/golden/` and the project's `regression-tests` memory). With correctness
established, the question is what to improve next. There is a concrete downstream
use case: feed the combined cross section(s) + full covariance into a top-quark-mass
fit (chi2 against a theory model, 1D parameter extraction), with an existing fit
pipeline that already treats other theory uncertainties (PDF, scale) as individual
nuisances whose own covariance matrix is added into the total experimental
covariance, then decomposed into a per-source breakdown the same way. This plan
answers four questions and ends with a concrete design for the gap that blocks the
downstream use case today — there is currently no machine-readable export of the
combined result, and no per-individual-systematic covariance breakdown (only
user-defined merged groups).

---

## Q1 — Statistical/mathematical assumptions currently baked in

- **Gaussian everywhere.** Measurements and nuisance priors are both Gaussian
  (chi-squared, not a general likelihood). No non-Gaussian uncertainty models, no
  bootstrap/toy-MC validation. (`objective.py:123-130`)
- **Neyman vs Pearson chi2** is the only model choice exposed (`--pearson`),
  switching how the observable block of the Hessian is scaled (`objective.py:114-123`).
- **Symmetric systematics ⇒ exactly quadratic chi2.** The code detects
  `Lk_up == -Lk_down` (`combiner.py:185-188`) and then uses closed-form HESSE errors
  instead of profiling — this is provably equivalent to MINOS for a quadratic chi2 and
  is why the fast path is also exact, not an approximation (`combiner.py:166-188`).
- **Asymmetric systematics are a linear kink at λ=0** (`Lk_up·λ` for λ≥0,
  `Lk_down·|λ|` for λ<0 — `objective.py:71-75`), not a smooth interpolation. This
  matches the C++ convention but means the chi2 has a discontinuous second derivative
  at λ=0; profile-likelihood (`brentq`) is required there and is already implemented.
- **Relative (multiplicative) systematics are nonlinear even when symmetric**
  (`rel_factor = 1 - k/x`, multiplied across systematics — `objective.py:84-89`).
  This is already flagged in the improvement roadmap as the reason the planned B2
  quadratic fast-path must gate on "all-absolute", not just "symmetric".
- **`lognormal` systematic tag is parsed but not implemented (resolved, fails loudly).**
  `parser.py` accepts `"lognormal"` as a systematic type, but `objective.py` only
  implements `"absolute"`/`"relative"`. This used to be a silent correctness gap
  (lognormal-tagged systematics were silently treated as absolute); `make_chi2`
  (`objective.py:65-80`) now raises `NotImplementedError` for any other tag instead,
  so a user who writes `lognormal` gets an immediate, explicit error rather than a
  wrong model with no warning. The model itself remains unimplemented (it was never
  finished in the original C++ Convino either) — only the silent-mis-modeling risk
  is closed.
- **Correlation scans are parsed but ignored.** `"(nom & low:high)"` syntax is fully
  parsed (`ScanRange`/`CorrelationScan` in `parser.py`) but only `nominal` is used
  (`combiner.py:314-320`). This is the C++ tool's `-s` option, unimplemented here.
- **No physical/boundary constraints (resolved, opt-in).** All parameters were
  unbounded in L-BFGS-B; nothing prevented a non-physical (e.g. negative) combined
  value if the data/systematics pulled it there. `--nonneg-combined` /
  `Combiner(..., nonneg_combined=True)` now floors the `nest` combined-observable
  parameters at 0 (nuisances stay unbounded); off by default since not every
  combined quantity is a non-negative cross section. The quadratic fast path's
  unconstrained one-shot solve and the general path's Newton-polish step both fall
  back to the bounded L-BFGS-B result when the bound would otherwise be violated.
- **PD-regularization is now warned (resolved).** `_nearest_positive_definite`
  (`combiner.py:817-843`) reflects negative eigenvalues of the prior correlation matrix
  to a small positive epsilon, and now `warnings.warn`s when this actually changes the
  matrix (eigenvalue genuinely below `-eps`, not float noise on an already-PD matrix) —
  the user is told their input correlation matrix was not positive-definite and got
  regularized, instead of this happening silently.
- **No goodness-of-fit output.** Only `chi2_min` is reported; no NDF, reduced chi2, or
  p-value (`result.py:306-313`). For a combination with hundreds of nuisance
  parameters, NDF is not simply `n_measurements - n_fitted` in the usual sense (most
  nuisances are profiled, not "fit" in the reduce-NDF sense) — this needs a defined
  convention before being added, not just a one-line computation.

## Q2 — Missing features (beyond what's already tracked)

**Done (closed 2026-06-16):**
- B2 (quadratic fast-path) and B3 (`value_and_grad` fusion) — `Combiner._minimize` now
  does an exact one-step linear solve for the symmetric/all-absolute/non-Pearson case,
  with a guarded Newton-polish + fused `fun`/`jac` for the general case. corrV2 full
  slow suite ~80s → ~22s.
- Formatter fidelity — decided to keep the extra `[impact table]`/`[merged impacts]`/
  `[covariance matrix for merged impacts]` sections as an intentional superset of the
  C++ reference rather than gating them; documented inline in `result.py`.
- PD-regularization warning — `_nearest_positive_definite` (combiner.py) now
  `warnings.warn`s when the prior correlation matrix needed regularizing.
- JSON/npz export — `CombinationResult.to_dict()` / `.export_npz()` / `.export_json()`
  plus `cli.py --export {npz,json,both}`.
- **Automatic statistical-vs-systematic decomposition** — `CombinationResult` gained
  `stat_only_covariance` and `total_syst_impact_up/down`, computed via the same
  frozen-Hessian-submatrix mechanism as user-defined impact groups, no manual
  "freeze everything" group needed anymore.
- **Public export API** — same `to_dict()`/`export_npz()`/`export_json()` as above;
  usable directly from Python (`Combiner.from_config(...).combine().to_dict()`), not
  just via the CLI flag.
- **`lognormal` systematics** — resolved via the doc's own fallback option: rather than
  implementing the (never-finished-in-C++-either) lognormal model, `objective.make_chi2`
  now raises `NotImplementedError` for any non-`absolute`/`relative` systematic type, so
  a `lognormal` tag fails loudly instead of being silently mis-modeled as `absolute`.

**Still open** (tracked in the project's improvement-roadmap memory): `--verbose`/timing
(in progress, uncommitted — see roadmap memory for exact resume point), pull/
impact-ranking plots, parallelized asymmetric impact refits, persistent JAX compile
cache, GPU note. (Correlation-scan `-s`, chi2/ndf+p-value, and `--no-impacts`/
`--impacts-only` are done — see roadmap memory.)

## Q3 — CLI usability

Today: 1 positional (`config`) + 3 flags (`--prefix`, `--pearson`, `--debug`) —
`cli.py:9-59`. Gaps relative to a typical scientific CLI:
- No `--verbose`/logging control (only on/off traceback via `--debug`).
- No way to skip impact computation (`--no-impacts`) when the user only wants the
  combined values quickly — impacts are the dominant cost even after the B1 speedup.
- No selective impact groups (`--impacts-only GROUP1,GROUP2`).
- No machine-readable export flag (the Q4 gap).
- ~~No `--version`.~~ Resolved: prints the installed package version (from
  `pyproject.toml` via `importlib.metadata`) and exits.
- Correlation-scan `-s` doesn't exist because the underlying feature isn't implemented
  (Q1/Q2).

No `--seed`/parallelism/device flags are needed yet — the fit is deterministic and JAX
device selection is an environment concern (`JAX_PLATFORM_NAME`), not a per-run CLI
concern; not worth adding until there's an actual multi-device use case.

## Q4 — Downstream top-mass fit: what output is needed, and how

**The gap:** `CombinationResult` already contains everything needed
(`combined_values`, `cov_full` — the post-fit covariance includes the `nest×nest`
estimate block, which is the marginal covariance over all profiled nuisance
parameters, exactly what a downstream χ² fit needs — and `impact_cov_groups`, the
per-group frozen covariance already computed for the impact tables). None of it is
exported in a form another program can load; it only exists inside the text file
formatter (`result.py`).

**Design for the export (NPZ + JSON, both):**

1. Add `CombinationResult.to_dict()` in `result.py` (or `combiner.py` next to the
   dataclass) returning a flat, self-describing dict:
   - `combined_names`, `combined_values`, `combined_err_up`, `combined_err_down`
   - `combined_covariance` = `cov_full[nsys:, nsys:]` (the slice the downstream fit
     actually needs), `combined_correlation` = `corr_full[nsys:, nsys:]`
   - `chi2_min`, `converged`
   - `impact_groups` (per-label up/down impacts) and `impact_cov_groups` (per-label
     `nest×nest` frozen covariance) — already computed, just surfaced
   - the new stat/syst split (next section): `stat_only_covariance`,
     `total_syst_covariance` (= `combined_covariance - stat_only_covariance`, or
     reported the same quadrature way as impacts, see below)
2. Add `export_npz(result, path)` (→ `np.savez_compressed`, mirrors
   `test/helpers.py:result_to_arrays`/`regen_golden.py` exactly, so it can reuse that
   helper rather than duplicating field lists) and `export_json(result, path)` (→
   nested dict via `to_dict()`, `np.ndarray` → `.tolist()`).
3. Add a CLI flag, e.g. `--export {npz,json,both}` (default: none, opt-in so existing
   behavior/perf is unchanged), writing `{prefix}_result.npz` / `{prefix}_result.json`
   alongside the existing text output. Wire it in `cli.py` next to `write_result`.

**Design for the automatic stat/syst split:**

Reuse the exact frozen-Hessian-submatrix mechanism already in `_compute_impacts`
(`combiner.py:506-631`), but with `frozen_idx = range(nsys)` (every systematic at
once) instead of one group's members. This gives, for free:
- `stat_only_covariance` (nest×nest): the combined-observable covariance with every
  systematic frozen — i.e. the pure statistical/measurement-only uncertainty.
- `total_syst_impact = sqrt(combined_err^2 - stat_only_err^2)` per observable, same
  quadrature convention already used for per-group impacts, so it's consistent with
  the existing impact tables (a user comparing "stat ⊕ each group ⊕ total syst" sees
  numbers computed the same way).

This is a small, additive change: one extra call into the existing frozen-fit
machinery with `frozen_idx` = all systematics, computed once per `combine()` call
(cheap — it's a single Hessian-submatrix slice + inverse, no re-fit needed for the
symmetric case, same cost profile as one impact group).

### Q4b — fitting into an existing PDF/scale-style covariance workflow

If the downstream mass fit already treats other theory uncertainties (PDF, scale) as
individual nuisances whose **own covariance matrix gets added** into the
experimental covariance, with the per-source breakdown coming out of that same
machinery, the recommendation is: don't give Convino's systematics a different
(nuisance-parameter / response-matrix) treatment than PDF/scale get — give them the
**same kind of object** (a named covariance matrix per source) so they drop into the
identical code path already used downstream, with zero new logic on that side.

**What "easiest" means concretely, and why it's nearly free to add:**

`combiner.py`'s `_compute_impacts` already computes, for each *user-defined* impact
group, a frozen covariance `cov_without_group` (group's systematics pinned to 0,
rest re-optimized) via a Hessian-submatrix slice — no refit, just a slice + inverse
of the already-computed post-fit Hessian (`combiner.py:568-619`). The only change
needed is to **also run that same loop with every individual systematic as its own
one-member group**, not just the user's merged groups, plus a "Stat" pseudo-group
that freezes *all* systematics at once (the stat/syst split above). No new
mathematics — it's the existing, already-validated mechanism applied to more
(trivial, single-element) groups.

**Export, per source `s` (every individual systematic name, `"Stat"`, `"TotalSyst"`,
and any existing user-defined groups):**

- `combined_covariance` (nest×nest) — the **total** experimental covariance. This is
  the one number that plays the same role as a PDF/scale total: it's what gets summed
  with `Cov_PDF + Cov_scale + ...` to get the full covariance for the central mass
  fit. **Do not also sum the per-source matrices below into that total — they would
  double-count, since `combined_covariance` already contains all of stat+syst.**
- `cov_without[s]` (nest×nest) — the same combined covariance with source `s` frozen
  out. This is the only other matrix needed.

**The breakdown recipe — identical in shape to a PDF/scale leave-one-out:**

1. Fit `m_top` once with `combined_covariance` (+ other covariance terms) →
   `sigma_m,total`.
2. For each source `s`, refit with `cov_without[s]` substituted for
   `combined_covariance` (everything else unchanged) → `sigma_m,without[s]`.
3. `sigma_m,s = sqrt(sigma_m,total^2 - sigma_m,without[s]^2)` — the same quadrature
   step already used for PDF eigenvectors/scale variations.
4. `"Stat"` and `"TotalSyst"` come out of the same loop for free (using the all-
   systematics-frozen and all-systematics-floating cases respectively) — a 3-tier
   stat/syst/per-source breakdown without any bespoke code.

If `nest == 1` (one combined cross-section number feeding the mass fit, the common
case), every matrix above is just a scalar variance, and step 2 reduces to "redo the
1-parameter fit with one number changed" — about as easy as this gets.

**The one honest caveat (linear approximation, same spirit as the rest of the
pipeline):** if two individual systematics are correlated with each other (via the
`[correlations]` block), `sigma_m,s` for each separately and `sigma_m,"TotalSyst"`
will not satisfy `Σ_s sigma_m,s^2 = sigma_m,TotalSyst^2` exactly — the quadrature
method ignores cross terms between sources, exactly like the existing merged-impact
tables already do, and exactly like most published ATLAS/CMS breakdowns of this kind.
If named systematics that *do* sum exactly are wanted, the only way is to group
everything that's mutually correlated into one named source first (Convino's
existing `[uncertainty impacts]` groups are exactly for this) — lean on that rather
than inventing new math, since it's already implemented and validated.

### Q4c — could the systematics be exported as explicit nuisance parameters instead?

(Kept for completeness / future reference — not the recommended path given Q4b.)

**Yes, and it's a legitimate alternative to the covariance method, not just a nicety.**
HEP combination practice has two ways to hand correlated systematics to a downstream
fit:

1. **Covariance method** (Q4 design above): collapse all nuisances into one
   `combined_covariance`, fit `theory(m_top)` against it directly. Simple, always
   available, but it's a *linearized, frozen* summary — it bakes in the assumption
   that the combination's nuisance parameters stay at their post-fit values
   regardless of what the downstream fit does.
2. **Nuisance-parameter method**: re-expose each systematic as an explicit floating
   parameter `λ_i` (unit-Gaussian-ish prior, correlated via the prior matrix) with a
   *response* describing how the combined value(s) shift per unit `λ_i`. The
   downstream fit profiles `λ_i` jointly with `m_top`. This is strictly more capable
   when:
   - the same physical systematic (e.g. luminosity, a JES source) is **shared**
     with another input to the downstream fit (the theory prediction, another
     measurement) — only the nuisance-parameter form lets the downstream fit
     re-correlate them; a covariance matrix cannot express "this number and that
     other analysis's number move together because of a shared source."
   - per-nuisance pull/impact diagnostics on the *extracted mass itself* are wanted
     (a standard HEP deliverable: "impact of JES on m_top"), not just on the
     intermediate combined cross section.
   - the combination's response to a nuisance is poorly summarized by one
     symmetric number (asymmetric systematics) and the symmetrization shouldn't be
     baked into a covariance entry before it reaches the mass fit.

   It costs: it's still a **linear (first-order Taylor) model** around the
   combination's best fit unless a literal joint re-fit is done (see the "exact"
   option below) — so it's an equally-valid-but-different approximation from the
   covariance method, not a strictly more correct one, *unless* nuisance sharing
   across analyses is actually the reason to want it.

**What to write out for the (recommended, linear) nuisance-parameter export** — all
derivable from quantities `combine()` already computes, no new fit required:

- `sys_names` (nsys), `pulls` (best-fit λ̂_i), `constraints` (post-fit σ_i) — already
  fields on `CombinationResult`.
- `pre_sys_corr` (the prior correlation ρ_ij the combination assumed) — already a
  field; needed if the downstream fit wants to apply the *same* prior penalty rather
  than treat each λ_i as independent.
- `nuisance_covariance` = `cov_full[:nsys, :nsys]` — the **post-fit** covariance of
  the nuisances (already computed, just not currently sliced out). Needed if the
  downstream fit wants to inherit the combination's posterior correlations between
  nuisances directly, rather than re-deriving them from `pre_sys_corr` + constraints.
- **New, not currently computed**: the response matrix
  `R[j, i] = ∂(combined_value_j) / ∂λ_i` at the best fit (`nest × nsys`). This is the
  signed, per-unit-λ slope a downstream nuisance-parameter model needs
  (`x_j(λ) ≈ x_j(λ̂) + Σ_i R[j,i]·(λ_i - λ̂_i)`). It is **not new math** — it's the
  standard multivariate-Gaussian conditional-slope identity applied to the existing
  post-fit covariance:
  `R = cov_full[nsys:, :nsys] @ inv(cov_full[:nsys, :nsys])`,
  one `nsys×nsys` solve, computed once in `combine()` right next to `corr_full`
  (`combiner.py` step 10, ~line 205-208). The already-printed "simple impact table"
  (`result.py:_print_simple_impacts`) is a *rescaled* version of the same
  correlation block (per-1σ-shift %, not a raw derivative) — `R` is the same
  information in fit-ready form.
  - **Caveat for asymmetric systematics**: `R` as defined is a single symmetric
    slope. If a systematic has `Lk_up != -Lk_down`, fidelity would call for an
    `R_up`/`R_down` pair (reusing the existing per-systematic `_profile_error` path,
    same mechanism already used for asymmetric impacts, just applied per-systematic
    instead of per-group). Current test setups are all symmetric, so this caveat is
    theoretical today but should be flagged in the export if any
    `responses_symmetric == False`, rather than silently handing out a misleading
    symmetric `R`.

  So: yes, exportable — `R`, `nuisance_covariance` (or `pre_sys_corr` + `constraints`
  if the downstream fit prefers to apply its own prior), `pulls`, `sys_names`. This
  slots into the same `to_dict()`/`export_npz`/`export_json` design as Q4, just with
  three more fields. No architecture change needed.

**If the exact (non-linearized) version is wanted instead**: that means not
exporting any summary at all, but literally adding the theory/mass-fit term into the
*same* chi2 that `combine()` minimizes, so `m_top` is profiled jointly with every
`λ_i` and `x_j` in one fit (the only way to capture non-linear response or let a
shared nuisance be exactly re-optimized rather than approximated by a frozen slope).
This requires `Combiner` to expose its chi2-building blocks (`setups`, `inv_C`,
`nsys`, `nest`, the JAX-built `chi2_fn`) as a composable object rather than a
single `combine()` entry point, so a caller can do
`chi2_total = chi2_fn(pars) + chi2_theory(m_top, pars[nsys:])` and re-minimize with
the same L-BFGS-B machinery. This is a real architecture change (new public API
surface, not just new output fields) — **only worth doing if the top-mass analysis
genuinely needs a shared nuisance re-optimized jointly**; for a standalone
"combined cross section → theory chi2 → m_top" fit (no shared nuisances with other
inputs), the linear `R`-based export above is the standard-practice answer and far
cheaper to deliver. Start with the linear export and revisit the joint-fit API only
if a concrete shared-nuisance need shows up.

---

## Recommended priority order (for a future implementation session)

1. **Export API + stat/syst split + per-individual-systematic frozen covariance**
   (Q4 / Q4b design above) — unblocks the actual next task; small, additive,
   reuses the existing `_compute_impacts` Hessian-submatrix machinery applied to
   every systematic as its own one-member group, no new fit and no new math.
2. **`lognormal` correctness gap** — either implement or fail loudly; silent
   mis-modeling is the highest-risk item found.
3. **PD-regularization warning** — cheap, prevents silently-wrong priors going
   unnoticed.
4. Carry forward existing improvement-roadmap NEXT items (B2 quadratic fast-path, B3
   `value_and_grad`, `--no-impacts`, formatter fidelity, correlation scan `-s`,
   chi2/ndf+p-value) at current relative priority — nothing here changes their
   ordering.

## Verification (for the future implementation session)

- Extend `test/golden/*.npz` regen to cover the new `stat_only_covariance` /
  `impact_cov_groups` export fields, so the export is itself regression-tested.
- Round-trip test: export npz/json for `atlas13test`, reload, assert values match the
  in-memory `CombinationResult` exactly (not just close) — this is a serialization
  test, should be bit-exact.
- Manually verify on one setup that
  `sqrt(sum of per-group impact^2 in quadrature) ≈ total syst impact` is *not* assumed
  exact (groups can overlap/correlate) — document that the stat/syst split and
  per-group impacts are independent quantities, not required to sum.
- Sanity-check the new per-individual-systematic `cov_without[s]` against the
  existing, already-validated "simple impact table"
  (`result.py:_print_simple_impacts`): for `nest==1`,
  `sqrt(combined_covariance - cov_without[s])` should reproduce the existing
  per-mille-level-validated simple-impact number for systematic `s` — a free
  correctness check against an already-C++-validated quantity, since it's the exact
  same Hessian-submatrix code path already exercised by the existing per-group
  impacts, just looped one systematic at a time.
