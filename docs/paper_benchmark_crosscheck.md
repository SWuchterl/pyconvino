# Crosscheck against the Convino paper's documented example

**Date:** 2026-07-23
**Input:** `old/Convino/examples/exampleconfig.txt` (+ `exampleMeasurement{1,2,3}.txt`,
`extra_correlations.txt`) — the worked example behind Figs. 14/15 of the
Convino paper (Kieseler, Section 5). Not previously crosschecked: only the
ATLAS/CMS `ConvinoSetups` (`test/test_regression.py`) had been validated
against C++ before this.

**Method:** ran the compiled C++ binary (`old/Convino/convino`, Pearson chi2 —
the default) and `python -m convino_jax.cli --pearson` on the identical input
files, diffed `result.txt` section by section.

## Central fit — matches

| Quantity | C++ | Python | rel. diff |
|---|---|---|---|
| chi2_min | 4.11332 | 4.11332 | 0 |
| combined_a | 798.69 +7.62619 −7.58072 | 798.69 +7.62616 −7.58088 | ~2e-5 |
| combined_b | 306.101 +5.50503 −5.39961 | 306.101 +5.50495 −5.39969 | ~1e-5 |

Confirms chi2 construction, Pearson/Neyman scaling, and all three input
formats (`[hessian]`, `[correlation matrix]`, `[not fitted]`) are ported
correctly.

## Per-systematic impacts — matches, except the smallest entries

| systematic | C++ (a, b) [%] | Python (a, b) [%] | rel. diff (a, b) |
|---|---|---|---|
| sys_a1 | 0.42234, 1.07526 | 0.42249, 1.07555 | 0.03%, 0.03% |
| sys_b1 | 0.43818, 0.06538 | 0.43820, 0.06564 | 0.01%, 0.4% |
| sys_c1 | 0.26322, 1.43027 | 0.26344, 1.43036 | 0.08%, 0.01% |
| sys_a2 | 0.11469, 0.10000 | 0.11468, 0.09992 | 0.01%, 0.08% |
| sys_b2 | 0.09430, 0.21306 | 0.09436, 0.21314 | 0.06%, 0.04% |
| sys_c2 | 0.46761, 0.47101 | 0.46766, 0.47126 | 0.01%, 0.05% |
| sys_d2 | 0.20781, 1.19106 | 0.20800, 1.19117 | 0.09%, 0.01% |
| sys_a3 | 0.01594, 0.03649 | 0.01523, 0.03564 | 4.5%, 2.3% |
| sys_b3 | 0.04469, 0.03518 | 0.04565, 0.03398 | 2.1%, 3.4% |
| sys_c3 | 0.02007, 0.00801 | 0.01961, 0.01220 | 2.3%, 52% |
| sys_d3 | 0.00117, 0.00063 | 0.00089, 0.00031 | 24%, 51% |
| sys_e3 | 0.00121, 0.00072 | 0.00123, 0.00066 | 1.6%, 8% |

`sys_a1`–`sys_d2` (from `exampleMeasurement1/2.txt`) agree to ≲0.1%. Group-level
impacts (`a_unc`/`b_unc`) and their covariances agree similarly (≲2%, up to
~11% on the smallest entry). Only `sys_a3`–`sys_e3` (from `exampleMeasurement3.txt`)
diverge, by 2–52%. Full text output: `out/paper_example_cpp_result.txt` /
`out/paper_example_py_result.txt`.

## Root cause of the sys_a3–sys_e3 divergence

Instrumented `Combiner._compute_impacts` to compare, per systematic, the
closed-form estimate (slice `H_fit`, invert — exact for a quadratic chi2)
against the profile-likelihood estimate the code actually uses whenever any
response in the combination is asymmetric (`sys_d3`'s `(+5-3)`).

- **The profile-likelihood path is correct, not the bug.** For `sys_a1`/`sys_c1`
  (which match C++ to <0.1%), the closed-form shortcut itself disagrees with
  C++ by ~2%; profile-likelihood is what matches. Gating the *whole*
  combination into MINOS/profile mode once any one parameter is asymmetric
  mirrors C++ (`combiner.py:331`), which always runs MINOS regardless of which
  parameter is asymmetric.
- **`sys_a3`–`sys_e3` have ~zero true impact.** Freezing `sys_d3` and
  re-minimizing moves chi2 by only `3.55e-7`; the frozen error matches the
  unfrozen error to 7 significant figures. The reported "impact" —
  `sqrt(err_full² − err_frozen²)` of two numbers agreeing to ~1e-7 — is
  reading off each optimizer's own convergence residual, not a physical
  difference. These systematics come from an extremely tightly self-constrained
  Hessian block (diagonal ~1944/1154/593, vs. O(1) elsewhere), so their true
  impact is at or below the noise floor of both SciPy (`gtol=1e-8` L-BFGS-B,
  `brentq xtol=1e-6`) and Minuit2/MINOS — two independent optimizers, so their
  residual noise has no reason to agree.
- **Why 2% up to 52%, not one number:** the *absolute* C++-vs-Python gap is
  small and roughly constant across all five (0.00002–0.0042 percentage
  points — one fixed-size noise floor). The *relative* number varies only
  because the true signal in the denominator shrinks toward zero
  (`sys_d3`, `sys_c3`-down have the smallest true impact → largest relative
  noise). One noise source through five tiny denominators, not five bugs.
- **Won't close by retrying:** existing tolerances are already far tighter
  than the ~1e-7 signal being resolved. Shrinking it further needs either
  chi2-scale-aware absolute convergence criteria (costly — see Finding 1
  below) or per-parameter (not combination-wide) symmetric/asymmetric
  handling (`responses_symmetric` is a single flag, `combiner.py:258`).
  Neither exists today. Doesn't affect any physics conclusion — the affected
  impacts are negligible either way.

## Finding 1 — asymmetric-impact path: fixed (both halves)

`Combiner._compute_impacts` (`combiner.py:933`) slices the post-fit Hessian
for symmetric responses (fast, exact); for asymmetric responses it instead
refits + profile-scans (`_minimize_frozen`/`_profile_error`,
`combiner.py:828-931`) **per impact group, per combined observable**. On this
tiny example (2 observables, 12 systematics) it originally took ~4-9 min wall
time for impacts alone across repeated runs (vs. <30s for everything else),
and over 45 min under CPU contention once — consistent with per-call JAX/XLA
dispatch overhead dominating.

**Test-coverage half**: `test/test_impacts_asymmetric.py`, gated behind
`CONVINO_SLOW_TESTS=1` (same convention as `corrV2`), runs
`ConvinoSetups/PaperExample/` (a copy of the paper's example, added so the
test doesn't depend on the sibling C++ checkout) and checks the "[simple
impact table]" output against the real C++ reference numbers: tight
(1%-margin) agreement on the 7 systematics with real signal, and a
sanity/non-NaN/small check on the 5 noise-floor ones. Its first draft had a
real bug — it compared against `impact_per_systematic` (the quadrature
frozen-fit impact used for `impact_groups`), not the different,
correlation-based quantity the "simple impact table" actually reports; fixed
by parsing `format_result()`'s text output directly, the same thing a user
sees.

**Performance half**: `combiner.py:250` used a plain, uncompiled `jax.grad`
for this path specifically to protect the impact quadrature's precision
(`objective.make_chi2` already returns a jitted `chi2_fn`, but the *fused*
jitted `value_and_grad_fn` reassociates floats at the ~1e-14 level, which
`sqrt(err_full^2 - err_frozen^2)` can amplify). Implemented the proposed fix:
a **separate**, non-fused `grad_search = jax.jit(jax.grad(chi2_fn))`, used
only for the L-BFGS-B/brentq search (`combiner.py:269`) — safe because
L-BFGS-B's own stopping rule (`gtol=1e-8`) is far looser than jit's
reassociation noise, so it converges to the same point either way, just
without paying eager dispatch overhead per op per call.

**Measured**: `test/test_impacts_asymmetric.py` — same C++-anchored
tolerances, same assertions — now runs in **43.2s**, down from the ~500-550s
baseline measured twice before the fix (**~12x**). Full fast suite (53
tests) unaffected, 13.75s. The ATLAS/CMS setups never take this branch at all
(`responses_symmetric=True` there), so nothing about their behavior changed.

## Finding 2 — differential normalisation (paper §2.4): implemented

**Fixed.** Ported `old/Convino/src/normaliser.cpp`'s algorithm — not a re-fit,
pure Monte Carlo error propagation on the already-combined result: draw
`N(combined_vals, cov)`, divide each sample by its own sum, take the
empirical covariance of the deviations from the nominal fraction.
`Combiner._normalise_differential` (vectorized NumPy, 1M draws in one batched
call) runs as combine()'s final step, gated by `cfg.is_differential and
cfg.normalise`, after impacts (matching C++'s ordering). Added
`CombinationResult.normalised: bool` (also in `to_dict()`/export) so callers
can tell. One deliberate divergence from C++, called out in a code comment:
C++ invalidates `chi2min_`/pulls/constraints/full-correlation afterward
(`-1`/empty); this port leaves them describing the pre-normalisation fit
instead of nulling them, since they're still meaningful.

**Verified** against a fresh C++ run on the paper's own
`differential_example.txt` (matching Pearson chi2 both sides): normalised
central bin fractions match **exactly** (`0.259876, 0.476774, 0.174705,
0.0886449`, both sides — the deterministic part of the algorithm). The
Monte-Carlo-estimated errors agree to <1% (`0.00296 vs 0.00298`, etc.) —
expected, since these are two independent 1M-draw MC estimates with
different RNGs, not a closed-form quantity. Full test suite unaffected (the
new step is a strict no-op unless both flags are set; no existing setup sets
them).

## Bottom line

Core algorithm (chi2, minimization, post-fit errors, impacts for symmetric
uncertainties) reproduces C++ to <0.1% on the paper's own example, matching
prior ATLAS/CMS-level agreement. The one non-tight spot is explained, not just
observed: near-zero-impact systematics hitting each optimizer's own noise
floor, not a modeling gap. Both gaps found are now fixed and verified: the
differential-normalisation feature is implemented and matches a fresh C++ run
exactly on its central values; the asymmetric-impact path has a passing
regression test against real C++ reference numbers **and** runs ~12x faster
(500s → 43s) after jitting the search-phase gradient, with no change to the
converged answer.
