"""
Combiner: orchestrates the full combination.

Steps performed:
1. Parse all measurement files and the config file.
2. Set up measurement matrices (LM, Lk, LD) for each measurement.
3. Build the global parameter vector layout:
      pars[:nsys]  = nuisance pulls (lambda), one per unique systematic
      pars[nsys:]  = combined observable values (x_comb), one per combined bin
4. Build the global prior inverse-covariance matrix from user correlations.
5. Build and JIT-compile the JAX chi2.
6. Minimization:
      - Quadratic fast path: when the chi2 is an exact quadratic form in
        `pars` (symmetric systematic responses, no "relative" systematics,
        and Pearson scaling off), its Hessian is constant and the minimum is
        reachable from any starting point in one linear solve.
      - General path (any "relative" systematic, or Pearson scaling on):
        two-pass scipy L-BFGS-B (pass 1: loose tolerance to get near the
        minimum; pass 2: tight tolerance for a precision fit), followed by a
        single Newton-correction step using the exact Hessian for extra
        precision.
7. Extract asymmetric errors via profile likelihood (MINOS equivalent).
8. Compute post-fit covariance from the Hessian of chi2 at the minimum.
9. Compute uncertainty impacts by re-fitting with systematic groups frozen.
10. Return a CombinationResult object.
"""

from __future__ import annotations

import copy
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import inv as scipy_inv
from scipy.optimize import brentq, minimize
from scipy.stats import chi2 as _chi2_dist

from .measurement import MeasurementSetup, setup_measurement
from .objective import make_chi2
from .parser import (
    ConfigData,
    MeasurementFileData,
    parse_config_file,
    parse_measurement_file,
)

jax.config.update("jax_enable_x64", True)


@dataclass
class CombinationResult:
    """All quantities needed to produce the output text file."""
    # Minimizer outcome
    chi2_min: float = 0.0
    converged: bool = False

    # Goodness of fit. ndf = (total number of individual input measurements,
    # i.e. sum of len(x_meas) over all measurement setups) - nest (the number
    # of fitted combined-observable parameters). Every nuisance parameter
    # (nsys of them) is excluded from both sides of that count: each is a
    # fitted parameter, but also carries at least a unit-Gaussian prior (the
    # global correlation prior, optionally sharpened by a per-measurement
    # Hessian block), so by the standard Wilks'-theorem treatment of
    # Gaussian-constrained profiled nuisances it contributes net zero degrees
    # of freedom — the same convention used for published ATLAS/CMS
    # combination chi2/ndf figures. (The original C++ Convino never computed
    # or printed ndf/p-value at all — only chi2min_ — so there is no fidelity
    # target to match here.) NaN when ndf <= 0 (e.g. a single, non-redundant
    # measurement with no systematics has nothing left to test for
    # consistency).
    ndf: int = 0
    chi2_per_ndf: float = float("nan")
    p_value: float = float("nan")

    # chi2 each input reaches on its own at its own best fit (the prior penalty
    # of its post-fit nuisance values; 0 for an input without [nuisance values]),
    # keyed by measurement file name, and chi2_min minus their sum: the part of
    # chi2_min that comes from combining, i.e. the tension between the inputs.
    chi2_standalone: dict[str, float] = field(default_factory=dict)
    chi2_tension: float = 0.0

    # Combined observables
    combined_names: list[str] = field(default_factory=list)
    combined_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    combined_err_up: np.ndarray = field(default_factory=lambda: np.zeros(0))
    combined_err_down: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Post-fit nuisance pulls and constraints
    sys_names: list[str] = field(default_factory=list)
    pulls: np.ndarray = field(default_factory=lambda: np.zeros(0))
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Post-fit correlation matrices (full, sys-only, est-only)
    corr_full: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    cov_full: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    all_names: list[str] = field(default_factory=list)  # sys_names + combined_names

    # Pre-combine system correlations (from input)
    pre_sys_corr: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # Impact groups: label → (abs_impact_up, abs_impact_down) per combined obs
    impact_groups: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # Per-group covariance of the combined observables with that group frozen
    # (nest × nest). Printed as [covariance matrix for merged impacts].
    impact_cov_groups: dict[str, np.ndarray] = field(default_factory=dict)

    # Group label of each nuisance (parallel to sys_names), from the user's
    # [uncertainty impacts] groups; "ungrouped" for nuisances in no group.
    # Used by the downstream global-impacts breakdown to bucket per-nuisance
    # global impacts (dm_i = Cov(POI, nu_i)) into additive per-group quadrature
    # sums. A nuisance appearing in several groups takes the first match.
    sys_group_labels: list[str] = field(default_factory=list)

    # Prior inverse-covariance (precision) of the nuisances, inv_C (nsys x nsys),
    # exactly as used in the chi2 prior term lambda^T inv_C lambda. The downstream
    # global-impacts systematic variance is g^T inv_C g with g_i = Cov(POI, nu_i)
    # (identity for independent unit priors; off-diagonal for correlated priors, e.g.
    # cross-experiment correlations, where g^T g alone would double-count).
    prior_inv_cov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # Effective auxiliary precision for the global-impacts stat/syst split:
    # prior_inv_cov plus the negative part of each input's nuisance block LD
    # (input Hessian information minus the unit prior). A negative LD direction
    # means the input gives a post-fit variance above the prior, which a
    # Gaussian profile fit cannot produce; counting it as data would give a
    # negative "data variance" (global stat < frozen stat). It is moved to the
    # auxiliary (syst) side instead. Equal to prior_inv_cov if no LD < 0.
    prior_inv_cov_eff: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # Best-fit full parameter vector
    pars_best: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nsys: int = 0
    nest: int = 0

    # Stat/syst split and per-individual-systematic breakdown, for feeding a
    # downstream fit's covariance-breakdown machinery (see
    # docs/improvement_plan.md Q4/Q4b). All derived from the same frozen-
    # Hessian-submatrix mechanism as impact_groups/impact_cov_groups above,
    # just applied to every systematic as its own one-member group plus one
    # group covering all of them.

    # Combined-observable covariance (nest x nest) with every systematic
    # frozen at 0 — the pure statistical/measurement-only uncertainty.
    stat_only_covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # Total systematic impact per combined observable: sqrt(full_err^2 -
    # stat_only_err^2), the same quadrature convention as impact_groups.
    total_syst_impact_up: np.ndarray = field(default_factory=lambda: np.zeros(0))
    total_syst_impact_down: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Per-individual-systematic impacts/frozen-covariance (one entry per
    # systematic name, independent of any user-defined [uncertainty impacts]
    # groups in impact_groups/impact_cov_groups above).
    impact_per_systematic: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    cov_per_systematic: dict[str, np.ndarray] = field(default_factory=dict)

    # Signed linear response matrix: impact_matrix[b, i] is the shift in
    # combined_values[b] for a +1σ variation of nuisance i, derived from the
    # post-fit Hessian cross-block A = -inv(H_xx) @ H_xt. Shape (nest, nsys).
    # NaN-filled if the Hessian was singular; shape (nest, 0) when nsys == 0.
    # Unlike impact_per_systematic (unsigned quadrature magnitudes), this gives
    # the full signed response vector needed for a nuisance-parameter downstream
    # fit (chi2 = (r - A @ theta)^T C_stat^{-1} (r - A @ theta) + ||theta||^2).
    impact_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # Impact-weighted mean nuisance pull per user-defined [uncertainty impacts]
    # group. Two weighting conventions:
    #   mean: weight_i = mean_b(|impact_per_systematic[i][0][b]|)
    #   norm: weight_i = norm_b(impact_per_systematic[i][0])
    # pull_g = Σ_i w_i * pull_i / Σ_i w_i  (sum over members of group g).
    # Empty when compute_impacts=False (--no-impacts) or nsys == 0.
    pull_per_group_mean: dict[str, float] = field(default_factory=dict)
    pull_per_group_norm: dict[str, float] = field(default_factory=dict)

    # True when [global] isDifferential/normalise were both set: combined_values/
    # combined_err_up/down and the combined-observable block of cov_full/corr_full
    # were bin-renormalised, so they describe shape (fractions summing to 1), not
    # the original absolute bin values. See Combiner._normalise_differential.
    normalised: bool = False


class Combiner:
    """
    Run the full Convino combination from a config file path.

    Usage
    -----
    result = Combiner.from_config("path/to/config.txt").combine()
    """

    def __init__(
        self,
        config: ConfigData,
        meas_data: list[MeasurementFileData],
        use_pearson: bool = False,
        compute_impacts: bool = True,
        impacts_only: list[str] | None = None,
        verbose: bool = False,
        pd_reg_method: str = "shift",
        nonneg_combined: bool = False,
        use_nuisance_values: bool = False,
    ):
        self.config = config
        self.meas_data = meas_data
        self.use_pearson = use_pearson
        self.compute_impacts = compute_impacts
        self.impacts_only = impacts_only
        self.verbose = verbose
        self.pd_reg_method = pd_reg_method
        self.nonneg_combined = nonneg_combined
        self.use_nuisance_values = use_nuisance_values
        if impacts_only is not None:
            unknown = sorted(set(impacts_only) - set(config.impact_groups))
            if unknown:
                raise ValueError(
                    f"--impacts-only requested unknown impact group(s) {unknown}; "
                    f"available: {sorted(config.impact_groups)}"
                )

    @classmethod
    def from_config(cls, config_path: str, **kw) -> Combiner:
        """Parse `config_path` and its measurement files; `kw` go to __init__."""
        t0 = time.perf_counter()
        cfg = parse_config_file(config_path)
        meas_data = [parse_measurement_file(p) for p in cfg.measurement_files]
        if kw.get("verbose"):
            print(
                f"[convino] parsed config + {len(meas_data)} measurement file(s): "
                f"{time.perf_counter() - t0:.3f}s",
                file=sys.stderr,
            )
        return cls(cfg, meas_data, **kw)

    def _vtime(self, label: str, t0: float) -> None:
        """Print elapsed time since `t0` if `self.verbose` (CLI --verbose)."""
        if self.verbose:
            print(f"[convino] {label}: {time.perf_counter() - t0:.3f}s", file=sys.stderr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(self) -> CombinationResult:
        cfg = self.config
        t_total = time.perf_counter()

        # 1. Setup measurements
        t0 = time.perf_counter()
        setups = [
            setup_measurement(d, self.use_nuisance_values) for d in self.meas_data
        ]

        # 2. Build global parameter index layout (mutates each setup in place)
        all_sys_names, combined_names, nsys, nest = self._assign_global_indices(setups)

        # 3. Build prior inverse-covariance
        free_names = {n for s in setups for n, p in zip(s.sys_names, s.prior_diag, strict=True) if p == 0}
        inv_C, C_exact = self._build_prior(all_sys_names, free_names)
        inv_C_eff = self._effective_prior(setups, inv_C)
        self._vtime("setup + prior", t0)

        # 4. Build chi2
        t0 = time.perf_counter()
        chi2_fn = make_chi2(setups, inv_C, nsys, nest, self.use_pearson)
        value_and_grad_fn = jax.jit(jax.value_and_grad(chi2_fn))
        hess_fn = jax.jit(jax.hessian(chi2_fn))
        # Gradient for the profile/impact refits (_minimize_frozen/_profile_error):
        # jitted, so the many L-BFGS-B iterations inside a refit -- and the
        # repeated refits inside a brentq profile scan -- reuse one cached XLA
        # executable instead of paying eager dispatch per op per call. This is
        # what made the asymmetric-response impacts path slow (see
        # docs/paper_benchmark_crosscheck.md Finding 1). Safe to jit here despite
        # the impact quadrature sqrt(err_full^2 - err_frozen^2)'s sensitivity to
        # float-reassociation noise: L-BFGS-B's own stopping rule (gtol=1e-8) is
        # far looser than jit's ~1e-14 reassociation noise, so it converges to the
        # same point either way, just faster. (This is a separate, non-fused jit
        # of grad alone -- NOT the fused, jitted value_and_grad_fn below, which
        # the main fit uses and which reassociates differently; validated against
        # real C++ reference numbers in test/test_impacts_asymmetric.py.)
        grad_search = jax.jit(jax.grad(chi2_fn))
        self._vtime("chi2 build", t0)

        # responses_symmetric: every systematic response is symmetric
        # (Lk_up == -Lk_down, no kink at lambda=0). Computed here (rather than
        # after _minimize, as in earlier versions) because it gates which
        # minimizer strategy to use, in addition to its other use below (MINOS
        # vs. HESSE errors).
        responses_symmetric = all(
            np.allclose(s.Lk_up, -s.Lk_down, atol=1e-9, rtol=1e-6)
            for s in setups
        )
        # No "relative" systematic anywhere: relative systematics are
        # multiplicative on x_comb (see the rel_factor branch in
        # objective._x_shifted), which is nonlinear in pars even when
        # symmetric. ("lognormal" is already rejected earlier in
        # make_chi2/setup_measurement, so only "absolute"/"relative" remain.)
        all_absolute = all(
            t == "absolute" for s in setups for t in s.sys_types
        )
        # Whenever the chi2 is an exact quadratic form in pars (symmetric
        # responses, no relative systematics, and not Pearson-rescaled — see
        # the use_pearson branch in objective.chi2, which makes LM_eff depend
        # nonlinearly on x_sh), its Hessian is constant and the gradient is
        # exactly linear: grad(p) = H @ (p - p_min) for ANY p. The true
        # minimum is then reachable from any starting point in one linear
        # solve, with no iterative optimizer needed at all.
        quadratic_fast_path = responses_symmetric and all_absolute and not self.use_pearson

        # 5. Initial parameter vector
        x0 = self._initial_params(setups, nsys, nest, combined_names)

        # 5b. Bounds (opt-in, --nonneg-combined): only the nest combined-value
        # parameters get a physical floor at 0; the nsys nuisance parameters
        # stay unbounded (Gaussian priors have no physical bound). Off by
        # default since not every combined quantity is a non-negative cross
        # section (e.g. asymmetries/ratios).
        bounds = None
        if self.nonneg_combined:
            bounds = [(None, None)] * nsys + [(0.0, None)] * nest

        # 6. Minimization: exact one-step solve for the quadratic case, or the
        # robust two-pass L-BFGS-B (with a Newton polish) otherwise. This is
        # also where the jitted value_and_grad_fn/hess_fn first get XLA-
        # compiled (jax.jit traces lazily, on first call), so this checkpoint
        # includes that one-time compile cost, not just optimizer iterations.
        t0 = time.perf_counter()
        pars_best, chi2_min, converged = self._minimize(
            value_and_grad_fn, x0, hess_fn=hess_fn,
            quadratic_fast_path=quadratic_fast_path, bounds=bounds,
        )
        self._vtime("minimize", t0)

        # 7. Post-fit Hessian → covariance
        #
        # H_fit is the Hessian of chi2 (not of -log L). For a chi2 the Taylor
        # expansion is chi2 = chi2_min + (1/2) dp^T H dp, so the parameter
        # covariance in the Gaussian approximation is 2*H^-1, and the 1-sigma
        # error (Delta chi2 = 1) is sqrt(2 * H^-1[i,i]). This is the Minuit
        # HESSE convention for an UP=1 chi2 fit. Applying the factor of 2 at the
        # source means every downstream quantity derived from cov_fit
        # (constraints, displayed covariance matrix, error-bracket hints) uses
        # the true variance; the correlation matrix is unaffected (factor
        # cancels) and profile-likelihood errors are computed independently.
        t0 = time.perf_counter()
        H_fit = np.array(hess_fn(pars_best))
        try:
            cov_fit = 2.0 * scipy_inv(H_fit)
        except np.linalg.LinAlgError:
            cov_fit = np.full_like(H_fit, np.nan)
            warnings.warn("Post-fit Hessian not invertible; covariance set to NaN", stacklevel=2)

        # 7b. Signed response matrix: shift in combined_values[b] for +1σ of
        # nuisance i, from the Hessian cross-block A = -inv(H_xx) @ H_xt.
        # H_xx = H_fit[nsys:, nsys:] is the observable-observable block
        # (= 2 * effective measurement inv-covariance), H_xt = H_fit[nsys:,
        # :nsys] is the observable-systematic cross block (= -2 * eff-inv-cov @
        # L), so A = -H_xx^{-1} H_xt = L_eff, the effective combined-space
        # response matrix. Uses the already-computed H_fit; no new fits needed.
        if nsys > 0:
            H_xx = H_fit[nsys:, nsys:]
            H_xt = H_fit[nsys:, :nsys]
            try:
                impact_matrix = -scipy_inv(H_xx) @ H_xt
            except np.linalg.LinAlgError:
                impact_matrix = np.full((nest, nsys), np.nan)
        else:
            impact_matrix = np.zeros((nest, 0))

        # 8. Errors on the combined observables.
        #
        # C++ Convino runs Minuit MINOS for every combined quantity
        # (combiner.cpp ~851: setAsMinosParameter). Convino's chi2 is gaussian,
        # i.e. EXACTLY quadratic whenever every systematic response is symmetric
        # (Lk_up == -Lk_down, no kink at lambda=0). For a quadratic chi2 the MINOS
        # error equals the HESSE error sqrt(2 * H^-1[k,k]) = sqrt(diag(cov_fit)),
        # which we already have from the post-fit covariance and which reproduces
        # the C++ MINOS result to every printed digit (verified on ATLAS8Only and
        # CMSOnly). HESSE is used in that case: it is the closed-form value of the
        # MINOS scan and is far more robust than brentq-profiling a fit with
        # several hundred nuisance parameters (where the inner minimisation does
        # not fully converge and yields spuriously small, asymmetric errors).
        #
        # Only a genuinely asymmetric response makes MINOS != HESSE; there we fall
        # back to the profile-likelihood scan to recover the true asymmetry.
        combined_vals = np.array(pars_best[nsys:])
        hesse_errs = np.sqrt(np.maximum(np.diag(cov_fit)[nsys:], 0.0))

        # responses_symmetric was already computed above (before _minimize),
        # since it also gates the quadratic fast path.

        combined_err_up = hesse_errs.copy()
        combined_err_down = hesse_errs.copy()
        if not responses_symmetric:
            for k in range(nest):
                eu, ed = self._profile_error(
                    chi2_fn, grad_search, pars_best, chi2_min, nsys + k, hesse_errs[k]
                )
                combined_err_up[k] = eu
                combined_err_down[k] = ed

        # 9. Pulls and constraints
        pulls = np.array(pars_best[:nsys])
        diag_cov_sys = np.sqrt(np.maximum(np.diag(cov_fit)[:nsys], 0.0))

        # 10. Correlation / covariance from post-fit Hessian
        corr_full = _cov_to_corr(cov_fit)

        self._vtime("post-fit covariance + errors", t0)

        # 12. Uncertainty impacts (and per-group frozen covariance matrices).
        # Both this step and step 13 below sit behind `self.compute_impacts`
        # (--no-impacts): they are the dominant runtime cost on setups with
        # many systematics/groups (see [[regression-tests]]), and a caller
        # who only wants the combined values/covariance can skip them
        # entirely. `self.impacts_only` (--impacts-only) instead narrows step
        # 12 to a subset of the user-defined groups without touching step 13
        # (the per-systematic breakdown is a separate, generally cheaper-per-
        # entry feature used by the export API).
        t0 = time.perf_counter()
        corr_est = corr_full[nsys:, nsys:]
        if self.compute_impacts:
            impact_groups_cfg = self.config.impact_groups
            if self.impacts_only is not None:
                impact_groups_cfg = {
                    label: impact_groups_cfg[label] for label in self.impacts_only
                }
            impact_groups, impact_cov_groups = self._compute_impacts(
                chi2_fn, grad_search, pars_best, chi2_min,
                all_sys_names, combined_err_up, combined_err_down,
                nsys, nest, corr_est, responses_symmetric, H_fit,
                impact_groups_cfg,
            )
        else:
            impact_groups, impact_cov_groups = {}, {}

        # 13. Stat/syst split + per-individual-systematic frozen covariance.
        # Same mechanism as step 12, just applied to every systematic as its
        # own one-member group plus a pseudo-group covering all of them (whose
        # frozen covariance IS the stat-only covariance, and whose "impact" IS
        # the total systematic impact).
        if not self.compute_impacts:
            # Skipped: leave NaN-filled placeholders of the right shape
            # (rather than the zero-size dataclass defaults) so downstream
            # consumers like to_dict()'s `combined_covariance -
            # stat_only_covariance` don't hit a shape mismatch.
            stat_only_covariance = np.full((nest, nest), np.nan)
            total_syst_impact_up = np.full(nest, np.nan)
            total_syst_impact_down = np.full(nest, np.nan)
            impact_per_systematic = {}
            cov_per_systematic = {}
        elif nsys > 0:
            per_sys_groups: dict[str, list[str]] = {
                name: [name] for name in all_sys_names
            }
            per_sys_groups["__stat_only__"] = list(all_sys_names)
            per_sys_impacts, per_sys_covs = self._compute_impacts(
                chi2_fn, grad_search, pars_best, chi2_min,
                all_sys_names, combined_err_up, combined_err_down,
                nsys, nest, corr_est, responses_symmetric, H_fit,
                per_sys_groups,
                true_frozen_cov=True,
            )
            total_syst_impact_up, total_syst_impact_down = per_sys_impacts.pop(
                "__stat_only__"
            )
            stat_only_covariance = per_sys_covs.pop("__stat_only__")
            impact_per_systematic = per_sys_impacts
            cov_per_systematic = per_sys_covs
        else:
            # No systematics: the combination is already stat-only.
            stat_only_covariance = cov_fit[nsys:, nsys:].copy()
            total_syst_impact_up = np.zeros(nest)
            total_syst_impact_down = np.zeros(nest)
            impact_per_systematic = {}
            cov_per_systematic = {}
        self._vtime("impacts", t0)

        # 13b. Group-level effective pulls: impact-weighted mean pull per
        # user-defined [uncertainty impacts] group. Two weighting conventions
        # are exported (mean: per-bin average |impact|; norm: Euclidean norm of
        # the impact vector over bins). Only available when impacts were computed
        # (both per-systematic breakdown above and user-defined groups) and nsys > 0.
        if self.compute_impacts and nsys > 0 and impact_per_systematic:
            sys_to_pull = {n: float(pulls[i]) for i, n in enumerate(all_sys_names)}
            pull_per_group_mean: dict[str, float] = {}
            pull_per_group_norm: dict[str, float] = {}
            for group_label, members in self.config.impact_groups.items():
                wm_sum = wn_sum = 0.0
                wm_pull = wn_pull = 0.0
                for name in members:
                    if name not in impact_per_systematic:
                        continue
                    impact_vec = np.asarray(impact_per_systematic[name][0])
                    wm = float(np.mean(np.abs(impact_vec)))
                    wn = float(np.linalg.norm(impact_vec))
                    p = sys_to_pull[name]
                    wm_sum += wm
                    wn_sum += wn
                    wm_pull += wm * p
                    wn_pull += wn * p
                pull_per_group_mean[group_label] = wm_pull / wm_sum if wm_sum > 0 else 0.0
                pull_per_group_norm[group_label] = wn_pull / wn_sum if wn_sum > 0 else 0.0
        else:
            pull_per_group_mean = {}
            pull_per_group_norm = {}

        # 14. Goodness of fit (ndf, chi2/ndf, p-value). See the ndf convention
        # documented on CombinationResult above.
        n_meas = sum(len(s.x_meas) for s in setups)
        ndf = n_meas - nest
        chi2_standalone = {
            Path(d.path).name: float(s.chi2_standalone)
            for d, s in zip(self.meas_data, setups, strict=True)
        }
        chi2_tension = float(chi2_min) - sum(chi2_standalone.values())
        if ndf > 0:
            chi2_per_ndf = float(chi2_min) / ndf
            p_value = float(_chi2_dist.sf(max(float(chi2_min), 0.0), ndf))
        else:
            chi2_per_ndf = float("nan")
            p_value = float("nan")

        # 15. Differential normalisation ([global] isDifferential + normalise,
        # paper Sec. 2.4). Purely a post-processing rescale of the already-
        # combined result: overwrites combined_vals/errs and the combined-
        # observable block of cov_fit/corr_full, in place, after impacts (which
        # describe the pre-normalisation fit, matching C++: normaliser.cpp runs
        # after the impact table is filled). Unlike C++ (which invalidates
        # chi2min_/pulls_/constraints_/the full correlation matrix afterwards by
        # setting them to -1/empty), chi2_min/ndf/pulls/constraints/impacts here
        # are deliberately left describing the pre-normalisation fit rather than
        # nulled out — they remain meaningful (goodness-of-fit of the combination
        # that was normalised), just on a different scale than combined_values.
        normalised = bool(cfg.is_differential and cfg.normalise)
        if normalised:
            t0 = time.perf_counter()
            cov_comb = cov_fit[nsys:, nsys:]
            combined_vals, cov_comb_normed = self._normalise_differential(
                combined_vals, cov_comb
            )
            combined_err_up = combined_err_down = np.sqrt(
                np.maximum(np.diag(cov_comb_normed), 0.0)
            )
            cov_fit = cov_fit.copy()
            cov_fit[nsys:, nsys:] = cov_comb_normed
            corr_full = _cov_to_corr(cov_fit)
            self._vtime("differential normalisation", t0)

        # Map each nuisance to its user-defined [uncertainty impacts] group
        # (parallel to all_sys_names). First match wins; "ungrouped" otherwise.
        _name_to_group = {}
        for _label, _members in cfg.impact_groups.items():
            for _m in _members:
                _name_to_group.setdefault(_m, _label)
        sys_group_labels = [_name_to_group.get(n, "ungrouped") for n in all_sys_names]

        self._vtime("total", t_total)

        return CombinationResult(
            chi2_min=float(chi2_min),
            converged=converged,
            ndf=ndf,
            chi2_per_ndf=chi2_per_ndf,
            p_value=p_value,
            chi2_standalone=chi2_standalone,
            chi2_tension=chi2_tension,
            combined_names=combined_names,
            combined_values=combined_vals,
            combined_err_up=combined_err_up,
            combined_err_down=combined_err_down,
            sys_names=all_sys_names,
            pulls=pulls,
            constraints=diag_cov_sys,  # post-fit sigma / prior sigma (prior sigma = 1)
            corr_full=corr_full,
            cov_full=cov_fit,
            normalised=normalised,
            all_names=all_sys_names + combined_names,
            pre_sys_corr=C_exact,  # exact user-specified prior C
            impact_groups=impact_groups,
            impact_cov_groups=impact_cov_groups,
            sys_group_labels=sys_group_labels,
            prior_inv_cov=np.asarray(inv_C),
            prior_inv_cov_eff=inv_C_eff,
            pars_best=np.array(pars_best),
            nsys=nsys,
            nest=nest,
            stat_only_covariance=stat_only_covariance,
            total_syst_impact_up=total_syst_impact_up,
            total_syst_impact_down=total_syst_impact_down,
            impact_per_systematic=impact_per_systematic,
            cov_per_systematic=cov_per_systematic,
            impact_matrix=impact_matrix,
            pull_per_group_mean=pull_per_group_mean,
            pull_per_group_norm=pull_per_group_norm,
        )

    def scan_correlations(
        self, n_steps: int = 6
    ) -> dict[str, tuple[np.ndarray, list[CombinationResult]]]:
        """
        Re-run the combination while sweeping a correlation assumption
        across its configured range, one named `[correlations]` group at a
        time. Python port of the C++ `-s` option (`combiner::scanCorrelations`
        / `single_correlationscan::scanVal`): for a group with a single
        `(nominal & low : high)` pair, the swept value at step i is
        `low + i*(high-low)/(n_steps-1)`; for a group with several pairs
        (moved together in lockstep, one config line scanning more than one
        systematic pair at once), each pair sweeps its own low/high range at
        the same step index, matching `single_correlationscan::scanVal` being
        called per-pair with the shared step. `n_steps=6` matches the
        original's hardcoded `single_correlationscan::nPoints()`.

        Groups where every pair has `low == high` (no actual range — just a
        plain nominal correlation, the common case) are skipped: scanning a
        single point burns a full re-fit for no information, unlike the
        reference implementation which scans every group unconditionally.

        Returns `{group_name: (scan_values, [CombinationResult, ...])}`, low
        to high. `scan_values[i]` is the swept correlation itself for a
        single-pair group, or the C++ reference's fallback `i/(n_steps-1)`
        fractional progress for a multi-pair group (where no single scalar
        correlation value applies).

        Impacts are not computed at the scan points (unlike the C++ reference):
        they are the dominant per-fit cost and a scan is about the combined
        values, errors and chi2.
        """
        if n_steps < 2:
            raise ValueError("scan_correlations: n_steps must be >= 2")

        results: dict[str, tuple[np.ndarray, list[CombinationResult]]] = {}
        for scan_idx, scan in enumerate(self.config.correlation_scans):
            if not any(sr.low != sr.high for sr in scan.ranges):
                continue

            step_results: list[CombinationResult] = []
            step_values: list[float] = []
            single_pair = len(scan.ranges) == 1
            for step in range(n_steps):
                cfg_step = copy.deepcopy(self.config)
                for sr in cfg_step.correlation_scans[scan_idx].ranges:
                    sr.nominal = sr.low + step * (sr.high - sr.low) / (n_steps - 1)

                # The per-step Combiner is deliberately built non-verbose: a
                # full phase breakdown per step per group would be far too
                # noisy. Instead we print our own coarser per-step line here,
                # timed from this loop, while the inner combine() stays
                # silent regardless of self.verbose.
                t0 = time.perf_counter()
                step_comb = copy.copy(self)
                step_comb.config = cfg_step
                step_comb.compute_impacts = False
                step_comb.verbose = False
                step_results.append(step_comb.combine())
                self._vtime(f"scan '{scan.name}' step {step + 1}/{n_steps}", t0)
                if single_pair:
                    sr0 = scan.ranges[0]
                    step_values.append(sr0.low + step * (sr0.high - sr0.low) / (n_steps - 1))
                else:
                    step_values.append(step / (n_steps - 1))

            results[scan.name] = (np.asarray(step_values), step_results)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_global_indices(
        self,
        setups: list[MeasurementSetup],
    ) -> tuple[list[str], list[str], int, int]:
        """
        Build the global parameter layout:
          pars[:nsys] = systematics (nuisance pulls)
          pars[nsys:] = combined observables

        Assigns est_global_idx and sys_global_idx on each MeasurementSetup
        in place.
        """
        cfg = self.config

        # Collect all unique systematics in encounter order
        all_sys_names = list(dict.fromkeys(n for ms in setups for n in ms.sys_names))

        # Build combined observable list from config
        # combined_names order matches config [observables] block order
        combined_names: list[str] = list(cfg.observables.keys())

        # Build mapping: estimate_name → combined_name → global index (nsys + k)
        est_to_combined: dict[str, str] = {}
        for cname, ests in cfg.observables.items():
            for e in ests:
                est_to_combined[e] = cname

        sys_to_idx = {n: i for i, n in enumerate(all_sys_names)}
        comb_to_idx = {n: i for i, n in enumerate(combined_names)}

        nsys = len(all_sys_names)
        nest = len(combined_names)

        for ms in setups:
            for e in ms.est_names:
                if e not in est_to_combined:
                    raise ValueError(
                        f"estimate '{e}' is not listed in any [observables] entry "
                        f"(estimates in this measurement: {ms.est_names})"
                    )
            ms.est_global_idx = [
                comb_to_idx[est_to_combined[e]] for e in ms.est_names
            ]
            ms.sys_global_idx = [sys_to_idx[s] for s in ms.sys_names]

        return all_sys_names, combined_names, nsys, nest

    def _build_prior(
        self, all_sys_names: list[str], free_names: set[str] = frozenset()
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the (nsys × nsys) inverse prior covariance.

        The prior covariance C has:
          C[i,i] = 1  (unit prior for each systematic)
          C[i,j] = rho_{ij}  (from [correlations] block in config)

        `free_names` (parameters an input fitted without a prior) get no
        prior here either: their row/column of inv_C is zero. They must not
        be correlated with anything, since there is no prior to correlate.

        Returns (inv_C, C_exact) where C_exact is the user-specified matrix
        (before PD regularisation) for display purposes.
        """
        nsys = len(all_sys_names)
        C = np.eye(nsys)
        name_to_idx = {n: i for i, n in enumerate(all_sys_names)}

        # NOTE: only the nominal correlation (sr.nominal) is used. The scan
        # ranges sr.low / sr.high parsed from the "(nom & low : high)" syntax
        # are deliberately not yet propagated — scanning the correlation
        # assumption to derive an additional uncertainty is a planned future
        # feature (the C++ tool exposes this via its "-s" option). The parser
        # already retains low/high so that work does not require re-parsing.
        # If the same pair is specified more than once, the last value wins.
        for scan in self.config.correlation_scans:
            for sr in scan.ranges:
                ia = name_to_idx.get(sr.name_a)
                ib = name_to_idx.get(sr.name_b)
                if ia is None or ib is None:
                    missing = sr.name_a if ia is None else sr.name_b
                    raise ValueError(
                        f"[correlations] entry ({sr.name_a!r}, {sr.name_b!r}) refers to "
                        f"unknown systematic {missing!r} (not in all_sys_names)"
                    )
                rho = sr.nominal
                C[ia, ib] = rho
                C[ib, ia] = rho

        C_exact = C.copy()  # exact user values for display
        C = _nearest_positive_definite(C, method=self.pd_reg_method, verbose=self.verbose)
        inv_C = scipy_inv(C)
        for name in sorted(free_names):
            i = name_to_idx[name]
            off = np.delete(C_exact[i], i)
            if np.any(off != 0):
                raise ValueError(
                    f"free parameter {name!r} appears in [correlations] with a "
                    "non-zero correlation; a parameter without prior cannot be "
                    "correlated through the prior"
                )
            inv_C[i, :] = 0.0
            inv_C[:, i] = 0.0
        return inv_C, C_exact

    def _initial_params(
        self,
        setups: list[MeasurementSetup],
        nsys: int,
        nest: int,
        combined_names: list[str],
    ) -> np.ndarray:
        """
        Start from: lambda = 0 for all systematics,
        x_comb = mean of contributing measurements for each combined observable.
        """
        x0 = np.zeros(nsys + nest)

        # For each combined observable, average over measurements that contribute.
        # ms.est_global_idx[k] is the combined index of estimate est_names[k]
        # (assigned in _assign_global_indices).
        idx = np.concatenate([np.asarray(ms.est_global_idx, dtype=int) for ms in setups])
        vals = np.concatenate([np.asarray(ms.x_meas, dtype=float) for ms in setups])
        comb_cnt = np.bincount(idx, minlength=nest)
        if (comb_cnt == 0).any():
            i = int(np.flatnonzero(comb_cnt == 0)[0])
            raise ValueError(
                f"observable {combined_names[i]!r} has no contributing estimates "
                f"(declared in [observables] but no measurement lists it)"
            )
        x0[nsys:] = np.bincount(idx, weights=vals, minlength=nest) / comb_cnt

        return x0

    @staticmethod
    def _bounds_violated(vec: np.ndarray, bounds) -> bool:
        """True if any component of `vec` falls outside its (lo, hi) bound."""
        if bounds is None:
            return False
        return any(
            (lo is not None and vec[i] < lo) or (hi is not None and vec[i] > hi)
            for i, (lo, hi) in enumerate(bounds)
        )

    def _minimize(self, value_and_grad_fn, x0: np.ndarray, hess_fn=None,
                  quadratic_fast_path: bool = False, bounds=None):
        """
        Find the chi2 minimum.

        `value_and_grad_fn` is a single jitted jax.value_and_grad callable, so
        each evaluation point triggers exactly one forward+backward XLA
        dispatch (instead of separately dispatching a jitted `fun` and a
        jitted `jac`, which would recompute the forward pass twice per point
        since reverse-mode autodiff already performs it as part of the
        gradient). Wired to scipy via the `jac=True` convention: `fun` returns
        a `(value, gradient)` tuple.

        Fast path (quadratic_fast_path=True): the chi2 is an exact quadratic
        form in `pars` with a parameter-independent Hessian H (this holds
        whenever every systematic response is symmetric, no systematic is
        "relative", and Pearson rescaling is off — see the gating logic in
        combine()). For such a chi2 the gradient is exactly linear,
        grad(p) = H @ (p - p_min), so the true minimum is reachable from ANY
        starting point in one linear solve:

            p_min = p0 - solve(H, grad(p0))

        No iterative optimizer is needed, and the result is exact (up to
        floating point), so `converged` is unconditionally True. This shortcut
        has no notion of bounds, so if `bounds` is given and the unconstrained
        solve lands outside them, it is discarded and control falls through to
        the general (bounded) path below instead.

        General path (otherwise, or as the fast-path's bounded fallback): the
        existing robust two-pass L-BFGS-B, followed by a single
        Newton-correction step using the exact Hessian evaluated at the
        L-BFGS-B solution. L-BFGS-B already lands inside the basin where
        Newton's quadratic convergence applies, so this step is safe and
        cheap; it just buys back precision (closer to machine epsilon) on top
        of the validated L-BFGS-B result, which is otherwise left untouched.
        The Newton step itself is also an unconstrained linear solve, so it is
        only accepted when `bounds` is respected too.
        """
        if quadratic_fast_path:
            assert hess_fn is not None, "quadratic_fast_path requires hess_fn"
            p0 = jnp.array(x0, dtype=jnp.float64)
            H = np.array(hess_fn(p0))
            _, g0 = value_and_grad_fn(p0)
            g0 = np.array(g0)
            step = np.linalg.solve(H, g0)
            p_min = np.array(x0, dtype=np.float64) - step
            if not self._bounds_violated(p_min, bounds):
                p_min_j = jnp.array(p_min, dtype=jnp.float64)
                chi2_min, _ = value_and_grad_fn(p_min_j)
                # Exact linear solve, not an iterative outcome that can fail
                # to converge.
                return p_min_j, float(chi2_min), True
            # Unconstrained minimum violates bounds: fall through to the
            # general bounded L-BFGS-B path below.

        def _fun_and_grad(pars):
            v, g = value_and_grad_fn(jnp.array(pars, dtype=jnp.float64))
            return float(v), np.array(g)

        # Pass 1: loose tolerance
        res1 = minimize(
            _fun_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-5},
        )

        # Pass 2: precision
        res2 = minimize(
            _fun_and_grad,
            res1.x,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-9},
        )

        # res2.success can be False even at a perfectly good minimum (L-BFGS-B
        # line-search quirks near machine precision), so fall back to an
        # explicit stationarity test on the gradient. The previous fallback
        # (res2.fun < res1.fun + 1e-6) was meaningless: pass 2 starts from
        # pass 1's point, so it is true for essentially every fit, converged or
        # not.
        x_best = res2.x
        fun_best = res2.fun
        _, g_final = value_and_grad_fn(jnp.array(x_best, dtype=jnp.float64))
        gnorm = float(np.linalg.norm(np.array(g_final)))
        converged = bool(res2.success or gnorm < 1e-4)

        # Newton-correction polish using the exact Hessian, evaluated once at
        # the L-BFGS-B solution. L-BFGS-B already lands inside the basin where
        # Newton's quadratic convergence applies, so this is a cheap precision
        # refinement layered on top of the (unmodified) robust optimizer
        # above, not a replacement for it. Guarded so it can only improve
        # things: if the Hessian solve fails, or the resulting point is not
        # at least as good (lower or equal chi2 and gradient norm), fall back
        # to the L-BFGS-B result unchanged.
        if hess_fn is not None:
            try:
                p_best = jnp.array(x_best, dtype=jnp.float64)
                H_best = np.array(hess_fn(p_best))
                g_best = np.array(g_final)
                step = np.linalg.solve(H_best, g_best)
                p_polished = np.array(x_best, dtype=np.float64) - step
                p_polished_j = jnp.array(p_polished, dtype=jnp.float64)
                chi2_polished, g_polished = value_and_grad_fn(p_polished_j)
                chi2_polished = float(chi2_polished)
                if (
                    np.all(np.isfinite(p_polished))
                    and chi2_polished <= fun_best
                    and not self._bounds_violated(p_polished, bounds)
                ):
                    gnorm_polished = float(np.linalg.norm(np.array(g_polished)))
                    # Only accept if it's at least as good in gradient norm
                    # too (chi2 at a quadratic-ish minimum can be flat enough
                    # that a tiny chi2 improvement hides a worse gradient).
                    if gnorm_polished <= gnorm:
                        x_best = p_polished
                        fun_best = chi2_polished
                        gnorm = gnorm_polished
                        converged = bool(converged or gnorm < 1e-4)
            except np.linalg.LinAlgError:
                pass  # keep the L-BFGS-B result unchanged

        return jnp.array(x_best, dtype=jnp.float64), fun_best, converged

    def _minimize_frozen(self, chi2_fn, grad_fn, p_init, frozen_idx, frozen_vals, options):
        """
        Minimize chi2 over all parameters except `frozen_idx`, which are pinned
        to `frozen_vals` (scalar or array). `grad_fn` (built once by the caller)
        supplies the analytic jacobian instead of rebuilding jax.grad per call.

        Returns the scipy OptimizeResult.
        """
        p_init = np.asarray(p_init, dtype=np.float64)
        n = len(p_init)
        free_mask = np.ones(n, dtype=bool)
        free_mask[frozen_idx] = False

        def _scatter(x_free):
            p = np.empty(n)
            p[free_mask] = x_free
            p[frozen_idx] = frozen_vals
            return jnp.array(p, dtype=jnp.float64)

        def obj(x_free):
            return float(chi2_fn(_scatter(x_free)))

        def jac(x_free):
            return np.array(grad_fn(_scatter(x_free)))[free_mask]

        res = minimize(obj, p_init[free_mask], method="L-BFGS-B",
                       jac=jac, options=options)
        return res

    def _profile_error(
        self,
        chi2_fn,
        grad_fn,
        pars_best: jnp.ndarray,
        chi2_min: float,
        param_idx: int,
        sigma_sym: float,
        extra_frozen_idx=None,
        extra_frozen_vals=None,
    ) -> tuple[float, float]:
        """
        Find asymmetric ±1σ errors via profile likelihood:
        profile_chi2(v) = min_{pars except param_idx} chi2(pars)  at  pars[param_idx] = v

        Solves: profile_chi2(v) - chi2_min = 1

        `extra_frozen_idx` / `extra_frozen_vals` pin additional parameters during
        the inner minimisation. They are used to profile a combined observable
        while an impact group is held frozen, so the frozen error is on the same
        profile-likelihood footing as the full error.
        """
        pars_arr = np.array(pars_best, dtype=np.float64)
        v0 = float(pars_arr[param_idx])
        _opts = {"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-8}

        extra_idx = list(extra_frozen_idx) if extra_frozen_idx is not None else []
        extra_vals = list(extra_frozen_vals) if extra_frozen_vals is not None else []

        def profile(v: float) -> float:
            """Minimize chi2 with pars[param_idx] = v (extras frozen)."""
            res = self._minimize_frozen(
                chi2_fn, grad_fn, pars_arr,
                [param_idx] + extra_idx, [float(v)] + extra_vals, _opts,
            )
            return res.fun

        target = chi2_min + 1.0

        # brentq operates on the absolute parameter value v (which can be
        # O(1e3-1e4)), so rtol dominates the achievable precision: rtol=1e-4 on
        # v~8000 leaves ~0.8 of slop in the returned error. Tight tolerances
        # bring the profile errors to ~5 significant figures, which also keeps
        # the impact quadrature sqrt(err_full^2 - err_frozen^2) accurate for the
        # tiny impacts where err_full ~= err_frozen.
        _XTOL, _RTOL, _MAXITER = 1e-6, 1e-10, 100

        # Upward error
        try:
            bracket_hi = v0 + max(sigma_sym * 5, abs(v0) * 0.1 + 1e-3)
            eu = brentq(lambda v: profile(v) - target,
                        v0, bracket_hi,
                        xtol=_XTOL, rtol=_RTOL, maxiter=_MAXITER) - v0
        except Exception:
            warnings.warn(
                f"profile (upward) error for parameter {param_idx} failed; "
                "falling back to the symmetric HESSE error", stacklevel=2
            )
            eu = sigma_sym  # fallback

        # Downward error
        try:
            bracket_lo = v0 - max(sigma_sym * 5, abs(v0) * 0.1 + 1e-3)
            ed = v0 - brentq(lambda v: profile(v) - target,
                             bracket_lo, v0,
                             xtol=_XTOL, rtol=_RTOL, maxiter=_MAXITER)
        except Exception:
            warnings.warn(
                f"profile (downward) error for parameter {param_idx} failed; "
                "falling back to the symmetric HESSE error", stacklevel=2
            )
            ed = sigma_sym  # fallback

        return float(eu), float(ed)

    @staticmethod
    def _effective_prior(setups, inv_C: np.ndarray) -> np.ndarray:
        """inv_C plus the negative part of every input's nuisance block LD.

        The chi2 holds lambda^T LD lambda per input and lambda^T inv_C lambda
        globally, so the data information on the nuisances is LD (input
        information minus the unit prior). Eigen-directions with LD < 0 are
        directions where the input's post-fit variance exceeds the prior; they
        are added to the auxiliary precision so that the downstream global
        stat (data part) stays >= the frozen stat. The fit is not changed.
        """
        inv_C_eff = np.array(inv_C, dtype=float, copy=True)
        for k, s in enumerate(setups):
            LD = np.asarray(s.LD, dtype=float)
            if LD.size == 0:
                continue
            e, V = np.linalg.eigh(0.5 * (LD + LD.T))
            neg = e < -1e-9 * max(1.0, float(np.abs(e).max()))
            if not np.any(neg):
                continue
            idx = np.asarray(s.sys_global_idx)
            inv_C_eff[np.ix_(idx, idx)] += (V[:, neg] * e[neg]) @ V[:, neg].T
            warnings.warn(
                f"[convino] input #{k}: {int(neg.sum())} nuisance "
                f"direction(s) with post-fit variance above the prior (min LD "
                f"eigenvalue {e.min():.3g}); moved to prior_inv_cov_eff for the "
                f"global-impacts split (fit unchanged)", stacklevel=2
            )
        return inv_C_eff

    def _compute_impacts(
        self,
        chi2_fn,
        grad_fn,
        pars_best: jnp.ndarray,
        chi2_min: float,
        all_sys_names: list[str],
        combined_err_up: np.ndarray,
        combined_err_down: np.ndarray,
        nsys: int,
        nest: int,
        corr_est: np.ndarray,
        responses_symmetric: bool,
        H_fit: np.ndarray,
        groups: dict[str, list[str]],
        true_frozen_cov: bool = False,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
        """
        For each entry in `groups` (label -> member systematic names), freeze
        the group's systematics at 0 and compute the impact in quadrature vs.
        the full error. Called once for the user-defined
        `self.config.impact_groups` and once for the per-individual-systematic
        (+ all-frozen "stat-only") breakdown — same mechanism, different group
        dict.

        The frozen-fit Hessian restricted to the free parameters is exactly the
        free-free submatrix of the full post-fit Hessian `H_fit`: freezing a
        parameter removes its row/column, and for the (quadratic) gaussian chi2
        the Hessian is constant, so the frozen minimum sits on the same
        paraboloid as `pars_best`. We therefore slice `H_fit` instead of doing a
        per-group L-BFGS-B refit followed by a freshly jitted jax.hessian — that
        per-group re-JIT + re-fit was the dominant cost (one XLA compile per
        group; 48 groups for the full combination). Only the genuinely
        asymmetric-response case still needs the refit + profile scan, where the
        submatrix serves as the bracket seed.

        `corr_est` is the (nest x nest) post-fit correlation of the combined
        observables from the FULL fit; C++ scales it by the frozen errors to
        build each per-group covariance (see combiner.cpp printout). With
        `true_frozen_cov` the covariance instead uses the correlation of the
        frozen fit itself (est block of 2*inv(H_restricted), i.e. the Schur
        complement of the full covariance), which is the actual covariance of
        the combined observables with the group frozen. Used for the exported
        stat_only_covariance / cov_per_systematic; the printed per-group
        matrices keep the C++ convention.

        Returns (impacts, frozen_covs) where:
          impacts[label]     = (impact_up, impact_down) per combined observable
          frozen_covs[label] = (nest x nest) covariance of the combined
                               observables with that group frozen
        """
        sys_to_idx = {n: i for i, n in enumerate(all_sys_names)}
        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        cov_results: dict[str, np.ndarray] = {}
        n = nsys + nest

        for label, members in groups.items():
            frozen_idx = [sys_to_idx[m] for m in members if m in sys_to_idx]
            if not frozen_idx:
                warnings.warn(
                    f"impact group '{label}' has no members matching any known "
                    f"systematic ({members}); skipping it", stacklevel=2
                )
                continue

            free_mask = np.ones(n, dtype=bool)
            free_mask[frozen_idx] = False
            free_indices = np.where(free_mask)[0]

            # x_comb indices within the free subspace. The combined observables
            # are never frozen, so they always occupy the last nest slots of the
            # free subspace, in combined_names order.
            n_free_sys = int(np.sum(free_mask[:nsys]))
            est_slice = slice(n_free_sys, n_free_sys + nest)
            try:
                # Restricted Hessian = free-free submatrix of the full Hessian
                # (see the docstring). Factor of 2: chi2 Hessian → true variance
                # (Minuit UP=1), same convention as cov_fit above, so err_frozen
                # is on the same footing as the combined_err_up/down it is
                # combined with.
                H_restricted = H_fit[np.ix_(free_indices, free_indices)]
                cov_restricted = 2.0 * scipy_inv(H_restricted)
                # HESSE frozen errors = sqrt of the diagonal of the est-block.
                # Also used as the bracket seed for the profile scan below.
                err_frozen_up = np.sqrt(np.maximum(
                    np.diag(cov_restricted)[est_slice], 0.0
                ))
                err_frozen_down = err_frozen_up.copy()

                # When the full errors were obtained by profiling (asymmetric
                # responses), the frozen errors must be profiled too, so the
                # impact quadrature sqrt(err_full^2 - err_frozen^2) combines
                # like-for-like. This mirrors C++, which refits the same
                # MINOS-configured fitter with the group frozen. In the
                # symmetric (purely quadratic) case HESSE == profile, so the
                # cheaper submatrix errors above are kept unchanged and no refit
                # is performed.
                # Frozen at their post-fit values (identical to freezing at 0
                # for inputs without [nuisance values], whose pulls are 0):
                # pinning a pulled nuisance back to 0 would both misdefine the
                # impact and turn every refit into a full re-minimisation.
                if not responses_symmetric:
                    frozen_vals = np.array(pars_best, dtype=np.float64)[frozen_idx]
                    res = self._minimize_frozen(
                        chi2_fn, grad_fn, pars_best, frozen_idx, frozen_vals,
                        {"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
                    )
                    p_frozen_full = np.array(pars_best, dtype=np.float64)
                    p_frozen_full[free_mask] = res.x
                    p_frozen_full[frozen_idx] = frozen_vals
                    eu_list, ed_list = [], []
                    for k in range(nest):
                        eu_f, ed_f = self._profile_error(
                            chi2_fn, grad_fn, p_frozen_full, res.fun,
                            nsys + k, float(err_frozen_up[k]),
                            extra_frozen_idx=frozen_idx,
                            extra_frozen_vals=list(frozen_vals),
                        )
                        eu_list.append(eu_f)
                        ed_list.append(ed_f)
                    err_frozen_up = np.array(eu_list)
                    err_frozen_down = np.array(ed_list)

                # Covariance of the combined observables with this group frozen,
                # printed as the [covariance matrix for merged impacts] section.
                # C++ builds it from the FULL-fit result correlation scaled by the
                # (symmetric) frozen errors: cov[i,j] = corr_est[i,j]*sig[i]*sig[j]
                # with sig = max(|err_up|, |err_down|). (Diagonal = sig**2 since
                # corr_est[i,i] = 1.)
                sym_frozen = np.maximum(np.abs(err_frozen_up), np.abs(err_frozen_down))
                if true_frozen_cov:
                    c_fr = cov_restricted[est_slice, est_slice]
                    corr_fr = _cov_to_corr(c_fr, floor=1e-300)
                    est_cov = corr_fr * np.outer(sym_frozen, sym_frozen)
                else:
                    est_cov = corr_est * np.outer(sym_frozen, sym_frozen)
            except np.linalg.LinAlgError:
                est_cov = np.full((nest, nest), np.nan)
                err_frozen_up = np.full(nest, np.nan)
                err_frozen_down = np.full(nest, np.nan)

            # Impact in quadrature: sqrt(|err_full^2 - err_frozen^2|)
            impact_up = np.sqrt(np.maximum(combined_err_up ** 2 - err_frozen_up ** 2, 0.0))
            impact_down = np.sqrt(np.maximum(combined_err_down ** 2 - err_frozen_down ** 2, 0.0))
            results[label] = (impact_up, impact_down)
            cov_results[label] = est_cov

        return results, cov_results

    def _normalise_differential(
        self,
        combined_vals: np.ndarray,
        cov_est: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bin-renormalisation for differential combinations ([global] isDifferential
        + normalise, paper Sec. 2.4): rescale each combined bin by the sum over all
        bins, propagating the uncertainty by Monte Carlo since the ratio is
        nonlinear in the (correlated) bin values. Ports
        old/Convino/src/normaliser.cpp's default (text-interface) behavior — no
        addFloatingBin support, since that is a C++-interface-only extension never
        exposed through the base config file.

        For each of n_iter = 1e6 draws x ~ N(combined_vals, cov_est): divide x by its
        own sum, subtract the nominal fraction, and accumulate the outer product.
        The result is the empirical covariance of the normalised fractions; C++
        does the same accumulation in a loop with a fixed seed(0)/1e6-iteration
        default, which this matches in spirit (MC estimates from the two RNGs will
        not agree bit-for-bit, only within the O(1/sqrt(n_iter)) MC uncertainty).

        Returns (normalised_vals, normalised_cov), both shape matching
        combined_vals/cov_est.
        """
        n_iter = 1_000_000
        rng = np.random.default_rng(0)
        sum_nominal = combined_vals.sum()
        nominal_frac = combined_vals / sum_nominal
        samples = rng.multivariate_normal(combined_vals, cov_est, size=n_iter)
        sum_varied = samples.sum(axis=1, keepdims=True)
        frac_dev = samples / sum_varied - nominal_frac
        normalised_cov = (frac_dev.T @ frac_dev) / n_iter
        return nominal_frac, normalised_cov


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _cov_to_corr(cov: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Correlation matrix of `cov` (variances floored at `floor`, unit diagonal)."""
    d = np.sqrt(np.maximum(np.diag(cov), floor))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _higham_nearest_corr(
    A: np.ndarray,
    eps: float = 1e-8,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Higham (2002) alternating-projection algorithm for the nearest correlation
    matrix (unit diagonal, positive-definite) in Frobenius norm.

    Alternates between:
      1. Dykstra-corrected projection onto the PSD cone (with eigenvalue floor eps).
      2. Projection onto the unit-diagonal hyperplane (set diagonal to 1.0).
    Converges quadratically; for the matrix sizes typical here (< 500 systematics)
    it needs only a handful of iterations.
    """
    Y = A.copy()
    delta_S = np.zeros_like(A)

    for _ in range(max_iter):
        R = Y - delta_S
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals_floored = np.maximum(eigvals, eps)
        X = eigvecs @ np.diag(eigvals_floored) @ eigvecs.T
        delta_S = X - R
        Y_prev = Y
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        norm_Y = max(1.0, np.linalg.norm(Y, "fro"))
        if np.linalg.norm(Y - Y_prev, "fro") / norm_Y < tol:
            break

    return Y


def _nearest_positive_definite(
    A: np.ndarray,
    method: str = "shift",
    verbose: bool = False,
) -> np.ndarray:
    """
    Return a positive-definite correlation matrix (unit diagonal) closest to A.

    Three methods are available:

    ``"shift"`` (default)
        Add the smallest δI that makes all eigenvalues ≥ eps, then renormalise
        rows/columns to restore unit diagonal.  All off-diagonal entries are
        uniformly scaled by 1/(1+δ), giving a clear physical interpretation:
        "your correlations were too large to be consistent; they have been
        damped by factor X."

    ``"clip"``
        Eigenvalue decomposition — reflect negative eigenvalues to eps, then
        renormalise rows/columns to restore unit diagonal.  The change is
        concentrated in the directions (eigenvectors) with negative eigenvalues,
        so the off-diagonal changes are not uniform.

    ``"higham"``
        Higham (2002) alternating-projection algorithm: returns the *nearest*
        correlation matrix in Frobenius norm.  Minimises the total change to
        the off-diagonal entries but requires an iterative solve.

    A ``UserWarning`` is always emitted when regularisation was actually needed
    (eigenvalue below floating-point noise floor), reporting the number and
    magnitude of negative eigenvalues plus the off-diagonal change statistics.
    With ``verbose=True`` an additional diagnostic line is printed to stderr.
    """
    if A.size == 0:
        return A  # stat-only combination — no systematics

    n = A.shape[0]
    A = (A + A.T) / 2.0  # enforce exact symmetry

    eigvals = np.linalg.eigvalsh(A)  # sorted ascending
    min_eigval = float(eigvals[0])
    max_abs_eigval = float(np.max(np.abs(eigvals)))
    # PD eigenvalue floor and FP-noise detection threshold (two orders of
    # magnitude apart so noise around 0 does not trigger regularisation).
    eps = 1e-8 * max(1.0, max_abs_eigval)
    noise_tol = 1e-10 * max(1.0, max_abs_eigval)

    if min_eigval >= eps:
        return A  # already comfortably positive-definite / invertible

    n_neg = int(np.sum(eigvals < -noise_tol))
    if n_neg > 0:
        eig_desc = f"{n_neg} negative eigenvalue(s), smallest λ={min_eigval:.4g}"
    else:
        eig_desc = (
            f"smallest eigenvalue λ={min_eigval:.4g} is below the invertibility "
            f"floor (rank-deficient, e.g. an exact ±1 correlation between two "
            f"systematics)"
        )

    if method == "clip":
        eigvals_full, eigvecs = np.linalg.eigh(A)
        eigvals_clipped = np.maximum(eigvals_full, eps)
        C_reg = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        d = np.sqrt(np.diag(C_reg))
        C_reg = C_reg / np.outer(d, d)
        method_msg = f"eigenvalue clip + renorm (eps={eps:.2g})"

    elif method == "shift":
        delta = float(-min_eigval + eps)
        C_reg = A + delta * np.eye(n)
        d = np.sqrt(np.diag(C_reg))  # = sqrt(1 + delta) uniformly
        C_reg = C_reg / np.outer(d, d)
        scale = 1.0 / (1.0 + delta)
        method_msg = (
            f"diagonal shift δ={delta:.4g}, "
            f"all correlations scaled by {scale:.6f}"
        )

    elif method == "higham":
        C_reg = _higham_nearest_corr(A, eps=eps)
        method_msg = "Higham (2002) nearest correlation matrix (Frobenius-optimal)"

    else:
        raise ValueError(
            f"Unknown pd_reg_method {method!r}; choose 'clip', 'shift', or 'higham'"
        )

    mask = ~np.eye(n, dtype=bool)
    max_change = float(np.max(np.abs(C_reg[mask] - A[mask])))
    rms_change = float(np.sqrt(np.mean((C_reg[mask] - A[mask]) ** 2)))

    warnings.warn(
        f"Prior correlation matrix is not positive-definite: "
        f"{eig_desc}. "
        f"Regularised via {method_msg}. "
        f"Off-diagonal change — max |Δρ|={max_change:.4g}, RMS |Δρ|={rms_change:.4g}.",
        stacklevel=3,
    )
    if verbose:
        print(
            f"[convino] PD regularisation ({method}): "
            f"{n_neg} neg. eigenvalue(s), λ_min={min_eigval:.6g}, "
            f"max |Δρ|={max_change:.6g}, RMS |Δρ|={rms_change:.6g}",
            file=sys.stderr,
        )

    return C_reg
