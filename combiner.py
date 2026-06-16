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
6. Two-pass minimization with scipy L-BFGS-B:
      pass 1: loose tolerance (fast, gets near the minimum)
      pass 2: tight tolerance (precision fit)
7. Extract asymmetric errors via profile likelihood (MINOS equivalent).
8. Compute post-fit covariance from the Hessian of chi2 at the minimum.
9. Compute uncertainty impacts by re-fitting with systematic groups frozen.
10. Return a CombinationResult object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, brentq
from scipy.linalg import inv as scipy_inv

jax.config.update("jax_enable_x64", True)

from .parser import (
    ConfigData,
    MeasurementFileData,
    parse_config_file,
    parse_measurement_file,
)
from .measurement import MeasurementSetup, setup_measurement
from .objective import make_chi2


@dataclass
class CombinationResult:
    """All quantities needed to produce the output text file."""
    # Minimizer outcome
    chi2_min: float = 0.0
    converged: bool = False

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

    # Best-fit full parameter vector
    pars_best: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nsys: int = 0
    nest: int = 0


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
        prefix: str = "convino",
    ):
        self.config = config
        self.meas_data = meas_data
        self.use_pearson = use_pearson
        self.prefix = prefix

    @classmethod
    def from_config(
        cls,
        config_path: str,
        use_pearson: bool = False,
        prefix: str = "convino",
    ) -> "Combiner":
        cfg = parse_config_file(config_path)
        meas_data = [parse_measurement_file(p) for p in cfg.measurement_files]
        return cls(cfg, meas_data, use_pearson=use_pearson, prefix=prefix)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(self) -> CombinationResult:
        cfg = self.config

        # 1. Setup measurements
        setups = [setup_measurement(d) for d in self.meas_data]

        # 2. Build global parameter index layout (mutates each setup in place)
        all_sys_names, combined_names, nsys, nest = self._assign_global_indices(setups)

        # 3. Build prior inverse-covariance
        inv_C, C_exact = self._build_prior(all_sys_names)

        # 4. Build chi2
        chi2_fn = make_chi2(setups, inv_C, nsys, nest, self.use_pearson)
        grad_fn = jax.jit(jax.grad(chi2_fn))
        hess_fn = jax.jit(jax.hessian(chi2_fn))
        # Eager gradient (constructed once) for the profile/impact refits. The
        # jitted grad_fn is NOT reused there: jit reassociates float ops at the
        # ~1e-14 level, which the impact quadrature sqrt(err_full^2 - err_frozen^2)
        # amplifies into the printed digits. The main fit uses the jitted grad_fn.
        grad_eager = jax.grad(chi2_fn)

        # 5. Initial parameter vector
        x0 = self._initial_params(setups, nsys, nest)

        # 6. Two-pass minimization
        pars_best, chi2_min, converged = self._minimize(chi2_fn, grad_fn, x0)

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
        H_fit = np.array(hess_fn(pars_best))
        try:
            cov_fit = 2.0 * scipy_inv(H_fit)
        except np.linalg.LinAlgError:
            cov_fit = np.full_like(H_fit, np.nan)
            warnings.warn("Post-fit Hessian not invertible; covariance set to NaN")

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

        responses_symmetric = all(
            np.allclose(s.Lk_up, -s.Lk_down, atol=1e-9, rtol=1e-6)
            for s in setups
        )

        combined_err_up = hesse_errs.copy()
        combined_err_down = hesse_errs.copy()
        if not responses_symmetric:
            for k in range(nest):
                eu, ed = self._profile_error(
                    chi2_fn, grad_eager, pars_best, chi2_min, nsys + k, hesse_errs[k]
                )
                combined_err_up[k] = eu
                combined_err_down[k] = ed

        # 9. Pulls and constraints
        pulls = np.array(pars_best[:nsys])
        diag_cov_sys = np.sqrt(np.maximum(np.diag(cov_fit)[:nsys], 0.0))
        constraints = diag_cov_sys  # post-fit sigma / prior sigma (prior sigma = 1)

        # 10. Correlation / covariance from post-fit Hessian
        diag_std = np.sqrt(np.maximum(np.diag(cov_fit), 1e-30))
        corr_full = cov_fit / np.outer(diag_std, diag_std)
        np.fill_diagonal(corr_full, 1.0)

        # 11. Pre-combine systematic correlations (exact user-specified prior C)
        pre_sys_corr = C_exact

        # 12. Uncertainty impacts (and per-group frozen covariance matrices)
        corr_est = corr_full[nsys:, nsys:]
        impact_groups, impact_cov_groups = self._compute_impacts(
            chi2_fn, grad_eager, pars_best, chi2_min,
            all_sys_names, combined_err_up, combined_err_down,
            nsys, nest, corr_est, responses_symmetric, H_fit,
        )

        return CombinationResult(
            chi2_min=float(chi2_min),
            converged=converged,
            combined_names=combined_names,
            combined_values=combined_vals,
            combined_err_up=combined_err_up,
            combined_err_down=combined_err_down,
            sys_names=all_sys_names,
            pulls=pulls,
            constraints=constraints,
            corr_full=corr_full,
            cov_full=cov_fit,
            all_names=all_sys_names + combined_names,
            pre_sys_corr=pre_sys_corr,
            impact_groups=impact_groups,
            impact_cov_groups=impact_cov_groups,
            pars_best=np.array(pars_best),
            nsys=nsys,
            nest=nest,
        )

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
        all_sys_names: list[str] = []
        seen_sys: set[str] = set()
        for ms in setups:
            for name in ms.sys_names:
                if name not in seen_sys:
                    all_sys_names.append(name)
                    seen_sys.add(name)

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

    def _build_prior(self, all_sys_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the (nsys × nsys) inverse prior covariance.

        The prior covariance C has:
          C[i,i] = 1  (unit prior for each systematic)
          C[i,j] = rho_{ij}  (from [correlations] block in config)

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
                    continue
                rho = sr.nominal
                C[ia, ib] = rho
                C[ib, ia] = rho

        C_exact = C.copy()  # exact user values for display
        # Ensure positive definiteness (reflect negative eigenvalues)
        C = _nearest_positive_definite(C)
        return scipy_inv(C), C_exact

    def _initial_params(
        self,
        setups: list[MeasurementSetup],
        nsys: int,
        nest: int,
    ) -> np.ndarray:
        """
        Start from: lambda = 0 for all systematics,
        x_comb = mean of contributing measurements for each combined observable.
        """
        x0 = np.zeros(nsys + nest)

        # For each combined observable, average over measurements that contribute.
        # ms.est_global_idx[k] is the combined index of estimate est_names[k]
        # (assigned in _assign_global_indices).
        comb_sum = np.zeros(nest)
        comb_cnt = np.zeros(nest, dtype=int)
        for ms in setups:
            for k, c_idx in enumerate(ms.est_global_idx):
                comb_sum[c_idx] += ms.x_meas[k]
                comb_cnt[c_idx] += 1

        for i in range(nest):
            if comb_cnt[i] > 0:
                x0[nsys + i] = comb_sum[i] / comb_cnt[i]
            else:
                x0[nsys + i] = 1.0  # fallback

        return x0

    def _minimize(self, chi2_fn, grad_fn, x0: np.ndarray):
        """Two-pass L-BFGS-B minimization."""
        def _scipy_grad(pars):
            return np.array(grad_fn(jnp.array(pars, dtype=jnp.float64)))

        # Pass 1: loose tolerance
        res1 = minimize(
            lambda p: float(chi2_fn(jnp.array(p, dtype=jnp.float64))),
            x0,
            method="L-BFGS-B",
            jac=_scipy_grad,
            options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-5},
        )

        # Pass 2: precision
        res2 = minimize(
            lambda p: float(chi2_fn(jnp.array(p, dtype=jnp.float64))),
            res1.x,
            method="L-BFGS-B",
            jac=_scipy_grad,
            options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-9},
        )

        # res2.success can be False even at a perfectly good minimum (L-BFGS-B
        # line-search quirks near machine precision), so fall back to an
        # explicit stationarity test on the gradient. The previous fallback
        # (res2.fun < res1.fun + 1e-6) was meaningless: pass 2 starts from
        # pass 1's point, so it is true for essentially every fit, converged or
        # not.
        gnorm = float(np.linalg.norm(
            np.array(grad_fn(jnp.array(res2.x, dtype=jnp.float64)))
        ))
        converged = bool(res2.success or gnorm < 1e-4)
        return jnp.array(res2.x, dtype=jnp.float64), res2.fun, converged

    def _minimize_frozen(self, chi2_fn, grad_fn, p_init, frozen_idx, frozen_vals, options):
        """
        Minimize chi2 over all parameters except `frozen_idx`, which are pinned
        to `frozen_vals` (scalar or array). `grad_fn` (built once by the caller)
        supplies the analytic jacobian instead of rebuilding jax.grad per call.

        Returns (scipy OptimizeResult, free_mask).
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
        return res, free_mask

    def _profile_error(
        self,
        chi2_fn,
        grad_fn,
        pars_best: jnp.ndarray,
        chi2_min: float,
        param_idx: int,
        sigma_sym: float,
        target_delta: float = 1.0,
        extra_frozen_idx=None,
        extra_frozen_vals=None,
    ) -> tuple[float, float]:
        """
        Find asymmetric ±1σ errors via profile likelihood:
        profile_chi2(v) = min_{pars except param_idx} chi2(pars)  at  pars[param_idx] = v

        Solves: profile_chi2(v) - chi2_min = target_delta = 1

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
            res, _ = self._minimize_frozen(
                chi2_fn, grad_fn, pars_arr,
                [param_idx] + extra_idx, [float(v)] + extra_vals, _opts,
            )
            return res.fun

        target = chi2_min + target_delta

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
                "falling back to the symmetric HESSE error"
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
                "falling back to the symmetric HESSE error"
            )
            ed = sigma_sym  # fallback

        return float(eu), float(ed)

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
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
        """
        For each impact group, freeze the group's systematics at 0 and compute
        the impact in quadrature vs. the full error.

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
        build each per-group covariance (see combiner.cpp printout).

        Returns (impacts, frozen_covs) where:
          impacts[label]     = (impact_up, impact_down) per combined observable
          frozen_covs[label] = (nest x nest) covariance of the combined
                               observables with that group frozen
        """
        sys_to_idx = {n: i for i, n in enumerate(all_sys_names)}
        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        cov_results: dict[str, np.ndarray] = {}
        n = nsys + nest

        for label, members in self.config.impact_groups.items():
            frozen_idx = [sys_to_idx[m] for m in members if m in sys_to_idx]
            if not frozen_idx:
                warnings.warn(
                    f"impact group '{label}' has no members matching any known "
                    f"systematic ({members}); skipping it"
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
                if not responses_symmetric:
                    res, _ = self._minimize_frozen(
                        chi2_fn, grad_fn, pars_best, frozen_idx, 0.0,
                        {"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
                    )
                    p_frozen_full = np.array(pars_best, dtype=np.float64)
                    p_frozen_full[free_mask] = res.x
                    p_frozen_full[frozen_idx] = 0.0
                    eu_list, ed_list = [], []
                    for k in range(nest):
                        eu_f, ed_f = self._profile_error(
                            chi2_fn, grad_fn, p_frozen_full, res.fun,
                            nsys + k, float(err_frozen_up[k]),
                            extra_frozen_idx=frozen_idx,
                            extra_frozen_vals=[0.0] * len(frozen_idx),
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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """
    Return the nearest positive-definite matrix to A.
    Negative eigenvalues are reflected to a small positive epsilon.

    Warns (once per call) when this actually changes the matrix, i.e. when at
    least one eigenvalue was negative beyond floating-point noise — not when
    every eigenvalue was already at or above the epsilon floor. A user whose
    prior correlation matrix was not positive-definite should be told that it
    got silently regularised; a matrix that was already fine (e.g. identity)
    should not generate noise on every run.
    """
    if A.size == 0:
        return A  # no systematics in the prior (e.g. stat-only combination)
    A = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A)
    eps = 1e-10 * max(1.0, np.max(np.abs(eigvals)))
    # Use a small negative tolerance so genuine floating-point noise around 0
    # (e.g. eigvals == -1e-16 for an already-PD matrix) does not trigger the
    # warning; only eigenvalues meaningfully below the epsilon floor count.
    if np.any(eigvals < -eps):
        warnings.warn(
            "Prior correlation matrix was not positive-definite "
            "(had negative eigenvalues); it has been regularized by "
            "reflecting negative eigenvalues to a small positive epsilon."
        )
    eigvals_clipped = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
