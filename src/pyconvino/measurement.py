"""
Measurement setup: decompose per-measurement Hessian into the matrices
(LM, Lk_up, Lk_down, LD) needed by the chi-squared objective.

Mathematical derivation
-----------------------
Given a Hessian H partitioned into blocks (with estimates first):

    H = | M        kappa  |
        | kappa^T  tildeC |

where M is (nHest × nHest), kappa is (nHest × nHlamb), tildeC is (nHlamb × nHlamb).

Response vectors:  ki = M^{-1} kappa[:,i]

Stored as: Lk_up[mu, i] = -ki[mu], Lk_down[mu, i] = +ki[mu]
(matches C++ uncertainty(-ki[mu], +ki[mu]))

LM = M (the estimate × estimate block of H)

LD (nHlamb × nHlamb):
    LD = tildeC - kappa^T @ M^{-1} @ kappa - diag(P)
    (derived from: LD[i,j] = tildeC[i,j]
                              - sum_{mu,nu} M[mu,nu]/2 * (Lk_symm[mu,i]*Lk_symm[nu,j]
                                                          + Lk_symm[nu,i]*Lk_symm[mu,j])
                              - P_i delta_{ij}
     with Lk_symm[mu,i] = -ki[mu])

P_i is the prior precision the input fit used for nuisance i: 1 for a
unit-Gaussian constraint (default), 0 for a parameter tagged "free" in
[systematics]. Subtracting it leaves the data-only Hessian H_d = H - diag(0,P);
the prior is re-added once, globally, by the combiner.

Nuisance central values ([nuisance values] block, lambda_hat, 0 if absent or
if use_nuisance_values is False, which is the default = the C++ Convino form):
the input's data-only likelihood is (theta - theta_d)^T H_d (theta - theta_d)
with H_d theta_d = H theta_hat (stationarity of the input fit). Written in
the objective's variables (see objective.py) this is

    (d - x - k(lambda - lambda_hat))^T M (...) + (lambda - lambda_hat)^T LD (lambda - lambda_hat)
    - 2 (lambda - lambda_hat)^T P lambda_hat + chi2_offset,

    chi2_offset = lambda_hat^T P LD^+ P lambda_hat   (>= 0, zero at theta_d)

so a standalone input is reproduced exactly (theta_hat, covariance H^-1) and
its minimum chi2 is chi2_standalone = chi2_offset + lambda_hat^T P lambda_hat,
the prior penalty the input fit already paid. Without lambda_hat (the original
Convino form) the nuisances of a profiled input are silently re-centred at 0.

Externalized systematics (from [not fitted]) extend Lk with user-supplied
(up, down) pairs; LD is zero for their indices — their constraint comes
entirely from the global prior. They always have lambda_hat = 0, P = 1.

For measurements without a Hessian (pure [not fitted]), LM is built from
the stat uncertainties: LM[i,j] = corr[i,j] / (stat_i * stat_j),
defaulting to the identity (uncorrelated estimates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .parser import MeasurementFileData


@dataclass
class MeasurementSetup:
    """Pre-computed matrices for a single measurement."""

    x_meas: np.ndarray  # central values, shape (nest,)
    LM: np.ndarray  # observable Hessian, shape (nest, nest)
    Lk_up: np.ndarray  # systematic responses (upward), shape (nest, nlamb)
    Lk_down: np.ndarray  # systematic responses (downward), shape (nest, nlamb)
    LD: np.ndarray  # residual systematic matrix, shape (nlamb, nlamb)

    est_names: list[str]  # estimate names in row order of LM / x_meas
    sys_names: list[str]  # systematic names in col order of Lk / LD
    sys_types: list[str]  # "absolute" or "relative" per systematic

    # Global parameter-vector indices (filled by Combiner, initially empty)
    est_global_idx: list[int] = field(default_factory=list)
    sys_global_idx: list[int] = field(default_factory=list)

    lambda_hat: np.ndarray = None  # input post-fit nuisance values, (nlamb,); default 0
    prior_diag: np.ndarray = (
        None  # prior precision P_i of the input fit, (nlamb,); default 1
    )
    chi2_offset: float = 0.0  # lambda_hat^T P LD^+ P lambda_hat
    chi2_standalone: float = 0.0  # chi2 of this input alone at its own best fit

    def __post_init__(self):
        nlamb = len(self.sys_names)
        if self.lambda_hat is None:
            self.lambda_hat = np.zeros(nlamb)
        if self.prior_diag is None:
            self.prior_diag = np.ones(nlamb)


def setup_measurement(
    data: MeasurementFileData, use_nuisance_values: bool = False
) -> MeasurementSetup:
    """
    Build the LM/Lk/LD matrices from raw file data.

    The function mirrors measurement::setup() in the C++ source.
    """
    H_has_data = bool(data.hessian)

    # ------------------------------------------------------------------
    # Identify which parameters are estimates vs. systematics
    # ------------------------------------------------------------------
    est_names_set = set(data.estimates.keys())
    hess_est_names = [n for n in data.hessian_names if n in est_names_set]
    hess_sys_names = [n for n in data.hessian_names if n not in est_names_set]

    # Externalized estimates (unusual but possible: estimates not in H)
    ext_est_names = [n for n in data.estimates if n not in set(data.hessian_names)]
    # Externalized systematics (from [not fitted] table)
    ext_sys_names = list(data.ext_sys_names)

    # Ordered names: Hessian params first, then externalized
    all_est_names = hess_est_names + ext_est_names
    all_sys_names = hess_sys_names + ext_sys_names

    nHest = len(hess_est_names)
    nHlamb = len(hess_sys_names)
    nest = len(all_est_names)
    nlamb = len(all_sys_names)

    hess_sys_set = set(hess_sys_names)
    for label, names in (
        ("[nuisance values]", data.nuisance_values),
        ("free", data.prior_free),
    ):
        unknown = sorted(set(names) - hess_sys_set)
        if unknown:
            raise ValueError(
                f"{data.path}: {label} entries {unknown} are not Hessian "
                "systematics (only nuisances of the input fit can carry a "
                "central value or be free)"
            )

    x_meas = np.array([data.estimates[n] for n in all_est_names], dtype=float)
    stat_errs = np.array(
        [data.stat_errors.get(n, 0.0) for n in all_est_names], dtype=float
    )

    # ------------------------------------------------------------------
    # Reorder Hessian: estimates before systematics
    # ------------------------------------------------------------------
    M = np.zeros((nHest, nHest))
    kappa = np.zeros((nHest, nHlamb))
    tildeC = np.zeros((nHlamb, nHlamb))
    TM = np.zeros((nHest, nHest))

    if H_has_data:
        H_orig = np.array(data.hessian, dtype=float)
        new_order = hess_est_names + hess_sys_names
        orig_idx = [data.hessian_names.index(n) for n in new_order]
        H = H_orig[np.ix_(orig_idx, orig_idx)]
        if nHest > 0:
            M = H[:nHest, :nHest]
            kappa = H[:nHest, nHest:]
            tildeC = H[nHest:, nHest:]
            try:
                TM = np.linalg.inv(M)
            except np.linalg.LinAlgError as err:
                raise ValueError(
                    f"{data.path}: estimate-block of the Hessian is singular "
                    "and cannot be inverted"
                ) from err

    # ------------------------------------------------------------------
    # Build LM (observable Hessian)
    # ------------------------------------------------------------------
    # Hessian block first; the other (externalized) estimates get 1/stat^2 on
    # the diagonal (identity correlation).
    LM = np.zeros((nest, nest))
    start = nHest if H_has_data else 0
    LM[:start, :start] = M
    i = start + np.flatnonzero(stat_errs[start:] > 0)
    LM[i, i] = 1.0 / stat_errs[i] ** 2

    # ------------------------------------------------------------------
    # Build Lk (systematic response vectors)
    # ------------------------------------------------------------------
    Lk_up = np.zeros((nest, nlamb))
    Lk_down = np.zeros((nest, nlamb))

    # Hessian-derived systematics: ki = TM @ kappa[:,i]
    # Convention: Lk_up[mu, i] = -ki[mu],  Lk_down[mu, i] = +ki[mu]
    if nHest > 0 and nHlamb > 0:
        ki_matrix = TM @ kappa  # shape (nHest, nHlamb)
        Lk_up[:nHest, :nHlamb] = -ki_matrix
        Lk_down[:nHest, :nHlamb] = ki_matrix

    # Externalized systematics: directly from [not fitted] table
    for i, sys_name in enumerate(ext_sys_names):
        col = nHlamb + i
        for mu, est_name in enumerate(all_est_names):
            if (
                est_name in data.externalized
                and sys_name in data.externalized[est_name]
            ):
                Lk_up[mu, col], Lk_down[mu, col] = data.externalized[est_name][sys_name]

    # ------------------------------------------------------------------
    # Build LD (residual systematic matrix, top-left nHlamb × nHlamb block)
    # ------------------------------------------------------------------
    prior_diag = np.array(
        [0.0 if n in data.prior_free else 1.0 for n in all_sys_names], dtype=float
    )
    nuis = data.nuisance_values if use_nuisance_values else {}
    lambda_hat = np.array([nuis.get(n, 0.0) for n in all_sys_names], dtype=float)

    LD = np.zeros((nlamb, nlamb))
    if nHlamb > 0:
        # LD = tildeC - kappa^T @ TM @ kappa - diag(P)
        LD[:nHlamb, :nHlamb] = (
            tildeC - kappa.T @ TM @ kappa - np.diag(prior_diag[:nHlamb])
        )

    # Sanity check matching C++ diagnostic
    for i in range(nHlamb):
        if LD[i, i] + prior_diag[i] < 0:
            raise ValueError(
                f"LD diagonal +P < 0 at index {i} (sys={all_sys_names[i]}). "
                "Check Hessian input."
            )

    # Constant making the data-only term vanish at the input's unconstrained
    # optimum (see module docstring). pinv: LD is singular along directions
    # the data do not constrain, where P*lambda_hat is 0 anyway.
    chi2_offset = 0.0
    p_lam = prior_diag[:nHlamb] * lambda_hat[:nHlamb]
    if nHlamb > 0 and np.any(p_lam):
        chi2_offset = float(
            p_lam @ np.linalg.pinv(LD[:nHlamb, :nHlamb], hermitian=True) @ p_lam
        )
    chi2_standalone = chi2_offset + float(lambda_hat @ (prior_diag * lambda_hat))

    # ------------------------------------------------------------------
    # Systematic types
    # ------------------------------------------------------------------
    sys_types = [data.sys_types.get(name, "absolute") for name in all_sys_names]

    return MeasurementSetup(
        x_meas=x_meas,
        LM=LM,
        Lk_up=Lk_up,
        Lk_down=Lk_down,
        LD=LD,
        est_names=all_est_names,
        sys_names=all_sys_names,
        sys_types=sys_types,
        lambda_hat=lambda_hat,
        prior_diag=prior_diag,
        chi2_offset=chi2_offset,
        chi2_standalone=chi2_standalone,
    )
