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
    LD = tildeC - kappa^T @ M^{-1} @ kappa - I
    (derived from: LD[i,j] = tildeC[i,j]
                              - sum_{mu,nu} M[mu,nu]/2 * (Lk_symm[mu,i]*Lk_symm[nu,j]
                                                          + Lk_symm[nu,i]*Lk_symm[mu,j])
                              - delta_{ij}
     with Lk_symm[mu,i] = -ki[mu])

Externalized systematics (from [not fitted]) extend Lk with user-supplied
(up, down) pairs; LD is zero for their indices — their constraint comes
entirely from the global prior.

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
    x_meas: np.ndarray        # central values, shape (nest,)
    LM: np.ndarray            # observable Hessian, shape (nest, nest)
    Lk_up: np.ndarray         # systematic responses (upward), shape (nest, nlamb)
    Lk_down: np.ndarray       # systematic responses (downward), shape (nest, nlamb)
    LD: np.ndarray            # residual systematic matrix, shape (nlamb, nlamb)

    est_names: list[str]      # estimate names in row order of LM / x_meas
    sys_names: list[str]      # systematic names in col order of Lk / LD
    sys_types: list[str]      # "absolute" or "relative" per systematic

    # Global parameter-vector indices (filled by Combiner, initially empty)
    est_global_idx: list[int] = field(default_factory=list)
    sys_global_idx: list[int] = field(default_factory=list)


def setup_measurement(data: MeasurementFileData) -> MeasurementSetup:
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

    x_meas = np.array([data.estimates[n] for n in all_est_names], dtype=float)
    stat_errs = np.array([data.stat_errors.get(n, 0.0) for n in all_est_names], dtype=float)

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
            TM = np.linalg.inv(M)

    # ------------------------------------------------------------------
    # Build LM (observable Hessian)
    # ------------------------------------------------------------------
    LM = np.zeros((nest, nest))
    if H_has_data and nHest > 0:
        LM[:nHest, :nHest] = M
        # Externalized estimates (if any): use their stat uncertainties
        for i in range(nHest, nest):
            if stat_errs[i] > 0:
                LM[i, i] = 1.0 / stat_errs[i] ** 2
    else:
        # Pure externalized: LM from stat uncertainties (identity correlation)
        for i in range(nest):
            if stat_errs[i] > 0:
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
            if est_name in data.externalized and sys_name in data.externalized[est_name]:
                entry = data.externalized[est_name][sys_name]
                Lk_up[mu, col] = entry.up
                Lk_down[mu, col] = entry.down

    # ------------------------------------------------------------------
    # Build LD (residual systematic matrix, top-left nHlamb × nHlamb block)
    # ------------------------------------------------------------------
    LD = np.zeros((nlamb, nlamb))
    if nHlamb > 0:
        # LD = tildeC - kappa^T @ TM @ kappa - I
        LD[:nHlamb, :nHlamb] = (
            tildeC
            - kappa.T @ TM @ kappa
            - np.eye(nHlamb)
        )

    # Sanity check matching C++ diagnostic
    for i in range(nHlamb):
        if LD[i, i] + 1.0 < 0:
            raise ValueError(
                f"LD diagonal +1 < 0 at index {i} (sys={all_sys_names[i]}). "
                "Check Hessian input."
            )

    # ------------------------------------------------------------------
    # Systematic types
    # ------------------------------------------------------------------
    sys_types = []
    for name in all_sys_names:
        t = data.sys_types.get(name, "absolute")
        sys_types.append(t)

    return MeasurementSetup(
        x_meas=x_meas,
        LM=LM,
        Lk_up=Lk_up,
        Lk_down=Lk_down,
        LD=LD,
        est_names=all_est_names,
        sys_names=all_sys_names,
        sys_types=sys_types,
    )
