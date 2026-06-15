"""
JAX-jitted chi-squared objective function.

Global parameter layout:
    pars[:nsys]  = nuisance pulls (lambda)
    pars[nsys:]  = combined observable values (x_comb), indexed by combined-obs index

For each measurement k the chi-squared contribution is:

    chi2_obs_k  = residual^T  LM  residual
    chi2_LD_k   = lambda_k^T  LD  lambda_k

where residual = x_meas - x_shifted and x_shifted is x_comb with all
systematic shifts applied (relative first, then absolute, matching C++).

Global prior:   chi2_prior = lambda_all^T  inv_C  lambda_all

Systematic eval:
    Lk_eval[mu,i] = Lk_up[mu,i] * lambda_i     if lambda_i >= 0
                  = Lk_down[mu,i] * |lambda_i|  if lambda_i <  0
"""

from __future__ import annotations
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_chi2(
    setups,
    inv_C: np.ndarray,
    nsys: int,
    nest: int,
    use_pearson: bool = False,
) -> Callable:
    """
    Return a JIT-compiled chi2(pars) function.

    Parameters
    ----------
    setups      : list of MeasurementSetup with est_global_idx and sys_global_idx set
    inv_C       : (nsys, nsys) inverse prior covariance
    nsys, nest  : parameter counts
    use_pearson : Pearson chi2 scaling (default Neyman)
    """
    # Convert to JAX arrays once
    meas_jax = []
    for ms in setups:
        meas_jax.append({
            "x_meas":   jnp.array(ms.x_meas),
            "LM":       jnp.array(ms.LM),
            "Lk_up":    jnp.array(ms.Lk_up),   # (nest_local, nlamb_local)
            "Lk_down":  jnp.array(ms.Lk_down),
            "LD":       jnp.array(ms.LD),
            "est_idx":  jnp.array(ms.est_global_idx, dtype=jnp.int32),
            "sys_idx":  jnp.array(ms.sys_global_idx, dtype=jnp.int32),
            "rel_mask": jnp.array([t == "relative" for t in ms.sys_types]),
        })
    inv_C_jax = jnp.array(inv_C)
    _eps = jnp.asarray(np.finfo(float).tiny)

    def _x_shifted(x_comb_local, x_meas, Lk_up, Lk_down, lambdas, rel_mask):
        """Vectorised systematic shift: relative first, then absolute."""
        # Lk_eval[mu, i]: shape (nest_local, nlamb_local)
        lam = lambdas[jnp.newaxis, :]           # (1, nlamb)
        k_eval = jnp.where(
            lam >= 0.0,
            Lk_up * lam,
            Lk_down * jnp.abs(lam),
        )  # (nest_local, nlamb_local)

        # Relative systematics: multiplicative
        x_safe = jnp.where(jnp.abs(x_meas) > 0, x_meas, _eps)
        rel_factor = jnp.where(rel_mask, 1.0 - k_eval / x_safe[:, jnp.newaxis], 1.0)
        x = x_comb_local * jnp.prod(rel_factor, axis=1)

        # Absolute systematics: additive
        abs_shift = jnp.where(rel_mask, 0.0, k_eval)
        x = x - jnp.sum(abs_shift, axis=1)
        return x

    def chi2(pars: jnp.ndarray) -> jnp.ndarray:
        pars = jnp.asarray(pars)
        total = jnp.zeros(())

        for md in meas_jax:
            x_meas      = md["x_meas"]
            LM          = md["LM"]
            Lk_up       = md["Lk_up"]
            Lk_down     = md["Lk_down"]
            LD          = md["LD"]
            est_idx     = md["est_idx"]
            sys_idx     = md["sys_idx"]
            rel_mask    = md["rel_mask"]

            x_comb_local  = pars[nsys + est_idx]   # (nest_local,)
            lambdas_local = pars[sys_idx]            # (nlamb_local,)

            x_sh = _x_shifted(x_comb_local, x_meas, Lk_up, Lk_down,
                               lambdas_local, rel_mask)
            residual = x_meas - x_sh  # (nest_local,)

            # Observable chi2
            if use_pearson:
                x_safe = jnp.where(jnp.abs(x_sh) > 0, x_sh,
                                   jnp.sign(x_sh + _eps) * _eps)
                scale = jnp.abs(x_meas / x_safe)           # (nest_local,)
                scale_mat = jnp.sqrt(jnp.outer(scale, scale))
                LM_eff = LM * scale_mat
            else:
                LM_eff = LM

            total += residual @ LM_eff @ residual

            # Systematic residual chi2
            total += lambdas_local @ LD @ lambdas_local

        # Global prior
        lambdas_all = pars[:nsys]
        total += lambdas_all @ inv_C_jax @ lambdas_all

        return total

    return jax.jit(chi2)
