"""Regression test for the nonlinear/general minimization path.

`Combiner._minimize` (combiner.py) has two strategies:
  - a quadratic fast path (exact one-step linear solve), taken when the chi2
    is an exact quadratic form in `pars` (symmetric systematic responses, no
    "relative" systematics, and Pearson scaling off);
  - a general path otherwise: the existing two-pass L-BFGS-B, with a single
    Newton-correction step (using the exact Hessian at the L-BFGS-B solution)
    layered on top for extra precision.

All three of the existing golden-tested setups (test/helpers.py SETUPS:
statonly, atlas13test, corrV2) use only "absolute" systematics, so none of
them exercises the general path or the Newton-polish step — this test fills
that gap.

There is no independently-verified reference value for this synthetic setup
(unlike the golden-snapshot tests, which were checked against the original
C++ Convino), so this is a self-consistency check, not a numeric regression
against a known-correct answer: it asserts that the Newton polish can only
move the solution toward (not away from) a stationary point of lower-or-equal
chi2, mirroring the safety guard implemented in `_minimize` itself.
"""

from __future__ import annotations

import unittest

import jax
import numpy as np

from pyconvino.combiner import Combiner
from pyconvino.measurement import MeasurementSetup
from pyconvino.objective import make_chi2

jax.config.update("jax_enable_x64", True)


def _make_relative_setup() -> MeasurementSetup:
    """Two estimates tied to one combined observable via a relative (i.e.
    multiplicative on x_comb) systematic plus one absolute systematic, so the
    chi2 is genuinely nonlinear in `pars` and the quadratic fast path must not
    be taken.
    """
    nlamb = 2
    return MeasurementSetup(
        x_meas=np.array([100.0, 102.0]),
        LM=np.array([[4.0, 0.5], [0.5, 3.0]]),
        # Lk_up != -Lk_down for the relative systematic on purpose (the
        # asymmetric response also independently disqualifies the fast path,
        # though the "relative" tag alone is already sufficient).
        Lk_up=np.array([[3.0, 1.0], [2.0, 0.5]]),
        Lk_down=np.array([[-2.5, -1.0], [-1.8, -0.5]]),
        LD=np.eye(nlamb),
        est_names=["estA", "estB"],
        sys_names=["sysRel", "sysAbs"],
        sys_types=["relative", "absolute"],
        est_global_idx=[0, 1],
        sys_global_idx=[0, 1],
    )


class NonlinearMinimizeTest(unittest.TestCase):
    def test_relative_systematic_uses_general_path_and_newton_polish_helps(self):
        setup = _make_relative_setup()
        nsys, nest = 2, 2

        chi2_fn = make_chi2([setup], inv_C=np.eye(nsys), nsys=nsys, nest=nest)
        value_and_grad_fn = jax.jit(jax.value_and_grad(chi2_fn))
        grad_fn = jax.grad(chi2_fn)  # eager, only used for the test's own checks
        hess_fn = jax.jit(jax.hessian(chi2_fn))

        x0 = np.array([0.0, 0.0, 101.0, 101.0])

        combiner = Combiner.__new__(Combiner)  # no config needed for _minimize
        combiner.use_pearson = False

        # Sanity: this setup is NOT eligible for the quadratic fast path
        # (relative systematic present), matching combine()'s gating logic.
        all_absolute = all(t == "absolute" for t in setup.sys_types)
        self.assertFalse(all_absolute)

        # Run the general path (quadratic_fast_path=False) with the Newton
        # polish enabled (hess_fn passed).
        pars_polished, chi2_polished, converged = combiner._minimize(
            value_and_grad_fn, x0, hess_fn=hess_fn, quadratic_fast_path=False
        )
        self.assertTrue(converged)
        self.assertTrue(np.all(np.isfinite(np.array(pars_polished))))

        gnorm_polished = float(np.linalg.norm(np.array(grad_fn(pars_polished))))

        # Reproduce the pre-polish (plain two-pass L-BFGS-B) result by calling
        # _minimize with hess_fn=None, so it skips the Newton-correction step
        # entirely. This isolates exactly what the polish step changed.
        pars_unpolished, chi2_unpolished, _ = combiner._minimize(
            value_and_grad_fn, x0, hess_fn=None, quadratic_fast_path=False
        )
        gnorm_unpolished = float(np.linalg.norm(np.array(grad_fn(pars_unpolished))))

        # (a) the post-polish gradient norm is smaller than (or very close to)
        # the pre-polish L-BFGS-B gradient norm.
        self.assertLessEqual(gnorm_polished, gnorm_unpolished + 1e-10)

        # (b) chi2_min did not increase.
        self.assertLessEqual(chi2_polished, chi2_unpolished + 1e-10)


if __name__ == "__main__":
    unittest.main()
