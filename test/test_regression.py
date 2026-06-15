"""Numeric regression tests against stored golden snapshots.

Run with either:
    python -m unittest test.test_regression
    pytest test/test_regression.py

If the numeric output is changed *intentionally*, regenerate the goldens with
    python -m test.regen_golden
and commit the updated test/golden/*.npz.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from .helpers import SETUPS, SLOW_SETUPS, compute_arrays, golden_path

_RUN_SLOW = bool(os.environ.get("CONVINO_SLOW_TESTS"))

# Per-key comparison tolerances. Impacts are sqrt(err_full^2 - err_frozen^2),
# which amplifies small differences, so they are checked more loosely than the
# directly-fitted quantities.
_DEFAULT_TOL = dict(rtol=1e-5, atol=1e-8)
_TOL = {
    "impact_up": dict(rtol=1e-4, atol=1e-6),
    "impact_down": dict(rtol=1e-4, atol=1e-6),
}


class RegressionTest(unittest.TestCase):
    def _check_setup(self, name: str) -> None:
        if name in SLOW_SETUPS and not _RUN_SLOW:
            self.skipTest(
                f"slow setup '{name}'; set CONVINO_SLOW_TESTS=1 to run it"
            )
        gpath = golden_path(name)
        if not gpath.exists():
            self.skipTest(
                f"golden {gpath.name} missing; run `python -m test.regen_golden {name}`"
            )

        golden = np.load(gpath, allow_pickle=False)
        actual = compute_arrays(name)

        # Every golden array must still be produced (a golden may store a
        # subset — large derived matrices are omitted for slow setups).
        missing = set(golden.files) - set(actual)
        self.assertFalse(missing, f"{name}: golden arrays no longer produced: {missing}")

        for key in golden.files:
            with self.subTest(setup=name, array=key):
                exp = golden[key]
                got = np.asarray(actual[key])
                self.assertEqual(exp.shape, got.shape,
                                 f"{name}/{key}: shape changed")

                if exp.dtype.kind in ("U", "S", "b"):
                    # strings (impact labels) / bools (converged): exact match
                    self.assertTrue(np.array_equal(exp, got),
                                    f"{name}/{key}: changed")
                else:
                    tol = _TOL.get(key, _DEFAULT_TOL)
                    if not np.allclose(got, exp, **tol):
                        diff = np.abs(got - exp)
                        i = int(np.nanargmax(diff))
                        self.fail(
                            f"{name}/{key}: max abs diff {diff.flat[i]:.3e} "
                            f"(golden={exp.flat[i]:.6g}, got={got.flat[i]:.6g}) "
                            f"exceeds tol {tol}"
                        )


def _make_test(name: str):
    def test(self):
        self._check_setup(name)
    test.__name__ = f"test_{name}"
    return test


for _name in SETUPS:
    setattr(RegressionTest, f"test_{_name}", _make_test(_name))


if __name__ == "__main__":
    unittest.main()
