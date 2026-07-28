"""Compare all three PD-regularisation methods on the extreme correlation setup.

This test builds the prior correlation matrix from
``Combination_ATLAS813CMS13_corrExtremeTest`` (the setup designed to produce a
non-positive-definite prior), runs all three regularisation methods, and prints
a side-by-side summary of the diagnostics and the resulting combined values.

It is gated behind ``CONVINO_SLOW_TESTS=1`` because the full combination
(even without impacts) takes on the order of a minute.  Run it manually to
compare methods before choosing the default:

    CONVINO_SLOW_TESTS=1 python -m pytest test/test_pd_regularization_extreme.py -s
"""

from __future__ import annotations

import os
import unittest
import warnings

import numpy as np

from .helpers import SETUPS_DIR


EXTREME_CONFIG = str(
    SETUPS_DIR / "Combination_ATLAS813CMS13_corrExtremeTest" / "rho_config.txt"
)
METHODS = ["shift", "clip", "higham"]


@unittest.skipUnless(
    os.environ.get("CONVINO_SLOW_TESTS"),
    "set CONVINO_SLOW_TESTS=1 to run the extreme-case method comparison",
)
class PDRegExtremeComparisonTest(unittest.TestCase):
    """Run the combination with each method and compare diagnostics + results."""

    @classmethod
    def setUpClass(cls):
        from pyconvino import Combiner

        cls.results = {}
        cls.warnings = {}

        for method in METHODS:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = Combiner.from_config(
                    EXTREME_CONFIG,
                    compute_impacts=False,
                    pd_reg_method=method,
                ).combine()
            cls.results[method] = result
            cls.warnings[method] = [
                str(w.message)
                for w in caught
                if issubclass(w.category, UserWarning)
                and "not positive-definite" in str(w.message)
            ]

        # Print the comparison table to stdout for human inspection.
        cls._print_summary()

    @classmethod
    def _print_summary(cls):
        print("\n" + "=" * 72)
        print("PD regularisation comparison — Combination_ATLAS813CMS13_corrExtremeTest")
        print("=" * 72)

        for method in METHODS:
            msgs = cls.warnings[method]
            print(f"\n--- method: {method!r} ---")
            if msgs:
                for m in msgs:
                    print(f"  WARNING: {m}")
            else:
                print("  (no PD regularisation warning — matrix was already PD)")

        print("\nCombined values per observable:")
        ref = cls.results[METHODS[0]]
        header = f"{'Observable':<55}" + "".join(f"  {m:>8}" for m in METHODS)
        print(header)
        print("-" * len(header))
        for i, name in enumerate(ref.combined_names):
            row = f"{name:<55}"
            for method in METHODS:
                row += f"  {cls.results[method].combined_values[i]:8.4f}"
            print(row)

        print("\nMax |Δvalue| relative to 'shift' method:")
        shift_vals = np.asarray(cls.results["shift"].combined_values)
        for method in METHODS:
            if method == "shift":
                continue
            delta = np.max(np.abs(np.asarray(cls.results[method].combined_values) - shift_vals))
            print(f"  {method}: max |Δ| = {delta:.6g}")

        print("=" * 72 + "\n")

    # -- actual assertions (all methods must produce valid, convergent results) --

    def test_all_methods_converge(self):
        for method in METHODS:
            self.assertTrue(
                self.results[method].converged,
                f"method '{method}' did not converge",
            )

    def test_all_methods_triggered_pd_warning(self):
        for method in METHODS:
            self.assertTrue(
                len(self.warnings[method]) > 0,
                f"method '{method}' should have triggered a PD warning on this setup",
            )

    def test_combined_values_agree_to_tolerance(self):
        """All three methods should give combined values that agree to 1 %."""
        ref = np.asarray(self.results["shift"].combined_values)
        for method in ["clip", "higham"]:
            vals = np.asarray(self.results[method].combined_values)
            np.testing.assert_allclose(
                vals, ref, rtol=0.01,
                err_msg=f"method '{method}' combined values deviate > 1 % from 'shift'",
            )
