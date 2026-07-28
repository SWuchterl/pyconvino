"""Self-consistency test for the opt-in --nonneg-combined / nonneg_combined bound.

Two independent stat-only measurements of a shared observable, both negative,
so the unconstrained weighted average is negative too (closed form:
(x1/s1^2 + x2/s2^2) / (1/s1^2 + 1/s2^2)). This is the symmetric/all-absolute/
non-Pearson case, so it exercises the quadratic fast path by default, and its
bounded fallback to L-BFGS-B when nonneg_combined=True pushes the unconstrained
solve outside the requested [0, inf) bound.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyconvino.combiner import Combiner

_MEASUREMENT_TEMPLATE = """\
[hessian]
[end hessian]

[estimates]
n_estimates = 1
name_0 = {name}
value_0 = {value}
[end estimates]

[not fitted]
[end not fitted]

[correlation matrix]
{name}   ({stat}) 1.0000
[end correlation matrix]

[systematics]
[end systematics]
"""

_CONFIG_TEMPLATE = """\
[global]
isDifferential = false
normalise = false
[end global]

[inputs]
nFiles = 2
file0 = measA.txt
file1 = measB.txt
[end inputs]

[observables]
combined = measA + measB
[end observables]
"""


class NonnegCombinedTest(unittest.TestCase):
    def _combine(self, nonneg_combined: bool):
        x1, s1 = -5.0, 1.0
        x2, s2 = -3.0, 1.0

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "measA.txt").write_text(
                _MEASUREMENT_TEMPLATE.format(name="measA", value=x1, stat=s1),
                encoding="utf-8",
            )
            (tdp / "measB.txt").write_text(
                _MEASUREMENT_TEMPLATE.format(name="measB", value=x2, stat=s2),
                encoding="utf-8",
            )
            (tdp / "config.txt").write_text(_CONFIG_TEMPLATE, encoding="utf-8")

            return Combiner.from_config(
                str(tdp / "config.txt"), nonneg_combined=nonneg_combined
            ).combine()

    def test_default_is_unbounded_and_negative(self):
        result = self._combine(nonneg_combined=False)
        expected = (-5.0 / 1.0**2 + -3.0 / 1.0**2) / (1.0 / 1.0**2 + 1.0 / 1.0**2)
        self.assertAlmostEqual(result.combined_values[0], expected, places=6)
        self.assertLess(result.combined_values[0], 0.0)
        self.assertTrue(result.converged)

    def test_nonneg_combined_clips_to_zero(self):
        result = self._combine(nonneg_combined=True)
        self.assertAlmostEqual(result.combined_values[0], 0.0, places=6)
        self.assertTrue(result.converged)

        unconstrained = self._combine(nonneg_combined=False)
        self.assertGreater(result.chi2_min, unconstrained.chi2_min)


if __name__ == "__main__":
    unittest.main()
