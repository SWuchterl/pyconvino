"""Self-consistency test for the ndf/chi2_per_ndf/p_value goodness-of-fit fields.

None of the committed ConvinoSetups fixtures actually average redundant
measurements of the *same* observable (every [observables] entry in
ConvinoSetups/*/rho_config.txt maps exactly one estimate to one combined
name — verified by grepping all of them), so every existing golden setup has
ndf == 0 by construction (see combiner.py's CombinationResult.ndf docstring
for the convention). This test builds the smallest case that actually
exercises ndf > 0: two independent stat-only measurements of one shared
observable, combined via a weighted average, where the chi2-minimum has a
well-known closed form.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scipy.stats import chi2 as chi2_dist

from convino_jax.combiner import Combiner

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


class GoodnessOfFitTest(unittest.TestCase):
    def test_two_measurement_average_has_closed_form_chi2(self):
        x1, s1 = 100.0, 2.0
        x2, s2 = 104.0, 3.0

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

            result = Combiner.from_config(str(tdp / "config.txt")).combine()

        # 2 measurements, 1 combined observable, no systematics: ndf = 2 - 1 = 1.
        self.assertEqual(result.nest, 1)
        self.assertEqual(result.nsys, 0)
        self.assertEqual(result.ndf, 1)

        # Closed-form chi2 minimum of a two-point weighted average:
        # chi2_min = (x1 - x2)^2 / (s1^2 + s2^2).
        expected_chi2 = (x1 - x2) ** 2 / (s1**2 + s2**2)
        self.assertAlmostEqual(result.chi2_min, expected_chi2, places=6)
        self.assertAlmostEqual(result.chi2_per_ndf, expected_chi2 / 1, places=6)
        self.assertAlmostEqual(
            result.p_value, float(chi2_dist.sf(expected_chi2, 1)), places=6
        )

    def test_single_measurement_has_no_degrees_of_freedom(self):
        # A single measurement directly defining its own observable (no
        # redundancy) has nothing left to test for consistency: ndf == 0,
        # chi2_min == 0 exactly, and chi2_per_ndf/p_value are NaN rather than
        # a misleading "perfect fit" p-value of 1.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "measA.txt").write_text(
                _MEASUREMENT_TEMPLATE.format(name="measA", value=100.0, stat=2.0),
                encoding="utf-8",
            )
            (tdp / "config.txt").write_text(
                """\
[global]
isDifferential = false
normalise = false
[end global]

[inputs]
nFiles = 1
file0 = measA.txt
[end inputs]

[observables]
combined = measA
[end observables]
""",
                encoding="utf-8",
            )

            result = Combiner.from_config(str(tdp / "config.txt")).combine()

        self.assertEqual(result.ndf, 0)
        self.assertAlmostEqual(result.chi2_min, 0.0, places=8)
        self.assertTrue(result.chi2_per_ndf != result.chi2_per_ndf)  # NaN
        self.assertTrue(result.p_value != result.p_value)  # NaN


if __name__ == "__main__":
    unittest.main()
