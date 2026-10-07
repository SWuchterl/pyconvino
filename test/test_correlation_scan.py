"""Self-consistency test for `Combiner.scan_correlations()` (CLI: `--scan`).

No real ConvinoSetups fixture has an actual `(nominal & low : high)` scan
range — every `[correlations]` block in the repo either has none at all, or
only the commented-out example in ATLAS13OnlyTest (grepped: no fixture has an
uncommented '&' in its rho_config.txt). This builds the smallest case that
exercises a real scan: two single-estimate measurements tied together by one
shared, scannable systematic-pair correlation.
"""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
{sys} stat
{name} {resp} {stat}
[end not fitted]

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

[correlations]
{correlations_line}
[end correlations]
"""


def _write_setup(tdp: Path, correlations_line: str) -> Path:
    (tdp / "measA.txt").write_text(
        _MEASUREMENT_TEMPLATE.format(name="measA", value=100.0, sys="sysA", resp=5.0, stat=2.0),
        encoding="utf-8",
    )
    (tdp / "measB.txt").write_text(
        _MEASUREMENT_TEMPLATE.format(name="measB", value=104.0, sys="sysB", resp=5.0, stat=2.0),
        encoding="utf-8",
    )
    config_path = tdp / "config.txt"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(correlations_line=correlations_line), encoding="utf-8"
    )
    return config_path


class CorrelationScanTest(unittest.TestCase):
    def test_scan_sweeps_the_prior_correlation_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = _write_setup(
                Path(td), "sysA = (0.0 & -0.9 : 0.9) sysB"
            )
            scan_results = Combiner.from_config(str(config_path)).scan_correlations()

        self.assertEqual(set(scan_results), {"sysA"})
        values, steps = scan_results["sysA"]
        expected_values = np.linspace(-0.9, 0.9, 6)
        np.testing.assert_allclose(values, expected_values)
        self.assertEqual(len(steps), 6)

        for val, res in zip(values, steps, strict=True):
            ia = res.sys_names.index("sysA")
            ib = res.sys_names.index("sysB")
            # The swept value lands in the prior exactly (it's a direct
            # assignment into the prior covariance, not a fitted quantity).
            self.assertAlmostEqual(float(res.pre_sys_corr[ia, ib]), float(val), places=10)

        # Two measurements sharing a positively-correlated systematic gain
        # less from averaging than when anti-correlated: the combined error
        # must increase monotonically with the scanned correlation.
        errs = [float(res.combined_err_up[0]) for res in steps]
        self.assertTrue(all(e2 > e1 for e1, e2 in itertools.pairwise(errs)))

    def test_scan_skips_groups_without_an_actual_range(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = _write_setup(Path(td), "sysA = (0.3) sysB")
            scan_results = Combiner.from_config(str(config_path)).scan_correlations()
        self.assertEqual(scan_results, {})

    def test_compute_impacts_defaults_to_false_during_scan(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = _write_setup(
                Path(td), "sysA = (0.0 & -0.9 : 0.9) sysB"
            )
            _, steps = Combiner.from_config(
                str(config_path)
            ).scan_correlations()["sysA"]
        for res in steps:
            self.assertEqual(res.impact_groups, {})

    def test_n_steps_below_two_raises(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = _write_setup(
                Path(td), "sysA = (0.0 & -0.9 : 0.9) sysB"
            )
            combiner = Combiner.from_config(str(config_path))
        with self.assertRaises(ValueError):
            combiner.scan_correlations(n_steps=1)


if __name__ == "__main__":
    unittest.main()
