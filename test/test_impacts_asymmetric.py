"""Regression test for the asymmetric-response impacts path against real C++ Convino.

`Combiner._compute_impacts` has two branches (see its docstring): symmetric
responses slice the post-fit Hessian (fast, exact); asymmetric responses
instead refit + profile-scan per impact group, per combined observable
(`_minimize_frozen`/`_profile_error`). None of the golden setups in
`test/helpers.py` are asymmetric ("all-absolute", per that module's own
comment), so this branch had no coverage at all before this test.

`ConvinoSetups/PaperExample/` is the worked example from the Convino paper
(Kieseler, Sec. 5, Figs. 14/15; `old/Convino/examples/exampleconfig.txt` and
its measurement files) — copied in so this test is self-contained and doesn't
depend on the sibling C++ checkout existing. It has one genuinely asymmetric
uncertainty (`exampleMeasurement3.txt`'s `sys_d3 (+5-3)`), which is enough to
disable the fast path for the whole combination.

Reference numbers below are the "[simple impact table]" (correlation-based
impact: corr_full[i, nsys+j] * symmetrized_error[j], see
`result._print_simple_impacts`) from the actual compiled C++ `convino` binary
run on the identical input files (Pearson chi2, no `--neyman`) -- see
`docs/paper_benchmark_crosscheck.md` for the full crosscheck writeup. This is
*not* the same quantity as `result.impact_per_systematic` (the quadrature
frozen-fit impact used for `impact_groups`) -- the printed "simple impact
table" is a linear correlation-times-error estimate, checked here via
`result.format_result()` to match exactly what a user actually sees in
`<prefix>_result.txt`.

Central values/errors and the 7 systematics from `exampleMeasurement1/2.txt`
match C++ to <0.1% and are checked tightly here. The 5 systematics from
`exampleMeasurement3.txt` have true impact at or below both Python's and
C++'s own optimizer noise floor (verified there by freezing them and seeing
chi2 barely move) -- their C++ and Python values genuinely disagree by up to
~50% *relative*, while differing by at most a few thousandths of a percentage
point *absolute*, so they are only checked for being small and finite, not
matched to C++.

This is slow: the asymmetric branch has no fast path, so every one of the 12
systematics (users groups + per-systematic breakdown) does a full refit plus
a profile-likelihood scan. Gated behind CONVINO_SLOW_TESTS=1 like corrV2 in
test_regression.py.
"""

from __future__ import annotations

import os
import re
import unittest

from pyconvino.combiner import Combiner
from pyconvino.result import format_result

_RUN_SLOW = bool(os.environ.get("CONVINO_SLOW_TESTS"))

# C++ reference: [combined (minimum chi^2=...)] section.
_CHI2_MIN_CPP = 4.11332
_COMBINED_CPP = {
    # name: (value, err_up, err_down)
    "combined_a": (798.69, 7.62619, 7.58072),
    "combined_b": (306.101, 5.50503, 5.39961),
}

# C++ reference: [simple impact table], columns (combined_a, combined_b), percent.
# Systematics from exampleMeasurement1.txt/exampleMeasurement2.txt: agree with
# Python to <0.1% (checked tightly, generous 1% margin below).
_TIGHT_IMPACTS_CPP = {
    "sys_a1": (0.42234, 1.07526),
    "sys_b1": (0.43818, 0.06538),
    "sys_c1": (0.26322, 1.43027),
    "sys_a2": (0.11469, 0.10000),
    "sys_b2": (0.09430, 0.21306),
    "sys_c2": (0.46761, 0.47101),
    "sys_d2": (0.20781, 1.19106),
}
# Systematics from exampleMeasurement3.txt: true impact ~0 (noise-floor regime,
# see module docstring) -- only checked for "small", not matched to C++.
_NOISE_FLOOR_SYS = ["sys_a3", "sys_b3", "sys_c3", "sys_d3", "sys_e3"]
_NOISE_FLOOR_CEILING_PCT = (
    0.1  # generous vs. the largest C++/Python value seen (~0.045%)
)


def _parse_simple_impact_table(text: str) -> dict[str, tuple[float, float]]:
    m = re.search(
        r"\[simple impact table: name, impact \[%\]\](.*?)\[end simple impact table",
        text,
        re.S,
    )
    table: dict[str, tuple[float, float]] = {}
    for line in m.group(1).strip().splitlines()[1:]:  # skip header row
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) == 3:
            table[parts[0]] = (float(parts[1]), float(parts[2]))
    return table


@unittest.skipUnless(
    _RUN_SLOW, "asymmetric impacts path is slow; set CONVINO_SLOW_TESTS=1"
)
class AsymmetricImpactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = Combiner.from_config(
            "ConvinoSetups/PaperExample/exampleconfig.txt", use_pearson=True
        ).combine()
        cls.simple_impacts = _parse_simple_impact_table(format_result(cls.result))

    def test_central_fit_matches_cpp(self):
        r = self.result
        self.assertAlmostEqual(r.chi2_min, _CHI2_MIN_CPP, delta=1e-3)
        for name, (val_cpp, up_cpp, down_cpp) in _COMBINED_CPP.items():
            i = r.combined_names.index(name)
            self.assertAlmostEqual(r.combined_values[i], val_cpp, delta=val_cpp * 1e-3)
            self.assertAlmostEqual(r.combined_err_up[i], up_cpp, delta=up_cpp * 1e-2)
            self.assertAlmostEqual(
                r.combined_err_down[i], down_cpp, delta=down_cpp * 1e-2
            )

    def test_dominant_systematic_impacts_match_cpp(self):
        # 1% relative margin: observed C++/Python agreement here is <0.1%.
        for sys_name, (a_cpp, b_cpp) in _TIGHT_IMPACTS_CPP.items():
            with self.subTest(sys=sys_name):
                a_py, b_py = self.simple_impacts[sys_name]
                self.assertAlmostEqual(a_py, a_cpp, delta=a_cpp * 0.01)
                self.assertAlmostEqual(b_py, b_cpp, delta=b_cpp * 0.01)

    def test_noise_floor_systematics_are_small(self):
        # Coverage goal here is not numeric agreement with C++ (see module
        # docstring) but catching regressions like NaN, inf, or a blown-up
        # impact -- i.e. that this branch still produces a sane answer at all.
        for sys_name in _NOISE_FLOOR_SYS:
            with self.subTest(sys=sys_name):
                a_py, b_py = self.simple_impacts[sys_name]
                self.assertTrue(a_py == a_py and b_py == b_py)  # not NaN
                self.assertLess(a_py, _NOISE_FLOOR_CEILING_PCT)
                self.assertLess(b_py, _NOISE_FLOOR_CEILING_PCT)


if __name__ == "__main__":
    unittest.main()
