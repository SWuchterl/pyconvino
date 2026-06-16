"""Fast unit test for the lognormal correctness gap.

`parser.py` accepts "lognormal" as a valid systematic type tag, but no
statistical model for it is implemented anywhere in the fit (the original
C++ Convino never finished it either: `measurement::setParameterType` throws
on it, and `fitfunctionBase::eval` has the lognormal branch explicitly
disabled with a "switched off for now"/"TBI" comment). Before this fix,
`objective.make_chi2` silently treated any systematic type other than
"relative" as "absolute" — so a user who tagged a systematic "lognormal" in
good faith got the wrong statistical model with zero warning.

This test locks in the fix: `make_chi2` must now raise `NotImplementedError`
as soon as it sees an unsupported systematic type, rather than silently
mis-modelling it. No fit/combination is performed — this runs in
milliseconds.
"""

from __future__ import annotations

import unittest

import numpy as np

from convino_jax.measurement import MeasurementSetup
from convino_jax.objective import make_chi2


def _make_setup(sys_types: list[str]) -> MeasurementSetup:
    """A minimal one-estimate, one-systematic MeasurementSetup."""
    nlamb = len(sys_types)
    return MeasurementSetup(
        x_meas=np.array([100.0]),
        LM=np.array([[1.0]]),
        Lk_up=np.zeros((1, nlamb)),
        Lk_down=np.zeros((1, nlamb)),
        LD=np.eye(nlamb),
        est_names=["est0"],
        sys_names=[f"sys{i}" for i in range(nlamb)],
        sys_types=sys_types,
        est_global_idx=[0],
        sys_global_idx=list(range(nlamb)),
    )


class LognormalUnsupportedTest(unittest.TestCase):
    def test_lognormal_raises(self):
        setup = _make_setup(["lognormal"])
        with self.assertRaises(NotImplementedError) as ctx:
            make_chi2([setup], inv_C=np.eye(1), nsys=1, nest=1)
        msg = str(ctx.exception)
        self.assertIn("sys0", msg)
        self.assertIn("lognormal", msg)

    def test_lognormal_mixed_with_supported_types_still_raises(self):
        setup = _make_setup(["absolute", "relative", "lognormal"])
        with self.assertRaises(NotImplementedError):
            make_chi2([setup], inv_C=np.eye(3), nsys=3, nest=1)

    def test_absolute_and_relative_still_supported(self):
        # Sanity check: the fix must not reject the two implemented types.
        setup = _make_setup(["absolute", "relative"])
        chi2_fn = make_chi2([setup], inv_C=np.eye(2), nsys=2, nest=1)
        pars = np.zeros(2 + 1)
        pars[2] = 100.0
        value = float(chi2_fn(pars))
        self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
