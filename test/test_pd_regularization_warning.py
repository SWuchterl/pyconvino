"""Fast unit tests for the PD-regularization warning and all three methods.

`combiner._nearest_positive_definite` can regularise a non-positive-definite
prior correlation matrix via three strategies: 'shift' (default), 'clip', and
'higham'.  All three must:
  - emit a UserWarning containing "not positive-definite" when needed,
  - produce a PD result (min eigenvalue > 0),
  - preserve the unit diagonal (correlation matrix structure),
  - remain silent for a matrix that is already PD.

No combination/fit is performed — this calls the utility function directly
and runs in milliseconds.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from pyconvino.combiner import _nearest_positive_definite

NON_PD = np.array(
    [
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.9],
        [0.9, -0.9, 1.0],
    ]
)


class PDRegularizationWarningTest(unittest.TestCase):
    def _run(self, A, method="shift"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _nearest_positive_definite(A, method=method)
        return result, caught

    def _assert_warns(self, caught):
        matched = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("not positive-definite" in str(w.message) for w in matched),
            f"expected a 'not positive-definite' warning, got: {[str(w.message) for w in caught]}",
        )

    def _assert_valid_corr(self, result):
        """Result must be PD and have unit diagonal."""
        self.assertGreater(np.linalg.eigvalsh(result).min(), 0.0)
        np.testing.assert_allclose(
            np.diag(result), np.ones(result.shape[0]), atol=1e-12
        )

    # -- per-method warning + validity ----------------------------------------

    def test_shift_warns_and_is_valid(self):
        self.assertLess(np.linalg.eigvalsh(NON_PD).min(), 0.0)
        result, caught = self._run(NON_PD, method="shift")
        self._assert_warns(caught)
        self._assert_valid_corr(result)

    def test_clip_warns_and_is_valid(self):
        result, caught = self._run(NON_PD, method="clip")
        self._assert_warns(caught)
        self._assert_valid_corr(result)

    def test_higham_warns_and_is_valid(self):
        result, caught = self._run(NON_PD, method="higham")
        self._assert_warns(caught)
        self._assert_valid_corr(result)

    # -- shift-specific: uniform scaling --------------------------------------

    def test_shift_uniform_offdiag_scaling(self):
        """All off-diagonal entries must be scaled by a common factor."""
        result, _ = self._run(NON_PD, method="shift")
        mask = ~np.eye(3, dtype=bool)
        # Where the original is non-zero, result / original should be constant.
        nonzero = NON_PD[mask] != 0
        ratios = result[mask][nonzero] / NON_PD[mask][nonzero]
        np.testing.assert_allclose(
            ratios,
            ratios[0],
            rtol=1e-10,
            err_msg="shift must scale all off-diag entries uniformly",
        )

    # -- Higham: Frobenius-optimal (≤ shift) ----------------------------------

    def test_higham_and_shift_similar_frobenius(self):
        """For a symmetric inconsistency all methods give comparable Frobenius change."""
        r_higham, _ = self._run(NON_PD, method="higham")
        r_shift, _ = self._run(NON_PD, method="shift")
        r_clip, _ = self._run(NON_PD, method="clip")
        mask = ~np.eye(3, dtype=bool)
        frobs = {
            "higham": np.sqrt(np.sum((r_higham[mask] - NON_PD[mask]) ** 2)),
            "shift": np.sqrt(np.sum((r_shift[mask] - NON_PD[mask]) ** 2)),
            "clip": np.sqrt(np.sum((r_clip[mask] - NON_PD[mask]) ** 2)),
        }
        # All three should be within 10 % of each other on this matrix
        # (for the symmetric NON_PD matrix shift≈higham; clip may differ a bit).
        values = list(frobs.values())
        spread = max(values) - min(values)
        self.assertLess(
            spread / max(values), 0.10, f"Frobenius distances spread >10 %: {frobs}"
        )

    # -- already-PD: no warning, identity returned ----------------------------

    def test_no_warning_for_already_positive_definite_matrix(self):
        A = np.eye(4)
        result, caught = self._run(A)
        self.assertEqual(
            [str(w.message) for w in caught],
            [],
            "identity matrix is already PD and must not trigger a warning",
        )
        np.testing.assert_allclose(result, A)

    def test_no_warning_for_well_conditioned_pd_matrix(self):
        A = np.array(
            [
                [1.0, 0.3, 0.1],
                [0.3, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        self.assertGreater(np.linalg.eigvalsh(A).min(), 0.0)
        _, caught = self._run(A)
        self.assertEqual([str(w.message) for w in caught], [])

    # -- invalid method -------------------------------------------------------

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError, msg="unknown method should raise"):
            _nearest_positive_definite(NON_PD, method="bogus")


if __name__ == "__main__":
    unittest.main()
