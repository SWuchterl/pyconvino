"""Fast unit test for the PD-regularization warning (item 3).

`combiner._nearest_positive_definite` reflects negative eigenvalues of the
prior correlation matrix to a small positive epsilon. Before this fix it did
so silently: a user whose `[correlations]` block produced a non-positive-
-definite matrix had it altered with no indication anything was wrong. This
test locks in that a `UserWarning` now fires exactly when regularization
actually changes something (a genuine negative eigenvalue), and not for a
matrix that is already positive-definite (e.g. the identity).

No combination/fit is performed — this calls the utility function directly
and runs in milliseconds.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from convino_jax.combiner import _nearest_positive_definite


class PDRegularizationWarningTest(unittest.TestCase):
    def test_warns_on_non_positive_definite_matrix(self):
        # A symmetric matrix with a clearly negative eigenvalue: correlation
        # of -1 between two systematics and +1 between another pair, which
        # together are inconsistent (not PD) for 3 variables.
        A = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ])
        # Confirm the premise: this matrix is indeed not PD.
        self.assertLess(np.linalg.eigvalsh(A).min(), 0.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _nearest_positive_definite(A)

        matched = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("not positive-definite" in str(w.message) for w in matched),
            f"expected a 'not positive-definite' warning, got: {[str(w.message) for w in caught]}",
        )
        # The regularised matrix should now be PD.
        self.assertGreater(np.linalg.eigvalsh(result).min(), 0.0)

    def test_no_warning_for_already_positive_definite_matrix(self):
        A = np.eye(4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _nearest_positive_definite(A)

        self.assertEqual(
            [str(w.message) for w in caught], [],
            "identity matrix is already PD and must not trigger a warning",
        )
        np.testing.assert_allclose(result, A)

    def test_no_warning_for_well_conditioned_pd_matrix(self):
        # A non-trivial but genuinely PD correlation matrix (no negative
        # eigenvalues, just off-diagonal structure).
        A = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        self.assertGreater(np.linalg.eigvalsh(A).min(), 0.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _nearest_positive_definite(A)

        self.assertEqual([str(w.message) for w in caught], [])


if __name__ == "__main__":
    unittest.main()
