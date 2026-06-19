"""Round-trip tests for the machine-readable export API (to_dict/export_npz/
export_json).

Exercises the actual end-to-end fit on a small, fast setup (atlas13test: 1
file, 7 observables, 8 impact groups) rather than a hand-built fixture, since
the point is to verify the *real* CombinationResult (impact groups, the new
per-systematic/stat-only breakdown) round-trips through both export formats
bit-for-bit (npz) or exactly (json, modulo float<->list conversion).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from convino_jax import Combiner, to_dict, export_npz, export_json

from .helpers import config_path


class ExportRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = Combiner.from_config(str(config_path("atlas13test"))).combine()
        cls.expected = to_dict(cls.result)

    def test_to_dict_shape(self):
        d = self.expected
        nsys, nest = self.result.nsys, self.result.nest
        self.assertEqual(d["combined_covariance"].shape, (nest, nest))
        self.assertEqual(d["combined_correlation"].shape, (nest, nest))
        self.assertEqual(d["stat_only_covariance"].shape, (nest, nest))
        self.assertEqual(d["total_syst_covariance"].shape, (nest, nest))
        self.assertEqual(len(d["sys_names"]), nsys)
        self.assertEqual(set(d["impact_per_systematic"].keys()), set(d["sys_names"]))
        # "Stat"/all-frozen pseudo-group must not leak into the per-systematic dict.
        self.assertNotIn("__stat_only__", d["impact_per_systematic"])
        self.assertNotIn("__stat_only__", d["cov_per_systematic"])

    def test_goodness_of_fit_fields_present(self):
        d = self.expected
        self.assertIn("ndf", d)
        self.assertIn("chi2_per_ndf", d)
        self.assertIn("p_value", d)
        self.assertEqual(d["ndf"], self.result.ndf)

    def test_total_syst_covariance_is_difference(self):
        d = self.expected
        np.testing.assert_allclose(
            d["total_syst_covariance"],
            d["combined_covariance"] - d["stat_only_covariance"],
        )

    def test_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.npz"
            export_npz(self.result, path)
            loaded = np.load(path, allow_pickle=False)

            np.testing.assert_array_equal(
                loaded["combined_values"], self.expected["combined_values"]
            )
            np.testing.assert_array_equal(
                loaded["combined_covariance"], self.expected["combined_covariance"]
            )
            np.testing.assert_array_equal(
                loaded["stat_only_covariance"], self.expected["stat_only_covariance"]
            )
            # A nested dict entry, flattened with the "__" separator.
            for label, cov in self.expected["impact_cov_groups"].items():
                np.testing.assert_array_equal(
                    loaded[f"impact_cov_groups__{label}"], cov
                )
            for name, ud in self.expected["impact_per_systematic"].items():
                np.testing.assert_array_equal(
                    loaded[f"impact_per_systematic__{name}__up"], ud["up"]
                )
                np.testing.assert_array_equal(
                    loaded[f"impact_per_systematic__{name}__down"], ud["down"]
                )

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.json"
            export_json(self.result, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))

            np.testing.assert_allclose(
                loaded["combined_values"], self.expected["combined_values"]
            )
            np.testing.assert_allclose(
                loaded["stat_only_covariance"], self.expected["stat_only_covariance"]
            )
            self.assertEqual(loaded["chi2_min"], self.expected["chi2_min"])
            self.assertEqual(loaded["converged"], self.expected["converged"])
            self.assertEqual(loaded["sys_names"], self.expected["sys_names"])

            for label, ud in self.expected["impact_groups"].items():
                np.testing.assert_allclose(loaded["impact_groups"][label]["up"], ud["up"])
                np.testing.assert_allclose(loaded["impact_groups"][label]["down"], ud["down"])

    # ------------------------------------------------------------------
    # Tests for impact_matrix (C) and pull_per_group (B)
    # ------------------------------------------------------------------

    def test_impact_matrix_shape(self):
        d = self.expected
        nsys, nest = self.result.nsys, self.result.nest
        self.assertEqual(d["impact_matrix"].shape, (nest, nsys))
        self.assertFalse(np.any(np.isnan(d["impact_matrix"])), "impact_matrix contains NaN")

    def test_impact_matrix_gram_is_psd(self):
        # A @ A.T is a Gram matrix, so its eigenvalues must all be >= 0.
        A = self.expected["impact_matrix"]
        eigvals = np.linalg.eigvalsh(A @ A.T)
        self.assertTrue(np.all(eigvals >= -1e-10 * np.max(np.abs(eigvals))),
                        f"A @ A.T has negative eigenvalue: {eigvals.min():.3e}")

    def test_impact_matrix_columns_bounded(self):
        # Each column is the response of all combined observables to one nuisance.
        # The norm of the shift should not exceed the total combined error.
        A = self.expected["impact_matrix"]
        max_err = float(np.max(self.expected["combined_err_up"]))
        col_norms = np.linalg.norm(A, axis=0)
        # No single nuisance should shift all observables by more than the
        # total error times a generous factor (any larger would be a clear bug).
        self.assertTrue(np.all(col_norms < 10.0 * max_err),
                        f"Unreasonably large impact_matrix column norm: {col_norms.max():.3e}")

    def test_impact_matrix_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.npz"
            export_npz(self.result, path)
            loaded = np.load(path, allow_pickle=False)
            np.testing.assert_array_equal(
                loaded["impact_matrix"], self.expected["impact_matrix"]
            )

    def test_pull_per_group_keys(self):
        # Both dicts should have exactly the same keys as impact_groups.
        d = self.expected
        expected_keys = set(d["impact_groups"].keys())
        self.assertEqual(set(d["pull_per_group_mean"].keys()), expected_keys)
        self.assertEqual(set(d["pull_per_group_norm"].keys()), expected_keys)

    def test_pull_per_group_finite(self):
        for label, v in self.expected["pull_per_group_mean"].items():
            self.assertTrue(np.isfinite(v), f"pull_per_group_mean[{label!r}] is not finite")
        for label, v in self.expected["pull_per_group_norm"].items():
            self.assertTrue(np.isfinite(v), f"pull_per_group_norm[{label!r}] is not finite")

    def test_pull_per_group_consistent_sign(self):
        # mean and norm weightings should give the same sign (or both zero).
        for label in self.expected["pull_per_group_mean"]:
            vm = self.expected["pull_per_group_mean"][label]
            vn = self.expected["pull_per_group_norm"][label]
            if abs(vm) > 1e-12 and abs(vn) > 1e-12:
                self.assertEqual(
                    np.sign(vm), np.sign(vn),
                    f"pull_per_group sign mismatch for group {label!r}: mean={vm:.4f}, norm={vn:.4f}"
                )

    def test_pull_per_group_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.npz"
            export_npz(self.result, path)
            loaded = np.load(path, allow_pickle=False)
            for label, v in self.expected["pull_per_group_mean"].items():
                np.testing.assert_allclose(
                    float(loaded[f"pull_per_group_mean__{label}"]), v
                )
            for label, v in self.expected["pull_per_group_norm"].items():
                np.testing.assert_allclose(
                    float(loaded[f"pull_per_group_norm__{label}"]), v
                )


if __name__ == "__main__":
    unittest.main()
