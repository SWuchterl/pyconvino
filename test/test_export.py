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


if __name__ == "__main__":
    unittest.main()
