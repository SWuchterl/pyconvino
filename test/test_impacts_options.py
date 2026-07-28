"""Tests for the `compute_impacts`/`impacts_only` options (CLI: `--no-impacts`
/`--impacts-only`).

Uses the `atlas8` fixture (14 user-defined impact groups, fast) rather than a
golden setup: these options don't change the underlying fit, only which of
the already-validated impact machinery runs, so a self-consistency check
against the full (`compute_impacts=True`) result is the right kind of test.
"""

from __future__ import annotations

import unittest

import numpy as np

from pyconvino.combiner import Combiner

from .helpers import config_path


class NoImpactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = Combiner.from_config(str(config_path("atlas8"))).combine()
        cls.skipped = Combiner.from_config(
            str(config_path("atlas8")), compute_impacts=False
        ).combine()

    def test_fit_itself_is_unaffected(self):
        # Skipping impacts must not change the combined values/errors/chi2 —
        # they come from steps 1-11, before the impact machinery runs.
        np.testing.assert_array_equal(
            self.skipped.combined_values, self.full.combined_values
        )
        np.testing.assert_array_equal(
            self.skipped.combined_err_up, self.full.combined_err_up
        )
        self.assertEqual(self.skipped.chi2_min, self.full.chi2_min)

    def test_impact_groups_empty(self):
        self.assertEqual(self.skipped.impact_groups, {})
        self.assertEqual(self.skipped.impact_cov_groups, {})
        self.assertEqual(self.skipped.impact_per_systematic, {})
        self.assertEqual(self.skipped.cov_per_systematic, {})

    def test_stat_only_and_total_syst_are_nan_placeholders(self):
        nest = self.skipped.nest
        self.assertEqual(self.skipped.stat_only_covariance.shape, (nest, nest))
        self.assertTrue(np.all(np.isnan(self.skipped.stat_only_covariance)))
        self.assertTrue(np.all(np.isnan(self.skipped.total_syst_impact_up)))
        self.assertTrue(np.all(np.isnan(self.skipped.total_syst_impact_down)))

    def test_to_dict_does_not_crash_on_nan_placeholders(self):
        from pyconvino.result import to_dict

        d = to_dict(self.skipped)
        self.assertEqual(d["total_syst_covariance"].shape, (self.skipped.nest,) * 2)
        self.assertTrue(np.all(np.isnan(d["total_syst_covariance"])))

    def test_format_result_skips_impact_sections(self):
        from pyconvino.result import format_result

        text = format_result(self.skipped)
        self.assertNotIn("[impact table]", text)
        self.assertNotIn("[merged impacts]", text)
        self.assertNotIn("[covariance matrix for merged impacts]", text)
        # Unaffected sections are still present.
        self.assertIn("[goodness of fit]", text)
        self.assertIn("[full correlation matrix]", text)


class ImpactsOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = Combiner.from_config(str(config_path("atlas8"))).combine()
        cls.labels = ["ATLAS_8TeV_pdf", "ATLAS_8TeV_scale_j"]
        cls.subset = Combiner.from_config(
            str(config_path("atlas8")), impacts_only=cls.labels
        ).combine()

    def test_only_requested_groups_present(self):
        self.assertEqual(set(self.subset.impact_groups), set(self.labels))
        self.assertEqual(set(self.subset.impact_cov_groups), set(self.labels))

    def test_subset_values_match_full_computation(self):
        for label in self.labels:
            np.testing.assert_array_equal(
                self.subset.impact_groups[label][0], self.full.impact_groups[label][0]
            )
            np.testing.assert_array_equal(
                self.subset.impact_groups[label][1], self.full.impact_groups[label][1]
            )
            np.testing.assert_array_equal(
                self.subset.impact_cov_groups[label], self.full.impact_cov_groups[label]
            )

    def test_per_systematic_breakdown_is_unaffected(self):
        # --impacts-only narrows step 12 (user-defined groups) only; step 13
        # (per-systematic/stat-only breakdown) is a separate mechanism.
        self.assertEqual(
            set(self.subset.impact_per_systematic), set(self.full.impact_per_systematic)
        )
        np.testing.assert_array_equal(
            self.subset.stat_only_covariance, self.full.stat_only_covariance
        )

    def test_unknown_label_raises_immediately(self):
        with self.assertRaises(ValueError):
            Combiner.from_config(
                str(config_path("atlas8")), impacts_only=["not_a_real_group"]
            )


if __name__ == "__main__":
    unittest.main()
