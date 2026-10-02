"""stat_only_covariance is the true frozen covariance (Schur complement of
cov_full), and prior_inv_cov_eff removes the negative input LD directions so the
global-impacts data part is PSD. CMSOnly has such directions (CMS dilepton input
with post-fit variances above the prior); ATLAS13OnlyTest has none."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from pyconvino import Combiner

from .helpers import SETUPS_DIR


def _run(setup):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = Combiner.from_config(str(SETUPS_DIR / setup / "rho_config.txt")).combine()
    return r, [str(x.message) for x in w]


class FrozenCovAndPriorEffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cms, cls.cms_warn = _run("CMSOnly")
        cls.atl, cls.atl_warn = _run("ATLAS13OnlyTest")

    def _schur(self, r):
        F, ns = np.asarray(r.cov_full), r.nsys
        return F[ns:, ns:] - F[ns:, :ns] @ np.linalg.solve(F[:ns, :ns], F[:ns, ns:])

    def test_stat_only_is_schur_complement(self):
        for r in (self.cms, self.atl):
            S, T = np.asarray(r.stat_only_covariance), self._schur(r)
            np.testing.assert_allclose(S, T, rtol=1e-6, atol=1e-9 * np.abs(T).max())

    def test_prior_eff_makes_data_part_psd(self):
        r, ns = self.cms, self.cms.nsys
        Ve, V = np.asarray(r.prior_inv_cov_eff), np.asarray(r.prior_inv_cov)
        self.assertGreater(np.abs(Ve - V).max(), 1e-3)
        self.assertGreater(np.linalg.eigvalsh(Ve).min(), 0.0)
        H = 2.0 * np.linalg.inv(np.asarray(r.cov_full))
        Hd = H.copy(); Hd[:ns, :ns] -= 2.0 * Ve
        e = np.linalg.eigvalsh(0.5 * (Hd + Hd.T))
        self.assertGreater(e.min(), -1e-8 * e.max())
        # global stat >= frozen stat for every observable
        G, C = np.asarray(r.cov_full)[ns:, :ns], np.asarray(r.cov_full)[ns:, ns:]
        stat_global = np.diag(C - G @ Ve @ G.T)
        self.assertTrue(np.all(stat_global >= np.diag(self._schur(r)) * (1 - 1e-8)))
        self.assertTrue(any("prior_inv_cov_eff" in m for m in self.cms_warn))

    def test_prior_eff_equals_prior_without_negative_ld(self):
        np.testing.assert_array_equal(self.atl.prior_inv_cov_eff, self.atl.prior_inv_cov)


if __name__ == "__main__":
    unittest.main()
