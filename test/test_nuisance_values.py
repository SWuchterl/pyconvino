"""[nuisance values] / "free": the combination must equal the exact joint fit.

Two linear-Gaussian toy analyses share one POI x and one nuisance lam
(unit prior). Each is fitted alone (full likelihood, prior included) and
written as a Convino input with its Hessian, POI estimate and post-fit lam.
Combining them must reproduce the single joint fit of both data sets with the
prior counted once — central values, covariance and chi2 — which the original
Convino form (lam re-centred at 0) does not.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyconvino.combiner import Combiner

P = np.diag([0.0, 1.0])  # prior precision in (x, lam)

# analysis A: precise on x, weak on lam; analysis B: the reverse
ANA = [
    (np.array([[1.0], [1.0], [0.0]]), np.array([[0.5], [0.5], [0.3]]), np.diag([0.05, 0.05, 1.0]) ** 2),
    (np.array([[0.3], [0.0], [0.0]]), np.array([[1.0], [1.0], [1.0]]), np.diag([0.5, 0.1, 0.1]) ** 2),
]


def _fit(A, B, S, y, prior):
    J = np.hstack([A, B]); W = np.linalg.inv(S)
    H = J.T @ W @ J + prior
    return np.linalg.solve(H, J.T @ W @ y), H


def _write(dirpath: Path, k: int, theta, H, with_values: bool, free: bool) -> str:
    lines = ["[hessian]", f"    est_{k} {float(H[0, 0])!r}", f"    lam {float(H[1, 0])!r} {float(H[1, 1])!r}", "[end hessian]",
             "[estimates]", "    n_estimates = 1", f"    name_0 = est_{k}", f"    value_0 = {float(theta[0])!r}",
             "[end estimates]", "[systematics]"]
    if free:
        lines.append("    lam = free")
    lines.append("[end systematics]")
    if with_values:
        lines += ["[nuisance values]", f"    lam = {float(theta[1])!r}", "[end nuisance values]"]
    p = dirpath / f"meas{k}.txt"
    p.write_text("\n".join(lines) + "\n")
    return p.name


def _config(dirpath: Path, files: list[str]) -> Path:
    lines = ["[inputs]", f"    nFiles = {len(files)}"] + [f"    file{i} = {f}" for i, f in enumerate(files)]
    lines += ["[end inputs]", "[observables]", "    comb = " + " + ".join(f"est_{k}" for k in range(len(files))), "[end observables]"]
    p = dirpath / "config.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


class NuisanceValuesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.ys = [A @ [0.0] + B @ [0.0] + rng.multivariate_normal(np.zeros(3), S) for A, B, S in ANA]
        self.ys[1] = self.ys[1] + 0.8 * ANA[1][1][:, 0]  # pull lam hard in analysis B
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _combine(self, with_values, free=(False, False), use_values=None):
        prior = [np.zeros((2, 2)) if f else P for f in free]
        fits = [_fit(A, B, S, y, pr) for (A, B, S), y, pr in zip(ANA, self.ys, prior, strict=True)]
        files = [_write(self.dir, k, th, H, with_values, free[k]) for k, (th, H) in enumerate(fits)]
        res = Combiner.from_config(
            str(_config(self.dir, files)), compute_impacts=False,
            use_nuisance_values=with_values if use_values is None else use_values,
        ).combine()
        return fits, res

    def test_block_ignored_unless_enabled(self):
        # The [nuisance values] block is opt-in: without the flag the result
        # must be the original (C++ Convino) one, block present or not.
        _, on = self._combine(with_values=True, use_values=False)
        _, off = self._combine(with_values=False)
        np.testing.assert_allclose(on.combined_values, off.combined_values, rtol=0, atol=0)
        np.testing.assert_allclose(on.pulls, off.pulls, rtol=0, atol=0)

    def test_matches_exact_joint_fit(self):
        fits, res = self._combine(with_values=True)
        # exact joint fit: both data sets, prior once
        H_tot = P.copy(); rhs = np.zeros(2)
        for (A, B, S), y in zip(ANA, self.ys, strict=True):
            J = np.hstack([A, B]); W = np.linalg.inv(S)
            H_tot += J.T @ W @ J; rhs += J.T @ W @ y
        theta = np.linalg.solve(H_tot, rhs)
        cov = np.linalg.inv(H_tot)
        chi2 = sum(float((y - np.hstack([A, B]) @ theta) @ np.linalg.inv(S) @ (y - np.hstack([A, B]) @ theta))
                   for (A, B, S), y in zip(ANA, self.ys, strict=True)) + theta[1] ** 2
        # the files carry no data residuals: each data term is relative to the
        # input's unconstrained (prior-free) optimum
        for (A, B, S), y in zip(ANA, self.ys, strict=True):
            th_d, _ = _fit(A, B, S, y, np.zeros((2, 2)))
            chi2 -= float((y - np.hstack([A, B]) @ th_d) @ np.linalg.inv(S) @ (y - np.hstack([A, B]) @ th_d))

        self.assertAlmostEqual(float(res.combined_values[0]), theta[0], places=10)
        self.assertAlmostEqual(float(res.pulls[0]), theta[1], places=10)
        self.assertAlmostEqual(float(res.cov_full[1, 1]), cov[0, 0], places=10)  # x is the last parameter
        self.assertAlmostEqual(float(res.cov_full[0, 0]), cov[1, 1], places=10)
        self.assertAlmostEqual(float(res.chi2_min), chi2, places=8)
        self.assertAlmostEqual(res.chi2_tension + sum(res.chi2_standalone.values()), res.chi2_min, places=10)

        # original Convino form: same covariance, different central value
        _, res0 = self._combine(with_values=False)
        self.assertAlmostEqual(float(res0.cov_full[1, 1]), cov[0, 0], places=10)
        self.assertNotAlmostEqual(float(res0.combined_values[0]), theta[0], places=3)
        self.assertEqual(res0.chi2_tension, res0.chi2_min)

    def test_standalone_input_is_reproduced(self):
        (th, H), = [_fit(*ANA[1], self.ys[1], P)]
        f = _write(self.dir, 0, th, H, True, False)
        res = Combiner.from_config(
            str(_config(self.dir, [f])), compute_impacts=False, use_nuisance_values=True
        ).combine()
        self.assertAlmostEqual(float(res.combined_values[0]), th[0], places=10)
        self.assertAlmostEqual(float(res.pulls[0]), th[1], places=10)
        self.assertAlmostEqual(float(res.constraints[0]), np.sqrt(np.linalg.inv(H)[1, 1]), places=10)
        # chi2_min = prior penalty of the input fit = lam^2 (1 + 1/D), D = data precision on lam
        D = H[1, 1] - 1.0 - H[0, 1] ** 2 / H[0, 0]
        self.assertAlmostEqual(float(res.chi2_min), th[1] ** 2 * (1 + 1 / D), places=8)
        self.assertAlmostEqual(float(res.chi2_tension), 0.0, places=8)

    def test_free_parameter_has_no_prior(self):
        # lam free in B: the joint fit must use P = 0 for it everywhere (A is
        # written with lam free too, so no prior exists at all)
        fits, res = self._combine(with_values=True, free=(True, True))
        H_tot = np.zeros((2, 2)); rhs = np.zeros(2)
        for (A, B, S), y in zip(ANA, self.ys, strict=True):
            J = np.hstack([A, B]); W = np.linalg.inv(S)
            H_tot += J.T @ W @ J; rhs += J.T @ W @ y
        theta = np.linalg.solve(H_tot, rhs)
        self.assertAlmostEqual(float(res.combined_values[0]), theta[0], places=10)
        self.assertAlmostEqual(float(res.pulls[0]), theta[1], places=10)
        self.assertAlmostEqual(float(res.cov_full[1, 1]), np.linalg.inv(H_tot)[0, 0], places=10)
        self.assertEqual(float(res.prior_inv_cov[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
