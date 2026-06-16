"""Tests for the `verbose` option (CLI: `--verbose`).

Uses the `atlas13test` fixture (fast) since verbose timing is about *whether*
checkpoints get printed, not their numeric value — no golden coverage needed.
"""

from __future__ import annotations

import contextlib
import io
import unittest

from convino_jax.combiner import Combiner

from .helpers import config_path


class VerboseTest(unittest.TestCase):
    def test_default_is_silent(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            Combiner.from_config(str(config_path("atlas13test"))).combine()
        self.assertEqual(stderr.getvalue(), "")

    def test_verbose_prints_phase_checkpoints(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            Combiner.from_config(
                str(config_path("atlas13test")), verbose=True
            ).combine()
        lines = [l for l in stderr.getvalue().splitlines() if l]
        self.assertTrue(lines, "expected at least one [convino] timing line")
        for line in lines:
            self.assertTrue(line.startswith("[convino] "))

        labels = {line.split(": ")[0] for line in lines}
        expected = {
            "[convino] parsed config + 1 measurement file(s)",
            "[convino] setup + prior",
            "[convino] chi2 build",
            "[convino] minimize",
            "[convino] post-fit covariance + errors",
            "[convino] impacts",
            "[convino] total",
        }
        self.assertEqual(labels, expected)

    def test_minimize_nonlinear_helper_unaffected_by_verbose_attr(self):
        # Combiner.__new__(Combiner) (used by test_minimize_nonlinear.py) never
        # sets self.verbose; _minimize/_compute_impacts/_profile_error must
        # never touch self.verbose or that bare-construction pattern would
        # AttributeError. Exercise that exact construction here as a guard.
        bare = Combiner.__new__(Combiner)
        bare.use_pearson = False
        self.assertFalse(hasattr(bare, "verbose"))
        # _vtime is the only method allowed to read self.verbose, and it is
        # only ever called from combine()/scan_correlations(), not from any
        # private helper such as _minimize.


class ScanVerboseTest(unittest.TestCase):
    def test_scan_prints_one_line_per_step_without_inner_combine_noise(self):
        import tempfile
        from pathlib import Path

        from .test_correlation_scan import _write_setup

        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as td:
            cfg_path = _write_setup(Path(td), "sysA = (0.0 & -0.9 : 0.9) sysB")
            with contextlib.redirect_stderr(stderr):
                Combiner.from_config(str(cfg_path), verbose=True).scan_correlations(
                    n_steps=3
                )

        lines = [l for l in stderr.getvalue().splitlines() if l]
        # One coarse line per scan step, none of the per-phase combine()
        # breakdown (which would require verbose=True on the inner Combiner).
        scan_lines = [l for l in lines if l.startswith("[convino] scan ")]
        self.assertEqual(len(scan_lines), 3)
        for i, line in enumerate(scan_lines, start=1):
            self.assertIn(f"step {i}/3", line)
        self.assertEqual(
            [l for l in lines if "chi2 build" in l or "post-fit" in l], []
        )


if __name__ == "__main__":
    unittest.main()
