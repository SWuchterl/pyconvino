"""Fast unit tests for the output-path handling (A1).

These run in milliseconds — no combination is performed. They lock in the
fail-fast behaviour: a --prefix that names a missing directory is created, and
an unwritable destination is rejected up front rather than after the fit.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from convino_jax.result import output_path_for, prepare_output_path


class OutputPathTest(unittest.TestCase):
    def test_output_path_suffix(self):
        self.assertEqual(output_path_for("foo"), Path("foo_result.txt"))
        self.assertEqual(
            output_path_for("dir/run1"), Path("dir/run1_result.txt")
        )

    def test_creates_missing_parent_dir(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "sub" / "deeper" / "run1")
            out = prepare_output_path(prefix)
            self.assertEqual(out, Path(prefix + "_result.txt"))
            self.assertTrue(out.parent.is_dir(),
                            "parent directory should have been created")
            # The returned path must be writable.
            out.write_text("ok", encoding="utf-8")
            self.assertEqual(out.read_text(encoding="utf-8"), "ok")

    def test_existing_writable_dir_ok(self):
        with tempfile.TemporaryDirectory() as d:
            out = prepare_output_path(str(Path(d) / "run"))
            self.assertEqual(out.parent, Path(d))

    def test_rejects_unwritable_parent(self):
        # The root filesystem is not writable for a normal user.
        with self.assertRaises(ValueError):
            prepare_output_path("/run1")

    def test_rejects_when_target_is_a_directory(self):
        with tempfile.TemporaryDirectory() as d:
            # Make <prefix>_result.txt already exist as a directory.
            target = Path(d) / "run_result.txt"
            target.mkdir()
            with self.assertRaises(ValueError):
                prepare_output_path(str(Path(d) / "run"))


if __name__ == "__main__":
    unittest.main()
