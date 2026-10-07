"""Byte-level formatter regression test.

format_result() must byte-for-byte match the C++ Convino layout. This test
feeds a fixed, hand-built CombinationResult (no fit, no float noise) through the
formatter and compares to a committed golden text file, so any change to the
output layout is caught deterministically — independent of the numeric fit.

Regenerate after an intentional formatting change:
    python -m test.regen_golden
"""

from __future__ import annotations

import unittest

from pyconvino import format_result

from .helpers import FORMATTER_GOLDEN, make_fixture_result


class FormatterTest(unittest.TestCase):
    def test_matches_golden(self):
        if not FORMATTER_GOLDEN.exists():
            self.skipTest(
                f"golden {FORMATTER_GOLDEN.name} missing; run `python -m test.regen_golden`"
            )
        expected = FORMATTER_GOLDEN.read_text(encoding="utf-8")
        actual = format_result(make_fixture_result())

        if actual != expected:
            exp_lines = expected.splitlines()
            act_lines = actual.splitlines()
            for i, (e, a) in enumerate(zip(exp_lines, act_lines, strict=True)):
                if e != a:
                    self.fail(
                        f"formatter output differs at line {i + 1}:\n"
                        f"  golden: {e!r}\n  actual: {a!r}"
                    )
            self.fail(
                f"formatter output length differs: golden {len(exp_lines)} lines, "
                f"actual {len(act_lines)} lines"
            )


if __name__ == "__main__":
    unittest.main()
