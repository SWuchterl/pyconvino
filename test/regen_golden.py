"""Regenerate the golden snapshots for the regression tests.

Run this only when the numeric output is *intentionally* changed and the new
values have been verified:

    python -m test.regen_golden            # regenerate all setups
    python -m test.regen_golden statonly   # regenerate one setup

Goldens are committed so the regression test can run without recomputation.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .helpers import (
    FORMATTER_GOLDEN,
    GOLDEN_DIR,
    SETUPS,
    compute_arrays,
    golden_path,
    make_fixture_result,
)


def _regen_formatter() -> None:
    from pyconvino import format_result

    FORMATTER_GOLDEN.write_text(format_result(make_fixture_result()), encoding="utf-8")
    print(f"wrote {FORMATTER_GOLDEN.name}")


def main(argv: list[str]) -> int:
    names = argv or list(SETUPS)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # The formatter golden is cheap and has no external inputs; regenerate it
    # whenever a full (no-args) regeneration is requested.
    if not argv:
        _regen_formatter()

    for name in names:
        if name not in SETUPS:
            print(f"unknown setup '{name}'; known: {list(SETUPS)}", file=sys.stderr)
            return 2
        t0 = time.time()
        arrays = compute_arrays(name)
        # Compressed: the correlation/covariance matrices are large but highly
        # compressible, keeping the committed goldens small.
        np.savez_compressed(golden_path(name), **arrays)
        print(f"wrote {golden_path(name).name} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
