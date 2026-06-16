"""Shared helpers for the regression test suite.

The regression tests run a small, representative set of combinations end-to-end
and compare the numeric result against a stored golden snapshot. They exist to
give the performance work (linear-solve / Newton fits, reusing the Hessian for
impacts, ...) a safety net: a refactor that changes the *answer* — rather than
just the last machine-precision digits — fails loudly.

The golden is a numeric snapshot of the CombinationResult arrays (.npz), not a
byte-for-byte diff of the formatted text. Optimiser-level noise of order the
convergence tolerance is expected and tolerated; a real change in a combined
value, error, pull or impact is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Repository root (the convino_jax package directory == this file's parent's parent)
REPO_ROOT = Path(__file__).resolve().parent.parent
SETUPS_DIR = REPO_ROOT / "ConvinoSetups"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Representative setups chosen for coverage at low runtime:
#   - statonly : 3 input files, 19 observables, NO impact groups
#                (fast; exercises multi-file combine + #!FILE includes + the
#                 correlation prior, without the expensive impact loop)
#   - atlas13test : 1 file, 7 observables, 8 impact groups
#                (exercises the [hessian]/[not fitted] parsing and the impact /
#                 frozen-refit code path on a small problem)
#   - corrV2 : the full 3-file ATLAS8+13 / CMS13 correlated combination
#                (19 observables, 48 impact groups). The heavyweight, realistic
#                case; its combined central values were validated once against
#                the original C++ Convino output (exact match on the values,
#                errors agreeing to ~0.05%). Slow (~90 s), so it is only run when
#                CONVINO_SLOW_TESTS=1 is set; its golden is always committed.
#   - atlas8 : single ATLAS-8TeV measurement, 8 observables, 14 impact groups.
#                A medium single-file case that exercises the impact / frozen
#                code path at moderate cost (so the impact code is covered
#                without needing the 80 s corrV2).
SETUPS: dict[str, str] = {
    "statonly": "Combination_ATLAS813CMS13_corrV2_statonly/rho_config.txt",
    "atlas13test": "ATLAS13OnlyTest/rho_config.txt",
    "atlas8": "ATLAS8Only/rho_config.txt",
    "corrV2": "Combination_ATLAS813CMS13_corrV2/rho_config.txt",
}

# Setups too slow to recompute on every test run. Their golden is still
# committed and the regression test only recomputes them when CONVINO_SLOW_TESTS
# is set in the environment. (atlas8 became fast enough to run by default after
# the impact-computation speedup, so it covers the impact path in the fast tier;
# corrV2 stays gated as the heavyweight full-combination validation.)
SLOW_SETUPS: frozenset[str] = frozenset({"corrV2"})


def config_path(name: str) -> Path:
    return SETUPS_DIR / SETUPS[name]


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.npz"


def result_to_arrays(result) -> dict[str, np.ndarray]:
    """Flatten the physics-relevant parts of a CombinationResult into named arrays.

    Impact groups are flattened into deterministically-ordered stacks so the
    snapshot is a plain dict of arrays that np.savez can store and np.allclose
    can compare.
    """
    out: dict[str, np.ndarray] = {
        "chi2_min": np.asarray(result.chi2_min, dtype=float),
        "ndf": np.asarray(result.ndf, dtype=int),
        "chi2_per_ndf": np.asarray(result.chi2_per_ndf, dtype=float),
        "p_value": np.asarray(result.p_value, dtype=float),
        "converged": np.asarray(result.converged, dtype=bool),
        "combined_values": np.asarray(result.combined_values, dtype=float),
        "combined_err_up": np.asarray(result.combined_err_up, dtype=float),
        "combined_err_down": np.asarray(result.combined_err_down, dtype=float),
        "pulls": np.asarray(result.pulls, dtype=float),
        "constraints": np.asarray(result.constraints, dtype=float),
        "corr_full": np.asarray(result.corr_full, dtype=float),
        "cov_full": np.asarray(result.cov_full, dtype=float),
    }

    # Impact groups (sorted by label for a stable ordering).
    labels = sorted(result.impact_groups.keys())
    if labels:
        up = np.stack([result.impact_groups[l][0] for l in labels])
        down = np.stack([result.impact_groups[l][1] for l in labels])
        out["impact_labels"] = np.asarray(labels)
        out["impact_up"] = up.astype(float)
        out["impact_down"] = down.astype(float)

    # Stat/syst split + per-individual-systematic breakdown (the export-API
    # addition, see docs/improvement_plan.md Q4/Q4b). Guard the new fields
    # with the same golden-regression net as everything else.
    out["stat_only_covariance"] = np.asarray(result.stat_only_covariance, dtype=float)
    out["total_syst_impact_up"] = np.asarray(result.total_syst_impact_up, dtype=float)
    out["total_syst_impact_down"] = np.asarray(result.total_syst_impact_down, dtype=float)

    sys_labels = sorted(result.impact_per_systematic.keys())
    if sys_labels:
        sys_up = np.stack([result.impact_per_systematic[l][0] for l in sys_labels])
        sys_down = np.stack([result.impact_per_systematic[l][1] for l in sys_labels])
        out["impact_per_sys_labels"] = np.asarray(sys_labels)
        out["impact_per_sys_up"] = sys_up.astype(float)
        out["impact_per_sys_down"] = sys_down.astype(float)
    return out


# For the large slow setups the full (nsys+nest)² correlation/covariance
# matrices dominate the golden file size (~6 MB) and barely compress. They are
# derived quantities, so we drop them from those goldens and keep the physics
# vectors (values, errors, pulls, constraints, impacts), which are what matter.
_LARGE_MATRIX_KEYS = ("corr_full", "cov_full")


def make_fixture_result():
    """A small, hand-built CombinationResult covering every output section.

    Fixed numbers (no fit) so format_result() is fully deterministic — this is
    the input for the byte-level formatter golden, which guards result.py
    formatting independently of any optimiser noise.
    """
    from scipy.stats import chi2 as chi2_dist

    from convino_jax import CombinationResult

    sys_names = ["sysA", "sysB", "sysC"]
    comb_names = ["obs1", "obs2"]
    nsys, nest = 3, 2
    all_names = sys_names + comb_names

    pre_sys_corr = np.array([
        [1.0, 0.3, 0.0],
        [0.3, 1.0, -0.2],
        [0.0, -0.2, 1.0],
    ])

    # A plausible 5x5 post-fit correlation (sys block, est block, cross terms).
    corr_full = np.array([
        [1.00, 0.25, 0.05, 0.40, -0.10],
        [0.25, 1.00, -0.15, 0.20, 0.30],
        [0.05, -0.15, 1.00, -0.05, 0.12],
        [0.40, 0.20, -0.05, 1.00, 0.50],
        [-0.10, 0.30, 0.12, 0.50, 1.00],
    ])
    std = np.array([0.8, 1.1, 0.6, 12.0, 7.5])
    cov_full = corr_full * np.outer(std, std)

    combined_values = np.array([107.5, 611.9])
    combined_err_up = np.array([12.24, 30.33])
    combined_err_down = np.array([12.10, 30.50])

    impact_up = np.array([5.1, 8.2])
    impact_down = np.array([5.0, 8.4])
    impact_cov = np.array([[5.1**2, 0.3 * 5.1 * 8.2],
                           [0.3 * 5.1 * 8.2, 8.2**2]])

    fixture_ndf = 7
    return CombinationResult(
        chi2_min=1.2345,
        ndf=fixture_ndf,
        chi2_per_ndf=1.2345 / fixture_ndf,
        p_value=float(chi2_dist.sf(1.2345, fixture_ndf)),
        converged=True,
        combined_names=comb_names,
        combined_values=combined_values,
        combined_err_up=combined_err_up,
        combined_err_down=combined_err_down,
        sys_names=sys_names,
        pulls=np.array([0.12, -0.34, 0.56]),
        constraints=np.array([0.95, 0.80, 1.00]),
        corr_full=corr_full,
        cov_full=cov_full,
        all_names=all_names,
        pre_sys_corr=pre_sys_corr,
        impact_groups={"groupX": (impact_up, impact_down)},
        impact_cov_groups={"groupX": impact_cov},
        pars_best=np.zeros(nsys + nest),
        nsys=nsys,
        nest=nest,
    )


# Byte-level formatter golden: format_result(make_fixture_result()).
FORMATTER_GOLDEN = GOLDEN_DIR / "formatter.txt"


def compute_arrays(name: str) -> dict[str, np.ndarray]:
    """Run the combination for `name` and return its array snapshot.

    For SLOW_SETUPS the large derived matrices are dropped to keep the
    committed golden small.
    """
    from convino_jax import Combiner

    result = Combiner.from_config(str(config_path(name))).combine()
    arrays = result_to_arrays(result)
    if name in SLOW_SETUPS:
        for k in _LARGE_MATRIX_KEYS:
            arrays.pop(k, None)
    return arrays
