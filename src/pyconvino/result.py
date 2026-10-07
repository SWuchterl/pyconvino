"""
Output formatter: produce the Convino result .txt file.
Matches combinationResult::printFullInfo() from the C++ source exactly.
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .combiner import CombinationResult

# ---------------------------------------------------------------------------
# C++ textFormatter::fixLength port
# ---------------------------------------------------------------------------


def _round_cpp(v: float, pres: float) -> float:
    """Port of C++ round(f, pres) = floor(f*(1/pres)+0.5)/(1/pres)."""
    inv_pres = 1.0 / pres
    return math.floor(v * inv_pres + 0.5) / inv_pres


def _fix_length(v: float, l: int) -> str:
    """
    Port of textFormatter::fixLength(double, size_t, truncate=true).

    For v > 0 : returns ' ' + first-l-chars  → total l+1 chars
    For v < 0 : l_eff = l+1, returns first l+1 chars → total l+1 chars
    For v == 0: returns first-l-chars (no leading space) → total l chars
    """
    l_eff = l + 1 if v < 0.0 else l

    s = f"{v:.{l_eff + 10}f}"  # fixed format, very high precision

    # Truncate to l_eff (take first l_eff chars) — truncate=true default
    if len(s) > l_eff:
        s = s[:l_eff]
    elif len(s) < l_eff:
        s = s + " " * (l_eff - len(s))

    # Handle trailing '.' edge case (replace with space, shorten by 1)
    if s.endswith("."):
        s = s[:-1] + " "

    # Prepend space for strictly positive values
    if v > 0.0:
        s = " " + s

    return s


# ---------------------------------------------------------------------------
# Matrix scaler and printing
# ---------------------------------------------------------------------------


def _compute_scaler(mat: np.ndarray) -> float:
    """
    Compute display scaler as in C++ triangularMatrix::printToStream.
    Brings (max - min) of matrix entries into [1, 10].
    """
    flat = mat.flatten()
    if flat.size == 0:
        return 1.0
    mn, mx = float(np.min(flat)), float(np.max(flat))
    delta = mx - mn
    if delta == 0.0:
        return 1.0
    scaler = 1.0
    while delta * scaler < 1.0:
        scaler *= 10.0
    while delta * scaler > 10.0:
        scaler *= 0.1
    return scaler


def _print_matrix(buf: io.StringIO, names: list[str], mat: np.ndarray) -> None:
    """
    Print a matrix block matching C++ triangularMatrix::printToStream.
    Writes all rows without a trailing newline on the last row
    (caller adds \\n via the section wrapper).
    """
    if not names:
        return

    scaler = _compute_scaler(mat)
    if scaler <= 0.01 or scaler >= 100.0:
        buf.write(f"multiplied by: {scaler:g}\n")
    else:
        scaler = 1.0

    maxnw = max(len(n) for n in names)

    for i, name in enumerate(names):
        buf.write(name.ljust(maxnw + 1))
        for j, v_raw in enumerate(mat[i]):
            v = _round_cpp(float(v_raw) * scaler, 1e-6)
            buf.write(_fix_length(v, 9))
            if j < len(mat[i]) - 1:
                buf.write(" ")
        if i < len(names) - 1:
            buf.write("\n")
    # No newline after last row — matches C++ printToStream


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _section(buf: io.StringIO, title: str, names: list[str], mat: np.ndarray) -> None:
    """Print [title] … matrix … [end title] with blank line after end."""
    buf.write(f"[{title}]\n")
    if names and mat.size > 0:
        _print_matrix(buf, names, mat)
        buf.write("\n")
    buf.write(f"[end {title}]\n\n")


def _print_nuisance_pulls(
    buf: io.StringIO,
    sys_names: list[str],
    pulls: np.ndarray,
    constraints: np.ndarray,
) -> None:
    """Port of C++ nuisance-pulls table."""
    # Header: fixLength("Name",11) + fixLength("pull",7) + "constraint\n"
    buf.write("Name".ljust(11) + "pull".ljust(7) + "constraint\n")

    if not sys_names:
        return

    maxlength = max(len(n) for n in sys_names)

    for i, name in enumerate(sys_names):
        pull = float(pulls[i])
        constr = float(constraints[i])

        buf.write(name.ljust(maxlength + 1))
        buf.write(" ")

        # C++: if pull < 0 → fixLength(pull,6)+"   "
        #      else        → " " + fixLength(pull,5) + "   "
        if pull < 0.0:
            buf.write(_fix_length(pull, 6))
        else:
            buf.write(" ")
            buf.write(_fix_length(pull, 5))
        buf.write("   ")

        buf.write(_fix_length(constr, 5))
        buf.write("\n")


def _print_simple_impacts(
    buf: io.StringIO,
    sys_names: list[str],
    comb_names: list[str],
    combined_vals: np.ndarray,
    corr_full: np.ndarray,
    err_up: np.ndarray,
    err_down: np.ndarray,
    nsys: int,
) -> None:
    """
    Port of C++ printSimpleImpactTable.
    impact[i,j] = corr_full[i, nsys+j] * max(|err_up[j]|, |err_down[j]|)
    rel[i,j]    = |impact / combined[j]| * 100  (printed with full precision)
    """
    nest = len(comb_names)
    sym_errs = np.maximum(np.abs(err_up), np.abs(err_down))

    maxnuis = max((len(n) for n in sys_names), default=1)
    maxcomb = max((len(n) for n in comb_names), default=1)

    buf.write("[simple impact table: name, impact [%]]\n")

    # Header
    buf.write(" " * maxnuis + " ")
    for cname in comb_names:
        buf.write(cname.ljust(maxcomb))
        buf.write(" | ")
    buf.write("\n")

    # Rows
    for i, sname in enumerate(sys_names):
        buf.write(sname.ljust(maxnuis))
        buf.write(" | ")
        for j in range(nest):
            corrcoef = float(corr_full[i, nsys + j])
            comberr = float(sym_errs[j])
            combined_j = float(combined_vals[j])
            impact = corrcoef * comberr
            rel = abs(impact / combined_j) * 100.0 if abs(combined_j) > 0 else 0.0
            buf.write(_fix_length(rel, maxcomb))
            buf.write(" | ")
        buf.write("\n")

    buf.write("[end simple impact table: name, impact [%]]\n")
    buf.write("\n")


def _print_impact_table(
    buf: io.StringIO,
    comb_names: list[str],
    combined_vals: np.ndarray,
    impact_groups: dict,
) -> None:
    """Print merged impact table from user-defined impact groups."""
    nest = len(comb_names)
    maxcomb = max((len(n) for n in comb_names), default=1)
    maxnuis = max((len(k) for k in impact_groups), default=1)

    buf.write("[impact table: name, impact [%]]\n")
    buf.write(" " * maxnuis + " ")
    for cname in comb_names:
        buf.write(cname.ljust(maxcomb))
        buf.write(" | ")
    buf.write("\n")

    # NOTE: only the upward impact is printed, matching the C++ reference
    # formatter (combinationResult::printFullInfo). The downward impact is
    # computed and stored on the result but intentionally not shown in this
    # table.
    for label, (imp_up, _imp_down) in impact_groups.items():
        buf.write(label.ljust(maxnuis))
        buf.write(" | ")
        for j in range(nest):
            impact = float(imp_up[j]) if j < len(imp_up) else 0.0
            combined_j = float(combined_vals[j])
            rel = abs(impact / combined_j) * 100.0 if abs(combined_j) > 0 else 0.0
            buf.write(_fix_length(rel, maxcomb))
            buf.write(" | ")
        buf.write("\n")


def _print_merged_impact_cov(
    buf: io.StringIO,
    comb_names: list[str],
    impact_cov_groups: dict,
) -> None:
    """
    Port of C++ [covariance matrix for merged impacts].

    One nest×nest block per impact group: the covariance of the combined
    observables with that group frozen (its diagonal is the frozen error²).
    Each block is formatted with the same triangularMatrix scaler/layout as the
    other covariance sections.
    """
    buf.write("[covariance matrix for merged impacts]\n")
    buf.write("[covariance matrix for impact table]\n")
    for label, cov in impact_cov_groups.items():
        buf.write(f"[{label}]\n")
        _print_matrix(buf, comb_names, np.asarray(cov))
        buf.write("\n")  # terminate the final matrix row
        buf.write(f"[end {label}]\n")
    buf.write("\n[end covariance matrix for merged impacts]\n")


# ---------------------------------------------------------------------------
# Main formatter
# ---------------------------------------------------------------------------


def format_result(result: CombinationResult) -> str:
    """Return the full result string matching C++ combinationResult::printFullInfo."""
    buf = io.StringIO()
    nsys = result.nsys
    sys_names = result.sys_names
    comb_names = result.combined_names
    all_names = sys_names + comb_names

    post_sys_corr = result.corr_full[:nsys, :nsys]
    post_est_corr = result.corr_full[nsys:, nsys:]

    # Post-combine result covariance uses symmetric errors (max of up/down)
    sym_errs = np.maximum(
        np.abs(result.combined_err_up), np.abs(result.combined_err_down)
    )
    post_est_cov = post_est_corr * np.outer(sym_errs, sym_errs)

    # 1. Pre-combine systematics correlations
    _section(
        buf, "pre-combine systematics correlations", sys_names, result.pre_sys_corr
    )

    # 2. Post-combine systematics correlations
    _section(buf, "post-combine systematics correlations", sys_names, post_sys_corr)

    # 3. Pre-combine estimate correlations (blank — not available pre-fit)
    buf.write("[pre-combine estimate correlations]\n")
    buf.write("\n[end pre-combine estimate correlations]\n\n")

    # 4. Post-combine result correlations
    _section(buf, "post-combine result correlations", comb_names, post_est_corr)

    # 5. Post-combine result covariance
    _section(buf, "post-combine result covariance", comb_names, post_est_cov)

    # 6. Combined values (printResultOnly)
    chi2_str = f"{result.chi2_min:g}"
    buf.write(f"[combined (minimum chi^2={chi2_str})]\n")
    for k, name in enumerate(comb_names):
        v = float(result.combined_values[k])
        eu = float(result.combined_err_up[k])
        ed = float(result.combined_err_down[k])
        buf.write(f"{name}: {v:g} +{eu:g} -{ed:g}\n")
    buf.write(f"[end combined (minimum chi^2={chi2_str})]\n")

    # Merged impact table inside printResultOnly (if groups present)
    if result.impact_groups:
        buf.write("\n")
        _print_impact_table(
            buf, comb_names, result.combined_values, result.impact_groups
        )
        buf.write("[end impact table]\n\n")

    # 6b. Goodness of fit. New section, no C++ equivalent: the reference
    # combinationResult only ever stores/prints chi2min_, never an ndf or
    # p-value (grepped combinationResult.cpp/combiner.cpp, no match) — see
    # the ndf convention documented on CombinationResult in combiner.py.
    buf.write("\n[goodness of fit]\n")
    buf.write(f"ndf: {result.ndf:d}\n")
    buf.write(f"chi2/ndf: {result.chi2_per_ndf:g}\n")
    buf.write(f"p-value: {result.p_value:g}\n")
    # Only written when an input actually carries [nuisance values]; without
    # them the tension is the whole chi2 and the output stays the C++ one.
    if any(result.chi2_standalone.values()):
        for name, v in result.chi2_standalone.items():
            buf.write(f"chi2 standalone {name}: {v:g}\n")
        buf.write(f"chi2 tension: {result.chi2_tension:g}\n")
    buf.write("[end goodness of fit]\n")

    # 7. Full correlation matrix
    _section(buf, "full correlation matrix", all_names, result.corr_full)

    # 8. Full covariance matrix
    _section(buf, "full covariance matrix", all_names, result.cov_full)

    # 9. Nuisance pulls
    buf.write("[nuisance pulls]\n")
    _print_nuisance_pulls(buf, sys_names, result.pulls, result.constraints)
    buf.write("[end nuisance pulls]\n")

    # 10. Simple impacts
    buf.write("\n[simple impacts]\n")
    _print_simple_impacts(
        buf,
        sys_names,
        comb_names,
        result.combined_values,
        result.corr_full,
        result.combined_err_up,
        result.combined_err_down,
        nsys,
    )
    buf.write("\n[end simple impacts]\n")

    # 11. Merged impacts again (if groups present)
    # Intentionally a superset of the C++ reference: that implementation never
    # populates impacttable_ for a printFullInfo run on this setup, so its output
    # omits sections 11/12 entirely. We print them deliberately (decided to keep,
    # not a fidelity bug) since the per-group covariance has no C++ equivalent at
    # all and is useful on its own.
    if result.impact_groups:
        buf.write("\n[merged impacts]\n\n")
        _print_impact_table(
            buf, comb_names, result.combined_values, result.impact_groups
        )
        buf.write("\n[end merged impacts]\n")

    # 12. Covariance matrices for the merged impact groups
    if result.impact_cov_groups:
        buf.write("\n")
        _print_merged_impact_cov(buf, comb_names, result.impact_cov_groups)

    return buf.getvalue()


def prepare_output_path(prefix: str) -> Path:
    """
    Resolve the output path for `prefix`, create any parent directory it names,
    and verify the location is writable — all BEFORE the (expensive) fit runs.

    A prefix may include a directory component (e.g. "out/run1" →
    "out/run1_result.txt"). The parent directory is created if missing, so the
    combination does not run for minutes only to fail at the final write.

    Returns the resolved output path. Raises a clear error if the directory
    cannot be created or written to.
    """
    out_path = Path(f"{prefix}_result.txt")
    parent = out_path.parent

    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(
            f"cannot create output directory '{parent}' for prefix '{prefix}': {e}"
        ) from e

    if not os.access(parent, os.W_OK):
        raise ValueError(
            f"output directory '{parent}' is not writable (prefix '{prefix}')"
        )
    # Catch the case where the target exists but is not a writable file
    # (e.g. a directory, or a read-only file).
    if out_path.is_dir():
        raise ValueError(f"output path '{out_path}' is a directory, not a file")
    if out_path.exists() and not os.access(out_path, os.W_OK):
        raise ValueError(f"output file '{out_path}' exists and is not writable")

    return out_path


def write_result(result: CombinationResult, prefix: str = "convino") -> None:
    """Write the result text file to <prefix>_result.txt.

    Creates the prefix's parent directory if needed.
    """
    text = format_result(result)
    out_path = prepare_output_path(prefix)
    out_path.write_text(text, encoding="utf-8")
    print(f"Result written to {out_path}")


# ---------------------------------------------------------------------------
# Correlation scan output (Combiner.scan_correlations / CLI --scan)
# ---------------------------------------------------------------------------


def format_scan_result(scan_results: dict) -> str:
    """
    Concatenate `format_result()` for every step of every scan group, with a
    `===` header identifying the group/step/value before each block. The C++
    reference's `scan_result.txt` concatenates raw `printFullInfo` blocks with
    no such header (relying on the console log printed alongside it for
    context); since there is no byte-fidelity target for this new file, the
    header is added for readability.
    """
    buf = io.StringIO()
    for name, (values, steps) in scan_results.items():
        for i, (val, res) in enumerate(zip(values, steps, strict=True)):
            buf.write(
                f"=== scan group '{name}': step {i + 1}/{len(steps)}, "
                f"correlation={float(val):g} ===\n"
            )
            buf.write(format_result(res))
            buf.write("\n")
    return buf.getvalue()


def write_scan_result(scan_results: dict, prefix: str = "convino") -> Path:
    """Write the scan text log to <prefix>_scan_result.txt."""
    text = format_scan_result(scan_results)
    out_path = Path(f"{prefix}_scan_result.txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"Scan result written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Machine-readable export (npz / json)
# ---------------------------------------------------------------------------
#
# Built for a downstream covariance-style fit (e.g. a top-quark-mass chi2
# fit) that treats other theory uncertainties (PDF, scale) as named
# covariance matrices summed into the total experimental covariance, with a
# per-source breakdown coming out of the same leave-one-out machinery. See
# docs/improvement_plan.md Q4/Q4b for the full design.


def to_dict(result: CombinationResult) -> dict:
    """
    Flatten a CombinationResult into a self-describing, nested dict suitable
    for a downstream fit (or for export_npz/export_json below).

    `combined_covariance`/`combined_correlation` are the nest×nest slices of
    `cov_full`/`corr_full` — the marginal covariance over all profiled
    nuisance parameters, i.e. exactly what a downstream chi2 fit needs.

    `ndf`/`chi2_per_ndf`/`p_value` are the goodness-of-fit figures; see the
    ndf convention documented on `CombinationResult` in combiner.py.

    `stat_only_covariance` is the same covariance with every systematic
    frozen (pure statistical/measurement uncertainty); `total_syst_covariance`
    is `combined_covariance - stat_only_covariance`. Do not sum
    `combined_covariance` with the per-source matrices below — it already
    contains all of stat+syst, summing would double-count.

    `impact_per_systematic`/`cov_per_systematic` give a leave-one-out
    breakdown per individual systematic (independent of any user-defined
    `[uncertainty impacts]` groups, which are exposed separately as
    `impact_groups`/`impact_cov_groups`). As with any quadrature-based
    impact breakdown, per-source numbers are not required to sum in
    quadrature to the total if systematics are mutually correlated.
    """
    nsys = result.nsys
    combined_covariance = np.asarray(result.cov_full)[nsys:, nsys:]
    combined_correlation = np.asarray(result.corr_full)[nsys:, nsys:]
    total_syst_covariance = combined_covariance - np.asarray(
        result.stat_only_covariance
    )
    # Cross-covariance between combined observables and nuisances, Cov(x_b, nu_i),
    # shape (nest, nsys), from the post-fit covariance block. This is the sole
    # ingredient a downstream fit needs for genuine additive "global impacts"
    # (arXiv:2307.04007): for a POI mu = w·x, the global impact of nuisance i is
    # dm_i = Cov(mu, nu_i) = w·x_sys_cov[:, i], and sum_i dm_i^2 is exactly the
    # global-impacts systematic variance (columns are additive by construction).
    # NB this is a DIFFERENT decomposition from total_syst_covariance (the
    # freeze/conditional method) — they need not agree; see the paper.
    x_sys_cov = np.asarray(result.cov_full)[nsys:, :nsys]

    return {
        "combined_names": list(result.combined_names),
        "combined_values": np.asarray(result.combined_values),
        "combined_err_up": np.asarray(result.combined_err_up),
        "combined_err_down": np.asarray(result.combined_err_down),
        "combined_covariance": combined_covariance,
        "combined_correlation": combined_correlation,
        "chi2_min": float(result.chi2_min),
        "ndf": int(result.ndf),
        "chi2_per_ndf": float(result.chi2_per_ndf),
        "p_value": float(result.p_value),
        # chi2_min = sum of the inputs' own prior penalties (chi2_standalone,
        # from their [nuisance values]) + chi2_tension (from combining them).
        "chi2_standalone": {k: float(v) for k, v in result.chi2_standalone.items()},
        "chi2_tension": float(result.chi2_tension),
        "converged": bool(result.converged),
        "normalised": bool(result.normalised),
        "sys_names": list(result.sys_names),
        "sys_group": list(result.sys_group_labels),
        "pulls": np.asarray(result.pulls),
        "constraints": np.asarray(result.constraints),
        "x_sys_cov": x_sys_cov,
        # Prior precision inv_C (nsys x nsys); the correct global-impacts systematic
        # variance is g^T inv_C g (g_i = Cov(POI, nu_i)), which reduces to g^T g only
        # for independent unit priors. Needed to avoid double-counting correlated priors.
        "prior_inv_cov": np.asarray(result.prior_inv_cov),
        # Effective auxiliary precision (prior_inv_cov + negative input LD part),
        # see CombinationResult.prior_inv_cov_eff; use it for the global stat/syst split.
        "prior_inv_cov_eff": np.asarray(result.prior_inv_cov_eff),
        "stat_only_covariance": np.asarray(result.stat_only_covariance),
        "total_syst_covariance": total_syst_covariance,
        "total_syst_impact_up": np.asarray(result.total_syst_impact_up),
        "total_syst_impact_down": np.asarray(result.total_syst_impact_down),
        "impact_groups": {
            label: {"up": np.asarray(up), "down": np.asarray(down)}
            for label, (up, down) in result.impact_groups.items()
        },
        "impact_cov_groups": {
            label: np.asarray(cov) for label, cov in result.impact_cov_groups.items()
        },
        "impact_per_systematic": {
            name: {"up": np.asarray(up), "down": np.asarray(down)}
            for name, (up, down) in result.impact_per_systematic.items()
        },
        "cov_per_systematic": {
            name: np.asarray(cov) for name, cov in result.cov_per_systematic.items()
        },
        # Signed linear response matrix (nest × nsys): shift in combined_values[b]
        # for +1σ of nuisance i. Columns ordered as sys_names. See CombinationResult
        # docstring for the derivation from the Hessian cross-block.
        "impact_matrix": np.asarray(result.impact_matrix),
        # Impact-weighted mean pull per user-defined [uncertainty impacts] group.
        # Two weighting conventions (see CombinationResult docstring).
        "pull_per_group_mean": {
            k: float(v) for k, v in result.pull_per_group_mean.items()
        },
        "pull_per_group_norm": {
            k: float(v) for k, v in result.pull_per_group_norm.items()
        },
    }


def _flatten_for_npz(d: dict, prefix: str = "") -> dict[str, np.ndarray]:
    """Recursively flatten a to_dict()-shaped dict into a flat dict of arrays,
    joining nested keys with '__' (e.g. impact_groups__GroupA__up)."""
    out: dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_for_npz(v, prefix=f"{key}__"))
        else:
            out[key] = np.asarray(v)
    return out


def export_npz(result: CombinationResult, path) -> None:
    """Export `to_dict(result)` to a compressed .npz file (flat keys, '__'-joined)."""
    flat: dict[str, Any] = _flatten_for_npz(to_dict(result))
    np.savez_compressed(path, **flat)


def export_json(result: CombinationResult, path) -> None:
    """Export `to_dict(result)` to a JSON file (nested, ndarrays as lists)."""
    import json

    data = json.dumps(to_dict(result), indent=2, default=lambda o: o.tolist())
    Path(path).write_text(data, encoding="utf-8")


def scan_results_to_dict(scan_results: dict) -> dict:
    """
    Flatten `Combiner.scan_correlations()`'s output into a `to_dict()`-shaped
    nested dict: `{group_name: {"scan_values": [...], "steps": [to_dict(...),
    ...]}}`. The pythonic replacement for the C++ reference's ROOT
    TGraphAsymmErrors output (`-p` plots) — `scan_values` plus each step's
    `combined_values`/`combined_err_up`/`combined_err_down`/`chi2_min` is the
    same data the original plotted, usable directly with matplotlib.
    """
    return {
        name: {
            "scan_values": np.asarray(values),
            "steps": [to_dict(res) for res in steps],
        }
        for name, (values, steps) in scan_results.items()
    }


def export_scan_npz(scan_results: dict, path) -> None:
    """Export scan results to a compressed .npz (flat keys, '__'-joined)."""
    flat: dict[str, Any] = {}
    for name, (values, steps) in scan_results.items():
        flat[f"{name}__scan_values"] = np.asarray(values)
        for i, res in enumerate(steps):
            flat.update(_flatten_for_npz(to_dict(res), prefix=f"{name}__step{i}__"))
    np.savez_compressed(path, **flat)


def export_scan_json(scan_results: dict, path) -> None:
    """Export scan results to a JSON file (nested, ndarrays as lists)."""
    import json

    data = json.dumps(
        scan_results_to_dict(scan_results), indent=2, default=lambda o: o.tolist()
    )
    Path(path).write_text(data, encoding="utf-8")
