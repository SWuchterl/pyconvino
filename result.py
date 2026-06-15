"""
Output formatter: produce the Convino result .txt file.
Matches combinationResult::printFullInfo() from the C++ source exactly.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

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
    if v < 0.0:
        l_eff = l + 1
    else:
        l_eff = l

    s = f"{v:.{l_eff + 10}f}"   # fixed format, very high precision

    # Truncate to l_eff (take first l_eff chars) — truncate=true default
    if len(s) > l_eff:
        s = s[:l_eff]
    elif len(s) < l_eff:
        s = s + ' ' * (l_eff - len(s))

    # Handle trailing '.' edge case (replace with space, shorten by 1)
    if s.endswith('.'):
        s = s[:-1] + ' '

    # Prepend space for strictly positive values
    if v > 0.0:
        s = ' ' + s

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
                buf.write(' ')
        if i < len(names) - 1:
            buf.write('\n')
    # No newline after last row — matches C++ printToStream


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _section(buf: io.StringIO, title: str, names: list[str], mat: np.ndarray) -> None:
    """Print [title] … matrix … [end title] with blank line after end."""
    buf.write(f"[{title}]\n")
    if names and mat.size > 0:
        _print_matrix(buf, names, mat)
        buf.write('\n')
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
        buf.write(' ')

        # C++: if pull < 0 → fixLength(pull,6)+"   "
        #      else        → " " + fixLength(pull,5) + "   "
        if pull < 0.0:
            buf.write(_fix_length(pull, 6))
        else:
            buf.write(' ')
            buf.write(_fix_length(pull, 5))
        buf.write('   ')

        buf.write(_fix_length(constr, 5))
        buf.write('\n')


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
    buf.write(' ' * maxnuis + ' ')
    for cname in comb_names:
        buf.write(cname.ljust(maxcomb))
        buf.write(' | ')
    buf.write('\n')

    # Rows
    for i, sname in enumerate(sys_names):
        buf.write(sname.ljust(maxnuis))
        buf.write(' | ')
        for j in range(nest):
            corrcoef = float(corr_full[i, nsys + j])
            comberr = float(sym_errs[j])
            combined_j = float(combined_vals[j])
            impact = corrcoef * comberr
            rel = abs(impact / combined_j) * 100.0 if abs(combined_j) > 0 else 0.0
            buf.write(_fix_length(rel, maxcomb))
            buf.write(' | ')
        buf.write('\n')

    buf.write("[end simple impact table: name, impact [%]]\n")
    buf.write('\n')


def _print_impact_table(
    buf: io.StringIO,
    comb_names: list[str],
    combined_vals: np.ndarray,
    impact_groups: dict,
) -> None:
    """Print merged impact table from user-defined impact groups."""
    nest = len(comb_names)
    maxcomb = max((len(n) for n in comb_names), default=1)
    maxnuis = max((len(k) for k in impact_groups.keys()), default=1)

    buf.write("[impact table: name, impact [%]]\n")
    buf.write(' ' * maxnuis + ' ')
    for cname in comb_names:
        buf.write(cname.ljust(maxcomb))
        buf.write(' | ')
    buf.write('\n')

    # NOTE: only the upward impact is printed, matching the C++ reference
    # formatter (combinationResult::printFullInfo). The downward impact is
    # computed and stored on the result but intentionally not shown in this
    # table.
    for label, (imp_up, _imp_down) in impact_groups.items():
        buf.write(label.ljust(maxnuis))
        buf.write(' | ')
        for j in range(nest):
            impact = float(imp_up[j]) if j < len(imp_up) else 0.0
            combined_j = float(combined_vals[j])
            rel = abs(impact / combined_j) * 100.0 if abs(combined_j) > 0 else 0.0
            buf.write(_fix_length(rel, maxcomb))
            buf.write(' | ')
        buf.write('\n')


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
        buf.write("\n")                       # terminate the final matrix row
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
    sym_errs = np.maximum(np.abs(result.combined_err_up), np.abs(result.combined_err_down))
    post_est_cov = post_est_corr * np.outer(sym_errs, sym_errs)

    # 1. Pre-combine systematics correlations
    _section(buf, "pre-combine systematics correlations",
             sys_names, result.pre_sys_corr)

    # 2. Post-combine systematics correlations
    _section(buf, "post-combine systematics correlations",
             sys_names, post_sys_corr)

    # 3. Pre-combine estimate correlations (blank — not available pre-fit)
    buf.write("[pre-combine estimate correlations]\n")
    buf.write("\n[end pre-combine estimate correlations]\n\n")

    # 4. Post-combine result correlations
    _section(buf, "post-combine result correlations",
             comb_names, post_est_corr)

    # 5. Post-combine result covariance
    _section(buf, "post-combine result covariance",
             comb_names, post_est_cov)

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
        buf.write('\n')
        _print_impact_table(buf, comb_names, result.combined_values,
                            result.impact_groups)
        buf.write("[end impact table]\n\n")

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
        buf, sys_names, comb_names,
        result.combined_values, result.corr_full,
        result.combined_err_up, result.combined_err_down,
        nsys,
    )
    buf.write("\n[end simple impacts]\n")

    # 11. Merged impacts again (if groups present)
    if result.impact_groups:
        buf.write("\n[merged impacts]\n\n")
        _print_impact_table(buf, comb_names, result.combined_values,
                            result.impact_groups)
        buf.write("\n[end merged impacts]\n")

    # 12. Covariance matrices for the merged impact groups
    if result.impact_cov_groups:
        buf.write("\n")
        _print_merged_impact_cov(buf, comb_names, result.impact_cov_groups)

    return buf.getvalue()


def write_result(result: CombinationResult, prefix: str = "convino") -> None:
    """Write the result text file to <prefix>_result.txt."""
    text = format_result(result)
    out_path = Path(f"{prefix}_result.txt")
    out_path.write_text(text)
    print(f"Result written to {out_path}")
