"""
Parse Convino text-based measurement and configuration files.

Supported formats
-----------------
Measurement files
  [hessian]         – lower-triangular Hessian matrix (optional constraint column)
  [correlation matrix] – lower-triangular correlation matrix with constraint column
  [not fitted]      – table of externalised (non-fitted) uncertainties
  [estimates]       – central values of the observables
  [systematics]     – type tags (absolute/relative/lognormal) per systematic;
                      "free" marks a Hessian parameter fitted without a prior
  [nuisance values] – post-fit central values (pulls) of the Hessian nuisances

Configuration files
  [global]          – isDifferential, normalise flags
  [inputs]          – list of measurement files
  [observables]     – association of estimates → combined name
  [correlations]    – cross-measurement systematic correlation assumptions
  [uncertainty impacts] – grouped uncertainty impacts to evaluate
  #!FILE = path     – include an external file into the correlations block
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Remove inline comment (everything from the first '#' on)."""
    pos = line.find("#")
    if pos >= 0:
        line = line[:pos]
    return line.strip()


def _read_blocks(path: Path) -> dict[str, list[str]]:
    """
    Read the file and return a dict mapping block-name → list of non-empty,
    comment-stripped lines inside that block.

    Handles the #!FILE = <path> directive by inlining the referenced file's
    content into the current block at the point of the directive.
    """
    blocks: dict[str, list[str]] = {}
    current_block: Optional[str] = None
    current_lines: list[str] = []

    def _consume(raw_lines: list[str], base_dir: Path) -> None:
        nonlocal current_block, current_lines
        for raw in raw_lines:
            # Handle #!FILE include directive BEFORE stripping comments
            m_file = re.match(r'\s*#!FILE\s*=\s*(.+)', raw)
            if m_file and current_block is not None:
                # Drop any trailing inline comment after the path. We cannot use
                # _strip_comment on the whole line because the directive itself
                # starts with '#'.
                inc = m_file.group(1)
                hash_pos = inc.find('#')
                if hash_pos >= 0:
                    inc = inc[:hash_pos]
                include_path = base_dir / inc.strip()
                inc_lines = include_path.read_text().splitlines()
                _consume(inc_lines, include_path.parent)
                continue

            line = _strip_comment(raw)
            if not line:
                continue

            # Detect block start: [marker] (but not [end marker])
            m_start = re.match(r'^\[([^\]]+)\]$', line, re.IGNORECASE)
            if m_start:
                tag = m_start.group(1).strip().lower()
                if tag.startswith('end '):
                    # Close the block
                    name = tag[4:].strip()
                    if current_block and current_block == name:
                        blocks[current_block] = current_lines
                        current_block = None
                        current_lines = []
                else:
                    if current_block is not None:
                        blocks[current_block] = current_lines  # save if no [end] marker
                    current_block = tag
                    current_lines = []
                continue

            if current_block is not None:
                current_lines.append(line)

    raw_text = Path(path).read_text().splitlines()
    _consume(raw_text, Path(path).parent)
    # Flush a block left open at end-of-file (i.e. with no explicit [end]
    # marker); otherwise its lines would be silently dropped.
    if current_block is not None:
        blocks[current_block] = current_lines
    return blocks


def _parse_uncertainty_str(s: str) -> tuple[float, float]:
    """
    Parse an asymmetric uncertainty string.

    Forms:
      "6.1"        → (6.1,  -6.1)  symmetric
      "(+5-3)"     → (5.0,  -3.0)
      "(-3+4)"     → (-3.0,  4.0)
      "(+5+3)"     → (5.0,   3.0)
    Returns (upvar, downvar) matching uncertainty::readFromString.
    """
    s = s.strip()
    if '(' not in s:
        v = float(s)
        return v, -v

    s = s.replace(' ', '')
    s = s.strip('()')
    # Find the separator between the up and down variations: the last '+'/'-'
    # that is the sign of the down value rather than part of an exponent
    # (e.g. the '-' in '1.2e-3' must not be treated as the separator).
    sep = -1
    for i in range(len(s) - 1, 0, -1):
        if s[i] in '+-' and s[i - 1] not in 'eE':
            sep = i
            break
    if sep <= 0:  # no separator found → a single (symmetric) value in parens
        v = float(s)
        return v, -v
    first = s[:sep]
    second = s[sep:]

    if first.startswith('+'):
        first = first[1:]
    if second.startswith('+'):
        second = second[1:]

    return float(first), float(second)


def _parse_triangular(lines: list[str]) -> tuple[list[str], list[Optional[float]], list[list[float]]]:
    """
    Parse a lower-triangular matrix block.

    The format is one parameter per row:
        name0 [constraint] val00
        name1 [constraint] val10 val11
        ...

    Constraints are optional. They appear as '(value)' immediately after the
    name (detected on the first non-empty row).

    Returns
    -------
    names       : list of parameter names
    constraints : list of constraint values (None if absent)
    matrix      : square symmetric matrix as list-of-lists
    """
    names: list[str] = []
    constraints: list[Optional[float]] = []
    rows: list[list[float]] = []

    for i, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        name = tokens[0]
        names.append(name)

        # A constraint is present on this row iff the token directly after the
        # name is a parenthesised value, e.g. '(0.5)'. Detected per row (not
        # inferred once from row 0) so a block whose rows are not all uniform
        # is still parsed correctly.
        if len(tokens) > 1 and tokens[1].startswith('(') and tokens[1].endswith(')'):
            constraints.append(float(tokens[1].strip('()')))
            values = [float(t) for t in tokens[2:]]
        else:
            constraints.append(None)
            values = [float(t) for t in tokens[1:]]

        rows.append(values)

    n = len(names)
    matrix = [[0.0] * n for _ in range(n)]
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            matrix[i][j] = v
            matrix[j][i] = v  # symmetrise

    return names, constraints, matrix


# ---------------------------------------------------------------------------
# Data classes (raw file data before mathematical setup)
# ---------------------------------------------------------------------------

@dataclass
class MeasurementFileData:
    """Raw data read from a single measurement file."""
    path: str = ""

    # Central values and stat errors keyed by estimate name
    estimates: dict[str, float] = field(default_factory=dict)
    stat_errors: dict[str, float] = field(default_factory=dict)

    # Hessian (if provided) – (names, matrix as 2D list)
    hessian_names: list[str] = field(default_factory=list)
    hessian: list[list[float]] = field(default_factory=list)  # symmetric n×n

    # Externalized (not-fitted) uncertainties: est_name → sys_name → (up, down)
    externalized: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    # Just the names of the externalised systematics in order
    ext_sys_names: list[str] = field(default_factory=list)

    # Systematic types: name → "absolute" | "relative" | "lognormal"
    sys_types: dict[str, str] = field(default_factory=dict)

    # Hessian nuisances fitted without a Gaussian prior in the input fit
    prior_free: set[str] = field(default_factory=set)
    # Post-fit central values of the Hessian nuisances (0 if absent)
    nuisance_values: dict[str, float] = field(default_factory=dict)


@dataclass
class ScanRange:
    """Correlation scan specification for a single (a, b) pair."""
    name_a: str = ""
    name_b: str = ""
    nominal: float = 0.0
    low: float = 0.0
    high: float = 0.0


@dataclass
class CorrelationScan:
    """One named set of scan ranges (may span multiple (a,b) pairs)."""
    name: str = ""
    ranges: list[ScanRange] = field(default_factory=list)


@dataclass
class ConfigData:
    """Raw data read from the top-level config file."""
    is_differential: bool = False
    normalise: bool = False

    measurement_files: list[str] = field(default_factory=list)

    # combined_name → [est_name, ...]
    observables: dict[str, list[str]] = field(default_factory=dict)

    # Correlation scans (first entry of each set is the nominal assumption)
    correlation_scans: list[CorrelationScan] = field(default_factory=list)

    # Impact groups: label → [sys_name, ...]
    impact_groups: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Measurement file parser
# ---------------------------------------------------------------------------

def parse_measurement_file(path: str | Path) -> MeasurementFileData:
    """Parse a Convino measurement file and return raw data."""
    path = Path(path)
    data = MeasurementFileData(path=str(path))
    blocks = _read_blocks(path)

    # ---- Hessian block ------------------------------------------------
    if 'hessian' in blocks and blocks['hessian']:
        names, _, matrix = _parse_triangular(blocks['hessian'])
        data.hessian_names = names
        data.hessian = matrix

    # ---- Correlation matrix block -------------------------------------
    elif 'correlation matrix' in blocks and blocks['correlation matrix']:
        lines = blocks['correlation matrix']
        names, constraints, corr = _parse_triangular(lines)
        n = len(names)

        if any(c is None for c in constraints):
            raise ValueError(
                f"{path}: correlation matrix requires constraints for all parameters"
            )

        # Zero out small off-diagonal correlations (|rho| < 1e-4) – matches C++
        corr = np.asarray(corr, dtype=float)
        sigma = np.asarray(constraints, dtype=float)
        corr[~np.eye(n, dtype=bool) & (np.abs(corr) < 1e-4)] = 0.0

        # Build covariance: cov[i,j] = (rho[i,j] * sigma_i) * sigma_j, invert to
        # Hessian. The (.. * sigma_i) * sigma_j association matches the original
        # scalar loop bit-for-bit (np.outer would reassociate to sigma_i*sigma_j).
        C = corr * sigma[:, None] * sigma[None, :]
        # Positive-definiteness must be tested via the eigenvalues (det > 0 is
        # NOT a PD test — an even number of negative eigenvalues also gives a
        # positive determinant) and BEFORE inverting, so an indefinite
        # covariance is rejected rather than silently inverted into a non-PD
        # Hessian.
        if float(np.linalg.eigvalsh(C).min()) <= 0:
            raise ValueError(f"{path}: covariance from correlation matrix not positive definite")
        try:
            H = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            raise ValueError(f"{path}: correlation matrix covariance not invertible")

        data.hessian_names = names
        data.hessian = H.tolist()

    # ---- Not-fitted block --------------------------------------------
    if 'not fitted' in blocks and blocks['not fitted']:
        lines = blocks['not fitted']
        # First line is the header. Each header column maps positionally to a
        # value column of the data rows (after the leading estimate name). The
        # 'stat' column may appear at any position, not only last.
        header = lines[0].split()
        sys_names = [h for h in header if h != 'stat']
        data.ext_sys_names = sys_names

        ext: dict[str, dict[str, tuple[float, float]]] = {}

        for line in lines[1:]:
            tokens = line.split()
            if not tokens:
                continue
            est_name = tokens[0]
            ext[est_name] = {}

            for col, hname in enumerate(header):
                tok_idx = col + 1  # skip the leading estimate name
                if tok_idx >= len(tokens):
                    break
                if hname == 'stat':
                    data.stat_errors[est_name] = float(tokens[tok_idx])
                else:
                    ext[est_name][hname] = _parse_uncertainty_str(tokens[tok_idx])

        data.externalized = ext

    # ---- Estimates block ----------------------------------------------
    if 'estimates' in blocks:
        lines = blocks['estimates']
        # Parse key=value pairs
        kv = _parse_kv(lines)
        n_est = int(kv.get('n_estimates', 0))
        for i in range(n_est):
            name = kv[f'name_{i}'].strip().rstrip(';')
            val_str = kv[f'value_{i}'].strip().rstrip(';')
            data.estimates[name] = float(val_str)

    # ---- Systematics block -------------------------------------------
    if 'systematics' in blocks:
        lines = blocks['systematics']
        for line in lines:
            if '=' not in line:
                continue
            parts = line.split('=', 1)
            name = parts[0].strip()
            stype = parts[1].strip().lower()
            if stype == 'free':
                data.prior_free.add(name)
            else:
                data.sys_types[name] = stype

    # ---- Nuisance values block ---------------------------------------
    if 'nuisance values' in blocks:
        for line in blocks['nuisance values']:
            if '=' not in line:
                continue
            name, val = line.split('=', 1)
            data.nuisance_values[name.strip()] = float(val)

    return data


# ---------------------------------------------------------------------------
# Config file parser
# ---------------------------------------------------------------------------

def parse_config_file(path: str | Path) -> ConfigData:
    """Parse a Convino top-level configuration file."""
    path = Path(path)
    config_dir = path.parent
    blocks = _read_blocks(path)
    cfg = ConfigData()

    # ---- Global block -------------------------------------------------
    if 'global' in blocks:
        kv = _parse_kv(blocks['global'])
        cfg.is_differential = _parse_bool(kv.get('isdifferential', 'false'))
        cfg.normalise = _parse_bool(kv.get('normalise', 'false'))

    # ---- Inputs block ------------------------------------------------
    if 'inputs' in blocks:
        kv = _parse_kv(blocks['inputs'])
        n = int(kv.get('nfiles', 0))
        for i in range(n):
            rel = kv[f'file{i}'].strip()
            cfg.measurement_files.append(str(config_dir / rel))

    # ---- Observables block -------------------------------------------
    if 'observables' in blocks:
        for line in blocks['observables']:
            if '=' not in line:
                continue
            parts = line.split('=', 1)
            combined_name = parts[0].strip()
            rhs = parts[1].strip()
            contributors = [s.strip() for s in rhs.split('+') if s.strip()]
            cfg.observables[combined_name] = contributors

    # ---- Correlations block ------------------------------------------
    if 'correlations' in blocks:
        _parse_correlations(blocks['correlations'], cfg, config_dir)

    # ---- Uncertainty impacts block -----------------------------------
    if 'uncertainty impacts' in blocks:
        for line in blocks['uncertainty impacts']:
            if '=' not in line:
                continue
            parts = line.split('=', 1)
            label = parts[0].strip()
            members = [s.strip() for s in parts[1].split('+') if s.strip()]
            cfg.impact_groups[label] = members

    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_kv(lines: list[str]) -> dict[str, str]:
    """Return a dict of key → value from key=value lines (case-insensitive keys)."""
    kv: dict[str, str] = {}
    for line in lines:
        if '=' not in line:
            continue
        parts = line.split('=', 1)
        kv[parts[0].strip().lower()] = parts[1].strip()
    return kv


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ('true', '1', 'yes')


def _parse_correlations(lines: list[str], cfg: ConfigData, base_dir: Path) -> None:
    """
    Parse the [correlations] block and populate cfg.correlation_scans.

    Syntax per line:
        sys_a = (nominal & low : high) sys_b + (nominal) sys_c + ...

    Scan range is optional: "(0.2)" with no "&" means low=high=nominal (no scan).
    """
    for line in lines:
        if '=' not in line:
            continue
        parts = line.split('=', 1)
        name_a = parts[0].strip()
        rhs = parts[1].strip()

        scan = CorrelationScan(name=name_a)

        contributions = [c.strip() for c in rhs.split('+') if c.strip()]
        for contrib in contributions:
            # Split on ')' to separate "(nominal & low : high)" from "sys_b"
            parens_end = contrib.find(')')
            if parens_end < 0:
                continue
            corr_spec = contrib[:parens_end + 1].strip().strip('()')
            name_b = contrib[parens_end + 1:].strip()
            if not name_b:
                continue

            sr = ScanRange(name_a=name_a, name_b=name_b)
            if '&' in corr_spec:
                nom_part, range_part = corr_spec.split('&', 1)
                sr.nominal = float(nom_part.strip())
                if ':' in range_part:
                    low_s, high_s = range_part.split(':', 1)
                    sr.low = float(low_s.strip())
                    sr.high = float(high_s.strip())
                else:
                    sr.low = sr.high = sr.nominal
            else:
                sr.nominal = float(corr_spec.strip())
                sr.low = sr.high = sr.nominal

            scan.ranges.append(sr)

        if scan.ranges:
            cfg.correlation_scans.append(scan)
