"""Compare Combiner and the two pyconvino modes on OUR ATLAS+CMS combination.

Combiner cannot take our full correlation model (its consistency check rejects
transitively linked nuisance parameters), so the comparison uses the reduced
model that it accepts: only the correlation pairs whose two nuisances appear in
no other pair. Both tools then see exactly the same assumptions.

Usage: python3 compare_ourcomb.py <convino-config> <combiner-log> <reduced-setup-dir>
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import numpy as np

from pyconvino import Combiner
from pyconvino.parser import parse_config_file


def isolated_pairs(cfg):
    pairs = [(r.name_a, r.name_b, r.nominal) for s in cfg.correlation_scans
             for r in s.ranges if abs(r.nominal) > 0]
    deg = defaultdict(int)
    for a, b, _ in pairs:
        deg[a] += 1
        deg[b] += 1
    return [(a, b, v) for a, b, v in pairs if deg[a] == 1 and deg[b] == 1]


def write_reduced_setup(config: str, outdir: str) -> str:
    """Copy the setup with only the isolated correlation pairs (symlinked inputs)."""
    cfg = parse_config_file(config)
    os.makedirs(outdir, exist_ok=True)
    src = os.path.dirname(os.path.abspath(config))
    for p in cfg.measurement_files:
        dst = os.path.join(outdir, os.path.basename(p))
        if not os.path.islink(dst):
            os.symlink(os.path.abspath(p), dst)          # AFS: symlink, never hardlink
    text = open(config).read()
    keep = isolated_pairs(cfg)
    block = "[correlations]\n" + "\n".join(
        f"  {a} = ({v:g}) {b}" for a, b, v in keep) + "\n[end correlations]"
    text = re.sub(r"\[correlations\].*?\[end correlations\]", block, text, flags=re.S)
    out = os.path.join(outdir, "rho_config.txt")
    open(out, "w").write(text)
    return out


def combiner_results(log: str) -> dict[str, tuple[float, float]]:
    res = {}
    for m in re.finditer(r"BLUE combination of (\S+) = (\S+) \+- (\S+)", open(log).read()):
        res[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return res


def main(config: str, log: str, reduced_dir: str) -> None:
    red_cfg = write_reduced_setup(config, reduced_dir)
    comb = combiner_results(log)
    runs = {}
    for use_values in (False, True):
        r = Combiner.from_config(red_cfg, compute_impacts=False,
                                 use_nuisance_values=use_values).combine()
        runs[use_values] = r
    r0, r1 = runs[False], runs[True]
    names = list(r0.combined_names)

    print(f"{'observable':46s} {'input':>12} {'Combiner':>12} {'pyconv def':>12} {'pyconv +val':>12}")
    print(f"{'':46s} {'+- unc':>12} {'+- unc':>12} {'+- unc':>12} {'+- unc':>12}")
    for i, n in enumerate(names):
        if n not in comb:
            continue
        cv, ce = comb[n]
        print(f"{n[:46]:46s} {'':>12} {cv:12.4f} {r0.combined_values[i]:12.4f} {r1.combined_values[i]:12.4f}")
        print(f"{'':46s} {'':>12} {ce:12.4f} {r0.combined_err_up[i]:12.4f} {r1.combined_err_up[i]:12.4f}")

    cv = np.array([comb[n][0] for n in names]); ce = np.array([comb[n][1] for n in names])
    for tag, r in (("default", r0), ("--use-nuisance-values", r1)):
        dv = (r.combined_values - cv) / ce
        du = r.combined_err_up / ce - 1
        print(f"\npyconvino {tag:22s} vs Combiner: value shift max {np.abs(dv).max():.3f} sigma, "
              f"uncertainty ratio min {du.min()+1:.4f} max {du.max()+1:.4f}")
    print(f"chi2_min: default {float(r0.chi2_min):.4g}, with values {float(r1.chi2_min):.4g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
