"""Translate one of our Convino setups into Combiner (Dado/Owen/Pinamonti) YAML.

Per measurement we write the form Combiner expects
(poiCovarianceMatrix + per-systematic impact/uncertainty/value + systematicCorrelations):

    O = (statistical covariance of the estimates) + sum_ext Gamma Gamma^T
    Gamma_ik = cov(estimate i, NP k)   -- the fit covariance block for profiled NPs,
                                          the [not fitted] response for externalised ones
    Sigma    = post-fit NP covariance for profiled NPs, unit for externalised ones

The Convino [correlations] block becomes byHandNPcorrelations. Usage:

    python3 to_combiner.py <convino-config> <out-dir> [--drop-chains]

--drop-chains keeps only correlation pairs whose two NPs appear in no other pair,
which is what Combiner's consistency check allows (see the closure-test note).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

from pyconvino.parser import parse_config_file, parse_measurement_file


def measurement_matrices(data):
    """Return (poi_names, poi_values, O, np_names, Gamma, Sigma, np_values)."""
    est_names = list(data.estimates)
    npoi = len(est_names)
    H = np.array(data.hessian, dtype=float) if data.hessian else None
    hn = list(data.hessian_names)

    prof_names, Gp, Sp, O = [], np.zeros((npoi, 0)), np.zeros((0, 0)), np.zeros((npoi, npoi))
    if H is not None:
        C = np.linalg.inv(H)
        ipoi = [hn.index(n) for n in est_names]
        prof_names = [n for n in hn if n not in est_names]
        ip = [hn.index(n) for n in prof_names]
        O = C[np.ix_(ipoi, ipoi)]
        Gp = C[np.ix_(ipoi, ip)]
        Sp = C[np.ix_(ip, ip)]

    ext_names = list(data.ext_sys_names)
    Ge = np.zeros((npoi, len(ext_names)))
    for i, e in enumerate(est_names):
        for k, s in enumerate(ext_names):
            u = data.externalized.get(e, {}).get(s)
            if u is not None:
                Ge[i, k] = u.up          # Convino stores cov(estimate, NP) as the "up" response
    O = O + Ge @ Ge.T                     # externalised sources add to the total POI covariance

    names = prof_names + ext_names
    G = np.hstack([Gp, Ge])
    S = np.eye(len(names))
    S[: len(prof_names), : len(prof_names)] = Sp
    vals = np.array([data.nuisance_values.get(n, 0.0) for n in names])
    return est_names, np.array([data.estimates[n] for n in est_names]), O, names, G, S, vals


def tri(m):
    return ["    - row: [" + ",".join(f"{m[i, j]:.12g}" for j in range(i + 1)) + "]"
            for i in range(len(m))]


def write_measurement(path, poi_names, poi_vals, O, np_names, G, S, np_vals, obs_of):
    sig = np.sqrt(np.diag(S))
    corr = S / np.outer(sig, sig)
    L = ["pois:"]
    for n, v in zip(poi_names, poi_vals):
        L += [f"  - name: {obs_of[n]}", f"    value: {v:.12g}"]
    L += ["", "poiCovarianceMatrix:", "  parameters: [" + ", ".join(obs_of[n] for n in poi_names) + "]",
          "  matrix:"] + tri(O) + ["", "systematics:"]
    for k, n in enumerate(np_names):
        L += [f"  - parameter: {n}",
              "    impact: [" + ", ".join(f"{G[i, k]:.12g}" for i in range(len(poi_names))) + "]",
              f"    uncertainty: {sig[k]:.12g}", f"    value: {np_vals[k]:.12g}"]
    L += ["", "systematicCorrelations:", "  parameters: [" + ", ".join(np_names) + "]",
          "  matrix:"] + tri(corr)
    open(path, "w").write("\n".join(L) + "\n")


def main(config: str, outdir: str, drop_chains: bool = False) -> None:
    cfg = parse_config_file(config)
    os.makedirs(outdir, exist_ok=True)
    obs_of = {est: obs for obs, ests in cfg.observables.items() for est in ests}

    names = []
    for path in cfg.measurement_files:
        data = parse_measurement_file(path)
        label = os.path.basename(path).replace(".txt", "")
        pn, pv, O, nn, G, S, nv = measurement_matrices(data)
        missing = [n for n in pn if n not in obs_of]
        if missing:
            raise SystemExit(f"{label}: estimates not mapped to an observable: {missing}")
        write_measurement(os.path.join(outdir, f"{label}.yml"), pn, pv, O, nn, G, S, nv, obs_of)
        names.append(label)
        print(f"  wrote {label}.yml: {len(pn)} POIs, {len(nn)} systematics")

    pairs = [(r.name_a, r.name_b, r.nominal) for s in cfg.correlation_scans
             for r in s.ranges if abs(r.nominal) > 0]
    if drop_chains:
        deg = defaultdict(int)
        for a, b, _ in pairs:
            deg[a] += 1
            deg[b] += 1
        kept = [(a, b, v) for a, b, v in pairs if deg[a] == 1 and deg[b] == 1]
        print(f"  correlations: {len(pairs)} non-zero, {len(kept)} kept as isolated pairs")
        pairs = kept

    L = ["general:", f"  outputPath: {os.path.basename(outdir)}_out/", "  debug: info"]
    if pairs:                                    # byHandNPcorrelations lives in the general block
        L += ["  byHandNPcorrelations:"]
        for a, b, v in pairs:
            L += [f"    - parameter1: {a}", f"      parameter2: {b}", f"      correlation: {v:.12g}"]
    L += ["", "measurements:"] + [f"  - name: {n}" for n in names] + [""]
    for n in names:
        L += [f"{n}:", f"  configFile: {outdir}/{n}.yml", ""]
    open(os.path.join(outdir, "main.yml"), "w").write("\n".join(L) + "\n")
    print(f"  wrote main.yml with {len(pairs)} by-hand correlations")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], "--drop-chains" in sys.argv)
