"""Closure test against Combiner v1.0.0 (Dado/Owen/Pinamonti) using the authors' own configs.

Builds each measurement's extended covariance exactly as Combiner does
(Measurement::CalculateFullCovarianceMatrix + CombinedCovarianceMatrix):

    C_meas = [[ O , G ], [ G^T , S ]]      G = per-systematic "impact" (= cov(POI,NP))
    C_AB   = [[ G_A G_B^T , G_A S_B ], [ S_A G_B^T , S_A S_B ]]   (rho = identity)

then runs BLUE (paper Eq. 2.3) and pyconvino in both modes on the same inputs.
Usage: python3 closure.py <combiner-repo-path>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

from convino_io import run_convino

CFG = "configs/paper"


def load_measurement(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (names, values, covariance) with the POIs first."""
    d = yaml.safe_load(open(path))
    pois = [p["name"] for p in d["pois"]]
    systs = [s["parameter"] for s in d.get("systematics", [])]
    vals = np.array([p["value"] for p in d["pois"]]
                    + [s.get("value", 0.0) for s in d.get("systematics", [])])
    npoi, nsys = len(pois), len(systs)

    if "fullCovarianceMatrix" in d:
        blk = d["fullCovarianceMatrix"]
        order = blk["parameters"]
        C = np.zeros((len(order), len(order)))
        for i, row in enumerate(blk["matrix"]):
            C[i, : i + 1] = row["row"]
            C[: i + 1, i] = row["row"]
        idx = [order.index(n) for n in pois + systs]
        C = C[np.ix_(idx, idx)]
    else:
        C = np.zeros((npoi + nsys, npoi + nsys))
        blk = d["poiCovarianceMatrix"]
        for i, row in enumerate(blk["matrix"]):
            C[i, : i + 1] = row["row"]
            C[: i + 1, i] = row["row"]
        for k, s in enumerate(d["systematics"]):
            C[:npoi, npoi + k] = C[npoi + k, :npoi] = s["impact"]
        sig = np.array([s.get("uncertainty", 1.0) for s in d["systematics"]])
        corr = np.eye(nsys)
        if "systematicCorrelations" in d:
            sc = d["systematicCorrelations"]
            m = np.eye(len(sc["parameters"]))
            for i, row in enumerate(sc["matrix"]):
                m[i, : i + 1] = row["row"]
                m[: i + 1, i] = row["row"]
            j = [sc["parameters"].index(n) for n in systs]
            corr = m[np.ix_(j, j)]
        C[npoi:, npoi:] = corr * np.outer(sig, sig)
    return pois + systs, vals, C


def blue(names_list, vals_list, covs_list, npoi_list, combine_nps, only_nps=None):
    """Paper Eq. 2.1-2.3 on the union of parameters. Returns (values, cov, weights, kept).

    combine_nps=False combines the POIs only; only_nps restricts the combined NPs
    to that list (Combiner's npsToFit). NPs that are not combined still enter
    through the covariance, exactly as Combiner drops their rows.
    """
    def kept_names(names, npoi):
        if not combine_nps:
            return names[:npoi]
        if only_nps is None:
            return names
        return names[:npoi] + [n for n in names[npoi:] if n in only_nps]

    union = []
    for names, npoi in zip(names_list, npoi_list):
        for n in kept_names(names, npoi):
            if n not in union:
                union.append(n)
    rows, y = [], []
    for m, (names, vals, C, npoi) in enumerate(zip(names_list, vals_list, covs_list, npoi_list)):
        keep = [names.index(n) for n in kept_names(names, npoi)]
        for i in keep:
            rows.append((m, i, names[i]))
            y.append(vals[i])
    n = len(rows)
    CAB = np.zeros((n, n))
    for a, (ma, ia, _) in enumerate(rows):
        for b, (mb, ib, _) in enumerate(rows):
            if ma == mb:
                CAB[a, b] = covs_list[ma][ia, ib]
            else:                                    # G_i rho G_j^T with rho = I over shared NPs
                Ca, Cb = covs_list[ma], covs_list[mb]
                na, nb = names_list[ma], names_list[mb]
                npa, npb = npoi_list[ma], npoi_list[mb]
                shared = [(na.index(s), nb.index(s)) for s in na[npa:] if s in nb[npb:]]
                CAB[a, b] = sum(Ca[ia, ja] * Cb[ib, jb] for ja, jb in shared)
    U = np.zeros((n, len(union)))
    for a, (_, _, nm) in enumerate(rows):
        U[a, union.index(nm)] = 1.0
    Ci = np.linalg.inv(CAB)
    cov = np.linalg.inv(U.T @ Ci @ U)
    W = cov @ U.T @ Ci
    return W @ np.array(y), cov, W, union


def write_convino(d: str, label: str, names, vals, C, npoi: int, use_values: bool) -> str:
    """Write one measurement as a Convino [correlation matrix] input; returns the file name."""
    sig = np.sqrt(np.diag(C))
    corr = C / np.outer(sig, sig)
    order = list(range(npoi, len(names))) + list(range(npoi))     # nuisances first, estimates last
    out = [f"{names[i]}_{label}" if i < npoi else names[i] for i in range(len(names))]
    L = ["[correlation matrix]"]
    for a, oi in enumerate(order):
        L.append(f"  {out[oi]} ({sig[oi]:.16g}) "
                 + " ".join(f"{corr[oi, order[b]]:.16g}" for b in range(a + 1)))
    L += ["[end correlation matrix]", "[estimates]", f"  n_estimates = {npoi}"]
    for i in range(npoi):
        L += [f"  name_{i} = {out[i]}", f"  value_{i} = {vals[i]:.16g}"]
    L += ["[end estimates]", "[systematics]", "[end systematics]"]
    if use_values and np.any(vals[npoi:]):
        L += ["[nuisance values]"] + [f"  {out[i]} = {vals[i]:.16g}"
                                      for i in range(npoi, len(names))] + ["[end nuisance values]"]
    fn = f"{label}.txt"
    open(os.path.join(d, fn), "w").write("\n".join(L) + "\n")
    return fn


def run_pyconvino(workdir: str, meas: dict, use_values: bool):
    """meas: label -> (names, vals, C, npoi). Observables combine the same POI across files."""
    import contextlib
    import io as _io

    from pyconvino import Combiner
    os.makedirs(workdir, exist_ok=True)
    files, pois = [], {}
    for label, (names, vals, C, npoi) in meas.items():
        files.append(write_convino(workdir, label, names, vals, C, npoi, use_values))
        for i in range(npoi):
            pois.setdefault(names[i], []).append(f"{names[i]}_{label}")
    L = ["[inputs]", f"  nFiles = {len(files)}"] + [f"  file{i} = {f}" for i, f in enumerate(files)]
    L += ["[end inputs]", "[observables]"] + [f"  {p} = " + " + ".join(v) for p, v in pois.items()]
    L += ["[end observables]"]
    cfg = os.path.join(workdir, "config.txt")
    open(cfg, "w").write("\n".join(L) + "\n")
    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
        return Combiner.from_config(cfg, compute_impacts=False,
                                    use_nuisance_values=use_values).combine()


def example2(repo: str) -> None:
    base = os.path.join(repo, CFG, "mt_example")
    meas = {k: load_measurement(f"{base}/ATLAS{v}.yml") for k, v in (("boosted", "boosted"), ("jpsi", "jpsi"))}
    print("\n================ Example 2: ATLAS boosted + J/psi ================")
    for k, (names, vals, C) in meas.items():
        sig = np.sqrt(np.diag(C))
        print(f"  {k:8s} {len(names)-1:3d} systematics, mtop = {vals[0]:.6f} +- {sig[0]:.6f}, "
              f"min eig of the extended covariance {np.linalg.eigvalsh(C).min():.3e}")

    names = [meas[k][0] for k in meas]
    vals = [meas[k][1] for k in meas]
    covs = [meas[k][2] for k in meas]
    print(f"\n{'combination':22s} {'mtop':>11} {'+/-':>10}   weights (mtop row)")
    for label, with_np, only in (("POI only", False, None), ("POI and recoil NP", True, ["alpha_Recoil"])):
        v, cov, W, union = blue(names, vals, covs, [1, 1], with_np, only)
        print(f"{label:22s} {v[0]:11.6f} {np.sqrt(cov[0,0]):10.6f}   "
              f"{np.array2string(W[0], precision=6)}  on {union}")
        if with_np:
            print(f"{'':22s} recoil NP {v[1]:+.6f} +- {np.sqrt(cov[1,1]):.6f}")
    print(f"\n{'mode':24s} {'mtop':>11} {'+/-':>10}   recoil pull")
    md = {k: (*meas[k], 1) for k in meas}
    for use_values in (False, True):
        r = run_pyconvino("conv_ex2", md, use_values)
        k = r.sys_names.index("alpha_Recoil")
        mode = "--use-nuisance-values" if use_values else "default (C++ Convino)"
        print(f"{mode:24s} {r.combined_values[0]:11.6f} {r.combined_err_up[0]:10.6f}   "
              f"{r.pulls[k]:+.4f} +- {r.constraints[k]:.4f}")


def main(repo: str) -> None:
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    base = os.path.join(repo, CFG, "simple_example")
    meas = {k: load_measurement(f"{base}/measurement_{k}.yml") for k in ("A", "B", "A_SRonly")}

    print("Inputs read from the Combiner configs (uncertainties = sqrt of the covariance diagonal):")
    for k, (names, vals, C) in meas.items():
        print(f"  {k:9s} " + "  ".join(f"{n.replace('alpha_',''):12s} {v:+.6f} +- {s:.6f}"
                                       for n, v, s in zip(names, vals, np.sqrt(np.diag(C)))))

    combos = [("A & B, POI and NPs", ("A", "B"), True), ("A & B, POI only", ("A", "B"), False),
              ("A SR & B, POI and NPs", ("A_SRonly", "B"), True),
              ("A SR & B, POI only", ("A_SRonly", "B"), False)]
    print(f"\n{'combination':24s} {'sigma(POI)':>11} {'sigma(NP1)':>11} {'sigma(NP2)':>11}   weights (POI row)")
    for label, keys, with_np in combos:
        names = [meas[k][0] for k in keys]
        v, cov, W, union = blue(names, [meas[k][1] for k in keys], [meas[k][2] for k in keys],
                                [1, 1], with_np)
        s = np.sqrt(np.diag(cov))
        extra = f"{s[1]:11.6f} {s[2]:11.6f}" if with_np else f"{'-':>11} {'-':>11}"
        print(f"{label:24s} {s[0]:11.6f} {extra}   {np.array2string(W[0], precision=6)}")

    # pyconvino on the same inputs: the two files as Convino measurements
    os.makedirs("conv_ex1", exist_ok=True)
    print("\npyconvino on the same inputs (Convino chi2; observable = the POI):")
    print(f"{'setup':24s} {'mode':22s} {'POI':>10} {'+/-':>10}   pulls")
    for label, keys, _ in combos[:1] + combos[2:3]:
        for use_values in (False, True):
            fits = {k.replace("_SRonly", "SR"): (meas[k][1], meas[k][2]) for k in keys}
            mu, eu, ed, chi2, r = run_convino("conv_ex1", fits, use_values,
                                              tag="clo_" + label.replace(" ", "").replace(",", ""))
            mode = "--use-nuisance-values" if use_values else "default (C++ Convino)"
            print(f"{label:24s} {mode:22s} {mu:10.6f} {eu:10.6f}   {np.array2string(r.pulls, precision=4)}")


if __name__ == "__main__":
    main(sys.argv[1])
    example2(sys.argv[1])
