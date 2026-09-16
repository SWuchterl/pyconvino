import os, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toymodel import *
from convino_io import run_convino

NTOY = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NPROC = 6
METHODS = ["blue_poi", "blue_np", "conv_nopulls", "conv_pulls", "sr_blue_np", "sr_conv_nopulls", "sr_conv_pulls"]

def one_chunk(args):
    seeds, wd = args
    os.makedirs(wd, exist_ok=True)
    rng = np.random.default_rng(seeds[0] + 12345)
    A, B = ANALYSES["A"], ANALYSES["B"]
    rows = []
    for s in seeds:
        g = rng.normal(size=2)                                   # common global observables
        nA = rng.poisson(asimov(A)); nB = rng.poisson(asimov(B))
        fA = fit(A, nA, g); fB = fit(B, nB, g); fS = fit(A, nA[:1], g, regions=[0])
        # report NP estimates relative to the prior centre g (the frame in which
        # a real input's pulls are quoted and the Convino global prior sits at 0)
        for f in (fA, fB, fS): f[0][1:] -= g
        r = {}
        for lab, (v, cov, W) in [("blue_poi", blue([fA[0], fB[0]], [fA[1], fB[1]], False)),
                                 ("blue_np", blue([fA[0], fB[0]], [fA[1], fB[1]], True)),
                                 ("sr_blue_np", blue([fS[0], fB[0]], [fS[1], fB[1]], True))]:
            r[lab] = (v[0], np.sqrt(cov[0, 0]))
        for lab, fits, pulls in [("conv_nopulls", {"A": fA, "B": fB}, False), ("conv_pulls", {"A": fA, "B": fB}, True),
                                 ("sr_conv_nopulls", {"A": fS, "B": fB}, False), ("sr_conv_pulls", {"A": fS, "B": fB}, True)]:
            mu, eu, ed, chi2, res = run_convino(wd, fits, pulls, tag=lab)
            r[lab] = (mu, 0.5 * (eu + ed))
        rows.append([s] + [x for m in METHODS for x in r[m]] + [fA[0][0], fB[0][0], fA[0][1], fA[0][2], fB[0][1], fB[0][2], g[0], g[1]])
    return np.array(rows)

if __name__ == "__main__":
    seeds = np.arange(NTOY)
    chunks = [(list(c), f"work{i}") for i, c in enumerate(np.array_split(seeds, NPROC))]
    with Pool(NPROC) as pool:
        parts = pool.map(one_chunk, chunks)
    data = np.vstack(parts)
    np.savez("toys.npz", data=data, methods=METHODS)
    print("done", data.shape)
