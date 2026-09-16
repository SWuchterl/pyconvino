"""Write the toy fits as Convino measurement files + config and run pyconvino in-process."""
import os, io, contextlib, numpy as np
from pyconvino.parser import parse_config_file, parse_measurement_file
from pyconvino import Combiner

def write_measurement(path, label, p, C, pulls=None):
    names = ["Syst1", "Syst2", f"mu_{label}"]
    sig = np.sqrt(np.diag(C)); corr = C / np.outer(sig, sig)
    order = [1, 2, 0]                      # NPs first, then the estimate (Convino convention)
    L = ["[correlation matrix]"]
    for i, oi in enumerate(order):
        row = "  ".join(f"{corr[oi, order[j]]:.12g}" for j in range(i + 1))
        L.append(f"  {names[i]:8s} ({sig[oi]:.12g})  {row}")
    L += ["[end correlation matrix]", "", "[estimates]", "  n_estimates = 1",
          f"  name_0 = mu_{label}", f"  value_0 = {p[0]:.12g}", "[end estimates]", "",
          "[systematics]", "[end systematics]"]
    if pulls is not None:
        L += ["", "[nuisance values]", f"  Syst1 = {p[1]:.12g}", f"  Syst2 = {p[2]:.12g}", "[end nuisance values]"]
    open(path, "w").write("\n".join(L) + "\n")

def write_config(path, files):
    L = ["[global]", "[end global]", "", "[inputs]", f"  nFiles = {len(files)}"]
    L += [f"  file{i} = {os.path.basename(f)}" for i, f in enumerate(files)]
    L += ["[end inputs]", "", "[observables]", "  mu = " + " + ".join(f"mu_{os.path.basename(f).split('_')[0]}" for f in files),
          "[end observables]", "", "[correlations]", "[end correlations]"]
    open(path, "w").write("\n".join(L) + "\n")

def run_convino(workdir, fits, pulls, tag="x"):
    """fits: dict label -> (p, C). Returns (mu, sigma_up, sigma_down, chi2min, result)."""
    files = []
    for label, (p, C) in fits.items():
        f = os.path.join(workdir, f"{label}_{tag}.txt"); write_measurement(f, label, p, C, p if pulls else None); files.append(f)
    cfg = os.path.join(workdir, f"config_{tag}.txt"); write_config(cfg, files)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        c = Combiner.from_config(cfg, compute_impacts=False, use_nuisance_values=pulls)
        r = c.combine()
    return float(r.combined_values[0]), float(r.combined_err_up[0]), float(r.combined_err_down[0]), float(r.chi2_min), r
