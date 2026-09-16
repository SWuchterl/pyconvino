"""Example 2 of Dado/Owen/Pinamonti: ATLAS boosted (profiled, HEPData ins2894561) + J/psi (stat-only, HEPData 167264)."""
import re, os, io, contextlib, numpy as np
from hep import rows
np.set_printoptions(precision=4, suppress=True, linewidth=150)

# ---------------- boosted input ----------------
z = np.load("boosted_cov.npz"); bn = list(z["names"]); C = z["C"]
_, npt = rows("hepdata/boosted__Post-fit_values_and_uncertaintes_for_nuisance_parameters.json"); bval = {n: float(a) for n, a, c in npt}
_, pre = rows("hepdata/boosted__Pre-fit_MC_statistical_uncertainties.json"); presig = {n: float(v) for n, v in pre}
i_mu = bn.index("$\\mu$"); keep = [i for i in range(len(bn)) if i != i_mu]      # marginalise mu
bn = [bn[i] for i in keep]; C = C[np.ix_(keep, keep)]
# gammas: prior N(1, sigma_pre) -> unit-prior NP  lam = (gamma-1)/sigma_pre
scale = np.ones(len(bn))
for i, n in enumerate(bn):
    m = re.match(r"\$m_(W|\{tj\})\$ bin (\d+) stat\.", n)
    if m:
        key = ("$m_{jj}$" if m.group(1) == "W" else "$m_{tj}$") + f" bin {m.group(2)}"
        scale[i] = 1.0 / presig[key]; bval[n] = (bval[n] - 1.0) * scale[i]
C = C * scale[:, None] * scale[None, :]
i_mt = bn.index("$m_t$"); mt_B = 172.95

# ---------------- jpsi input ----------------
_, js = rows("hepdata/jpsi__All_Systematic_Uncertainties.json")
jn = [r[1] for r in js]; # HEPData quotes the shift of m_t when the +1 sigma model replaces the nominal (data variation);
# the profiled covariance uses d m_t / d g (template variation) -> opposite sign.
jimp = -np.array([float(r[2]) for r in js]); mt_J = 172.17; stat_J = 0.80

# ---------------- correlation model (paper Sec. 5): shared names ----------------
M = {"Recoil": "Recoil", "$t\\bar{t}$ PS and had.": "Parton shower and hadronisation, $\\overline{m_J}$",
     "$t\\bar{t}$ $h_{damp}$": "hdamp setting", "$t\\bar{t}$ ME": "Matrix element matching",
     "$t\\bar{t}$ $\\alpha_{s}^{FSR}$": "Final-state radiation", "$t\\bar{t}$ $\\alpha_{s}^{ISR}$": "var3c eigentune variation",
     "$t\\bar{t}$ $\\mu_{r}$": "Renormalisation scale", "$t\\bar{t}$ $\\mu_{f}$": "Factorisation scale",
     "$t\\bar{t}$ UE": "Underlying event", "$t\\bar{t}$ CR": "CR1", "$t\\bar{t}$ NNLO rew.": "NNLO re-weighting",
     "single-top ME": "Single-top matrix element matching", "single-top PS and had.": "Single-top parton shower and hadronisation",
     "single-top $h_{damp}$": "Single-top hdamp setting", "single-top $\\alpha_{s}^{ISR}$": "Single-top var3c eigentune variation",
     "single-top $\\alpha_{s}^{FSR}$": "Single-top FSR", "single-top $\\mu_{r}$": "Single-top renormalisation scale setting",
     "single-top $\\mu_{f}$": "Single-top factorisation scale setting",
     "$t\\bar{t}$-single-top interference": "Single-top ttbar interference removal",
     "Luminosity": "Luminosity", "Pile-up": "Pileup re-weighting",
     "JES $\\eta$ intercalibration total stat.": "JES $\\eta$ intercalibration statistics",
     "JES $\\eta$ intercalibration modelling": "JES $\\eta$-intercalibration modelling",
     "JES $\\eta$ intercalibration non-closure 2018 data": "JES $\\eta$-intercalibration 2018 data",
     "JES $\\eta$ intercalibration non-closure pos-$\\eta$": "JES $\\eta$-intercalibration positive $\\eta$",
     "JES $\\eta$ intercalibration non-closure neg-$\\eta$": "JES $\\eta$-intercalibration negative $\\eta$",
     "JES pile-up, offset $\\mu$": "JES pileup $\\mu$", "JES pile-up, offset NPV": "JES pileup $N_{\\mathrm{PV}}$",
     "JES pile-up, $p_T$ term": "JES pileup $p_T$-dependence", "JES pile-up, $\\rho$ topology": "JES pileup $\\rho$-topology",
     "JES flavour composition": "JES flavour composition (prop.)", "JES flavour response": "JES flavour response (prop.)",
     "JES punch through": "JES punch through", "JER data vs MC": "JER data-MC agreement", "Jet vertex fraction": "JVT",
     "$b$-tag extrap. high-$p_T$": "b-tagging high-$p_T$ extrapolation", "$b$-tag extrap. $c\\rightarrow\\tau$": "b-tagging high-$p_T$ charm extrapolation",
     "MET soft track resolution (para.)": "$E_T^{\\text{miss}}$ soft track ResoPara", "MET soft track resolution (perp.)": "$E_T^{\\text{miss}}$ soft track ResoPerp",
     "Electron energy scale": "Egamma scale", "Electron energy resolution": "Egamma resolution", "Electron trigger eff.": "Electron trigger SF",
     "Electron reconstruction eff.": "Electron reconstruction", "Electron identification eff.": "Electron identification", "Electron isolation eff.": "Electron isolation",
     "Muon momentum scale": "Muon energy scale", "Muon momentum resolution (CB)": "Muon CB", "Muon sagitta data": "Muon sagitta data statistics",
     "Muon sagitta resolution (bias)": "Muon sagitta resolution bias", "Muon trigger eff. (stat)": "Muon trigger SF (stat.)", "Muon trigger eff. (syst)": "Muon trigger SF (syst.)",
     "Muon identification eff. (stat)": "Muon identification (stat.)", "Muon identification eff. (syst)": "Muon identification (syst.)",
     "Muon isolation eff. (stat)": "Muon isolation (stat.)", "Muon isolation eff. (syst)": "Muon isolation (syst.)",
     "Muon track-to-vertex association eff. (syst)": "Muon TTVA SF (syst.)"}
for i in range(1, 7): M[f"JES statistical {i}"] = f"JES statistical {i}"
for i in range(1, 3): M[f"JES detector {i}"] = f"JES detector {i}"
for i in range(1, 4): M[f"JES mixed {i}"] = f"JES mixed {i}"
for i in range(1, 5): M[f"JES modelling {i}"] = f"JES modelling {i}"
for i in range(1, 13): M[f"JER effectiveNP {i}" + ("Resterm" if i == 12 else "")] = f"JER {i}"
for i in range(1, 7): M[f"$b$-tag b {i}"] = f"b-tagging B {i-1}"
for i in range(1, 4): M[f"$b$-tag c {i}"] = f"b-tagging C {i-1}"
for i in range(1, 4): M[f"$b$-tag light {i}"] = f"b-tagging Light {i-1}"
for i in range(1, 31): M[f"$t\\bar{{t}}$ PDF {i}"] = f"PDF4LHC {i}"
bset = set(bn); jset = set(jn)
missing = [(j, b) for j, b in M.items() if j not in jset or b not in bset]
M = {j: b for j, b in M.items() if j in jset and b in bset}
def clean(s): return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]", "_", s)).strip("_")
shared = {}  # convino name for every parameter
for j, b in M.items(): shared[("J", j)] = shared[("B", b)] = "S_" + clean(b)
for b in bn:
    if ("B", b) not in shared: shared[("B", b)] = ("mt_boosted" if b == "$m_t$" else "B_" + clean(b))
for j in jn:
    if ("J", j) not in shared: shared[("J", j)] = "J_" + clean(j)
assert len(set(shared[("B", b)] for b in bn)) == len(bn) and len(set(shared[("J", j)] for j in jn)) == len(jn)

def write_inputs(d, pulls):
    os.makedirs(d, exist_ok=True)
    sig = np.sqrt(np.diag(C)); corr = C / np.outer(sig, sig)
    order = [i for i in range(len(bn)) if i != i_mt] + [i_mt]
    L = ["[correlation matrix]"]
    for a, oi in enumerate(order):
        L.append(f"  {shared[('B', bn[oi])]} ({sig[oi]:.10g}) " + " ".join(f"{corr[oi, order[b]]:.10g}" for b in range(a + 1)))
    L += ["[end correlation matrix]", "[estimates]", "  n_estimates = 1", "  name_0 = mt_boosted", f"  value_0 = {mt_B}", "[end estimates]", "[systematics]", "[end systematics]"]
    if pulls:
        L += ["[nuisance values]"] + [f"  {shared[('B', b)]} = {bval[b]:.10g}" for b in bn if b != "$m_t$"] + ["[end nuisance values]"]
    open(f"{d}/boosted.txt", "w").write("\n".join(L) + "\n")
    L = ["[not fitted]", "  " + " ".join(shared[("J", j)] for j in jn) + " stat",
         "  mt_jpsi " + " ".join(f"{v:.6g}" for v in jimp) + f" {stat_J}", "[end not fitted]",
         "[estimates]", "  n_estimates = 1", "  name_0 = mt_jpsi", f"  value_0 = {mt_J}", "[end estimates]", "[systematics]", "[end systematics]"]
    open(f"{d}/jpsi.txt", "w").write("\n".join(L) + "\n")
    open(f"{d}/config.txt", "w").write("[global]\n[end global]\n[inputs]\n  nFiles = 2\n  file0 = boosted.txt\n  file1 = jpsi.txt\n[end inputs]\n"
                                       "[observables]\n  mt = mt_boosted + mt_jpsi\n[end observables]\n[correlations]\n[end correlations]\n")

# ---------------- BLUE (paper Eq. 2.3) on a common NP axis ----------------
allnp = sorted(set(shared[("B", b)] for b in bn if b != "$m_t$") | set(shared[("J", j)] for j in jn))
col = {n: k for k, n in enumerate(allnp)}
def gamma_B(rowidx):
    G = np.zeros((len(rowidx), len(allnp)))
    for r, i in enumerate(rowidx):
        for k, b in enumerate(bn):
            if b != "$m_t$": G[r, col[shared[("B", b)]]] = C[i, k]
    return G
def blue_general(ests, covs, gammas):
    n = [len(e) for e in ests]; blocks = [[covs[i] if i == j else gammas[i] @ gammas[j].T for j in range(2)] for i in range(2)]
    CAB = np.block(blocks); U = np.vstack([np.eye(n[0])] + [np.eye(n[0])[:n[1]]] if False else [np.eye(n[0]), np.eye(n[0])[:, :][:n[1]]])
    Ci = np.linalg.inv(CAB); W = np.linalg.solve(U.T @ Ci @ U, U.T @ Ci)
    return W @ np.concatenate(ests), np.linalg.inv(U.T @ Ci @ U), W, CAB

GJ_mt = np.zeros((1, len(allnp)))
for j, v in zip(jn, jimp): GJ_mt[0, col[shared[("J", j)]]] = v
cov_J_mt = np.array([[stat_J ** 2 + (jimp ** 2).sum()]])
print(f"J/psi total sigma {np.sqrt(cov_J_mt[0,0]):.3f}; boosted sigma {np.sqrt(C[i_mt,i_mt]):.3f}")
print("unmapped pairs (not in tables):", missing)
rho = (gamma_B([i_mt]) @ GJ_mt.T)[0, 0] / np.sqrt(C[i_mt, i_mt] * cov_J_mt[0, 0]); print(f"correlation mt_boosted vs mt_jpsi: {rho:.3f}   (paper: 0.01)")

if __name__ == "__main__":
    # POI only
    v, cov, W, _ = blue_general([np.array([mt_B]), np.array([mt_J])], [C[np.ix_([i_mt], [i_mt])], cov_J_mt], [gamma_B([i_mt]), GJ_mt])
    print(f"BLUE POI only        : mt = {v[0]:.3f} +- {np.sqrt(cov[0,0]):.3f}   weights {W[0]}   (paper 0.50; 0.90/0.10)")
    # POI + recoil NP: boosted (mt, recoil) ; jpsi (mt, recoil prior 0 +- 1)
    i_rec = bn.index("Recoil"); GJ2 = np.vstack([GJ_mt, np.eye(len(allnp))[col["S_Recoil"]]])
    covJ2 = np.diag([cov_J_mt[0, 0], 1.0]); covJ2[0, 1] = covJ2[1, 0] = jimp[jn.index("Recoil")]   # cov(mt_J, recoil prior) = Gamma_recoil
    v, cov, W, _ = blue_general([np.array([mt_B, bval["Recoil"]]), np.array([mt_J, 0.0])], [C[np.ix_([i_mt, i_rec], [i_mt, i_rec])], covJ2], [gamma_B([i_mt, i_rec]), GJ2])
    print(f"BLUE POI + recoil NP : mt = {v[0]:.3f} +- {np.sqrt(cov[0,0]):.3f}   weights(mt row: mtB, recB, mtJ, recJ) {W[0]}   (paper 0.48; 0.84/-0.17/0.16/0.17)")
    # POI + all shared NPs (jpsi NPs: prior 0 +- 1 each)
    S = [shared[("B", b)] for b in bn if shared[("B", b)].startswith("S_")]; iB = [i_mt] + [bn.index(b) for b in bn if shared[("B", b)] in S]
    GJn = np.vstack([GJ_mt] + [np.eye(len(allnp))[col[n]] for n in S]); covJn = np.eye(len(S) + 1); covJn[0, 0] = cov_J_mt[0, 0]
    for a, n in enumerate(S): covJn[0, a + 1] = covJn[a + 1, 0] = GJ_mt[0, col[n]]
    estB = np.array([mt_B] + [bval[b] for b in bn if shared[("B", b)] in S])
    estJ = np.concatenate([[mt_J], np.zeros(len(S))])
    v, cov, W, _ = blue_general([estB, estJ], [C[np.ix_(iB, iB)], covJn], [gamma_B(iB), GJn])
    print(f"BLUE POI + all {len(S)} shared NPs: mt = {v[0]:.3f} +- {np.sqrt(cov[0,0]):.3f}   (recoil weights B/J {W[0, 1+S.index('S_Recoil')]:+.3f}/{W[0, len(S)+2+S.index('S_Recoil')]:+.3f})")
    # Convino
    from pyconvino import Combiner
    for pulls in (False, True):
        d = "conv_pulls" if pulls else "conv_nopulls"; write_inputs(d, pulls)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = Combiner.from_config(
                f"{d}/config.txt", compute_impacts=False, use_nuisance_values=pulls
            ).combine()
        k = r.sys_names.index("S_Recoil"); kp = r.sys_names.index("S_Parton_shower_and_hadronisation_overline_m_J")
        print(f"pyconvino pulls={pulls!s:5}: mt = {r.combined_values[0]:.3f} +{r.combined_err_up[0]:.3f} -{r.combined_err_down[0]:.3f}  chi2min {r.chi2_min:.2f} ndf {r.ndf}  "
              f"Recoil pull {r.pulls[k]:+.3f} constr {r.constraints[k]:.3f}  PS(mJ) pull {r.pulls[kp]:+.3f} constr {r.constraints[kp]:.3f}")
