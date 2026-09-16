"""Toy of Dado/Owen/Pinamonti Example 1: SR + 2 CR, one POI mu, two NPs."""
import numpy as np
from scipy.optimize import minimize

# decoded from Fig. 1 (vector paths): s0, b0 per region [SR, CR1, CR2], abs. syst half-widths per region
ANALYSES = {
    "A": dict(s=np.array([650., 0., 0.]),  b=np.array([1000., 100., 100.]),
              d=np.array([[80., 6., 5.], [60., 5., 3.]])),
    "B": dict(s=np.array([200., 0., 0.]),  b=np.array([1500., 500., 600.]),
              d=np.array([[60., 15., 24.], [90., 35., 54.]])),
}

def expected(an, mu, th, regions):
    s, b, d = an["s"], an["b"], an["d"]
    nu = s * mu + b * (1.0 + d[0] / b * th[0] + d[1] / b * th[1])
    return nu[regions]

def nll(p, an, n, g, regions):
    nu = expected(an, p[0], p[1:], regions)
    return np.sum(nu - n * np.log(nu)) + 0.5 * np.sum((p[1:] - g) ** 2)

def hessian(p, an, n, g, regions):
    # exact Hessian of nll for the linear model: sum_r  n_r/nu_r^2 * grad_nu grad_nu^T  + prior
    s, b, d = an["s"], an["b"], an["d"]
    nu = expected(an, p[0], p[1:], regions)
    J = np.stack([s[regions], d[0][regions], d[1][regions]], axis=1)   # (nreg, 3)
    H = (J * (n / nu ** 2)[:, None]).T @ J
    H[1:, 1:] += np.eye(2)
    return H

def fit(an, n, g, regions=(0, 1, 2), start=(1.0, 0.0, 0.0)):
    regions = list(regions)
    res = minimize(nll, np.array(start), args=(an, n, g, regions), method="BFGS",
                   jac=lambda p, *a: _grad(p, *a), options={"gtol": 1e-10})
    p = res.x
    C = np.linalg.inv(hessian(p, an, n, g, regions))
    return p, C

def _grad(p, an, n, g, regions):
    s, b, d = an["s"], an["b"], an["d"]
    nu = expected(an, p[0], p[1:], regions)
    J = np.stack([s[regions], d[0][regions], d[1][regions]], axis=1)
    gr = J.T @ (1 - n / nu)
    gr[1:] += p[1:] - g
    return gr

def asimov(an, regions=(0, 1, 2)):
    regions = list(regions)
    return expected(an, 1.0, np.zeros(2), regions)

# ---------------- BLUE (paper Eq. 2.3) ----------------
def gamma(C):
    """Gamma_ik = cov(parameter i, NP k): impact of NP k on estimate i (unit prior)."""
    return C[:, 1:]

def blue(ests, covs, use_np=True):
    """ests/covs: per-analysis (3,) and (3,3) for (mu, th1, th2). Returns combined values, cov, weights."""
    sel = slice(None) if use_np else slice(0, 1)
    npar = 3 if use_np else 1
    blocks = []
    for i, Ci in enumerate(covs):
        row = []
        for j, Cj in enumerate(covs):
            row.append(Ci[sel, sel] if i == j else (gamma(Ci) @ gamma(Cj).T)[sel, sel])
        blocks.append(row)
    CAB = np.block(blocks)
    U = np.vstack([np.eye(npar)] * len(covs))
    y = np.concatenate([e[sel] for e in ests])
    Ci = np.linalg.inv(CAB)
    W = np.linalg.solve(U.T @ Ci @ U, U.T @ Ci)      # (npar, ntot) BLUE weights
    return W @ y, np.linalg.inv(U.T @ Ci @ U), W
