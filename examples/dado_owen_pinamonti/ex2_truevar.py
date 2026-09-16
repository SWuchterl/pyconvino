"""True variance of the out-of-the-box Convino estimator for Example 2 (linear in d_B, d_J; ignores lambda_hat)."""
import io, contextlib, numpy as np, ex2
from ex2 import *
from pyconvino import Combiner
def run2(dB, dJ, pulls):
    ex2.mt_B = dB; ex2.mt_J = dJ
    ex2.write_inputs("tv", pulls)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r = Combiner.from_config("tv/config.txt", compute_impacts=False, use_nuisance_values=pulls).combine()
    return float(r.combined_values[0]), float(r.combined_err_up[0])
sB, sJ = np.sqrt(C[i_mt, i_mt]), np.sqrt(cov_J_mt[0, 0]); Sigma = np.array([[sB**2, rho*sB*sJ], [rho*sB*sJ, sJ**2]])
for pulls in (False, True):
    m0, e0 = run2(172.95, 172.17, pulls); wB = run2(173.95, 172.17, pulls)[0] - m0; wJ = run2(172.95, 173.17, pulls)[0] - m0
    w = np.array([wB, wJ]); print(f"pulls={pulls!s:5}: weights on (mt_B, mt_J) = {w}  sum {w.sum():.4f}; reported sigma {e0:.3f}; "
          f"true sigma of w.d (POI part only) {np.sqrt(w @ Sigma @ w):.3f}")
print("BLUE POI-only optimal sigma 0.507 for reference; the pulls estimator also carries lambda_hat terms, whose variance is included in the exact 0.493.")
