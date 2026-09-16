# Mathematical extensions of pyconvino relative to the C++ Convino reference

This document specifies, in a self-contained and citable form, the mathematics
that `pyconvino` adds to or reformulates relative to the original C++
Convino tool [1]. It is intended as a baseline for internal documentation and
for the methods section of a paper: every quantity is defined, every extension
is derived from the shared base model, and each item is classified as one of

- **New** — a quantity the reference tool does not compute at all;
- **Reformulation** — the same physical result obtained by a different (usually
  closed-form) route, validated to agree with the reference;
- **Port** — an existing reference algorithm reimplemented, with any deliberate
  behavioural divergence called out explicitly.

The base combination method (the χ² with simultaneously constrained
uncertainties, the split of each input covariance into a measurement-constraining
and a nuisance-constraining part, the Neyman/Pearson choice, and the
profile-likelihood errors) is *not* reproduced here — it is defined in the
Convino paper [1], and `pyconvino` reproduces it to the sub-per-mille level on
the paper's own worked example and on the ATLAS+CMS top-quark cross-section
combinations (see `docs/paper_benchmark_crosscheck.md`). This document covers
only what is different.

---

## 1. Notation and the shared χ²

The global parameter vector is

$$
\theta \;=\; (\lambda,\, x), \qquad
\lambda \in \mathbb{R}^{n_\mathrm{sys}}, \quad
x \in \mathbb{R}^{n_\mathrm{est}} ,
$$

with the `nsys` nuisance pulls $\lambda_i$ ordered first and the `nest`
combined-observable values $x_b$ last. For each input measurement $k$ with
local estimates $x^{(k)}_\mathrm{meas}$, define the shifted prediction
$\tilde{x}^{(k)}(x,\lambda)$ by mapping the relevant combined values into the
measurement's estimate space and applying its systematic shifts. Per systematic
$i$ the shift magnitude is the piecewise-linear ("kinked") response

$$
k_i(\lambda_i) \;=\;
\begin{cases}
L^{\uparrow}_i\,\lambda_i, & \lambda_i \ge 0,\\[2pt]
L^{\downarrow}_i\,|\lambda_i|, & \lambda_i < 0,
\end{cases}
$$

applied additively for `absolute` systematics
($\tilde{x} \mathrel{-}= \sum_i k_i$) and multiplicatively for `relative`
ones ($\tilde{x} \mathrel{*}= \prod_i (1 - k_i/x_\mathrm{meas})$). With the
residual $r^{(k)} = x^{(k)}_\mathrm{meas} - \tilde{x}^{(k)}$, the objective is

$$
\chi^2(\lambda, x) \;=\;
\sum_k \Big[\, r^{(k)\mathsf{T}} M^{(k)}\, r^{(k)}
          \;+\; \lambda^{(k)\mathsf{T}} D^{(k)} \lambda^{(k)} \,\Big]
\;+\; \lambda^{\mathsf{T}} C^{-1} \lambda ,
\tag{1}
$$

where $M^{(k)}$ and $D^{(k)}$ are the measurement- and nuisance-constraining
blocks obtained from the input Hessian/covariance of measurement $k$ (the split
of [1]), and $C$ is the $n_\mathrm{sys}\times n_\mathrm{sys}$ **prior
correlation matrix** of the nuisances across measurements ($C_{ii}=1$,
$C_{ij}=\rho_{ij}$ from the `[correlations]` block). The Pearson option
replaces $M^{(k)}$ by
$M^{(k)}_\mathrm{eff} = M^{(k)}\odot \sqrt{s\,s^{\mathsf{T}}}$ with
$s_b = |x^{(k)}_{\mathrm{meas},b}/\tilde{x}^{(k)}_b|$; the default is Neyman
($M_\mathrm{eff}=M$). Equation (1) is the object `pyconvino` minimises, and it
is the C++ reference model.

Throughout, $\hat\theta$ denotes the best fit, $H$ the Hessian of $\chi^2$ at
$\hat\theta$, and the post-fit covariance uses the Minuit UP $=1$ convention for
a χ² (as opposed to a $-2\ln L$):

$$
\operatorname{cov}(\theta) \;=\; 2\,H^{-1},
\qquad \sigma_{\theta_j} = \sqrt{2\,(H^{-1})_{jj}} .
\tag{2}
$$

The factor of $2$ follows from $\Delta\chi^2 = 1$ defining the $1\sigma$
interval together with the quadratic expansion
$\chi^2 = \chi^2_\mathrm{min} + \tfrac12\,\delta\theta^{\mathsf{T}} H\,\delta\theta$.
We partition $H$ into nuisance ($\lambda$) and observable ($x$) blocks:

$$
H \;=\;
\begin{pmatrix} H_{\lambda\lambda} & H_{\lambda x}\\[2pt]
               H_{x\lambda} & H_{xx} \end{pmatrix},
\qquad H_{x\lambda} = H_{\lambda x}^{\mathsf{T}} .
\tag{3}
$$

---

## 2. Exact quadratic minimisation, and HESSE ≡ MINOS  *(reformulation)*

**When it applies.** Whenever (i) every systematic response is symmetric,
$L^{\uparrow}_i = -L^{\downarrow}_i$ (no kink at $\lambda_i = 0$), (ii) no
systematic is `relative` (all shifts additive), and (iii) Pearson rescaling is
off, the objective (1) is an *exact* quadratic form in $\theta$ with a
**constant** Hessian $H$. The gradient is then exactly linear,

$$
\nabla\chi^2(\theta) \;=\; H\,(\theta - \hat\theta)
\quad\text{for all } \theta,
$$

so the minimum is reached from any starting point $\theta_0$ in a single linear
solve,

$$
\hat\theta \;=\; \theta_0 - H^{-1}\,\nabla\chi^2(\theta_0).
\tag{4}
$$

The reference tool always runs an iterative Minuit minimisation; `pyconvino`
detects the quadratic case and uses (4), which is exact to floating point in one
step (the general non-quadratic case — relative systematics, Pearson, or
asymmetric responses — still uses an iterative L-BFGS-B fit with a guarded
Newton polish). This is a numerical reformulation: the result is identical, and
is validated against the reference.

**HESSE ≡ MINOS in the quadratic case.** For a quadratic χ² the profile-
likelihood (MINOS) interval of any parameter equals its parabolic (HESSE)
interval,

$$
\sigma^{\mathrm{MINOS}}_{\theta_j} \;=\; \sigma^{\mathrm{HESSE}}_{\theta_j}
\;=\; \sqrt{2\,(H^{-1})_{jj}}
\quad\text{(symmetric responses).}
$$

The reference tool configures every combined observable as a MINOS parameter
unconditionally. `pyconvino` uses the closed-form HESSE value (2) whenever the
responses are symmetric — provably the same number, and far more robust than
profiling a fit with several hundred nuisances — and falls back to an explicit
profile-likelihood scan,
$\{\,v : \min_{\theta\setminus\theta_j}\chi^2\big|_{\theta_j=v} - \chi^2_\mathrm{min} = 1\,\}$,
only when at least one response is genuinely asymmetric, where the two differ.

---

## 3. Positive-definite regularisation of the nuisance prior  *(new)*

The user-specified correlations $\rho_{ij}$ need not form a positive-definite
(PD) matrix — e.g. a cycle $\rho_{AB}=\rho_{BC}=0.9,\ \rho_{AC}=-0.9$ is
inconsistent. Since (1) needs $C^{-1}$, a non-PD $C$ must be regularised.
`pyconvino` exposes three methods, always emits a `UserWarning` reporting the
number and size of the negative eigenvalues and the induced off-diagonal change,
and reports the *exact* user matrix separately for display. The reference tool
does not expose the choice of regularisation as a user-visible, quantified step.

Let $C = C^{\mathsf{T}}$ have eigen-decomposition $C = Q\Lambda Q^{\mathsf{T}}$
with sorted eigenvalues $\lambda_1 \le \dots \le \lambda_n$, and let
$\varepsilon = 10^{-8}\max(1,\max_j|\lambda_j|)$ be the PD floor. Regularisation
is triggered only when $\lambda_1 < -10^{-10}\max(1,\max_j|\lambda_j|)$ (two
orders of magnitude below the floor, so floating-point noise around $0$ does not
trigger it).

**`shift` (default).** Add the smallest isotropic shift that lifts the spectrum
to the floor, then renormalise to unit diagonal:

$$
\delta = -\lambda_1 + \varepsilon,\qquad
\hat C = \frac{C + \delta I}{1+\delta},\qquad
\hat C_{ij} = \frac{\rho_{ij}}{1+\delta}\ (i\ne j).
$$

Every off-diagonal is scaled by the **same** factor $1/(1+\delta)$, giving the
interpretation "all correlations damped by a common factor because they were
mutually inconsistent."

**`clip`.** Reflect the sub-floor eigenvalues to $\varepsilon$ and renormalise:

$$
\hat C_0 = Q\,\mathrm{diag}\!\big(\max(\lambda_j,\varepsilon)\big)\,Q^{\mathsf{T}},
\qquad
\hat C_{ij} = \frac{(\hat C_0)_{ij}}{\sqrt{(\hat C_0)_{ii}(\hat C_0)_{jj}}} .
$$

The change is concentrated in the offending eigenvector directions and is zero
for uncorrelated pairs.

**`higham`.** The nearest correlation matrix in Frobenius norm, via Higham's
(2002) alternating projections [2] between the PSD cone (with eigenvalue floor
$\varepsilon$, Dykstra-corrected) and the unit-diagonal hyperplane:

$$
\hat C = \arg\min_{\substack{X \succeq 0\\ X_{ii}=1}} \lVert X - C\rVert_F .
$$

This minimises the total squared perturbation at the cost of an iterative solve.

**Invariance property.** For the standard quadratic case (all-absolute,
symmetric, Neyman) the regularisation method changes *only the nuisance pulls*,
not the combined values or their uncertainties: from (1), $x$ is determined by
the data blocks $M^{(k)}$, while $C$ enters only the prior penalty that governs
how far $\lambda$ is pulled. `shift` is the default for its single-factor
interpretability; `higham` is available when a minimum-perturbation argument is
required.

---

## 4. Goodness of fit: ndf, χ²/ndf, p-value  *(new)*

The reference tool reports only $\chi^2_\mathrm{min}$. `pyconvino` adds a
goodness-of-fit summary. The degrees of freedom use the convention

$$
\mathrm{ndf} \;=\; N_\mathrm{meas} - n_\mathrm{est},
\tag{5}
$$

where $N_\mathrm{meas} = \sum_k \dim x^{(k)}_\mathrm{meas}$ is the total number
of individual input estimate values across all measurements, and
$n_\mathrm{est}$ is the number of combined observables. The nuisance parameters
contribute **net zero** to the ndf: each carries a unit-Gaussian prior term in
(1) that adds one expected unit to $\chi^2_\mathrm{min}$ while also adding one
fitted parameter, so the two cancel — the standard treatment of Gaussian-
constrained nuisances. When $\mathrm{ndf} > 0$,

$$
\chi^2/\mathrm{ndf} = \frac{\chi^2_\mathrm{min}}{\mathrm{ndf}},
\qquad
p = \int_{\chi^2_\mathrm{min}}^{\infty} f_{\chi^2}(t;\mathrm{ndf})\,\mathrm{d}t
  = \mathrm{sf}_{\chi^2}\!\big(\chi^2_\mathrm{min};\mathrm{ndf}\big);
$$

both are `nan` when $\mathrm{ndf} \le 0$ (e.g. a combination in which each
observable has exactly one contributing measurement, as in the current
ATLAS+CMS fixtures).

---

## 5. Frozen-Hessian impact and covariance decomposition  *(reformulation)*

The impact of a group $g$ of systematics is defined, as in the reference, by the
quadrature difference between the full and the "group-frozen" errors,

$$
\mathrm{impact}_g \;=\; \sqrt{\,\sigma_\mathrm{full}^2 - \sigma_{\mathrm{frozen}(g)}^2\,},
\tag{6}
$$

where $\sigma_{\mathrm{frozen}(g)}$ is the combined-observable error obtained
with the members of $g$ pinned to $\lambda = 0$ and all other parameters
re-optimised. The reference computes $\sigma_{\mathrm{frozen}(g)}$ by an explicit
re-fit per group.

**Reformulation.** For the quadratic χ² of Section 2 the Hessian is constant, so
the group-frozen minimum lies on the *same* paraboloid as $\hat\theta$; pinning a
set $F$ of parameters simply deletes their rows and columns. The frozen
covariance is therefore the corresponding submatrix inverse of the *already
computed* full Hessian, with no re-fit and no re-JIT:

$$
\operatorname{cov}_{\mathrm{frozen}(F)} \;=\; 2\,\big(H_{\bar F\bar F}\big)^{-1},
\qquad \bar F = \text{free indices},
\tag{7}
$$

and $\sigma_{\mathrm{frozen}(g)}$ is the square root of the observable-block
diagonal of (7). This is exact for the symmetric/quadratic case (validated to
match the reference's per-mille-level impact numbers); only genuinely asymmetric
responses retain the explicit re-fit plus profile scan, with (7) supplying the
bracket seed. The per-group covariance printed to output uses the reference's
own construction from the full-fit correlation $\varrho$ scaled by the frozen
errors, $\operatorname{cov}_{ij}=\varrho_{ij}\,\sigma_i\sigma_j$ with
$\sigma = \max(|\sigma^\uparrow|,|\sigma^\downarrow|)$.

---

## 6. Automatic statistical / systematic split  *(new)*

Applying (6)–(7) with the frozen set $F$ equal to **all** systematics
($\bar F = x$) gives the two quantities the reference does not report:

$$
\operatorname{cov}_\mathrm{stat} \;=\; 2\,H_{xx}^{-1}
\quad(\text{pure statistical / measurement-only covariance}),
\tag{8}
$$

$$
\operatorname{cov}_\mathrm{syst} \;=\; \operatorname{cov}_\mathrm{comb}
   - \operatorname{cov}_\mathrm{stat},
\qquad
\sigma^\mathrm{syst}_b = \sqrt{(\sigma^\mathrm{comb}_b)^2 - (\sigma^\mathrm{stat}_b)^2},
\tag{9}
$$

where $\operatorname{cov}_\mathrm{comb} = 2\,(H^{-1})_{xx}$ is the full post-fit
covariance of the combined observables (stat $\oplus$ all systematics profiled).
This is computed at negligible cost from the same submatrix machinery as any
impact group and is emitted automatically for every combination (no manual
"freeze everything" group needed). By construction the per-group/per-systematic
impacts (6) do **not** sum in quadrature to (9) when the sources are correlated —
they are leave-one-out contributions, not an orthogonal decomposition.

**Diagonal vs off-diagonal convention.** Equation (8) is the definition used for
the reported per-observable statistical *variances* — the diagonal, which the
combined-observable systematic impact
$\sigma^\mathrm{syst}_b$ in (9) uses and which equals
$2\,(H_{xx}^{-1})_{bb}$ exactly. For the *off-diagonal* entries of the exported
stat-only and per-source covariance matrices, `pyconvino` follows the reference
tool's construction of Section 5 — the full-fit correlation $\varrho$ scaled by
the frozen errors, $\operatorname{cov}_{ij}=\varrho_{ij}\,\sigma_i\sigma_j$ —
rather than the raw frozen-Hessian block $2\,(H_{xx}^{-1})_{ij}$. The two agree
on the diagonal (the physically reported quantity) but not in general off the
diagonal; the exact-algebra identity of Section 7 is stated for the raw
frozen-Hessian block.

---

## 7. Signed response matrix and the $A\,\Sigma_\lambda\,A^{\mathsf T}$ identity  *(new)*

Define the **signed linear response matrix** $A \in \mathbb{R}^{n_\mathrm{est}
\times n_\mathrm{sys}}$, whose entry $A_{bi}$ is the shift in combined value
$x_b$ per $+1\sigma$ of nuisance $i$, from the Hessian cross-block:

$$
A \;=\; -\,H_{xx}^{-1}\,H_{x\lambda}.
\tag{10}
$$

This is the conditional-mean slope of $x$ given $\lambda$ for the post-fit
Gaussian — the effective combined-space response matrix $L_\mathrm{eff}$ of [1] —
and is obtained from the post-fit Hessian at no extra fit cost. The reference tool reports only a rescaled per-1σ percentage version
of this information ("simple impact table"); the raw signed derivative (10),
needed to feed a downstream nuisance-parameter fit, is new.

**Exact covariance identity.** Define the raw frozen-Hessian systematic
covariance from the observable block only — i.e. using (8) and
$\operatorname{cov}_\mathrm{comb} = 2\,(H^{-1})_{xx}$ directly, before the
off-diagonal rescaling convention of Sections 5–6:

$$
\operatorname{cov}^{H}_\mathrm{syst}
\;\equiv\; 2\big[(H^{-1})_{xx} - H_{xx}^{-1}\big].
$$

Using the Schur-complement form of the inverse block,
$\,(H^{-1})_{xx} = H_{xx}^{-1} + H_{xx}^{-1}H_{x\lambda}
(H_{\lambda\lambda}-H_{\lambda x}H_{xx}^{-1}H_{x\lambda})^{-1}
H_{\lambda x}H_{xx}^{-1}$, together with (10) and the post-fit nuisance
covariance
$\Sigma_\lambda \equiv \operatorname{cov}(\lambda) =
2\,(H^{-1})_{\lambda\lambda} = 2\,(H_{\lambda\lambda}-H_{\lambda x}
H_{xx}^{-1}H_{x\lambda})^{-1}$, one obtains the **general identity**

$$
\boxed{\;\operatorname{cov}^{H}_\mathrm{syst}
   \;=\; A\,\Sigma_\lambda\,A^{\mathsf{T}}\;}
\tag{11}
$$

which holds exactly (not to first order). It has been verified to hold to
machine precision ($\sim\!10^{-12}$, full matrix) on the ATLAS-only, CMS-only and
combined ATLAS+CMS setups, **including** setups whose nuisances are genuinely
constrained ($\Sigma_\lambda \ne I$). In the special case where the fit does
*not* constrain the nuisances beyond their prior — $\Sigma_\lambda = I$ — (11)
reduces to

$$
\operatorname{cov}^{H}_\mathrm{syst} \;=\; A\,A^{\mathsf{T}}.
\tag{12}
$$

Equation (12) holds for the ATLAS-only fixtures, where each combined observable
is a free parameter that absorbs all measurement tension, so every nuisance stays
at its prior (pulls $\hat\lambda = 0$, constraints $=1$, hence
$\Sigma_\lambda = I$); it **fails** for CMS-only, whose nuisances are constrained
(constraints down to $\sim\!0.03$), where the general form (11) must be used.

**Reported quantities.** The per-observable systematic variance actually reported
(the diagonal of (9), $(\sigma^\mathrm{syst}_b)^2$) equals the diagonal of (11),
$\big(A\,\Sigma_\lambda\,A^{\mathsf{T}}\big)_{bb}$, exactly (verified to
$\sim\!10^{-12}$ on all setups above). Off the diagonal, the *exported*
`total_syst_covariance` uses the reference tool's correlation-rescaling
convention (Sections 5–6), so it matches (11) on the diagonal but not in general
off it; (11) is the statement about the underlying Hessian, and is the one to use
when the full systematic covariance is needed exactly.

**Downstream use.** $A$ enables a nuisance-parameter formulation of a downstream
χ² that is equivalent to the covariance form via the Woodbury identity:

$$
\chi^2(\theta_\mathrm{np}, \mu)
= \big(r - A\,\theta_\mathrm{np}\big)^{\mathsf T}
   \operatorname{cov}_\mathrm{stat}^{-1}
  \big(r - A\,\theta_\mathrm{np}\big) + \lVert \theta_\mathrm{np}\rVert^2 ,
$$

with $r = x_\mathrm{data} - x_\mathrm{theory}(\mu)$. Profiling $\theta_\mathrm{np}$
is equivalent (Woodbury) to a covariance fit with effective covariance
$\operatorname{cov}_\mathrm{stat} + A A^{\mathsf T}$, while exposing per-nuisance
pulls and impacts on the downstream parameter $\mu$ directly. This effective
covariance equals the combination's own $\operatorname{cov}_\mathrm{comb}$ when
the nuisances are unconstrained ($\Sigma_\lambda = I$, the standard Convino
configuration); when they are constrained, use the general covariance form (11)
— i.e. weight the penalty by $\Sigma_\lambda^{-1}$ — or feed
$\operatorname{cov}_\mathrm{comb}$ directly.

---

## 8. Group-level effective pulls  *(new)*

For a user-defined group $g$ of systematics, `pyconvino` reports an
impact-weighted mean of the member pulls, in two weighting conventions:

$$
\bar\lambda_g \;=\; \frac{\sum_{i\in g} w_i\,\hat\lambda_i}{\sum_{i\in g} w_i},
\qquad
w_i \;=\;
\begin{cases}
\operatorname{mean}_b\big(|\mathrm{impact}_{i,b}|\big), & \text{(``mean'')},\\[4pt]
\big\lVert \mathrm{impact}_{i,\cdot}\big\rVert_2, & \text{(``norm'')},
\end{cases}
$$

where $\mathrm{impact}_{i,b}$ is the per-observable impact (6) of the single
systematic $i$. This condenses a group's constraint into one number for
group-level downstream-shift estimates; it has no reference-tool equivalent. On
the current fixtures every $\bar\lambda_g = 0$, consistent with the
$\hat\lambda = 0$ observation of Section 7 — non-zero group pulls arise only when
a systematic is shared across measurements with conflicting responses.

---

## 9. Differential bin normalisation  *(port, one divergence)*

For a differential combination flagged `isDifferential` and `normalise`
(paper Sec. 2.4), the combined bin values are renormalised to fractions of their
sum. Because the ratio $x_b / \sum_{b'} x_{b'}$ is nonlinear in the correlated
bin values, the uncertainty is propagated by Monte Carlo, reproducing the
reference's `normaliser` algorithm: draw $N$ samples
$x^{(s)} \sim \mathcal{N}(\hat x, \operatorname{cov}_\mathrm{comb})$, form the
per-draw fraction deviations from the nominal fraction
$f = \hat x / \sum_b \hat x_b$, and take the empirical covariance

$$
d^{(s)} = \frac{x^{(s)}}{\sum_b x^{(s)}_b} - f,
\qquad
\operatorname{cov}_\mathrm{norm} = \frac{1}{N}\sum_{s=1}^{N} d^{(s)} d^{(s)\mathsf T},
\qquad N = 10^6 .
$$

The nominal fractions $f$ (the deterministic part) match the reference exactly;
the MC errors agree within the $\mathcal{O}(1/\sqrt{N})$ statistical precision of
two independent RNGs.

**Deliberate divergence.** After normalisation the reference invalidates
$\chi^2_\mathrm{min}$, the pulls, the constraints and the full correlation matrix
(setting them to $-1$/empty). `pyconvino` leaves these describing the
pre-normalisation fit instead of nulling them: they remain meaningful as the
goodness-of-fit and nuisance summary of the combination that *was* normalised,
now on a different (shape) scale than `combined_values`. A boolean
`normalised` flag records that the step ran.

---

## 10. `lognormal` systematics  *(aligned: fail loudly)*

The `lognormal` systematic type is accepted by the file-format parser but has no
implemented statistical model — it was never completed in the C++ reference
either, which throws on it. `pyconvino` likewise raises `NotImplementedError` at
χ²-build time rather than silently treating a `lognormal` tag as `absolute`.
This is a behavioural alignment (both tools refuse it), noted here only so the
supported model set — `absolute` and `relative` — is unambiguous.

---

## 11. Post-fit nuisance values of profiled inputs  *(new, opt-in)*

**Off by default** (`--use-nuisance-values` / `use_nuisance_values=True`), so
the default result is the original Convino one. Full derivation, formulas and
file format: **[`nuisance_values.md`](nuisance_values.md)**. In short: the
Convino chi2 has no field for an input's fitted nuisance values, so a profiled
input is silently re-centred at $\hat\lambda = 0$ (Dado/Owen/Pinamonti 2026,
App. A). With the option on, each measurement uses
$\delta\lambda = \lambda - \hat\lambda$ plus a linear term
$-2\,\delta\lambda^{\mathsf T} P\hat\lambda$ and a constant
$\hat\lambda^{\mathsf T} P D^{+} P \hat\lambda$; the global prior stays
centred at 0.

**Impacts.** Frozen refits (asymmetric-response path, §5) pin the frozen
nuisances at their post-fit values, not at 0. Without `[nuisance values]` the
two coincide (all pulls are 0 there).

**What changes.** The Hessian is unchanged — the exact and the original
objective differ by a term linear in $\theta$ — so every covariance, error,
impact and `x_sys_cov` is identical. Only the central values and post-fit
pulls move. The original form therefore reports the covariance of the exact
estimator while computing a different, less precise one; the extension makes
the two consistent. For an NP not correlated with any other input the
profiled POI likelihood is the same either way, so `noCorr`-type setups are
unaffected.

## 12. Summary

| # | Item | Class | Reference behaviour |
|---|------|-------|---------------------|
| 2 | Exact one-step quadratic solve; HESSE ≡ MINOS when symmetric | Reformulation | Always iterative Minuit + MINOS |
| 3 | Three-method, warned PD regularisation of the nuisance prior | New | Not a user-visible, quantified step |
| 4 | ndf, χ²/ndf, p-value with the $N_\mathrm{meas}-n_\mathrm{est}$ convention | New | Reports only $\chi^2_\mathrm{min}$ |
| 5 | Frozen-Hessian-submatrix impacts (no per-group re-fit) | Reformulation | Explicit re-fit per group |
| 6 | Automatic stat/syst covariance split | New | Not reported |
| 7 | Signed response matrix $A$; exact identity $\operatorname{cov}^{H}_\mathrm{syst}=A\Sigma_\lambda A^{\mathsf T}$ | New | Only a rescaled % table |
| 8 | Group-level effective pulls | New | Not reported |
| 9 | Differential normalisation | Port | Reproduced; pre-norm summary kept, not nulled |
| 10 | `lognormal` fail-loud | Aligned | Also refuses |
| 11 | Post-fit nuisance values of profiled inputs (opt-in, off by default) | New | Re-centres every input's nuisances at 0 |

All **reformulation** items are validated to reproduce the reference to the
per-mille level or better on the paper's worked example and the ATLAS+CMS
combinations (`docs/paper_benchmark_crosscheck.md`); all **new** items reduce to
quantities derivable from the already-computed post-fit Hessian of the shared
model (1), so they introduce output, not a change to the combination itself.
The one exception is item 11, which changes the objective — and is therefore
off unless the user asks for it.

---

## References

[1] J. Kieseler, *A method and tool for combining differential or inclusive
measurements obtained with simultaneously constrained uncertainties*,
Eur. Phys. J. C **77** (2017) 792, arXiv:1706.01681,
doi:10.1140/epjc/s10052-017-5345-0. (`docs/convino_paper.pdf`; source tool:
<https://github.com/jkiesele/Convino>.)

[2] N. J. Higham, *Computing the nearest correlation matrix — a problem from
finance*, IMA J. Numer. Anal. **22** (2002) 329–343,
doi:10.1093/imanum/22.3.329.
