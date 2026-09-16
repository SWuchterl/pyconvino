# Post-fit nuisance values of a profiled input (`[nuisance values]`)

**Default: off.** Without `--use-nuisance-values` pyconvino calculates exactly
what the original C++ Convino calculates. The block is then read and checked,
but it has no effect on the result.

```bash
convino config.txt --use-nuisance-values        # CLI
Combiner.from_config(cfg, use_nuisance_values=True)   # Python API
```

Turn it on only if at least one input is a full-likelihood (profiled) fit.
For all other inputs the two forms are identical, because their nuisances
are at zero anyway.

---

## 1. The problem

An input that is a profile-likelihood fit reports two things: the fitted
parameters θ̂ = (d, λ̂), where d are the POIs and λ̂ the nuisance parameters
(NPs), and the covariance matrix. Convino uses the covariance, but it has no
field for λ̂. Its χ² for one measurement is

$$
\chi^2_k = (d - x - k\lambda)^{\mathsf T} M (d - x - k\lambda)
         + \lambda^{\mathsf T} D \lambda + \lambda^{\mathsf T}\lambda .
$$

Set the two derivatives to zero. The first gives `d − x − kλ = 0`; put that
into the second and `Dλ + λ = 0` remains, so **λ = 0 and x = d**. The
measurement is therefore treated as if its fit had returned all NPs at zero.

That is correct for an input that cannot constrain its NPs (they must stay at
zero there). It is wrong for an input that does constrain them: the
information "this experiment measured λ₃ = +0.9 ± 0.4" is thrown away, and
the combination keeps the smaller uncertainty that this measurement caused
without keeping the measurement itself. Reference: Dado, Owen and Pinamonti
(2026), Appendix A.

## 2. The extension

Let H be the input's Hessian and P the prior precision that the input fit used
for its NPs (P = 1 for a unit-Gaussian constraint, P = 0 for a parameter
tagged `free`). Convino removes this prior from each input and adds it once,
globally. What is left is the data-only form

$$
\chi^2_{\text{data},k}(\theta) = (\theta - \hat\theta_d)^{\mathsf T} H_d (\theta - \hat\theta_d),
\qquad H_d = H - \mathrm{diag}(0, P).
$$

θ̂_d is the optimum of the data alone. The input does not report it, but the
input fit was stationary, so `H_d θ̂_d = H θ̂`. Only θ̂ and H are necessary.
Written in Convino's variables, with **δλ = λ − λ̂**:

$$
\boxed{\;
\chi^2_k = (d - x - k\,\delta\lambda)^{\mathsf T} M (d - x - k\,\delta\lambda)
 + \delta\lambda^{\mathsf T} D\, \delta\lambda
 \;\underbrace{-\; 2\,\delta\lambda^{\mathsf T} P \hat\lambda}_{\text{new}}
 \;+\; \underbrace{\hat\lambda^{\mathsf T} P D^{+} P \hat\lambda}_{\text{new, constant}} \;}
$$

Three changes against the original form:

1. **λ → λ − λ̂** in the shift term and in the D term. This alone is not
   enough: it would move the minimum to λ = λ̂ with no penalty.
2. **The linear term −2 δλᵀ P λ̂.** This is the prior gradient that the input
   fit balanced against its own data at λ̂. It is not optional. It is what
   makes the combined fit pay a price for moving λ away from λ̂ *and* for
   keeping λ̂ away from the global prior centre.
3. **The constant λ̂ᵀ P D⁺ P λ̂** (D⁺ = pseudo-inverse, because D is singular
   in the directions that the data do not constrain, where P λ̂ = 0 anyway).
   It sets the zero point so that the data term vanishes at θ̂_d.

The global prior λᵀ C⁻¹ λ is unchanged. It is centred at zero, which is
correct: zero is where the external knowledge of the NP sits.

## 3. What this gives

- **A standalone input is reproduced exactly**: the combination of one input
  returns its own θ̂ and its own covariance H⁻¹. The original form returns the
  same covariance but sets all NPs to zero.
- **The minimum χ² is no longer zero.** It becomes
  `χ²_standalone = λ̂ᵀ (P + P D⁺ P) λ̂`, the prior penalty that the input fit
  already paid. The text output writes this per input as `chi2 standalone
  <file>`, and `chi2 tension = χ²_min − Σ χ²_standalone`, which is the part
  that comes from the combination itself. These two lines appear only when an
  input really carries nuisance values, so the default text output stays
  byte-identical to the C++ one. Use the tension, not χ²_min, to
  judge agreement between the inputs.
- **The covariance does not change.** The new terms are linear and constant in
  λ, so the Hessian is identical. Only the central values, the pulls and χ²
  move. Uncertainties, correlations and impacts stay bit-identical.
- **Free parameters** (`name = free` in `[systematics]`) have P = 0. Their
  linear term and their part of the constant are zero, and they get no global
  prior either.

## 4. File format

```
[nuisance values]
    CMS_modelling_ttbar_bfrag = 0.170976
    CMS_scale_j_flavorBottom  = -0.412
[end nuisance values]
```

One line per NP, value in units of the prior width. Every name must be a
parameter of the `[hessian]` or `[correlation matrix]` block of the same
file; an unknown name is an error. Missing names are zero. Parameters from
`[not fitted]` are externalised, so they are always zero.

## 5. Verification

1. **Algebra.** `test/test_nuisance_values.py` builds two linear-Gaussian toy
   inputs, writes them as Convino files and shows that the combination equals
   the exact joint fit of both data sets with the prior counted once (values,
   covariance and χ² to machine precision), which the original form does not.
   A standalone input reproduces its own fit exactly. It also checks that the
   block is inert unless the option is on.
2. **Against an independent implementation.** On the published example of
   Dado, Owen and Pinamonti (ATLAS boosted + J/ψ top-quark mass) this option
   reproduces their Combiner v1.0.0 BLUE result to all six digits, in both the
   central value and the uncertainty (172.863654 ± 0.483478). Their code is a
   different method, a different language and a different author group.
3. **Coverage.** In 5000 pseudo-experiments the reported uncertainty covers
   correctly (RMS/σ = 0.99), where the original form gives 1.30 and 1.60.
4. **The input mapping.** For our CMS 13 TeV input the 234 post-fit
   constraints of the Convino file agree with the independent HEPData
   `np_impacts_pulls` table to 0.00 % under the name mapping used to attach the
   pulls, so the pulls are attached to the right nuisances. The largest pull
   difference is 3·10⁻⁶ (rounding in the published values).
5. **The one convention that is ambiguous is inert.** A `free` parameter
   quoted around 1 instead of 0 (CMS `rateTT0Jet`) has P = 0, so its value
   drops out: removing it or shifting it by −1 changes the combination by
   10⁻¹² σ and leaves χ² unchanged.

Limits: like the whole Convino and BLUE framework, this is exact only in the
Gaussian regime of the input fit, and it assumes the input's nuisance prior was
a unit Gaussian (P = 1) except where `free` is given.

Full closure test against the paper and their code, including the
pseudo-experiments: `paper_dado_owen_pinamonti_reproduction.md`.
