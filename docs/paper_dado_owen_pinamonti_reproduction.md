# Closure test against Dado, Owen and Pinamonti (2026) and the Combiner tool

**Date:** 2026-09-11 (first version 2026-09-04, against the unpublished draft)
**Paper:** arXiv:2609.11461v1, "Combinations of measurements that are simultaneous
fits of parameters of interest and systematic uncertainties using the BLUE method".
**Their code:** Combiner v1.0.0, `https://gitlab.cern.ch/tdado/combiner`,
commit 9016bb6 (2026-09-03), Zenodo DOI 10.5281/zenodo.12007777.
**Scripts:** `examples/dado_owen_pinamonti/` (see its README).

The paper says that Convino under-estimates the combined uncertainty when an
input is a profile-likelihood fit, because the Convino χ² does not use the
input's fitted nuisance values (their Appendix 7). The pyconvino option
`--use-nuisance-values` (`docs/nuisance_values.md`) adds these values. This
document gives the closure test of both against their published code.

The published version contains the configuration files of both paper examples.
All numbers below therefore use **the authors' own inputs**, not inputs read
from a figure. Combiner was built here from source (ROOT 6.40, GCC 11.5) and
run on those configs.

---

## 1. Example 1 — two toy analyses (paper Section 4)

Two analyses measure one signal strength μ. Each has a signal region and two
control regions, and two nuisance parameters (NPs) with a unit Gaussian prior.
Analysis A is precise on μ and weak on the NPs, analysis B the opposite.
"A SR only" is analysis A fitted in the signal region alone, which makes it a
statistics-only measurement.

### 1.1 Our BLUE against their code

Our implementation of the paper's Eq. 2.1–2.3 is in
`examples/dado_owen_pinamonti/closure.py`. It builds each measurement's
extended covariance exactly as Combiner does
(`Measurement::CalculateFullCovarianceMatrix`), that is
`C = [[O, Γ],[Γᵀ, Σ]]` with `Γ` the per-systematic `impact`, and the
cross-measurement block `[[Γ_A Γ_Bᵀ, Γ_A Σ_B],[Σ_A Γ_Bᵀ, Σ_A Σ_B]]`.

| Combination | σ(POI) ours | σ(POI) runCombiner | σ(NP1) / σ(NP2) ours | paper Table 1 |
|---|---|---|---|---|
| A & B, POI and NPs | 0.098345 | 0.098345 | 0.821405 / 0.467258 | 0.10 / 0.82 / 0.47 |
| A & B, POI only | 0.123133 | 0.123133 | – | 0.12 |
| A SR & B, POI and NPs | 0.106567 | 0.106567 | 0.915044 / 0.482771 | 0.11 / 0.92 / 0.48 |
| A SR & B, POI only | 0.159339 | 0.159339 | – | 0.16 |

Every digit agrees. The BLUE weights agree as well:

| Combination | weights (μ row), ours = runCombiner | paper Table 2 |
|---|---|---|
| A & B, POI and NPs | 1.022353, 0.030037, 0.103312, −0.022353, −0.030037, −0.103312 | 1.02, −0.02, 0.03, −0.03, 0.10, −0.10 ✓ |
| A & B, POI only | 0.893824, 0.106176 | 0.89, 0.11 ✓ |
| A SR & B, POI and NPs | 1.046377, **0.128721**, **0.096479**, −0.046377, −0.128721, −0.096479 | 1.05, −0.05, **0.10**, −0.10, **0.11**, −0.11 ✗ |
| A SR & B, POI only | **0.833932, 0.166068** | **1.05, −0.05** ✗ |

**Two entries of the published Table 2 disagree with the authors' own code.**
The NP weights of the third row should read 0.13 / −0.13 / 0.10 / −0.10, and
the fourth row should read 0.83 / 0.17 — the printed values repeat the POI
weights of the row above. The uncertainties in Table 1 are correct, and our
code and theirs agree exactly, so this looks like an error in the table, not
in the method. Worth reporting to the authors.

For the record, the earlier reproduction that read the inputs off Figure 1
was correct: it gave 0.129 / 0.097 for these weights, and it gave 0.166 for
"A SR only", where the draft printed 0.13. The published Table 1 now says
0.17.

### 1.2 The two pyconvino modes

Written as Convino inputs, the same two measurements give:

| Mode | μ | σ(μ) | pulls |
|---|---|---|---|
| default (C++ Convino form) | 1.000000 | 0.098323 | 0, 0 |
| `--use-nuisance-values` | 1.000000 | 0.098323 | 0, 0 |
| BLUE with POI and NPs | 1.000000 | 0.098345 | – |

On the Asimov data set every fitted NP is zero, so the two pyconvino modes are
identical here. **The difference between the modes appears only in the
pseudo-experiments** (Section 3), where the fitted NPs move away from zero.

## 2. Example 2 — ATLAS boosted + J/ψ top-quark mass (paper Section 5)

Inputs: 133 systematics for the boosted measurement (the 15 Barlow–Beeston
MC-statistics parameters of the HEPData record are dropped, and the POI
variance is correspondingly reduced from 0.53268² to 0.525732²), 251 for the
J/ψ measurement, 113 of them shared by name. The J/ψ impacts carry the
opposite sign to the HEPData table, the boosted ones do not.

| Combination | m_t ours | σ ours | runCombiner | paper Table 4 |
|---|---|---|---|---|
| POI only | 172.870801 | 0.499652 | 172.871 ± 0.499652 | 0.50 |
| POI and recoil NP | 172.878434 | 0.484252 | 172.878 ± 0.484252 | 0.48 |
| recoil NP itself | −0.369297 | 0.329951 | −0.369297 ± 0.32995 | – |
| POI and all 113 shared NPs | 172.863654 | 0.483478 | – | "negligible improvement" ✓ |

μ-row weights, ours and theirs: 0.900368 / 0.099632 (POI only) and
0.843919 / −0.165966 / 0.156081 / 0.165966 (POI and recoil NP). The paper
prints 0.90 / 0.10 and 0.84 / −0.17 / 0.16 / 0.17.

The same two inputs through pyconvino:

| Mode | m_t | σ | recoil pull |
|---|---|---|---|
| default (C++ Convino form) | 172.824691 | 0.483478 | −0.104 ± 0.329 |
| `--use-nuisance-values` | **172.863654** | **0.483478** | −0.382 ± 0.329 |
| BLUE with all 113 shared NPs | **172.863654** | **0.483478** | – |

**pyconvino with the nuisance values reproduces the paper's BLUE exactly**, to
all six digits, in both the central value and the uncertainty. The default
form gives the same uncertainty but a central value 39 MeV away, and it pulls
the recoil NP from −0.38 back to −0.10, as the paper's Appendix predicts.

## 3. Pseudo-experiments (paper Table 3)

5000 pseudo-experiments of the Example 1 model (`toys.py`, `toymodel.py`): the
global observables are drawn once per experiment and shared by both analyses,
the event counts are Poisson, both analyses are re-fitted and every combination
is redone. The generative model is the one of the paper; its Asimov fit
reproduces the authors' config inputs to 0.03 %, so it is a fair test of the
estimators. The statistical precision is 1.0 % on each RMS.

| Combination of A and B | reported σ | RMS of μ | RMS/σ | paper Table 3 |
|---|---|---|---|---|
| BLUE, POI only | 0.1231 | 0.1221 | 0.99 | 0.12 / 0.12 |
| BLUE, POI and NPs | 0.0983 | 0.0975 | 0.99 | 0.10 / 0.10 |
| pyconvino default (C++ form) | 0.0983 | **0.1280** | **1.30** | 0.10 / 0.12 |
| pyconvino `--use-nuisance-values` | 0.0983 | 0.0975 | 0.99 | – |

| Combination of A SR and B | reported σ | RMS of μ | RMS/σ |
|---|---|---|---|
| BLUE, POI and NPs | 0.1064 | 0.1060 | 1.00 |
| pyconvino default (C++ form) | 0.1064 | **0.1698** | **1.60** |
| pyconvino `--use-nuisance-values` | 0.1064 | 0.1060 | 1.00 |

One number does not match exactly: the paper prints an RMS of 0.12 for Convino,
we measure 0.1280 ± 0.0013, which would print as 0.13. The Asimov inputs of our
toy model agree with theirs to 0.03 %, so the model is not the cause; the
remaining candidates are the pseudo-experiment sample and the difference between
the C++ Convino and pyconvino in this configuration (they agree on the Asimov
point, 0.098323 against a reported 0.10). In our run the default Convino form is
slightly worse than the POI-only BLUE (0.128 against 0.122), while the paper
gives both as 0.12. The conclusion is unchanged.

The paper's claim is confirmed with their own inputs: the default Convino form
reports 0.098 but has a true spread of 0.128, so it under-estimates by 30 %,
and by 60 % when the second input is statistics-only — which is the class our
ATLAS+CMS combination belongs to. With the nuisance values the reported
uncertainty covers correctly.

Per pseudo-experiment, pyconvino `--use-nuisance-values` and the paper's BLUE
give the same combined μ: the RMS of the difference is 0.0020 for A & B (from
the Poisson non-linearity and the prior counting of Section 4) and **exactly
zero** for A SR & B, where the two methods are algebraically identical.

## 4. Where the two methods differ, and by how much

The BLUE of the paper treats each input's NP estimate as an independent
measurement of that NP. Each input's covariance already contains the unit
Gaussian prior of the NP. When **two** inputs both constrain the same NP, that
prior is therefore counted twice. pyconvino removes each input's own prior
(the `−diag(P)` in `D`) and adds it once globally, which is the exact joint
likelihood.

Measured on these inputs (data information on the NPs with the POI profiled
out, `closure.py`):

| Inputs | Both constrain the NPs? | pyconvino + values | BLUE | exact joint fit |
|---|---|---|---|---|
| A & B | yes | 0.098323 | 0.098345 | 0.098323 |
| A SR & B | no (A SR has none) | 0.106567 | 0.106567 | 0.106567 |
| boosted & J/ψ | no (J/ψ has none) | 0.483478 | 0.483478 | – |

So the two methods agree exactly whenever at most one input constrains a given
NP, which covers both paper examples and our ATLAS+CMS combination (only the
CMS 13 TeV input is profiled). They differ by 0.02 % in the one case where
both inputs constrain the same NPs, and there pyconvino equals the exact joint
fit. The difference grows with the number of inputs that constrain the same
NPs; it is not visible in 5000 pseudo-experiments.

## 5. Their tool on our ATLAS+CMS combination

`examples/dado_owen_pinamonti/to_combiner.py` translates one of our Convino
setups into Combiner YAML: per measurement it writes
`O = (statistical covariance) + Σ_ext Γ Γᵀ`, the per-systematic `impact`
(= cov(estimate, NP)), the post-fit NP uncertainties and values, and the NP
correlation matrix. Our `[correlations]` block becomes `byHandNPcorrelations`.

### 5.1 Combiner cannot take our correlation model

```
ERROR CorrelationMa...:247 | Inconsistent by-hand NP correlation setup:
"CMS_modelling_singletop_fsr" and "CMS_modelling_singletop_isr" already share a
measurement, but the by-hand correlations provided transitively imply a
correlation between them. Please remove one of the linking correlations.
```

Our model has 77 non-zero cross-experiment correlations (0.25, 0.50, 0.75)
between **differently named** nuisance parameters — no name is shared by two of
our three inputs. 44 of those nuisances appear in more than one pair, which
links them into groups (the largest has 16 members: one CMS jet-scale nuisance
partially correlated with 8 ATLAS 8 TeV and 7 ATLAS 13 TeV components). Combiner
requires every pair inside such a group to be given explicitly and refuses when
two members come from the same measurement. That is the normal shape of an
ATLAS/CMS correlation model, and Convino handles it natively through the prior
correlation matrix.

A second limitation appears in the log: Combiner clamps any post-fit nuisance
uncertainty above 1 ("setting to 1"). Ten of our 234 CMS nuisances are above 1
(up to 1.095).

### 5.2 On the reduced model that Combiner accepts

Keeping only the 14 correlation pairs whose two nuisances appear in no other
pair, both tools run on identical assumptions
(`examples/dado_owen_pinamonti/compare_ourcomb.py`):

| | central values | uncertainties | cross-experiment ρ |
|---|---|---|---|
| the three inputs standalone | – | – | – |
| **Combiner** (BLUE with NPs) | = inputs | = inputs | **0** (max 5·10⁻¹⁴) |
| **pyconvino default** | = inputs | up to 0.6 % smaller | up to 0.017 |
| **pyconvino `--use-nuisance-values`** | ATLAS bins shift up to **0.21 σ** | up to 0.6 % smaller | up to 0.017 |

On our full correlation model pyconvino gives cross-experiment ρ up to 0.138,
uncertainties up to 3.3 % smaller than the inputs, and a shift of up to 0.12 σ
from the nuisance values.

**Why Combiner returns the inputs.** In our combination every observable is
measured by exactly one input, and no nuisance parameter is shared by name.
Partially correlated nuisances are moved out of the combined parameters
(`ModelBuilder::AddByHandNPsToNotFitList`), so nothing is combined: the BLUE
weight matrix is a permutation and the result is each input's own value and
uncertainty, with no correlation induced between the experiments. The
information that the profiled CMS fit carries about nuisances that are
*partially* correlated with ATLAS ones cannot enter a BLUE that treats a shared
nuisance as one parameter measured twice. This is precisely what the Convino
prior correlation matrix is for, and it is the reason our combination uses
Convino rather than BLUE.

### 5.3 A bug in their output files

`Covariance_POIs.txt` and `Correlation_POIs.txt` (and the matching plots) are
mis-indexed when the POIs are not the first parameters of the model. In our run
the rows labelled with the 15 ATLAS observables carry the values of the first 15
CMS nuisance parameters:

| row label in `Covariance_POIs.txt` | value written | the parameter it belongs to |
|---|---|---|
| ATLAS_8TeV_..._rho_0p0_0p25 | 0.525645 | CMS_modelling_ttbar_bfrag |
| ATLAS_8TeV_..._rho_0p25_0p325 | 1.0 | CMS_modelling_ttbar_bfragPythiaDefault |
| ATLAS_13TeV_..._rho_0p8_1p0 | 0.046075 | CMS_norm_zjets_3j |

The correct ATLAS values (14975.3 = 122.37², …) are in
`Covariance_all_parameters.txt`, which is indexed by name and is right. The
headline numbers (`Parameters.txt`, the log, `Weights.txt`) are also right,
because they go through `GetCombinedResult(name)`. In both paper examples every
measurement has one POI of the same name, so the POIs come first and the bug
cannot show. Worth reporting to the authors.

## 6. How to repeat this

```bash
git clone https://gitlab.cern.ch/tdado/combiner.git && cd combiner
git submodule update --init
mkdir build && cd build && cmake .. && make -j6          # needs ROOT
./build/bin/runCombiner configs/paper/simple_example/mainConfig.yml
./build/bin/runCombiner configs/paper/simple_example/mainConfig.yml npsNotToFit=".*"
./build/bin/runCombiner configs/paper/mt_example/mainMt.yml
# POI-only for the mt example needs a config copy without the npsToFit line,
# otherwise alpha_Recoil is both selected and excluded and the tool aborts.

cd pyconvino/examples/dado_owen_pinamonti
python3 closure.py <path-to-combiner-repo>     # needs pyyaml
python3 toys.py 5000                           # pseudo-experiments

# their tool on our combination
python3 to_combiner.py ../../ConvinoSetups/Combination_ATLAS813CMS13_corrV2/rho_config.txt <dir> [--drop-chains]
python3 compare_ourcomb.py <our-config> <their-log> <reduced-setup-dir>
```

The HS3 likelihoods of Example 1 are published at
`https://doi.org/10.5525/gla.researchdata.2414`. They are not used here: the
toy model in `toymodel.py` reproduces the authors' Asimov inputs to 0.03 %.
