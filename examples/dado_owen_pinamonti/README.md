# Dado / Owen / Pinamonti (2026) — closure test

Paper: arXiv:2609.11461v1. Their tool: Combiner v1.0.0,
`https://gitlab.cern.ch/tdado/combiner` (needs ROOT; build with cmake).
Results and discussion: `docs/paper_dado_owen_pinamonti_reproduction.md`.

- `closure.py <combiner-repo>` — **the closure test**. Reads the authors' own
  configs (`configs/paper/`), rebuilds each measurement's extended covariance the
  way Combiner does, runs BLUE (paper Eq. 2.1–2.3) and pyconvino in both modes on
  the same inputs. Needs `pyyaml`.
- `toys.py [N]` — N pseudo-experiments of Example 1 (paper Table 3), seven
  combination variants. `toys_2000.npz` / `toys_5000.npz` hold the stored runs.
  Run with `OMP_NUM_THREADS=1`, one process per core.
- `toymodel.py` — the Example 1 likelihood (signal region + two control regions,
  Poisson). Its Asimov fit reproduces the authors' config inputs to 0.03 %, so it
  is used as the generative model for the pseudo-experiments.
- `convino_io.py` — writes a toy fit as Convino input files and runs pyconvino.
- `ex2.py`, `ex2_truevar.py`, `hepdata/`, `boosted_cov.npz` — the first
  reproduction of Example 2, built from HEPData before the configs were published.
  Superseded by `closure.py` for the numbers, kept for the HEPData bookkeeping
  (sign conventions, the MC-stat parameters, the name mapping).

Scratch outputs (`conv_ex1/`, `conv_ex2/`, `work*/`, `toys.npz`) are ignored by git.
