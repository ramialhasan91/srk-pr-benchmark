# SRK / Peng–Robinson saturation-property benchmark — error budget edition

Reproducible benchmark of the SRK and Peng–Robinson (PR) cubic equations of state against reference
Helmholtz-energy equations of state (via CoolProp) for 37 pure fluids in five chemical families:
saturation pressure, saturated-liquid density, saturated-vapor density and enthalpy of vaporization,
1480 saturation states, Tr = 0.50–0.985.

Each cubic is combined with **seven α-functions** — six generalized ones that need only (Tc, pc, ω)
(classical Soave; the 2019 Pina-Martinez update; Twu-95; the consistent generalized Twu-91 of
Jaubert, Privat and co-workers; the exponential Coquelet function; the consistent Mahmoodi–Sedigh
(MS) function) and, as a fitted reference, the component-specific consistent Twu-91 parameters of the
1800-fluid compilation of Piña-Martinez et al. (2022) — and with a three-rung **volume-translation
ladder** (Rackett-anchored, density-anchored, critical-matched) on both cubics. Every pooled statistic
carries a fluid-level bootstrap confidence interval and every model comparison an exact paired sign
test (206 tests). Two internal consistency checks (translation invariance to 1e-14; departure-function
vs Clapeyron enthalpy of vaporization to 6e-7 at every point), an anchoring-temperature sweep and a
critical-constant sensitivity analysis of the fitted reference are included.

Companion code and data for the manuscript *"An Error Budget for the SRK and PR Equations of State:
α-Function Parameterization and Volume Translation in a Reproducible Pure-Fluid Saturation-Property
Benchmark"* (submitted to *Thermo*).

## Repository structure

```
.
├── code/                          Analysis pipeline (run in the order below)
│   ├── eos.py                     Cubic EoS core: SRK/PR with the seven α-functions (analytic
│   │                              derivatives), fugacity coefficients, saturation-pressure solver,
│   │                              residual enthalpy, translation-invariance quadrature check.
│   ├── fluid_set.py               The 37 fluids, families and polar/associating subgroups.
│   ├── 00_extract_twu_fit_params.py  (optional) regenerates data/twu_fit_params.csv; needs `thermo`.
│   ├── 01_run_benchmark.py        Runs the benchmark: 14 (cubic, α) combinations × 4 properties,
│   │                              translation ladder on every combination, Clapeyron and
│   │                              translation-invariance checks, anchor sweep, Halm–Stiel factors.
│   ├── 04_stats.py                Fluid-level cluster-bootstrap 95% CIs, exact binomial sign tests,
│   │                              near-critical Zc regression.
│   ├── 06_twufit_source_constants.py  Re-evaluates the fitted reference with the critical constants of
│   │                              its source compilation (critical-constant confound, Table S15).
│   ├── 02_make_tables.py          Every table of the paper and SI (LaTeX + tables/tables.json).
│   ├── 03_make_figures.py         Every figure (PDF + 600-dpi PNG) and the graphical abstract.
│   └── 05_check_manuscript_numbers.py  Cross-checks every number quoted in the text against the data.
│
├── data/
│   ├── twu_fit_params.csv         Component-specific Twu-91 parameters of tc-RK/tc-PR for the 37
│   │                              fluids (from the SI of Piña-Martinez et al. 2022; verified line by line).
│   ├── twu_fit_source_critical_constants.csv  Tc, pc of the same source compilation for the 37 fluids.
│   ├── twufit_source_constants.csv / .json    Twu-fit re-evaluated with the source constants (06).
│   ├── deviations_full.csv        Per-point results and % deviations (1480 rows).
│   ├── fluids.csv                 Fluid metadata: Tc, pc, ω, Zc_ref, Z_RA, χ, translation constants.
│   ├── anchor_sweep.csv           Pooled translated liquid-density AAD vs anchoring temperature.
│   ├── key_numbers.json           Headline statistics (overall, by range, by family, per fluid).
│   ├── stats.json                 Bootstrap CIs, sign tests, near-critical regression.
│   └── run_metadata.json          Library versions, solver settings, convergence and check record.
│
├── figures/                       fig1–fig7 (main text; file numbers = figure numbers), figS1–figS3
│                                  (SI), graphical_abstract.
├── tables/                        Generated LaTeX tables and tables.json.
├── manuscript/                    Manuscript sources: main.md, si.md, refs.py, make_word.py
│                                  (MDPI-style Word build), make_letters.py.
├── requirements.txt
├── LICENSE                        MIT.
└── README.md
```

No experimental data are transcribed anywhere in the repository except the published α-function
constants; every reference value is computed at run time from the reference equations of state in
CoolProp.

## Reproducing the results

Requires Python ≥ 3.11 and the pinned versions in `requirements.txt` (CoolProp 8.0.0, NumPy 2.4.4,
pandas 3.0.2, matplotlib 3.10.x). Run the scripts **in this order**:

```bash
pip install -r requirements.txt
python code/01_run_benchmark.py    # -> data/*.csv, key_numbers.json, run_metadata.json  (~2 min)
python code/04_stats.py            # -> data/stats.json
python code/06_twufit_source_constants.py    # -> data/twufit_source_constants.csv/.json
python code/02_make_tables.py      # -> tables/*.tex, tables/tables.json
python code/03_make_figures.py     # -> figures/*.pdf, *.png
python code/05_check_manuscript_numbers.py   # optional: 372 checks against the quoted numbers
```

To rebuild the Word files (needs `python-docx` and `pandoc`):

```bash
python manuscript/make_word.py     # -> submission/Main_Manuscript_Thermo.docx, Supplementary_Materials_Thermo.docx,
                                   #    submission/figures_numbered/, submission/Graphical_Abstract.png
```

The pipeline is deterministic (fixed bootstrap seed); a clean run reproduces the committed
`data/`, `tables/` and `figures/`.

## Model naming

`<cubic>_<alpha>` with cubic ∈ {srk, pr} and alpha ∈ {s (Soave), s19 (Soave-19), t95 (Twu-95),
tc (Twu-c), coq (Coquelet), ms (MS), tf (Twu-fit)}; translated variants carry the suffix `_vt`
(density-anchored), `_vtc` (Rackett-anchored) or `_vtz` (critical-matched, diagnostic only).

## Citation

> R. Alhasan, "An Error Budget for the SRK and PR Equations of State: α-Function Parameterization and
> Volume Translation in a Reproducible Pure-Fluid Saturation-Property Benchmark," *Thermo* (submitted, 2026).

Software archive: https://doi.org/10.5281/zenodo.21370716 (concept DOI; version 2.0).

## License

MIT; see [LICENSE](LICENSE).
