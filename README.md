# SRK / Peng–Robinson saturation-property benchmark — error budget edition

Reproducible benchmark of the SRK and Peng–Robinson (PR) cubic equations of state against reference
Helmholtz-energy equations of state (via CoolProp) for 37 pure fluids in five chemical families:
saturation pressure, saturated-liquid density, saturated-vapor density and enthalpy of vaporization,
1480 saturation states, Tr = 0.50–0.985.

Each cubic is combined with **seven α-functions** — six generalized ones that need only (Tc, pc, ω)
(classical Soave; the 2019 Piña-Martinez update; Twu-95; the consistent generalized Twu-91 of
Jaubert, Privat and co-workers; the exponential Coquelet function; the consistent Mahmoodi–Sedigh
(MS) function) and, as an out-of-sample fitted reference, the component-specific consistent Twu-91
parameters of the 1800-fluid compilation of Piña-Martinez et al. (2022) — and with a three-level
**volume-translation ladder** (untranslated, Rackett-anchored, density-anchored; a critical-matched
constant serves as a diagnostic) on both cubics. In addition, the α-function of each cubic is refitted
per fluid to the same reference vapor pressures (one-parameter Soave form and three-parameter
consistent Twu-91 form), so that the SRK–PR difference is measured at matched parameterization.
Every pooled statistic carries a fluid-level bootstrap confidence interval; every model comparison
carries a paired-difference interval, an exact sign test and a multiplicity adjustment (Holm within
three families defined by the question each comparison answers, Benjamini–Yekutieli over all
242 comparisons).
Internal consistency checks (translation invariance to 1e-14; departure-function vs Clapeyron
enthalpy of vaporization to 6e-7 at every point), an anchoring-temperature sweep, the PR78 variant,
the critical-constant dependence of fitted parameters, the input sensitivities to Tc, pc and ω, and
the physical diagnostics behind the interpretation (sensitivity to the attraction parameter,
near-critical closure exponents, Zc dependence of the liquid-density error, Clapeyron split of the
enthalpy of vaporization) are included.

Companion code and data for the manuscript *"An Error Budget for the SRK and PR Equations of State:
α-Function Parameterization and Volume Translation in a Reproducible Pure-Fluid Saturation-Property
Benchmark"* (*Thermo*, under review; manuscript Thermo-4636081). Release **v2.1.0** is the revised
version; the previous Zenodo version is the submitted one.

## Repository structure

```
.
├── run_all.py                     Runs the whole pipeline in dependency order (see below).
├── code/                          Analysis pipeline
│   ├── eos.py                     Cubic EoS core: SRK/PR with the seven α-functions (analytic
│   │                              derivatives), fugacity coefficients, saturation-pressure solver,
│   │                              residual enthalpy, translation-invariance quadrature check.
│   ├── fluid_set.py               The 37 fluids, families and polar/associating subgroups.
│   ├── 00_extract_twu_fit_params.py  (optional) regenerates data/twu_fit_params.csv; needs `thermo`.
│   ├── 01_run_benchmark.py        The benchmark: 14 (cubic, α) combinations × 4 properties, the three
│   │                              translation constants on every combination, Clapeyron and
│   │                              translation-invariance checks, anchor sweep, Halm–Stiel factors.
│   ├── 04_stats.py                Fluid-level bootstrap 95% CIs, exact sign tests, near-critical Zc
│   │                              regression.
│   ├── 06_twufit_source_constants.py  Fitted reference re-evaluated with the critical constants of its
│   │                              source compilation (critical-constant confound, tabS_twufit_src).
│   ├── 07_insample_fit.py         In-sample α-function fits per fluid and cubic (Soave-is, Twu-is):
│   │                              SLSQP under the consistency constraints, six starts per fit.
│   ├── 08_paired_statistics.py    Paired bootstrap intervals on the same resamples as 04 for the 206
│   │                              predictive comparisons, Holm within their families, mean signed deviations.
│   ├── 09_insample_evaluation.py  The in-sample optima re-evaluated with eos.py; their statistics, the
│   │                              final Holm families F1–F3 (Table 4), Benjamini–Yekutieli over all 242
│   │                              comparisons (Table S14), the floor-plus-loss decomposition.
│   ├── 10_pr78_shift_input_error.py  PR78 variant; dimensionless shifts s = c/b; absolute Δhvap
│   │                              deviations; model-form vs input error (compilation constants,
│   │                              sensitivities to Tc, pc, ω).
│   ├── 11_physical_diagnostics.py Sensitivity to the attraction parameter, spread of the generalized
│   │                              α-functions, near-critical exponents, Zc dependence of the liquid-
│   │                              density error, Clapeyron split of Δhvap, n-decane example.
│   ├── 02_make_tables.py          Every table of the paper and SI (LaTeX + tables/tables.json).
│   ├── 03_make_figures.py         Every figure (PDF + 600-dpi PNG) and the graphical abstract.
│   └── 05_check_manuscript_numbers.py  Checks 775 numbers quoted in the two texts against the data.
│
├── data/                          Inputs (two files) and every generated result
│   ├── twu_fit_params.csv         Component-specific Twu-91 parameters of tc-RK/tc-PR for the 37
│   │                              fluids (from the SI of Piña-Martinez et al. 2022; verified line by line).
│   ├── twu_fit_source_critical_constants.csv  Tc, pc of the same source compilation for the 37 fluids.
│   ├── deviations_full.csv        Per-point results and % deviations (1480 rows)           [01]
│   ├── fluids.csv                 Fluid metadata: Tc, pc, ω, Zc_ref, Z_RA, χ, translation constants [01]
│   ├── anchor_sweep.csv, key_numbers.json, run_metadata.json                                 [01]
│   ├── stats.json                 Bootstrap CIs, sign tests, near-critical regression        [04]
│   ├── twufit_source_constants.csv / .json                                                   [06]
│   ├── insample_fit.csv, insample_fit_starts.csv, insample_fit_meta.json                      [07]
│   ├── paired_comparisons.csv, comparison_families.json, signed_deviations*.csv               [08]
│   ├── insample_points.csv, insample_pooled.csv, insample_per_fluid.csv, insample_comparisons.csv,
│   │   holm_extended.csv, by_all_comparisons.csv, insample_decomposition.json,
│   │   insample_alpha_closeness.csv, insample_verification.json                              [09]
│   ├── pr78_*.csv/.json, volume_shift.csv, volume_shift_summary.json, hvap_absolute*.csv,
│   │   compilation_constants*.csv/.json, input_sensitivities.csv                             [10]
│   └── physical_diagnostics.json, physics_*.csv                                              [11]
│
├── figures/                       fig1–fig7 (main text; file numbers = figure numbers), figS1–figS3
│                                  (SI), graphical_abstract.
├── tables/                        Generated LaTeX tables and tables.json (35 tables).
├── manuscript/                    Manuscript sources: main.md, si.md, refs.py, make_word.py
│                                  (MDPI-style Word build), make_letters.py.
├── requirements.txt               Pinned library versions (environment.yml: the same for conda).
├── CITATION.cff
├── LICENSE                        MIT.
└── README.md
```

No experimental data are transcribed anywhere in the repository except the published α-function
constants and the critical constants and Twu parameters of the source compilation; every reference
value is computed at run time from the reference equations of state in CoolProp.

## Reproducing the results

Requires Python ≥ 3.11 and the pinned versions in `requirements.txt` (Python 3.13.16, CoolProp 8.0.0,
NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, matplotlib 3.11.2; `environment.yml` for conda). The scripts depend on each other in the order below,
which `run_all.py` follows:

```bash
pip install -r requirements.txt
python run_all.py                  # everything, about 30 min on one core (the in-sample fit takes 20-25 min)
```

or step by step:

```bash
python code/01_run_benchmark.py              # -> data/deviations_full.csv, fluids.csv, anchor_sweep.csv, key_numbers.json, run_metadata.json  (~2 min)
python code/04_stats.py                      # -> data/stats.json
python code/06_twufit_source_constants.py    # -> data/twufit_source_constants.csv/.json
python code/07_insample_fit.py               # -> data/insample_fit.csv, insample_fit_starts.csv, insample_fit_meta.json  (20-25 min; --quick for a smoke test)
python code/08_paired_statistics.py          # -> data/paired_comparisons.csv, comparison_families.json, signed_deviations*.csv
python code/09_insample_evaluation.py        # -> data/insample_*.csv/.json, holm_extended.csv, by_all_comparisons.csv
python code/10_pr78_shift_input_error.py     # -> data/pr78_*, volume_shift*, hvap_absolute*, compilation_constants*, input_sensitivities.csv
python code/11_physical_diagnostics.py       # -> data/physical_diagnostics.json, physics_*.csv
python code/02_make_tables.py                # -> tables/*.tex, tables/tables.json
python code/03_make_figures.py               # -> figures/*.pdf, *.png
python code/05_check_manuscript_numbers.py   # 775 checks against the numbers quoted in the texts; exit code 1 on any mismatch
```

To rebuild the Word files (needs `python-docx` and `pandoc`):

```bash
python manuscript/make_word.py     # -> submission/Main_Manuscript_Thermo.docx, Supplementary_Materials_Thermo.docx,
                                   #    submission/figures_numbered/, submission/Graphical_Abstract.png
```

The pipeline is deterministic (fixed bootstrap seed, 20260714, shared by scripts 04, 08 and 09); a
clean run reproduces the committed `data/`, `tables/` and `figures/`. The benchmark itself (01, 04, 06)
gives identical results under Python 3.11.15 with NumPy 2.4.4 and pandas 3.0.2 (the versions of the
submitted release); with other versions of SciPy the in-sample optimizer may stop at a slightly
different point in the flat region of its objective, which changes the fitted parameters in the third
decimal and the AADs by less than 0.001 points. `data/run_metadata.json`, `data/insample_fit_meta.json`
and `data/physical_diagnostics.json` record the versions used for the committed files.

Smoke test (about 3 min): `python run_all.py --quick` runs the benchmark, the statistics and the
in-sample fit for four fluids and writes `*_quick` files that are not used by the tables.

## Model naming

`<cubic>_<alpha>` with cubic ∈ {srk, pr} and alpha ∈ {s (Soave), s19 (Soave-19), t95 (Twu-95),
tc (Twu-c), coq (Coquelet), ms (MS), tf (Twu-fit)}; the in-sample fits are `sfit` (Soave-is),
`tfit` (Twu-is) and `tunc` (Twu-91 without the consistency constraints, diagnostic only);
translated variants carry the suffix `_vt` (density-anchored), `_vtc` (Rackett-anchored) or
`_vtz` (critical-matched, diagnostic only).

## What changed in v2.1.0

Revision after peer review: the in-sample fits (07, 09), the paired statistics with multiplicity
adjustment and the signed deviations (08), the PR78 variant, dimensionless shifts, absolute Δhvap
deviations and input-error analysis (10), the physical diagnostics (11), the comparative Table 1,
a two-panel Figure 7, Tables S16–S26 and `run_all.py`; `05_check_manuscript_numbers.py` now covers
the revised texts, and the Benjamini–Yekutieli adjustment is applied over all 242 comparisons (09). The benchmark itself (01, 04, 06) and the committed `deviations_full.csv` are
unchanged from the submitted version.

## Citation

> R. Alhasan, "An Error Budget for the SRK and PR Equations of State: α-Function Parameterization and
> Volume Translation in a Reproducible Pure-Fluid Saturation-Property Benchmark," *Thermo* (under
> review, 2026).

Software archive: https://doi.org/10.5281/zenodo.21370716 (concept DOI, resolves to the latest
archived version; the Zenodo record lists the version DOI of each release). Version DOI of v2.1.0
(this revision): to be added here when the release is archived.

## License

MIT; see [LICENSE](LICENSE).
