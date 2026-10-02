"""
Statistical analysis of the benchmark results.

The state points are not independent: the 40 points of one fluid lie on a
smooth deviation curve, so the effective sample size for pooled statistics is
the number of fluids, not the number of points. Three things are done here:

1. Confidence intervals for pooled AADs by a CLUSTER bootstrap: the fluids are
   resampled with replacement (B = 10000, fixed seed), the pooled AAD is
   recomputed over the points of the resampled fluids, and the 2.5/97.5
   percentiles give a 95% CI. (Implemented through per-fluid sums of |dev| and
   point counts, which is exactly equivalent to concatenating the resampled
   point sets and much faster.)

2. Per-fluid model comparisons ("model A beats model B for k of n fluids")
   with an exact two-sided binomial sign test at the fluid level: PR vs SRK
   at every alpha-function, every pair of alpha-functions within each cubic
   (21 pairs per cubic), and the translated-density comparisons, for all
   four properties. Ties (identical per-fluid AADs) are excluded from n;
   none occur in the present data. The p-values are nominal (no adjustment
   for multiple comparisons) and are reported as descriptive evidence.

3. The near-critical density mechanism: the signed saturated-liquid-density
   deviation at the highest grid temperature (Tr = 0.985) against the limiting
   value implied by the critical compressibility factors,
   100 (Zc_ref / Zc_model - 1): Pearson correlation and OLS slope/intercept.

Outputs ../data/stats.json.
"""

import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

sys.path.insert(0, HERE)
from eos import CubicEOS  # noqa: E402

B_BOOT = 10_000
SEED = 20260714

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))
meta = json.load(open(os.path.join(DATA, "run_metadata.json")))
TAGS = meta["models"]
fluids = sorted(df.fluid.unique())
n_f = len(fluids)
rng = np.random.default_rng(SEED)

# One shared set of bootstrap resamples (fluid indices) for every statistic.
IDX = rng.integers(0, n_f, size=(B_BOOT, n_f))


def aad(s):
    return float(np.nanmean(np.abs(s)))


def cluster_boot_ci(col, subset=None):
    d = df if subset is None else df[subset]
    g = d.groupby("fluid")[col]
    sums = g.apply(lambda s: np.nansum(np.abs(s))).reindex(fluids).to_numpy()
    cnts = g.apply(lambda s: int(np.isfinite(s).sum())).reindex(fluids).to_numpy(dtype=float)
    stats = sums[IDX].sum(axis=1) / cnts[IDX].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(aad=aad(d[col]), ci_lo=float(lo), ci_hi=float(hi))


# ------------------------------------------------------------------ #
# 1. Cluster-bootstrap 95% CIs for every pooled AAD used in the paper
CI = {}
for t in TAGS:
    for prop, col in (("psat", "dev_p"), ("h", "dev_h"), ("rho", "dev_rho"), ("rhov", "dev_rhov")):
        CI[f"{prop}_{t}"] = cluster_boot_ci(f"{col}_{t}")
    for sfx in ("vt", "vtc"):
        CI[f"rho_{t}_{sfx}"] = cluster_boot_ci(f"dev_rho_{t}_{sfx}")
        CI[f"rhov_{t}_{sfx}"] = cluster_boot_ci(f"dev_rhov_{t}_{sfx}")
# pooled below Tr = 0.90 (the practically dominant range) for the ladder
below = df.Tr < 0.90
CI_BELOW090 = {}
for t in ("srk_s", "pr_s"):
    CI_BELOW090[f"rho_{t}"] = cluster_boot_ci(f"dev_rho_{t}", below)
    for sfx in ("vt", "vtc"):
        CI_BELOW090[f"rho_{t}_{sfx}"] = cluster_boot_ci(f"dev_rho_{t}_{sfx}", below)

# ------------------------------------------------------------------ #
# 2. Exact two-sided binomial sign tests on per-fluid AADs
def per_fluid_aad(col):
    return {f: aad(df.loc[df.fluid == f, col]) for f in fluids}


def sign_test(col_a, col_b):
    """Exact two-sided sign test that model A has lower per-fluid AAD than B."""
    A, Bv = per_fluid_aad(col_a), per_fluid_aad(col_b)
    wins = sum(A[f] < Bv[f] for f in fluids)
    ties = sum(A[f] == Bv[f] for f in fluids)
    n = n_f - ties
    k = max(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)
    return dict(wins=int(wins), n=int(n), p_two_sided=float(p))


PROPCOL = {"psat": "dev_p", "h": "dev_h", "rho": "dev_rho", "rhov": "dev_rhov"}
ALPHAS = ["s", "s19", "t95", "tc", "coq", "ms", "tf"]
comparisons = []
for prop in ("psat", "h", "rho", "rhov"):
    # cubic-vs-cubic at fixed alpha
    for a in ALPHAS:
        comparisons.append((prop, f"pr_{a}", f"srk_{a}"))
    # every pair of alpha-functions within each cubic (later alpha vs earlier alpha)
    for c in ("srk", "pr"):
        for j, a in enumerate(ALPHAS):
            for b in ALPHAS[:j]:
                comparisons.append((prop, f"{c}_{a}", f"{c}_{b}"))
tests = {}
for prop, a, b in comparisons:
    tests[f"{prop}:{a}_vs_{b}"] = sign_test(f"{PROPCOL[prop]}_{a}", f"{PROPCOL[prop]}_{b}")
n_ties = sum(n_f - v["n"] for v in tests.values())
# translated-density comparisons (classical Soave alpha)
tests["rho:pr_s_vt_vs_srk_s_vt"] = sign_test("dev_rho_pr_s_vt", "dev_rho_srk_s_vt")
tests["rho:pr_s_vtc_vs_srk_s_vtc"] = sign_test("dev_rho_pr_s_vtc", "dev_rho_srk_s_vtc")
tests["rho:pr_s_vtc_vs_pr_s"] = sign_test("dev_rho_pr_s_vtc", "dev_rho_pr_s")
tests["rho:srk_s_vtc_vs_srk_s"] = sign_test("dev_rho_srk_s_vtc", "dev_rho_srk_s")
tests["rho:pr_s_vt_vs_pr_s_vtc"] = sign_test("dev_rho_pr_s_vt", "dev_rho_pr_s_vtc")
tests["rho:srk_s_vt_vs_srk_s_vtc"] = sign_test("dev_rho_srk_s_vt", "dev_rho_srk_s_vtc")
tests["rhov:pr_s_vt_vs_pr_s"] = sign_test("dev_rhov_pr_s_vt", "dev_rhov_pr_s")
tests["rhov:srk_s_vt_vs_srk_s"] = sign_test("dev_rhov_srk_s_vt", "dev_rhov_srk_s")
tests["rhov:pr_s_vtc_vs_pr_s"] = sign_test("dev_rhov_pr_s_vtc", "dev_rhov_pr_s")
tests["rhov:srk_s_vtc_vs_srk_s"] = sign_test("dev_rhov_srk_s_vtc", "dev_rhov_srk_s")
n_ties += sum(n_f - tests[k]["n"] for k in tests if "_vt" in k)

# ------------------------------------------------------------------ #
# 3. Near-critical density deviation vs the Zc mismatch (classical alpha)
Zc_model = {"srk": CubicEOS.PARAMS["SRK"]["Zc"], "pr": CubicEOS.PARAMS["PR"]["Zc"]}
top = df.loc[df.groupby("fluid").Tr.idxmax()].set_index("fluid")  # Tr = 0.985 rows
zc = fl.set_index("fluid").Zc_ref

near_crit = {}
for cub in ("srk", "pr"):
    x = np.array([100.0 * (zc[f] / Zc_model[cub] - 1.0) for f in fluids])  # limit
    y = np.array([top.loc[f, f"dev_rho_{cub}_s"] for f in fluids])         # observed
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    near_crit[cub] = dict(
        pearson_r=r, ols_slope=float(slope), ols_intercept=float(intercept),
        x_limit_min=float(x.min()), x_limit_max=float(x.max()),
        resid_min=float((y - x).min()), resid_max=float((y - x).max()),
        x_limit=dict(zip(fluids, x.round(3))), y_obs=dict(zip(fluids, y.round(3))),
    )

# ------------------------------------------------------------------ #
# 4. Generalized-vs-fitted Twu parameters: how far the generalized consistent
#    correlations L(omega), M(omega) sit from the component-specific values of
#    the 1800-fluid compilation for the benchmark fluids (a check on the
#    transcribed correlations and a measure of the generalization loss).
gen_vs_fit = {}
for cub in ("srk", "pr"):
    dL = fl[f"L_tc_{cub}"] - fl[f"L_tf_{cub}"]
    dM = fl[f"M_tc_{cub}"] - fl[f"M_tf_{cub}"]
    gen_vs_fit[cub] = dict(
        mean_abs_dL=float(dL.abs().mean()), mean_abs_dM=float(dM.abs().mean()),
        corr_L=float(np.corrcoef(fl[f"L_tc_{cub}"], fl[f"L_tf_{cub}"])[0, 1]),
        corr_M=float(np.corrcoef(fl[f"M_tc_{cub}"], fl[f"M_tf_{cub}"])[0, 1]),
        N_fit_min=float(fl[f"N_tf_{cub}"].min()), N_fit_max=float(fl[f"N_tf_{cub}"].max()),
    )

out = dict(seed=SEED, B=B_BOOT, n_fluids=n_f, ci=CI, ci_below_090=CI_BELOW090,
           sign_tests=tests, n_sign_tests=len(tests), n_ties_total=int(n_ties),
           Zc_model=Zc_model, near_critical=near_crit,
           generalized_vs_fitted_twu=gen_vs_fit)
with open(os.path.join(DATA, "stats.json"), "w") as f:
    json.dump(out, f, indent=2)

for k, v in CI.items():
    if k.startswith(("psat_", "h_")):
        print(f"{k:14s} {v['aad']:.2f}  ({v['ci_lo']:.2f}-{v['ci_hi']:.2f})")
for k, v in tests.items():
    if k.startswith("psat"):
        print(f"{k:28s} wins {v['wins']}/{v['n']}  p = {v['p_two_sided']:.2g}")
for cub in ("srk", "pr"):
    nc = near_crit[cub]
    print(f"near-critical {cub}: r = {nc['pearson_r']:.3f}, slope = {nc['ols_slope']:.2f}, "
          f"intercept = {nc['ols_intercept']:.2f}")
print(json.dumps(gen_vs_fit, indent=1))
print(f"{len(tests)} sign tests, {n_ties} ties in total")
print("Wrote stats.json")
