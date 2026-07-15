"""
Statistical analysis of the benchmark results.

The 1280 state points are not independent: the 40 points of one fluid lie on
a smooth deviation curve, so the effective sample size for pooled statistics
is the number of fluids (32), not the number of points. Two consequences are
handled here:

1. Confidence intervals for pooled AADs are computed by a CLUSTER bootstrap:
   the 32 fluids are resampled with replacement (B = 10000), the pooled AAD
   is recomputed over the points of the resampled fluids, and the 2.5/97.5
   percentiles give a 95% CI.

2. Per-fluid model comparisons ("model A beats model B for k of 32 fluids")
   are tested with an exact two-sided binomial sign test at the fluid level.

3. The near-critical density mechanism: for each fluid, the signed
   saturated-liquid-density deviation at the highest grid temperature
   (Tr = 0.985) is compared with the limiting value implied by the critical
   compressibility factors, 100*(Zc_ref/Zc_model - 1), including a Pearson
   correlation and an ordinary least-squares slope.

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
fluids = sorted(df.fluid.unique())
rng = np.random.default_rng(SEED)


def aad(s):
    return float(np.nanmean(np.abs(s)))


# ------------------------------------------------------------------ #
# 1. Cluster-bootstrap 95% CIs for the pooled AADs
def cluster_boot_ci(col):
    groups = {f: df.loc[df.fluid == f, col].to_numpy() for f in fluids}
    point = aad(df[col])
    stats = np.empty(B_BOOT)
    for i in range(B_BOOT):
        sample = rng.choice(fluids, size=len(fluids), replace=True)
        pooled = np.concatenate([groups[f] for f in sample])
        stats[i] = np.nanmean(np.abs(pooled))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(aad=point, ci_lo=float(lo), ci_hi=float(hi))


CI_COLS = {
    "psat_srk": "dev_p_srk", "psat_pr": "dev_p_pr",
    "psat_srktwu": "dev_p_srktwu", "psat_prtwu": "dev_p_prtwu",
    "rho_srk": "dev_rho_srk", "rho_pr": "dev_rho_pr", "rho_prvt": "dev_rho_prvt",
    "rho_prvtc": "dev_rho_prvtc",
    "rhov_srk": "dev_rhov_srk", "rhov_pr": "dev_rhov_pr",
    "rhov_srktwu": "dev_rhov_srktwu", "rhov_prtwu": "dev_rhov_prtwu",
    "rhov_prvt": "dev_rhov_prvt",
    "h_srk": "dev_h_srk", "h_pr": "dev_h_pr",
    "h_srktwu": "dev_h_srktwu", "h_prtwu": "dev_h_prtwu",
}
ci = {name: cluster_boot_ci(col) for name, col in CI_COLS.items()}

# ------------------------------------------------------------------ #
# 2. Exact two-sided binomial sign tests on per-fluid AADs
def per_fluid_aad(col):
    return {f: aad(df.loc[df.fluid == f, col]) for f in fluids}


def sign_test(col_a, col_b):
    """Exact two-sided sign test that model A has lower per-fluid AAD than B."""
    A, Bv = per_fluid_aad(col_a), per_fluid_aad(col_b)
    wins = sum(A[f] < Bv[f] for f in fluids)
    ties = sum(A[f] == Bv[f] for f in fluids)
    n = len(fluids) - ties
    k = max(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)
    return dict(wins=int(wins), n=int(n), p_two_sided=float(p))


tests = {
    "psat_pr_vs_srk": sign_test("dev_p_pr", "dev_p_srk"),
    "psat_srktwu_vs_srk": sign_test("dev_p_srktwu", "dev_p_srk"),
    "psat_prtwu_vs_pr": sign_test("dev_p_prtwu", "dev_p_pr"),
    "psat_prtwu_vs_srktwu": sign_test("dev_p_prtwu", "dev_p_srktwu"),
    "h_pr_vs_srk": sign_test("dev_h_pr", "dev_h_srk"),
    "h_srktwu_vs_srk": sign_test("dev_h_srktwu", "dev_h_srk"),
    "h_prtwu_vs_pr": sign_test("dev_h_prtwu", "dev_h_pr"),
    "rho_pr_vs_srk": sign_test("dev_rho_pr", "dev_rho_srk"),
    "rhov_pr_vs_srk": sign_test("dev_rhov_pr", "dev_rhov_srk"),
    "rhov_prtwu_vs_pr": sign_test("dev_rhov_prtwu", "dev_rhov_pr"),
}

# ------------------------------------------------------------------ #
# 3. Near-critical density deviation vs the Zc mismatch
Zc_model = {"srk": CubicEOS.PARAMS["SRK"]["Zc"], "pr": CubicEOS.PARAMS["PR"]["Zc"]}
top = df.loc[df.groupby("fluid").Tr.idxmax()].set_index("fluid")  # Tr = 0.985 rows
zc = fl.set_index("fluid").Zc_ref

near_crit = {}
for tag in ("srk", "pr"):
    x = np.array([100.0 * (zc[f] / Zc_model[tag] - 1.0) for f in fluids])  # limit
    y = np.array([top.loc[f, f"dev_rho_{tag}"] for f in fluids])           # observed
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    near_crit[tag] = dict(
        pearson_r=r, ols_slope=float(slope), ols_intercept=float(intercept),
        x_limit=dict(zip(fluids, x.round(3))), y_obs=dict(zip(fluids, y.round(3))),
    )

out = dict(seed=SEED, B=B_BOOT, ci=ci, sign_tests=tests,
           Zc_model=Zc_model, near_critical=near_crit)
with open(os.path.join(DATA, "stats.json"), "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps({k: v for k, v in out.items() if k != "near_critical"}, indent=2))
for tag in ("srk", "pr"):
    nc = near_crit[tag]
    print(f"near-critical {tag}: r = {nc['pearson_r']:.3f}, "
          f"slope = {nc['ols_slope']:.2f}, intercept = {nc['ols_intercept']:.2f}")
print("Wrote stats.json")
