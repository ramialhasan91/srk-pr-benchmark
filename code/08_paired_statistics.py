"""Paired statistics (Section 2.5): paired fluid-level bootstrap of the pooled-AAD difference for the 206
comparisons of the predictive models, Holm adjustment within their families F1-F3, and mean signed
deviations by model, property, range and family. 09_insample_evaluation.py adds the in-sample
comparisons to F1 and F2, recomputes the Holm adjustment within the final families and the
Benjamini-Yekutieli adjustment over all 242 comparisons; the tables use those.

Reads the committed outputs (deviations_full.csv, fluids.csv, run_metadata.json, stats.json) and
reproduces the bootstrap index matrix of 04_stats.py (same seed, same B), so that the paired intervals
use exactly the resamples of the per-model intervals.
Writes ../data/paired_comparisons.csv, signed_deviations.csv, signed_deviations_per_fluid.csv,
       comparison_families.json
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

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))
meta = json.load(open(os.path.join(DATA, "run_metadata.json")))
st = json.load(open(os.path.join(DATA, "stats.json")))
TAGS = meta["models"]
fluids = sorted(df.fluid.unique())
n_f = len(fluids)

B_BOOT, SEED = st["B"], st["seed"]
rng = np.random.default_rng(SEED)
IDX = rng.integers(0, n_f, size=(B_BOOT, n_f))


def aad(s):
    return float(np.nanmean(np.abs(s)))


def sums_counts(col, subset=None):
    d = df if subset is None else df[subset]
    g = d.groupby("fluid")[col]
    sums = g.apply(lambda s: np.nansum(np.abs(s))).reindex(fluids).to_numpy()
    cnts = g.apply(lambda s: int(np.isfinite(s).sum())).reindex(fluids).to_numpy(dtype=float)
    return sums, cnts


def boot_stats(col, subset=None):
    sums, cnts = sums_counts(col, subset)
    return sums[IDX].sum(axis=1) / cnts[IDX].sum(axis=1)


# --- 0. reproduce the per-model CIs of stats.json (check that IDX is identical)
chk = []
for key in ("psat_pr_s", "psat_srk_s", "psat_pr_t95", "h_srk_tf", "rho_pr_s_vt", "rhov_srk_s"):
    prop, rest = key.split("_", 1)
    col = {"psat": "dev_p", "h": "dev_h", "rho": "dev_rho", "rhov": "dev_rhov"}[prop] + "_" + rest
    b = boot_stats(col)
    lo, hi = np.percentile(b, [2.5, 97.5])
    chk.append((key, st["ci"][key]["ci_lo"], float(lo), st["ci"][key]["ci_hi"], float(hi)))
max_ci_diff = max(max(abs(a - b), abs(c - d)) for _, a, b, c, d in chk)
print("CI reproduction check, max |diff| =", max_ci_diff)
assert max_ci_diff < 1e-9, "bootstrap resamples differ from 04_stats.py"

# --- 1. the 206 comparisons, in the order of 04_stats.py
PROPCOL = {"psat": "dev_p", "h": "dev_h", "rho": "dev_rho", "rhov": "dev_rhov"}
ALPHAS = ["s", "s19", "t95", "tc", "coq", "ms", "tf"]
comparisons = []
for prop in ("psat", "h", "rho", "rhov"):
    for a in ALPHAS:
        comparisons.append((prop, f"pr_{a}", f"srk_{a}"))
    for c in ("srk", "pr"):
        for j, a in enumerate(ALPHAS):
            for b in ALPHAS[:j]:
                comparisons.append((prop, f"{c}_{a}", f"{c}_{b}"))
vt_pairs = [("rho", "pr_s_vt", "srk_s_vt"), ("rho", "pr_s_vtc", "srk_s_vtc"), ("rho", "pr_s_vtc", "pr_s"),
            ("rho", "srk_s_vtc", "srk_s"), ("rho", "pr_s_vt", "pr_s_vtc"), ("rho", "srk_s_vt", "srk_s_vtc"),
            ("rhov", "pr_s_vt", "pr_s"), ("rhov", "srk_s_vt", "srk_s"), ("rhov", "pr_s_vtc", "pr_s"),
            ("rhov", "srk_s_vtc", "srk_s")]
comparisons += vt_pairs
assert len(comparisons) == 206


def colname(prop, tag):
    parts = tag.split("_")
    base = f"{PROPCOL[prop]}_{parts[0]}_{parts[1]}"
    return base + (f"_{parts[2]}" if len(parts) == 3 else "")


def per_fluid_aad(col):
    return {f: aad(df.loc[df.fluid == f, col]) for f in fluids}


def sign_test(col_a, col_b):
    A, Bv = per_fluid_aad(col_a), per_fluid_aad(col_b)
    wins = sum(A[f] < Bv[f] for f in fluids)
    ties = sum(A[f] == Bv[f] for f in fluids)
    n = n_f - ties
    k = max(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)
    return int(wins), int(n), float(p)


boot_cache = {}


def boot(col):
    if col not in boot_cache:
        boot_cache[col] = boot_stats(col)
    return boot_cache[col]


rows = []
for prop, a, b in comparisons:
    ca, cb = colname(prop, a), colname(prop, b)
    key = f"{prop}:{a}_vs_{b}"
    wins, n, p = sign_test(ca, cb)
    s_old = st["sign_tests"][key]
    assert (wins, n) == (s_old["wins"], s_old["n"]) and abs(p - s_old["p_two_sided"]) < 1e-12, key
    aad_a, aad_b = aad(df[ca]), aad(df[cb])
    d_boot = boot(ca) - boot(cb)
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    # per-fluid paired differences (for the median and the sign of the per-fluid effect)
    pfa, pfb = per_fluid_aad(ca), per_fluid_aad(cb)
    pf_diff = np.array([pfa[f] - pfb[f] for f in fluids])
    rows.append(dict(key=key, prop=prop, A=a, B=b, aad_A=aad_a, aad_B=aad_b, dAAD=aad_a - aad_b,
                     ci_lo=float(lo), ci_hi=float(hi), ci_excludes_zero=bool(lo > 0 or hi < 0),
                     boot_sd=float(np.std(d_boot, ddof=1)),
                     pf_diff_median=float(np.median(pf_diff)), pf_diff_min=float(pf_diff.min()),
                     pf_diff_max=float(pf_diff.max()),
                     wins=wins, n=n, p=p))
res = pd.DataFrame(rows).set_index("key")

# --- 2. multiplicity adjustment
def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def benjamini_yekutieli(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    cm = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(1.0, m * cm * p[i] / (rank + 1))
        running = min(running, val)
        adj[i] = running
    return adj


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(1.0, m * p[i] / (rank + 1))
        running = min(running, val)
        adj[i] = running
    return adj


F1 = [f"psat:pr_{a}_vs_srk_{a}" for a in ALPHAS] + [f"h:pr_{a}_vs_srk_{a}" for a in ALPHAS] + \
     ["rho:pr_s_vs_srk_s", "rho:pr_s_vt_vs_srk_s_vt"]
F2_table3 = ["psat:srk_s19_vs_srk_s", "psat:pr_s19_vs_pr_s", "psat:srk_t95_vs_srk_s", "psat:pr_t95_vs_pr_s",
             "psat:srk_tc_vs_srk_t95", "psat:pr_tc_vs_pr_t95", "psat:pr_tc_vs_pr_s", "psat:srk_ms_vs_srk_t95",
             "psat:pr_ms_vs_pr_s", "psat:pr_ms_vs_pr_t95", "psat:srk_coq_vs_srk_s", "psat:srk_coq_vs_srk_t95",
             "psat:pr_coq_vs_pr_s", "psat:srk_tf_vs_srk_t95", "psat:srk_tf_vs_srk_ms", "psat:pr_tf_vs_pr_s",
             "psat:pr_tf_vs_pr_ms", "h:srk_t95_vs_srk_s", "h:pr_t95_vs_pr_s"]
F2_text_only = ["psat:srk_ms_vs_srk_s", "psat:srk_tf_vs_srk_s"]
F2 = F2_table3 + F2_text_only
F3 = ["rho:pr_s_vtc_vs_pr_s", "rho:srk_s_vtc_vs_srk_s", "rho:pr_s_vt_vs_pr_s_vtc", "rhov:pr_s_vt_vs_pr_s"]
assert len(F1) == 16 and len(F2) == 21 and len(F3) == 4
for fam_name, fam in (("F1", F1), ("F2", F2), ("F3", F3)):
    res.loc[fam, "family"] = fam_name
    res.loc[fam, "p_holm_family"] = holm(res.loc[fam, "p"].to_numpy())
res["family"] = res["family"].fillna("descriptive")
# sensitivity variants
table3 = F1 + F2_table3 + F3
res.loc[table3, "p_holm_table3_single_family"] = holm(res.loc[table3, "p"].to_numpy())
res["p_holm_all206"] = holm(res["p"].to_numpy())
res["q_BY_all206"] = benjamini_yekutieli(res["p"].to_numpy())
res["q_BH_all206"] = benjamini_hochberg(res["p"].to_numpy())
res["sig_nominal"] = res.p < 0.05
res["sig_holm_family"] = res.p_holm_family < 0.05
res["sig_holm_table3"] = res.p_holm_table3_single_family < 0.05
res["sig_holm_all206"] = res.p_holm_all206 < 0.05
res["sig_BY_all206"] = res.q_BY_all206 < 0.05
res.to_csv(os.path.join(DATA, "paired_comparisons.csv"))

# --- 3. mean signed deviations (bias)
BIN_EDGES = [0.50, 0.70, 0.90, 0.9851]
BIN_LABELS = ["0.50-0.70", "0.70-0.90", "0.90-0.985"]
df["Tr_bin"] = pd.cut(df.Tr, BIN_EDGES, right=False, labels=BIN_LABELS)
FAMILY_ORDER = ["Inorganic gases", "Alkanes", "Aromatics", "Polar/associating", "Refrigerants"]
subsets = {"All": np.ones(len(df), bool), "< 0.90": (df.Tr < 0.90).to_numpy()}
for b in BIN_LABELS:
    subsets[b] = (df.Tr_bin == b).to_numpy()
for fam in FAMILY_ORDER:
    subsets[fam] = (df.family == fam).to_numpy()
for sg in ("H-bonding", "polar aprotic"):
    subsets[sg] = (df.subgroup == sg).to_numpy()

model_cols = []
for t in TAGS:
    for prop, col in PROPCOL.items():
        model_cols.append((prop, t, f"{col}_{t}"))
for t in ("srk_s", "pr_s"):
    for sfx in ("vtc", "vt"):
        model_cols.append(("rho", f"{t}_{sfx}", f"dev_rho_{t}_{sfx}"))
        model_cols.append(("rhov", f"{t}_{sfx}", f"dev_rhov_{t}_{sfx}"))
bias_rows = []
for prop, tag, col in model_cols:
    rec = dict(prop=prop, model=tag)
    for name, mask in subsets.items():
        rec[f"bias|{name}"] = float(np.nanmean(df.loc[mask, col]))
        rec[f"aad|{name}"] = float(np.nanmean(np.abs(df.loc[mask, col])))
    # fluid-level bootstrap CI of the overall bias (same resamples)
    g = df.groupby("fluid")[col]
    sums = g.apply(lambda s: np.nansum(s)).reindex(fluids).to_numpy()
    cnts = g.apply(lambda s: int(np.isfinite(s).sum())).reindex(fluids).to_numpy(dtype=float)
    bb = sums[IDX].sum(axis=1) / cnts[IDX].sum(axis=1)
    rec["bias_ci_lo|All"], rec["bias_ci_hi|All"] = [float(x) for x in np.percentile(bb, [2.5, 97.5])]
    # share of points with positive deviation (overprediction)
    rec["frac_positive|All"] = float(np.nanmean(df[col] > 0))
    bias_rows.append(rec)
bias = pd.DataFrame(bias_rows)
bias.to_csv(os.path.join(DATA, "signed_deviations.csv"), index=False)

# per-fluid signed mean deviations (psat and h), all 14 models — for the SI if wanted
pf_bias = df.groupby("fluid")[[f"dev_p_{t}" for t in TAGS] + [f"dev_h_{t}" for t in TAGS]].mean()
pf_bias.to_csv(os.path.join(DATA, "signed_deviations_per_fluid.csv"))

# --- 4. summary printout
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
cols = ["family", "aad_A", "aad_B", "dAAD", "ci_lo", "ci_hi", "wins", "n", "p", "p_holm_family",
        "p_holm_table3_single_family", "p_holm_all206", "q_BY_all206"]
print(res.loc[F1 + F2 + F3, cols].round(4).to_string())
print()
print("nominal p<0.05 in F1/F2/F3 that lose significance after Holm (family):")
sel = res[(res.family != "descriptive") & res.sig_nominal & ~res.sig_holm_family]
print(sel[cols].round(4).to_string())
print()
print("counts: nominal sig all206 =", int(res.sig_nominal.sum()), "| Holm all206 =", int(res.sig_holm_all206.sum()),
      "| BY all206 =", int(res.sig_BY_all206.sum()), "| BH all206 =", int((res.q_BH_all206 < 0.05).sum()))
print("paired CI excludes zero (all 206):", int(res.ci_excludes_zero.sum()))
print("agreement sign-test nominal vs CI-excludes-zero:", int((res.sig_nominal == res.ci_excludes_zero).sum()), "/ 206")
json.dump(dict(F1=F1, F2=F2, F2_table3=F2_table3, F2_text_only=F2_text_only, F3=F3, seed=SEED, B=B_BOOT),
          open(os.path.join(DATA, "comparison_families.json"), "w"), indent=1)
print("wrote paired_comparisons.csv, signed_deviations.csv, signed_deviations_per_fluid.csv")
