"""Evaluates the in-sample optima of 07_insample_fit.py with eos.py, computes their statistics with the
machinery of 08_paired_statistics.py (same resamples, sign tests, paired intervals, Holm within the
final families, Benjamini-Yekutieli over all 242 comparisons) and the two-way decomposition of the
per-fluid log-AAD.

Reads  ../data/insample_fit.csv (or the file given as first argument), deviations_full.csv, fluids.csv,
       stats.json, key_numbers.json, comparison_families.json, paired_comparisons.csv
Writes ../data/insample_points.csv, insample_pooled.csv, insample_per_fluid.csv, insample_comparisons.csv,
       holm_extended.csv, by_all_comparisons.csv, insample_decomposition.json, insample_alpha_closeness.csv,
       insample_verification.json
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
FITS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "insample_fit.csv")
from eos import R, CubicEOS  # noqa: E402

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv")).set_index("fluid")
st = json.load(open(os.path.join(DATA, "stats.json")))
key = json.load(open(os.path.join(DATA, "key_numbers.json")))
fits = pd.read_csv(FITS)
fam = json.load(open(os.path.join(DATA, "comparison_families.json")))
wp2 = pd.read_csv(os.path.join(DATA, "paired_comparisons.csv")).set_index("key")
ALPHAS = ["s", "s19", "t95", "tc", "coq", "ms", "tf"]
NEW = ["sfit", "tfit", "tunc"]
PROPCOL = {"psat": "dev_p", "h": "dev_h", "rho": "dev_rho", "rhov": "dev_rhov"}
FAMILY_ORDER = ["Inorganic gases", "Alkanes", "Aromatics", "Polar/associating", "Refrigerants"]
fluids_all = sorted(df.fluid.unique())
fluids_fit = sorted(fits.fluid.unique())
quick = len(fluids_fit) < len(fluids_all)


def pct(model, ref):
    return (model - ref) / ref * 100.0


def aad(s):
    return float(np.nanmean(np.abs(s)))


# ------------------------------------------------------------------ 1. evaluate the optima with eos.py
CLAP_H = 1e-4
rows, clap = [], {t: [] for t in [f"{c}_{n}" for c in ("srk", "pr") for n in NEW]}
for fluid in fluids_fit:
    g = df[df.fluid == fluid].sort_values("Tr")
    Tc, pc, w = fl.loc[fluid, "Tc_K"], fl.loc[fluid, "pc_MPa"] * 1e6, fl.loc[fluid, "omega"]
    models = {}
    for cubic in ("SRK", "PR"):
        f = fits[(fits.fluid == fluid) & (fits.cubic == cubic)].iloc[0]
        cl = cubic.lower()
        m = CubicEOS(cubic, Tc, pc, w, alpha="s")
        m.m = float(f.m_fit)
        models[f"{cl}_sfit"] = m
        models[f"{cl}_tfit"] = CubicEOS(cubic, Tc, pc, w, alpha="tf", twu_params=(f.L_fit, f.M_fit, f.N_fit))
        if np.isfinite(f.get("L_unc", np.nan)):
            models[f"{cl}_tunc"] = CubicEOS(cubic, Tc, pc, w, alpha="tf", twu_params=(f.L_unc, f.M_unc, f.N_unc))
    for _, r in g.iterrows():
        T = r.T_K
        rec = dict(fluid=fluid, family=r.family, subgroup=r.subgroup, Tr=r.Tr, T_K=T)
        for tag, m in models.items():
            p, Zl, Zv = m.psat(T)
            if not np.isfinite(p):
                for prop, col in PROPCOL.items():
                    rec[f"{col}_{tag}"] = np.nan
                continue
            v_l, v_v = Zl * R * T / p, Zv * R * T / p
            h = m.h_res(T, p, Zv) - m.h_res(T, p, Zl)
            rec[f"dev_p_{tag}"] = pct(p, r.p_ref_Pa)
            rec[f"dev_rho_{tag}"] = pct(1.0 / v_l, r.rho_ref_molm3)
            rec[f"dev_rhov_{tag}"] = pct(1.0 / v_v, r.rhov_ref_molm3)
            rec[f"dev_h_{tag}"] = pct(h, r.hvap_ref_Jmol)
            rec[f"hvap_{tag}_Jmol"] = h
            p_lo, _, _ = m.psat(T - CLAP_H * T)
            p_hi, _, _ = m.psat(T + CLAP_H * T)
            if np.isfinite(p_lo) and np.isfinite(p_hi):
                clap[tag].append(abs(T * (v_v - v_l) * (p_hi - p_lo) / (2.0 * CLAP_H * T) / h - 1.0))
        rows.append(rec)
new = pd.DataFrame(rows)
new.to_csv(os.path.join(DATA, "insample_points.csv"), index=False)
clap_max = {t: (float(np.max(v)) if v else np.nan) for t, v in clap.items()}
print("Clapeyron check at the optima (max rel diff):", {k: f"{v:.1e}" for k, v in clap_max.items()})

# reproduction of the fit script's AADs (vectorized solver) by eos.py
chk = []
for fluid in fluids_fit:
    for cubic in ("SRK", "PR"):
        f = fits[(fits.fluid == fluid) & (fits.cubic == cubic)].iloc[0]
        cl = cubic.lower()
        g = new[new.fluid == fluid]
        chk.append(abs(aad(g[f"dev_p_{cl}_sfit"]) - f.psat_aad_sfit))
        chk.append(abs(aad(g[f"dev_p_{cl}_tfit"]) - f.psat_aad_tfit))
        chk.append(abs(aad(g[f"dev_h_{cl}_tfit"]) - f.h_aad_tfit))
print("max |AAD(eos.py) - AAD(vectorized)| =", max(chk))

# merged point table: committed 14 combinations + new models
dfm = df.merge(new.drop(columns=["family", "subgroup", "T_K"]), on=["fluid", "Tr"], how="inner")
TAGS = key["models"] + [f"{c}_{n}" for c in ("srk", "pr") for n in NEW if f"dev_p_{c}_{n}" in new.columns]
fluids = sorted(dfm.fluid.unique())
n_f = len(fluids)

# ------------------------------------------------------------------ 2. bootstrap machinery (as in 08_paired_statistics.py)
rng = np.random.default_rng(st["seed"])
IDX = rng.integers(0, len(fluids_all), size=(st["B"], len(fluids_all)))
if quick:   # restrict resamples to the fitted fluids (diagnostic runs only)
    sel = [fluids_all.index(f) for f in fluids]
    rng = np.random.default_rng(st["seed"])
    IDX = rng.integers(0, n_f, size=(st["B"], n_f))


def sums_counts(col, subset=None):
    d = dfm if subset is None else dfm[subset]
    g = d.groupby("fluid")[col]
    sums = g.apply(lambda s: np.nansum(np.abs(s))).reindex(fluids).to_numpy()
    cnts = g.apply(lambda s: int(np.isfinite(s).sum())).reindex(fluids).to_numpy(dtype=float)
    return sums, cnts


boot_cache = {}


def boot(col, subset=None, keyname=None):
    k = keyname or col
    if k not in boot_cache:
        s, c = sums_counts(col, subset)
        boot_cache[k] = s[IDX].sum(axis=1) / c[IDX].sum(axis=1)
    return boot_cache[k]


def ci_of(col, subset=None, keyname=None):
    b = boot(col, subset, keyname)
    lo, hi = np.percentile(b, [2.5, 97.5])
    d = dfm if subset is None else dfm[subset]
    return dict(aad=aad(d[col]), ci_lo=float(lo), ci_hi=float(hi))


def per_fluid_aad(col):
    return {f: aad(dfm.loc[dfm.fluid == f, col]) for f in fluids}


def sign_test(col_a, col_b):
    A, Bv = per_fluid_aad(col_a), per_fluid_aad(col_b)
    wins = sum(A[f] < Bv[f] for f in fluids)
    ties = sum(A[f] == Bv[f] for f in fluids)
    n = n_f - ties
    k = max(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)
    return int(wins), int(n), float(p)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# check: reproduce a paired CI of 08_paired_statistics.py with the merged frame (full runs only)
if not quick:
    c = ci_of("dev_p_pr_s")
    assert abs(c["ci_lo"] - st["ci"]["psat_pr_s"]["ci_lo"]) < 1e-9 and abs(c["ci_hi"] - st["ci"]["psat_pr_s"]["ci_hi"]) < 1e-9

# ------------------------------------------------------------------ 3. pooled statistics of the new models
BIN_EDGES = [0.50, 0.70, 0.90, 0.9851]
BIN_LABELS = ["0.50-0.70", "0.70-0.90", "0.90-0.985"]
dfm["Tr_bin"] = pd.cut(dfm.Tr, BIN_EDGES, right=False, labels=BIN_LABELS)
subsets = {"All": np.ones(len(dfm), bool), "< 0.90": (dfm.Tr < 0.90).to_numpy()}
for b in BIN_LABELS:
    subsets[b] = (dfm.Tr_bin == b).to_numpy()
for fm in FAMILY_ORDER:
    subsets[fm] = (dfm.family == fm).to_numpy()
for sg in ("H-bonding", "polar aprotic"):
    subsets[sg] = (dfm.subgroup == sg).to_numpy()
pooled = []
for t in TAGS:
    for prop, col in PROPCOL.items():
        rec = dict(model=t, prop=prop)
        for name, mask in subsets.items():
            rec[f"aad|{name}"] = aad(dfm.loc[mask, f"{col}_{t}"])
            rec[f"bias|{name}"] = float(np.nanmean(dfm.loc[mask, f"{col}_{t}"]))
        c = ci_of(f"{col}_{t}")
        rec["ci_lo|All"], rec["ci_hi|All"] = c["ci_lo"], c["ci_hi"]
        pooled.append(rec)
pooled = pd.DataFrame(pooled)
pooled.to_csv(os.path.join(DATA, "insample_pooled.csv"), index=False)
perfluid = []
for f in fluids:
    rec = dict(fluid=f, family=fl.loc[f, "family"], subgroup=fl.loc[f, "subgroup"], omega=fl.loc[f, "omega"])
    g = dfm[dfm.fluid == f]
    for t in TAGS:
        for prop, col in PROPCOL.items():
            rec[f"{prop}_{t}"] = aad(g[f"{col}_{t}"])
    perfluid.append(rec)
perfluid = pd.DataFrame(perfluid).set_index("fluid")
perfluid.to_csv(os.path.join(DATA, "insample_per_fluid.csv"))

# ------------------------------------------------------------------ 4. paired comparisons
comps = []
# F1 additions: cubic form at the in-sample optimum
for prop in ("psat", "h", "rho", "rhov"):
    for n in ("sfit", "tfit"):
        comps.append((prop, f"pr_{n}", f"srk_{n}", "F1" if prop in ("psat", "h") else "descriptive"))
# alpha-flexibility and in- vs out-of-sample comparisons at fixed cubic
for prop in ("psat", "h"):
    for c in ("srk", "pr"):
        f2fam = "F2" if prop == "psat" else "descriptive"           # the psat exchanges are the ones quoted in the text
        comps.append((prop, f"{c}_tfit", f"{c}_sfit", f2fam))      # three vs one fitted parameter
        comps.append((prop, f"{c}_tfit", f"{c}_tf", f2fam))        # in-sample vs out-of-sample fitted Twu
        comps.append((prop, f"{c}_sfit", f"{c}_s", f2fam))         # in-sample Soave vs generalized Soave
        best_gen = {"srk": "t95", "pr": "s"}[c]
        comps.append((prop, f"{c}_tfit", f"{c}_{best_gen}", "descriptive"))  # in-sample Twu vs best generalized
        if best_gen != "s":
            comps.append((prop, f"{c}_sfit", f"{c}_{best_gen}", "descriptive"))
        comps.append((prop, f"{c}_tfit", f"{c}_ms", "descriptive"))
        comps.append((prop, f"{c}_tunc", f"{c}_tfit", "descriptive"))  # unconstrained vs constrained (diagnostic)
comps.append(("psat", "pr_tunc", "srk_tunc", "descriptive"))
comps.append(("h", "pr_tunc", "srk_tunc", "descriptive"))
rows = []
for prop, a, b, famname in comps:
    ca, cb = f"{PROPCOL[prop]}_{a}", f"{PROPCOL[prop]}_{b}"
    if ca not in dfm.columns or cb not in dfm.columns:
        continue
    wins, n, p = sign_test(ca, cb)
    d_boot = boot(ca) - boot(cb)
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    pfa, pfb = per_fluid_aad(ca), per_fluid_aad(cb)
    pfd = np.array([pfa[f] - pfb[f] for f in fluids])
    lr = np.array([np.log(pfa[f] / pfb[f]) for f in fluids])
    # fluid-level bootstrap of the mean log-ratio (same resamples)
    lr_boot = lr[IDX].mean(axis=1)
    rows.append(dict(key=f"{prop}:{a}_vs_{b}", prop=prop, A=a, B=b, family=famname, aad_A=aad(dfm[ca]), aad_B=aad(dfm[cb]),
                     dAAD=aad(dfm[ca]) - aad(dfm[cb]), ci_lo=float(lo), ci_hi=float(hi),
                     pf_diff_median=float(np.median(pfd)), pf_diff_q1=float(np.percentile(pfd, 25)),
                     pf_diff_q3=float(np.percentile(pfd, 75)), pf_diff_min=float(pfd.min()), pf_diff_max=float(pfd.max()),
                     mean_log_ratio=float(lr.mean()), geo_ratio=float(np.exp(lr.mean())),
                     geo_ratio_lo=float(np.exp(np.percentile(lr_boot, 2.5))), geo_ratio_hi=float(np.exp(np.percentile(lr_boot, 97.5))),
                     wins=wins, n=n, p=p))
cmp_ = pd.DataFrame(rows).set_index("key")

# Holm within the final families (full runs only): F1 = 16 predictive tests + 4 in-sample; F2 = 21 + the in-sample exchanges
if not quick:
    F1_new = [k for k in cmp_.index if cmp_.loc[k, "family"] == "F1"]
    F2_new = [k for k in cmp_.index if cmp_.loc[k, "family"] == "F2"]
    allp = pd.concat([wp2.loc[fam["F1"] + fam["F2"], ["p"]], cmp_.loc[F1_new + F2_new, ["p"]]])
    fam1 = fam["F1"] + F1_new
    fam2 = fam["F2"] + F2_new
    adj = {}
    for keys in (fam1, fam2):
        a = holm(allp.loc[keys, "p"].to_numpy())
        adj.update(dict(zip(keys, a)))
    cmp_["p_holm_family_ext"] = [adj.get(k, np.nan) for k in cmp_.index]
    ext = pd.DataFrame({"p": allp.p, "p_holm_ext": [adj[k] for k in allp.index],
                        "family": ["F1" if k in fam1 else "F2" for k in allp.index]})
    ext["p_holm_wp2"] = [wp2.loc[k, "p_holm_family"] if k in wp2.index else np.nan for k in ext.index]
    ext.to_csv(os.path.join(DATA, "holm_extended.csv"))
    # Benjamini-Yekutieli over all 242 comparisons (206 of script 08 + these 36): one adjustment valid under
    # arbitrary dependence for every row of tabS_signtests_full
    allq = pd.concat([wp2[["p"]], cmp_[["p"]]])
    m_ = len(allq)
    cm_ = np.sum(1.0 / np.arange(1, m_ + 1))
    order_ = np.argsort(allq.p.to_numpy())
    adjq, running = np.empty(m_), 1.0
    for rank in range(m_ - 1, -1, -1):
        i_ = order_[rank]
        running = min(running, min(1.0, m_ * cm_ * allq.p.to_numpy()[i_] / (rank + 1)))
        adjq[i_] = running
    allq["q_BY_all"] = adjq
    allq.to_csv(os.path.join(DATA, "by_all_comparisons.csv"))
    cmp_["q_BY_all"] = [allq.loc[k, "q_BY_all"] for k in cmp_.index]
    print("Benjamini-Yekutieli over", m_, "comparisons: significant at q < 0.05:", int((allq.q_BY_all < 0.05).sum()),
          "| of the original 206 at the 206-set adjustment:", int((wp2.q_BY_all206 < 0.05).sum()))
cmp_.to_csv(os.path.join(DATA, "insample_comparisons.csv"))

# ------------------------------------------------------------------ 5. two-way decomposition of log-AAD (fluid as block)
def decomposition(alphas, prop, use_log=True):
    """y[cubic, alpha, fluid] = (log) per-fluid AAD; balanced 2 x J x n_f design."""
    Y = np.empty((2, len(alphas), n_f))
    for i, c in enumerate(("srk", "pr")):
        for j, a in enumerate(alphas):
            for k, f in enumerate(fluids):
                v = perfluid.loc[f, f"{prop}_{c}_{a}"]
                Y[i, j, k] = np.log(v) if use_log else v
    I, J, K = Y.shape
    gm = Y.mean()
    ss_tot = float(((Y - gm) ** 2).sum())
    ss_c = float(J * K * ((Y.mean(axis=(1, 2)) - gm) ** 2).sum())
    ss_a = float(I * K * ((Y.mean(axis=(0, 2)) - gm) ** 2).sum())
    ss_f = float(I * J * ((Y.mean(axis=(0, 1)) - gm) ** 2).sum())
    ca_means = Y.mean(axis=2)
    ss_ca = float(K * ((ca_means - Y.mean(axis=(1, 2))[:, None] - Y.mean(axis=(0, 2))[None, :] + gm) ** 2).sum())
    cf_means = Y.mean(axis=1)
    ss_cf = float(J * ((cf_means - Y.mean(axis=(1, 2))[:, None] - Y.mean(axis=(0, 1))[None, :] + gm) ** 2).sum())
    af_means = Y.mean(axis=0)
    ss_af = float(I * ((af_means - Y.mean(axis=(0, 2))[:, None] - Y.mean(axis=(0, 1))[None, :] + gm) ** 2).sum())
    ss_res = ss_tot - ss_c - ss_a - ss_f - ss_ca - ss_cf - ss_af
    ss_model = ss_tot - ss_f           # everything except the fluid block
    # cubic effect per alpha-function: mean over fluids of (PR - SRK) in log-AAD = log of the geometric mean ratio
    eff = {a: float((Y[1, j, :] - Y[0, j, :]).mean()) for j, a in enumerate(alphas)}
    return dict(prop=prop, alphas=alphas, n_alpha=len(alphas), use_log=use_log, ss_total=ss_tot, ss_fluid=ss_f,
                ss_cubic=ss_c, ss_alpha=ss_a, ss_cubic_x_alpha=ss_ca, ss_cubic_x_fluid=ss_cf, ss_alpha_x_fluid=ss_af,
                ss_residual=ss_res, ss_excl_fluid=ss_model,
                share_excl_fluid=dict(cubic=ss_c / ss_model, alpha=ss_a / ss_model, cubic_x_alpha=ss_ca / ss_model,
                                      cubic_x_fluid=ss_cf / ss_model, alpha_x_fluid=ss_af / ss_model, residual=ss_res / ss_model),
                cubic_effect_by_alpha=eff,
                cubic_main_effect=float(Y[1].mean() - Y[0].mean()))


decomp = {}
sets = {"14 combinations (paper)": ALPHAS, "18 combinations (paper + in-sample)": ALPHAS + ["sfit", "tfit"],
        "in-sample only": ["sfit", "tfit"], "one-procedure generalized pairs": ["t95", "coq", "ms"],
        "fitted forms only (tf, sfit, tfit)": ["tf", "sfit", "tfit"]}
for name, al in sets.items():
    for prop in ("psat", "h"):
        decomp[f"{name}|{prop}|log"] = decomposition(al, prop, True)
        decomp[f"{name}|{prop}|points"] = decomposition(al, prop, False)
json.dump(decomp, open(os.path.join(DATA, "insample_decomposition.json"), "w"), indent=1)

# ------------------------------------------------------------------ 6. verification summaries from the fit records
ver = {}
ver["clapeyron_max"] = clap_max
ver["aad_reproduction_max_diff"] = float(max(chk))
ver["n_fits"] = int(len(fits))
ver["starts_within_0p001_min"] = int(fits.n_starts_within_0p001.min())
ver["starts_within_0p001_mean"] = float(fits.n_starts_within_0p001.mean())
ver["n_fits_single_start_only"] = int((fits.n_starts_within_0p001 <= 1).sum())
ver["active_Q3"] = int(fits.active_Q3.sum())
ver["active_P2"] = int(fits.active_P2.sum())
ver["active_M1"] = int(fits.active_M1.sum())
ver["active_any"] = int((fits.active_Q3 | fits.active_P2 | fits.active_M1 | fits.active_L0 | fits.active_d1).sum())
ver["aad_tfit_le_tf_all"] = bool((fits.psat_aad_tfit <= fits.aad_twu_tf + 1e-9).all())
ver["max_aad_tfit_minus_tf"] = float((fits.psat_aad_tfit - fits.aad_twu_tf).max())
ver["mean_aad_tf_minus_tfit"] = float((fits.aad_twu_tf - fits.psat_aad_tfit).mean())
ver["max_aad_ls_minus_l1_twu"] = float((fits.aad_twu_at_ls - fits.psat_aad_tfit).max())
ver["mean_aad_ls_minus_l1_twu"] = float((fits.aad_twu_at_ls - fits.psat_aad_tfit).mean())
ver["max_aad_ls_minus_l1_soave"] = float((fits.aad_soave_at_ls - fits.aad_soave_fit).max())
alk = fits[fits.fluid.map(lambda f: fl.loc[f, "family"]) == "Alkanes"]
ver["alkanes_param_offsets"] = {c: dict(mean_abs_dL=float((alk[alk.cubic == c].L_fit - alk[alk.cubic == c].L_tf).abs().mean()),
                                        mean_abs_dM=float((alk[alk.cubic == c].M_fit - alk[alk.cubic == c].M_tf).abs().mean()),
                                        mean_abs_dN=float((alk[alk.cubic == c].N_fit - alk[alk.cubic == c].N_tf).abs().mean()),
                                        mean_aad_tf=float(alk[alk.cubic == c].aad_twu_tf.mean()),
                                        mean_aad_tfit=float(alk[alk.cubic == c].psat_aad_tfit.mean()))
                               for c in ("SRK", "PR")}
ver["spread_params_among_equivalent_starts"] = dict(L=float(fits.spread_L.max()), M=float(fits.spread_M.max()), N=float(fits.spread_N.max()))
# closeness of the in-sample alpha-function to the published [8] set and to the generalized Twu-c on the grid
alpha_rows = []
for fluid in fluids_fit:
    g = df[df.fluid == fluid].sort_values("Tr")
    Tr = g.Tr.to_numpy()
    for cubic in ("SRK", "PR"):
        f = fits[(fits.fluid == fluid) & (fits.cubic == cubic)].iloc[0]
        def tw(L, M, N):
            return Tr ** (N * (M - 1.0)) * np.exp(L * (1.0 - Tr ** (N * M)))
        a_fit, a_tf, a_gen = tw(f.L_fit, f.M_fit, f.N_fit), tw(f.L_tf, f.M_tf, f.N_tf), tw(f.L_gen, f.M_gen, 2.0)
        a_sf = (1.0 + f.m_fit * (1.0 - np.sqrt(Tr))) ** 2
        alpha_rows.append(dict(fluid=fluid, cubic=cubic, family=fl.loc[fluid, "family"],
                               max_rel_dalpha_vs_tf=float(np.max(np.abs(a_fit / a_tf - 1.0)) * 100),
                               mean_rel_dalpha_vs_tf=float(np.mean(np.abs(a_fit / a_tf - 1.0)) * 100),
                               max_rel_dalpha_vs_gen=float(np.max(np.abs(a_fit / a_gen - 1.0)) * 100),
                               max_rel_dalpha_vs_soavefit=float(np.max(np.abs(a_fit / a_sf - 1.0)) * 100),
                               dalpha_vs_tf_at_Trmin=float((a_fit[0] / a_tf[0] - 1.0) * 100),
                               dalpha_vs_tf_at_Trmax=float((a_fit[-1] / a_tf[-1] - 1.0) * 100)))
alpha_df = pd.DataFrame(alpha_rows)
alpha_df.to_csv(os.path.join(DATA, "insample_alpha_closeness.csv"), index=False)
ver["alpha_closeness_vs_tf"] = {c: dict(max_pct=float(alpha_df[alpha_df.cubic == c].max_rel_dalpha_vs_tf.max()),
                                        mean_of_max_pct=float(alpha_df[alpha_df.cubic == c].max_rel_dalpha_vs_tf.mean()),
                                        alkanes_mean_of_max_pct=float(alpha_df[(alpha_df.cubic == c) & (alpha_df.family == "Alkanes")].max_rel_dalpha_vs_tf.mean()))
                                for c in ("SRK", "PR")}
ver["unconstrained_consistent_count"] = int(fits.unc_consistent.sum()) if "unc_consistent" in fits else None
ver["unconstrained_gain_max"] = float((fits.psat_aad_tfit - fits.aad_twu_unc).max()) if "aad_twu_unc" in fits else None
json.dump(ver, open(os.path.join(DATA, "insample_verification.json"), "w"), indent=1)

pd.set_option("display.width", 250)
print(json.dumps(ver, indent=1))
print(cmp_[["family", "aad_A", "aad_B", "dAAD", "ci_lo", "ci_hi", "geo_ratio", "wins", "n", "p"]].round(4).to_string())
for name in sets:
    for prop in ("psat", "h"):
        d = decomp[f"{name}|{prop}|log"]
        print(name, prop, {k: round(v, 3) for k, v in d["share_excl_fluid"].items()}, "cubic eff by alpha:",
              {k: round(v, 3) for k, v in d["cubic_effect_by_alpha"].items()})
print("wrote insample_points.csv, insample_pooled.csv, insample_per_fluid.csv, insample_comparisons.csv, insample_decomposition.json, insample_verification.json")
