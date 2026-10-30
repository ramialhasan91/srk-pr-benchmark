"""Three additions of the revision, each with the protocol of 01_run_benchmark.py (same temperatures, same
reference values):
(A) PR78 (tabS_pr78): the 1978 Robinson-Peng switch
      m = 0.379642 + 1.48503 w - 0.164423 w^2 + 0.016666 w^3  for w > 0.49, else PR76,
    for all 37 fluids with the Clapeyron check and the ladder re-anchored on the variant; it differs from
    PR76 for methanol and ethanol only.
(B) The dimensionless shift s = c/b (b = Omega_b R Tc / pc) for the constants of tabS_vt, and the absolute
    deviations of the enthalpy of vaporization by range (tabS_nearcrit).
(C) Model-form versus input error (tabS_srcconst, tabS_sens): (a) all 14 combinations re-evaluated with
    the DIPPR-based Tc, pc of the compilation of tabS_twufit_src (omega unchanged; points with T >= Tc_src
    dropped, as in script 06), pooled AADs with reference and compilation constants on the same points;
    (b) sensitivities of the saturation properties to Tc, pc and omega by central differences (PR- and
    SRK-Soave, every grid point), summarized at Tr = 0.50, 0.70, 0.90 and 0.985.

Reads  ../data/deviations_full.csv, fluids.csv, key_numbers.json, run_metadata.json, twu_fit_params.csv,
       twu_fit_source_critical_constants.csv, twufit_source_constants.csv
Writes ../data/pr78_points.csv, pr78_summary.csv, pr78.json; volume_shift.csv, volume_shift_summary.json,
       hvap_absolute.csv, hvap_absolute_per_fluid.csv; compilation_constants_per_fluid.csv,
       compilation_constants_points.csv, compilation_constants.json, input_sensitivities.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
from eos import R, CubicEOS  # noqa: E402

# ====================================================================== #
# (A) PR78 alpha-variant
df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv")).set_index("fluid")
key = json.load(open(os.path.join(DATA, "key_numbers.json")))


def m_pr76(w):
    return 0.37464 + 1.54226 * w - 0.26992 * w * w


def m_pr78(w):
    return 0.379642 + 1.48503 * w - 0.164423 * w ** 2 + 0.016666 * w ** 3 if w > 0.49 else m_pr76(w)


def pct(model, ref):
    return (model - ref) / ref * 100.0


CLAP_H = 1e-4
rows, clap = [], []
mvals = {}
for fluid, g in df.groupby("fluid", sort=False):
    g = g.sort_values("Tr")
    r0 = fl.loc[fluid]
    Tc, pc, w, cp_name = r0.Tc_K, r0.pc_MPa * 1e6, r0.omega, r0.coolprop
    m = CubicEOS("PR", Tc, pc, w, alpha="s")
    assert abs(m.m - m_pr76(w)) < 1e-14
    m.m = m_pr78(w)                       # the only change: the slope parameter
    mvals[fluid] = dict(omega=w, m76=m_pr76(w), m78=m_pr78(w))
    # translation constants re-anchored on the variant (protocol of script 01)
    Tr_anchor = r0.Tr_anchor
    T_anchor = Tr_anchor * Tc
    v_ref_anchor = 1.0 / PropsSI("Dmolar", "T", T_anchor, "Q", 0, cp_name)
    Z_RA = 0.29056 - 0.08775 * w
    v_ra_anchor = R * Tc / pc * Z_RA ** (1.0 + (1.0 - Tr_anchor) ** (2.0 / 7.0))
    v_anchor, _ = m.sat_liq_volume(T_anchor)
    c_fit, c_corr = v_anchor - v_ref_anchor, v_anchor - v_ra_anchor
    for _, r in g.iterrows():
        T = r.T_K
        p, Zl, Zv = m.psat(T)
        assert np.isfinite(p), (fluid, T)
        v_l, v_v = Zl * R * T / p, Zv * R * T / p
        h_vap = m.h_res(T, p, Zv) - m.h_res(T, p, Zl)
        p_lo, _, _ = m.psat(T - CLAP_H * T)
        p_hi, _, _ = m.psat(T + CLAP_H * T)
        h_clap = T * (v_v - v_l) * (p_hi - p_lo) / (2.0 * CLAP_H * T)
        clap.append(abs(h_clap / h_vap - 1.0))
        rows.append(dict(fluid=fluid, family=r.family, subgroup=r.subgroup, Tr=r.Tr, T_K=T,
                         dev_p=pct(p, r.p_ref_Pa), dev_rho=pct(1.0 / v_l, r.rho_ref_molm3),
                         dev_rhov=pct(1.0 / v_v, r.rhov_ref_molm3), dev_h=pct(h_vap, r.hvap_ref_Jmol),
                         dev_rho_vt=pct(1.0 / (v_l - c_fit), r.rho_ref_molm3),
                         dev_rho_vtc=pct(1.0 / (v_l - c_corr), r.rho_ref_molm3),
                         dev_rhov_vt=pct(1.0 / (v_v - c_fit), r.rhov_ref_molm3),
                         # the committed PR-Soave (PR76) values at the same point
                         dev_p_76=r.dev_p_pr_s, dev_rho_76=r.dev_rho_pr_s, dev_rhov_76=r.dev_rhov_pr_s,
                         dev_h_76=r.dev_h_pr_s, dev_rho_vt_76=r.dev_rho_pr_s_vt, dev_rho_vtc_76=r.dev_rho_pr_s_vtc,
                         dev_rhov_vt_76=r.dev_rhov_pr_s_vt))
    mvals[fluid].update(c_fit_cm3mol=c_fit * 1e6, c_corr_cm3mol=c_corr * 1e6,
                        c_fit76_cm3mol=r0.c_fit_pr_s_cm3mol, c_corr76_cm3mol=r0.c_corr_pr_s_cm3mol)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA, "pr78_points.csv"), index=False)
print(f"Clapeyron check (PR78 variant, all 1480 points): max rel diff = {max(clap):.2e}")

# --- environment check: for omega <= 0.49 the variant IS PR-Soave; reproduce the committed numbers
same = out[out.fluid.map(lambda f: mvals[f]["omega"] <= 0.49)]
for a, b in (("dev_p", "dev_p_76"), ("dev_rho", "dev_rho_76"), ("dev_rhov", "dev_rhov_76"), ("dev_h", "dev_h_76"),
             ("dev_rho_vt", "dev_rho_vt_76"), ("dev_rho_vtc", "dev_rho_vtc_76")):
    print(f"  reproduction of committed PR-Soave {a:12s}: max |diff| = {np.abs(same[a] - same[b]).max():.2e} "
          f"({len(same.fluid.unique())} fluids)")
changed = sorted(f for f in mvals if mvals[f]["omega"] > 0.49)
print("fluids with omega > 0.49:", changed, {f: mvals[f] for f in changed})
print("n-decane:", mvals["n-decane"])

# --- AADs by range for the two alcohols, PR78 vs PR76
BINS = [("0.50-0.70", 0.50, 0.70), ("0.70-0.90", 0.70, 0.90), ("0.90-0.985", 0.90, 0.9851), ("< 0.90", 0.0, 0.90),
        ("All", 0.0, 1.0)]
PROPS = [("psat", "dev_p"), ("h", "dev_h"), ("rhov", "dev_rhov"), ("rho", "dev_rho"), ("rho_vt", "dev_rho_vt"),
         ("rho_vtc", "dev_rho_vtc"), ("rhov_vt", "dev_rhov_vt")]
summary = []
for fluid in changed:
    g = out[out.fluid == fluid]
    for lab, lo, hi in BINS:
        s = g[(g.Tr >= lo) & (g.Tr < hi)]
        rec = dict(fluid=fluid, range=lab, n=len(s))
        for prop, col in PROPS:
            rec[f"{prop}_PR78"] = float(np.mean(np.abs(s[col])))
            rec[f"{prop}_PR76"] = float(np.mean(np.abs(s[col + "_76"])))
            rec[f"{prop}_bias_PR78"] = float(np.mean(s[col]))
            rec[f"{prop}_bias_PR76"] = float(np.mean(s[col + "_76"]))
        summary.append(rec)
summ = pd.DataFrame(summary)
summ.to_csv(os.path.join(DATA, "pr78_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(summ[["fluid", "range", "n", "psat_PR76", "psat_PR78", "h_PR76", "h_PR78", "rhov_PR76", "rhov_PR78",
            "rho_PR76", "rho_PR78", "rho_vt_PR76", "rho_vt_PR78"]].round(3).to_string())

# --- low-temperature spot values and the signed deviation curve of the two alcohols
spot = {}
for fluid in changed:
    g = out[out.fluid == fluid].sort_values("Tr")
    spot[fluid] = dict(Tr050_dev_p_PR76=float(g.dev_p_76.iloc[0]), Tr050_dev_p_PR78=float(g.dev_p.iloc[0]),
                       Tr050=float(g.Tr.iloc[0]),
                       max_abs_dev_p_PR76=float(g.dev_p_76.abs().max()), max_abs_dev_p_PR78=float(g.dev_p.abs().max()),
                       dev_p_near_Tr07_PR76=float(g.iloc[(g.Tr - 0.7).abs().argmin()].dev_p_76),
                       dev_p_near_Tr07_PR78=float(g.iloc[(g.Tr - 0.7).abs().argmin()].dev_p),
                       dev_p_at_Tr_points={f"{t:.3f}": (float(a), float(b)) for t, a, b in
                                           zip(g.Tr, g.dev_p_76, g.dev_p)})
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "dev_p_at_Tr_points"} for k, v in spot.items()}, indent=1))

# --- pooled effect: H-bonding subgroup, polar/associating family and all fluids, PR78 replacing PR76
pooled = {}
for lab, mask in (("All (37)", np.ones(len(out), bool)), ("Polar/associating (10)", (out.family == "Polar/associating").to_numpy()),
                  ("H-bonding (4)", (out.subgroup == "H-bonding").to_numpy())):
    rec = {}
    for prop, col in PROPS[:4]:
        rec[f"{prop}_PR76"] = float(np.mean(np.abs(out.loc[mask, col + "_76"])))
        rec[f"{prop}_PR78"] = float(np.mean(np.abs(out.loc[mask, col])))
    for sub, lo, hi in (("<0.90", 0.0, 0.90), ("0.50-0.70", 0.50, 0.70)):
        mm = mask & (out.Tr >= lo).to_numpy() & (out.Tr < hi).to_numpy()
        for prop, col in PROPS[:2]:
            rec[f"{prop}_{sub}_PR76"] = float(np.mean(np.abs(out.loc[mm, col + "_76"])))
            rec[f"{prop}_{sub}_PR78"] = float(np.mean(np.abs(out.loc[mm, col])))
    pooled[lab] = rec
print(json.dumps(pooled, indent=1))
json.dump(dict(m=mvals, spot=spot, pooled=pooled, clapeyron_max=float(max(clap))),
          open(os.path.join(DATA, "pr78.json"), "w"), indent=1)
print("wrote pr78_points.csv, pr78_summary.csv, pr78.json")

# ====================================================================== #
# (B) dimensionless volume shift and absolute enthalpy-of-vaporization deviations
df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))
meta = json.load(open(os.path.join(DATA, "run_metadata.json")))
TAGS = meta["models"]
FAMILY_ORDER = ["Inorganic gases", "Alkanes", "Aromatics", "Polar/associating", "Refrigerants"]

# ------------------------------------------------------------------ s = c/b
rows = []
for _, r in fl.iterrows():
    Tc, pc = r.Tc_K, r.pc_MPa * 1e6
    rec = dict(fluid=r.fluid, family=r.family, subgroup=r.subgroup if isinstance(r.subgroup, str) else "",
               omega=r.omega, Zc_ref=r.Zc_ref)
    for cub in ("srk", "pr"):
        b = CubicEOS.PARAMS[cub.upper()]["Omega_b"] * R * Tc / pc * 1e6   # cm3/mol
        rec[f"b_{cub}_cm3mol"] = b
        for kind in ("fit", "corr"):
            rec[f"c_{kind}_{cub}_cm3mol"] = r[f"c_{kind}_{cub}_s_cm3mol"]
            rec[f"s_{kind}_{cub}"] = r[f"c_{kind}_{cub}_s_cm3mol"] / b
        rec[f"c_zc_{cub}_cm3mol"] = r[f"c_zc_{cub}_cm3mol"]
        rec[f"s_zc_{cub}"] = r[f"c_zc_{cub}_cm3mol"] / b
        rec[f"c_lit_{cub}_cm3mol"] = r[f"c_lit_{cub}_cm3mol"]
        rec[f"s_lit_{cub}"] = r[f"c_lit_{cub}_cm3mol"] / b
    rows.append(rec)
sh = pd.DataFrame(rows)
sh.to_csv(os.path.join(DATA, "volume_shift.csv"), index=False)
pd.set_option("display.width", 250)
print(sh[["fluid", "omega", "b_srk_cm3mol", "c_fit_srk_cm3mol", "s_fit_srk", "s_corr_srk", "b_pr_cm3mol",
          "c_fit_pr_cm3mol", "s_fit_pr", "s_corr_pr", "s_lit_pr"]].round(4).to_string())
summ = {}
for col in ("s_fit_srk", "s_corr_srk", "s_zc_srk", "s_lit_srk", "s_fit_pr", "s_corr_pr", "s_zc_pr", "s_lit_pr"):
    summ[col] = dict(min=float(sh[col].min()), fluid_min=str(sh.loc[sh[col].idxmin(), "fluid"]),
                     max=float(sh[col].max()), fluid_max=str(sh.loc[sh[col].idxmax(), "fluid"]),
                     mean=float(sh[col].mean()), median=float(sh[col].median()),
                     n_negative=int((sh[col] < 0).sum()),
                     by_family={fam: float(sh.loc[sh.family == fam, col].mean()) for fam in FAMILY_ORDER})
summ["corr_s_fit_srk_vs_pr"] = float(np.corrcoef(sh.s_fit_srk, sh.s_fit_pr)[0, 1])
summ["corr_s_fit_pr_vs_lit_pr"] = float(np.corrcoef(sh.s_fit_pr, sh.s_lit_pr)[0, 1])
summ["mean_abs_diff_s_fit_pr_minus_lit_pr"] = float((sh.s_fit_pr - sh.s_lit_pr).abs().mean())
# reference point: Peneloux (1982) reported c/b of about 0.1 for alkanes on SRK
print(json.dumps(summ, indent=1))
json.dump(summ, open(os.path.join(DATA, "volume_shift_summary.json"), "w"), indent=1)

# ------------------------------------------------------------------ absolute hvap deviations
# fine bins of Table S13 plus the pooled range below 0.90
BINS = [("< 0.90", 0.0, 0.90), ("0.90-0.95", 0.90, 0.95), ("0.95-0.985", 0.95, 0.9851), ("All", 0.0, 1.0)]
# per-fluid reference hvap at the grid point nearest max(0.70, Tr_min) (the anchoring convention)
h07 = {}
for f, g in df.groupby("fluid"):
    g = g.sort_values("Tr")
    tr_a = max(0.70, g.Tr.iloc[0])
    h07[f] = float(g.iloc[(g.Tr - tr_a).abs().argmin()].hvap_ref_Jmol)
df["h07_ref"] = df.fluid.map(h07)
df["RTc"] = df.fluid.map(fl.set_index("fluid").Tc_K) * R
rows = []
for t in TAGS:
    d_abs = (df[f"hvap_{t}_Jmol"] - df.hvap_ref_Jmol)          # J/mol, signed
    for lab, lo, hi in BINS:
        m = (df.Tr >= lo) & (df.Tr < hi)
        rec = dict(model=t, range=lab, n=int(m.sum()),
                   mean_abs_dev_kJmol=float(np.mean(np.abs(d_abs[m])) / 1e3),
                   mean_signed_dev_kJmol=float(np.mean(d_abs[m]) / 1e3),
                   median_abs_dev_kJmol=float(np.median(np.abs(d_abs[m])) / 1e3),
                   mean_ref_hvap_kJmol=float(np.mean(df.hvap_ref_Jmol[m]) / 1e3),
                   rel_AAD_pct=float(np.mean(np.abs(df[f"dev_h_{t}"][m]))),
                   # absolute deviation normalized per fluid by its hvap at Tr = 0.70, then pooled (%)
                   abs_dev_rel_h07_pct=float(np.mean(np.abs(d_abs[m]) / df.h07_ref[m]) * 100.0),
                   signed_dev_rel_h07_pct=float(np.mean(d_abs[m] / df.h07_ref[m]) * 100.0),
                   # normalized by R Tc (dimensionless), pooled
                   abs_dev_over_RTc=float(np.mean(np.abs(d_abs[m]) / df.RTc[m])),
                   # reference hvap relative to the Tr = 0.70 value, pooled (shows the shrinking denominator)
                   ref_hvap_rel_h07_pct=float(np.mean(df.hvap_ref_Jmol[m] / df.h07_ref[m]) * 100.0))
        rows.append(rec)
ha = pd.DataFrame(rows)
ha.to_csv(os.path.join(DATA, "hvap_absolute.csv"), index=False)
print(ha[ha.model.isin(["srk_s", "pr_s", "srk_t95", "pr_t95", "srk_tc", "pr_tc", "srk_tf", "pr_tf"])].round(3).to_string())

# per-fluid near-critical absolute deviations for the classical pair (to find the fluids that dominate)
pf = []
for f, g in df.groupby("fluid"):
    rec = dict(fluid=f, h07_kJmol=h07[f] / 1e3)
    for t in ("srk_s", "pr_s", "pr_t95", "pr_tf"):
        d = (g[f"hvap_{t}_Jmol"] - g.hvap_ref_Jmol)
        for lab, lo, hi in BINS[:3]:
            m = (g.Tr >= lo) & (g.Tr < hi)
            rec[f"{t}|{lab}|abs_kJmol"] = float(np.mean(np.abs(d[m])) / 1e3)
            rec[f"{t}|{lab}|rel_pct"] = float(np.mean(np.abs(g[f"dev_h_{t}"][m])))
    pf.append(rec)
pf = pd.DataFrame(pf)
pf.to_csv(os.path.join(DATA, "hvap_absolute_per_fluid.csv"), index=False)
print(pf[["fluid", "h07_kJmol", "pr_s|< 0.90|abs_kJmol", "pr_s|0.90-0.95|abs_kJmol", "pr_s|0.95-0.985|abs_kJmol",
          "pr_s|< 0.90|rel_pct", "pr_s|0.90-0.95|rel_pct", "pr_s|0.95-0.985|rel_pct"]].round(3).to_string())
print("wrote volume_shift.csv, volume_shift_summary.json, hvap_absolute.csv, hvap_absolute_per_fluid.csv")

# ====================================================================== #
# (C) model-form error versus input error
df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv")).set_index("fluid")
tf = pd.read_csv(os.path.join(DATA, "twu_fit_params.csv"), comment="#").set_index("fluid")
src = pd.read_csv(os.path.join(DATA, "twu_fit_source_critical_constants.csv"), comment="#").set_index("fluid")
old = pd.read_csv(os.path.join(DATA, "twufit_source_constants.csv")).set_index("fluid")
meta = json.load(open(os.path.join(DATA, "run_metadata.json")))
TAGS = meta["models"]
ALPHAS = ["s", "s19", "t95", "tc", "coq", "ms", "tf"]
PROPS = {"psat": "dev_p", "h": "dev_h", "rho": "dev_rho", "rhov": "dev_rhov"}
FAMILY_ORDER = ["Inorganic gases", "Alkanes", "Aromatics", "Polar/associating", "Refrigerants"]


def pct(model, ref):
    return (model - ref) / ref * 100.0


def make(cubic, a, fluid, Tc, pc, w):
    kw = {}
    if a == "tf":
        kw["twu_params"] = (tf.loc[fluid, f"L_{cubic}"], tf.loc[fluid, f"M_{cubic}"], tf.loc[fluid, f"N_{cubic}"])
    return CubicEOS(cubic, Tc, pc, w, alpha=a, **kw)


def evaluate(m, g):
    """Per-point deviations of model m on the rows g (NaN where T >= Tc of the model)."""
    out = {k: np.full(len(g), np.nan) for k in PROPS}
    for i, (_, r) in enumerate(g.iterrows()):
        T = r.T_K
        if T >= m.Tc:
            continue
        p, Zl, Zv = m.psat(T)
        if not np.isfinite(p):
            continue
        v_l, v_v = Zl * R * T / p, Zv * R * T / p
        out["psat"][i] = pct(p, r.p_ref_Pa)
        out["rho"][i] = pct(1.0 / v_l, r.rho_ref_molm3)
        out["rhov"][i] = pct(1.0 / v_v, r.rhov_ref_molm3)
        out["h"][i] = pct(m.h_res(T, p, Zv) - m.h_res(T, p, Zl), r.hvap_ref_Jmol)
    return out


# ------------------------------------------------------------ (a) compilation constants
rows, pts = [], []
for fluid, g in df.groupby("fluid", sort=False):
    g = g.sort_values("Tr")
    Tc_ref, pc_ref, w = fl.loc[fluid, "Tc_K"], fl.loc[fluid, "pc_MPa"] * 1e6, fl.loc[fluid, "omega"]
    Tc_s, pc_s = src.loc[fluid, "Tc_src_K"], src.loc[fluid, "pc_src_bar"] * 1e5
    rec = dict(fluid=fluid, family=g.family.iloc[0], subgroup=g.subgroup.iloc[0] if isinstance(g.subgroup.iloc[0], str) else "",
               dTc_pct=pct(Tc_s, Tc_ref), dpc_pct=pct(pc_s, pc_ref))
    for cubic in ("SRK", "PR"):
        cl = cubic.lower()
        for a in ALPHAS:
            tag = f"{cl}_{a}"
            dev = evaluate(make(cubic, a, fluid, Tc_s, pc_s, w), g)
            keep = np.isfinite(dev["psat"])
            rec[f"n_{tag}"] = int(keep.sum())
            for prop, col in PROPS.items():
                rec[f"{prop}_{tag}_src"] = float(np.mean(np.abs(dev[prop][keep])))
                rec[f"{prop}_{tag}_ref"] = float(np.mean(np.abs(g[f"{col}_{tag}"].to_numpy()[keep])))
                rec[f"{prop}_bias_{tag}_src"] = float(np.mean(dev[prop][keep]))
                rec[f"{prop}_bias_{tag}_ref"] = float(np.mean(g[f"{col}_{tag}"].to_numpy()[keep]))
            for i in range(len(g)):
                pts.append(dict(fluid=fluid, tag=tag, Tr=g.Tr.iloc[i], keep=bool(keep[i]),
                                **{f"dev_{prop}_src": dev[prop][i] for prop in PROPS},
                                **{f"dev_{prop}_ref": g[f"{col}_{tag}"].iloc[i] for prop, col in PROPS.items()}))
    rows.append(rec)
res = pd.DataFrame(rows).set_index("fluid")
res.to_csv(os.path.join(DATA, "compilation_constants_per_fluid.csv"))
ptsdf = pd.DataFrame(pts)
ptsdf.to_csv(os.path.join(DATA, "compilation_constants_points.csv"), index=False)

# check against the committed Table S15 numbers (script 06) for the Twu-fit variants
for cl in ("srk", "pr"):
    for prop in PROPS:
        d = np.abs(res[f"{prop}_{cl}_tf_src"] - old[f"{prop}_{cl}_tf_src"]).max()
        d2 = np.abs(res[f"{prop}_{cl}_tf_ref"] - old[f"{prop}_{cl}_tf_ref"]).max()
        print(f"check vs 06_twufit_source_constants: {prop}_{cl}_tf src max|diff| = {d:.2e}, ref max|diff| = {d2:.2e}")
dropped = {tag: int((ptsdf[(ptsdf.tag == tag)].keep == False).sum()) for tag in TAGS}
print("points dropped (T >= Tc_src):", dropped)

# pooled (point-weighted over kept points), overall and below Tr = 0.90, plus by family / subgroup
def pooled(mask):
    d = {}
    for tag in TAGS:
        q = ptsdf[(ptsdf.tag == tag) & ptsdf.keep & mask.reindex(ptsdf.index, fill_value=False)]
        for prop in PROPS:
            d[f"{prop}_{tag}_src"] = float(np.mean(np.abs(q[f"dev_{prop}_src"])))
            d[f"{prop}_{tag}_ref"] = float(np.mean(np.abs(q[f"dev_{prop}_ref"])))
            d[f"{prop}_bias_{tag}_src"] = float(np.mean(q[f"dev_{prop}_src"]))
            d[f"{prop}_bias_{tag}_ref"] = float(np.mean(q[f"dev_{prop}_ref"]))
        d[f"n_{tag}"] = int(len(q))
    return d


fam_of = ptsdf.fluid.map(res.family)
sub_of = ptsdf.fluid.map(res.subgroup)
summary = {
    "All": pooled(pd.Series(True, index=ptsdf.index)),
    "< 0.90": pooled(ptsdf.Tr < 0.90),
    "0.50-0.70": pooled(ptsdf.Tr < 0.70),
}
for fam in FAMILY_ORDER:
    summary[fam] = pooled(fam_of == fam)
for sg in ("H-bonding", "polar aprotic"):
    summary[sg] = pooled(sub_of == sg)

# per-fluid largest changes (psat) for the generalized combinations
chg = {}
for tag in TAGS:
    d = res[f"psat_{tag}_src"] - res[f"psat_{tag}_ref"]
    chg[tag] = dict(max_increase=float(d.max()), fluid_max_increase=str(d.idxmax()),
                    max_decrease=float(d.min()), fluid_max_decrease=str(d.idxmin()),
                    mean_abs_change=float(d.abs().mean()), n_worse=int((d > 0).sum()), n_better=int((d < 0).sum()))
# pooled psat change vs |dpc|: the psat change per fluid correlates with the pc mismatch
corr = {}
for tag in ("srk_s", "pr_s", "srk_t95", "pr_ms"):
    dpf = res[f"psat_bias_{tag}_src"] - res[f"psat_bias_{tag}_ref"]
    corr[tag] = dict(corr_bias_change_vs_dpc=float(np.corrcoef(res.dpc_pct, dpf)[0, 1]),
                     slope_bias_change_vs_dpc=float(np.polyfit(res.dpc_pct, dpf, 1)[0]))

pd.set_option("display.width", 250)
print()
print("pooled AAD (%), all kept points: ref -> src")
for tag in TAGS:
    s = summary["All"]
    print(f"{tag:9s}  psat {s[f'psat_{tag}_ref']:.3f} -> {s[f'psat_{tag}_src']:.3f} | h {s[f'h_{tag}_ref']:.3f} -> {s[f'h_{tag}_src']:.3f}"
          f" | rho {s[f'rho_{tag}_ref']:.3f} -> {s[f'rho_{tag}_src']:.3f} | rhov {s[f'rhov_{tag}_ref']:.3f} -> {s[f'rhov_{tag}_src']:.3f}"
          f" | n={s[f'n_{tag}']}")
print(json.dumps(chg, indent=1))
print(json.dumps(corr, indent=1))
print("max |dTc| =", res.dTc_pct.abs().max(), res.dTc_pct.abs().idxmax(), "; max |dpc| =", res.dpc_pct.abs().max(), res.dpc_pct.abs().idxmax())
print("mean |dTc| =", res.dTc_pct.abs().mean(), "; mean |dpc| =", res.dpc_pct.abs().mean(), "; median |dpc| =", res.dpc_pct.abs().median())

# ------------------------------------------------------------ (b) sensitivities
EPS_T, EPS_P, EPS_W = 5e-4, 5e-4, 0.005
sens_rows = []
for fluid, g in df.groupby("fluid", sort=False):
    g = g.sort_values("Tr")
    Tc, pc, w = fl.loc[fluid, "Tc_K"], fl.loc[fluid, "pc_MPa"] * 1e6, fl.loc[fluid, "omega"]
    for cubic in ("SRK", "PR"):
        base = make(cubic, "s", fluid, Tc, pc, w)
        mT = (make(cubic, "s", fluid, Tc * (1 - EPS_T), pc, w), make(cubic, "s", fluid, Tc * (1 + EPS_T), pc, w))
        mP = (make(cubic, "s", fluid, Tc, pc * (1 - EPS_P), w), make(cubic, "s", fluid, Tc, pc * (1 + EPS_P), w))
        mW = (make(cubic, "s", fluid, Tc, pc, w - EPS_W), make(cubic, "s", fluid, Tc, pc, w + EPS_W))
        for _, r in g.iterrows():
            T = r.T_K
            vals = {}
            for name, (lo_m, hi_m), step in (("Tc", mT, 2 * EPS_T), ("pc", mP, 2 * EPS_P), ("w", mW, 2 * EPS_W)):
                out = {}
                for which, m in (("lo", lo_m), ("hi", hi_m)):
                    p, Zl, Zv = m.psat(T)
                    out[which] = dict(p=p, rho=p / (Zl * R * T), rhov=p / (Zv * R * T),
                                      h=m.h_res(T, p, Zv) - m.h_res(T, p, Zl))
                for prop in ("p", "rho", "rhov", "h"):
                    vals[f"dln{prop}_d{name}"] = (np.log(out["hi"][prop]) - np.log(out["lo"][prop])) / step
            sens_rows.append(dict(fluid=fluid, cubic=cubic, Tr=r.Tr, omega=w, **vals))
sens = pd.DataFrame(sens_rows)
sens.to_csv(os.path.join(DATA, "input_sensitivities.csv"), index=False)
# summary at the grid points nearest Tr = 0.50, 0.70, 0.90, 0.985 (fluids with Tr_min <= target)
summ = {}
for cubic in ("SRK", "PR"):
    summ[cubic] = {}
    for target in (0.50, 0.70, 0.90, 0.985):
        sel = []
        for fluid, g in sens[sens.cubic == cubic].groupby("fluid"):
            if g.Tr.min() > target + 0.013:
                continue
            sel.append(g.iloc[(g.Tr - target).abs().argmin()])
        sel = pd.DataFrame(sel)
        summ[cubic][f"Tr={target}"] = {c: dict(median=float(sel[c].median()), min=float(sel[c].min()),
                                               max=float(sel[c].max()), n=int(len(sel)))
                                       for c in sens.columns if c.startswith("dln")}
json.dump(dict(summary=summary, changes=chg, corr=corr, dropped=dropped, sensitivities=summ,
               eps=dict(Tc=EPS_T, pc=EPS_P, omega=EPS_W)),
          open(os.path.join(DATA, "compilation_constants.json"), "w"), indent=1)
for cubic in ("SRK", "PR"):
    print(cubic)
    for k, v in summ[cubic].items():
        print(" ", k, {c: (round(d["median"], 3), round(d["min"], 3), round(d["max"], 3)) for c, d in v.items()
                       if c in ("dlnp_dTc", "dlnp_dpc", "dlnp_dw", "dlnh_dTc", "dlnh_dpc", "dlnh_dw", "dlnrho_dTc", "dlnrho_dpc", "dlnrho_dw")})
print("wrote compilation_constants_per_fluid.csv, compilation_constants_points.csv, compilation_constants.json, input_sensitivities.csv")
