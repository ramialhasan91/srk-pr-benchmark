"""Physical diagnostics behind Sections 3 and 4 of the paper (tabS_physics, fig7 panel a):
(a) near-critical closure: distance of the liquid from the critical density at the top grid point, for the
    cubic and the reference, and the effective exponent of the coexistence-curve width (rho' - rho'') ~ (1 - Tr)^beta;
(b) sensitivity of psat and rho' to the attraction parameter at fixed T and b (PR- and SRK-Soave; Tr = 0.5, 0.7, 0.9);
(c) dependence of the untranslated liquid-density error on the reference Zc;
(d) effective exponent of the enthalpy of vaporization near Tc;
(e) spread of the generalized alphas around the classical one and its effect on psat and rho';
(f) heavy-alkane example (n-decane, Tr = 0.50): alpha change, sensitivity, psat change;
(g) Clapeyron split of the enthalpy-of-vaporization error into a volumetric (Z'' - Z') and a slope part;
(h) the Zc regression with the compressibility factor at the reducing point of each reference equation.
Reads  ../data/deviations_full.csv, fluids.csv, insample_points.csv
Writes ../data/physical_diagnostics.json, physics_nearcrit_closure.csv, physics_alpha_sensitivity.csv,
       physics_zc_density.csv, physics_hvap_exponent.csv, physics_alpha_spread.csv, physics_hvap_split.csv
"""
import json
import os
import sys

import platform

import CoolProp
import numpy as np
import pandas as pd
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
from eos import CubicEOS, R  # noqa: E402

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv")).set_index("fluid")
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                CoolProp=CoolProp.__version__)
res = {"versions": VERSIONS}

# ---------------------------------------------------------------- (a)
rows = []
for fluid, g in df.groupby("fluid", sort=False):
    g = g.sort_values("Tr")
    f = fl.loc[fluid]
    Tc, pc = f.Tc_K, f.pc_MPa * 1e6
    rc_ref = pc / (f.Zc_ref * R * Tc)
    top = g.iloc[-1]
    out = dict(fluid=fluid, family=f.family, subgroup=f.subgroup, Zc_ref=f.Zc_ref, Tr_top=top.Tr,
               red_ref=top.rho_ref_molm3 / rc_ref - 1.0)
    hi = g[g.Tr >= 0.93]
    x = np.log(1.0 - hi.Tr.to_numpy())
    w_ref = (hi.rho_ref_molm3 - hi.rhov_ref_molm3).to_numpy() / rc_ref
    out["beta_ref"] = float(np.polyfit(x, np.log(w_ref), 1)[0])
    for cub, Zc in (("srk", 1.0 / 3.0), ("pr", 0.3074)):
        rc_eos = pc / (Zc * R * Tc)
        rho_l = top[f"rho_{cub}_s_molm3"]
        out[f"red_{cub}"] = rho_l / rc_eos - 1.0
        rl = hi[f"rho_{cub}_s_molm3"].to_numpy()
        rv = (hi.rhov_ref_molm3 * (1.0 + hi[f"dev_rhov_{cub}_s"] / 100.0)).to_numpy()
        out[f"beta_{cub}"] = float(np.polyfit(x, np.log((rl - rv) / rc_eos), 1)[0])
        # liquid- and vapor-density deviations at the top point (Tr = 0.985) and the Zc limit of the liquid
        out[f"dev_top_{cub}"] = 100.0 * (rho_l / top.rho_ref_molm3 - 1.0)
        out[f"dev_top_rhov_{cub}"] = float(top[f"dev_rhov_{cub}_s"])
        out[f"limit_{cub}"] = 100.0 * (f.Zc_ref / Zc - 1.0)
    rows.append(out)
a = pd.DataFrame(rows)
a.to_csv(os.path.join(DATA, "physics_nearcrit_closure.csv"), index=False)
res["closure"] = {
    "n_fluids": int(len(a)),
    "cubic_closer_srk": int((a.red_srk < a.red_ref).sum()),
    "cubic_closer_pr": int((a.red_pr < a.red_ref).sum()),
    "median_red_ref": float(a.red_ref.median()), "median_red_srk": float(a.red_srk.median()),
    "median_red_pr": float(a.red_pr.median()),
    "beta_ref_median": float(a.beta_ref.median()), "beta_ref_range": [float(a.beta_ref.min()), float(a.beta_ref.max())],
    "beta_srk_median": float(a.beta_srk.median()), "beta_srk_range": [float(a.beta_srk.min()), float(a.beta_srk.max())],
    "beta_pr_median": float(a.beta_pr.median()), "beta_pr_range": [float(a.beta_pr.min()), float(a.beta_pr.max())],
    "dev_below_limit_srk": int((a.dev_top_srk < a.limit_srk).sum()),
    "dev_below_limit_pr": int((a.dev_top_pr < a.limit_pr).sum()),
    # mean signed deviations of both branches at the top point (for the vapor the two volumetric effects partly cancel)
    "mean_dev_top_rho_srk": float(a.dev_top_srk.mean()), "mean_dev_top_rho_pr": float(a.dev_top_pr.mean()),
    "mean_dev_top_rhov_srk": float(a.dev_top_rhov_srk.mean()), "mean_dev_top_rhov_pr": float(a.dev_top_rhov_pr.mean()),
}

# ---------------------------------------------------------------- (b)
EPS = 1e-4
rows = []
for fluid, f in fl.iterrows():
    for cub in ("SRK", "PR"):
        for trq in (0.5, 0.7, 0.9):
            if trq < f.Tr_min - 1e-9:
                continue
            T = trq * f.Tc_K
            vals = {}
            for sgn in (+1, -1):
                e = CubicEOS(cub, f.Tc_K, f.pc_MPa * 1e6, f.omega, alpha="s")
                e.Oa *= (1.0 + sgn * EPS)          # scales a(T) at fixed T and b
                p, Zl, Zv = e.psat(T)
                vals[sgn] = (np.log(p), np.log(p / (Zl * R * T)), np.log(p / (Zv * R * T)))
            d = [(vals[1][k] - vals[-1][k]) / (np.log(1 + EPS) - np.log(1 - EPS)) for k in range(3)]
            rows.append(dict(fluid=fluid, cubic=cub, Tr=trq, dlnp_dlna=d[0], dlnrhol_dlna=d[1], dlnrhov_dlna=d[2]))
b = pd.DataFrame(rows)
b.to_csv(os.path.join(DATA, "physics_alpha_sensitivity.csv"), index=False)
sens = {}
for (cub, trq), g in b.groupby(["cubic", "Tr"]):
    sens[f"{cub}|{trq}"] = dict(n=int(len(g)),
                                dlnp_median=float(g.dlnp_dlna.median()), dlnp_min=float(g.dlnp_dlna.min()), dlnp_max=float(g.dlnp_dlna.max()),
                                dlnrhol_median=float(g.dlnrhol_dlna.median()), dlnrhol_min=float(g.dlnrhol_dlna.min()),
                                dlnrhol_max=float(g.dlnrhol_dlna.max()),
                                ratio_median=float((g.dlnp_dlna / g.dlnrhol_dlna).abs().median()))
res["alpha_sensitivity"] = sens

# ---------------------------------------------------------------- (c)
rows = []
for fluid, g in df.groupby("fluid", sort=False):
    f = fl.loc[fluid]
    lo = g[g.Tr < 0.70]
    rows.append(dict(fluid=fluid, family=f.family, Zc_ref=f.Zc_ref,
                     bias_lo_pr=float(lo.dev_rho_pr_s.mean()), bias_lo_srk=float(lo.dev_rho_srk_s.mean()),
                     bias_all_pr=float(g.dev_rho_pr_s.mean()), bias_all_srk=float(g.dev_rho_srk_s.mean()),
                     bias_lo_pr_vt=float(lo.dev_rho_pr_s_vt.mean())))
c = pd.DataFrame(rows)
c.to_csv(os.path.join(DATA, "physics_zc_density.csv"), index=False)
zc = {}
for col in ("bias_lo_pr", "bias_lo_srk", "bias_all_pr", "bias_all_srk"):
    cc = c.dropna(subset=[col])
    r = float(np.corrcoef(cc.Zc_ref, cc[col])[0, 1])
    slope, icpt = np.polyfit(cc.Zc_ref, cc[col], 1)
    zc[col] = dict(r=r, n=int(len(cc)), slope_pct_per_0p01Zc=float(slope * 0.01), Zc_at_zero=float(-icpt / slope))
res["zc_density"] = zc
res["zc_density"]["pr_lo_positive_fluids"] = sorted(c[c.bias_lo_pr > 0].fluid.tolist())
res["zc_density"]["pr_lo_positive_min_Zc"] = float(c[c.bias_lo_pr > 0].Zc_ref.min())
res["zc_density"]["pr_lo_negative_max_Zc"] = float(c[c.bias_lo_pr < 0].Zc_ref.max())

# ---------------------------------------------------------------- (d)
# effective exponent of the enthalpy of vaporization near Tc: hvap ~ (1 - Tr)^beta_h
rows = []
for fluid, g in df.groupby("fluid", sort=False):
    hi = g[g.Tr >= 0.93].sort_values("Tr")
    x = np.log(1.0 - hi.Tr.to_numpy())
    out = dict(fluid=fluid, beta_h_ref=float(np.polyfit(x, np.log(hi.hvap_ref_Jmol.to_numpy()), 1)[0]))
    for cub in ("srk", "pr"):
        out[f"beta_h_{cub}"] = float(np.polyfit(x, np.log(hi[f"hvap_{cub}_s_Jmol"].to_numpy()), 1)[0])
    rows.append(out)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(DATA, "physics_hvap_exponent.csv"), index=False)
res["hvap_exponent"] = {k: dict(median=float(d[k].median()), min=float(d[k].min()), max=float(d[k].max()))
                        for k in ("beta_h_ref", "beta_h_srk", "beta_h_pr")}

# ---------------------------------------------------------------- (e)
# spread of the generalized alphas around the classical one and its effect on psat and rho', point by point
GEN = ("s19", "t95", "tc", "coq", "ms")
rows = []
for fluid, f in fl.iterrows():
    g = df[df.fluid == fluid]
    for cub in ("SRK", "PR"):
        base = CubicEOS(cub, f.Tc_K, f.pc_MPa * 1e6, f.omega, alpha="s")
        a0 = np.array([base.alpha(T)[0] for T in g.T_K])
        for al in GEN:
            e = CubicEOS(cub, f.Tc_K, f.pc_MPa * 1e6, f.omega, alpha=al)
            a1 = np.array([e.alpha(T)[0] for T in g.T_K])
            c = cub.lower()
            rows.append(dict(fluid=fluid, cubic=cub, alpha=al,
                             max_dalpha_pct=float(np.max(np.abs(a1 / a0 - 1.0)) * 100),
                             max_dpsat_pts=float(np.max(np.abs(g[f"dev_p_{c}_{al}"] - g[f"dev_p_{c}_s"]))),
                             max_drho_pts=float(np.max(np.abs(g[f"dev_rho_{c}_{al}"] - g[f"dev_rho_{c}_s"])))))
e_ = pd.DataFrame(rows)
e_.to_csv(os.path.join(DATA, "physics_alpha_spread.csv"), index=False)
res["alpha_spread"] = dict(max_dalpha_pct=float(e_.max_dalpha_pct.max()),
                           median_of_max_dalpha_pct=float(e_.max_dalpha_pct.median()),
                           p90_of_max_dalpha_pct=float(e_.max_dalpha_pct.quantile(0.9)),
                           max_dpsat_pts=float(e_.max_dpsat_pts.max()),
                           max_drho_pts=float(e_.max_drho_pts.max()),
                           median_of_max_dpsat_pts=float(e_.max_dpsat_pts.median()),
                           median_of_max_drho_pts=float(e_.max_drho_pts.median()))

# ---------------------------------------------------------------- (f)
# heavy-alkane example: the larger low-temperature alpha of Twu-95 and MS removes the PR overprediction of n-decane at Tr = 0.50
ex = {}
f = fl.loc["n-decane"]
g0 = df[df.fluid == "n-decane"].sort_values("Tr").iloc[0]
T0 = float(g0.T_K)
for cub in ("SRK", "PR"):
    c = cub.lower()
    base = CubicEOS(cub, f.Tc_K, f.pc_MPa * 1e6, f.omega, alpha="s")
    p0 = base.psat(T0)[0]
    vals = {}
    for sgn in (+1, -1):
        e = CubicEOS(cub, f.Tc_K, f.pc_MPa * 1e6, f.omega, alpha="s")
        e.Oa *= (1.0 + sgn * EPS)
        vals[sgn] = np.log(e.psat(T0)[0])
    out = dict(Tr=float(g0.Tr), dlnp_dlna=float((vals[1] - vals[-1]) / (np.log(1 + EPS) - np.log(1 - EPS))),
               dev_classical=float(g0[f"dev_p_{c}_s"]))
    for al in ("t95", "ms"):
        e = CubicEOS(cub, f.Tc_K, f.pc_MPa * 1e6, f.omega, alpha=al)
        out[f"dalpha_pct_{al}"] = float(100 * (e.alpha(T0)[0] / base.alpha(T0)[0] - 1))
        out[f"dpsat_pct_{al}"] = float(100 * (e.psat(T0)[0] / p0 - 1))
        out[f"dev_{al}"] = float(g0[f"dev_p_{c}_{al}"])
    ex[cub] = out
res["decane_example"] = ex

# ---------------------------------------------------------------- (g)
# Clapeyron split: dh = R T^2 dZ dlnp/dT with dZ = Z'' - Z', so ln(dh_EoS/dh_ref) = ln(dZ_EoS/dZ_ref) + ln(slope_EoS/slope_ref);
#     the first part is volumetric, the second follows the alpha function
ins = pd.read_csv(os.path.join(DATA, "insample_points.csv"))
mm = df.merge(ins.drop(columns=["family", "subgroup", "T_K"]), on=["fluid", "Tr"])
T_, p_, rl_, rv_ = mm.T_K, mm.p_ref_Pa, mm.rho_ref_molm3, mm.rhov_ref_molm3
dZr = p_ / (rv_ * R * T_) - p_ / (rl_ * R * T_)
RANGES = {"lo": mm.Tr < 0.70, "mid": (mm.Tr >= 0.70) & (mm.Tr < 0.90), "nc": mm.Tr >= 0.90}
rows = []
for cub in ("srk", "pr"):
    for al in ("s", "s19", "t95", "tc", "coq", "ms", "tf", "sfit", "tfit", "tunc"):
        pE = p_ * (1 + mm[f"dev_p_{cub}_{al}"] / 100)
        rvE = rv_ * (1 + mm[f"dev_rhov_{cub}_{al}"] / 100)
        rlE = rl_ * (1 + mm[f"dev_rho_{cub}_{al}"] / 100)
        lh = np.log(1 + mm[f"dev_h_{cub}_{al}"] / 100)
        lz = np.log((pE / (rvE * R * T_) - pE / (rlE * R * T_)) / dZr)
        lzv = np.log((pE / (rvE * R * T_)) / (p_ / (rv_ * R * T_)))
        for rg, msk in RANGES.items():
            rows.append(dict(cubic=cub, alpha=al, range=rg, h=float(100 * lh[msk].mean()), dZ=float(100 * lz[msk].mean()),
                             slope=float(100 * (lh - lz)[msk].mean()), Zv=float(100 * lzv[msk].mean())))
gsp = pd.DataFrame(rows)
gsp.to_csv(os.path.join(DATA, "physics_hvap_split.csv"), index=False)
SEVEN = ("s", "s19", "t95", "tc", "coq", "ms", "tf")
hs = {}
for cub in ("srk", "pr"):
    for rg in RANGES:
        g7 = gsp[(gsp.cubic == cub) & (gsp.range == rg) & gsp.alpha.isin(SEVEN)]
        gi = gsp[(gsp.cubic == cub) & (gsp.range == rg) & (gsp.alpha == "tfit")].iloc[0]
        hs[f"{cub}|{rg}"] = {q: [float(g7[q].min()), float(g7[q].max())] for q in ("h", "dZ", "slope", "Zv")}
        hs[f"{cub}|{rg}"]["tfit"] = {q: float(gi[q]) for q in ("h", "dZ", "slope", "Zv")}
        hs[f"{cub}|{rg}"]["dZ_share_min"] = float((g7.dZ / g7.h).min())
res["hvap_split"] = hs

# ---------------------------------------------------------------- (h)
# Zc regression with the reducing-point compressibility factor, which differs from the critical-point value only where the
#     reference equation's critical point is displaced from its reducing point (methyl chloride, n-heptane; marginally oxygen, R-134a)
import CoolProp.CoolProp as CP  # noqa: E402
from fluid_set import FLUIDS  # noqa: E402
red = {}
for cpn, (lab, _, _) in FLUIDS.items():
    Tred, rred = CP.PropsSI("T_reducing", cpn), CP.PropsSI("rhomolar_reducing", cpn)
    pred = CP.PropsSI("P", "T", Tred, "Dmolar", rred, cpn)
    red[lab] = dict(T_red=Tred, p_red=pred, Zc_red=pred / (rred * R * Tred),
                    dTc_pct=100 * (fl.loc[lab].Tc_K / Tred - 1), dpc_pct=100 * (fl.loc[lab].pc_MPa * 1e6 / pred - 1))
cz = pd.read_csv(os.path.join(DATA, "physics_zc_density.csv")).dropna(subset=["bias_lo_pr"]).reset_index(drop=True)
cz["Zc_red"] = [red[f]["Zc_red"] for f in cz.fluid]
zr = {}
for col in ("bias_lo_srk", "bias_lo_pr"):
    s1, i1 = np.polyfit(cz.Zc_ref, cz[col], 1)
    resid = cz[col] - (s1 * cz.Zc_ref + i1)
    order = resid.abs().sort_values(ascending=False).index
    z2 = cz.Zc_ref.where(~cz.fluid.isin(["methyl chloride", "n-heptane"]), cz.Zc_red)
    zr[col] = dict(r_red=float(np.corrcoef(cz.Zc_red, cz[col])[0, 1]), r_red_two=float(np.corrcoef(z2, cz[col])[0, 1]),
                   farthest=[cz.fluid[k] for k in order[:3]],
                   resid={cz.fluid[k]: float(resid[k]) for k in order[:3]},
                   equiv_Zc={cz.fluid[k]: float(cz.Zc_ref[k] + resid[k] / s1) for k in order[:3]})
res["zc_reducing"] = dict(regression=zr, displaced={f: red[f] for f in ("methyl chloride", "n-heptane", "oxygen", "R-134a")},
                          n_dTc_above_0p05pct=int(sum(abs(v["dTc_pct"]) > 0.05 for v in red.values())))

json.dump(res, open(os.path.join(DATA, "physical_diagnostics.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
