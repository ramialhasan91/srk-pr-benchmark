"""
Benchmark of the SRK and PR cubic equations of state, each combined with seven
alpha functions and with a constant volume translation, against reference
multiparameter (Helmholtz-energy) equations of state as implemented in
CoolProp, for saturation pressure, saturated-liquid density, saturated-vapor
density, and enthalpy of vaporization.

Models (tag = <cubic>_<alpha>)
------------------------------
Cubics : srk, pr
Alphas : s    classical Soave form, original m(omega) (Soave 1972 / PR 1976)
         s19  Soave form, updated generalized m(omega) (Pina-Martinez et al. 2019)
         t95  generalized Twu alpha function (Twu et al. 1995)
         tc   generalized consistent Twu-91 (N = 2; PR: Pina-Martinez et al.
              2022, SRK: Pina-Martinez et al. 2018)
         coq  exponential alpha function of Coquelet et al. 2004, generalized
              coefficients as compiled by Xiao & Yang 2025
         ms   consistent generalized alpha function of Mahmoodi & Sedigh 2017,
              coefficients as compiled by Xiao & Yang 2025
         tf   component-specific consistent Twu-91 parameters from the
              1800-fluid compilation of Pina-Martinez et al. 2022
              (data/twu_fit_params.csv, verified against its SI Table S3).
              NOT predictive: fitted reference.
The first six alphas require nothing beyond (Tc, pc, omega).

Volume translation (density only), v_t = v - c, "translation ladder":
  c_corr  zero-data: translated liquid volume matches the Rackett prediction
          at Tr = 0.70 with Z_RA = 0.29056 - 0.08775 omega (Yamada-Gunn), i.e.
          the Peneloux (1982) construction;
  c_fit   one datum: matches the REFERENCE liquid volume at Tr = 0.70 (or at
          the lowest gridded Tr if the triple point lies above 0.70 Tc);
  c_zc    critical datum: matches the reference critical volume,
          c = (Zc_model - Zc_ref) R Tc / pc (diagnostic only).
The ladder is evaluated for every (cubic, alpha) combination; the paper
reports it for the classical-alpha variants (VT_ALPHA) and quotes the alpha
sensitivity. Translated saturated-vapor densities are recorded for the
density-anchored and the Rackett-anchored constants.

Validation
----------
* Volume-translation invariance: ln(phi_translated) = ln(phi) - c p/(RT)
  checked by quadrature of the exact departure integral (both phases, all
  fluids, three temperatures, both cubics).
* Clapeyron self-consistency: the departure-function enthalpy of vaporization
  is checked against T (v'' - v') dpsat/dT (central differences of the solved
  saturation pressure) at EVERY state point for all 14 combinations.
* Anchor sensitivity: the pooled translated liquid-density AAD is recomputed
  with the anchoring temperature swept over the whole grid (both cubics).

Protocol
--------
For each fluid: 40 reduced temperatures, Tr in [max(0.50, Tt/Tc + 0.01), 0.985].
Critical constants (Tc, pc) and acentric factor omega are taken from the
reference EoS itself (via CoolProp), so deviations reflect model form only.
The Halm-Stiel polar factor chi = log10[psat_ref(0.6 Tc)/pc] + 1.70 omega + 1.552
is evaluated from the reference EoS as a descriptor of each fluid's departure
from two-parameter corresponding states (undefined when 0.6 Tc < Tt).

Outputs (../data/):
  deviations_full.csv   per-point results and % deviations
  fluids.csv            fluid metadata, polar factor, translation constants
  anchor_sweep.csv      pooled translated liquid-density AAD vs anchoring Tr
  key_numbers.json      headline statistics quoted in the manuscript
  run_metadata.json     library versions, solver settings, convergence record,
                        and the two validation checks
"""

import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import CoolProp
from CoolProp.CoolProp import PropsSI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eos import R, CubicEOS, ALPHA_LABELS  # noqa: E402
from fluid_set import FLUIDS, FAMILY_ORDER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

N_T = 40
TR_MAX = 0.985
TR_ANCHOR = 0.70
PSAT_TOL = 1e-11          # convergence tolerance on the fugacity ratio
PSAT_TOL_FALLBACK = 1e-7  # accepted near Tc if the iteration limit is hit
CLAP_H = 1e-4             # relative step for the Clapeyron finite difference

CUBICS = ["SRK", "PR"]
ALPHAS = ["s", "s19", "t95", "tc", "coq", "ms", "tf"]
TAGS = [f"{c.lower()}_{a}" for c in CUBICS for a in ALPHAS]
PREDICTIVE_TAGS = [t for t in TAGS if not t.endswith("_tf")]
VT_ALPHA = "s"            # alpha used for the translation ladder in the paper (classical Peneloux practice); alpha sensitivity is reported

tf = pd.read_csv(os.path.join(DATA, "twu_fit_params.csv"), comment="#").set_index("fluid")


def pct(model, ref):
    return (model - ref) / ref * 100.0


def v_rackett(Tr, Tc, pc, Z_RA):
    """Rackett saturated-liquid molar volume with a correlated Z_RA."""
    return R * Tc / pc * Z_RA ** (1.0 + (1.0 - Tr) ** (2.0 / 7.0))


def make_models(disp, Tc, pc, w):
    models = {}
    for cubic in CUBICS:
        for a in ALPHAS:
            kw = {}
            if a == "tf":
                kw["twu_params"] = (tf.loc[disp, f"L_{cubic}"], tf.loc[disp, f"M_{cubic}"],
                                    tf.loc[disp, f"N_{cubic}"])
            models[f"{cubic.lower()}_{a}"] = CubicEOS(cubic, Tc, pc, w, alpha=a, **kw)
    return models


rows, meta = [], []
clap_rel = {tag: [] for tag in TAGS}
clap_skipped = 0
vt_nonphysical = 0

for cp_name, (disp, family, subgroup) in FLUIDS.items():
    Tc = PropsSI("Tcrit", cp_name)
    pc = PropsSI("pcrit", cp_name)
    w = PropsSI("acentric", cp_name)
    Tt = PropsSI("Ttriple", cp_name)
    rhoc_ref = PropsSI("rhomolar_critical", cp_name)  # mol/m^3
    Zc_ref = pc / (rhoc_ref * R * Tc)

    Tr_min = max(0.50, Tt / Tc + 0.01)
    Tr_grid = np.linspace(Tr_min, TR_MAX, N_T)

    # Halm-Stiel polar factor from the reference EoS (NaN if 0.6 Tc < Tt)
    if 0.60 * Tc > Tt:
        chi = np.log10(PropsSI("P", "T", 0.60 * Tc, "Q", 0, cp_name) / pc) + 1.70 * w + 1.552
    else:
        chi = np.nan

    models = make_models(disp, Tc, pc, w)

    # ---- translation constants, per (cubic, alpha) ----------------------- #
    Tr_anchor = max(TR_ANCHOR, Tr_min)
    T_anchor = Tr_anchor * Tc
    v_ref_anchor = 1.0 / PropsSI("Dmolar", "T", T_anchor, "Q", 0, cp_name)
    Z_RA = 0.29056 - 0.08775 * w
    v_ra_anchor = v_rackett(Tr_anchor, Tc, pc, Z_RA)
    cvals = {}
    for tag, m in models.items():
        v_anchor, _ = m.sat_liq_volume(T_anchor)
        cvals[tag] = dict(fit=v_anchor - v_ref_anchor,           # one datum
                          corr=v_anchor - v_ra_anchor,           # zero data
                          zc=(m.Zc - Zc_ref) * R * Tc / pc)      # critical datum

    m_meta = dict(fluid=disp, coolprop=cp_name, family=family, subgroup=subgroup,
                  Tc_K=Tc, pc_MPa=pc / 1e6, omega=w, Zc_ref=Zc_ref, Z_RA=Z_RA,
                  chi_HalmStiel=chi, Tt_K=Tt, Tr_min=Tr_min, Tr_anchor=Tr_anchor)
    for cubic in CUBICS:
        cl = cubic.lower()
        for a in ALPHAS:
            m_meta[f"c_fit_{cl}_{a}_cm3mol"] = cvals[f"{cl}_{a}"]["fit"] * 1e6
            m_meta[f"c_corr_{cl}_{a}_cm3mol"] = cvals[f"{cl}_{a}"]["corr"] * 1e6
        m_meta[f"c_zc_{cl}_cm3mol"] = cvals[f"{cl}_{VT_ALPHA}"]["zc"] * 1e6
        m_meta[f"c_lit_{cl}_cm3mol"] = float(tf.loc[disp, f"c_{cubic}_cm3mol"])
        m_meta[f"L_tf_{cl}"] = float(tf.loc[disp, f"L_{cubic}"])
        m_meta[f"M_tf_{cl}"] = float(tf.loc[disp, f"M_{cubic}"])
        m_meta[f"N_tf_{cl}"] = float(tf.loc[disp, f"N_{cubic}"])
        gm = models[f"{cl}_tc"].twu_single
        m_meta[f"L_tc_{cl}"], m_meta[f"M_tc_{cl}"] = gm[0], gm[1]
    meta.append(m_meta)

    for Tr in Tr_grid:
        T = Tr * Tc
        p_ref = PropsSI("P", "T", T, "Q", 0, cp_name)
        rho_ref = PropsSI("Dmolar", "T", T, "Q", 0, cp_name)    # mol/m^3
        rhov_ref = PropsSI("Dmolar", "T", T, "Q", 1, cp_name)   # mol/m^3
        h_ref = (PropsSI("Hmolar", "T", T, "Q", 1, cp_name)
                 - PropsSI("Hmolar", "T", T, "Q", 0, cp_name))  # J/mol

        rec = dict(fluid=disp, family=family, subgroup=subgroup, Tr=Tr, T_K=T,
                   p_ref_Pa=p_ref, rho_ref_molm3=rho_ref,
                   rhov_ref_molm3=rhov_ref, hvap_ref_Jmol=h_ref)

        for tag, model in models.items():
            p, Zl, Zv = model.psat(T)
            cols_nan = [f"p_{tag}_Pa", f"dev_p_{tag}", f"rho_{tag}_molm3",
                        f"dev_rho_{tag}", f"dev_rhov_{tag}", f"hvap_{tag}_Jmol",
                        f"dev_h_{tag}", f"dev_rho_{tag}_vt", f"dev_rho_{tag}_vtc",
                        f"dev_rhov_{tag}_vt", f"dev_rhov_{tag}_vtc"]
            if tag.endswith(f"_{VT_ALPHA}"):
                cols_nan.append(f"dev_rho_{tag}_vtz")
            if not np.isfinite(p):
                for col in cols_nan:
                    rec[col] = np.nan
                continue
            v_l = Zl * R * T / p
            v_v = Zv * R * T / p
            h_vap = model.h_res(T, p, Zv) - model.h_res(T, p, Zl)
            rec[f"p_{tag}_Pa"] = p
            rec[f"dev_p_{tag}"] = pct(p, p_ref)
            rec[f"rho_{tag}_molm3"] = 1.0 / v_l
            rec[f"dev_rho_{tag}"] = pct(1.0 / v_l, rho_ref)
            rec[f"dev_rhov_{tag}"] = pct(1.0 / v_v, rhov_ref)
            rec[f"hvap_{tag}_Jmol"] = h_vap
            rec[f"dev_h_{tag}"] = pct(h_vap, h_ref)
            # Clapeyron self-consistency at every point
            hstep = CLAP_H * T
            p_lo, _, _ = model.psat(T - hstep)
            p_hi, _, _ = model.psat(T + hstep)
            if np.isfinite(p_lo) and np.isfinite(p_hi):
                h_clap = T * (v_v - v_l) * (p_hi - p_lo) / (2.0 * hstep)
                clap_rel[tag].append(abs(h_clap / h_vap - 1.0))
            else:
                clap_skipped += 1
            # translated densities (liquid: three constants; vapor: the two
            # low-temperature-anchored constants)
            c = cvals[tag]
            for suffix, cc in (("vt", c["fit"]), ("vtc", c["corr"])):
                rec[f"dev_rho_{tag}_{suffix}"] = pct(1.0 / (v_l - cc), rho_ref)
                rec[f"dev_rhov_{tag}_{suffix}"] = pct(1.0 / (v_v - cc), rhov_ref)
            if tag.endswith(f"_{VT_ALPHA}"):
                # Translated volume set to NaN when v - c <= 0 (occurs only for
                # the diagnostic Zc-matched constant at low Tr).
                if v_l - c["zc"] > 0.0:
                    rec[f"dev_rho_{tag}_vtz"] = pct(1.0 / (v_l - c["zc"]), rho_ref)
                else:
                    rec[f"dev_rho_{tag}_vtz"] = np.nan
                    vt_nonphysical += 1
        rows.append(rec)

df = pd.DataFrame(rows)
fl = pd.DataFrame(meta)

df.to_csv(os.path.join(DATA, "deviations_full.csv"), index=False)
fl.to_csv(os.path.join(DATA, "fluids.csv"), index=False)

# --------------------------------------------------------------------- #
# Sanity checks and convergence accounting
n_expected = len(FLUIDS) * N_T
fail_counts = {tag: int(df[f"dev_p_{tag}"].isna().sum()) for tag in TAGS}
prop = df[df.fluid == "propane"]
prop_check = prop.iloc[(prop.Tr - 0.7).abs().argmin()]
assert abs(prop_check.dev_p_pr_s) < 5.0, "PR propane psat sanity check failed"
print(f"State points: {len(df)} (expected {n_expected}); "
      f"failed psat solves per model: {fail_counts}")

clap_all = np.concatenate([np.asarray(v) for v in clap_rel.values()])
clap_stats = {
    "relative_step": CLAP_H,
    "n_checks": int(clap_all.size),
    "n_skipped": int(clap_skipped),
    "max_rel_diff": float(np.max(clap_all)),
    "median_rel_diff": float(np.median(clap_all)),
    "per_model_max": {tag: float(np.max(v)) for tag, v in clap_rel.items()},
}
print(f"Clapeyron check: {clap_stats['n_checks']} points, "
      f"max |rel diff| = {clap_stats['max_rel_diff']:.2e}, "
      f"median = {clap_stats['median_rel_diff']:.2e}")
assert clap_stats["max_rel_diff"] < 1e-3, "Clapeyron consistency check failed"

# --------------------------------------------------------------------- #
# Numerical verification that a constant volume translation leaves the
# fugacity-equality condition (hence psat and Delta h_vap) unchanged, for
# both cubics (classical Soave alpha, VT_ALPHA), every fluid, three temperatures, both phases.
vt_residuals = []
for cp_name, (disp, family, subgroup) in FLUIDS.items():
    row = fl[fl.fluid == disp].iloc[0]
    Tc, pc, w = PropsSI("Tcrit", cp_name), PropsSI("pcrit", cp_name), PropsSI("acentric", cp_name)
    for cubic in CUBICS:
        m = CubicEOS(cubic, Tc, pc, w, alpha=VT_ALPHA)
        c = row[f"c_fit_{cubic.lower()}_{VT_ALPHA}_cm3mol"] * 1e-6
        for Tr in (max(0.55, row.Tr_min), 0.75, 0.95):
            T = Tr * Tc
            p, Zl, Zv = m.psat(T)
            if not np.isfinite(p):
                continue
            A, B = m.AB(T, p)
            for Z in (Zl, Zv):
                lhs = m.lnphi_translated_quadrature(T, p, Z, c)
                rhs = m.lnphi(Z, A, B) - c * p / (R * T)
                vt_residuals.append(abs(lhs - rhs))
vt_max_residual = float(np.max(vt_residuals))
print(f"VT invariance identity: {len(vt_residuals)} checks, "
      f"max |residual| in ln(phi) = {vt_max_residual:.2e}")
assert vt_max_residual < 1e-8, "volume-translation invariance check failed"

# --------------------------------------------------------------------- #
# Anchor-sensitivity sweep (both cubics, classical alpha): pooled translated
# liquid-density AAD as a function of the anchoring reduced temperature (per
# fluid, the anchor is the grid point nearest max(Tr_a, Tr_min)).
ANCHORS = np.linspace(0.50, TR_MAX, N_T)
sweep = []
for cubic in CUBICS:
    tag = f"{cubic.lower()}_{VT_ALPHA}"
    per_fluid = {}
    for f, g in df.groupby("fluid"):
        g = g.sort_values("Tr")
        per_fluid[f] = (g.Tr.to_numpy(), 1.0 / g[f"rho_{tag}_molm3"].to_numpy(),
                        g.rho_ref_molm3.to_numpy())
    for Tr_a in ANCHORS:
        devs = []
        for f, (Tr_g, v_g, rho_ref_g) in per_fluid.items():
            idx = int(np.argmin(np.abs(Tr_g - max(Tr_a, Tr_g[0]))))
            c = v_g[idx] - 1.0 / rho_ref_g[idx]
            devs.append(np.abs(pct(1.0 / (v_g - c), rho_ref_g)))
        sweep.append((cubic, Tr_a, float(np.nanmean(np.concatenate(devs)))))
sweep_df = pd.DataFrame(sweep, columns=["cubic", "Tr_anchor", "aad_rho_vt"])
sweep_df.to_csv(os.path.join(DATA, "anchor_sweep.csv"), index=False)
anchor_summary = {}
for cubic, s in sweep_df.groupby("cubic"):
    s = s.reset_index(drop=True)
    i_min = int(s.aad_rho_vt.idxmin())
    i_070 = int((s.Tr_anchor - 0.70).abs().idxmin())
    flat = s.Tr_anchor[s.aad_rho_vt <= s.aad_rho_vt[i_min] + 0.25]
    untrans = float(np.nanmean(np.abs(df[f"dev_rho_{cubic.lower()}_{VT_ALPHA}"])))
    worse = s.Tr_anchor[s.aad_rho_vt > untrans]
    anchor_summary[cubic] = {
        "aad_min": float(s.aad_rho_vt[i_min]), "Tr_at_min": float(s.Tr_anchor[i_min]),
        "aad_at_070": float(s.aad_rho_vt[i_070]),
        "flat_window_Tr": [float(flat.min()), float(flat.max())],
        "first_Tr_worse_than_untranslated": float(worse.min()) if len(worse) else None,
        "aad_at_top": float(s.aad_rho_vt.iloc[-1]),
    }
print("Anchor sweep:", json.dumps(anchor_summary, indent=1))

with open(os.path.join(DATA, "run_metadata.json"), "w") as fjson:
    json.dump({
        "python": platform.python_version(),
        "coolprop": CoolProp.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_fluids": len(FLUIDS),
        "n_T_per_fluid": N_T,
        "n_points": int(len(df)),
        "Tr_max": TR_MAX,
        "models": TAGS,
        "alpha_labels": ALPHA_LABELS,
        "psat_solver": "successive substitution on fugacity ratio, Wilson start",
        "psat_tol_fugacity_ratio": PSAT_TOL,
        "psat_tol_fallback_near_Tc": PSAT_TOL_FALLBACK,
        "failed_psat_solves": fail_counts,
        "vt_nonphysical_points_zc_constant": vt_nonphysical,
        "clapeyron_check": clap_stats,
        "vt_invariance_checks": len(vt_residuals),
        "vt_invariance_max_abs_residual_lnphi": vt_max_residual,
        "anchor_sweep": anchor_summary,
    }, fjson, indent=2)

# --------------------------------------------------------------------- #
# Headline statistics for the manuscript
def aad(s):
    return float(np.nanmean(np.abs(s)))


def bias(s):
    return float(np.nanmean(s))


BIN_EDGES = [0.50, 0.70, 0.90, 0.9851]
BIN_LABELS = ["0.50-0.70", "0.70-0.90", "0.90-0.985"]
df["Tr_bin"] = pd.cut(df.Tr, BIN_EDGES, right=False, labels=BIN_LABELS)
df["Tr_bin_fine"] = pd.cut(df.Tr, [0.50, 0.70, 0.90, 0.95, 0.9851], right=False,
                           labels=["0.50-0.70", "0.70-0.90", "0.90-0.95", "0.95-0.985"])

PROPS = {"psat": "dev_p", "rho": "dev_rho", "rhov": "dev_rhov", "h": "dev_h"}
VT_SUFFIXES = ["vt", "vtc"]


def block(g):
    """All headline AADs for a subset g of the point table."""
    out = {}
    for prop, col in PROPS.items():
        for t in TAGS:
            out[f"{prop}_{t}"] = aad(g[f"{col}_{t}"])
        if prop == "rho":
            for t in TAGS:
                out[f"rho_bias_{t}"] = bias(g[f"dev_rho_{t}"])
                for sfx in VT_SUFFIXES:
                    out[f"rho_{t}_{sfx}"] = aad(g[f"dev_rho_{t}_{sfx}"])
            for cubic in CUBICS:
                t = f"{cubic.lower()}_{VT_ALPHA}"
                out[f"rho_{t}_vtz"] = aad(g[f"dev_rho_{t}_vtz"])
        if prop == "rhov":
            for t in TAGS:
                for sfx in VT_SUFFIXES:
                    out[f"rhov_{t}_{sfx}"] = aad(g[f"dev_rhov_{t}_{sfx}"])
    return out


key = {
    "n_fluids": len(FLUIDS), "n_points": int(len(df)), "models": TAGS,
    "failed_psat_solves": fail_counts,
    "overall": block(df),
    "below_090": block(df[df.Tr < 0.90]),
    "by_family": {fam: block(g) for fam, g in df.groupby("family")},
    "by_subgroup": {sg: block(g) for sg, g in df.groupby("subgroup") if sg},
    "by_bin": {str(b): block(g) for b, g in df.groupby("Tr_bin", observed=True)},
    "by_bin_fine": {str(b): block(g) for b, g in df.groupby("Tr_bin_fine", observed=True)},
    "per_fluid": {},
}
for f, g in df.groupby("fluid"):
    key["per_fluid"][f] = block(g)

# vapor density mirrors psat at low Tr:
lo = df[df.Tr < 0.70]
key["overall"]["corr_rhov_psat_pr_s_lowTr"] = float(np.corrcoef(lo.dev_rhov_pr_s, lo.dev_p_pr_s)[0, 1])
key["overall"]["corr_rhov_psat_pr_tc_lowTr"] = float(np.corrcoef(lo.dev_rhov_pr_tc, lo.dev_p_pr_tc)[0, 1])

# Deviation near the acentric-factor anchor Tr = 0.7 (grid points within 0.02)
near07 = df.loc[(df.Tr - 0.70).abs() < 0.02]
key["near_Tr07_psat_AAD"] = {t: aad(near07[f"dev_p_{t}"]) for t in TAGS}

# Per-fluid win counts for a set of pairwise comparisons (sign tests in 04)
def wins(prop, a, b):
    return int(sum(key["per_fluid"][f][f"{prop}_{a}"] < key["per_fluid"][f][f"{prop}_{b}"]
                   for f in key["per_fluid"]))


key["win_counts"] = {}
for prop in ("psat", "h", "rho", "rhov"):
    key["win_counts"][prop] = {f"pr_{a}_vs_srk_{a}": wins(prop, f"pr_{a}", f"srk_{a}") for a in ALPHAS}
    for cubic in ("srk", "pr"):
        for j, a in enumerate(ALPHAS):          # every pair, later alpha vs earlier alpha
            for b in ALPHAS[:j]:
                key["win_counts"][prop][f"{cubic}_{a}_vs_{cubic}_{b}"] = wins(
                    prop, f"{cubic}_{a}", f"{cubic}_{b}")

# Extreme points quoted in the text
def extreme(fluid, tag, col="dev_p", Tr_target=0.50):
    g = df[df.fluid == fluid]
    r = g.iloc[(g.Tr - Tr_target).abs().argmin()]
    return float(r[f"{col}_{tag}"])


key["spot_values"] = {
    "n-decane_Tr050_dev_p": {t: extreme("n-decane", t) for t in TAGS},
    "water_Tr050_dev_p": {t: extreme("water", t) for t in TAGS},
    "methanol_Tr050_dev_p": {t: extreme("methanol", t) for t in TAGS},
    "methane_rho_AAD": {t: key["per_fluid"]["methane"][f"rho_{t}"] for t in TAGS},
}

# Alpha sensitivity of the translated liquid density (max over alphas of the
# difference in pooled AAD relative to the classical-alpha ladder)
key["vt_alpha_sensitivity"] = {}
for cubic in CUBICS:
    cl = cubic.lower()
    for sfx in VT_SUFFIXES:
        vals = {a: key["overall"][f"rho_{cl}_{a}_{sfx}"] for a in ALPHAS}
        key["vt_alpha_sensitivity"][f"{cl}_{sfx}"] = dict(
            values=vals, max_abs_diff_vs_classical=float(max(abs(v - vals[VT_ALPHA]) for v in vals.values())))

with open(os.path.join(DATA, "key_numbers.json"), "w") as fjson:
    json.dump(key, fjson, indent=2)

print(json.dumps({k: v for k, v in key["overall"].items()
                  if k.startswith(("psat_", "h_")) or k.endswith(("_vt", "_vtc"))}, indent=1))
print("Wrote deviations_full.csv, fluids.csv, anchor_sweep.csv, "
      "key_numbers.json, run_metadata.json")
