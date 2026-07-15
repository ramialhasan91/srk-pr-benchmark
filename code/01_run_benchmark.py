"""
Benchmark of SRK and PR cubic equations of state against reference
multiparameter (Helmholtz-energy) equations of state, as implemented in
CoolProp, for saturation pressure, saturated-liquid density, saturated-vapor
density, and enthalpy of vaporization.

Models
------
SRK      : Soave-Redlich-Kwong, classical Soave alpha function.
PR       : Peng-Robinson, classical Soave alpha function.
SRK-Twu  : SRK with the generalized Twu (1995) alpha function (Part 2).
PR-Twu   : PR with the generalized Twu (1995) alpha function (Part 1).
PR-VT    : PR with a constant per-fluid volume translation. Three choices of
           the constant c are compared (the "translation ladder"):
             c_fit  - translated saturated-liquid volume matches the REFERENCE
                      value at Tr = 0.70 (one liquid-density datum per fluid);
             c_corr - matches the RACKETT prediction at Tr = 0.70 instead,
                      with Z_RA = 0.29056 - 0.08775*omega (Yamada-Gunn), i.e.
                      zero data beyond (Tc, pc, omega); the PR analogue of
                      Peneloux's original construction for SRK;
             c_Zc   - matches the reference CRITICAL volume,
                      c = (Zc_PR - Zc_ref) * R * Tc / pc.
           Constant translation leaves psat, Delta h_vap, and cp unchanged
           (verified numerically below); it alters densities only.

Validation
----------
* Volume-translation invariance: ln(phi_translated) = ln(phi) - c*p/(R*T)
  checked by quadrature of the exact departure integral (both phases,
  all fluids, three temperatures).
* Clapeyron self-consistency: the departure-function enthalpy of
  vaporization, Eq. h_res(V) - h_res(L), is checked against
  T * (v'' - v') * dpsat/dT with the slope from central differences of the
  solved saturation pressure, at EVERY state point for all four
  (cubic, alpha) combinations.
* Anchor sensitivity: the pooled PR-VT liquid-density AAD is recomputed with
  the anchoring temperature swept over the whole reduced-temperature grid.

Protocol
--------
For each fluid: 40 reduced temperatures, Tr in [max(0.50, Tt/Tc + 0.01), 0.985].
Critical constants (Tc, pc) and acentric factor omega are taken from the
reference EoS itself (via CoolProp), so deviations reflect model form only.

Outputs (../data/):
  deviations_full.csv   per-point results and % deviations
  fluids.csv            fluid metadata (Tc, pc, omega, Zc_ref, Z_RA, family,
                        c_fit, c_corr, c_Zc, ...)
  anchor_sweep.csv      pooled PR-VT density AAD vs anchoring temperature
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
from eos import R, CubicEOS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

# --------------------------------------------------------------------- #
# Fluid set: 32 fluids in 5 chemical families. CoolProp names on the left,
# display names on the right. Hydrogen and helium are excluded (quantum
# fluids for which classical corresponding-states cubics are not intended).
FLUIDS = {
    # Inorganic / light gases
    "Nitrogen":        ("nitrogen",         "Inorganic gases"),
    "Oxygen":          ("oxygen",           "Inorganic gases"),
    "Argon":           ("argon",            "Inorganic gases"),
    "CarbonMonoxide":  ("carbon monoxide",  "Inorganic gases"),
    "CarbonDioxide":   ("carbon dioxide",   "Inorganic gases"),
    "HydrogenSulfide": ("hydrogen sulfide", "Inorganic gases"),
    # Alkanes (n-, iso-, cyclo-)
    "Methane":     ("methane",     "Alkanes"),
    "Ethane":      ("ethane",      "Alkanes"),
    "Propane":     ("propane",     "Alkanes"),
    "n-Butane":    ("n-butane",    "Alkanes"),
    "IsoButane":   ("isobutane",   "Alkanes"),
    "n-Pentane":   ("n-pentane",   "Alkanes"),
    "n-Hexane":    ("n-hexane",    "Alkanes"),
    "n-Heptane":   ("n-heptane",   "Alkanes"),
    "n-Octane":    ("n-octane",    "Alkanes"),
    "n-Decane":    ("n-decane",    "Alkanes"),
    "CycloHexane": ("cyclohexane", "Alkanes"),
    # Aromatics
    "Benzene":      ("benzene",      "Aromatics"),
    "Toluene":      ("toluene",      "Aromatics"),
    "EthylBenzene": ("ethylbenzene", "Aromatics"),
    "o-Xylene":     ("o-xylene",     "Aromatics"),
    "m-Xylene":     ("m-xylene",     "Aromatics"),
    "p-Xylene":     ("p-xylene",     "Aromatics"),
    # Polar / associating
    "Water":    ("water",    "Polar/associating"),
    "Methanol": ("methanol", "Polar/associating"),
    "Ethanol":  ("ethanol",  "Polar/associating"),
    "Acetone":  ("acetone",  "Polar/associating"),
    "Ammonia":  ("ammonia",  "Polar/associating"),
    # Refrigerants (HFC / HFO)
    "R32":     ("R-32",     "Refrigerants"),
    "R125":    ("R-125",    "Refrigerants"),
    "R134a":   ("R-134a",   "Refrigerants"),
    "R1234yf": ("R-1234yf", "Refrigerants"),
}

N_T = 40
TR_MAX = 0.985
PSAT_TOL = 1e-11          # convergence tolerance on the fugacity ratio
PSAT_TOL_FALLBACK = 1e-7  # accepted near Tc if the iteration limit is hit
CLAP_H = 1e-4             # relative step for the Clapeyron finite difference

# The four (cubic, alpha) combinations benchmarked for psat, rho', rho'', hvap.
MODEL_TAGS = [
    ("srk",    "SRK", "soave"),
    ("pr",     "PR",  "soave"),
    ("srktwu", "SRK", "twu"),
    ("prtwu",  "PR",  "twu"),
]


def pct(model, ref):
    return (model - ref) / ref * 100.0


def v_rackett(Tr, Tc, pc, Z_RA):
    """Rackett saturated-liquid molar volume with a correlated Z_RA."""
    return R * Tc / pc * Z_RA ** (1.0 + (1.0 - Tr) ** (2.0 / 7.0))


rows, meta = [], []
clap_rel = {tag: [] for tag, _, _ in MODEL_TAGS}
clap_skipped = 0
vt_nonphysical = 0

for cp_name, (disp, family) in FLUIDS.items():
    Tc = PropsSI("Tcrit", cp_name)
    pc = PropsSI("pcrit", cp_name)
    w = PropsSI("acentric", cp_name)
    Tt = PropsSI("Ttriple", cp_name)
    rhoc_ref = PropsSI("rhomolar_critical", cp_name)  # mol/m^3
    Zc_ref = pc / (rhoc_ref * R * Tc)

    Tr_min = max(0.50, Tt / Tc + 0.01)
    Tr_grid = np.linspace(Tr_min, TR_MAX, N_T)

    models = {tag: CubicEOS(cubic, Tc, pc, w, alpha=alpha)
              for tag, cubic, alpha in MODEL_TAGS}
    pr = models["pr"]

    # ---- translation ladder: three choices of the constant c ---------- #
    Tr_anchor = max(0.70, Tr_min)
    T_anchor = Tr_anchor * Tc
    v_pr_anchor, _ = pr.sat_liq_volume(T_anchor)
    v_ref_anchor = 1.0 / PropsSI("Dmolar", "T", T_anchor, "Q", 0, cp_name)
    Z_RA = 0.29056 - 0.08775 * w
    c_fit = v_pr_anchor - v_ref_anchor                        # one datum
    c_corr = v_pr_anchor - v_rackett(Tr_anchor, Tc, pc, Z_RA)  # zero data
    c_zc = (pr.Zc - Zc_ref) * R * Tc / pc                      # critical datum

    meta.append(
        dict(fluid=disp, coolprop=cp_name, family=family, Tc_K=Tc,
             pc_MPa=pc / 1e6, omega=w, Zc_ref=Zc_ref, Z_RA=Z_RA,
             Tr_min=Tr_min, Tr_anchor=Tr_anchor,
             c_VT_cm3mol=c_fit * 1e6, c_corr_cm3mol=c_corr * 1e6,
             c_zc_cm3mol=c_zc * 1e6)
    )

    for Tr in Tr_grid:
        T = Tr * Tc
        p_ref = PropsSI("P", "T", T, "Q", 0, cp_name)
        rho_ref = PropsSI("Dmolar", "T", T, "Q", 0, cp_name)    # mol/m^3
        rhov_ref = PropsSI("Dmolar", "T", T, "Q", 1, cp_name)   # mol/m^3
        h_ref = (PropsSI("Hmolar", "T", T, "Q", 1, cp_name)
                 - PropsSI("Hmolar", "T", T, "Q", 0, cp_name))  # J/mol

        rec = dict(fluid=disp, family=family, Tr=Tr, T_K=T,
                   p_ref_Pa=p_ref, rho_ref_molm3=rho_ref,
                   rhov_ref_molm3=rhov_ref, hvap_ref_Jmol=h_ref)

        for tag, model in models.items():
            p, Zl, Zv = model.psat(T)
            if np.isfinite(p):
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
                if tag == "pr":
                    # Translated volumes are set to NaN when v - c <= 0
                    # (occurs only for the diagnostic Zc-matched constant,
                    # for methanol at low Tr, where c_Zc exceeds the entire
                    # PR liquid volume).
                    for suffix, c in (("prvt", c_fit), ("prvtc", c_corr),
                                      ("prvtz", c_zc)):
                        if v_l - c > 0.0:
                            rec[f"dev_rho_{suffix}"] = pct(1.0 / (v_l - c), rho_ref)
                        else:
                            rec[f"dev_rho_{suffix}"] = np.nan
                            vt_nonphysical += 1
                    rec["rho_prvt_molm3"] = 1.0 / (v_l - c_fit)
                    rec["dev_rhov_prvt"] = pct(1.0 / (v_v - c_fit), rhov_ref)
            else:
                for col in (f"p_{tag}_Pa", f"dev_p_{tag}", f"rho_{tag}_molm3",
                            f"dev_rho_{tag}", f"dev_rhov_{tag}",
                            f"hvap_{tag}_Jmol", f"dev_h_{tag}"):
                    rec[col] = np.nan
                if tag == "pr":
                    for col in ("dev_rho_prvt", "dev_rho_prvtc", "dev_rho_prvtz",
                                "rho_prvt_molm3", "dev_rhov_prvt"):
                        rec[col] = np.nan
        rows.append(rec)

df = pd.DataFrame(rows)
fl = pd.DataFrame(meta)

df.to_csv(os.path.join(DATA, "deviations_full.csv"), index=False)
fl.to_csv(os.path.join(DATA, "fluids.csv"), index=False)

# --------------------------------------------------------------------- #
# Sanity checks and convergence accounting
n_expected = len(FLUIDS) * N_T
fail_counts = {tag: int(df[f"dev_p_{tag}"].isna().sum()) for tag, _, _ in MODEL_TAGS}
prop = df[df.fluid == "propane"]
prop_check = prop.iloc[(prop.Tr - 0.7).abs().argmin()]
assert abs(prop_check.dev_p_pr) < 5.0, "PR propane psat sanity check failed"
print(f"State points: {len(df)} (expected {n_expected}); "
      f"failed psat solves per model: {fail_counts}")

# Clapeyron consistency summary
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
# fugacity-equality condition (hence psat and Delta h_vap) unchanged:
# ln(phi_translated) = ln(phi) - c*p/(R*T) is checked by direct quadrature
# of the exact departure integral of the translated EoS, in both phases,
# for every fluid at three temperatures.
vt_residuals = []
for cp_name, (disp, family) in FLUIDS.items():
    row = fl[fl.fluid == disp].iloc[0]
    Tc, pc, w = PropsSI("Tcrit", cp_name), PropsSI("pcrit", cp_name), PropsSI("acentric", cp_name)
    pr = CubicEOS("PR", Tc, pc, w)
    c = row.c_VT_cm3mol * 1e-6
    for Tr in (max(0.55, row.Tr_min), 0.75, 0.95):
        T = Tr * Tc
        p, Zl, Zv = pr.psat(T)
        if not np.isfinite(p):
            continue
        A, B = pr.AB(T, p)
        for Z in (Zl, Zv):
            lhs = pr.lnphi_translated_quadrature(T, p, Z, c)
            rhs = pr.lnphi(Z, A, B) - c * p / (R * T)
            vt_residuals.append(abs(lhs - rhs))
vt_max_residual = float(np.max(vt_residuals))
print(f"VT invariance identity: {len(vt_residuals)} checks, "
      f"max |residual| in ln(phi) = {vt_max_residual:.2e}")
assert vt_max_residual < 1e-8, "volume-translation invariance check failed"

# --------------------------------------------------------------------- #
# Anchor-sensitivity sweep: pooled PR-VT liquid-density AAD as a function of
# the anchoring reduced temperature (per fluid, the anchor is the grid point
# nearest max(Tr_a, Tr_min), mirroring the paper's convention).
ANCHORS = np.linspace(0.50, TR_MAX, N_T)
per_fluid = {}
for f, g in df.groupby("fluid"):
    g = g.sort_values("Tr")
    per_fluid[f] = (g.Tr.to_numpy(),
                    1.0 / g.rho_pr_molm3.to_numpy(),
                    g.rho_ref_molm3.to_numpy())
sweep = []
for Tr_a in ANCHORS:
    devs = []
    for f, (Tr_g, v_pr_g, rho_ref_g) in per_fluid.items():
        idx = int(np.argmin(np.abs(Tr_g - max(Tr_a, Tr_g[0]))))
        c = v_pr_g[idx] - 1.0 / rho_ref_g[idx]
        devs.append(np.abs(pct(1.0 / (v_pr_g - c), rho_ref_g)))
    sweep.append((Tr_a, float(np.nanmean(np.concatenate(devs)))))
sweep_df = pd.DataFrame(sweep, columns=["Tr_anchor", "aad_rho_prvt"])
sweep_df.to_csv(os.path.join(DATA, "anchor_sweep.csv"), index=False)
i_min = int(sweep_df.aad_rho_prvt.idxmin())
i_070 = int((sweep_df.Tr_anchor - 0.70).abs().idxmin())
anchor_summary = {
    "aad_min": float(sweep_df.aad_rho_prvt[i_min]),
    "Tr_at_min": float(sweep_df.Tr_anchor[i_min]),
    "aad_at_070": float(sweep_df.aad_rho_prvt[i_070]),
    "flat_window_Tr": [
        float(sweep_df.Tr_anchor[sweep_df.aad_rho_prvt
                                 <= sweep_df.aad_rho_prvt[i_min] + 0.25].min()),
        float(sweep_df.Tr_anchor[sweep_df.aad_rho_prvt
                                 <= sweep_df.aad_rho_prvt[i_min] + 0.25].max()),
    ],
}
print("Anchor sweep:", anchor_summary)

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
        "psat_solver": "successive substitution on fugacity ratio, Wilson start",
        "psat_tol_fugacity_ratio": PSAT_TOL,
        "psat_tol_fallback_near_Tc": PSAT_TOL_FALLBACK,
        "failed_psat_solves": fail_counts,
        "vt_nonphysical_points": vt_nonphysical,
        "clapeyron_check": clap_stats,
        "vt_invariance_checks": len(vt_residuals),
        "vt_invariance_max_abs_residual_lnphi": vt_max_residual,
        "anchor_sweep": anchor_summary,
    }, fjson, indent=2)

# --------------------------------------------------------------------- #
# Headline statistics for the manuscript
def aad(s):
    return float(np.nanmean(np.abs(s)))


bins = pd.cut(df.Tr, [0.50, 0.70, 0.90, 0.9851], right=False,
              labels=["0.50-0.70", "0.70-0.90", "0.90-0.985"])
df["Tr_bin"] = bins

TAGS = [t for t, _, _ in MODEL_TAGS]
VT_TAGS = ["prvt", "prvtc", "prvtz"]

key = {
    "n_fluids": len(FLUIDS),
    "n_points": int(len(df)),
    "failed_psat_solves": fail_counts,
    "overall": {},
    "by_family": {}, "by_bin": {},
    "per_fluid_psat": {}, "per_fluid_rho": {}, "per_fluid_h": {},
}
for t in TAGS:
    key["overall"][f"psat_AAD_{t}"] = aad(df[f"dev_p_{t}"])
    key["overall"][f"rho_AAD_{t}"] = aad(df[f"dev_rho_{t}"])
    key["overall"][f"rho_bias_{t}"] = float(np.nanmean(df[f"dev_rho_{t}"]))
    key["overall"][f"rhov_AAD_{t}"] = aad(df[f"dev_rhov_{t}"])
    key["overall"][f"h_AAD_{t}"] = aad(df[f"dev_h_{t}"])
for t in VT_TAGS:
    key["overall"][f"rho_AAD_{t}"] = aad(df[f"dev_rho_{t}"])
key["overall"]["rhov_AAD_prvt"] = aad(df.dev_rhov_prvt)
# vapor density mirrors psat at low Tr:
lo = df[df.Tr < 0.70]
key["overall"]["corr_rhov_psat_pr_lowTr"] = float(
    np.corrcoef(lo.dev_rhov_pr, lo.dev_p_pr)[0, 1])

for fam, g in df.groupby("family"):
    key["by_family"][fam] = {}
    for t in TAGS:
        key["by_family"][fam][f"psat_{t}"] = aad(g[f"dev_p_{t}"])
        key["by_family"][fam][f"rho_{t}"] = aad(g[f"dev_rho_{t}"])
        key["by_family"][fam][f"rho_bias_{t}"] = float(np.nanmean(g[f"dev_rho_{t}"]))
        key["by_family"][fam][f"rhov_{t}"] = aad(g[f"dev_rhov_{t}"])
        key["by_family"][fam][f"h_{t}"] = aad(g[f"dev_h_{t}"])
    for t in VT_TAGS:
        key["by_family"][fam][f"rho_{t}"] = aad(g[f"dev_rho_{t}"])

for b, g in df.groupby("Tr_bin", observed=True):
    key["by_bin"][str(b)] = {}
    for t in TAGS:
        key["by_bin"][str(b)][f"psat_{t}"] = aad(g[f"dev_p_{t}"])
        key["by_bin"][str(b)][f"rho_{t}"] = aad(g[f"dev_rho_{t}"])
        key["by_bin"][str(b)][f"rhov_{t}"] = aad(g[f"dev_rhov_{t}"])
        key["by_bin"][str(b)][f"h_{t}"] = aad(g[f"dev_h_{t}"])
    for t in VT_TAGS:
        key["by_bin"][str(b)][f"rho_{t}"] = aad(g[f"dev_rho_{t}"])
    key["by_bin"][str(b)]["rhov_prvt"] = aad(g.dev_rhov_prvt)

for f, g in df.groupby("fluid"):
    key["per_fluid_psat"][f] = {t: aad(g[f"dev_p_{t}"]) for t in TAGS}
    key["per_fluid_rho"][f] = {t: aad(g[f"dev_rho_{t}"]) for t in TAGS}
    for t in VT_TAGS:
        key["per_fluid_rho"][f][t] = aad(g[f"dev_rho_{t}"])
    key["per_fluid_h"][f] = {t: aad(g[f"dev_h_{t}"]) for t in TAGS}

# Improvement factor of volume translation, per fluid (median across fluids)
imp = []
for f, g in df.groupby("fluid"):
    a_pr, a_vt = aad(g.dev_rho_pr), aad(g.dev_rho_prvt)
    if a_vt > 0:
        imp.append(a_pr / a_vt)
key["median_rho_improvement_factor_PRVT"] = float(np.median(imp))

# Pairwise per-fluid win counts on psat AAD (for sign tests)
key["win_counts_psat"] = {
    "pr_beats_srk": int(sum(key["per_fluid_psat"][f]["pr"] < key["per_fluid_psat"][f]["srk"]
                            for f in key["per_fluid_psat"])),
    "prtwu_beats_pr": int(sum(key["per_fluid_psat"][f]["prtwu"] < key["per_fluid_psat"][f]["pr"]
                              for f in key["per_fluid_psat"])),
    "srktwu_beats_srk": int(sum(key["per_fluid_psat"][f]["srktwu"] < key["per_fluid_psat"][f]["srk"]
                                for f in key["per_fluid_psat"])),
    "prtwu_beats_srktwu": int(sum(key["per_fluid_psat"][f]["prtwu"] < key["per_fluid_psat"][f]["srktwu"]
                                  for f in key["per_fluid_psat"])),
}
key["win_counts_h"] = {
    "prtwu_beats_pr": int(sum(key["per_fluid_h"][f]["prtwu"] < key["per_fluid_h"][f]["pr"]
                              for f in key["per_fluid_h"])),
    "srktwu_beats_srk": int(sum(key["per_fluid_h"][f]["srktwu"] < key["per_fluid_h"][f]["srk"]
                                for f in key["per_fluid_h"])),
}

# Deviation exactly at the acentric-factor anchor Tr = 0.7 (nearest grid pt)
near07 = df.loc[(df.Tr - 0.70).abs() < 0.02]
key["near_Tr07_psat_AAD"] = {t: aad(near07[f"dev_p_{t}"]) for t in TAGS}

with open(os.path.join(DATA, "key_numbers.json"), "w") as fjson:
    json.dump(key, fjson, indent=2)

print(json.dumps(key["overall"], indent=2))
print("Wrote deviations_full.csv, fluids.csv, anchor_sweep.csv, "
      "key_numbers.json, run_metadata.json")
