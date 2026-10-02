"""
Critical-constant confound of the fitted (Twu-fit) reference.

The component-specific Twu-91 parameters of the 1800-fluid compilation of
Pina-Martinez et al. (2022) were regressed with the critical constants of
that compilation (DIPPR-based), which differ from the critical constants of
the reference equations of state used everywhere else in this benchmark (by
up to 0.6% in Tc and 3.6% in pc for the 37 fluids; see
data/twu_fit_source_critical_constants.csv). Because a cubic EoS reproduces
psat(Tc) = pc exactly, a mismatch in pc propagates almost one-to-one into the
vapor-pressure deviation at every temperature. This script re-evaluates the
Twu-fit variants of SRK and PR with the critical constants of the source
compilation, at the same absolute temperatures and against the same reference
values as the main run, so that the effect of the constants can be separated
from the quality of the fitted alpha-function parameters.

Only the fitted reference is affected: the generalized alpha-functions are
evaluated with the reference-EoS constants throughout, which is the
consistent choice for a benchmark of predictive models.

Reads  ../data/deviations_full.csv, fluids.csv, twu_fit_params.csv,
       twu_fit_source_critical_constants.csv
Writes ../data/twufit_source_constants.csv  (per-fluid AADs, both constant sets)
       ../data/twufit_source_constants.json (pooled numbers quoted in the SI)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eos import R, CubicEOS  # noqa: E402
from fluid_set import FAMILY_ORDER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv")).set_index("fluid")
tf = pd.read_csv(os.path.join(DATA, "twu_fit_params.csv"), comment="#").set_index("fluid")
src = pd.read_csv(os.path.join(DATA, "twu_fit_source_critical_constants.csv"), comment="#").set_index("fluid")

PROPS = {"psat": "dev_p", "h": "dev_h", "rho": "dev_rho", "rhov": "dev_rhov"}


def pct(model, ref):
    return (model - ref) / ref * 100.0


def aad(s):
    return float(np.nanmean(np.abs(s)))


rows, dropped = [], 0
for fluid, g in df.groupby("fluid", sort=False):
    g = g.sort_values("Tr")
    Tc_ref, pc_ref, w = fl.loc[fluid, "Tc_K"], fl.loc[fluid, "pc_MPa"] * 1e6, fl.loc[fluid, "omega"]
    Tc_s, pc_s = src.loc[fluid, "Tc_src_K"], src.loc[fluid, "pc_src_bar"] * 1e5
    rec = dict(fluid=fluid, family=g.family.iloc[0],
               subgroup=(g.subgroup.iloc[0] if isinstance(g.subgroup.iloc[0], str) else ""),
               Tc_ref_K=Tc_ref, Tc_src_K=Tc_s,
               dTc_pct=pct(Tc_s, Tc_ref), pc_ref_MPa=pc_ref / 1e6, pc_src_MPa=pc_s / 1e6,
               dpc_pct=pct(pc_s, pc_ref))
    for cubic in ("SRK", "PR"):
        cl = cubic.lower()
        m = CubicEOS(cubic, Tc_s, pc_s, w, alpha="tf",
                     twu_params=(tf.loc[fluid, f"L_{cubic}"], tf.loc[fluid, f"M_{cubic}"],
                                 tf.loc[fluid, f"N_{cubic}"]))
        dev = {k: [] for k in PROPS}
        keep = []
        for _, r in g.iterrows():
            T = r.T_K
            if T >= Tc_s:               # above the source critical temperature
                dropped += 1
                keep.append(False)
                continue
            p, Zl, Zv = m.psat(T)
            if not np.isfinite(p):
                dropped += 1
                keep.append(False)
                continue
            keep.append(True)
            v_l, v_v = Zl * R * T / p, Zv * R * T / p
            h_vap = m.h_res(T, p, Zv) - m.h_res(T, p, Zl)
            dev["psat"].append(pct(p, r.p_ref_Pa))
            dev["rho"].append(pct(1.0 / v_l, r.rho_ref_molm3))
            dev["rhov"].append(pct(1.0 / v_v, r.rhov_ref_molm3))
            dev["h"].append(pct(h_vap, r.hvap_ref_Jmol))
        keep = np.asarray(keep)
        rec[f"n_points_{cl}"] = int(keep.sum())
        for prop, col in PROPS.items():
            rec[f"{prop}_{cl}_tf_src"] = aad(np.asarray(dev[prop]))
            # same points, reference-EoS constants (main run)
            rec[f"{prop}_{cl}_tf_ref"] = aad(g[f"{col}_{cl}_tf"].to_numpy()[keep])
            rec[f"{prop}_{cl}_tf_ref_allpts"] = aad(g[f"{col}_{cl}_tf"])
        # signed mean vapor-pressure deviation (bias) with both constant sets
        rec[f"psat_bias_{cl}_tf_src"] = float(np.mean(dev["psat"]))
        rec[f"psat_bias_{cl}_tf_ref"] = float(np.nanmean(g[f"dev_p_{cl}_tf"].to_numpy()[keep]))
    rows.append(rec)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA, "twufit_source_constants.csv"), index=False)

# pooled numbers (point-weighted, as in the main text)
pooled = {"n_points_dropped_above_source_Tc": int(dropped)}
for cl in ("srk", "pr"):
    for prop in PROPS:
        for which in ("src", "ref"):
            cols = out[[f"{prop}_{cl}_tf_{which}", f"n_points_{cl}"]]
            pooled[f"{prop}_{cl}_tf_{which}"] = float(
                (cols.iloc[:, 0] * cols.iloc[:, 1]).sum() / cols.iloc[:, 1].sum())
    pooled[f"psat_wins_src_better_{cl}"] = int((out[f"psat_{cl}_tf_src"] < out[f"psat_{cl}_tf_ref"]).sum())
    pooled[f"h_wins_src_better_{cl}"] = int((out[f"h_{cl}_tf_src"] < out[f"h_{cl}_tf_ref"]).sum())
def pooled_block(o):
    d = {}
    for cl in ("srk", "pr"):
        for prop in ("psat", "h"):
            for which in ("src", "ref"):
                d[f"{prop}_{cl}_tf_{which}"] = float(
                    (o[f"{prop}_{cl}_tf_{which}"] * o[f"n_points_{cl}"]).sum() / o[f"n_points_{cl}"].sum())
    return d


pooled["by_family"] = {fam: pooled_block(out[out.family == fam]) for fam in FAMILY_ORDER}
pooled["by_subgroup"] = {sg: pooled_block(out[out.subgroup == sg])
                         for sg in sorted(set(out.subgroup)) if sg}
pooled["max_abs_dTc_pct"] = float(out.dTc_pct.abs().max())
pooled["max_abs_dpc_pct"] = float(out.dpc_pct.abs().max())
pooled["fluid_max_abs_dpc"] = str(out.loc[out.dpc_pct.abs().idxmax(), "fluid"])
pooled["per_fluid"] = {
    r.fluid: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
              for k, v in r._asdict().items() if k not in ("Index", "fluid", "family", "subgroup")}
    for r in out.itertuples()}
with open(os.path.join(DATA, "twufit_source_constants.json"), "w") as f:
    json.dump(pooled, f, indent=2)

pd.set_option("display.width", 200)
print(out[["fluid", "dTc_pct", "dpc_pct", "psat_srk_tf_ref", "psat_srk_tf_src",
           "psat_pr_tf_ref", "psat_pr_tf_src", "h_pr_tf_ref", "h_pr_tf_src"]].round(2).to_string())
print(json.dumps({k: v for k, v in pooled.items() if k not in ("per_fluid", "by_family")}, indent=1))
print(json.dumps(pooled["by_family"], indent=1))
print(json.dumps(pooled["by_subgroup"], indent=1))
print("Wrote twufit_source_constants.csv / .json")
