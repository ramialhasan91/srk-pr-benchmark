"""Checks the numbers quoted in the manuscript and the Supplementary Materials against the generated data.
Part A: the original benchmark (key_numbers.json, stats.json, run_metadata.json, fluids.csv);
Part B: the revision (outputs of scripts 06-11, tables.json). Each check: description, value quoted in
the text, computed value, tolerance. Exits with status 1 on any mismatch.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
k = json.load(open(os.path.join(DATA, "key_numbers.json")))
st = json.load(open(os.path.join(DATA, "stats.json")))
md = json.load(open(os.path.join(DATA, "run_metadata.json")))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))
df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
o, b, f, s, pf = k["overall"], k["by_bin"], k["by_family"], k["by_subgroup"], k["per_fluid"]
ci, tests, nc = st["ci"], st["sign_tests"], st["near_critical"]
b090 = k["below_090"]
checks = []


def C(desc, quoted, computed, tol=0.006):
    checks.append((desc, quoted, computed, tol))


def spot(fluid, tag, Tr=0.5, col="dev_p"):
    g = df[df.fluid == fluid]
    return float(g.iloc[(g.Tr - Tr).abs().argmin()][f"{col}_{tag}"])


# ###################################################################### #
# Part A: numbers of the original benchmark
# ---- counts
C("n fluids", 37, k["n_fluids"], 0)
C("n points", 1480, k["n_points"], 0)
C("Clapeyron checks", 20720, md["clapeyron_check"]["n_checks"], 0)
C("Clapeyron max", 6e-7, md["clapeyron_check"]["max_rel_diff"], 1e-7)
C("Clapeyron median", 5e-8, md["clapeyron_check"]["median_rel_diff"], 1e-8)
C("VT checks", 444, md["vt_invariance_checks"], 0)
C("VT residual", 1.2e-14, md["vt_invariance_max_abs_residual_lnphi"], 2e-15)
# ---- abstract / overall
C("gap Soave", 0.68, o["psat_srk_s"] - o["psat_pr_s"])
C("gap S19", 0.33, o["psat_srk_s19"] - o["psat_pr_s19"])
C("gap T95", 0.00, o["psat_srk_t95"] - o["psat_pr_t95"])
C("gap Tc", 0.36, o["psat_srk_tc"] - o["psat_pr_tc"])
C("gap Tf", 0.16, o["psat_srk_tf"] - o["psat_pr_tf"])
C("gap coq", 0.08, o["psat_srk_coq"] - o["psat_pr_coq"]); C("gap ms", 0.14, o["psat_srk_ms"] - o["psat_pr_ms"])
C("psat srk coq", 1.45, o["psat_srk_coq"]); C("psat pr coq", 1.37, o["psat_pr_coq"])
C("psat srk ms", 1.23, o["psat_srk_ms"]); C("psat pr ms", 1.10, o["psat_pr_ms"])
C("bin1 coq srk", 2.50, b["0.50-0.70"]["psat_srk_coq"]); C("bin1 coq pr", 2.43, b["0.50-0.70"]["psat_pr_coq"])
C("bin1 ms srk", 1.79, b["0.50-0.70"]["psat_srk_ms"]); C("bin1 ms pr", 1.65, b["0.50-0.70"]["psat_pr_ms"])
C("Hbond coq srk", 4.6, s["H-bonding"]["psat_srk_coq"], 0.05); C("Hbond ms srk", 3.9, s["H-bonding"]["psat_srk_ms"], 0.05)
C("Hbond coq pr", 4.4, s["H-bonding"]["psat_pr_coq"], 0.05); C("Hbond ms pr", 3.9, s["H-bonding"]["psat_pr_ms"], 0.05)
C("gap h Tf", 0.22, o["h_srk_tf"] - o["h_pr_tf"])
for t, v in (("srk_s", 1.78), ("srk_s19", 1.81), ("srk_t95", 1.20), ("srk_tc", 1.51), ("srk_tf", 0.92),
             ("pr_s", 1.10), ("pr_s19", 1.48), ("pr_t95", 1.20), ("pr_tc", 1.15), ("pr_tf", 0.76)):
    C(f"psat {t}", v, o[f"psat_{t}"])
for t, v in (("srk_s", 2.74), ("pr_s", 2.10), ("srk_t95", 2.29), ("pr_t95", 2.19), ("srk_tf", 2.04), ("pr_tf", 1.82)):
    C(f"h {t}", v, o[f"h_{t}"])
C("CI psat srk_s lo", 1.41, ci["psat_srk_s"]["ci_lo"]); C("CI psat srk_s hi", 2.25, ci["psat_srk_s"]["ci_hi"])
C("CI psat pr_s lo", 0.89, ci["psat_pr_s"]["ci_lo"]); C("CI psat pr_s hi", 1.39, ci["psat_pr_s"]["ci_hi"])
C("CI psat srk_t95", 0.85, ci["psat_srk_t95"]["ci_lo"]); C("CI psat srk_t95 hi", 1.67, ci["psat_srk_t95"]["ci_hi"])
C("CI psat pr_t95 hi", 1.66, ci["psat_pr_t95"]["ci_hi"])
C("CI h srk_s", 2.38, ci["h_srk_s"]["ci_lo"]); C("CI h srk_s hi", 3.20, ci["h_srk_s"]["ci_hi"])
C("CI h pr_s", 1.83, ci["h_pr_s"]["ci_lo"]); C("CI h pr_s hi", 2.47, ci["h_pr_s"]["ci_hi"])
C("rho srk", 13.12, o["rho_srk_s"]); C("rho pr", 6.51, o["rho_pr_s"])
C("rho srk CI", 11.19, ci["rho_srk_s"]["ci_lo"]); C("rho srk CI hi", 15.11, ci["rho_srk_s"]["ci_hi"])
C("rho pr CI", 5.28, ci["rho_pr_s"]["ci_lo"]); C("rho pr CI hi", 7.92, ci["rho_pr_s"]["ci_hi"])
C("rho alpha spread srk", 0.07, max(o[f"rho_srk_{a}"] for a in ("s", "s19", "t95", "tc", "coq", "ms", "tf")) - min(o[f"rho_srk_{a}"] for a in ("s", "s19", "t95", "tc", "coq", "ms", "tf")), 0.02)
C("rho srk vt", 4.83, o["rho_srk_s_vt"]); C("rho pr vt", 3.11, o["rho_pr_s_vt"])
C("rho srk vtc", 6.33, o["rho_srk_s_vtc"]); C("rho pr vtc", 5.08, o["rho_pr_s_vtc"])
# ---- temperature dependence
C("bin2 psat srk", 1.15, b["0.70-0.90"]["psat_srk_s"]); C("bin2 psat pr", 0.52, b["0.70-0.90"]["psat_pr_s"])
C("near07 srk", 0.23, k["near_Tr07_psat_AAD"]["srk_s"]); C("near07 pr", 0.49, k["near_Tr07_psat_AAD"]["pr_s"])
C("near07 pr s19", 1.68, k["near_Tr07_psat_AAD"]["pr_s19"])
C("bin1 psat srk", 2.66, b["0.50-0.70"]["psat_srk_s"]); C("bin1 psat pr", 1.90, b["0.50-0.70"]["psat_pr_s"])
hb = [b090[f"h_{t}"] for t in k["models"] if not t.endswith("tf")]
C("h below 0.90 min", 1.6, min(hb), 0.05); C("h below 0.90 max", 2.3, max(hb), 0.05)
htop = [b["0.90-0.985"][f"h_{t}"] for t in k["models"]]
C("h near-crit min", 3.8, min(htop), 0.05); C("h near-crit max", 4.9, max(htop), 0.05)
C("psat near-crit max", 1.27, max(b["0.90-0.985"][f"psat_{t}"] for t in k["models"]))
# ---- families
C("polar psat srk", 2.84, f["Polar/associating"]["psat_srk_s"]); C("polar psat pr", 1.55, f["Polar/associating"]["psat_pr_s"])
C("methanol worst min", 5.04, min(pf["methanol"][f"psat_{t}"] for t in k["models"] if not t.endswith("tf")))
C("methanol worst max", 8.43, max(pf["methanol"][f"psat_{t}"] for t in k["models"] if not t.endswith("tf")))
C("water Tr0.5 srk", -18.7, spot("water", "srk_s"), 0.05); C("water Tr0.5 pr", -11.0, spot("water", "pr_s"), 0.05)
C("decane Tr0.5 pr", 11.2, spot("n-decane", "pr_s"), 0.05); C("decane Tr0.5 srk", -2.5, spot("n-decane", "srk_s"), 0.05)
C("heptane Tr0.5 pr", 5.1, spot("n-heptane", "pr_s"), 0.05); C("octane Tr0.5 pr", 7.9, spot("n-octane", "pr_s"), 0.05)
C("decane Tr0.5 pr t95", -3.0, spot("n-decane", "pr_t95"), 0.05); C("decane Tr0.5 pr s19", 6.6, spot("n-decane", "pr_s19"), 0.05)
C("decane AAD pr s", 2.19, pf["n-decane"]["psat_pr_s"]); C("decane AAD pr t95", 0.96, pf["n-decane"]["psat_pr_t95"])
C("alkanes srk s", 1.35, f["Alkanes"]["psat_srk_s"]); C("alkanes srk t95", 0.69, f["Alkanes"]["psat_srk_t95"])
C("alkanes pr s", 1.01, f["Alkanes"]["psat_pr_s"]); C("alkanes pr t95", 0.69, f["Alkanes"]["psat_pr_t95"])
C("aromatics pr s", 0.89, f["Aromatics"]["psat_pr_s"]); C("aromatics pr t95", 1.22, f["Aromatics"]["psat_pr_t95"])
C("Hbond pr s", 2.57, s["H-bonding"]["psat_pr_s"]); C("Hbond pr t95", 3.99, s["H-bonding"]["psat_pr_t95"])
C("srk s19", 1.81, o["psat_srk_s19"]); C("pr s19", 1.48, o["psat_pr_s19"])
# ---- sign tests (wins)
W = lambda key: tests[key]["wins"]
C("PR vs SRK soave psat", 31, W("psat:pr_s_vs_srk_s"), 0); C("PR vs SRK t95 psat", 22, W("psat:pr_t95_vs_srk_t95"), 0)
C("PR vs SRK tf psat", 31, W("psat:pr_tf_vs_srk_tf"), 0); C("PR vs SRK soave h", 34, W("h:pr_s_vs_srk_s"), 0)
C("PR vs SRK coq", 37, W("psat:pr_coq_vs_srk_coq"), 0); C("PR vs SRK ms", 36, W("psat:pr_ms_vs_srk_ms"), 0)
C("srk ms vs t95", 21, W("psat:srk_ms_vs_srk_t95"), 0); C("srk ms vs s", 34, W("psat:srk_ms_vs_srk_s"), 0)
C("pr ms vs s", 24, W("psat:pr_ms_vs_pr_s"), 0); C("srk coq vs t95 (t95 better 28)", 9, W("psat:srk_coq_vs_srk_t95"), 0)
C("pr coq vs t95 (t95 better 27)", 10, W("psat:pr_coq_vs_pr_t95"), 0)
C("p srk ms vs t95", 0.51, tests["psat:srk_ms_vs_srk_t95"]["p_two_sided"], 0.01); C("p pr ms vs s", 0.10, tests["psat:pr_ms_vs_pr_s"]["p_two_sided"], 0.01)
C("PR vs SRK t95 h", 24, W("h:pr_t95_vs_srk_t95"), 0)
C("srk t95 vs s", 34, W("psat:srk_t95_vs_srk_s"), 0); C("pr t95 vs s", 25, W("psat:pr_t95_vs_pr_s"), 0)
C("pr tc vs t95", 24, W("psat:pr_tc_vs_pr_t95"), 0); C("srk tc vs t95 (t95 better 32)", 5, W("psat:srk_tc_vs_srk_t95"), 0)
C("srk s19 vs s", 13, W("psat:srk_s19_vs_srk_s"), 0); C("pr s19 vs s", 9, W("psat:pr_s19_vs_pr_s"), 0)
C("srk tf vs t95", 23, W("psat:srk_tf_vs_srk_t95"), 0); C("pr tf vs s", 29, W("psat:pr_tf_vs_pr_s"), 0)
C("rho PR vs SRK", 31, W("rho:pr_s_vs_srk_s"), 0); C("rho PR-VT vs SRK-VT", 37, W("rho:pr_s_vt_vs_srk_s_vt"), 0)
C("rho srk vtc vs srk", 36, W("rho:srk_s_vtc_vs_srk_s"), 0); C("rho pr vtc vs pr", 22, W("rho:pr_s_vtc_vs_pr_s"), 0)
C("rhov pr vt vs pr", 30, W("rhov:pr_s_vt_vs_pr_s"), 0)
C("p PR vs SRK soave", 4e-5, tests["psat:pr_s_vs_srk_s"]["p_two_sided"], 1e-5)
C("p pr t95 vs s", 0.05, tests["psat:pr_t95_vs_pr_s"]["p_two_sided"], 0.005)
C("p pr s19 vs s", 3e-3, tests["psat:pr_s19_vs_pr_s"]["p_two_sided"], 1e-3)
C("p srk tc vs t95", 7e-6, tests["psat:srk_tc_vs_srk_t95"]["p_two_sided"], 2e-6)
# ---- fitted reference
C("Hbond tf srk", 1.20, s["H-bonding"]["psat_srk_tf"]); C("Hbond tf pr", 0.98, s["H-bonding"]["psat_pr_tf"])
C("water tf srk", 0.43, pf["water"]["psat_srk_tf"]); C("water tf pr", 0.30, pf["water"]["psat_pr_tf"])
C("methanol tf srk", 1.60, pf["methanol"]["psat_srk_tf"]); C("methanol tf pr", 0.44, pf["methanol"]["psat_pr_tf"])
C("methanol Tr0.5 srk s", -38, spot("methanol", "srk_s"), 0.5); C("methanol Tr0.5 pr s", -27, spot("methanol", "pr_s"), 0.5)
C("methanol Tr0.5 srk tf", -6.5, spot("methanol", "srk_tf"), 0.05); C("methanol Tr0.5 pr tf", 0.1, spot("methanol", "pr_tf"), 0.05)
C("water Tr0.5 tf below 1", 1.0, max(abs(spot("water", "srk_tf")), abs(spot("water", "pr_tf"))), 1.0)
apr = [s["polar aprotic"][f"psat_srk_{a}"] for a in ("s", "s19", "t95", "tc", "coq", "ms")]
C("aprotic srk min", 1.0, min(apr), 0.05); C("aprotic srk max", 1.8, max(apr), 0.05)
apr = [s["polar aprotic"][f"psat_pr_{a}"] for a in ("s", "s19", "t95", "tc", "coq", "ms")]
C("aprotic pr min", 0.9, min(apr), 0.05); C("aprotic pr max", 1.5, max(apr), 0.05)
hbr = [s["H-bonding"][f"psat_srk_{a}"] for a in ("s", "s19", "t95", "tc", "coq", "ms")]
C("Hbond srk min", 3.9, min(hbr), 0.05); C("Hbond srk max", 4.6, max(hbr), 0.05)
hbr = [s["H-bonding"][f"psat_pr_{a}"] for a in ("s", "s19", "t95", "tc", "coq", "ms")]
C("Hbond pr min", 2.6, min(hbr), 0.05); C("Hbond pr max", 4.4, max(hbr), 0.05)
chi = fl.set_index("fluid").chi_HalmStiel
C("chi methanol", 0.043, chi["methanol"], 0.0006); C("chi water", 0.023, chi["water"], 0.0006)
C("chi R-32", 0.015, chi["R-32"], 0.0006); C("chi ethanol", 0.004, chi["ethanol"], 0.0006)
C("Hbond rho pr ethanol", 7.71, pf["ethanol"]["rho_pr_s"]); C("Hbond rho pr water", 19.59, pf["water"]["rho_pr_s"])
C("Hbond rho pr vt min", 3.7, min(pf[x]["rho_pr_s_vt"] for x in ("water", "methanol", "ethanol", "ammonia")), 0.05)
C("Hbond rho pr vt max", 6.3, max(pf[x]["rho_pr_s_vt"] for x in ("water", "methanol", "ethanol", "ammonia")), 0.05)
# ---- density section
C("bias inorg srk", -5.8, f["Inorganic gases"]["rho_bias_srk_s"], 0.05); C("bias polar srk", -18.0, f["Polar/associating"]["rho_bias_srk_s"], 0.05)
C("bias alk pr", 0.5, f["Alkanes"]["rho_bias_pr_s"], 0.05); C("bias arom pr", -2.8, f["Aromatics"]["rho_bias_pr_s"], 0.05)
C("bias inorg pr", 6.4, f["Inorganic gases"]["rho_bias_pr_s"], 0.05); C("bias polar pr", -7.3, f["Polar/associating"]["rho_bias_pr_s"], 0.05)
C("methane rho srk", 4.7, pf["methane"]["rho_srk_s"], 0.05); C("methane rho pr", 8.8, pf["methane"]["rho_pr_s"], 0.05)
C("rho alpha per-point max", 0.6, max((df[f"dev_rho_{c}_{a}"] - df[f"dev_rho_{c}_s"]).abs().max() for c in ("srk", "pr") for a in ("s19", "t95", "tc", "coq", "ms", "tf")), 0.05)
C("b090 rho srk", 11.59, b090["rho_srk_s"]); C("b090 rho srk vtc", 4.32, b090["rho_srk_s_vtc"]); C("b090 rho srk vt", 2.64, b090["rho_srk_s_vt"])
C("b090 rho pr", 5.68, b090["rho_pr_s"]); C("b090 rho pr vtc", 3.59, b090["rho_pr_s_vtc"]); C("b090 rho pr vt", 1.35, b090["rho_pr_s_vt"])
C("b090 CI srk vt", 2.42, st["ci_below_090"]["rho_srk_s_vt"]["ci_lo"]); C("b090 CI srk vt hi", 2.90, st["ci_below_090"]["rho_srk_s_vt"]["ci_hi"])
C("b090 CI pr vt", 1.20, st["ci_below_090"]["rho_pr_s_vt"]["ci_lo"]); C("b090 CI pr vt hi", 1.55, st["ci_below_090"]["rho_pr_s_vt"]["ci_hi"])
C("bin1 rho pr vt", 0.47, b["0.50-0.70"]["rho_pr_s_vt"]); C("water rho pr vt", 6.3, pf["water"]["rho_pr_s_vt"], 0.05)
C("polar rho pr vtc", 8.9, f["Polar/associating"]["rho_pr_s_vtc"], 0.05)
frow = fl.set_index("fluid")
C("water c_corr pr", 0.2, frow.loc["water", "c_corr_pr_s_cm3mol"], 0.05); C("water c_fit pr", 4.3, frow.loc["water", "c_fit_pr_s_cm3mol"], 0.05)
C("top rho pr", 10.24, b["0.90-0.985"]["rho_pr_s"]); C("top rho pr vt", 10.97, b["0.90-0.985"]["rho_pr_s_vt"]); C("top rho pr vtc", 11.72, b["0.90-0.985"]["rho_pr_s_vtc"])
C("top rho srk", 19.96, b["0.90-0.985"]["rho_srk_s"]); C("top rho srk vt", 14.61, b["0.90-0.985"]["rho_srk_s_vt"]); C("top rho srk vtc", 15.30, b["0.90-0.985"]["rho_srk_s_vtc"])
C("c_fit pr negative count", 23, int((fl.c_fit_pr_s_cm3mol < 0).sum()), 0)
C("c_zc all positive", 1, int((fl.c_zc_pr_cm3mol > 0).all() and (fl.c_zc_srk_cm3mol > 0).all()), 0)
C("Zc ref min", 0.219, fl.Zc_ref.min(), 0.0006); C("Zc ref max", 0.294, fl.Zc_ref.max(), 0.0006)
r = fl.c_zc_pr_cm3mol / fl.c_fit_pr_s_cm3mol
C("ratio min", 4.4, r[fl.c_fit_pr_s_cm3mol > 0].min(), 0.05)
C("methanol c_zc", 45.9, frow.loc["methanol", "c_zc_pr_cm3mol"], 0.05)
C("nonphysical pr", 3, int(df.dev_rho_pr_s_vtz.isna().sum()), 0); C("nonphysical srk", 26, int(df.dev_rho_srk_s_vtz.isna().sum()), 0)
C("vtz overall pr", 132, o["rho_pr_s_vtz"], 0.5); C("vtz bin1 pr", 252, b["0.50-0.70"]["rho_pr_s_vtz"], 0.5); C("vtz top pr", 20.5, b["0.90-0.985"]["rho_pr_s_vtz"], 0.05)
C("nc srk slope", 0.97, nc["srk"]["ols_slope"], 0.005); C("nc pr slope", 0.97, nc["pr"]["ols_slope"], 0.005)
C("nc srk r", 0.98, nc["srk"]["pearson_r"], 0.005); C("nc pr r", 0.98, nc["pr"]["pearson_r"], 0.005)
C("nc srk intercept", -4.6, nc["srk"]["ols_intercept"], 0.05); C("nc pr intercept", -2.7, nc["pr"]["ols_intercept"], 0.05)
C("limit srk min", -34, nc["srk"]["x_limit_min"], 0.5); C("limit srk max", -12, nc["srk"]["x_limit_max"], 0.5)
C("limit pr min", -29, nc["pr"]["x_limit_min"], 0.5); C("limit pr max", -4, nc["pr"]["x_limit_max"], 0.5)
an = md["anchor_sweep"]
C("anchor PR min", 3.04, an["PR"]["aad_min"]); C("anchor PR Tr min", 0.75, an["PR"]["Tr_at_min"], 0.005); C("anchor PR 0.70", 3.12, an["PR"]["aad_at_070"])
C("anchor PR flat hi", 0.79, an["PR"]["flat_window_Tr"][1], 0.005); C("anchor PR worse", 0.90, an["PR"]["first_Tr_worse_than_untranslated"], 0.005); C("anchor PR top", 44, an["PR"]["aad_at_top"], 0.5)
C("anchor SRK flat lo", 0.59, an["SRK"]["flat_window_Tr"][0], 0.005); C("anchor SRK flat hi", 0.75, an["SRK"]["flat_window_Tr"][1], 0.005)
C("anchor SRK min", 4.83, an["SRK"]["aad_min"]); C("anchor SRK top", 70, an["SRK"]["aad_at_top"], 0.5)
d = fl.c_lit_pr_cm3mol - fl.c_fit_pr_s_cm3mol
C("c_lit mean abs diff", 1.6, d.abs().mean(), 0.05); C("c_lit corr", 0.97, np.corrcoef(fl.c_lit_pr_cm3mol, fl.c_fit_pr_s_cm3mol)[0, 1], 0.005)
# ---- vapor density
rv = [o[f"rhov_{t}"] for t in k["models"]] + [o["rhov_pr_s_vt"], o["rhov_srk_s_vt"]]
C("rhov min", 1.7, min(rv), 0.05); C("rhov max", 2.7, max(rv), 0.05)
C("rhov srk bins min", 1.3, min(b[x]["rhov_srk_s"] for x in b), 0.05); C("rhov srk bins max", 3.4, max(b[x]["rhov_srk_s"] for x in b), 0.05)
C("corr rhov psat", 0.98, o["corr_rhov_psat_pr_s_lowTr"], 0.005)
C("rhov top pr", 2.80, b["0.90-0.985"]["rhov_pr_s"]); C("rhov top pr vt", 2.12, b["0.90-0.985"]["rhov_pr_s_vt"])

# ---- revision: sign-test p-values and win counts quoted in the text
P = lambda key: tests[key]["p_two_sided"]
C("p pr t95 vs s (0.047)", 0.047, P("psat:pr_t95_vs_pr_s"), 0.001); C("p pr ms vs s (0.099)", 0.099, P("psat:pr_ms_vs_pr_s"), 0.001)
C("p srk s19 vs s (0.099)", 0.099, P("psat:srk_s19_vs_srk_s"), 0.001); C("p pr tc vs t95 (0.099)", 0.099, P("psat:pr_tc_vs_pr_t95"), 0.001)
C("p srk tf vs t95 (0.19)", 0.19, P("psat:srk_tf_vs_srk_t95"), 0.005); C("p srk tf vs ms (0.32)", 0.32, P("psat:srk_tf_vs_srk_ms"), 0.005)
C("p pr tf vs ms (0.51)", 0.51, P("psat:pr_tf_vs_pr_ms"), 0.005); C("p pr coq vs s (0.51)", 0.51, P("psat:pr_coq_vs_pr_s"), 0.005)
C("p srk coq vs s (0.003)", 0.003, P("psat:srk_coq_vs_srk_s"), 0.0005); C("wins srk coq vs s", 28, W("psat:srk_coq_vs_srk_s"), 0)
C("wins pr coq vs s", 16, W("psat:pr_coq_vs_pr_s"), 0)
C("wins srk tf vs s", 30, W("psat:srk_tf_vs_srk_s"), 0); C("p srk tf vs s (2e-4)", 2e-4, P("psat:srk_tf_vs_srk_s"), 0.5e-4)
C("p pr tf vs s (8e-4)", 8e-4, P("psat:pr_tf_vs_pr_s"), 0.5e-4)
C("wins srk tf vs ms", 22, W("psat:srk_tf_vs_srk_ms"), 0); C("wins pr tf vs ms", 21, W("psat:pr_tf_vs_pr_ms"), 0)
C("wins pr tc vs t95", 24, W("psat:pr_tc_vs_pr_t95"), 0); C("wins srk tc vs t95", 5, W("psat:srk_tc_vs_srk_t95"), 0)
C("p PR vs SRK t95 psat (0.32)", 0.32, P("psat:pr_t95_vs_srk_t95"), 0.005); C("p PR vs SRK t95 h (0.099)", 0.099, P("h:pr_t95_vs_srk_t95"), 0.001)
C("PR vs SRK tf h wins", 33, W("h:pr_tf_vs_srk_tf"), 0); C("p PR vs SRK tf h (1e-6)", 1e-6, P("h:pr_tf_vs_srk_tf"), 0.5e-6)
C("p PR vs SRK soave h (1e-7)", 1e-7, P("h:pr_s_vs_srk_s"), 0.5e-7)
hw = [W(f"h:pr_{a}_vs_srk_{a}") for a in ("s", "s19", "tc", "coq", "ms", "tf")]
C("h cubic wins min (29)", 29, min(hw), 0); C("h cubic wins max (35)", 35, max(hw), 0)
hg = [o[f"h_srk_{a}"] - o[f"h_pr_{a}"] for a in ("t95", "tc", "coq", "ms")]
C("h gap modern min 0.1", 0.1, min(hg), 0.02); C("h gap modern max 0.3", 0.3, max(hg), 0.05); C("h gap soave 0.64", 0.64, o["h_srk_s"] - o["h_pr_s"])
C("p rhov pr vt vs pr (2e-4)", 2e-4, P("rhov:pr_s_vt_vs_pr_s"), 0.5e-4); C("p rho pr vtc vs pr (0.32)", 0.32, P("rho:pr_s_vtc_vs_pr_s"), 0.005)
C("n sign tests", 206, st["n_sign_tests"], 0); C("ties", 0, st["n_ties_total"], 0)
# PR equivalent set: Soave, MS, Twu-c pairwise p >= 0.10
C("pr ms vs tc p >= 0.10", 1, int(min(P("psat:pr_ms_vs_pr_s"), P("psat:pr_tc_vs_pr_s"), P("psat:pr_ms_vs_pr_tc")) >= 0.09), 0)
C("pr ms vs t95 wins 28", 28, W("psat:pr_ms_vs_pr_t95"), 0); C("p pr ms vs t95 0.003", 0.003, P("psat:pr_ms_vs_pr_t95"), 0.0005)
# ---- revision: MS lowest generalized in the lowest bin, alpha behaviour by fluid
for c in ("srk", "pr"):
    C(f"ms lowest bin1 {c}", 1, int(min(b["0.50-0.70"][f"psat_{c}_{a}"] for a in ("s", "s19", "t95", "tc", "coq", "ms")) == b["0.50-0.70"][f"psat_{c}_ms"]), 0)
gen = ("s", "s19", "t95", "tc", "coq", "ms")
for fluid in ("hydrogen sulfide", "dimethyl ether", "ethylene oxide", "methyl chloride", "tetrahydrofuran", "R-125", "R-1234yf"):
    for c in ("srk", "pr"):
        C(f"tf worse than best generalized {fluid} {c}", 1, int(pf[fluid][f"psat_{c}_tf"] > min(pf[fluid][f"psat_{c}_{a}"] for a in gen)), 0)
dd = [pf[fl_][f"psat_{c}_tf"] - min(pf[fl_][f"psat_{c}_{a}"] for a in gen) for fl_ in ("ethylene oxide", "tetrahydrofuran") for c in ("srk", "pr")]
C("EO/THF tf loss min 1", 1.0, min(dd), 0.35); C("EO/THF tf loss max 2", 2.0, max(dd), 0.1)
amm = [pf["ammonia"][f"psat_{c}_{a}"] for c in ("srk", "pr") for a in gen]
C("ammonia gen min 0.7", 0.7, min(amm), 0.05); C("ammonia gen max 2.4", 2.4, max(amm), 0.05)
for fluid in ("water", "methanol", "ethanol"):
    C(f"PR-Soave best generalized {fluid}", 1, int(pf[fluid]["psat_pr_s"] == min(pf[fluid][f"psat_{c}_{a}"] for c in ("srk", "pr") for a in gen)), 0)
apro = ("acetone", "dimethyl ether", "sulfur dioxide", "ethylene oxide", "methyl chloride", "tetrahydrofuran")
C("aprotic best generalized < 1%", 1, int(all(min(pf[x][f"psat_{c}_{a}"] for c in ("srk", "pr") for a in gen) < 1.0 for x in apro)), 0)
C("t95 improves all aprotic on SRK", 1, int(all(pf[x]["psat_srk_t95"] < pf[x]["psat_srk_s"] for x in apro)), 0)
C("t95 improves DME/SO2/MeCl on PR", 1, int(all(pf[x]["psat_pr_t95"] < pf[x]["psat_pr_s"] for x in ("dimethyl ether", "sulfur dioxide", "methyl chloride"))), 0)
C("PR-Soave best generalized acetone/EO/THF", 1, int(all(pf[x]["psat_pr_s"] == min(pf[x][f"psat_{c}_{a}"] for c in ("srk", "pr") for a in gen) for x in ("acetone", "ethylene oxide", "tetrahydrofuran"))), 0)
C("alkanes tf srk", 0.7, f["Alkanes"]["psat_srk_tf"], 0.05); C("alkanes tf pr", 0.5, f["Alkanes"]["psat_pr_tf"], 0.05)
# ---- revision: Soave-19 slope differences
import sys
sys.path.insert(0, HERE)
from eos import M_SOAVE  # noqa: E402
dm_pr = np.array([M_SOAVE[("PR", "s19")](x) - M_SOAVE[("PR", "s")](x) for x in fl.omega])
dm_srk = np.array([M_SOAVE[("SRK", "s19")](x) - M_SOAVE[("SRK", "s")](x) for x in fl.omega])
C("PR m19-m min 0.007", 0.007, dm_pr.min(), 0.0006); C("PR m19-m max 0.017", 0.017, dm_pr.max(), 0.0006)
C("SRK |m19-m| < 0.003", 1, int(np.abs(dm_srk).max() < 0.003), 0)
# ---- revision: density section by family (all T) and vapor density
C("PR vtc inorg 8.0->2.6", 8.0, f["Inorganic gases"]["rho_pr_s"], 0.05); C("PR vtc inorg after", 2.6, f["Inorganic gases"]["rho_pr_s_vtc"], 0.05)
C("PR vtc alk 5.2", 5.2, f["Alkanes"]["rho_pr_s"], 0.05); C("PR vtc alk after 2.9", 2.9, f["Alkanes"]["rho_pr_s_vtc"], 0.05)
C("PR vtc arom 3.7", 3.7, f["Aromatics"]["rho_pr_s"], 0.05); C("PR vtc arom after 4.1", 4.1, f["Aromatics"]["rho_pr_s_vtc"], 0.05)
C("PR vtc refr 6.2", 6.2, f["Refrigerants"]["rho_pr_s"], 0.05); C("PR vtc refr after 6.7", 6.7, f["Refrigerants"]["rho_pr_s_vtc"], 0.05)
C("PR alkanes rho AAD 5.2", 5.2, f["Alkanes"]["rho_pr_s"], 0.05)
C("Hbond rho pr", 14.0, s["H-bonding"]["rho_pr_s"], 0.05); C("Hbond rho pr vt", 4.83, s["H-bonding"]["rho_pr_s_vt"])
C("Hbond rho pr removes ~2/3", 0.66, 1 - s["H-bonding"]["rho_pr_s_vt"] / s["H-bonding"]["rho_pr_s"], 0.03)
top2 = sorted(pf, key=lambda x: -pf[x]["rho_pr_s_vt"])[:2]
C("largest PR-VT errors water+methanol", 1, int(set(top2) == {"water", "methanol"}), 0)
C("rhov top srk", 3.32, b["0.90-0.985"]["rhov_srk_s"]); C("rhov top srk vt", 2.23, b["0.90-0.985"]["rhov_srk_s_vt"])
mrv = [pf["methanol"][f"rhov_{c}_{a}"] for c in ("srk", "pr") for a in gen]
C("methanol rhov min 8", 8, min(mrv), 0.5); C("methanol rhov max 12", 12, max(mrv), 0.5)
C("h bin2 srk 2.36", 2.36, b["0.70-0.90"]["h_srk_s"]); C("h bin2 pr 1.76", 1.76, b["0.70-0.90"]["h_pr_s"])
best = sorted((min(pf[x][f"psat_{t}"] for t in k["models"]), x) for x in pf)[:5]
C("best per-fluid psat min 0.19", 0.19, best[0][0]); C("best per-fluid psat 5th 0.30", 0.30, best[4][0])
C("best five fluids", 1, int({x for _, x in best} == {"carbon dioxide", "oxygen", "ethane", "water", "nitrogen"}), 0)
C("vt alpha max 0.13", 0.13, max(abs(o[f"rho_{c}_{a}_{sfx}"] - o[f"rho_{c}_s_{sfx}"]) for c in ("srk", "pr") for a in gen + ("tf",) for sfx in ("vt", "vtc")), 0.006)
C("SRK-VT coquelet 4.70", 4.70, o["rho_srk_coq_vt"])
C("psat alpha span min 0.8", 0.8, min(o[f"psat_{t}"] for t in k["models"]), 0.05); C("psat alpha span max 1.8", 1.8, max(o[f"psat_{t}"] for t in k["models"]), 0.05)
C("h alpha span min 1.8", 1.8, min(o[f"h_{t}"] for t in k["models"]), 0.05); C("h alpha span max 2.8", 2.8, max(o[f"h_{t}"] for t in k["models"]), 0.05)
# ---- revision: SI Figure S3 fluids and translation constants
C("argon c_corr pr", -3.29, frow.loc["argon", "c_corr_pr_s_cm3mol"]); C("argon c_fit pr", -3.40, frow.loc["argon", "c_fit_pr_s_cm3mol"])
C("water c_corr pr 0.20", 0.20, frow.loc["water", "c_corr_pr_s_cm3mol"]); C("water c_fit pr 4.35", 4.35, frow.loc["water", "c_fit_pr_s_cm3mol"])
C("methane c_fit pr -4.1", -4.1, frow.loc["methane", "c_fit_pr_s_cm3mol"], 0.05)
C("decane c_corr pr 17.1", 17.1, frow.loc["n-decane", "c_corr_pr_s_cm3mol"], 0.05); C("decane c_fit pr 11.7", 11.7, frow.loc["n-decane", "c_fit_pr_s_cm3mol"], 0.05)
C("R134a c_corr pr 0.06", 0.06, frow.loc["R-134a", "c_corr_pr_s_cm3mol"]); C("R134a c_fit pr 1.03", 1.03, frow.loc["R-134a", "c_fit_pr_s_cm3mol"])
for fluid, key_, val in (("methane", "rho_pr_s_vtc", 2.7), ("methane", "rho_pr_s_vt", 2.6), ("methane", "rho_srk_s_vt", 4.3), ("methane", "rho_srk_s_vtc", 4.3),
                         ("n-decane", "rho_pr_s", 7.5), ("n-decane", "rho_pr_s_vtc", 3.4), ("n-decane", "rho_srk_s", 18.1), ("n-decane", "rho_srk_s_vtc", 5.2),
                         ("R-134a", "rho_pr_s", 4.2), ("R-134a", "rho_pr_s_vtc", 4.2), ("R-134a", "rho_srk_s", 15.3), ("R-134a", "rho_srk_s_vtc", 4.9),
                         ("methanol", "rho_srk_s", 25.8), ("methanol", "rho_srk_s_vtc", 8.0), ("methanol", "rho_pr_s", 16.2), ("methanol", "rho_pr_s_vtc", 7.2),
                         ("methanol", "rho_srk_s_vt", 7.4), ("methanol", "rho_pr_s_vt", 5.3), ("methane", "rho_pr_s", 8.8)):
    C(f"figS3 {fluid} {key_}", val, pf[fluid][key_], 0.05)
# ---- revision: critical-constant confound of the fitted reference
tfs = json.load(open(os.path.join(DATA, "twufit_source_constants.json")))
pfs = tfs["per_fluid"]
C("tf src pooled srk", 0.73, tfs["psat_srk_tf_src"]); C("tf src pooled pr", 0.60, tfs["psat_pr_tf_src"])
C("tf ref pooled srk", 0.92, tfs["psat_srk_tf_ref"]); C("tf ref pooled pr", 0.76, tfs["psat_pr_tf_ref"])
C("tf gap src 0.13", 0.13, tfs["psat_srk_tf_src"] - tfs["psat_pr_tf_src"])
C("EO pr tf ref 2.52", 2.52, pfs["ethylene oxide"]["psat_pr_tf_ref"]); C("EO pr tf src 0.68", 0.68, pfs["ethylene oxide"]["psat_pr_tf_src"])
C("THF pr tf ref 2.31", 2.31, pfs["tetrahydrofuran"]["psat_pr_tf_ref"]); C("THF pr tf src 0.46", 0.46, pfs["tetrahydrofuran"]["psat_pr_tf_src"])
C("MeCl pr tf ref 1.25", 1.25, pfs["methyl chloride"]["psat_pr_tf_ref"]); C("MeCl pr tf src 0.33", 0.33, pfs["methyl chloride"]["psat_pr_tf_src"])
C("aprotic pr tf ref 1.42", 1.42, tfs["by_subgroup"]["polar aprotic"]["psat_pr_tf_ref"]); C("aprotic pr tf src 0.65", 0.65, tfs["by_subgroup"]["polar aprotic"]["psat_pr_tf_src"])
C("Hbond srk tf src 1.11", 1.11, tfs["by_subgroup"]["H-bonding"]["psat_srk_tf_src"]); C("Hbond pr tf src 1.15", 1.15, tfs["by_subgroup"]["H-bonding"]["psat_pr_tf_src"])
C("max dTc 0.6", 0.6, tfs["max_abs_dTc_pct"], 0.05); C("max dpc 3.6", 3.6, tfs["max_abs_dpc_pct"], 0.05)
C("max dpc fluid", 1, int(tfs["fluid_max_abs_dpc"] == "methyl chloride"), 0)
big = {x for x in pfs if abs(pfs[x]["dpc_pct"]) > 1.5}
C("dpc > 1.5% set", 1, int(big == {"methanol", "ethanol", "ethylene oxide", "methyl chloride", "tetrahydrofuran"}), 0)
C("no points dropped", 0, tfs["n_points_dropped_above_source_Tc"], 0)

# ---- proofreading pass: scoped claims
gen7 = ("s19", "t95", "tc", "coq", "ms", "tf")
C("vt alpha max change by range 0.26", 0.26, max(abs(b[x][f"rho_{c}_{a}_{sfx}"] - b[x][f"rho_{c}_s_{sfx}"]) for c in ("srk", "pr") for a in gen7 for sfx in ("vt", "vtc") for x in b), 0.006)
lo_ = df[df.Tr < 0.90]
C("vt alpha max change <0.90 <= 0.13", 1, int(max(abs(np.nanmean(np.abs(lo_[f"dev_rho_{c}_{a}_{sfx}"])) - np.nanmean(np.abs(lo_[f"dev_rho_{c}_s_{sfx}"]))) for c in ("srk", "pr") for a in gen7 for sfx in ("vt", "vtc")) <= 0.13), 0)
rs = fl.c_zc_srk_cm3mol / fl.c_fit_srk_s_cm3mol
C("SRK c_zc/c_fit min 3", 3.3, rs.min(), 0.05); C("SRK c_zc/c_fit max 40", 40, rs.max(), 1.0)
rp = (fl.c_zc_pr_cm3mol / fl.c_fit_pr_s_cm3mol)[fl.c_fit_pr_s_cm3mol > 0]
C("PR c_zc/c_fit min 4.4", 4.4, rp.min(), 0.05); C("PR c_zc/c_fit max several hundred", 1, int(rp.max() > 200), 0)
C("h gap s19 0.46", 0.46, o["h_srk_s19"] - o["h_pr_s19"])
eth = [pf["ethanol"][f"psat_{c}_{a}"] for c in ("srk", "pr") for a in gen]
C("ethanol gen min 2", 2.0, min(eth), 0.05); C("ethanol gen max 5 (4.58)", 4.6, max(eth), 0.05)
# library versions quoted in Section 2.4 against the metadata of the committed runs
_ifm = json.load(open(os.path.join(DATA, "insample_fit_meta.json")))["versions"]
for _lab, _quoted, _rec in (("python", "3.13.16", md["python"]), ("numpy", "2.5.3", md["numpy"]), ("pandas", "3.0.5", md["pandas"]),
                            ("coolprop", "8.0.0", md["coolprop"]), ("scipy", "1.18.1", _ifm["scipy"]), ("python (fits)", "3.13.16", _ifm["python"])):
    C(f"version {_lab} {_quoted}", 1, int(_rec == _quoted), 0)
fine = k["by_bin_fine"]
C("psat smaller in upper half (all four classical/t95)", 1, int(all(fine["0.95-0.985"][f"psat_{t}"] < fine["0.90-0.95"][f"psat_{t}"] for t in ("srk_s", "pr_s", "srk_t95", "pr_t95"))), 0)
C("h, rho, rhov larger in upper half", 1, int(all(fine["0.95-0.985"][f"{p_}_{t}"] > fine["0.90-0.95"][f"{p_}_{t}"] for p_ in ("h", "rho", "rhov") for t in ("srk_s", "pr_s", "srk_t95", "pr_t95"))), 0)

# ###################################################################### #
# Part B: numbers added in the revision
cmp2 = pd.read_csv(os.path.join(DATA, "paired_comparisons.csv")).set_index("key")
holm = pd.read_csv(os.path.join(DATA, "holm_extended.csv")).set_index("key")
cmp3 = pd.read_csv(os.path.join(DATA, "insample_comparisons.csv")).set_index("key")
pool3 = pd.read_csv(os.path.join(DATA, "insample_pooled.csv")).set_index(["model", "prop"])
pf3 = pd.read_csv(os.path.join(DATA, "insample_per_fluid.csv")).set_index("fluid")
fits = pd.read_csv(os.path.join(DATA, "insample_fit.csv"))
clo = pd.read_csv(os.path.join(DATA, "insample_alpha_closeness.csv"))
pr78 = pd.read_csv(os.path.join(DATA, "pr78_summary.csv")).set_index(["fluid", "range"])
pr78p = pd.read_csv(os.path.join(DATA, "pr78_points.csv"))
pr78j = json.load(open(os.path.join(DATA, "pr78.json")))
habs = pd.read_csv(os.path.join(DATA, "hvap_absolute.csv")).set_index(["model", "range"])
tab = json.load(open(os.path.join(DATA, "compilation_constants.json")))
tabpf = pd.read_csv(os.path.join(DATA, "compilation_constants_per_fluid.csv")).set_index("fluid")
sens = pd.read_csv(os.path.join(DATA, "input_sensitivities.csv"))
o = k["overall"]
GEN = ("s", "s19", "t95", "tc", "coq", "ms")

# ---- Abstract / Section 3.1: gaps
gaps = {a: o[f"psat_srk_{a}"] - o[f"psat_pr_{a}"] for a in GEN}
C("abstract: other generalized gaps min 0.0", 0.0, min(gaps[a] for a in GEN if a != "s"), 0.05)
C("abstract: other generalized gaps max 0.4", 0.4, max(gaps[a] for a in GEN if a != "s"), 0.05)
C("3.1 Soave-is gap 0.58", 0.58, -cmp3.loc["psat:pr_sfit_vs_srk_sfit", "dAAD"])
C("3.1 Twu-is gap 0.15", 0.15, -cmp3.loc["psat:pr_tfit_vs_srk_tfit", "dAAD"])
# ---- Section 2.5: family sizes
C("F1 size 20", 20, int((holm.family == "F1").sum()), 0)
C("F2 size 27", 27, int((holm.family == "F2").sum()), 0)
C("F3 size 4", 4, int((cmp2.family == "F3").sum()), 0)
C("n exploratory tests 206", 206, len(cmp2), 0)
# ---- Section 3.2.1: signed deviations and absolute hvap
C("SRK bias < 0.70 (-2.6)", -2.6, pool3.loc[("srk_s", "psat"), "bias|0.50-0.70"], 0.05)
C("PR bias < 0.70 (+0.6)", 0.6, pool3.loc[("pr_s", "psat"), "bias|0.50-0.70"], 0.05)
r = habs.loc["pr_s"]
C("hvap abs < 0.90 (0.40)", 0.40, r.loc["< 0.90", "mean_abs_dev_kJmol"])
C("hvap abs 0.90-0.95 (0.27)", 0.27, r.loc["0.90-0.95", "mean_abs_dev_kJmol"])
C("hvap abs 0.95-0.985 (0.55)", 0.55, r.loc["0.95-0.985", "mean_abs_dev_kJmol"])
C("hvap abs rel h07 < 0.90 (1.5)", 1.5, r.loc["< 0.90", "abs_dev_rel_h07_pct"], 0.05)
C("hvap abs rel h07 0.90-0.95 (1.3)", 1.3, r.loc["0.90-0.95", "abs_dev_rel_h07_pct"], 0.05)
C("hvap abs rel h07 0.95-0.985 (2.7)", 2.7, r.loc["0.95-0.985", "abs_dev_rel_h07_pct"], 0.05)
C("ref hvap falls to 38%", 38, r.loc["0.95-0.985", "ref_hvap_rel_h07_pct"], 0.5)
C("rel AAD 1.6 -> 7.3", 1.6, r.loc["< 0.90", "rel_AAD_pct"], 0.05); C("rel AAD 7.3", 7.3, r.loc["0.95-0.985", "rel_AAD_pct"], 0.05)
C("signed near-crit -0.5 kJ/mol", -0.5, r.loc["0.95-0.985", "mean_signed_dev_kJmol"], 0.05)
# ---- Section 3.2.2: paired intervals and adjusted p-values
def ci(key):
    return cmp2.loc[key, "dAAD"], cmp2.loc[key, "ci_lo"], cmp2.loc[key, "ci_hi"]
d, lo, hi = ci("psat:pr_t95_vs_pr_s")
C("PR t95 vs s +0.10", 0.10, d); C("PR t95 vs s lo -0.11", -0.11, lo); C("PR t95 vs s hi +0.33", 0.33, hi)
C("PR t95 vs s wins 25", 25, cmp2.loc["psat:pr_t95_vs_pr_s", "wins"], 0)
C("PR t95 vs s Holm 0.52", 0.52, holm.loc["psat:pr_t95_vs_pr_s", "p_holm_ext"], 0.005)
C("SRK t95 vs s Holm 3e-6", 3e-6, holm.loc["psat:srk_t95_vs_srk_s", "p_holm_ext"], 0.5e-6)
C("SRK t95 vs s wins 34", 34, cmp2.loc["psat:srk_t95_vs_srk_s", "wins"], 0)
d, lo, hi = ci("psat:pr_tc_vs_pr_t95")
C("PR tc vs t95 -0.05", -0.05, d); C("lo -0.20", -0.20, lo); C("hi +0.11", 0.11, hi)
C("SRK tc worse than t95 for 32", 32, 37 - cmp2.loc["psat:srk_tc_vs_srk_t95", "wins"], 0)
d, lo, hi = ci("psat:srk_ms_vs_srk_t95")
C("SRK ms vs t95 +0.03", 0.03, d); C("lo -0.04", -0.04, lo); C("hi +0.12", 0.12, hi)
d, lo, hi = ci("psat:pr_ms_vs_pr_s")
C("PR ms vs s 0.00", 0.00, d); C("lo -0.22", -0.22, lo); C("hi +0.23", 0.23, hi)
d, lo, hi = ci("psat:pr_coq_vs_pr_s")
C("PR coq vs s +0.27", 0.27, d); C("lo +0.06", 0.06, lo); C("hi +0.53", 0.53, hi)
C("PR coq vs s wins 16", 16, cmp2.loc["psat:pr_coq_vs_pr_s", "wins"], 0)
C("SRK coq vs s wins 28", 28, cmp2.loc["psat:srk_coq_vs_srk_s", "wins"], 0)
C("PR-Soave-19 bias -1.06", -1.06, pool3.loc[("pr_s19", "psat"), "bias|All"])
fams = ["Inorganic gases", "Alkanes", "Aromatics", "Polar/associating", "Refrigerants"]
C("PR-Soave-19 bias negative in every family", 1, int(all(pool3.loc[("pr_s19", "psat"), f"bias|{f}"] < 0 for f in fams)), 0)
# ---- Section 3.2.3: the gap
d, lo, hi = ci("psat:pr_t95_vs_srk_t95")
C("Twu-95 pair 0.00", 0.00, d); C("lo -0.02", -0.02, lo); C("hi +0.01", 0.01, hi)
d, lo, hi = ci("h:pr_t95_vs_srk_t95")
C("Twu-95 pair h -0.11", -0.11, d); C("lo -0.19", -0.19, lo); C("hi -0.03", -0.03, hi)
C("h srk t95 2.29", 2.29, o["h_srk_t95"]); C("h pr t95 2.19", 2.19, o["h_pr_t95"])
C("coq gap 0.08", 0.08, -cmp2.loc["psat:pr_coq_vs_srk_coq", "dAAD"]); C("coq gap upper 0.10", 0.10, -cmp2.loc["psat:pr_coq_vs_srk_coq", "ci_lo"])
C("ms gap 0.14", 0.14, -cmp2.loc["psat:pr_ms_vs_srk_ms", "dAAD"]); C("ms gap upper 0.18", 0.18, -cmp2.loc["psat:pr_ms_vs_srk_ms", "ci_lo"])
C("coq pair wins 37", 37, cmp2.loc["psat:pr_coq_vs_srk_coq", "wins"], 0); C("ms pair wins 36", 36, cmp2.loc["psat:pr_ms_vs_srk_ms", "wins"], 0)
C("tf gap 0.16", 0.16, -cmp2.loc["psat:pr_tf_vs_srk_tf", "dAAD"]); C("tf gap CI 0.08", 0.08, -cmp2.loc["psat:pr_tf_vs_srk_tf", "ci_hi"]); C("tf gap CI 0.25", 0.25, -cmp2.loc["psat:pr_tf_vs_srk_tf", "ci_lo"])
C("tf gap h 0.22", 0.22, -cmp2.loc["h:pr_tf_vs_srk_tf", "dAAD"])
r3 = cmp3.loc["psat:pr_sfit_vs_srk_sfit"]
C("Soave-is PR 0.83", 0.83, r3.aad_A); C("Soave-is SRK 1.41", 1.41, r3.aad_B); C("Soave-is CI 0.39", 0.39, -r3.ci_hi); C("Soave-is CI 0.78", 0.78, -r3.ci_lo); C("Soave-is wins 31", 31, r3.wins, 0)
r3 = cmp3.loc["psat:pr_tfit_vs_srk_tfit"]
C("Twu-is PR 0.25", 0.25, r3.aad_A); C("Twu-is SRK 0.40", 0.40, r3.aad_B); C("Twu-is CI 0.09", 0.09, -r3.ci_hi); C("Twu-is CI 0.22", 0.22, -r3.ci_lo); C("Twu-is wins 36", 36, r3.wins, 0)
C("Twu-is Holm 1e-8", 1e-8, holm.loc["psat:pr_tfit_vs_srk_tfit", "p_holm_ext"], 0.5e-8)
r3 = cmp3.loc["h:pr_tfit_vs_srk_tfit"]
C("Twu-is h PR 1.75", 1.75, r3.aad_A); C("Twu-is h SRK 1.94", 1.94, r3.aad_B); C("Twu-is h gap 0.19", 0.19, -r3.dAAD); C("CI 0.15", 0.15, -r3.ci_hi); C("CI 0.24", 0.24, -r3.ci_lo); C("wins 34", 34, r3.wins, 0)
for rng, val in (("0.50-0.70", 0.05), ("0.70-0.90", 0.21), ("0.90-0.985", 0.25)):
    C(f"Twu-is SRK deficit {rng}", val, pool3.loc[("srk_tfit", "psat"), f"aad|{rng}"] - pool3.loc[("pr_tfit", "psat"), f"aad|{rng}"])
C("active alpha''' SRK 37", 37, int(fits[fits.cubic.str.upper() == "SRK"].active_Q3.sum()), 0)
C("active alpha''' PR 22", 22, int(fits[fits.cubic.str.upper() == "PR"].active_Q3.sum()), 0)
C("unc SRK 0.24", 0.24, pool3.loc[("srk_tunc", "psat"), "aad|All"]); C("unc PR 0.22", 0.22, pool3.loc[("pr_tunc", "psat"), "aad|All"])
C("unc gap 0.02", 0.02, pool3.loc[("srk_tunc", "psat"), "aad|All"] - pool3.loc[("pr_tunc", "psat"), "aad|All"])
floor_srk, floor_pr = pool3.loc[("srk_tfit", "psat"), "aad|All"], pool3.loc[("pr_tfit", "psat"), "aad|All"]
C("floor difference 0.15", 0.15, floor_srk - floor_pr)
lossdiff = {a: (o[f"psat_srk_{a}"] - floor_srk) - (o[f"psat_pr_{a}"] - floor_pr) for a in GEN}
C("loss diff Twu-95 -0.15", -0.15, lossdiff["t95"]); C("loss diff Soave +0.53", 0.53, lossdiff["s"])
C("loss diff min is t95", 1, int(min(lossdiff, key=lossdiff.get) == "t95"), 0); C("loss diff max is s", 1, int(max(lossdiff, key=lossdiff.get) == "s"), 0)
# ---- Section 3.2.4
C("in-sample better than published, SRK every fluid", 37, cmp3.loc["psat:srk_tfit_vs_srk_tf", "wins"], 0)
C("in-sample better than published, PR every fluid", 37, cmp3.loc["psat:pr_tfit_vs_pr_tf", "wins"], 0)
C("by 0.52 on average", 0.52, -0.5 * (cmp3.loc["psat:srk_tfit_vs_srk_tf", "dAAD"] + cmp3.loc["psat:pr_tfit_vs_pr_tf", "dAAD"]))
C("alpha closeness < 0.8%", 1, int(clo.max_rel_dalpha_vs_tf.max() < 0.8), 0)
# ---- Section 3.3
C("PR78 m methanol +0.009", 0.009, pr78j["m"]["methanol"]["m78"] - pr78j["m"]["methanol"]["m76"], 0.0006)
C("PR78 m ethanol +0.016", 0.016, pr78j["m"]["ethanol"]["m78"] - pr78j["m"]["ethanol"]["m76"], 0.0006)
C("methanol 5.04 -> 5.69", 5.04, pr78.loc[("methanol", "All"), "psat_PR76"]); C("methanol PR78 5.69", 5.69, pr78.loc[("methanol", "All"), "psat_PR78"])
C("ethanol 2.02 -> 2.33", 2.02, pr78.loc[("ethanol", "All"), "psat_PR76"]); C("ethanol PR78 2.33", 2.33, pr78.loc[("ethanol", "All"), "psat_PR78"])
m05 = pr78p[(pr78p.fluid == "methanol")].sort_values("Tr").iloc[0]
C("methanol Tr0.5 PR76 -26.5", -26.5, m05.dev_p_76, 0.05); C("methanol Tr0.5 PR78 -30.1", -30.1, m05.dev_p, 0.05)
for fluid, srk, pr in (("methanol", 6.32, 4.47), ("water", 3.50, 1.92)):
    C(f"Soave-is {fluid} SRK", srk, pf3.loc[fluid, "psat_srk_sfit"]); C(f"Soave-is {fluid} PR", pr, pf3.loc[fluid, "psat_pr_sfit"])
C("Twu-is H-bond SRK 0.67", 0.67, pool3.loc[("srk_tfit", "psat"), "aad|H-bonding"]); C("Twu-is H-bond PR 0.42", 0.42, pool3.loc[("pr_tfit", "psat"), "aad|H-bonding"])
C("Twu-is alkanes SRK 0.33", 0.33, pool3.loc[("srk_tfit", "psat"), "aad|Alkanes"]); C("Twu-is alkanes PR 0.21", 0.21, pool3.loc[("pr_tfit", "psat"), "aad|Alkanes"])
for fluid, srk, pr in (("water", 0.42, 0.19), ("methanol", 1.38, 0.27), ("ethanol", 0.66, 1.07), ("ammonia", 0.23, 0.16)):
    C(f"Twu-is {fluid} SRK", srk, pf3.loc[fluid, "psat_srk_tfit"]); C(f"Twu-is {fluid} PR", pr, pf3.loc[fluid, "psat_pr_tfit"])
C("methanol SRK unconstrained 0.15", 0.15, pf3.loc["methanol", "psat_srk_tunc"])
C("ethanol SRK unconstrained 0.59", 0.59, pf3.loc["ethanol", "psat_srk_tunc"]); C("ethanol PR unconstrained 1.07", 1.07, pf3.loc["ethanol", "psat_pr_tunc"])
C("methanol SRK alpha''' active", 1, int(fits[(fits.fluid == "methanol") & (fits.cubic.str.upper() == "SRK")].active_Q3.iloc[0]), 0)
# ---- Section 3.4: near-critical share of the translated PR error
a = df.dev_rho_pr_s_vt.abs(); hi = df.Tr >= 0.90
C("share of PR-VT error from Tr>=0.90 (65%)", 65, 100 * a[hi].sum() / a.sum(), 0.5)
C("share of points Tr>=0.90 (18%)", 18, 100 * hi.mean(), 0.5)
C("PR-VT pooled 3.11", 3.11, a.mean())
C("PR-VT rho < 0.90 (1.35)", 1.35, a[~hi].mean()); C("PR-VT rhov < 0.90 (1.80)", 1.80, df.dev_rhov_pr_s_vt.abs()[~hi].mean())
# ---- Section 4.1: PR equivalent set interval width
w = max(max(abs(cmp2.loc[key, "ci_lo"]), abs(cmp2.loc[key, "ci_hi"])) for key in ("psat:pr_tc_vs_pr_s", "psat:pr_ms_vs_pr_s", "psat:pr_ms_vs_pr_tc"))
C("PR equivalent set within +-0.26 (max |bound| <= 0.26 and > 0.25)", 1, int(0.25 < w <= 0.26), 0)
# ---- Section 4.4: non-associating fluids, 0.70-0.90
sub = dict(zip(fl.fluid, fl.subgroup))
m = (df.Tr >= 0.70) & (df.Tr < 0.90) & (df.fluid.map(sub) != "H-bonding")
C("non-assoc n fluids 33", 33, df.loc[m, "fluid"].nunique(), 0)
C("non-assoc psat SRK 1.09", 1.09, df.loc[m, "dev_p_srk_s"].abs().mean()); C("non-assoc psat PR 0.44", 0.44, df.loc[m, "dev_p_pr_s"].abs().mean())
C("non-assoc h SRK 2.12", 2.12, df.loc[m, "dev_h_srk_s"].abs().mean()); C("non-assoc h PR 1.52", 1.52, df.loc[m, "dev_h_pr_s"].abs().mean())
# ---- Section 4.5: sensitivities and compilation constants
S = tab["sensitivities"]["PR"]
C("dlnp/dlnTc at 0.7 (-10)", -10, S["Tr=0.7"]["dlnp_dTc"]["median"], 0.3); C("dlnp/dlnTc at 0.5 (-15)", -15, S["Tr=0.5"]["dlnp_dTc"]["median"], 0.5)
C("dlnp/dw at 0.7 (2.3 per 0.01 -> 2.3%)", 2.3, -S["Tr=0.7"]["dlnp_dw"]["median"], 0.05); C("dlnp/dw at 0.5 (6.6%)", 6.6, -S["Tr=0.5"]["dlnp_dw"]["median"], 0.05)
C("dlnh/dw about 1", 1.0, S["Tr=0.7"]["dlnh_dw"]["median"], 0.15)
C("dlnp/dlnpc = 1", 1.0, S["Tr=0.7"]["dlnp_dpc"]["median"], 1e-5); C("dlnh/dlnpc = 0", 0.0, S["Tr=0.7"]["dlnh_dpc"]["median"], 1e-6)
A = tab["summary"]["All"]
dps = {f"{c}_{a}": A[f"psat_{c}_{a}_src"] - A[f"psat_{c}_{a}_ref"] for c in ("srk", "pr") for a in GEN}
C("max pooled change 0.12", 0.12, max(abs(v) for v in dps.values()), 0.006)
C("SRK-Soave 1.78 -> 1.77", 1.77, A["psat_srk_s_src"]); C("PR-Soave 1.10 -> 1.22", 1.22, A["psat_pr_s_src"])
for c in ("srk", "pr"):
    ref_order = sorted(GEN, key=lambda a: A[f"psat_{c}_{a}_ref"]); src_order = sorted(GEN, key=lambda a: A[f"psat_{c}_{a}_src"])
    C(f"ranking unchanged {c}", 1, int(ref_order == src_order), 0)
chg = max(((tabpf[f"psat_{c}_{a}_src"] - tabpf[f"psat_{c}_{a}_ref"]).max(), f"{c}_{a}", (tabpf[f"psat_{c}_{a}_src"] - tabpf[f"psat_{c}_{a}_ref"]).idxmax()) for c in ("srk", "pr") for a in GEN)
C("max per-fluid change 1.75", 1.75, chg[0]); C("max per-fluid change is ethylene oxide", 1, int(chg[2] == "ethylene oxide"), 0)
lin = {}
for cubic, t in (("PR", "pr_s"), ("SRK", "srk_s")):
    s_ = sens[sens.cubic == cubic].groupby("fluid")[["dlnp_dTc", "dlnp_dpc"]].mean()
    pred = (s_.dlnp_dpc * tabpf.dpc_pct + s_.dlnp_dTc * tabpf.dTc_pct).reindex(tabpf.index)
    act = tabpf[f"psat_bias_{t}_src"] - tabpf[f"psat_bias_{t}_ref"]
    lin[cubic] = float((pred - act).abs().max())
C("linear prediction of bias change within 0.11", 0.11, max(lin.values()), 0.006)
C("max dTc 0.6", 0.6, tabpf.dTc_pct.abs().max(), 0.05); C("max dpc 3.6", 3.6, tabpf.dpc_pct.abs().max(), 0.05)

# ---- Supplementary Materials text (sections 10, 11, 13)
bias2 = pd.read_csv(os.path.join(DATA, "signed_deviations.csv")).set_index(["prop", "model"])
models14 = [f"{c}_{a}" for c in ("srk", "pr") for a in GEN + ("tf",)]
gen12 = [f"{c}_{a}" for c in ("srk", "pr") for a in GEN]
hb = [bias2.loc[("psat", m), "bias|H-bonding"] for m in gen12 if m != "pr_s"]
C("SI: H-bond bias others min -3.2", -3.2, min(hb), 0.05); C("SI: H-bond bias others max -2.2", -2.2, max(hb), 0.05)
C("SI: H-bond bias PR-Soave -0.9", -0.9, bias2.loc[("psat", "pr_s"), "bias|H-bonding"], 0.05)
C("SI: h bias < 0.90 positive for all 14", 1, int(all(bias2.loc[("h", m), "bias|< 0.90"] > 0 for m in models14)), 0)
C("SI: h bias near-crit negative for all 14", 1, int(all(bias2.loc[("h", m), "bias|0.90-0.985"] < 0 for m in models14)), 0)
C("SI: SRK rho bias negative every range", 1, int(all(bias2.loc[("rho", m), f"bias|{r}"] < 0 for m in models14 if m.startswith("srk") for r in ("0.50-0.70", "0.70-0.90", "0.90-0.985"))), 0)
C("SI: SRK-VT rho bias lowest range +1.3", 1.3, bias2.loc[("rho", "srk_s_vt"), "bias|0.50-0.70"], 0.05)
C("SI: PR-VT rho bias lowest range ~0", 0.0, bias2.loc[("rho", "pr_s_vt"), "bias|0.50-0.70"], 0.1)
C("SI: rhov bias overall negative for all 14", 1, int(all(bias2.loc[("rhov", m), "bias|All"] < 0 for m in models14)), 0)
off = [bias2.loc[("rhov", m), "bias|0.50-0.70"] - bias2.loc[("psat", m), "bias|0.50-0.70"] for m in models14]
C("SI: rhov - psat bias offset min -0.7", -0.7, min(off), 0.05); C("SI: rhov - psat bias offset max -0.6", -0.6, max(off), 0.05)
C("SI: PR-Soave-19 only |bias| > 1 among generalized", 1, int([m for m in gen12 if abs(bias2.loc[("psat", m), "bias|All"]) > 1] == ["pr_s19"]), 0)
ver = json.load(open(os.path.join(DATA, "insample_verification.json")))
C("SI: n fits 74", 74, ver["n_fits"], 0); C("SI: starts within 0.001 min 2", 2, ver["starts_within_0p001_min"], 0)
C("SI: all six starts for 63 fits", 63, int((fits.n_starts_within_0p001 == fits.n_starts).sum()), 0)
sp = ver["spread_params_among_equivalent_starts"]
C("SI: spread L 0.017", 0.017, sp["L"], 0.0006); C("SI: spread M 0.002", 0.002, sp["M"], 0.0006); C("SI: spread N 0.099", 0.099, sp["N"], 0.0006)
C("SI: LS stage within 0.20 (Twu)", 0.20, ver["max_aad_ls_minus_l1_twu"], 0.006); C("SI: LS stage within 0.39 (Soave)", 0.39, ver["max_aad_ls_minus_l1_soave"], 0.006)
C("SI: active third-derivative 59 of 74", 59, ver["active_Q3"], 0); C("SI: P2 never active", 0, ver["active_P2"], 0); C("SI: M=1 never active", 0, ver["active_M1"], 0)
C("SI: unconstrained consistent 13", 13, ver["unconstrained_consistent_count"], 0); C("SI: unconstrained gain max 1.23", 1.23, ver["unconstrained_gain_max"], 0.006)
C("SI: AAD reproduction 1.1e-9", 1.1e-9, ver["aad_reproduction_max_diff"], 0.1e-9); C("SI: Clapeyron fits <= 6e-7", 1, int(max(ver["clapeyron_max"].values()) <= 6e-7), 0)
C("SI: soave scan unimodal all", 1, int(fits.soave_scan_unimodal.all()), 0)
C("SI: worsened generalized combos 10", 10, sum(v > 0 for v in dps.values()), 0)
big = set()
for c in ("srk", "pr"):
    for a in GEN:
        dd = tabpf[f"psat_{c}_{a}_src"] - tabpf[f"psat_{c}_{a}_ref"]; big |= set(dd[dd.abs() > 0.5].index)
C("SI: big-change fluids set", 1, int(big == {"ammonia", "dimethyl ether", "ethylene oxide", "hydrogen sulfide", "methyl chloride", "tetrahydrofuran"}), 0)
mc = tabpf.loc["methyl chloride"]; s_pr = sens[sens.cubic == "PR"].groupby("fluid")[["dlnp_dTc"]].mean()
C("SI: MeCl dpc -3.6", -3.6, mc.dpc_pct, 0.05); C("SI: MeCl dTc -0.6", -0.6, mc.dTc_pct, 0.05)
C("SI: MeCl Tc effect +4.7", 4.7, mc.dTc_pct * s_pr.loc["methyl chloride", "dlnp_dTc"], 0.05)
C("SI: MeCl net bias change +1.1", 1.1, mc.psat_bias_pr_s_src - mc.psat_bias_pr_s_ref, 0.05)
C("SI: rho change 0.1-0.3", 1, int(0.1 <= min(A[f"rho_{m}_src"] - A[f"rho_{m}_ref"] for m in gen12) and max(A[f"rho_{m}_src"] - A[f"rho_{m}_ref"] for m in gen12) <= 0.3), 0)
C("SI: rhov change < 0.1", 1, int(max(abs(A[f"rhov_{m}_src"] - A[f"rhov_{m}_ref"]) for m in gen12) < 0.1), 0)
C("SI: h change < 0.05", 1, int(max(abs(A[f"h_{m}_src"] - A[f"h_{m}_ref"]) for m in gen12) < 0.05), 0)
C("SI: ethanol Tr0.5 PR76 -8.3", -8.3, pr78p[pr78p.fluid == "ethanol"].sort_values("Tr").iloc[0].dev_p_76, 0.05)
C("SI: ethanol Tr0.5 PR78 -16.0", -16.0, pr78p[pr78p.fluid == "ethanol"].sort_values("Tr").iloc[0].dev_p, 0.05)
C("SI: PR78 liquid density change < 0.15", 1, int(max(abs(pr78.loc[(f, "All"), "rho_PR78"] - pr78.loc[(f, "All"), "rho_PR76"]) for f in ("methanol", "ethanol")) < 0.15 and max(abs(pr78.loc[(f, "All"), "rho_vt_PR78"] - pr78.loc[(f, "All"), "rho_vt_PR76"]) for f in ("methanol", "ethanol")) < 0.15), 0)
C("SI: PR78 h and rhov worsen", 1, int(all(pr78.loc[(f, "All"), f"{p_}_PR78"] > pr78.loc[(f, "All"), f"{p_}_PR76"] for f in ("methanol", "ethanol") for p_ in ("h", "rhov"))), 0)


# ---- physical diagnostics quoted in the main text and SI (physical_diagnostics.json, physics_*.csv)
ph = json.load(open(os.path.join(DATA, "physical_diagnostics.json")))
sa = ph["alpha_sensitivity"]
C("physics: dlnp/dlna PR 0.7 (-5.6)", -5.6, sa["PR|0.7"]["dlnp_median"], 0.05)
C("physics: dlnp/dlna PR 0.5 (-9.9)", -9.9, sa["PR|0.5"]["dlnp_median"], 0.05)
C("physics: dlnp/dlna SRK 0.7 (-5.5, SI)", -5.4, sa["SRK|0.7"]["dlnp_median"], 0.05)
C("physics: dlnrho/dlna PR 0.7 (0.4)", 0.4, sa["PR|0.7"]["dlnrhol_median"], 0.05)
C("physics: dlnrho/dlna PR 0.5 (0.2)", 0.2, sa["PR|0.5"]["dlnrhol_median"], 0.05)
C("physics: dlnrho/dlna SRK 0.7 (0.4, SI)", 0.4, sa["SRK|0.7"]["dlnrhol_median"], 0.05)
sp = ph["alpha_spread"]
C("physics: max alpha difference 1.9%", 1.9, sp["max_dalpha_pct"], 0.05)
C("physics: max psat change 19.4 points", 19.4, sp["max_dpsat_pts"], 0.05)
C("physics: max rho change 0.57 points", 0.57, sp["max_drho_pts"], 0.006)
eh = ph["hvap_exponent"]
C("physics: hvap exponent cubics all round to 0.48 (min)", 1, int(round(min(eh["beta_h_srk"]["min"], eh["beta_h_pr"]["min"]), 2) == 0.48), 0)
C("physics: hvap exponent cubics all round to 0.48 (max)", 1, int(round(max(eh["beta_h_srk"]["max"], eh["beta_h_pr"]["max"]), 2) == 0.48), 0)
C("physics: hvap exponent ref median 0.40", 0.40, eh["beta_h_ref"]["median"], 0.006)
C("physics: hvap exponent ref min 0.37", 0.37, eh["beta_h_ref"]["min"], 0.006)
C("physics: hvap exponent ref max 0.48", 0.48, eh["beta_h_ref"]["max"], 0.006)
cl = ph["closure"]
C("physics: width exponent cubic min 0.47", 0.47, min(cl["beta_srk_range"][0], cl["beta_pr_range"][0]), 0.006)
C("physics: width exponent cubic max 0.49", 0.49, max(cl["beta_srk_range"][1], cl["beta_pr_range"][1]), 0.006)
C("physics: width exponent ref min 0.32", 0.32, cl["beta_ref_range"][0], 0.006)
C("physics: width exponent ref max 0.42", 0.42, cl["beta_ref_range"][1], 0.006)
C("physics: closer to own rho_c SRK 37", 37, cl["cubic_closer_srk"], 0)
C("physics: closer to own rho_c PR 36", 36, cl["cubic_closer_pr"], 0)
zz = ph["zc_density"]
C("physics: Zc r SRK 0.95", 0.95, zz["bias_lo_srk"]["r"], 0.006); C("physics: Zc r PR 0.95", 0.95, zz["bias_lo_pr"]["r"], 0.006)
C("physics: Zc n 36", 36, zz["bias_lo_pr"]["n"], 0)
C("physics: Zc slope SRK 3.6", 3.6, zz["bias_lo_srk"]["slope_pct_per_0p01Zc"], 0.05)
C("physics: Zc slope PR 4.1", 4.1, zz["bias_lo_pr"]["slope_pct_per_0p01Zc"], 0.05)
C("physics: Zc zero SRK 0.295", 0.295, zz["bias_lo_srk"]["Zc_at_zero"], 0.0006)
C("physics: Zc zero PR 0.26", 0.26, zz["bias_lo_pr"]["Zc_at_zero"], 0.005)
C("physics: Zc zero PR 0.263 (SI)", 0.263, zz["bias_lo_pr"]["Zc_at_zero"], 0.0006)
chi_ = fl.set_index("fluid").chi_HalmStiel
C("physics: methanol psat(0.6) 10% above normal", 10, 100 * (10 ** chi_["methanol"] - 1), 0.5)
C("physics: water psat(0.6) 6% above normal", 6, 100 * (10 ** chi_["water"] - 1), 0.5)
zcr = fl.set_index("fluid").Zc_ref
hbz = [zcr[x] for x in ("water", "methanol", "ethanol", "ammonia")]
C("physics: H-bond Zc min 0.22", 0.22, min(hbz), 0.005); C("physics: H-bond Zc max 0.25", 0.25, max(hbz), 0.005)
C("physics: Zc max of set 0.294", 0.294, zcr.max(), 0.0006); C("physics: Zc min of set 0.219", 0.219, zcr.min(), 0.0006)
C("physics: methyl chloride Zc 0.294 (joint highest with oxygen)", 1, int(abs(zcr["methyl chloride"] - 0.294) < 0.0005 and abs(zcr["oxygen"] - 0.294) < 0.0005 and zcr.drop(["methyl chloride", "oxygen"]).max() < 0.2935), 0)
_zd = pd.read_csv(os.path.join(DATA, "physics_zc_density.csv")).dropna(subset=["bias_lo_pr"]).set_index("fluid")
for _col, _res, _lab in (("bias_lo_srk", -7.4, "SRK"), ("bias_lo_pr", -8.3, "PR")):
    _s, _i = np.polyfit(_zd.Zc_ref, _zd[_col], 1); _r = _zd[_col] - (_s * _zd.Zc_ref + _i)
    C(f"physics: methyl chloride farthest from {_lab} line", 1, int(_r.abs().idxmax() == "methyl chloride"), 0)
    C(f"physics: methyl chloride residual {_lab} {_res}", _res, _r["methyl chloride"], 0.05)
    C(f"physics: methyl chloride equivalent Zc {_lab} 0.274", 0.274, _zd.Zc_ref["methyl chloride"] + _r["methyl chloride"] / _s, 0.0006)
# translation constants in covolumes and the methanol negative-volume region (SI Section 5)
R_ = 8.314462618
for cub, Ob, lo, hi in (("srk", 0.08664, 0.45, 1.32), ("pr", 0.07780, 0.17, 1.14)):
    bcm = Ob * R_ * fl.Tc_K / (fl.pc_MPa * 1e6) * 1e6
    s_zc = fl[f"c_zc_{cub}_cm3mol"] / bcm
    C(f"physics: c_Zc/b {cub} min {lo}", lo, s_zc.min(), 0.006); C(f"physics: c_Zc/b {cub} max {hi}", hi, s_zc.max(), 0.006)
for cub, Ob, lo, hi in (("srk", 0.08664, 0.01, 0.36), ("pr", 0.07780, -0.17, 0.23)):
    bcm = Ob * R_ * fl.Tc_K / (fl.pc_MPa * 1e6) * 1e6
    s_fit = fl[f"c_fit_{cub}_s_cm3mol"] / bcm
    C(f"physics: s_fit {cub} min {lo}", lo, s_fit.min(), 0.006); C(f"physics: s_fit {cub} max {hi}", hi, s_fit.max(), 0.006)
mneg = df[(df.fluid == "methanol") & df.dev_rho_pr_s_vtz.isna()]
C("physics: methanol negative volumes below Tr 0.53", 1, int(len(mneg) > 0 and mneg.Tr.max() < 0.53), 0)

# ---- n-decane mechanism (Section 4.2, SI Physical Diagnostics), vapor Z'' below Tr 0.70 (Section 3.5)
dx = ph["decane_example"]
C("physics: decane PR Twu-95 dalpha 1.2 (1.19 SI)", 1.19, dx["PR"]["dalpha_pct_t95"], 0.005)
C("physics: decane PR MS dalpha 1.0 (0.99 SI)", 0.99, dx["PR"]["dalpha_pct_ms"], 0.005)
C("physics: decane PR sensitivity -11.5 (-11.49 SI)", -11.49, dx["PR"]["dlnp_dlna"], 0.005)
C("physics: decane PR Twu-95 psat change -13 (-12.80 SI)", -12.80, dx["PR"]["dpsat_pct_t95"], 0.005)
C("physics: decane PR MS psat change -11 (-10.72 SI)", -10.72, dx["PR"]["dpsat_pct_ms"], 0.005)
C("physics: decane PR dev classical +11.2", 11.2, dx["PR"]["dev_classical"], 0.05)
C("physics: decane PR dev Twu-95 -3.0", -3.0, dx["PR"]["dev_t95"], 0.05)
C("physics: decane PR dev MS -0.7", -0.7, dx["PR"]["dev_ms"], 0.05)
C("physics: decane SRK |dalpha| < 0.2 both", 1, int(max(abs(dx["SRK"]["dalpha_pct_t95"]), abs(dx["SRK"]["dalpha_pct_ms"])) < 0.2), 0)
C("physics: decane SRK dev classical -2.51", -2.51, dx["SRK"]["dev_classical"], 0.005)
for _al in ("t95", "ms"):
    _pred = 100 * (np.exp(dx["PR"]["dlnp_dlna"] * np.log(1 + dx["PR"]["dalpha_pct_" + _al] / 100)) - 1)
    C(f"physics: decane sensitivity predicts {_al} psat change within 0.1 points", 1, int(abs(_pred - dx["PR"]["dpsat_pct_" + _al]) < 0.1), 0)
_Zv = df.p_ref_Pa / (df.rhov_ref_molm3 * 8.314462618 * df.T_K)
C("physics: reference Z'' below Tr 0.70 min 0.90", 0.90, _Zv[df.Tr < 0.70].min(), 0.005)
C("physics: reference Z'' below Tr 0.70 max < 1", 1, int(_Zv[df.Tr < 0.70].max() < 1.0), 0)
_hb = df[df.fluid.isin(["water", "methanol", "ethanol", "ammonia"])]
C("physics: H-bond PR density reduced by about two thirds", 0.66, 1 - _hb.dev_rho_pr_s_vt.abs().mean() / _hb.dev_rho_pr_s.abs().mean(), 0.04)
C("physics: sensitivity n at Tr 0.5/0.7/0.9 = 28/36/37", 1, int([sa["PR|0.5"]["n"], sa["PR|0.7"]["n"], sa["PR|0.9"]["n"]] == [28, 36, 37]), 0)

# ---- sign of the PR density-anchored constant against Zc (SI Section 5); Figure S2b statements
_neg = fl[fl.c_fit_pr_s_cm3mol < 0]; _pos = fl[fl.c_fit_pr_s_cm3mol >= 0]
C("physics: PR c_fit negative for 23", 23, len(_neg), 0); C("physics: PR c_fit non-negative for 14", 14, len(_pos), 0)
C("physics: PR c_fit negative Zc min 0.265", 0.265, _neg.Zc_ref.min(), 0.0006); C("physics: PR c_fit negative Zc max 0.294", 0.294, _neg.Zc_ref.max(), 0.0006)
C("physics: PR c_fit non-negative Zc min 0.219", 0.219, _pos.Zc_ref.min(), 0.0006); C("physics: PR c_fit non-negative Zc max 0.266", 0.266, _pos.Zc_ref.max(), 0.0006)
import sys as _sys
_sys.path.insert(0, HERE)
from eos import CubicEOS as _CE
_b = _CE("PR", 600.0, 3e6, 0.30, alpha="s")
_d = {al: [(_CE("PR", 600.0, 3e6, 0.30, alpha=al).alpha(t * 600.0)[0] / _b.alpha(t * 600.0)[0] - 1) * 100 for t in np.arange(0.5, 0.986, 0.005)] for al in ("s19", "t95", "tc", "coq", "ms")}
_tg = np.arange(0.5, 0.986, 0.005)
C("physics: S2b all within 0.7% (rounded)", 1, int(max(max(abs(x) for x in v) for v in _d.values()) < 0.75), 0)
C("physics: S2b Soave-19 above everywhere", 1, int(min(_d["s19"]) > 0), 0)
for _al in ("t95", "coq", "ms"):
    _cross = _tg[np.argmax(np.array(_d[_al]) < 0)]
    C(f"physics: S2b {_al} crosses below near 0.65", 0.65, _cross, 0.03)
    C(f"physics: S2b {_al} below classical above crossing", 1, int(all(x <= 0.005 for x, t in zip(_d[_al], _tg) if t > _cross)), 0)
C("physics: S2b Twu-c within 0.03% above Tr 0.7", 1, int(max(abs(x) for x, t in zip(_d["tc"], _tg) if t >= 0.7) < 0.035), 0)

# ---- Clapeyron split, SRK-PR gap, hydrogen-bonding scope, Rackett level, displaced critical points, input error
TJ = json.load(open(os.path.join(HERE, "..", "tables", "tables.json")))
def trow(tab, label):
    return [r for r in TJ[tab]["rows"] if isinstance(r, list) and r[0] == label][0]
def num(x):
    return float(str(x).replace("$", "").replace("+", "").replace("−", "-").split(" ")[0])
hs = ph["hvap_split"]
C("physics: dZ part lo min 0.6", 0.6, min(hs["srk|lo"]["dZ"][0], hs["pr|lo"]["dZ"][0]), 0.05)
C("physics: dZ part lo max 0.7", 0.7, max(hs["srk|lo"]["dZ"][1], hs["pr|lo"]["dZ"][1]), 0.05)
C("physics: dZ part mid SRK 1.5", 1.5, hs["srk|mid"]["dZ"][0], 0.05); C("physics: dZ part mid SRK 1.6", 1.6, hs["srk|mid"]["dZ"][1], 0.05)
C("physics: dZ part mid PR 1.0", 1.0, hs["pr|mid"]["dZ"][0], 0.05); C("physics: dZ part mid PR 1.3", 1.3, hs["pr|mid"]["dZ"][1], 0.05)
C("physics: Z'' mid SRK 2.1", 2.1, hs["srk|mid"]["Zv"][0], 0.05); C("physics: Z'' mid SRK 2.2", 2.2, hs["srk|mid"]["Zv"][1], 0.05)
C("physics: Z'' mid PR 1.1", 1.1, hs["pr|mid"]["Zv"][0], 0.05); C("physics: Z'' mid PR 1.3", 1.3, hs["pr|mid"]["Zv"][1], 0.05)
C("physics: dZ larger share mid (both cubics)", 1, int(hs["srk|mid"]["dZ_share_min"] > 0.5 and hs["pr|mid"]["dZ_share_min"] > 0.5), 0)
C("physics: dZ dominant near-critical (both cubics)", 1, int(hs["srk|nc"]["dZ_share_min"] > 0.5 and hs["pr|nc"]["dZ_share_min"] > 0.5), 0)
C("physics: Twu-is slope part lo ~0", 1, int(abs(hs["srk|lo"]["tfit"]["slope"]) < 0.05 and abs(hs["pr|lo"]["tfit"]["slope"]) < 0.05), 0)
C("physics: Twu-is hvap lo +0.7 SRK", 0.7, hs["srk|lo"]["tfit"]["h"], 0.05); C("physics: Twu-is hvap lo +0.7 PR", 0.7, hs["pr|lo"]["tfit"]["h"], 0.05)
C("physics: near-critical slope and dZ negative", 1, int(max(hs["srk|nc"]["dZ"][1], hs["pr|nc"]["dZ"][1], hs["srk|nc"]["slope"][1], hs["pr|nc"]["slope"][1]) < 0), 0)
sh = {m: num(trow("tabS_signed_h", m)[2]) for m in ("SRK-Soave", "SRK-Soave-19", "SRK-Twu-95", "SRK-Twu-c", "SRK-Coquelet", "SRK-MS", "SRK-Twu-fit$^{\\dagger}$",
                                                   "PR-Soave", "PR-Soave-19", "PR-Twu-95", "PR-Twu-c", "PR-Coquelet", "PR-MS", "PR-Twu-fit$^{\\dagger}$")}
C("physics: S17 mid SRK min 2.0", 2.0, min(v for k, v in sh.items() if k.startswith("SRK")), 0.05)
C("physics: S17 mid SRK max 2.3", 2.3, max(v for k, v in sh.items() if k.startswith("SRK")), 0.05)
C("physics: S17 mid PR min 1.4", 1.4, min(v for k, v in sh.items() if k.startswith("PR")), 0.05)
C("physics: S17 mid PR max 2.3", 2.3, max(v for k, v in sh.items() if k.startswith("PR")), 0.05)
hv = {m: num(trow("tab_hvap", m)[5]) for m in sh}
C("physics: hvap alpha range 1.8", 1.8, min(hv.values()), 0.05); C("physics: hvap alpha range 2.8", 2.8, max(hv.values()), 0.05)
C("physics: Twu-is hvap SRK 1.94", 1.94, num(trow("tab_hvap", "SRK-Twu-is$^{\\ddagger}$")[5]), 0.005)
C("physics: Twu-is hvap PR 1.75", 1.75, num(trow("tab_hvap", "PR-Twu-is$^{\\ddagger}$")[5]), 0.005)
C("physics: Twu-is psat SRK 0.40", 0.40, num(trow("tab_psat", "SRK-Twu-is$^{\\ddagger}$")[5]), 0.005)
C("physics: Twu-is psat PR 0.25", 0.25, num(trow("tab_psat", "PR-Twu-is$^{\\ddagger}$")[5]), 0.005)
ps = {m: [num(x) for x in trow("tab_psat", m)[1:4]] for m in sh}
C("physics: psat grows above 0.90 for 13 of 14", 13, sum(v[2] > v[1] for v in ps.values()), 0)
C("physics: psat near-critical below 1.3", 1, int(max(v[2] for v in ps.values()) < 1.3), 0)
_up = 0
for c_ in ("srk", "pr"):
    for a_ in ("s", "s19", "t95", "tc", "coq", "ms", "tf"):
        lo_ = df[(df.Tr >= 0.90) & (df.Tr < 0.95)][f"dev_p_{c_}_{a_}"].abs().mean(); hi_ = df[df.Tr >= 0.95][f"dev_p_{c_}_{a_}"].abs().mean()
        _up += int(hi_ < lo_)
C("physics: psat falls within near-critical range for all 14", 14, _up, 0)
C("physics: Soave-19 PR below 0.70: 1.90 -> 2.52", 1, int(abs(ps["PR-Soave"][0] - 1.90) < 0.005 and abs(ps["PR-Soave-19"][0] - 2.52) < 0.005), 0)
hx = pd.read_csv(os.path.join(DATA, "holm_extended.csv")).set_index("key")
C("physics: Soave-19 SRK-PR not significant after Holm", 1, int(hx.loc["psat:pr_s19_vs_srk_s19", "p_holm_ext"] > 0.05), 0)
C("physics: Twu-c SRK-PR significant after Holm", 1, int(hx.loc["psat:pr_tc_vs_srk_tc", "p_holm_ext"] < 0.05), 0)
C("physics: Coquelet PR sign test not significant", 1, int(hx.loc["psat:pr_coq_vs_pr_s", "p"] > 0.05), 0)
po = {m: num(trow("tab_psat", m)[5]) for m in sh}
C("physics: gap Coquelet 0.08", 0.08, po["SRK-Coquelet"] - po["PR-Coquelet"], 0.011)
C("physics: gap MS 0.14", 0.14, po["SRK-MS"] - po["PR-MS"], 0.011)
C("physics: gap Twu-fit 0.16", 0.16, po["SRK-Twu-fit$^{\\dagger}$"] - po["PR-Twu-fit$^{\\dagger}$"], 0.011)
# hydrogen-bonding scope
GEN6 = ("s", "s19", "t95", "tc", "coq", "ms")
pfa = {f"{c_}_{a_}": df.groupby("fluid")[f"dev_p_{c_}_{a_}"].apply(lambda s_: s_.abs().mean()) for c_ in ("srk", "pr") for a_ in GEN6 + ("tf",)}
best = {c_: pd.concat([pfa[f"{c_}_{a_}"] for a_ in GEN6], axis=1).min(axis=1) for c_ in ("srk", "pr")}
three = ["water", "methanol", "ethanol"]
C("physics: only water/methanol/ethanol above 2% on both cubics", 1, int(set(best["srk"][(best["srk"] > 2.0) & (best["pr"] > 2.0)].index) == set(three)), 0)
C("physics: best SRK three 3.2", 3.2, best["srk"][three].min(), 0.05); C("physics: best SRK three 7.0", 7.0, best["srk"][three].max(), 0.05)
C("physics: best PR three 2.0", 2.0, best["pr"][three].min(), 0.05); C("physics: best PR three 5.0", 5.0, best["pr"][three].max(), 0.05)
aprot = fl.set_index("fluid").index[fl.set_index("fluid").subgroup == "polar aprotic"]
C("physics: aprotic each below 1% on PR with best", 1, int(best["pr"][aprot].max() < 1.0), 0)
allgen3 = pd.concat([pfa[f"{c_}_{a_}"][three] for c_ in ("srk", "pr") for a_ in GEN6])
C("physics: rule 3: 2-8% (min)", 2.0, allgen3.min(), 0.05); C("physics: rule 3: 2-8% (max)", 8.0, allgen3.max(), 0.5)
C("physics: ammonia PR-Soave 0.7", 0.7, pfa["pr_s"]["ammonia"], 0.05)
C("physics: ammonia range 0.7-2.4", 1, int(abs(min(pfa[f"{c_}_{a_}"]["ammonia"] for c_ in ("srk", "pr") for a_ in GEN6) - 0.71) < 0.01 and abs(max(pfa[f"{c_}_{a_}"]["ammonia"] for c_ in ("srk", "pr") for a_ in GEN6) - 2.42) < 0.01), 0)
for f_ in ("water", "methanol"):
    for c_ in ("srk", "pr"):
        C(f"physics: Twu-fit reduces {f_} {c_} several-fold", 1, int(best[c_][f_] / pfa[f"{c_}_tf"][f_] > 3), 0)
insp = pd.read_csv(os.path.join(DATA, "insample_points.csv"))
tis = {c_: insp.groupby("fluid")[f"dev_p_{c_}_tfit"].apply(lambda s_: s_.abs().mean()) for c_ in ("srk", "pr")}
eth = [tis["srk"]["ethanol"], tis["pr"]["ethanol"], pfa["srk_tf"]["ethanol"], pfa["pr_tf"]["ethanol"]]
C("physics: ethanol fitted 0.7", 0.7, min(eth), 0.05); C("physics: ethanol fitted 1.8", 1.8, max(eth), 0.05)
s15 = {r[0]: r for r in TJ["tabS_twufit_src"]["rows"] if isinstance(r, list)}
C("physics: S15 H-bond SRK change -0.09", -0.09, num(s15["H-bonding (pooled)"][4]) - num(s15["H-bonding (pooled)"][3]), 0.006)
C("physics: S15 H-bond PR change +0.17", 0.17, num(s15["H-bonding (pooled)"][6]) - num(s15["H-bonding (pooled)"][5]), 0.006)
C("physics: S15 aprotic SRK change -0.7", -0.7, num(s15["Polar aprotic (pooled)"][4]) - num(s15["Polar aprotic (pooled)"][3]), 0.05)
C("physics: S15 aprotic PR change -0.8", -0.8, num(s15["Polar aprotic (pooled)"][6]) - num(s15["Polar aprotic (pooled)"][5]), 0.05)
rws = [r for r in TJ["tabS_twufit_src"]["rows"] if isinstance(r, list) and not str(r[0]).endswith("(pooled)")]
for c_, col_, val_ in (("srk", 4, 0.34), ("pr", 6, 0.35)):
    C(f"physics: Twu-fit(src) - Twu-is {c_} {val_}", val_, np.mean([num(r[col_]) - tis[c_][r[0]] for r in rws]), 0.006)
# Rackett-anchored level on PR
u_ = df.groupby("fluid").dev_rho_pr_s.apply(lambda s_: s_.abs().mean()); v_ = df.groupby("fluid").dev_rho_pr_s_vtc.apply(lambda s_: s_.abs().mean())
fam_ = fl.set_index("fluid").family; sub_ = fl.set_index("fluid").subgroup
C("physics: VT_corr worsens 15 PR fluids", 15, int((v_ > u_).sum()), 0)
C("physics: VT_corr worsens 5 of 6 aprotic", 5, int((v_ > u_)[sub_ == "polar aprotic"].sum()), 0)
C("physics: VT_corr worsens 5 of 6 aromatics", 5, int((v_ > u_)[fam_ == "Aromatics"].sum()), 0)
C("physics: VT_corr helps all inorganic gases", 1, int((v_ <= u_)[fam_ == "Inorganic gases"].all()), 0)
C("physics: VT_corr helps 10 of 11 alkanes", 10, int((v_ < u_)[fam_ == "Alkanes"].sum()), 0)
C("physics: VT_corr helps methanol", 1, int(v_["methanol"] < u_["methanol"]), 0)
flc = fl.set_index("fluid")
for f_, cc_, cf_ in (("acetone", -0.68, 10.50), ("water", 0.20, 4.35), ("methanol", 5.37, 6.89), ("argon", -3.29, -3.40)):
    C(f"physics: c_corr {f_}", cc_, flc.c_corr_pr_s_cm3mol[f_], 0.006); C(f"physics: c_fit {f_}", cf_, flc.c_fit_pr_s_cm3mol[f_], 0.006)
# displaced critical points of two reference equations
zrd = ph["zc_reducing"]
C("physics: displaced: only 2 fluids with dTc > 0.05%", 2, zrd["n_dTc_above_0p05pct"], 0)
d_ = zrd["displaced"]
C("physics: methyl chloride Tc 2.3 K above reducing", 2.3, flc.Tc_K["methyl chloride"] - d_["methyl chloride"]["T_red"], 0.05)
C("physics: n-heptane Tc 1.1 K above reducing", 1.1, flc.Tc_K["n-heptane"] - d_["n-heptane"]["T_red"], 0.05)
C("physics: methyl chloride pc 3.6% above", 3.6, d_["methyl chloride"]["dpc_pct"], 0.05); C("physics: n-heptane pc 1.5% above", 1.5, d_["n-heptane"]["dpc_pct"], 0.05)
C("physics: methyl chloride Zc red 0.269", 0.269, d_["methyl chloride"]["Zc_red"], 0.0006); C("physics: n-heptane Zc red 0.263", 0.263, d_["n-heptane"]["Zc_red"], 0.0006)
C("physics: n-heptane Zc ref 0.275", 0.275, flc.Zc_ref["n-heptane"], 0.0006)
import CoolProp.CoolProp as _CP
for f_, cpn_, val_ in (("methyl chloride", "R40", -5.9), ("n-heptane", "n-Heptane", -3.1)):
    C(f"physics: {f_} rhoc {val_}% vs reducing", val_, 100 * (_CP.PropsSI("rhomolar_critical", cpn_) / _CP.PropsSI("rhomolar_reducing", cpn_) - 1), 0.05)
rg_ = zrd["regression"]
C("physics: farthest = methyl chloride, 3rd = n-heptane (SRK)", 1, int(rg_["bias_lo_srk"]["farthest"][0] == "methyl chloride" and rg_["bias_lo_srk"]["farthest"][2] == "n-heptane"), 0)
C("physics: farthest = methyl chloride, 3rd = n-heptane (PR)", 1, int(rg_["bias_lo_pr"]["farthest"][0] == "methyl chloride" and rg_["bias_lo_pr"]["farthest"][2] == "n-heptane"), 0)
C("physics: n-heptane residual SRK -3.7", -3.7, rg_["bias_lo_srk"]["resid"]["n-heptane"], 0.05); C("physics: n-heptane residual PR -4.3", -4.3, rg_["bias_lo_pr"]["resid"]["n-heptane"], 0.05)
C("physics: n-heptane equivalent Zc 0.264", 0.264, rg_["bias_lo_pr"]["equiv_Zc"]["n-heptane"], 0.0006)
C("physics: r with reducing Zc of the two 0.97 (SRK)", 0.97, rg_["bias_lo_srk"]["r_red_two"], 0.005); C("physics: r with reducing Zc of the two 0.97 (PR)", 0.97, rg_["bias_lo_pr"]["r_red_two"], 0.005)
# input error wording
tpf = pd.read_csv(os.path.join(DATA, "compilation_constants_per_fluid.csv"))
imp = [int((tpf[f"psat_{c_}_{a_}_src"] < tpf[f"psat_{c_}_{a_}_ref"]).sum()) for c_ in ("srk", "pr") for a_ in GEN6]
C("physics: 13-20 fluids improve with compilation constants (min)", 13, min(imp), 0); C("physics: 13-20 fluids improve (max)", 20, max(imp), 0)
s25 = [r for r in TJ["tabS_sens"]["rows"] if isinstance(r, list) and "0.7" in r[0]][0]
C("physics: omega sensitivity -2.13 to -2.45 at 0.7", 1, int(s25[4].replace("−", "-") == "-2.31 (-2.45 to -2.13)"), 0)

# ---- revision pass after the independent check: vapor branch near Tc, unconstrained per-fluid wins, in-sample bounds
_top = df[df.Tr >= 0.985]
C("vapor branch top point PR -1.3", -1.3, _top.dev_rhov_pr_s.mean(), 0.05); C("vapor branch top point SRK -6.3", -6.3, _top.dev_rhov_srk_s.mean(), 0.05)
C("liquid branch top point PR -15.7", -15.7, _top.dev_rho_pr_s.mean(), 0.05); C("liquid branch top point SRK -24.0", -24.0, _top.dev_rho_srk_s.mean(), 0.05)
_cl = json.load(open(os.path.join(DATA, "physical_diagnostics.json")))["closure"]
C("tabS_physics vapor top point PR", -1.3, _cl["mean_dev_top_rhov_pr"], 0.05); C("tabS_physics vapor top point SRK", -6.3, _cl["mean_dev_top_rhov_srk"], 0.05)
C("tabS_physics liquid top point PR", -15.7, _cl["mean_dev_top_rho_pr"], 0.05); C("tabS_physics liquid top point SRK", -24.0, _cl["mean_dev_top_rho_srk"], 0.05)
_cmp3 = pd.read_csv(os.path.join(DATA, "insample_comparisons.csv")).set_index("key")
C("unconstrained PR wins 35 of 37", 35, int(_cmp3.loc["psat:pr_tunc_vs_srk_tunc", "wins"]), 0)
C("unconstrained gap 0.02", 0.02, -float(_cmp3.loc["psat:pr_tunc_vs_srk_tunc", "dAAD"]), 0.006)
_fl2 = TJ["tabS_floorloss"]["rows"]
C("Soave-is loss part +0.43", 0.43, float([r for r in _fl2 if isinstance(r, list) and r[0].startswith("Soave-is")][0][7]), 0.006)
_insp = pd.read_csv(os.path.join(DATA, "insample_points.csv"))
_tis = {f: _insp[_insp.fluid == f].dev_p_pr_tfit.abs().mean() for f in ("water", "methanol", "ammonia", "ethanol")}
C("Twu-is PR water/methanol/ammonia min 0.16", 0.16, min(_tis[f] for f in ("water", "methanol", "ammonia")), 0.006)
C("Twu-is PR water/methanol/ammonia max 0.27", 0.27, max(_tis[f] for f in ("water", "methanol", "ammonia")), 0.006)
C("Twu-is PR ethanol 1.07", 1.07, _tis["ethanol"], 0.006)
_byq = pd.read_csv(os.path.join(DATA, "by_all_comparisons.csv"), index_col=0)
C("BY over 242 comparisons", 242, len(_byq), 0)
_p206 = pd.read_csv(os.path.join(DATA, "paired_comparisons.csv")).set_index("key")
# ---- v6 text (submitted wording restored): ranges quoted in Sections 3.1 and 4.4
_gaps = [-float(_p206.loc[f"psat:pr_{a_}_vs_srk_{a_}", "dAAD"]) for a_ in ("s19", "tc", "coq", "ms")]
C("3.1 gap range other generalized 0.08", 0.08, min(_gaps), 0.006); C("3.1 gap range other generalized 0.36", 0.36, max(_gaps), 0.006)
_fam = pd.read_csv(os.path.join(DATA, "fluids.csv")).set_index("fluid")["family"]
_famaad = [df[df.fluid.map(_fam) == f_][c_].abs().mean() for f_ in _fam.unique() for c_ in ("dev_rho_srk_s", "dev_rho_pr_s")]
C("4.4 rule 4 family AAD min ~4", 4, min(_famaad), 0.5); C("4.4 rule 4 family AAD max ~18", 18, max(_famaad), 0.5)
C("BY-242 changes no status among the 206", 0, int(sum((_p206.loc[k, "q_BY_all206"] < 0.05) != (_byq.loc[k, "q_BY_all"] < 0.05) for k in _p206.index)), 0)

bad = 0
for desc, q, c, tol in checks:
    ok = abs(q - c) <= tol + 1e-12
    bad += (not ok)
    if not ok:
        print(f"MISMATCH  {desc}: text {q} vs data {c}")
print(f"{len(checks)} checks, {bad} mismatches")
raise SystemExit(1 if bad else 0)
