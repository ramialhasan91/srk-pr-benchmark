"""
Cross-check of every headline number quoted in the manuscript text against
the generated data (key_numbers.json, stats.json, run_metadata.json,
fluids.csv). Each entry: (description, value quoted in the text, computed
value, tolerance). Fails loudly on any mismatch.
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
C("python 3.11", 1, int(md["python"].startswith("3.11")), 0)
fine = k["by_bin_fine"]
C("psat smaller in upper half (all four classical/t95)", 1, int(all(fine["0.95-0.985"][f"psat_{t}"] < fine["0.90-0.95"][f"psat_{t}"] for t in ("srk_s", "pr_s", "srk_t95", "pr_t95"))), 0)
C("h, rho, rhov larger in upper half", 1, int(all(fine["0.95-0.985"][f"{p_}_{t}"] > fine["0.90-0.95"][f"{p_}_{t}"] for p_ in ("h", "rho", "rhov") for t in ("srk_s", "pr_s", "srk_t95", "pr_t95"))), 0)

bad = 0
for desc, q, c, tol in checks:
    ok = abs(q - c) <= tol + 1e-12
    bad += (not ok)
    if not ok:
        print(f"MISMATCH  {desc}: text {q} vs data {c}")
print(f"{len(checks)} checks, {bad} mismatches")
raise SystemExit(1 if bad else 0)
