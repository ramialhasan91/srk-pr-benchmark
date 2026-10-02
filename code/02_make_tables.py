"""
Generate every table of the manuscript and of the Supporting Information from
the benchmark data (data/*.csv, key_numbers.json, stats.json).

Two outputs per table:
  tables/<name>.tex   booktabs LaTeX (for the LaTeX sources)
  tables/tables.json  the same content as structured data (header rows, body
                      rows, caption), consumed by make_word.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TABS = os.path.join(HERE, "..", "tables")
os.makedirs(TABS, exist_ok=True)

sys.path.insert(0, HERE)
from fluid_set import FAMILY_ORDER  # noqa: E402

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))
key = json.load(open(os.path.join(DATA, "key_numbers.json")))
st = json.load(open(os.path.join(DATA, "stats.json")))
meta = json.load(open(os.path.join(DATA, "run_metadata.json")))

TAGS = meta["models"]
ALPHA_LABEL = {"s": "Soave", "s19": "Soave-19", "t95": "Twu-95", "tc": "Twu-c", "coq": "Coquelet", "ms": "MS", "tf": "Twu-fit"}
ALPHAS = ["s", "s19", "t95", "tc", "coq", "ms", "tf"]
GEN_ALPHAS = ["s", "s19", "t95", "tc", "coq", "ms"]
CUBIC_LABEL = {"srk": "SRK", "pr": "PR"}


def label(tag):
    c, a = tag.split("_")
    return f"{CUBIC_LABEL[c]}-{ALPHA_LABEL[a]}"


def short(tag):
    """Column header form: cubic on one line, alpha on the next."""
    c, a = tag.split("_")
    return (CUBIC_LABEL[c], ALPHA_LABEL[a])


def f2(x):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def ci(name, src=None):
    d = (src or st["ci"])[name]
    return f"{d['aad']:.2f} ({d['ci_lo']:.2f}--{d['ci_hi']:.2f})"


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def pfmt(p):
    """p-value as plain text with Unicode superscripts (renders identically in Word and LaTeX with fontspec):
    two decimals above 0.10, three decimals between 0.001 and 0.10, one significant figure below."""
    if p >= 0.10:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    return f"{m:.0f} × 10{str(e).translate(SUP)}"


BIN_ROWS = [("0.50-0.70", "0.50–0.70"), ("0.70-0.90", "0.70–0.90"), ("0.90-0.985", "0.90–0.985")]
FAM_ROWS = FAMILY_ORDER
SUB_ROWS = [("H-bonding", "H-bonding (4)"), ("polar aprotic", "polar aprotic (6)")]
fam_n = fl.groupby("family").size().to_dict()

tables = {}


def emit(name, caption, header, rows, colspec=None, notes=None, label_tex=None):
    """Store a table (structured) and write its LaTeX version."""
    tables[name] = dict(caption=caption, header=header, rows=rows, notes=notes or "")
    ncol = len(rows[0]) if rows else len(header[-1])
    colspec = colspec or ("l" + "c" * (ncol - 1))
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{tab:{label_tex or name}}}", "\\small",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    for h in header:
        lines.append(" & ".join(h) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        if r is None:
            lines.append("\\midrule")
        elif isinstance(r, str):
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{r}}}}} \\\\")
        else:
            lines.append(" & ".join(str(x) for x in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\footnotesize {notes}")
    lines += ["\\end{table}", ""]
    with open(os.path.join(TABS, name + ".tex"), "w") as f:
        f.write("\n".join(lines))
    print("wrote", name)


# ====================================================================== #
# Table: the alpha functions (descriptive)
emit(
    "tab_models",
    "The $\\alpha$-functions compared. The first six are generalized (input: $T_c$, $p_c$, $\\omega$ only); "
    "the seventh uses component-specific parameters and serves only as a fitted reference.",
    [["Label", "Form", "Parameterization", "Fitted to", "Ref."]],
    [
        ["Soave", "Soave, Eq. {eq:soave}", "original $m(\\omega)$ of each cubic (SRK 1972; PR 1976)",
         "hydrocarbon $p^{sat}$ (SRK: at $T_r = 0.7$; PR: normal boiling point to $T_c$)", "[cite:soave1972,peng1976]"],
        ["Soave-19", "Soave, Eq. {eq:soave}", "updated third-degree polynomial $m(\\omega)$ for RK and PR",
         "$p^{sat}$, $\\Delta h_{vap}$, $c_p^{liq}$ of 1721 compounds, triple point to $T_c$", "[cite:pina2019]"],
        ["Twu-95", "Twu, Eq. {eq:twu95}", "universal $\\alpha^{(0)}$, $\\alpha^{(1)}$, linear in $\\omega$",
         "hydrocarbon $p^{sat}$, triple point to $T_c$", "[cite:twu1995a,twu1995b]"],
        ["Twu-c", "Twu-91, Eq. {eq:twu91}", "consistent, $N=2$, $L(\\omega)$, $M(\\omega)$ quadratic",
         "$p^{sat}$, $\\Delta h_{vap}$, $c_p^{liq}$ (PR: 1800 fluids; RK: 2018 set)", "[cite:pina2022,pina2018]"],
        ["Coquelet", "exponential, Eq. {eq:coq}", "$C_1$--$C_3$ quadratic in $\\omega$ (correlations of Mahmoodi and Sedigh for PR and SRK, as compiled by Xiao and Yang)",
         "$p^{sat}$ of pure compounds", "[cite:coquelet2004,mahmoodi2017,xiao2025]"],
        ["MS", "exponential, consistent, Eq. {eq:ms}", "$C_1$, $C_2$ quadratic in $\\omega$, $C_3$ Gaussian in $\\omega$ (as compiled by Xiao and Yang)",
         "$p^{sat}$ of pure compounds with consistency constraints", "[cite:mahmoodi2017,xiao2025]"],
        ["Twu-fit", "Twu-91, Eq. {eq:twu91}", "component-specific consistent $(L, M, N)$ per fluid",
         "same database as Twu-c, per fluid", "[cite:pina2022]"],
    ],
    colspec="llp{4.6cm}p{4.2cm}l",
)

# ====================================================================== #
# Table: overall AADs with bootstrap CIs
rows = []
for c in ("srk", "pr"):
    for a in ALPHAS:
        t = f"{c}_{a}"
        rows.append([label(t) + ("$^{\\dagger}$" if a == "tf" else ""),
                     ci(f"psat_{t}"), ci(f"h_{t}"), ci(f"rho_{t}"), ci(f"rhov_{t}")])
rows.append(None)
for c in ("srk", "pr"):
    C = CUBIC_LABEL[c]
    rows.append([f"{C}-VT$_{{corr}}$", "as " + C, "as " + C, ci(f"rho_{c}_s_vtc"), ci(f"rhov_{c}_s_vtc")])
    rows.append([f"{C}-VT", "as " + C, "as " + C, ci(f"rho_{c}_s_vt"), ci(f"rhov_{c}_s_vt")])
emit(
    "tab_overall",
    "Overall average absolute deviation (AAD, \\%) from the reference equations of state, pooled over all "
    f"{key['n_fluids']} fluids and {key['n_points']} saturation states, with fluid-level bootstrap 95\\% confidence "
    "intervals in parentheses. The translated variants (classical Soave $\\alpha$; VT$_{corr}$, Rackett-anchored constant "
    "requiring no data; VT, density-anchored constant requiring one reference liquid density; Section 2.2) share the "
    "$p^{sat}$ and $\\Delta h_{vap}$ of the untranslated model (exact invariance, Section 2.2), and the same constant "
    "is applied to the vapor volume. $^{\\dagger}$Component-specific parameters; fitted reference, not a "
    "predictive model.",
    [["Model", "$p^{sat}$", "$\\Delta h_{vap}$", "$\\rho'$", "$\\rho''$"]],
    rows,
)

# ====================================================================== #
# Tables: psat and hvap AAD by temperature range, family and subgroup (one row per model)
FAM_SHORT = {"Inorganic gases": "Inorg.", "Alkanes": "Alk.", "Aromatics": "Arom.", "Polar/associating": "Polar",
             "Refrigerants": "Refr."}


def by_group_table(prop, name, caption):
    hdr1 = ["", "Reduced-temperature range", "", "", "", "", "Chemical family", "", "", "", "", "Subgroup", ""]
    hdr2 = ["Model", "0.50--0.70", "0.70--0.90", "0.90--0.985", "< 0.90", "All"] + \
           [f"{FAM_SHORT[fam]} ({fam_n[fam]})" for fam in FAM_ROWS] + ["H-bond (4)", "Aprotic (6)"]
    rows = []
    for c in ("srk", "pr"):
        for a in ALPHAS:
            t = f"{c}_{a}"
            if a == "tf":
                rows.append(None)
            rows.append([label(t) + ("$^{\\dagger}$" if a == "tf" else "")]
                        + [f2(key["by_bin"][k][f"{prop}_{t}"]) for k, _ in BIN_ROWS]
                        + [f2(key["below_090"][f"{prop}_{t}"]), f2(key["overall"][f"{prop}_{t}"])]
                        + [f2(key["by_family"][fam][f"{prop}_{t}"]) for fam in FAM_ROWS]
                        + [f2(key["by_subgroup"][k][f"{prop}_{t}"]) for k, _ in SUB_ROWS])
        if c == "srk":
            rows.append(None)
    emit(name, caption, [hdr1, hdr2], rows, colspec="l" + "c" * 12)


by_group_table(
    "psat", "tab_psat",
    "AAD (\\%) in vapor pressure by reduced-temperature range (five columns: the three ranges, the pooled range "
    "$T_r < 0.90$ and all points), by chemical family (number of fluids in parentheses: inorganic gases, alkanes, "
    "aromatics, polar/associating, refrigerants) and by polar/associating subgroup (hydrogen-bonding, polar aprotic), "
    "for the two cubics with each $\\alpha$-function ({tab:tab_models}). All models are strictly predictive except "
    "Twu-fit ($^{\\dagger}$, component-specific parameters, fitted reference). Volume translation does not alter this "
    "property, so the translated variants are omitted.",
)
by_group_table(
    "h", "tab_hvap",
    "AAD (\\%) in the enthalpy of vaporization by reduced-temperature range, chemical family and polar/associating "
    "subgroup. Layout and model set as in {tab:tab_psat}; translated variants are identical to the untranslated models "
    "for this property and are omitted.",
)

# ====================================================================== #
# Table: polar / associating fluids, per fluid
pol = fl[fl.family == "Polar/associating"].copy()
pol["order"] = pol.subgroup.map({"H-bonding": 0, "polar aprotic": 1})
pol = pol.sort_values(["order", "omega"])
rows = []
for sg, lab in SUB_ROWS:
    rows.append(lab.split(" (")[0])
    for _, r in pol[pol.subgroup == sg].iterrows():
        pf = key["per_fluid"][r.fluid]
        rows.append([r.fluid, f"{r.omega:.3f}", f"{r.chi_HalmStiel:.3f}",
                     f2(pf["psat_srk_s"]), f2(pf["psat_srk_t95"]), f2(pf["psat_srk_tf"]),
                     f2(pf["psat_pr_s"]), f2(pf["psat_pr_ms"]), f2(pf["psat_pr_tf"]),
                     f2(pf["rho_pr_s"]), f2(pf["rho_pr_s_vt"])])
emit(
    "tab_polar",
    "The polar and associating fluids, one by one: acentric factor, Halm--Stiel polar factor $\\chi$ from the "
    "reference EoS, per-fluid AAD (\\%) in vapor pressure for the classical $\\alpha$-function, for the generalized "
    "$\\alpha$-function with the lowest pooled vapor-pressure AAD on each cubic (Twu-95 for SRK; MS for PR, which ties with "
    "Soave at two decimals in {tab:tab_overall}) "
    "and for the component-specific fitted reference (Twu-fit$^{\\dagger}$), and per-fluid AAD (\\%) in saturated-liquid "
    "density for untranslated and density-anchored translated PR. Fluids are ordered by subgroup and acentric factor; "
    "the other $\\alpha$-functions are tabulated per fluid in {stab:tabS_perfluid_psat_srk} and {stab:tabS_perfluid_psat_pr}.",
    [["", "", "", "$p^{sat}$, SRK", "", "", "$p^{sat}$, PR", "", "", "$\\rho'$, PR", ""],
     ["Fluid", "$\\omega$", "$\\chi$", "Soave", "Twu-95", "Twu-fit$^{\\dagger}$", "Soave", "MS",
      "Twu-fit$^{\\dagger}$", "PR", "PR-VT"]],
    rows, colspec="lcccccccccc",
)

# ====================================================================== #
# Table: translation ladder by Tr range, both cubics
cols = [("srk_s", ""), ("srk_s", "vtc"), ("srk_s", "vt"), ("pr_s", ""), ("pr_s", "vtc"), ("pr_s", "vt")]


def ladder_val(block, t, sfx):
    return block[f"rho_{t}_{sfx}"] if sfx else block[f"rho_{t}"]


rows = []
for k, lab in BIN_ROWS:
    rows.append([lab] + [f2(ladder_val(key["by_bin"][k], t, s)) for t, s in cols])
cb = st["ci_below_090"]
rows.append(["< 0.90 (pooled)"] + [
    ci(f"rho_{t}" + (f"_{s}" if s else ""), cb) for t, s in cols])
rows.append(["All"] + [ci(f"rho_{t}" + (f"_{s}" if s else "")) for t, s in cols])
emit(
    "tab_ladder",
    "The volume-translation ladder: AAD (\\%) in saturated-liquid density for the untranslated cubics and for the two "
    "low-temperature-anchored choices of the constant $c$, by reduced-temperature range (classical Soave $\\alpha$; "
    "re-anchoring on any other $\\alpha$-function changes the pooled rows by at most 0.13 points and the range rows by at most "
    "0.26 points, {stab:tabS_vt_alpha}). "
    "VT$_{corr}$ (Rackett-anchored) sets $c$ so that the translated liquid volume matches the Rackett--Yamada--Gunn "
    "correlation at $T_r = 0.70$ (no input beyond $T_c$, $p_c$, $\\omega$); VT (density-anchored) sets $c$ so that it "
    "matches one reference liquid density at $T_r = 0.70$. Bootstrap 95\\% confidence intervals are given for the pooled rows. "
    "The third conceivable constant, matching the reference critical volume, is not a usable model and is discussed "
    "in the text; all constants are tabulated per fluid in {stab:tabS_vt}.",
    [["", "SRK", "", "", "PR", "", ""],
     ["$T_r$ range", "none", "VT$_{corr}$", "VT", "none", "VT$_{corr}$", "VT"]],
    rows, colspec="lcccccc",
)

# ====================================================================== #
# Table: saturated vapor density by Tr range
rcols = ["srk_s", "pr_s", "srk_t95", "pr_t95", "srk_tc", "pr_tc"]
rows = []
for k, lab in BIN_ROWS:
    rows.append([lab] + [f2(key["by_bin"][k][f"rhov_{t}"]) for t in rcols]
                + [f2(key["by_bin"][k]["rhov_srk_s_vt"]), f2(key["by_bin"][k]["rhov_pr_s_vt"])])
rows.append(["All"] + [f2(key["overall"][f"rhov_{t}"]) for t in rcols]
            + [f2(key["overall"]["rhov_srk_s_vt"]), f2(key["overall"]["rhov_pr_s_vt"])])
emit(
    "tab_rhov",
    "AAD (\\%) in the saturated-vapor density by reduced-temperature range for the classical, Twu-95 and Twu-c "
    "$\\alpha$-functions on both cubics and for the density-anchored translated cubics (SRK-VT, PR-VT; classical "
    "$\\alpha$). The remaining combinations lie in the same 1.7--2.7\\% band when pooled ({tab:tab_overall}); "
    "per-fluid values are given in {stab:tabS_perfluid_rhov_srk} and {stab:tabS_perfluid_rhov_pr}.",
    [["$T_r$ range", "SRK-Soave", "PR-Soave", "SRK-Twu-95", "PR-Twu-95", "SRK-Twu-c", "PR-Twu-c", "SRK-VT", "PR-VT"]],
    rows, colspec="lcccccccc",
)

# ====================================================================== #
# Table: key sign tests (main text) and full matrix (SI)
def st_row(desc, prop_label, k):
    t = st["sign_tests"][k]
    return [desc, prop_label, f"{t['wins']}/{t['n']}", pfmt(t["p_two_sided"])]


P, H, RL, RV = "$p^{sat}$", "$\\Delta h_{vap}$", "$\\rho'$", "$\\rho''$"
rows = [
    "Cubic form, at fixed $\\alpha$-function",
    st_row("PR vs. SRK (Soave)", P, "psat:pr_s_vs_srk_s"),
    st_row("PR vs. SRK (Soave-19)", P, "psat:pr_s19_vs_srk_s19"),
    st_row("PR vs. SRK (Twu-95)", P, "psat:pr_t95_vs_srk_t95"),
    st_row("PR vs. SRK (Twu-c)", P, "psat:pr_tc_vs_srk_tc"),
    st_row("PR vs. SRK (Coquelet)", P, "psat:pr_coq_vs_srk_coq"),
    st_row("PR vs. SRK (MS)", P, "psat:pr_ms_vs_srk_ms"),
    st_row("PR vs. SRK (Twu-fit$^{\\dagger}$)", P, "psat:pr_tf_vs_srk_tf"),
    st_row("PR vs. SRK (Soave)", H, "h:pr_s_vs_srk_s"),
    st_row("PR vs. SRK (Soave-19)", H, "h:pr_s19_vs_srk_s19"),
    st_row("PR vs. SRK (Twu-95)", H, "h:pr_t95_vs_srk_t95"),
    st_row("PR vs. SRK (Twu-c)", H, "h:pr_tc_vs_srk_tc"),
    st_row("PR vs. SRK (Coquelet)", H, "h:pr_coq_vs_srk_coq"),
    st_row("PR vs. SRK (MS)", H, "h:pr_ms_vs_srk_ms"),
    st_row("PR vs. SRK (Twu-fit$^{\\dagger}$)", H, "h:pr_tf_vs_srk_tf"),
    st_row("PR vs. SRK (Soave)", RL, "rho:pr_s_vs_srk_s"),
    st_row("PR-VT vs. SRK-VT", RL, "rho:pr_s_vt_vs_srk_s_vt"),
    st_row("PR vs. SRK (Soave)", RV, "rhov:pr_s_vs_srk_s"),
    "$\\alpha$-function, at fixed cubic",
    st_row("SRK-Soave-19 vs. SRK-Soave", P, "psat:srk_s19_vs_srk_s"),
    st_row("PR-Soave-19 vs. PR-Soave", P, "psat:pr_s19_vs_pr_s"),
    st_row("SRK-Twu-95 vs. SRK-Soave", P, "psat:srk_t95_vs_srk_s"),
    st_row("PR-Twu-95 vs. PR-Soave", P, "psat:pr_t95_vs_pr_s"),
    st_row("SRK-Twu-c vs. SRK-Twu-95", P, "psat:srk_tc_vs_srk_t95"),
    st_row("PR-Twu-c vs. PR-Twu-95", P, "psat:pr_tc_vs_pr_t95"),
    st_row("PR-Twu-c vs. PR-Soave", P, "psat:pr_tc_vs_pr_s"),
    st_row("SRK-MS vs. SRK-Twu-95", P, "psat:srk_ms_vs_srk_t95"),
    st_row("PR-MS vs. PR-Soave", P, "psat:pr_ms_vs_pr_s"),
    st_row("PR-MS vs. PR-Twu-95", P, "psat:pr_ms_vs_pr_t95"),
    st_row("SRK-Coquelet vs. SRK-Soave", P, "psat:srk_coq_vs_srk_s"),
    st_row("SRK-Coquelet vs. SRK-Twu-95", P, "psat:srk_coq_vs_srk_t95"),
    st_row("PR-Coquelet vs. PR-Soave", P, "psat:pr_coq_vs_pr_s"),
    st_row("SRK-Twu-fit$^{\\dagger}$ vs. SRK-Twu-95", P, "psat:srk_tf_vs_srk_t95"),
    st_row("SRK-Twu-fit$^{\\dagger}$ vs. SRK-MS", P, "psat:srk_tf_vs_srk_ms"),
    st_row("PR-Twu-fit$^{\\dagger}$ vs. PR-Soave", P, "psat:pr_tf_vs_pr_s"),
    st_row("PR-Twu-fit$^{\\dagger}$ vs. PR-MS", P, "psat:pr_tf_vs_pr_ms"),
    st_row("SRK-Twu-95 vs. SRK-Soave", H, "h:srk_t95_vs_srk_s"),
    st_row("PR-Twu-95 vs. PR-Soave", H, "h:pr_t95_vs_pr_s"),
    "Volume translation",
    st_row("PR-VT$_{corr}$ vs. PR", RL, "rho:pr_s_vtc_vs_pr_s"),
    st_row("SRK-VT$_{corr}$ vs. SRK", RL, "rho:srk_s_vtc_vs_srk_s"),
    st_row("PR-VT vs. PR-VT$_{corr}$", RL, "rho:pr_s_vt_vs_pr_s_vtc"),
    st_row("PR-VT vs. PR", RV, "rhov:pr_s_vt_vs_pr_s"),
]
emit(
    "tab_signtests",
    f"Paired per-fluid comparisons: number of the {key['n_fluids']} fluids for which the first-named model has the "
    "lower AAD (wins), with the nominal two-sided $p$-value of the exact binomial sign test (Section 2.5). No ties occurred. "
    "The complete set of pairwise comparisons is given in {stab:tabS_signtests_full}. $^{\\dagger}$Component-specific parameters.",
    [["Comparison", "Property", "Wins", "$p$"]],
    rows, colspec="llcc",
)

def cell(k):
    if k not in st["sign_tests"]:
        return "--"
    t = st["sign_tests"][k]
    return f"{t['wins']}/{t['n']} ({pfmt(t['p_two_sided'])})"


def alab(a):
    return ALPHA_LABEL[a] + ("$^{\\dagger}$" if a == "tf" else "")


def vlab(x):
    """Label of a possibly translated model tag such as pr_s_vt."""
    parts = x.split("_")
    base = CUBIC_LABEL[parts[0]]
    if len(parts) == 3:
        base += {"vt": "-VT", "vtc": "-VT$_{corr}$"}[parts[2]]
    return base


rows = ["PR vs. SRK at a fixed $\\alpha$-function"]
for a in ALPHAS:
    rows.append([alab(a)] + [cell(f"{prop}:pr_{a}_vs_srk_{a}") for prop in ("psat", "h", "rho", "rhov")])
for c in ("srk", "pr"):
    rows.append(f"$\\alpha$-function pairs, {CUBIC_LABEL[c]} (first-named vs. second-named)")
    for j, a in enumerate(ALPHAS):
        for b in ALPHAS[:j]:
            rows.append([f"{alab(a)} vs. {alab(b)}"]
                        + [cell(f"{prop}:{c}_{a}_vs_{c}_{b}") for prop in ("psat", "h", "rho", "rhov")])
rows.append("Volume translation (classical Soave $\\alpha$)")
for a, b in (("pr_s_vt", "srk_s_vt"), ("pr_s_vtc", "srk_s_vtc"), ("pr_s_vtc", "pr_s"), ("srk_s_vtc", "srk_s"),
             ("pr_s_vt", "pr_s_vtc"), ("srk_s_vt", "srk_s_vtc"), ("pr_s_vt", "pr_s"), ("srk_s_vt", "srk_s")):
    rows.append([f"{vlab(a)} vs. {vlab(b)}", "--", "--", cell(f"rho:{a}_vs_{b}"), cell(f"rhov:{a}_vs_{b}")])
emit(
    "tabS_signtests_full",
    f"Complete set of the {st['n_sign_tests']} exact two-sided binomial sign tests on per-fluid AADs: number of the "
    f"{key['n_fluids']} fluids for which the first-named model has the lower AAD, with the nominal two-sided $p$-value "
    "in parentheses, for the four properties (columns). No ties occurred. The translated variants share $p^{sat}$ and "
    "$\\Delta h_{vap}$ with the untranslated models, and the comparison of VT with VT$_{corr}$ was computed for the "
    "liquid density only. $^{\\dagger}$Component-specific parameters.",
    [["Comparison", "$p^{sat}$", "$\\Delta h_{vap}$", "$\\rho'$", "$\\rho''$"]], rows, colspec="lcccc",
)

# ====================================================================== #
# SI tables
# S1: alpha-function constants
emit(
    "tabS_alpha_constants",
    "Constants of the generalized $\\alpha$-functions. Soave form, Eq. {eq:soave}: $m(\\omega)$ correlations. "
    "Twu-95, Eq. {eq:twu95}: universal subcritical constants of $\\alpha^{(0)}$ and $\\alpha^{(1)}$ [cite:twu1995a,twu1995b]. "
    "Twu-c, Eq. {eq:twu91}: $N = 2$ and quadratic $L(\\omega)$, $M(\\omega)$ [cite:pina2018,pina2022]. "
    "Coquelet, Eq. {eq:coq}, and MS, Eq. {eq:ms}: generalized coefficients as compiled by Xiao and Yang "
    "[cite:xiao2025] from [cite:coquelet2004,mahmoodi2017].",
    [["$\\alpha$-function", "EoS", "Expression / constants"]],
    [
        ["Soave", "SRK", "$m = 0.480 + 1.574\\,\\omega - 0.176\\,\\omega^2$"],
        ["Soave", "PR", "$m = 0.37464 + 1.54226\\,\\omega - 0.26992\\,\\omega^2$"],
        ["Soave-19", "SRK", "$m = 0.4810 + 1.5963\\,\\omega - 0.2963\\,\\omega^2 + 0.1223\\,\\omega^3$"],
        ["Soave-19", "PR", "$m = 0.3919 + 1.4996\\,\\omega - 0.2721\\,\\omega^2 + 0.1063\\,\\omega^3$"],
        ["Twu-95", "SRK", "$\\alpha^{(0)}$: $L = 0.141599$, $M = 0.919422$, $N = 2.496441$; "
                          "$\\alpha^{(1)}$: $L = 0.500315$, $M = 0.799457$, $N = 3.291790$"],
        ["Twu-95", "PR", "$\\alpha^{(0)}$: $L = 0.125283$, $M = 0.911807$, $N = 1.948150$; "
                         "$\\alpha^{(1)}$: $L = 0.511614$, $M = 0.784054$, $N = 2.812520$"],
        ["Twu-c", "SRK", "$L = 0.1359 + 0.7535\\,\\omega + 0.0611\\,\\omega^2$; "
                         "$M = 0.8787 - 0.2063\\,\\omega + 0.1709\\,\\omega^2$; $N = 2$"],
        ["Twu-c", "PR", "$L = 0.0544 + 0.7536\\,\\omega + 0.0297\\,\\omega^2$; "
                        "$M = 0.8678 - 0.1785\\,\\omega + 0.1401\\,\\omega^2$; $N = 2$"],
        ["Coquelet", "SRK", "$C_1 = 0.53591 + 1.4492\\,\\omega - 0.13969\\,\\omega^2$; "
                            "$C_2 = -0.2741 + 0.59006\\,\\omega - 0.54412\\,\\omega^2$; "
                            "$C_3 = 0.54293 + 0.53258\\,\\omega - 1.4927\\,\\omega^2$"],
        ["Coquelet", "PR", "$C_1 = 0.40464 + 1.3361\\,\\omega - 0.07987\\,\\omega^2$; "
                           "$C_2 = -0.08139 + 0.96493\\,\\omega - 0.85454\\,\\omega^2$; "
                           "$C_3 = 0.31953 + 0.001007\\,\\omega - 0.88858\\,\\omega^2$"],
        ["MS", "SRK", "$C_1 = 0.47941 + 1.673\\,\\omega - 0.23356\\,\\omega^2$; "
                        "$C_2 = 0.53827 + 2.1529\\,\\omega - 0.7143\\,\\omega^2$; "
                        "$C_3 = 1.7482\\,\\exp\\{-[(\\omega - 1.456)/1.4542]^4\\}$"],
        ["MS", "PR", "$C_1 = 0.36818 + 1.4801\\,\\omega - 0.14407\\,\\omega^2$; "
                       "$C_2 = 0.19422 + 1.9061\\,\\omega - 0.46577\\,\\omega^2$; "
                       "$C_3 = 1.514\\,\\exp\\{-[(\\omega - 1.6645)/1.5474]^4\\}$"],
    ],
    colspec="llp{9cm}",
)

# S2: fluid set
rows = []
for fam in FAM_ROWS:
    rows.append(fam)
    for _, r in fl[fl.family == fam].sort_values("omega").iterrows():
        rows.append([r.fluid, r.subgroup if isinstance(r.subgroup, str) else "", f"{r.Tc_K:.2f}", f"{r.pc_MPa:.3f}", f"{r.omega:.4f}",
                     f"{r.Zc_ref:.4f}", "--" if not np.isfinite(r.chi_HalmStiel) else f"{r.chi_HalmStiel:.3f}",
                     f"{r.Tr_min:.3f}"])
emit(
    "tabS_fluids",
    "Fluid set. Critical constants, acentric factors and critical compressibility factors are taken from the "
    "reference equations of state via CoolProp; $\\chi$ is the Halm--Stiel polar factor evaluated from the "
    "reference vapor pressure at $T_r = 0.6$ (undefined for carbon dioxide, whose triple point lies above "
    "$0.6\\,T_c$); $T_{r,\\min}$ is the lower end of the temperature grid.",
    [["Fluid", "Subgroup", "$T_c$ (K)", "$p_c$ (MPa)", "$\\omega$", "$Z_c^{ref}$", "$\\chi$", "$T_{r,\\min}$"]],
    rows, colspec="llcccccc",
)

# S3: fitted Twu parameters used (transcribed subset)
rows = []
for fam in FAM_ROWS:
    rows.append(fam)
    for _, r in fl[fl.family == fam].sort_values("omega").iterrows():
        rows.append([r.fluid, f"{r.L_tf_srk:.4f}", f"{r.M_tf_srk:.4f}", f"{r.N_tf_srk:.4f}",
                     f"{r.L_tf_pr:.4f}", f"{r.M_tf_pr:.4f}", f"{r.N_tf_pr:.4f}"])
emit(
    "tabS_twufit",
    "Component-specific consistent Twu-91 parameters used for the fitted reference (Twu-fit), from the 1800-fluid "
    "compilation of Pi\\~na-Martinez et al. [cite:pina2022] (tc-RK and tc-PR parameter sets), reproduced here for the "
    "benchmark fluids so that the results can be regenerated without external databases.",
    [["", "SRK (tc-RK set)", "", "", "PR (tc-PR set)", "", ""],
     ["Fluid", "$L$", "$M$", "$N$", "$L$", "$M$", "$N$"]],
    rows, colspec="lcccccc",
)

# S4-S7: per-fluid AADs
def per_fluid_table(name, prop, caption, cols, hdr):
    rows = []
    for fam in FAM_ROWS:
        rows.append(fam)
        for _, r in fl[fl.family == fam].sort_values("omega").iterrows():
            pf = key["per_fluid"][r.fluid]
            rows.append([r.fluid, f"{r.omega:.3f}"] + [f2(pf[c]) for c in cols])
    emit(name, caption, hdr, rows, colspec="lc" + "c" * len(cols))


def hdr_cubic(c, extra=None):
    h1 = ["", ""] + [CUBIC_LABEL[c]] * len(ALPHAS) + ([f"{CUBIC_LABEL[c]}-VT"] if extra else [])
    h2 = ["Fluid", "$\\omega$"] + [ALPHA_LABEL[a] + ("$^{\\dagger}$" if a == "tf" else "") for a in ALPHAS] + ([""] if extra else [])
    return [h1, h2]


for c in ("srk", "pr"):
    C = CUBIC_LABEL[c]
    per_fluid_table(f"tabS_perfluid_psat_{c}", "psat",
                    f"Per-fluid AAD (\\%) in vapor pressure over the full temperature grid of each fluid, {C} with each "
                    "$\\alpha$-function. $^{\\dagger}$Component-specific parameters.",
                    [f"psat_{c}_{a}" for a in ALPHAS], hdr_cubic(c))
    per_fluid_table(f"tabS_perfluid_hvap_{c}", "h",
                    f"Per-fluid AAD (\\%) in the enthalpy of vaporization over the full temperature grid of each fluid, {C} "
                    "with each $\\alpha$-function. A constant volume translation leaves this property unchanged. "
                    "$^{\\dagger}$Component-specific parameters.",
                    [f"h_{c}_{a}" for a in ALPHAS], hdr_cubic(c))
per_fluid_table("tabS_perfluid_rho", "rho",
                "Per-fluid AAD (\\%) in saturated-liquid density for the untranslated cubics (classical Soave "
                "$\\alpha$; the $\\alpha$-function changes these values by less than 0.3 points for any fluid) and "
                "for the Rackett-anchored (VT$_{corr}$) and density-anchored (VT) translations.",
                ["rho_srk_s", "rho_srk_s_vtc", "rho_srk_s_vt", "rho_pr_s", "rho_pr_s_vtc", "rho_pr_s_vt"],
                [["", "", "SRK", "", "", "PR", "", ""],
                 ["Fluid", "$\\omega$", "none", "VT$_{corr}$", "VT", "none", "VT$_{corr}$", "VT"]])
for c in ("srk", "pr"):
    C = CUBIC_LABEL[c]
    per_fluid_table(f"tabS_perfluid_rhov_{c}", "rhov",
                    f"Per-fluid AAD (\\%) in saturated-vapor density over the full temperature grid of each fluid, {C} "
                    f"with each $\\alpha$-function and density-anchored translated {C} (classical $\\alpha$, last column). "
                    "$^{\\dagger}$Component-specific parameters.",
                    [f"rhov_{c}_{a}" for a in ALPHAS] + [f"rhov_{c}_s_vt"], hdr_cubic(c, extra=True))

# S8: translation constants
rows = []
for fam in FAM_ROWS:
    rows.append(fam)
    for _, r in fl[fl.family == fam].sort_values("omega").iterrows():
        rows.append([r.fluid, f"{r.Zc_ref:.4f}", f"{r.Tr_anchor:.3f}",
                     f"{r.c_fit_srk_s_cm3mol:.2f}", f"{r.c_corr_srk_s_cm3mol:.2f}", f"{r.c_zc_srk_cm3mol:.2f}",
                     f"{r.c_fit_pr_s_cm3mol:.2f}", f"{r.c_corr_pr_s_cm3mol:.2f}", f"{r.c_zc_pr_cm3mol:.2f}",
                     f"{r.c_lit_pr_cm3mol:.2f}"])
emit(
    "tabS_vt",
    "Per-fluid constants of the translated models: reference critical compressibility factor, anchoring reduced "
    "temperature and, in cm$^3$ mol$^{-1}$, the density-anchored constant $c_{fit}$, the Rackett-anchored constant $c_{corr}$ "
    "and the critical-matched constant $c_{Z_c} = (Z_c^{EoS} - Z_c^{ref})RT_c/p_c$ (diagnostic only) for SRK and PR "
    "(classical Soave $\\alpha$), and, for comparison, the tc-PR translation constant of Pi\\~na-Martinez et al. "
    "[cite:pina2022], $c_{lit}$, which is anchored to the saturated-liquid volume at $T_r = 0.8$.",
    [["", "", "", "SRK (cm$^3$ mol$^{-1}$)", "", "", "PR (cm$^3$ mol$^{-1}$)", "", "", ""],
     ["Fluid", "$Z_c^{ref}$", "$T_{r,anchor}$", "$c_{fit}$", "$c_{corr}$", "$c_{Z_c}$", "$c_{fit}$", "$c_{corr}$", "$c_{Z_c}$", "$c_{lit}$"]],
    rows, colspec="lccccccccc",
)

# S9: alpha sensitivity of the translated densities
rows = []
for a in ALPHAS:
    rows.append([ALPHA_LABEL[a] + ("$^{\\dagger}$" if a == "tf" else ""),
                 f2(key["overall"][f"rho_srk_{a}"]), f2(key["overall"][f"rho_srk_{a}_vtc"]),
                 f2(key["overall"][f"rho_srk_{a}_vt"]),
                 f2(key["overall"][f"rho_pr_{a}"]), f2(key["overall"][f"rho_pr_{a}_vtc"]),
                 f2(key["overall"][f"rho_pr_{a}_vt"])])
emit(
    "tabS_vt_alpha",
    "Sensitivity of the saturated-liquid-density AAD (\\%, all fluids and temperatures) to the $\\alpha$-function, "
    "for the untranslated cubics and for both translation constants re-anchored on each $\\alpha$-variant. "
    "$^{\\dagger}$Component-specific parameters.",
    [["", "SRK", "", "", "PR", "", ""],
     ["$\\alpha$-function", "none", "VT$_{corr}$", "VT", "none", "VT$_{corr}$", "VT"]],
    rows, colspec="lcccccc",
)

# S10: near-critical fine bins
fine = key["by_bin_fine"]
mods = ["srk_s", "pr_s", "srk_t95", "pr_t95", "srk_tc", "pr_tc"]
rows = []
for prop, lab in (("psat", "Vapor pressure"), ("h", "Enthalpy of vaporization"),
                  ("rho", "Saturated-liquid density"), ("rhov", "Saturated-vapor density")):
    rows.append(lab)
    for k, blab in (("0.90-0.95", "0.90–0.95"), ("0.95-0.985", "0.95–0.985")):
        rows.append([blab] + [f2(fine[k][f"{prop}_{t}"]) for t in mods] +
                    ([f2(fine[k][f"{prop}_pr_s_vt"])] if prop in ("rho", "rhov") else ["as PR-Soave"]))
emit(
    "tabS_nearcrit",
    "The near-critical range split in two: AAD (\\%) for $0.90 \\le T_r < 0.95$ and $0.95 \\le T_r \\le 0.985$. "
    "PR-VT is the density-anchored translated PR (classical $\\alpha$); it shares the vapor pressure and enthalpy of "
    "vaporization of PR-Soave.",
    [["Property / range", "SRK-Soave", "PR-Soave", "SRK-Twu-95", "PR-Twu-95", "SRK-Twu-c", "PR-Twu-c", "PR-VT"]],
    rows, colspec="lccccccc",
)

# S: critical-constant confound of the fitted reference
tfs = json.load(open(os.path.join(DATA, "twufit_source_constants.json")))
tfc = pd.read_csv(os.path.join(DATA, "twufit_source_constants.csv")).set_index("fluid")
rows = []
for fam in FAM_ROWS:
    rows.append(fam)
    for _, r in fl[fl.family == fam].sort_values("omega").iterrows():
        q = tfc.loc[r.fluid]
        rows.append([r.fluid, f"{q.dTc_pct:+.2f}", f"{q.dpc_pct:+.2f}",
                     f2(q.psat_srk_tf_ref), f2(q.psat_srk_tf_src), f2(q.psat_pr_tf_ref), f2(q.psat_pr_tf_src),
                     f2(q.h_srk_tf_ref), f2(q.h_srk_tf_src), f2(q.h_pr_tf_ref), f2(q.h_pr_tf_src)])
rows.append(None)
for lab, blk in (("All fluids (pooled)", tfs), ("H-bonding (pooled)", tfs["by_subgroup"]["H-bonding"]),
                 ("Polar aprotic (pooled)", tfs["by_subgroup"]["polar aprotic"])):
    rows.append([lab, "", ""] + [f2(blk[k]) for k in ("psat_srk_tf_ref", "psat_srk_tf_src", "psat_pr_tf_ref",
                                                        "psat_pr_tf_src", "h_srk_tf_ref", "h_srk_tf_src",
                                                        "h_pr_tf_ref", "h_pr_tf_src")])
emit(
    "tabS_twufit_src",
    "Critical-constant confound of the fitted reference. Relative difference (\\%) between the critical temperature "
    "and pressure listed in the source compilation [cite:pina2022] and those of the reference equations of state, and "
    "per-fluid AAD (\\%) in vapor pressure and enthalpy of vaporization of SRK-Twu-fit and PR-Twu-fit evaluated with "
    "the reference-EoS constants (ref, as in the main text) and with the constants of the source compilation (src), "
    "at the same temperatures and against the same reference values. The pooled rows are point-weighted means.",
    [["", "", "", "$p^{sat}$, SRK", "", "$p^{sat}$, PR", "", "$\\Delta h_{vap}$, SRK", "", "$\\Delta h_{vap}$, PR", ""],
     ["Fluid", "$\\Delta T_c$", "$\\Delta p_c$", "ref", "src", "ref", "src", "ref", "src", "ref", "src"]],
    rows, colspec="lcccccccccc",
)

with open(os.path.join(TABS, "tables.json"), "w") as f:
    json.dump(tables, f, indent=1)
print("wrote tables.json with", len(tables), "tables")
