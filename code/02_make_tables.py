"""Every table of the manuscript and of the Supplementary Materials, from data/*.csv, key_numbers.json,
stats.json and the outputs of scripts 06-11. Part A builds the tables of the original benchmark; Part B
adds the tables of the revision (tab_compare, tabS_signed_*, tabS_insample*, tabS_floorloss, tabS_pr78,
tabS_srcconst, tabS_sens, tabS_physics) and extends tab_models, tab_overall, tab_psat, tab_hvap,
tab_signtests, tabS_signtests_full, tabS_vt and tabS_nearcrit.

Outputs: tables/<name>.tex (booktabs LaTeX) and tables/tables.json (the same content as structured
data, read by manuscript/make_word.py).
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
    """Store a table (structured); the LaTeX files are written at the end from the final content."""
    ncol = len(rows[0]) if rows and isinstance(rows[0], list) else len(header[-1])
    tables[name] = dict(caption=caption, header=header, rows=rows, notes=notes or "",
                        colspec=colspec or ("l" + "c" * (ncol - 1)), label_tex=label_tex or name)
    print("built", name)


def write_tex(name, t):
    """booktabs LaTeX version of a stored table."""
    ncol = len(t["header"][-1])
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{t['caption']}}}",
             f"\\label{{tab:{t.get('label_tex') or name}}}", "\\small",
             f"\\begin{{tabular}}{{{t['colspec']}}}", "\\toprule"]
    for h in t["header"]:
        lines.append(" & ".join(h) + " \\\\")
    lines.append("\\midrule")
    for r in t["rows"]:
        if r is None:
            lines.append("\\midrule")
        elif isinstance(r, str):
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{r}}}}} \\\\")
        else:
            lines.append(" & ".join(str(x) for x in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if t.get("notes"):
        lines.append(f"\\par\\smallskip\\footnotesize {t['notes']}")
    lines += ["\\end{table}", ""]
    with open(os.path.join(TABS, name + ".tex"), "w") as f:
        f.write("\n".join(lines))


# ###################################################################### #
# Part A: tables of the original benchmark
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


# ###################################################################### #
# Part B: tables of the revision (inputs: outputs of scripts 06-11)
# ====================================================================== #
ALPHA_LABEL.update({"sfit": "Soave-is", "tfit": "Twu-is", "tunc": "Twu-is (unconstrained)"})
P, H, RL, RV = "$p^{sat}$", "$\\Delta h_{vap}$", "$\\rho'$", "$\\rho''$"
PROP_LABEL = {"psat": P, "h": H, "rho": RL, "rhov": RV}

wp2 = pd.read_csv(os.path.join(DATA, "paired_comparisons.csv")).set_index("key")
bias = pd.read_csv(os.path.join(DATA, "signed_deviations.csv"))
fam = json.load(open(os.path.join(DATA, "comparison_families.json")))
wp3c = pd.read_csv(os.path.join(DATA, "insample_comparisons.csv")).set_index("key")
holm_ext = pd.read_csv(os.path.join(DATA, "holm_extended.csv"), index_col=0)
pooled3 = pd.read_csv(os.path.join(DATA, "insample_pooled.csv"))
perfluid3 = pd.read_csv(os.path.join(DATA, "insample_per_fluid.csv")).set_index("fluid")
fits = pd.read_csv(os.path.join(DATA, "insample_fit.csv"))
pr78s = pd.read_csv(os.path.join(DATA, "pr78_summary.csv"))
pr78 = json.load(open(os.path.join(DATA, "pr78.json")))
shift = pd.read_csv(os.path.join(DATA, "volume_shift.csv")).set_index("fluid")
hv = pd.read_csv(os.path.join(DATA, "hvap_absolute.csv"))
tab4 = json.load(open(os.path.join(DATA, "compilation_constants.json")))
tabpf = pd.read_csv(os.path.join(DATA, "compilation_constants_per_fluid.csv")).set_index("fluid")


def pfmt(p):
    if p >= 0.10:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    if round(m) >= 10:
        m, e = 1.0, e + 1
    return f"{m:.0f} × 10{str(e).translate(SUP)}"


def f2(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def sf2(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if abs(x) < 0.5 * 10 ** (-nd):
        return f"{0.0:.{nd}f}"
    return f"{x:+.{nd}f}"


def label(tag):
    parts = tag.split("_")
    if len(parts) == 3:
        return CUBIC_LABEL[parts[0]] + {"vt": "-VT", "vtc": "-VT$_{corr}$"}[parts[2]]
    c, a = parts
    return f"{CUBIC_LABEL[c]}-{ALPHA_LABEL[a]}" + ("$^{\\dagger}$" if a == "tf" else "") + ("$^{\\ddagger}$" if a in ("sfit", "tfit") else "")


# emit(): the storing version defined in Part A


def pool3(tag, prop="psat"):
    return pooled3[(pooled3.model == tag) & (pooled3.prop == prop)].iloc[0]


def ci3(tag, prop):
    r = pool3(tag, prop)
    return f"{r['aad|All']:.2f} ({r['ci_lo|All']:.2f}--{r['ci_hi|All']:.2f})"


# ====================================================================== #
# 1. Comparative table (new Table 1, Introduction)
emit(
    "tab_compare",
    "Comparative assessments of cubic equations of state closest to the present benchmark, by the elements that define "
    "a benchmark: the cubics and $\\alpha$-functions compared, whether the $\\alpha$-functions are generalized (predictive) "
    "or component-fitted, the properties, the fluids and the reference standard, the treatment of volume translation, the "
    "statistics reported, and whether the $\\alpha$-function is exchanged across both cubics under one protocol and the "
    "code released. The large compilations provide breadth of chemistry; the present work provides attribution and "
    "statistical resolution on a narrower set.",
    [["Study", "Cubics; $\\alpha$-functions", "Properties", "Fluids; standard", "Translation", "Statistics", "$\\alpha$ exchanged; code"]],
    [
        ["Trebble and Bishnoi 1986 [cite:trebble1986]", "ten cubics; $\\alpha$-functions as published", "$p^{sat}$, saturated volumes",
         "polar and non-polar compounds; experimental data", "none", "average deviations", "no; no"],
        ["Young et al. 2016 [cite:young2016]", "PR; 20 $\\alpha$-functions, generalized and component-fitted", "$p^{sat}$",
         "pure compounds of several classes; experimental data", "none", "average deviations", "no (PR only); no"],
        ["Le Guennec et al. 2016 [cite:leguennec2016b]", "tc-PR, tc-RK; consistent Twu-91, component-fitted",
         "$p^{sat}$, $\\Delta h_{vap}$, $c_p^{liq}$, $v^{liq}$", "pure compounds; DIPPR pseudo-experimental data",
         "one constant per fluid ($T_r = 0.8$)", "average deviations", "no; parameters tabulated"],
        ["Bell et al. 2018 [cite:bell2018]", "PR; consistent Twu-91, fitted per fluid", "$p^{sat}$, $\\Delta h_{vap}$",
         "> 2500 fluids; critically evaluated experimental data (NIST TDE)", "none", "average deviations", "no; parameters released"],
        ["Pi\\~na-Martinez et al. 2019 [cite:pina2019]", "RK, PR; updated generalized Soave $m(\\omega)$",
         "$p^{sat}$, $\\Delta h_{vap}$, $c_p^{liq}$", "1721 compounds; DIPPR pseudo-experimental data", "none",
         "average deviations", "same database for both cubics; no"],
        ["Pi\\~na-Martinez et al. 2022 [cite:pina2022]", "SRK, PR (classical Soave); tc-RK, tc-PR (consistent Twu-91, component-fitted)",
         "$p^{sat}$, $\\Delta h_{vap}$, $c_p^{liq}$, $v^{liq}$", "1800 fluids, $\\approx$300,000 points; DIPPR pseudo-experimental data",
         "one constant per fluid ($T_r = 0.8$; tc models)", "MAPE, pooled and per compound; self-/non-self-associating split", "no; parameters tabulated"],
        ["Mohamed et al. 2024 [cite:mohamed2024]", "PR, RK; Soave (generalized) and Twu-91 (fitted per component, consistency constraints)",
         "$p^{sat}$, $\\Delta h_{vap}$, $c_p^{liq}$ (fit); $\\rho'$, speed of sound (test set)",
         "147 components grouped by interaction type; NIST and compiled data", "in a test set only",
         "average deviations; consistency", "Soave and Twu on both; no"],
        ["Xiao and Yang 2025 [cite:xiao2025]", "PR, SRK, PTV, YFR; Soave, Twu, Coquelet, MS, Heyen", "viscosity by residual-entropy scaling",
         "124 fluids; REFPROP 10", "none", "AAD and bias", "Coquelet and MS on PR and SRK; no"],
        ["This work", "SRK, PR; six generalized $\\alpha$-functions, one out-of-sample fitted reference, in-sample fits of two forms",
         "$p^{sat}$, $\\Delta h_{vap}$, $\\rho'$, $\\rho''$", "37 fluids, 1480 states; reference Helmholtz-energy EoS; compilation critical constants as a second input set",
         "Rackett- and density-anchored constants, critical-matched diagnostic; anchor sweep; $Z_c$ dependence",
         "fluid-level bootstrap CIs; paired tests with Holm adjustment; paired-difference CIs; signed deviations",
         "yes, one protocol; open pipeline (MIT, Zenodo)"],
    ],
    colspec="p{2.3cm}p{2.9cm}p{2.0cm}p{2.6cm}p{2.0cm}p{2.3cm}p{2.1cm}",
)

# ====================================================================== #
# 2. Table of models: explicit alpha(Tr) and the PR76/PR78 statement
tm = tables["tab_models"]
tm["header"] = [["Label", "Form, $\\alpha(T_r)$", "Parameterization", "Fitted to", "Ref."]]
tm["rows"][0][1] = "Soave, Eq. {eq:soave}: $[1 + m(1 - \\sqrt{T_r})]^2$"
tm["rows"][0][2] = "original $m(\\omega)$ of each cubic (SRK 1972; PR 1976, i.e., PR76; the 1978 modification PR78 is reported in {stab:tabS_pr78})"
tm["rows"][1][1] = "Soave, Eq. {eq:soave}"
tm["rows"][2][1] = "Twu, Eq. {eq:twu95}: $\\alpha^{(0)} + \\omega(\\alpha^{(1)} - \\alpha^{(0)})$"
tm["rows"][3][1] = "Twu-91, Eq. {eq:twu91}: $T_r^{N(M-1)} \\exp[L(1 - T_r^{NM})]$"
tm["rows"][4][1] = "exponential, Eq. {eq:coq}"
tm["rows"][5][1] = "exponential, consistent, Eq. {eq:ms}"
tm["rows"][6][1] = "Twu-91, Eq. {eq:twu91}"
tm["rows"].append(["Soave-is$^{\\ddagger}$ / Twu-is$^{\\ddagger}$", "Eqs. {eq:soave} / {eq:twu91}", "$m$, or consistent $(L, M, N)$, fitted per fluid and cubic to the reference $p^{sat}$ of this benchmark (Section 2.5)",
                   "in-sample; the paper's own 40-point grid", "this work"])
tm["colspec"] = "p{2.2cm}p{4.2cm}p{4.6cm}p{3.0cm}p{1.4cm}"
tm["caption"] = ("The $\\alpha$-functions compared, with their functional form, parameterization, fitting database and source. The "
                 "first six are generalized (input: $T_c$, $p_c$, $\\omega$ only); the seventh uses component-specific parameters "
                 "and serves only as an out-of-sample fitted reference; the last row denotes the in-sample fits of Section 2.5, which "
                 "are not models but the per-fluid optimum of each form on this benchmark. The constants of all generalized functions "
                 "are listed in {stab:tabS_alpha_constants}.")

# ====================================================================== #
# 3. Table 2 (overall) and Tables psat / hvap by range: in-sample rows
to = tables["tab_overall"]
rows = to["rows"]
ins_rows = []
for c in ("srk", "pr"):
    for a in ("sfit", "tfit"):
        t = f"{c}_{a}"
        ins_rows.append([label(t), ci3(t, "psat"), ci3(t, "h"), ci3(t, "rho"), ci3(t, "rhov")])
# insert after the 14 model rows (before the None rule of the translated block)
i_rule = rows.index(None)
to["rows"] = rows[:i_rule] + [None] + ins_rows + rows[i_rule:]
to["caption"] = to["caption"].replace(
    "$^{\\dagger}$Component-specific parameters; fitted reference, not a predictive model.",
    "$^{\\dagger}$Component-specific parameters of [cite:pina2022]; out-of-sample fitted reference, not a predictive model. "
    "$^{\\ddagger}$In-sample fits (Section 2.5): the per-fluid optimum of the one-parameter Soave form (Soave-is) and of the "
    "three-parameter consistent Twu-91 form (Twu-is) on the reference vapor pressures of this benchmark; not models, but the "
    "lowest vapor-pressure AAD each cubic reaches with that form (for Twu-is, the floor of Section 2.5).")

BIN_ROWS = [("0.50-0.70", "0.50–0.70"), ("0.70-0.90", "0.70–0.90"), ("0.90-0.985", "0.90–0.985")]
FAM_ROWS = FAMILY_ORDER
SUB_ROWS = [("H-bonding", "H-bonding (4)"), ("polar aprotic", "polar aprotic (6)")]
for name, prop in (("tab_psat", "psat"), ("tab_hvap", "h")):
    t = tables[name]
    new_rows = []
    srk_rows = [r for r in t["rows"]]
    # rebuild: SRK block (7 + rule + 2 in-sample), rule, PR block
    out = []
    block = {"srk": [], "pr": []}
    for r in t["rows"]:
        if isinstance(r, list):
            block["srk" if r[0].startswith("SRK") else "pr"].append(r)
    for c in ("srk", "pr"):
        rs = block[c]
        out += rs[:6] + [None] + rs[6:]
        for a in ("sfit", "tfit"):
            q = pool3(f"{c}_{a}", prop)
            out.append([label(f"{c}_{a}")] + [f2(q[f"aad|{k}"]) for k, _ in BIN_ROWS] + [f2(q["aad|< 0.90"]), f2(q["aad|All"])]
                       + [f2(q[f"aad|{fm}"]) for fm in FAM_ROWS] + [f2(q[f"aad|{k}"]) for k, _ in SUB_ROWS])
        if c == "srk":
            out.append(None)
    t["rows"] = out
    t["caption"] = t["caption"].replace("(\\textdagger, component-specific parameters, fitted reference)", "")
    if "$^{\\ddagger}$" not in t["caption"]:
        t["caption"] += (" $^{\\ddagger}$In-sample fits (Section 2.5): the lowest vapor-pressure AAD each cubic reaches with the form "
                         "(for Twu-is, the floor)." if name == "tab_psat" else
                         " $^{\\ddagger}$In-sample fits (Section 2.5): $\\alpha$-functions fitted to the vapor pressure; $\\Delta h_{vap}$ "
                         "follows without further fitting.")

# ====================================================================== #
# 4. Table 3 (sign tests): paired-difference interval and Holm-adjusted p; in-sample rows; the two text-only exchanges
def row_from(key_, desc, prop_label, src, holm_src=None):
    r = src.loc[key_]
    ph = holm_src.loc[key_, "p_holm_ext"] if (holm_src is not None and key_ in holm_src.index) else np.nan
    return [desc, prop_label, f"{r.dAAD:+.2f} ({r.ci_lo:+.2f} to {r.ci_hi:+.2f})", f"{int(r.wins)}/{int(r.n)}", pfmt(r.p),
            pfmt(ph) if np.isfinite(ph) else "--"]


rows = ["Cubic form, at fixed $\\alpha$-function (family F1)"]
for a in ALPHAS:
    rows.append(row_from(f"psat:pr_{a}_vs_srk_{a}", f"PR vs. SRK ({ALPHA_LABEL[a]}{'$^{\\dagger}$' if a == 'tf' else ''})", P, wp2, holm_ext))
for a in ("sfit", "tfit"):
    rows.append(row_from(f"psat:pr_{a}_vs_srk_{a}", f"PR vs. SRK ({ALPHA_LABEL[a]}$^{{\\ddagger}}$)", P, wp3c, holm_ext))
for a in ALPHAS:
    rows.append(row_from(f"h:pr_{a}_vs_srk_{a}", f"PR vs. SRK ({ALPHA_LABEL[a]}{'$^{\\dagger}$' if a == 'tf' else ''})", H, wp2, holm_ext))
for a in ("sfit", "tfit"):
    rows.append(row_from(f"h:pr_{a}_vs_srk_{a}", f"PR vs. SRK ({ALPHA_LABEL[a]}$^{{\\ddagger}}$)", H, wp3c, holm_ext))
rows.append(row_from("rho:pr_s_vs_srk_s", "PR vs. SRK (Soave)", RL, wp2, holm_ext))
rows.append(row_from("rho:pr_s_vt_vs_srk_s_vt", "PR-VT vs. SRK-VT", RL, wp2, holm_ext))
rows.append(row_from("rhov:pr_s_vs_srk_s", "PR vs. SRK (Soave)", RV, wp2))
rows.append("$\\alpha$-function, at fixed cubic (family F2)")
F2_DESC = [("psat:srk_s19_vs_srk_s", "SRK-Soave-19 vs. SRK-Soave"), ("psat:pr_s19_vs_pr_s", "PR-Soave-19 vs. PR-Soave"),
           ("psat:srk_t95_vs_srk_s", "SRK-Twu-95 vs. SRK-Soave"), ("psat:pr_t95_vs_pr_s", "PR-Twu-95 vs. PR-Soave"),
           ("psat:srk_tc_vs_srk_t95", "SRK-Twu-c vs. SRK-Twu-95"), ("psat:pr_tc_vs_pr_t95", "PR-Twu-c vs. PR-Twu-95"),
           ("psat:pr_tc_vs_pr_s", "PR-Twu-c vs. PR-Soave"), ("psat:srk_ms_vs_srk_t95", "SRK-MS vs. SRK-Twu-95"),
           ("psat:srk_ms_vs_srk_s", "SRK-MS vs. SRK-Soave"), ("psat:pr_ms_vs_pr_s", "PR-MS vs. PR-Soave"),
           ("psat:pr_ms_vs_pr_t95", "PR-MS vs. PR-Twu-95"), ("psat:srk_coq_vs_srk_s", "SRK-Coquelet vs. SRK-Soave"),
           ("psat:srk_coq_vs_srk_t95", "SRK-Coquelet vs. SRK-Twu-95"), ("psat:pr_coq_vs_pr_s", "PR-Coquelet vs. PR-Soave"),
           ("psat:srk_tf_vs_srk_s", "SRK-Twu-fit$^{\\dagger}$ vs. SRK-Soave"), ("psat:srk_tf_vs_srk_t95", "SRK-Twu-fit$^{\\dagger}$ vs. SRK-Twu-95"),
           ("psat:srk_tf_vs_srk_ms", "SRK-Twu-fit$^{\\dagger}$ vs. SRK-MS"), ("psat:pr_tf_vs_pr_s", "PR-Twu-fit$^{\\dagger}$ vs. PR-Soave"),
           ("psat:pr_tf_vs_pr_ms", "PR-Twu-fit$^{\\dagger}$ vs. PR-MS")]
for k, d in F2_DESC:
    rows.append(row_from(k, d, P, wp2, holm_ext))
for c in ("srk", "pr"):
    C = CUBIC_LABEL[c]
    rows.append(row_from(f"psat:{c}_sfit_vs_{c}_s", f"{C}-Soave-is$^{{\\ddagger}}$ vs. {C}-Soave", P, wp3c, holm_ext))
    rows.append(row_from(f"psat:{c}_tfit_vs_{c}_sfit", f"{C}-Twu-is$^{{\\ddagger}}$ vs. {C}-Soave-is$^{{\\ddagger}}$", P, wp3c, holm_ext))
    rows.append(row_from(f"psat:{c}_tfit_vs_{c}_tf", f"{C}-Twu-is$^{{\\ddagger}}$ vs. {C}-Twu-fit$^{{\\dagger}}$", P, wp3c, holm_ext))
rows.append(row_from("h:srk_t95_vs_srk_s", "SRK-Twu-95 vs. SRK-Soave", H, wp2, holm_ext))
rows.append(row_from("h:pr_t95_vs_pr_s", "PR-Twu-95 vs. PR-Soave", H, wp2, holm_ext))
rows.append("Volume translation (family F3)")
for k, d, pl in (("rho:pr_s_vtc_vs_pr_s", "PR-VT$_{corr}$ vs. PR", RL), ("rho:srk_s_vtc_vs_srk_s", "SRK-VT$_{corr}$ vs. SRK", RL),
                 ("rho:pr_s_vt_vs_pr_s_vtc", "PR-VT vs. PR-VT$_{corr}$", RL), ("rhov:pr_s_vt_vs_pr_s", "PR-VT vs. PR", RV)):
    r = wp2.loc[k]
    rows.append([d, pl, f"{r.dAAD:+.2f} ({r.ci_lo:+.2f} to {r.ci_hi:+.2f})", f"{int(r.wins)}/{int(r.n)}", pfmt(r.p), pfmt(r.p_holm_family)])
emit(
    "tab_signtests",
    "Paired per-fluid comparisons: pooled difference in AAD between the first- and the second-named model (percentage points; "
    "negative = first-named model more accurate) with its paired fluid-level bootstrap 95\\% confidence interval, the number of "
    "the 37 fluids for which the first-named model has the lower AAD (wins), the nominal two-sided $p$-value of the exact binomial "
    "sign test, and the $p$-value adjusted by the Holm procedure within its family (F1: cubic form at fixed "
    "$\\alpha$-function, 20 tests; F2: $\\alpha$-function exchanges discussed in the text, 27 tests; F3: volume translation, 4 "
    "tests; Section 2.5). No ties occurred. The complete set of comparisons, with the Benjamini--Yekutieli adjustment over all "
    "242 of them, is given in {stab:tabS_signtests_full}. $^{\\dagger}$Component-specific parameters; $^{\\ddagger}$in-sample fit.",
    [["Comparison", "Property", "$\\Delta$AAD (95\\% CI)", "Wins", "$p$", "$p_{Holm}$"]],
    rows, colspec="llcccc",
)

# ====================================================================== #
# 5. Table S14 (full set): long format with paired intervals and BY q
rows = []
PROPS = ["psat", "h", "rho", "rhov"]


by_all = pd.read_csv(os.path.join(DATA, "by_all_comparisons.csv"), index_col=0)


def long_row(k, r, holm_col):
    prop, rest = k.split(":")
    a, b = rest.split("_vs_")
    fam_ = r.get("family", "descriptive")
    return [f"{label(a)} vs. {label(b)}", PROP_LABEL[prop], f"{r.dAAD:+.2f}", f"{r.ci_lo:+.2f} to {r.ci_hi:+.2f}",
            f"{int(r.wins)}/{int(r.n)}", pfmt(r.p), (fam_ if fam_ != "descriptive" else "--"),
            pfmt(holm_col) if np.isfinite(holm_col) else "--", pfmt(by_all.loc[k, "q_BY_all"])]


for prop in PROPS:
    rows.append(f"{PROP_LABEL[prop]}: PR vs. SRK at fixed $\\alpha$-function")
    for a in ALPHAS:
        k = f"{prop}:pr_{a}_vs_srk_{a}"
        rows.append(long_row(k, wp2.loc[k], holm_ext.loc[k, "p_holm_ext"] if k in holm_ext.index else np.nan))
    for c in ("srk", "pr"):
        rows.append(f"{PROP_LABEL[prop]}: $\\alpha$-function pairs, {CUBIC_LABEL[c]}")
        for j, a in enumerate(ALPHAS):
            for b in ALPHAS[:j]:
                k = f"{prop}:{c}_{a}_vs_{c}_{b}"
                rows.append(long_row(k, wp2.loc[k], holm_ext.loc[k, "p_holm_ext"] if k in holm_ext.index else np.nan))
rows.append("Volume translation (classical Soave $\\alpha$)")
for k in ["rho:pr_s_vt_vs_srk_s_vt", "rho:pr_s_vtc_vs_srk_s_vtc", "rho:pr_s_vtc_vs_pr_s", "rho:srk_s_vtc_vs_srk_s",
          "rho:pr_s_vt_vs_pr_s_vtc", "rho:srk_s_vt_vs_srk_s_vtc", "rhov:pr_s_vt_vs_pr_s", "rhov:srk_s_vt_vs_srk_s",
          "rhov:pr_s_vtc_vs_pr_s", "rhov:srk_s_vtc_vs_srk_s"]:
    r = wp2.loc[k]
    hc = holm_ext.loc[k, "p_holm_ext"] if k in holm_ext.index else (r.p_holm_family if r.family != "descriptive" else np.nan)
    rows.append(long_row(k, r, hc))
rows.append("In-sample fits (Section 2.5; $^{\\ddagger}$)")
for k, r in wp3c.iterrows():
    hc = holm_ext.loc[k, "p_holm_ext"] if k in holm_ext.index else np.nan
    rows.append(long_row(k, r, hc))
n_total = len([r for r in rows if isinstance(r, list)])
emit(
    "tabS_signtests_full",
    f"Complete set of the {n_total} paired per-fluid comparisons: pooled difference in AAD (percentage points; negative = "
    "first-named model more accurate) with its paired fluid-level bootstrap 95\\% confidence interval, number of the 37 fluids "
    "for which the first-named model has the lower AAD, nominal two-sided $p$-value of the exact binomial sign test, "
    f"family (F1, F2, F3; -- = exploratory), Holm-adjusted $p$ within the family, and Benjamini--Yekutieli "
    f"adjusted $q$ over all {n_total} comparisons (valid under arbitrary dependence). No ties occurred. "
    "The translated variants share $p^{sat}$ and $\\Delta h_{vap}$ with the untranslated models. $^{\\dagger}$Component-specific "
    "parameters; $^{\\ddagger}$in-sample fit; 'unconstrained' = Twu-91 fitted without the consistency constraints (diagnostic only).",
    [["Comparison", "Property", "$\\Delta$AAD", "95\\% CI", "Wins", "$p$", "Family", "$p_{Holm}$", "$q_{BY}$"]],
    rows, colspec="llccccccc",
)

# ====================================================================== #
# 6. Table S11 (translation constants): dimensionless shift columns
tv = tables["tabS_vt"]
tv["header"] = [["", "", "", "SRK (cm$^3$ mol$^{-1}$)", "", "", "", "PR (cm$^3$ mol$^{-1}$)", "", "", "", ""],
                ["Fluid", "$Z_c^{ref}$", "$T_{r,anchor}$", "$c_{fit}$", "$c_{corr}$", "$c_{Z_c}$", "$s_{fit}$", "$c_{fit}$", "$c_{corr}$", "$c_{Z_c}$", "$s_{fit}$", "$c_{lit}$"]]
new_rows = []
for r in tv["rows"]:
    if isinstance(r, list):
        q = shift.loc[r[0]]
        new_rows.append(r[:6] + [f"{q.s_fit_srk:.3f}"] + r[6:9] + [f"{q.s_fit_pr:.3f}"] + r[9:])
    else:
        new_rows.append(r)
tv["rows"] = new_rows
tv["colspec"] = "l" + "c" * 11
tv["caption"] = tv["caption"].replace(
    "for SRK and PR (classical Soave $\\alpha$), and, for comparison,",
    "for SRK and PR (classical Soave $\\alpha$), the dimensionless shift $s_{fit} = c_{fit}/b$ with $b = \\Omega_b R T_c/p_c$ "
    "(the form used in PVT packages), and, for comparison,")

# ====================================================================== #
# 7. Table S13 (near-critical): absolute enthalpy-of-vaporization deviations
tn = tables["tabS_nearcrit"]
mods = ["srk_s", "pr_s", "srk_t95", "pr_t95", "srk_tc", "pr_tc"]


def hv_val(tag, rg, col):
    return hv[(hv.model == tag) & (hv.range == rg)].iloc[0][col]


tn["rows"].append("Enthalpy of vaporization, mean absolute deviation (kJ mol$^{-1}$)")
for rg, lab in (("< 0.90", "< 0.90"), ("0.90-0.95", "0.90–0.95"), ("0.95-0.985", "0.95–0.985")):
    tn["rows"].append([lab] + [f"{hv_val(t, rg, 'mean_abs_dev_kJmol'):.2f}" for t in mods] + ["as PR-Soave"])
tn["rows"].append("Enthalpy of vaporization, mean signed deviation (kJ mol$^{-1}$)")
for rg, lab in (("< 0.90", "< 0.90"), ("0.90-0.95", "0.90–0.95"), ("0.95-0.985", "0.95–0.985")):
    tn["rows"].append([lab] + [f"{hv_val(t, rg, 'mean_signed_dev_kJmol'):+.2f}" for t in mods] + ["as PR-Soave"])
tn["rows"].append("Enthalpy of vaporization, absolute deviation relative to the fluid's $\\Delta h_{vap}$ at $T_r = 0.70$ (\\%)")
for rg, lab in (("< 0.90", "< 0.90"), ("0.90-0.95", "0.90–0.95"), ("0.95-0.985", "0.95–0.985")):
    tn["rows"].append([lab] + [f"{hv_val(t, rg, 'abs_dev_rel_h07_pct'):.2f}" for t in mods] + ["as PR-Soave"])
tn["caption"] = ("The near-critical range split in two: AAD (\\%) for $0.90 \\le T_r < 0.95$ and $0.95 \\le T_r \\le 0.985$, and, "
                 "for the enthalpy of vaporization, the absolute deviation (mean $|\\Delta h_{vap}^{EoS} - \\Delta h_{vap}^{ref}|$ in "
                 "kJ mol$^{-1}$), its mean signed value and the absolute deviation divided by each fluid's reference $\\Delta h_{vap}$ "
                 "at $T_r = 0.70$ (the grid point nearest the anchoring temperature; pooled over fluids), for $T_r < 0.90$ and the two "
                 "near-critical ranges; the mean reference $\\Delta h_{vap}$ is "
                 f"{hv_val('pr_s', '< 0.90', 'mean_ref_hvap_kJmol'):.1f}, {hv_val('pr_s', '0.90-0.95', 'mean_ref_hvap_kJmol'):.1f} and "
                 f"{hv_val('pr_s', '0.95-0.985', 'mean_ref_hvap_kJmol'):.1f} kJ mol$^{{-1}}$ in the three ranges. "
                 "PR-VT is the density-anchored translated PR (classical $\\alpha$); it shares the vapor pressure and enthalpy of "
                 "vaporization of PR-Soave.")

# ====================================================================== #
# 8. New SI tables: signed deviations (S16a-d)
BCOLS = ["bias|0.50-0.70", "bias|0.70-0.90", "bias|0.90-0.985", "bias|< 0.90", "bias|All", "bias|Inorganic gases", "bias|Alkanes",
         "bias|Aromatics", "bias|Polar/associating", "bias|Refrigerants", "bias|H-bonding", "bias|polar aprotic"]
HDR1 = ["", "Reduced-temperature range", "", "", "", "", "Chemical family", "", "", "", "", "Subgroup", ""]
HDR2 = ["Model", "0.50--0.70", "0.70--0.90", "0.90--0.985", "< 0.90", "All (95\\% CI)", "Inorg. (6)", "Alk. (11)", "Arom. (6)", "Polar (10)",
        "Refr. (4)", "H-bond (4)", "Aprotic (6)"]


def bias_rows(prop, extra=()):
    b = bias[bias.prop == prop].set_index("model")
    rows = []
    for c in ("srk", "pr"):
        for a in ALPHAS:
            t = f"{c}_{a}"
            r = b.loc[t]
            vals = [sf2(r[k]) for k in BCOLS]
            vals[4] = f"{r['bias|All']:+.2f} ({r['bias_ci_lo|All']:+.2f} to {r['bias_ci_hi|All']:+.2f})"
            rows.append([label(t)] + vals)
        if c == "srk":
            rows.append(None)
    if extra:
        rows.append(None)
        for t in extra:
            r = b.loc[t]
            vals = [sf2(r[k]) for k in BCOLS]
            vals[4] = f"{r['bias|All']:+.2f} ({r['bias_ci_lo|All']:+.2f} to {r['bias_ci_hi|All']:+.2f})"
            rows.append([label(t)] + vals)
    return rows


for prop, nm, cap in (("psat", "tabS_signed_psat", "vapor pressure"), ("h", "tabS_signed_h", "enthalpy of vaporization"),
                      ("rho", "tabS_signed_rho", "saturated-liquid density (untranslated cubics, and the translated variants of the classical pair)"),
                      ("rhov", "tabS_signed_rhov", "saturated-vapor density (the same constants applied to the vapor volume)")):
    extra = ("srk_s_vtc", "srk_s_vt", "pr_s_vtc", "pr_s_vt") if prop in ("rho", "rhov") else ()
    emit(nm,
         f"Mean signed deviation (\\%; positive = overprediction) in {cap} by reduced-temperature range, chemical family and "
         "polar/associating subgroup, with the fluid-level bootstrap 95\\% confidence interval of the overall value. "
         "$^{\\dagger}$Component-specific parameters.",
         [HDR1, HDR2], bias_rows(prop, extra), colspec="l" + "c" * 12)

# ====================================================================== #
# 9. In-sample fits: per-fluid AADs (S17) and parameters (S18), floor + loss (S19)
rows = []
for fam_ in FAMILY_ORDER:
    rows.append(fam_)
    for _, r in fl[fl.family == fam_].sort_values("omega").iterrows():
        q = perfluid3.loc[r.fluid]
        f_ = {c: fits[(fits.fluid == r.fluid) & (fits.cubic == c)].iloc[0] for c in ("SRK", "PR")}
        rows.append([r.fluid, f2(q.psat_srk_s), f2(q.psat_srk_sfit), f2(q.psat_srk_tf), f2(q.psat_srk_tfit),
                     f2(f_["SRK"].aad_twu_unc), f2(q.h_srk_tfit), f2(q.psat_pr_s), f2(q.psat_pr_sfit), f2(q.psat_pr_tf), f2(q.psat_pr_tfit),
                     f2(f_["PR"].aad_twu_unc), f2(q.h_pr_tfit), f"{q.psat_pr_tfit - q.psat_srk_tfit:+.2f}"])
rows.append(None)
for lab, grp in (("All fluids (pooled)", "All"), ("Alkanes (pooled)", "Alkanes"), ("H-bonding (pooled)", "H-bonding")):
    rows.append([lab] + [f2(pool3(t, "psat")[f"aad|{grp}"]) for t in ("srk_s", "srk_sfit", "srk_tf", "srk_tfit", "srk_tunc")]
                + [f2(pool3("srk_tfit", "h")[f"aad|{grp}"])]
                + [f2(pool3(t, "psat")[f"aad|{grp}"]) for t in ("pr_s", "pr_sfit", "pr_tf", "pr_tfit", "pr_tunc")]
                + [f2(pool3("pr_tfit", "h")[f"aad|{grp}"]), f"{pool3('pr_tfit', 'psat')[f'aad|{grp}'] - pool3('srk_tfit', 'psat')[f'aad|{grp}']:+.2f}"])
emit(
    "tabS_insample",
    "In-sample $\\alpha$-function fits (Section 2.5): per-fluid AAD (\\%) in vapor pressure with the generalized Soave correlation "
    "(Soave), the Soave form with $m$ fitted per fluid (Soave-is), the out-of-sample component-specific Twu-91 parameters of "
    "[cite:pina2022] (Twu-fit$^{\\dagger}$), the consistent Twu-91 form with $(L, M, N)$ fitted per fluid under the consistency "
    "constraints (Twu-is), and, as a diagnostic only, the Twu-91 form fitted without the consistency constraints (unc.); AAD (\\%) "
    "in the enthalpy of vaporization implied by the in-sample Twu-91 fit; and the PR $-$ SRK difference at the Twu-is optimum "
    "(percentage points). Pooled rows are point-weighted means.",
    [["", "SRK: $p^{sat}$", "", "", "", "", "$\\Delta h_{vap}$", "PR: $p^{sat}$", "", "", "", "", "$\\Delta h_{vap}$", "PR $-$ SRK"],
     ["Fluid", "Soave", "Soave-is", "Twu-fit$^{\\dagger}$", "Twu-is", "unc.", "Twu-is", "Soave", "Soave-is", "Twu-fit$^{\\dagger}$", "Twu-is", "unc.", "Twu-is", "Twu-is"]],
    rows, colspec="l" + "c" * 13,
)

rows = []
for fam_ in FAMILY_ORDER:
    rows.append(fam_)
    for _, r in fl[fl.family == fam_].sort_values("omega").iterrows():
        a = {c: fits[(fits.fluid == r.fluid) & (fits.cubic == c)].iloc[0] for c in ("SRK", "PR")}

        def act(x):
            s = []
            if x.active_Q3:
                s.append("α‴")
            if x.active_P2:
                s.append("α″")
            if x.active_M1:
                s.append("M = 1")
            return ", ".join(s) if s else "--"
        rows.append([r.fluid, f"{a['SRK'].m_gen:.4f}", f"{a['SRK'].m_fit:.4f}", f"{a['SRK'].L_fit:.4f}", f"{a['SRK'].M_fit:.4f}", f"{a['SRK'].N_fit:.4f}", act(a["SRK"]),
                     f"{a['PR'].m_gen:.4f}", f"{a['PR'].m_fit:.4f}", f"{a['PR'].L_fit:.4f}", f"{a['PR'].M_fit:.4f}", f"{a['PR'].N_fit:.4f}", act(a["PR"])])
emit(
    "tabS_insample_params",
    "Parameters of the in-sample fits: the generalized and the fitted Soave slope parameter $m$, the fitted consistent Twu-91 "
    "parameters $(L, M, N)$ and the consistency condition active at the optimum (α‴: third-derivative condition; "
    "α″: second-derivative condition; M = 1: bound), for SRK and PR. The Twu-91 parameters are partly interchangeable "
    "over the grid (different $(L, M, N)$ give nearly the same $\\alpha(T_r)$), so the optimum AAD of {stab:tabS_insample} is "
    "better determined than the parameters themselves; the in-sample $\\alpha(T_r)$ differs from that of the published parameters "
    "[cite:pina2022] by less than 0.8\\% anywhere on the grid for every fluid.",
    [["", "SRK", "", "", "", "", "", "PR", "", "", "", "", ""],
     ["Fluid", "$m$ (gen.)", "$m$ (is)", "$L$", "$M$", "$N$", "active", "$m$ (gen.)", "$m$ (is)", "$L$", "$M$", "$N$", "active"]],
    rows, colspec="l" + "c" * 12,
)

fs, fp = pool3("srk_tfit")["aad|All"], pool3("pr_tfit")["aad|All"]
rows = [["Floor (Twu-is$^{\\ddagger}$)", f2(fs), "--", f2(fp), "--", f"{fs - fp:+.2f}", f"{fs - fp:+.2f}", "--"]]
for a in ALPHAS:
    s_, p_ = key["overall"][f"psat_srk_{a}"], key["overall"][f"psat_pr_{a}"]
    rows.append([ALPHA_LABEL[a] + ("$^{\\dagger}$" if a == "tf" else ""), f2(s_), f2(s_ - fs), f2(p_), f2(p_ - fp), f"{s_ - p_:+.2f}",
                 f"{fs - fp:+.2f}", f"{(s_ - fs) - (p_ - fp):+.2f}"])
s_, p_ = pool3("srk_sfit")["aad|All"], pool3("pr_sfit")["aad|All"]
rows.append(["Soave-is$^{\\ddagger}$", f2(s_), f2(s_ - fs), f2(p_), f2(p_ - fp), f"{s_ - p_:+.2f}", f"{fs - fp:+.2f}", f"{(s_ - fs) - (p_ - fp):+.2f}"])
emit(
    "tabS_floorloss",
    "The pooled vapor-pressure AAD (\\%) of every combination written as the floor of its cubic (the in-sample optimum of the "
    "consistent Twu-91 form, Twu-is) plus the parameterization loss of its $\\alpha$-function, and the SRK--PR gap (percentage "
    "points, SRK minus PR) as the difference of the floors plus the difference of the losses. The floor difference is the "
    "contribution of the cubic form at the flexibility of the consistent three-parameter form; the loss difference is what the "
    "$\\alpha$-function of the row gives up against that form on each cubic. For an $\\alpha$-function fitted to the same data by the same "
    "procedure on both cubics, the sum of the two is the contribution of the cubic form at that flexibility (fitted Soave form: "
    "$0.15 + 0.43 = 0.58$ points); for the published correlations the loss difference also contains their regression histories "
    "(Section 3.2.3 of the main text). $^{\\dagger}$Component-specific parameters; $^{\\ddagger}$in-sample fit.",
    [["", "SRK", "", "PR", "", "SRK $-$ PR gap", "", ""], ["$\\alpha$-function", "AAD", "loss", "AAD", "loss", "total", "floor part", "loss part"]],
    rows, colspec="lccccccc",
)

# ====================================================================== #
# 10. PR78 (S20)
RANGES = ["0.50-0.70", "0.70-0.90", "0.90-0.985", "< 0.90", "All"]
RLAB = {"0.50-0.70": "0.50--0.70", "0.70-0.90": "0.70--0.90", "0.90-0.985": "0.90--0.985", "< 0.90": "< 0.90", "All": "All"}
rows = []
mm = pr78["m"]
for f_ in ("methanol", "ethanol"):
    rows.append(f"{f_} ($\\omega = {mm[f_]['omega']:.3f}$; $m$ = {mm[f_]['m76']:.4f} (PR76), {mm[f_]['m78']:.4f} (PR78))")
    for rg in RANGES:
        r = pr78s[(pr78s.fluid == f_) & (pr78s.range == rg)].iloc[0]
        rows.append([RLAB[rg], f2(r.psat_PR76), f2(r.psat_PR78), f2(r.h_PR76), f2(r.h_PR78), f2(r.rhov_PR76), f2(r.rhov_PR78),
                     f2(r.rho_PR76), f2(r.rho_PR78), f2(r.rho_vt_PR76), f2(r.rho_vt_PR78)])
sm, se = pr78["spot"]["methanol"], pr78["spot"]["ethanol"]
emit(
    "tabS_pr78",
    "PR76 versus PR78: AAD (\\%) of PR with the 1976 correlation for $m(\\omega)$ (PR76, used throughout the main text) and with "
    "the 1978 modification [cite:robinson1978], $m = 0.379642 + 1.48503\\,\\omega - 0.164423\\,\\omega^2 + 0.016666\\,\\omega^3$ for "
    "$\\omega > 0.49$ (PR78), for the two fluids of the set above the switch, by reduced-temperature range; the density-anchored "
    "translation is re-anchored on the variant. The other 35 fluids, including n-decane ($\\omega = 0.488$), are unaffected. "
    f"At $T_r = 0.50$ the vapor-pressure deviation changes from {sm['Tr050_dev_p_PR76']:+.1f}\\% to {sm['Tr050_dev_p_PR78']:+.1f}\\% for "
    f"methanol and from {se['Tr050_dev_p_PR76']:+.1f}\\% to {se['Tr050_dev_p_PR78']:+.1f}\\% for ethanol.",
    [["", "$p^{sat}$", "", "$\\Delta h_{vap}$", "", "$\\rho''$", "", "$\\rho'$", "", "$\\rho'$, PR-VT", ""],
     ["$T_r$ range", "PR76", "PR78", "PR76", "PR78", "PR76", "PR78", "PR76", "PR78", "PR76", "PR78"]],
    rows, colspec="l" + "c" * 10,
)

# ====================================================================== #
# 11. Compilation constants (S21) and sensitivities (S22)
A = tab4["summary"]["All"]
rows = []
for c in ("srk", "pr"):
    for a in ALPHAS:
        t = f"{c}_{a}"
        ch = tab4["changes"][t]
        rows.append([label(t), f2(A[f"psat_{t}_ref"]), f2(A[f"psat_{t}_src"]), f2(A[f"h_{t}_ref"]), f2(A[f"h_{t}_src"]),
                     f2(A[f"rho_{t}_ref"]), f2(A[f"rho_{t}_src"]), f2(A[f"rhov_{t}_ref"]), f2(A[f"rhov_{t}_src"]),
                     f"{ch['max_increase']:+.2f} ({ch['fluid_max_increase']})", f"{ch['n_worse']}/{ch['n_better']}"])
    if c == "srk":
        rows.append(None)
emit(
    "tabS_srcconst",
    "Model-form error versus input error: pooled AAD (\\%) of the fourteen combinations evaluated with the critical constants of "
    "the reference equations of state (ref, as in the main text) and with the critical temperatures and pressures listed in the "
    "compilation of [cite:pina2022] (src; acentric factors unchanged, same absolute temperatures, same reference values; "
    f"differences between the two sets of constants up to {tabpf.dTc_pct.abs().max():.1f}\\% in $T_c$ and {tabpf.dpc_pct.abs().max():.1f}\\% in $p_c$, "
    "{stab:tabS_twufit_src}), with the largest per-fluid increase of the vapor-pressure AAD (percentage points, fluid in "
    "parentheses) and the number of fluids whose vapor-pressure AAD becomes worse/better. For the fitted reference the "
    "compilation constants are those its parameters were regressed with (Section 3.2.4); for the twelve generalized "
    "combinations they are an input error on top of the model-form error.",
    [["", "$p^{sat}$", "", "$\\Delta h_{vap}$", "", "$\\rho'$", "", "$\\rho''$", "", "Largest increase, $p^{sat}$", "Worse/better"],
     ["Model", "ref", "src", "ref", "src", "ref", "src", "ref", "src", "", ""]],
    rows, colspec="l" + "c" * 10,
)

S = tab4["sensitivities"]["PR"]


def srange(d):
    return f"{d['median']:.2f} ({d['min']:.2f} to {d['max']:.2f})"


rows = []
for tr in ("Tr=0.5", "Tr=0.7", "Tr=0.9", "Tr=0.985"):
    q = S[tr]
    rows.append([tr.replace("Tr=", "$T_r$ = "), str(q["dlnp_dTc"]["n"]), srange(q["dlnp_dTc"]), f"{q['dlnp_dpc']['median']:.2f}", srange(q["dlnp_dw"]),
                 srange(q["dlnh_dTc"]), f"{abs(q['dlnh_dpc']['median']):.2f}", srange(q["dlnh_dw"]), srange(q["dlnrho_dTc"]),
                 f"{q['dlnrho_dpc']['median']:.2f}", srange(q["dlnrho_dw"])])
emit(
    "tabS_sens",
    "Input sensitivities of the PR-Soave saturation properties at fixed absolute temperature: logarithmic derivatives with "
    "respect to $T_c$, $p_c$ and $\\omega$ (central differences, $\\pm 0.05\\%$ in $T_c$ and $p_c$, $\\pm 0.005$ in $\\omega$), "
    "evaluated for every fluid at the grid point nearest $T_r$ = 0.50, 0.70, 0.90 and 0.985; median over the $n$ fluids present at "
    "that $T_r$ with the range over the fluids in parentheses. The SRK-Soave vapor-pressure sensitivities agree within 3\\%. At fixed $T$, $T_c$ and "
    "$\\omega$ every two-parameter cubic gives $p^{sat}$, $\\rho'$ and $\\rho''$ exactly proportional to $p_c$ and $\\Delta h_{vap}$ "
    "independent of $p_c$; $\\partial \\ln p^{sat}/\\partial \\omega$ is close to $-\\ln 10$ at $T_r = 0.70$ because the $m(\\omega)$ "
    "correlations follow the definition of the acentric factor.",
    [["", "", "$p^{sat}$", "", "", "$\\Delta h_{vap}$", "", "", "$\\rho'$", "", ""],
     ["", "$n$", "$\\partial \\ln p/\\partial \\ln T_c$", "$\\partial \\ln p/\\partial \\ln p_c$", "$\\partial \\ln p/\\partial \\omega$",
      "$\\partial \\ln \\Delta h/\\partial \\ln T_c$", "$\\partial \\ln \\Delta h/\\partial \\ln p_c$", "$\\partial \\ln \\Delta h/\\partial \\omega$",
      "$\\partial \\ln \\rho'/\\partial \\ln T_c$", "$\\partial \\ln \\rho'/\\partial \\ln p_c$", "$\\partial \\ln \\rho'/\\partial \\omega$"]],
    rows, colspec="l" + "c" * 10,
)

# Physical diagnostics (tabS_physics): attraction-parameter sensitivities, alpha spread, near-critical exponents,\n# Zc dependence of the density error, Clapeyron split, n-decane example
phys = json.load(open(os.path.join(DATA, "physical_diagnostics.json")))
spread = pd.read_csv(os.path.join(DATA, "physics_alpha_spread.csv"))
expo_w = pd.read_csv(os.path.join(DATA, "physics_nearcrit_closure.csv"))
expo_h = pd.read_csv(os.path.join(DATA, "physics_hvap_exponent.csv"))
sens_a = pd.read_csv(os.path.join(DATA, "physics_alpha_sensitivity.csv"))


def mr(series, nd=2):
    return f"{series.median():.{nd}f} ({series.min():.{nd}f} to {series.max():.{nd}f})"


rows = ["Sensitivity to the attraction parameter at fixed $T$ and $b$ (classical $\\alpha$; median, range over the fluids)"]
for q, lab in (("dlnp_dlna", "$\\partial \\ln p^{sat}/\\partial \\ln a$"), ("dlnrhol_dlna", "$\\partial \\ln \\rho'/\\partial \\ln a$")):
    for trq in (0.5, 0.7, 0.9):
        g = {c: sens_a[(sens_a.cubic == c) & (sens_a.Tr == trq)][q] for c in ("SRK", "PR")}
        rows.append([f"{lab}, $T_r$ = {trq:.1f}", mr(g["SRK"]), mr(g["PR"]), "--"])
rows.append("Exchange of the generalized $\\alpha$-functions (five functions against the classical one, every grid point)")
for q, lab in (("max_dalpha_pct", "largest relative difference in $\\alpha$ (\\%)"),
               ("max_dpsat_pts", "largest change of the $p^{sat}$ deviation (points)"),
               ("max_drho_pts", "largest change of the $\\rho'$ deviation (points)")):
    rows.append([lab, f2(spread[spread.cubic == "SRK"][q].max()), f2(spread[spread.cubic == "PR"][q].max()), "--"])
rows.append("Closure of the coexistence curve, $0.93 \\le T_r \\le 0.985$ (classical $\\alpha$; median, range over the fluids)")
rows.append(["effective exponent of $\\rho' - \\rho''$", mr(expo_w.beta_srk, 3), mr(expo_w.beta_pr, 3), mr(expo_w.beta_ref, 3)])
rows.append(["effective exponent of $\\Delta h_{vap}$", mr(expo_h.beta_h_srk, 3), mr(expo_h.beta_h_pr, 3), mr(expo_h.beta_h_ref, 3)])
cl = phys["closure"]
rows.append(["fluids for which $(\\rho' - \\rho_c)/\\rho_c$ at $T_r = 0.985$ is smaller for the cubic, with its own $\\rho_c$, than for the reference",
             f"{cl['cubic_closer_srk']}/{cl['n_fluids']}", f"{cl['cubic_closer_pr']}/{cl['n_fluids']}", "--"])
rows.append(["mean signed deviation of $\\rho'$ at $T_r = 0.985$ (\\%; untranslated)", f"{cl['mean_dev_top_rho_srk']:.1f}", f"{cl['mean_dev_top_rho_pr']:.1f}", "--"])
rows.append(["mean signed deviation of $\\rho''$ at $T_r = 0.985$ (\\%)", f"{cl['mean_dev_top_rhov_srk']:.1f}", f"{cl['mean_dev_top_rhov_pr']:.1f}", "--"])
rows.append("Mean signed $\\rho'$ deviation below $T_r = 0.70$ against the reference $Z_c$ (untranslated, classical $\\alpha$; 36 fluids)")
zc_ = phys["zc_density"]
rows.append(["correlation coefficient $r$", f"{zc_['bias_lo_srk']['r']:.2f}", f"{zc_['bias_lo_pr']['r']:.2f}", "--"])
rows.append(["slope (points per 0.01 in $Z_c$)", f"{zc_['bias_lo_srk']['slope_pct_per_0p01Zc']:.1f}", f"{zc_['bias_lo_pr']['slope_pct_per_0p01Zc']:.1f}", "--"])
rows.append(["$Z_c$ at zero mean deviation", f"{zc_['bias_lo_srk']['Zc_at_zero']:.3f}", f"{zc_['bias_lo_pr']['Zc_at_zero']:.3f}", "--"])
zrd = phys["zc_reducing"]["regression"]
rows.append(["$r$ with the reducing-point $Z_c$ of methyl chloride and n-heptane", f"{zrd['bias_lo_srk']['r_red_two']:.2f}", f"{zrd['bias_lo_pr']['r_red_two']:.2f}", "--"])
hsp = phys["hvap_split"]
rows.append("Enthalpy of vaporization split by the Clapeyron equation (mean logarithmic deviation, \\%; range over the seven "
            "$\\alpha$-functions, in-sample Twu-91 in brackets)")
for rg, lab in (("lo", "$0.50 \\le T_r < 0.70$"), ("mid", "$0.70 \\le T_r < 0.90$"), ("nc", "$0.90 \\le T_r \\le 0.985$")):
    for q, qlab in (("h", "$\\Delta h_{vap}$"), ("dZ", "$\\Delta Z_{vap}$ part")) + ((("Zv", "$Z''$"),) if rg == "mid" else ()):
        rows.append([f"{lab}: {qlab}"] + [f"{hsp[c + '|' + rg][q][0]:.2f} to {hsp[c + '|' + rg][q][1]:.2f} [{hsp[c + '|' + rg]['tfit'][q]:.2f}]"
                                          for c in ("srk", "pr")] + ["--"])
dx = phys["decane_example"]
rows.append("Heavy-alkane example: n-decane at $T_r = 0.50$ (Twu-95 and MS against the classical $\\alpha$)")
rows.append(["change of $\\alpha$ (\\%), Twu-95 / MS"] + [f"{dx[c]['dalpha_pct_t95']:+.2f} / {dx[c]['dalpha_pct_ms']:+.2f}" for c in ("SRK", "PR")] + ["--"])
rows.append(["$\\partial \\ln p^{sat}/\\partial \\ln a$"] + [f"{dx[c]['dlnp_dlna']:.2f}" for c in ("SRK", "PR")] + ["--"])
rows.append(["resulting change of $p^{sat}$ (\\%), Twu-95 / MS"] + [f"{dx[c]['dpsat_pct_t95']:+.2f} / {dx[c]['dpsat_pct_ms']:+.2f}" for c in ("SRK", "PR")] + ["--"])
rows.append(["$p^{sat}$ deviation (\\%): classical; Twu-95 / MS"] + [f"{dx[c]['dev_classical']:+.2f}; {dx[c]['dev_t95']:+.2f} / {dx[c]['dev_ms']:+.2f}" for c in ("SRK", "PR")] + ["--"])
emit(
    "tabS_physics",
    "Physical diagnostics behind the interpretation in Sections 3 and 4 of the main text. Sensitivities are central "
    "differences ($\\pm 0.01\\%$ in $\\Omega_a$, which scales $a$ at fixed $T$ and $b$) at $T_r$ = 0.5, 0.7 and 0.9 exactly, for the fluids whose grid includes that temperature (28, 36 and 37 fluids). "
    "Effective exponents are least-squares slopes of $\\ln(\\rho' - \\rho'')$ and $\\ln \\Delta h_{vap}$ against $\\ln(1 - T_r)$ over "
    "the grid points with $T_r \\ge 0.93$; for the cubics the width is normalized by their own critical density, for the reference "
    "equations by the reference critical density. The $Z_c$ regression excludes carbon dioxide, which has no grid point below $T_r = 0.70$; "
    "the reducing-point $Z_c$ is $p/(\\rho R T)$ of the reference equation at its reducing temperature and density, which differs from the "
    "critical-point value of Table S2 for methyl chloride (0.269 against 0.294) and n-heptane (0.263 against 0.275), whose reference "
    "equations place the critical point 2.3 K and 1.1 K above the reducing temperature. The Clapeyron split writes "
    "$\\ln(\\Delta h_{vap}^{EoS}/\\Delta h_{vap}^{ref}) = \\ln(\\Delta Z_{vap}^{EoS}/\\Delta Z_{vap}^{ref}) + \\ln(s^{EoS}/s^{ref})$ "
    "with $\\Delta Z_{vap} = Z'' - Z'$ and $s = d\\ln p^{sat}/dT$, each model at its own saturation state; the first term is the "
    "volumetric part, the second, the difference of the two rows, follows the $\\alpha$-function.",
    [["Diagnostic", "SRK", "PR", "Reference"]],
    rows, colspec="p{6.2cm}ccc",
)


# ====================================================================== #
# LaTeX files and tables.json from the final content
for name, t in tables.items():
    write_tex(name, t)
with open(os.path.join(TABS, "tables.json"), "w") as f:
    json.dump(tables, f, indent=1)
print("wrote", len(tables), "tables (.tex and tables.json)")
