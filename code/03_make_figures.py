"""Generate the manuscript and SI figures (vector PDF + 600-dpi PNG) and the graphical abstract."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, HERE)
from fluid_set import FAMILY_ORDER  # noqa: E402
from eos import CubicEOS  # noqa: E402

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))
key = json.load(open(os.path.join(DATA, "key_numbers.json")))
st = json.load(open(os.path.join(DATA, "stats.json")))
TAGS = json.load(open(os.path.join(DATA, "run_metadata.json")))["models"]

# Colorblind-safe (Okabe-Ito) encodings, fixed order.
CUBIC_C = {"srk": "#0072B2", "pr": "#D55E00"}          # cubic form -> hue
ALPHA_C = {"s": "#0072B2", "s19": "#56B4E9", "t95": "#D55E00", "tc": "#009E73", "coq": "#CC79A7", "ms": "#E69F00", "tf": "white"}
ALPHA_L = {"s": "Soave", "s19": "Soave-19", "t95": "Twu-95", "tc": "Twu-c", "coq": "Coquelet", "ms": "MS", "tf": "Twu-fit"}
ALPHA_LS = {"s": "-", "t95": "--", "tc": "-.", "tf": ":"}
VT_C = {"none": None, "vtc": "0.78", "vt": "0.42"}       # translation rungs -> greys (hues are reserved for the cubics/alphas)
FAM_MARK = {
    "Inorganic gases": "o", "Alkanes": "s", "Aromatics": "^", "Polar/associating": "D", "Refrigerants": "v",
}
ZC_MODEL = {"srk": CubicEOS.PARAMS["SRK"]["Zc"], "pr": CubicEOS.PARAMS["PR"]["Zc"]}

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.4, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "hatch.linewidth": 0.6,
})


def save(fig, name, dpi=600):
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=dpi)
    plt.close(fig)
    print("wrote", name)


def aad(s):
    return np.nanmean(np.abs(s))


def lab(tag):
    c, a = tag.split("_")
    return f"{c.upper()}-{ALPHA_L[a]}"


# ================================================================== #
# Figure 1: overall summary, four properties, all models, bootstrap CIs
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4))
alphas = ["s", "s19", "t95", "tc", "coq", "ms", "tf"]
w = 0.12


def bar_group(ax, prop, ylabel, title):
    for gi, c in enumerate(("srk", "pr")):
        for ai, a in enumerate(alphas):
            t = f"{c}_{a}"
            d = st["ci"][f"{prop}_{t}"]
            x = gi + (ai - 3) * w
            ax.bar(x, d["aad"], w * 0.92, color=ALPHA_C[a], edgecolor="k", linewidth=0.5,
                   hatch="////" if a == "tf" else None, zorder=3)
            ax.errorbar(x, d["aad"], yerr=[[d["aad"] - d["ci_lo"]], [d["ci_hi"] - d["aad"]]],
                        fmt="none", ecolor="k", elinewidth=0.7, capsize=1.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SRK", "PR"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)


bar_group(axes[0, 0], "psat", r"AAD in $p^{\rm sat}$ (%)", "(a) vapor pressure")
bar_group(axes[0, 1], "h", r"AAD in $\Delta h_{\rm vap}$ (%)", "(b) enthalpy of vaporization")
bar_group(axes[1, 1], "rhov", r"AAD in $\rho''$ (%)", "(d) saturated-vapor density")
# (c) liquid density: untranslated (classical alpha) and the two predictive translations
ax = axes[1, 0]
wv = 0.26
for gi, c in enumerate(("srk", "pr")):
    for vi, (sfx, lbl, col, hatch) in enumerate((("", "untranslated", CUBIC_C[c], None),
                                                  ("_vtc", r"VT$_{\rm corr}$ (Rackett)", VT_C["vtc"], None),
                                                  ("_vt", "VT (one datum)", VT_C["vt"], None))):
        d = st["ci"][f"rho_{c}_s{sfx}"]
        x = gi + (vi - 1) * wv
        ax.bar(x, d["aad"], wv * 0.9, color=col, edgecolor="k", linewidth=0.5, hatch=hatch, zorder=3)
        ax.errorbar(x, d["aad"], yerr=[[d["aad"] - d["ci_lo"]], [d["ci_hi"] - d["aad"]]],
                    fmt="none", ecolor="k", elinewidth=0.7, capsize=1.5, zorder=4)
ax.set_xticks([0, 1])
ax.set_xticklabels(["SRK", "PR"])
ax.set_ylabel(r"AAD in $\rho'$ (%)")
ax.set_title("(c) saturated-liquid density", loc="left", fontweight="bold")
ax.grid(axis="y", alpha=0.25, lw=0.4)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor="white", edgecolor="k", label="untranslated (cubic color)"),
                   Patch(facecolor=VT_C["vtc"], edgecolor="k", label=r"VT$_{\rm corr}$: Rackett-anchored $c$"),
                   Patch(facecolor=VT_C["vt"], edgecolor="k", label="VT: one reference density")],
          frameon=False, loc="upper right", handlelength=1.2)
handles = [Patch(facecolor=ALPHA_C[a], edgecolor="k", hatch="////" if a == "tf" else None,
                 label=ALPHA_L[a] + (" (fitted reference)" if a == "tf" else "")) for a in alphas]
fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handlelength=1.2, title=r"$\alpha$-function (panels a, b, d)", title_fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.90))
save(fig, "fig1_overall_summary")

# ================================================================== #
# Figure 2: temperature-resolved psat and hvap deviations, four fluids
SHOW = ["methane", "n-decane", "R-134a", "water"]
CURVES = [("srk_s", "SRK-Soave", CUBIC_C["srk"], "-"), ("pr_s", "PR-Soave", CUBIC_C["pr"], "-"),
          ("srk_t95", "SRK-Twu-95", CUBIC_C["srk"], "--"), ("pr_t95", "PR-Twu-95", CUBIC_C["pr"], "--"),
          ("pr_tf", "PR-Twu-fit (fitted reference)", "0.45", ":")]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.2), sharex=True)
for j, f in enumerate(SHOW):
    g = df[df.fluid == f].sort_values("Tr")
    for i, col in enumerate(("dev_p", "dev_h")):
        ax = axes[i, j]
        ax.axhline(0, color="0.6", lw=0.8)
        for tag, lbl, colr, ls in CURVES:
            ax.plot(g.Tr, g[f"{col}_{tag}"], color=colr, ls=ls, label=lbl, lw=1.3 if tag != "pr_tf" else 1.1)
        ax.set_xlim(0.5, 1.0)
        ax.grid(alpha=0.2, lw=0.4)
    axes[0, j].set_title(f)
    axes[1, j].set_xlabel(r"$T_r = T/T_c$")
axes[0, 0].set_ylabel(r"$p^{\rm sat}$ deviation (%)")
axes[1, 0].set_ylabel(r"$\Delta h_{\rm vap}$ deviation (%)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handlelength=2.0)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig2_dev_vs_Tr")

# ================================================================== #
# Figure 3: per-fluid AADs (psat and hvap), all fluids, dot plot
order = []
for fam in FAMILY_ORDER:
    sub = fl[fl.family == fam].copy()
    sub["sg"] = sub.subgroup.map({"H-bonding": 0, "polar aprotic": 1}).fillna(0)
    sub = sub.sort_values(["sg", "omega"])
    order += list(sub.fluid)
ypos = {}
y = 0
fam_bounds = []
for fam in FAMILY_ORDER:
    sub = [f for f in order if fl.set_index("fluid").family[f] == fam]
    start = y
    for f in sub:
        ypos[f] = y
        y += 1
    fam_bounds.append((fam, start, y - 1))
    y += 0.8  # gap between families
DOTS = [("srk_s", "SRK-Soave", CUBIC_C["srk"], "o", CUBIC_C["srk"]),
        ("pr_s", "PR-Soave", CUBIC_C["pr"], "s", CUBIC_C["pr"]),
        ("srk_t95", "SRK-Twu-95", CUBIC_C["srk"], "o", "white"),
        ("pr_t95", "PR-Twu-95", CUBIC_C["pr"], "s", "white"),
        ("pr_tf", "PR-Twu-fit (fitted reference)", "0.3", "x", "0.3")]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 8.6), sharey=True)
for ax, prop, xlabel in zip(axes, ("psat", "h"), (r"AAD in $p^{\rm sat}$ (%)", r"AAD in $\Delta h_{\rm vap}$ (%)")):
    for f in order:
        pf = key["per_fluid"][f]
        for tag, lbl, edge, mk, face in DOTS:
            ax.scatter(pf[f"{prop}_{tag}"], ypos[f], marker=mk, s=22 if mk != "x" else 26,
                       facecolor=face, edgecolor=edge, linewidth=0.8, zorder=3,
                       label=lbl if f == order[0] else None)
    for fam, a, b in fam_bounds:
        ax.axhspan(a - 0.45, b + 0.45, color="0.5", alpha=0.07, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", which="both", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
axes[0].set_yticks([ypos[f] for f in order])
axes[0].set_yticklabels(order)
axes[0].invert_yaxis()
for fam, a, b in fam_bounds:
    axes[1].text(1.02, (a + b) / 2, fam, transform=axes[1].get_yaxis_transform(), rotation=90,
                 va="center", ha="left", fontsize=7.5, color="0.35")
axes[0].set_xlim(0.15, 15)
axes[1].set_xlim(0.4, 15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handletextpad=0.3)
fig.tight_layout(rect=(0, 0, 0.97, 0.955))
save(fig, "fig3_perfluid_dots")

# ================================================================== #
# Figure 5: liquid density by family: (a) mean signed deviation, untranslated;
# (b) AAD for the translation ladder
short = ["Inorg.\ngases", "Alkanes", "Aromatics", "Polar/\nassoc.", "Refrig."]
x = np.arange(len(FAMILY_ORDER))
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
ax = axes[0]
wb = 0.36
for i, c in enumerate(("srk", "pr")):
    vals = [key["by_family"][fam][f"rho_bias_{c}_s"] for fam in FAMILY_ORDER]
    ax.bar(x + (i - 0.5) * wb, vals, wb * 0.92, color=CUBIC_C[c], edgecolor="k", linewidth=0.4,
           label=c.upper(), zorder=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel(r"mean signed deviation in $\rho'$ (%)")
ax.set_title("(a) untranslated cubics: bias by family", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="lower left")
ax = axes[1]
wb = 0.13
items = [("srk", "", "SRK"), ("srk", "_vtc", r"SRK-VT$_{\rm corr}$"), ("srk", "_vt", "SRK-VT"),
         ("pr", "", "PR"), ("pr", "_vtc", r"PR-VT$_{\rm corr}$"), ("pr", "_vt", "PR-VT")]
for i, (c, sfx, lbl) in enumerate(items):
    vals = [key["by_family"][fam][f"rho_{c}_s{sfx}"] for fam in FAMILY_ORDER]
    col = CUBIC_C[c] if sfx == "" else VT_C[sfx.strip("_")]
    hatch = None if c == "pr" else "////"
    ax.bar(x + (i - 2.5) * wb, vals, wb * 0.92, color=col, edgecolor="k", linewidth=0.4,
           hatch=hatch, label=lbl, zorder=3)
ax.set_ylabel(r"AAD in $\rho'$ (%)")
ax.set_title("(b) translation ladder by family", loc="left", fontweight="bold")
ax.set_ylim(0, 24)
ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=1.3, loc="upper left")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig5_density_by_family")

# ================================================================== #
# Figure 6: liquid-density deviation vs Tr (propane, water), ladder for both cubics
LAD = [("dev_rho_srk_s", "SRK", CUBIC_C["srk"], "-"), ("dev_rho_srk_s_vtc", r"SRK-VT$_{\rm corr}$", CUBIC_C["srk"], "-."),
       ("dev_rho_srk_s_vt", "SRK-VT", CUBIC_C["srk"], "--"),
       ("dev_rho_pr_s", "PR", CUBIC_C["pr"], "-"), ("dev_rho_pr_s_vtc", r"PR-VT$_{\rm corr}$", CUBIC_C["pr"], "-."),
       ("dev_rho_pr_s_vt", "PR-VT", CUBIC_C["pr"], "--")]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharex=True)
for ax, f in zip(axes, ["propane", "water"]):
    g = df[df.fluid == f].sort_values("Tr")
    ax.axhline(0, color="0.6", lw=0.8)
    for col, lbl, colr, ls in LAD:
        ax.plot(g.Tr, g[col], color=colr, ls=ls, label=lbl)
    ax.set_title(f)
    ax.set_xlabel(r"$T_r = T/T_c$")
    ax.set_xlim(0.5, 1.0)
    ax.grid(alpha=0.2, lw=0.4)
axes[0].set_ylabel(r"$\rho'$ deviation (%)")
axes[0].legend(frameon=False, loc="lower left", ncol=2, columnspacing=1.0, fontsize=7)
fig.tight_layout()
save(fig, "fig6_rholiq_dev_vs_Tr")

# ================================================================== #
# Figure 7: near-critical density deviation vs the Zc-mismatch limit
top = df.loc[df.groupby("fluid").Tr.idxmax()].set_index("fluid")
zc = fl.set_index("fluid").Zc_ref
fluids = sorted(df.fluid.unique())
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for c, mk in (("srk", "o"), ("pr", "s")):
    xv = np.array([100.0 * (zc[f] / ZC_MODEL[c] - 1.0) for f in fluids])
    yv = np.array([top.loc[f, f"dev_rho_{c}_s"] for f in fluids])
    r = np.corrcoef(xv, yv)[0, 1]
    ax.scatter(xv, yv, marker=mk, s=30, facecolor=CUBIC_C[c], edgecolor="k", linewidth=0.4, zorder=3,
               label=f"{c.upper()} ($r={r:.2f}$)")
lims = [-40, 0]
ax.plot(lims, lims, color="0.4", lw=0.9, ls="--", zorder=2, label="1:1")
ax.set_xlim(lims)
ax.set_ylim(-45, 0)
ax.set_xlabel(r"$Z_c$-mismatch limit, $100\,(Z_c^{\rm ref}/Z_c^{\rm EoS}-1)$ (%)")
ax.set_ylabel(r"$\rho'$ deviation at $T_r=0.985$ (%)")
ax.grid(alpha=0.25, lw=0.4)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig7_zc_mechanism")

# ================================================================== #
# Figure 4: polar and associating fluids: generalized vs fitted alpha, vs Tr
POL = ["ammonia", "methanol", "ethanol", "acetone", "ethylene oxide", "sulfur dioxide"]
PCURVES = [("srk_s", "SRK-Soave", CUBIC_C["srk"], "-"), ("pr_s", "PR-Soave", CUBIC_C["pr"], "-"),
           ("srk_t95", "SRK-Twu-95", CUBIC_C["srk"], "--"), ("pr_tc", "PR-Twu-c", CUBIC_C["pr"], "-."),
           ("srk_tf", "SRK-Twu-fit", "0.45", ":"), ("pr_tf", "PR-Twu-fit", "0.1", ":")]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True)
for ax, f in zip(axes.ravel(), POL):
    g = df[df.fluid == f].sort_values("Tr")
    ax.axhline(0, color="0.6", lw=0.8)
    for tag, lbl, colr, ls in PCURVES:
        ax.plot(g.Tr, g[f"dev_p_{tag}"], color=colr, ls=ls, label=lbl, lw=1.2)
    ax.set_title(f)
    ax.set_xlim(0.5, 1.0)
    ax.grid(alpha=0.2, lw=0.4)
for ax in axes[1]:
    ax.set_xlabel(r"$T_r = T/T_c$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$p^{\rm sat}$ deviation (%)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.0, handlelength=1.8)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "fig4_polar_dev_vs_Tr")

# ================================================================== #
# Figure S1: anchor sweep, both cubics
sw = pd.read_csv(os.path.join(DATA, "anchor_sweep.csv"))
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for c in ("SRK", "PR"):
    s = sw[sw.cubic == c]
    ax.plot(s.Tr_anchor, s.aad_rho_vt, color=CUBIC_C[c.lower()], lw=1.6, label=f"{c}-VT (anchored at $T_{{r,a}}$)")
    ax.axhline(aad(df[f"dev_rho_{c.lower()}_s"]), color=CUBIC_C[c.lower()], lw=1.0, ls="--",
               label=f"{c}, untranslated")
    i070 = (s.Tr_anchor - 0.70).abs().idxmin()
    ax.plot(s.Tr_anchor[i070], s.aad_rho_vt[i070], "o", ms=6, mfc="white", mec=CUBIC_C[c.lower()], mew=1.4, zorder=4)
ax.set_xlabel(r"anchoring reduced temperature $T_{r,a}$")
ax.set_ylabel(r"pooled AAD in $\rho'$ (%)")
ax.set_yscale("log")
ax.set_xlim(0.5, 1.0)
ax.grid(alpha=0.25, which="both", lw=0.4)
ax.legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout()
save(fig, "figS1_anchor_sweep")

# ================================================================== #
# Figure S2: the alpha functions themselves: m(omega) and alpha(Tr)
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
ax = axes[0]
wgrid = np.linspace(-0.05, 0.7, 200)
from eos import M_SOAVE  # noqa: E402
for c in ("SRK", "PR"):
    for a, ls in (("s", "-"), ("s19", "--")):
        ax.plot(wgrid, [M_SOAVE[(c, a)](w_) for w_ in wgrid], color=CUBIC_C[c.lower()], ls=ls,
                label=f"{c}-{ALPHA_L[a]}")
ax.set_xlabel(r"acentric factor $\omega$")
ax.set_ylabel(r"Soave slope parameter $m$")
ax.set_title(r"(a) $m(\omega)$ correlations", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7)
ax = axes[1]
Tr = np.linspace(0.5, 1.0, 200)
for a, col in (("s", ALPHA_C["s"]), ("s19", ALPHA_C["s19"]), ("t95", ALPHA_C["t95"]), ("tc", ALPHA_C["tc"]),
               ("coq", ALPHA_C["coq"]), ("ms", ALPHA_C["ms"])):
    m = CubicEOS("PR", 500.0, 4e6, 0.30, alpha=a)
    al = np.array([m.alpha(t * 500.0)[0] for t in Tr])
    ref = np.array([CubicEOS("PR", 500.0, 4e6, 0.30, alpha="s").alpha(t * 500.0)[0] for t in Tr])
    ax.plot(Tr, 100 * (al / ref - 1), color=col, label=f"PR-{ALPHA_L[a]}")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel(r"$T_r$")
ax.set_ylabel(r"$100\,[\alpha/\alpha_{\rm Soave} - 1]$ (%)")
ax.set_title(r"(b) PR $\alpha$-functions at $\omega = 0.30$", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7, ncol=2)
for ax in axes:
    ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
save(fig, "figS2_alpha_functions")

# ================================================================== #
# Figure S3: the translation ladder resolved in temperature for four further
# fluids (same encoding as Figure 5): methane (density inversion), n-decane,
# R-134a, methanol
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=True)
for ax, f in zip(axes.ravel(), ["methane", "n-decane", "R-134a", "methanol"]):
    g = df[df.fluid == f].sort_values("Tr")
    ax.axhline(0, color="0.6", lw=0.8)
    for col_name, lbl, colr, ls in LAD:
        ax.plot(g.Tr, g[col_name], color=colr, ls=ls, label=lbl)
    ax.set_title(f)
    ax.set_xlim(0.5, 1.0)
    ax.grid(alpha=0.2, lw=0.4)
for ax in axes[1]:
    ax.set_xlabel(r"$T_r = T/T_c$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\rho'$ deviation (%)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handlelength=2.0, fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "figS3_ladder_dev_vs_Tr")

# ================================================================== #
# Graphical abstract (MDPI: >= 1100 x 560 px): two ladders, side by side
fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), dpi=200)
ax = axes[0]
pairs = [("s", "Soave\n(1972/76)"), ("s19", "Soave-19"), ("t95", "Twu-95"), ("tc", "Twu-c"), ("coq", "Coquelet"), ("ms", "MS"), ("tf", "Twu-fit\n(fitted)")]
xs = np.arange(len(pairs))
for i, c in enumerate(("srk", "pr")):
    vals = [key["overall"][f"psat_{c}_{a}"] for a, _ in pairs]
    ax.bar(xs + (i - 0.5) * 0.38, vals, 0.36, color=CUBIC_C[c], edgecolor="k", linewidth=0.6,
           label=c.upper(), zorder=3)
    for xx, v in zip(xs + (i - 0.5) * 0.38, vals):
        ax.text(xx, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([p[1] for p in pairs], fontsize=10)
ax.set_ylabel("vapor-pressure AAD, 37 fluids (%)", fontsize=11)
ax.set_title(r"$\alpha$-function: sets the SRK–PR gap", fontsize=13, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=11, loc="upper right")
ax.set_ylim(0, 2.3)
ax = axes[1]
lad = [("", "untranslated"), ("_vtc", "Rackett-anchored\n(no data)"), ("_vt", "density-anchored\n(one reference datum)")]
xs = np.arange(len(lad))
for i, c in enumerate(("srk", "pr")):
    vals = [key["overall"][f"rho_{c}_s{s}"] for s, _ in lad]
    ax.bar(xs + (i - 0.5) * 0.38, vals, 0.36, color=CUBIC_C[c], edgecolor="k", linewidth=0.6,
           label=c.upper(), zorder=3)
    for xx, v in zip(xs + (i - 0.5) * 0.38, vals):
        ax.text(xx, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([p[1] for p in lad], fontsize=10)
ax.set_ylabel("liquid-density AAD, 37 fluids (%)", fontsize=11)
ax.set_title("volume translation: sets the density error", fontsize=13, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=11, loc="upper right")
ax.set_ylim(0, 15)
for ax in axes:
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
fig.suptitle("SRK and PR vs. reference equations of state: where the errors come from", fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(FIGS, "graphical_abstract.png"), dpi=200)
fig.savefig(os.path.join(FIGS, "graphical_abstract.pdf"))
plt.close(fig)
print("wrote graphical_abstract")
print("done")
