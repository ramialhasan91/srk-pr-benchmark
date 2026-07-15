"""Generate the manuscript figures (vector PDF + PNG previews)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))

C = {"SRK": "#0072B2", "PR": "#D55E00", "PR-VT": "#009E73"}
MODEL_STYLE = {  # tag -> (label, color, linestyle)
    "srk": ("SRK", C["SRK"], "-"),
    "pr": ("PR", C["PR"], "-"),
    "srktwu": ("SRK-Twu", C["SRK"], "--"),
    "prtwu": ("PR-Twu", C["PR"], "--"),
}
FAM_MARK = {
    "Inorganic gases": ("o", "#0072B2"),
    "Alkanes": ("s", "#D55E00"),
    "Aromatics": ("^", "#009E73"),
    "Polar/associating": ("D", "#CC79A7"),
    "Refrigerants": ("v", "#E69F00"),
}
FAMILY_ORDER = list(FAM_MARK.keys())
ZC_MODEL = {"srk": 1.0 / 3.0, "pr": 0.3074}

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.4, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150)
    plt.close(fig)
    print("wrote", name)


def aad(s):
    return np.nanmean(np.abs(s))


SHOW = ["methane", "n-decane", "water", "R-134a"]

# ------------------------------------------------------------------ #
# Figure 1: psat percent deviation vs Tr, four representative fluids,
# four (cubic, alpha) combinations
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True)
for ax, f in zip(axes.ravel(), SHOW):
    g = df[df.fluid == f].sort_values("Tr")
    ax.axhline(0, color="0.6", lw=0.8)
    for tag, (lbl, col, ls) in MODEL_STYLE.items():
        ax.plot(g.Tr, g[f"dev_p_{tag}"], color=col, ls=ls, label=lbl)
    ax.set_title(f)
    ax.set_xlim(0.5, 1.0)
for ax in axes[1]:
    ax.set_xlabel(r"$T_r = T/T_c$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$100\,(p^{\rm sat}_{\rm cubic}-p^{\rm sat}_{\rm ref})/p^{\rm sat}_{\rm ref}$")
axes[0, 0].legend(frameon=False, loc="lower right", ncol=2,
                  columnspacing=0.8, handlelength=1.6)
fig.tight_layout()
save(fig, "fig1_psat_dev_vs_Tr")

# ------------------------------------------------------------------ #
# Figure 2: per-fluid psat AAD vs acentric factor, one panel per model
per = []
for f, g in df.groupby("fluid"):
    fam = g.family.iloc[0]
    w = fl.loc[fl.fluid == f, "omega"].iloc[0]
    per.append((f, fam, w, *[aad(g[f"dev_p_{t}"]) for t in MODEL_STYLE]))
per = pd.DataFrame(per, columns=["fluid", "family", "omega", *MODEL_STYLE])

fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.2), sharey=True, sharex=True)
for ax, tag in zip(axes.ravel(), MODEL_STYLE):
    lbl = MODEL_STYLE[tag][0]
    for fam in FAMILY_ORDER:
        s = per[per.family == fam]
        mk, col = FAM_MARK[fam]
        ax.scatter(s.omega, s[tag], marker=mk, s=26, facecolor=col,
                   edgecolor="k", linewidth=0.4, label=fam, zorder=3)
    ax.set_title(lbl)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.4)
for ax in axes[1]:
    ax.set_xlabel(r"acentric factor $\omega$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"AAD in $p^{\rm sat}$ (%)")
axes[0, 0].legend(frameon=False, loc="upper left", handletextpad=0.2,
                  borderaxespad=0.2, labelspacing=0.3)
fig.tight_layout()
save(fig, "fig2_psat_aad_vs_omega")

# ------------------------------------------------------------------ #
# Figure 3: enthalpy-of-vaporization deviation vs Tr, same four fluids.
# The legend sits above the panels (figure-level) so it never overlaps
# the curves.
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.9), sharex=True)
for ax, f in zip(axes.ravel(), SHOW):
    g = df[df.fluid == f].sort_values("Tr")
    ax.axhline(0, color="0.6", lw=0.8)
    for tag, (lbl, col, ls) in MODEL_STYLE.items():
        ax.plot(g.Tr, g[f"dev_h_{tag}"], color=col, ls=ls, label=lbl)
    ax.set_title(f)
    ax.set_xlim(0.5, 1.0)
for ax in axes[1]:
    ax.set_xlabel(r"$T_r = T/T_c$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$100\,(\Delta h_{\rm cubic}-\Delta h_{\rm ref})/\Delta h_{\rm ref}$")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=1.8)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "fig3_hvap_dev_vs_Tr")

# ------------------------------------------------------------------ #
# Figure 4: saturated-liquid density deviation vs Tr (propane, water)
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharex=True)
for ax, f in zip(axes, ["propane", "water"]):
    g = df[df.fluid == f].sort_values("Tr")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(g.Tr, g.dev_rho_srk, color=C["SRK"], label="SRK")
    ax.plot(g.Tr, g.dev_rho_pr, color=C["PR"], label="PR")
    ax.plot(g.Tr, g.dev_rho_prvt, color=C["PR-VT"], label="PR-VT")
    ax.set_title(f)
    ax.set_xlabel(r"$T_r = T/T_c$")
    ax.set_xlim(0.5, 1.0)
axes[0].set_ylabel(r"$100\,(\rho'_{\rm cubic}-\rho'_{\rm ref})/\rho'_{\rm ref}$")
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "fig4_rholiq_dev_vs_Tr")

# ------------------------------------------------------------------ #
# Figure 5: grouped bars, AAD by family, three properties
short = ["Inorg.\ngases", "Alkanes", "Aromatics", "Polar/\nassoc.", "Refrig."]
x = np.arange(len(FAMILY_ORDER))

fam_stats = {}
for fam in FAMILY_ORDER:
    g = df[df.family == fam]
    fam_stats[fam] = dict(
        **{f"p_{t}": aad(g[f"dev_p_{t}"]) for t in MODEL_STYLE},
        **{f"h_{t}": aad(g[f"dev_h_{t}"]) for t in MODEL_STYLE},
        r_srk=aad(g.dev_rho_srk), r_pr=aad(g.dev_rho_pr), r_prvt=aad(g.dev_rho_prvt),
    )
fs = pd.DataFrame(fam_stats).T.loc[FAMILY_ORDER]

fig, axes = plt.subplots(3, 1, figsize=(6.3, 6.9))

w = 0.19
for i, (tag, (lbl, col, ls)) in enumerate(MODEL_STYLE.items()):
    hatch = "//" if ls == "--" else None
    axes[0].bar(x + (i - 1.5) * w, fs[f"p_{tag}"], w, color=col,
                hatch=hatch, edgecolor="k", linewidth=0.4, label=lbl)
    axes[1].bar(x + (i - 1.5) * w, fs[f"h_{tag}"], w, color=col,
                hatch=hatch, edgecolor="k", linewidth=0.4, label=lbl)
axes[0].set_ylabel(r"AAD in $p^{\rm sat}$ (%)")
axes[0].set_title("Vapor pressure")
axes[1].set_ylabel(r"AAD in $\Delta h_{\rm vap}$ (%)")
axes[1].set_title("Enthalpy of vaporization")

w = 0.26
for i, (tag, lbl) in enumerate((("r_srk", "SRK"), ("r_pr", "PR"), ("r_prvt", "PR-VT"))):
    axes[2].bar(x + (i - 1) * w, fs[tag], w, color=C[lbl],
                edgecolor="k", linewidth=0.4, label=lbl)
axes[2].set_ylabel(r"AAD in $\rho'$ (%)")
axes[2].set_title("Saturated liquid density")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.legend(frameon=False, ncol=2, columnspacing=0.9)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig5_aad_by_family")

# ------------------------------------------------------------------ #
# Figure 6: near-critical density deviation vs the Zc-mismatch limit
top = df.loc[df.groupby("fluid").Tr.idxmax()].set_index("fluid")
zc = fl.set_index("fluid").Zc_ref
fluids = sorted(df.fluid.unique())

fig, ax = plt.subplots(figsize=(4.6, 3.6))
for tag, mk in (("srk", "o"), ("pr", "s")):
    lbl = MODEL_STYLE[tag][0]
    xv = np.array([100.0 * (zc[f] / ZC_MODEL[tag] - 1.0) for f in fluids])
    yv = np.array([top.loc[f, f"dev_rho_{tag}"] for f in fluids])
    r = np.corrcoef(xv, yv)[0, 1]
    ax.scatter(xv, yv, marker=mk, s=30, facecolor=MODEL_STYLE[tag][1],
               edgecolor="k", linewidth=0.4, zorder=3,
               label=f"{lbl} ($r={r:.2f}$)")
lims = [-40, 0]
ax.plot(lims, lims, color="0.4", lw=0.9, ls="--", zorder=2, label="1:1")
ax.set_xlim(lims)
ax.set_ylim(-42, 0)
ax.set_xlabel(r"$Z_c$-mismatch limit, $100\,(Z_c^{\rm ref}/Z_c^{\rm model}-1)$ (%)")
ax.set_ylabel(r"$\rho'$ deviation at $T_r=0.985$ (%)")
ax.grid(alpha=0.25, lw=0.4)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig6_zc_mechanism")

# ------------------------------------------------------------------ #
# Supplementary Figure S1: sensitivity of the PR-VT density AAD to the
# anchoring temperature
sw = pd.read_csv(os.path.join(DATA, "anchor_sweep.csv"))
pr_aad = aad(df.dev_rho_pr)

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(sw.Tr_anchor, sw.aad_rho_prvt, color=C["PR-VT"], lw=1.6,
        label="PR-VT (anchored at $T_{r,a}$)")
ax.axhline(pr_aad, color=C["PR"], lw=1.1, ls="--",
           label="PR, untranslated")
i070 = (sw.Tr_anchor - 0.70).abs().idxmin()
ax.plot(sw.Tr_anchor[i070], sw.aad_rho_prvt[i070], "o", ms=6,
        mfc="white", mec=C["PR-VT"], mew=1.4, zorder=4,
        label=r"$T_{r,a} = 0.70$ (this work)")
ax.set_xlabel(r"anchoring reduced temperature $T_{r,a}$")
ax.set_ylabel(r"pooled AAD in $\rho'$ (%)")
ax.set_yscale("log")
ax.set_xlim(0.5, 1.0)
ax.grid(alpha=0.25, which="both", lw=0.4)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "figS1_anchor_sweep")

# ------------------------------------------------------------------ #
# Supplementary Figure S2: per-fluid hvap AAD vs acentric factor,
# one panel per (cubic, alpha) combination (mirrors Figure 2)
per_h = []
for f, g in df.groupby("fluid"):
    fam = g.family.iloc[0]
    w = fl.loc[fl.fluid == f, "omega"].iloc[0]
    per_h.append((f, fam, w, *[aad(g[f"dev_h_{t}"]) for t in MODEL_STYLE]))
per_h = pd.DataFrame(per_h, columns=["fluid", "family", "omega", *MODEL_STYLE])

fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.2), sharey=True, sharex=True)
for ax, tag in zip(axes.ravel(), MODEL_STYLE):
    lbl = MODEL_STYLE[tag][0]
    for fam in FAMILY_ORDER:
        sub = per_h[per_h.family == fam]
        mk, col = FAM_MARK[fam]
        ax.scatter(sub.omega, sub[tag], marker=mk, s=26, facecolor=col,
                   edgecolor="k", linewidth=0.4, label=fam, zorder=3)
    ax.set_title(lbl)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.4)
for ax in axes[1]:
    ax.set_xlabel(r"acentric factor $\omega$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"AAD in $\Delta h_{\rm vap}$ (%)")
axes[0, 0].legend(frameon=False, loc="upper left", handletextpad=0.2,
                  borderaxespad=0.2, labelspacing=0.3)
fig.tight_layout()
save(fig, "figS2_hvap_aad_vs_omega")

# ------------------------------------------------------------------ #
# Supplementary Figure S3: the translation ladder resolved in temperature
# for two representative fluids (PR, Rackett-anchored, fitted)
LADDER_STYLE = (("dev_rho_pr", "PR", C["PR"], "-"),
                ("dev_rho_prvtc", r"PR-VT$_{\rm corr}$", "#CC79A7", "-."),
                ("dev_rho_prvt", "PR-VT", C["PR-VT"], "--"))
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharex=True)
for ax, f in zip(axes, ["propane", "water"]):
    g = df[df.fluid == f].sort_values("Tr")
    ax.axhline(0, color="0.6", lw=0.8)
    for col_name, lbl, col, ls in LADDER_STYLE:
        ax.plot(g.Tr, g[col_name], color=col, ls=ls, label=lbl)
    ax.set_title(f)
    ax.set_xlabel(r"$T_r = T/T_c$")
    ax.set_xlim(0.5, 1.0)
axes[0].set_ylabel(r"$100\,(\rho'_{\rm model}-\rho'_{\rm ref})/\rho'_{\rm ref}$")
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "figS3_ladder_dev_vs_Tr")

print("done")
