"""Generate the LaTeX tables for the main and supplementary manuscripts."""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TABL = os.path.join(HERE, "..", "tables")
os.makedirs(TABL, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv"))

FAMILY_ORDER = ["Inorganic gases", "Alkanes", "Aromatics",
                "Polar/associating", "Refrigerants"]
TAGS = ["srk", "pr", "srktwu", "prtwu"]


def aad(s):
    return np.nanmean(np.abs(s))


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


fl_sorted = fl.copy()
fl_sorted["fam_rank"] = fl_sorted.family.map({f: i for i, f in enumerate(FAMILY_ORDER)})
fl_sorted = fl_sorted.sort_values(["fam_rank", "omega"])

# ------------------------------------------------------------------ #
# Fluid-set table (tab:fluids)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Fluid set. Critical constants and acentric factors are taken "
    r"from the reference equations of state via CoolProp \cite{Bell2014}. "
    r"$T_{r,\min}$ is the lower end of the temperature grid. "
    r"The per-fluid volume-translation constants of the translated-PR "
    r"models are listed in Table~\ref{tab:vt}.}",
    r"\label{tab:fluids}",
    r"\small",
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"Fluid & Family & $T_c$ (K) & $p_c$ (MPa) & $\omega$ & $T_{r,\min}$ \\",
    r"\midrule",
]
for _, r in fl_sorted.iterrows():
    lines.append(
        f"{r.fluid} & {r.family} & {r.Tc_K:.2f} & {r.pc_MPa:.3f} & "
        f"{r.omega:.4f} & {r.Tr_min:.3f} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_fluids.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Temperature-binned AAD table, psat and liquid density (tab:bins)
bins = pd.cut(df.Tr, [0.50, 0.70, 0.90, 0.9851], right=False,
              labels=[r"$[0.50,0.70)$", r"$[0.70,0.90)$", r"$[0.90,0.985]$"])
df["bin"] = bins


def bin_row(name, g):
    vals = [aad(g[f"dev_p_{t}"]) for t in TAGS]
    vals += [aad(g.dev_rho_srk), aad(g.dev_rho_pr), aad(g.dev_rho_prvt)]
    return (name, *vals)


rows = [bin_row(str(b), g) for b, g in df.groupby("bin", observed=True)]
rows.append(bin_row("All", df))

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Average absolute deviation (AAD, \%) from the reference "
    r"equations of state, pooled over all 32 fluids, by reduced-temperature "
    r"range. SRK-Twu and PR-Twu use the generalized Twu $\alpha$-function "
    r"\cite{Twu1995a,Twu1995b} in place of the Soave form. PR-VT differs "
    r"from PR only for the liquid density; its vapor pressure is identical "
    r"to PR.}",
    r"\label{tab:bins}",
    r"\small",
    r"\begin{tabular}{lrrrrrrr}",
    r"\toprule",
    r" & \multicolumn{4}{c}{$p^{\mathrm{sat}}$} & \multicolumn{3}{c}{$\rho'$} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
    r"$T_r$ range & SRK & PR & SRK-Twu & PR-Twu & SRK & PR & PR-VT \\",
    r"\midrule",
]
for name, *vals in rows:
    pre = r"\midrule" + "\n" if name == "All" else ""
    lines.append(pre + name + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_bins.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Family-resolved AAD table (tab:family)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{AAD (\%) by chemical family, pooled over all fluids and "
    r"temperatures in each family.}",
    r"\label{tab:family}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrrrrrrrr}",
    r"\toprule",
    r" & & \multicolumn{4}{c}{$p^{\mathrm{sat}}$} & \multicolumn{3}{c}{$\rho'$} \\",
    r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
    r"Family & $N$ & SRK & PR & SRK-Twu & PR-Twu & SRK & PR & PR-VT \\",
    r"\midrule",
]
for fam in FAMILY_ORDER:
    g = df[df.family == fam]
    n = g.fluid.nunique()
    vals = [aad(g[f"dev_p_{t}"]) for t in TAGS]
    vals += [aad(g.dev_rho_srk), aad(g.dev_rho_pr), aad(g.dev_rho_prvt)]
    lines.append(f"{fam} & {n} & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_family.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Enthalpy-of-vaporization table (tab:hvap)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{AAD (\%) in the enthalpy of vaporization "
    r"$\Delta h_{\mathrm{vap}}$, by reduced-temperature range and by "
    r"chemical family. A constant volume translation leaves "
    r"$\Delta h_{\mathrm{vap}}$ unchanged, so PR-VT is identical to PR "
    r"for this property.}",
    r"\label{tab:hvap}",
    r"\small",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r" & SRK & PR & SRK-Twu & PR-Twu \\",
    r"\midrule",
]
for b, g in df.groupby("bin", observed=True):
    vals = [aad(g[f"dev_h_{t}"]) for t in TAGS]
    lines.append(str(b) + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
vals = [aad(df[f"dev_h_{t}"]) for t in TAGS]
lines.append(r"\midrule")
lines.append("All & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines.append(r"\midrule")
for fam in FAMILY_ORDER:
    g = df[df.family == fam]
    vals = [aad(g[f"dev_h_{t}"]) for t in TAGS]
    lines.append(fam + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_hvap.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Volume-translation ladder table (tab:ladder)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{The volume-translation ladder: AAD (\%) in saturated-liquid "
    r"density for untranslated PR and for two predictive choices of the "
    r"constant $c$, by reduced-temperature range. PR-VT$_{\mathrm{corr}}$ "
    r"anchors $c$ to the Rackett--Yamada--Gunn correlation at "
    r"$T_r = 0.70$ (no data beyond $T_c$, $p_c$, $\omega$); PR-VT anchors "
    r"$c$ to one reference liquid density at $T_r = 0.70$. The third "
    r"conceivable constant, matching the reference critical volume, is "
    r"unusable and is discussed in the text; all three constants are "
    r"tabulated per fluid in Table~S6 of the Supplementary Material.}",
    r"\label{tab:ladder}",
    r"\small",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"$T_r$ range & PR & PR-VT$_{\mathrm{corr}}$ & PR-VT \\",
    r"\midrule",
]
for b, g in df.groupby("bin", observed=True):
    vals = [aad(g.dev_rho_pr), aad(g.dev_rho_prvtc), aad(g.dev_rho_prvt)]
    lines.append(str(b) + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
vals = [aad(df.dev_rho_pr), aad(df.dev_rho_prvtc), aad(df.dev_rho_prvt)]
lines.append(r"\midrule")
lines.append("All & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_ladder.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Saturated-vapor-density table (tab:rhov)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{AAD (\%) in the saturated-vapor density $\rho''$, by "
    r"reduced-temperature range, for the four $(\text{cubic}, \alpha)$ "
    r"combinations and for the translated PR-VT (fitted constant).}",
    r"\label{tab:rhov}",
    r"\small",
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"$T_r$ range & SRK & PR & SRK-Twu & PR-Twu & PR-VT \\",
    r"\midrule",
]
for b, g in df.groupby("bin", observed=True):
    vals = [aad(g[f"dev_rhov_{t}"]) for t in TAGS] + [aad(g.dev_rhov_prvt)]
    lines.append(str(b) + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
vals = [aad(df[f"dev_rhov_{t}"]) for t in TAGS] + [aad(df.dev_rhov_prvt)]
lines.append(r"\midrule")
lines.append("All & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_rhov.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Appendix table: per-fluid psat and density AADs (tab:perfluid)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Per-fluid AAD (\%) in vapor pressure and saturated-liquid "
    r"density over the full temperature grid of each fluid.}",
    r"\label{tab:perfluid}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrrrrrrrr}",
    r"\toprule",
    r" & & \multicolumn{4}{c}{$p^{\mathrm{sat}}$} & \multicolumn{3}{c}{$\rho'$} \\",
    r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
    r"Fluid & $\omega$ & SRK & PR & SRK-Twu & PR-Twu & SRK & PR & PR-VT \\",
    r"\midrule",
]
for fam in FAMILY_ORDER:
    sub = fl_sorted[fl_sorted.family == fam]
    lines.append(r"\multicolumn{9}{l}{\itshape " + fam + r"}\\")
    for _, r in sub.iterrows():
        g = df[df.fluid == r.fluid]
        vals = [aad(g[f"dev_p_{t}"]) for t in TAGS]
        vals += [aad(g.dev_rho_srk), aad(g.dev_rho_pr), aad(g.dev_rho_prvt)]
        lines.append(f"\\quad {r.fluid} & {r.omega:.3f} & "
                     + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_perfluid.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Appendix table: per-fluid enthalpy-of-vaporization AADs (tab:perfluidh)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Per-fluid AAD (\%) in the enthalpy of vaporization over the "
    r"full temperature grid of each fluid.}",
    r"\label{tab:perfluidh}",
    r"\footnotesize",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Fluid & SRK & PR & SRK-Twu & PR-Twu \\",
    r"\midrule",
]
for fam in FAMILY_ORDER:
    sub = fl_sorted[fl_sorted.family == fam]
    lines.append(r"\multicolumn{5}{l}{\itshape " + fam + r"}\\")
    for _, r in sub.iterrows():
        g = df[df.fluid == r.fluid]
        vals = [aad(g[f"dev_h_{t}"]) for t in TAGS]
        lines.append(f"\\quad {r.fluid} & "
                     + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_perfluid_h.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Appendix table: translation constants and reference Zc (tab:vt)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Per-fluid constants of the translated-PR models: reference "
    r"critical compressibility factor $Z_c^{\mathrm{ref}}$, anchoring "
    r"temperature $T_{r,\mathrm{anchor}}$, fitted constant $c$ (translated "
    r"PR liquid volume matches the reference value at "
    r"$T_{r,\mathrm{anchor}}$), correlation constant $c_{\mathrm{corr}}$ "
    r"(matches the Rackett--Yamada--Gunn prediction instead, no data "
    r"required), and critical-matched constant "
    r"$c_{Z_c} = (Z_c^{\mathrm{PR}} - Z_c^{\mathrm{ref}})\,R T_c/p_c$ "
    r"(diagnostic only; see text). All $c$ in cm$^3$\,mol$^{-1}$.}",
    r"\label{tab:vt}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4.5pt}",
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"Fluid & $Z_c^{\mathrm{ref}}$ & $T_{r,\mathrm{anchor}}$ & $c$ & "
    r"$c_{\mathrm{corr}}$ & $c_{Z_c}$ \\",
    r"\midrule",
]
for fam in FAMILY_ORDER:
    sub = fl_sorted[fl_sorted.family == fam]
    lines.append(r"\multicolumn{6}{l}{\itshape " + fam + r"}\\")
    for _, r in sub.iterrows():
        lines.append(f"\\quad {r.fluid} & {r.Zc_ref:.4f} & {r.Tr_anchor:.3f} & "
                     f"{r.c_VT_cm3mol:.2f} & {r.c_corr_cm3mol:.2f} & "
                     f"{r.c_zc_cm3mol:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_vt.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Supplementary table: per-fluid saturated-vapor-density AADs (tab:perfluidrv)
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Per-fluid AAD (\%) in the saturated-vapor density over the "
    r"full temperature grid of each fluid.}",
    r"\label{tab:perfluidrv}",
    r"\footnotesize",
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"Fluid & SRK & PR & SRK-Twu & PR-Twu & PR-VT \\",
    r"\midrule",
]
for fam in FAMILY_ORDER:
    sub = fl_sorted[fl_sorted.family == fam]
    lines.append(r"\multicolumn{6}{l}{\itshape " + fam + r"}\\")
    for _, r in sub.iterrows():
        g = df[df.fluid == r.fluid]
        vals = [aad(g[f"dev_rhov_{t}"]) for t in TAGS] + [aad(g.dev_rhov_prvt)]
        lines.append(f"\\quad {r.fluid} & "
                     + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_perfluid_rhov.tex"), "\n".join(lines) + "\n")

# ------------------------------------------------------------------ #
# Supplementary table: sign-test summary (tab:signtests), from stats.json
with open(os.path.join(DATA, "stats.json")) as f:
    stats = json.load(f)

TEST_LABELS = [
    ("psat_pr_vs_srk", "PR vs.\\ SRK", r"$p^{\mathrm{sat}}$"),
    ("psat_srktwu_vs_srk", "SRK-Twu vs.\\ SRK", r"$p^{\mathrm{sat}}$"),
    ("psat_prtwu_vs_pr", "PR-Twu vs.\\ PR", r"$p^{\mathrm{sat}}$"),
    ("psat_prtwu_vs_srktwu", "PR-Twu vs.\\ SRK-Twu", r"$p^{\mathrm{sat}}$"),
    ("h_pr_vs_srk", "PR vs.\\ SRK", r"$\Delta h_{\mathrm{vap}}$"),
    ("h_srktwu_vs_srk", "SRK-Twu vs.\\ SRK", r"$\Delta h_{\mathrm{vap}}$"),
    ("h_prtwu_vs_pr", "PR-Twu vs.\\ PR", r"$\Delta h_{\mathrm{vap}}$"),
    ("rho_pr_vs_srk", "PR vs.\\ SRK", r"$\rho'$"),
    ("rhov_pr_vs_srk", "PR vs.\\ SRK", r"$\rho''$"),
    ("rhov_prtwu_vs_pr", "PR-Twu vs.\\ PR", r"$\rho''$"),
]


def fmt_p(p):
    if p >= 0.995:
        return "1.0"
    if p >= 0.01:
        return f"{p:.2f}"
    mant, expo = f"{p:.0e}".split("e")
    return rf"${mant}\times10^{{{int(expo)}}}$"


lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Exact two-sided binomial sign tests on per-fluid AADs for all "
    r"pairwise model comparisons quoted in the main text. ``Wins'' counts "
    r"the fluids for which the first-named model has the lower AAD "
    r"($n = 32$, no ties occurred).}",
    r"\label{tab:signtests}",
    r"\small",
    r"\begin{tabular}{llrr}",
    r"\toprule",
    r"Comparison & Property & Wins & $p$ \\",
    r"\midrule",
]
for key, label, prop in TEST_LABELS:
    t = stats["sign_tests"][key]
    lines.append(f"{label} & {prop} & {t['wins']}/{t['n']} & {fmt_p(t['p_two_sided'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write(os.path.join(TABL, "tab_signtests.tex"), "\n".join(lines) + "\n")

print("done")
