"""In-sample alpha-function fit per fluid and cubic (Section 2.5 of the paper; tables tabS_insample,
tabS_insample_params, tabS_floorloss). For every fluid and both cubics the alpha function is refitted
to the reference vapor pressures on the paper's 40-point grid with one objective and one algorithm:
    objective  F(theta) = sum_i [ln psat_EoS(T_i; theta) - ln psat_ref(T_i)]^2
    forms      (a) Soave, one parameter m (bounded scalar minimization)
               (b) consistent Twu-91 (L, M, N) (SLSQP under the consistency constraints, multi-start)
The consistency constraints of Le Guennec et al. (alpha >= 0, alpha' <= 0, alpha'' >= 0, alpha''' <= 0
for every Tr) are written for the Twu-91 form with a = N(M-1), b = N M, y = L b Tr^b:
    alpha'  <= 0  <=>  a <= 0 and L b >= 0
    alpha'' >= 0  <=>  P2(y) = y^2 + (1 - b - 2a) y + (a^2 - a) >= 0 for all y >= 0
    alpha'''<= 0  <=>  Q3(y) = y^3 - 3(a+b-1) y^2 + (3a^2 + 3ab - 6a + b^2 - 3b + 2) y - (a^3 - 3a^2 + 2a) >= 0 for all y >= 0
(derived symbolically; the 74 published tc-RK/tc-PR sets satisfy them to rounding). A vectorized
saturation solver (the scheme of eos.py with closed-form cubic roots and Newton polish) runs inside the
optimizer and is verified against eos.py on the committed parameter sets before use.

Reads  ../data/deviations_full.csv, fluids.csv, twu_fit_params.csv
Writes ../data/insample_fit.csv (parameters, objective, AADs, constraint activity, start record),
       insample_fit_starts.csv (every start of every fit), insample_fit_meta.json (solver check, time, versions)
Runtime 20-25 min (74 fits x 6 starts); --quick fits a subset of fluids and writes *_quick files.
"""
import json
import os
import sys
import time

import platform

import CoolProp
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
QUICK = "--quick" in sys.argv
from eos import R, CubicEOS, TWU_CONSISTENT_GENERALIZED, M_SOAVE  # noqa: E402

df = pd.read_csv(os.path.join(DATA, "deviations_full.csv"))
fl = pd.read_csv(os.path.join(DATA, "fluids.csv")).set_index("fluid")
tf = pd.read_csv(os.path.join(DATA, "twu_fit_params.csv"), comment="#").set_index("fluid")
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                CoolProp=CoolProp.__version__)
PARAMS = CubicEOS.PARAMS
SQ2 = np.sqrt(2.0)
BOUNDS_TWU = [(0.0, 5.0), (1e-4, 1.0), (1e-3, 8.0)]


# ------------------------------------------------------------------ alpha functions (vectorized)
def alpha_soave(Tr, m):
    f = 1.0 + m * (1.0 - np.sqrt(Tr))
    return f * f, -m * f / np.sqrt(Tr)


def alpha_twu(Tr, L, M, N):
    al = Tr ** (N * (M - 1.0)) * np.exp(L * (1.0 - Tr ** (N * M)))
    dln = N * (M - 1.0) / Tr - L * M * N * Tr ** (N * M - 1.0)
    return al, al * dln


# ------------------------------------------------------------------ consistency margins
def consistency_margins(L, M, N):
    """Minimum over y >= 0 of P2 and Q3 (>= 0 means consistent), and the first-derivative margin."""
    a, b = N * (M - 1.0), N * M
    # P2: quadratic, minimum on y >= 0
    c1 = 1.0 - b - 2.0 * a
    c0 = a * a - a
    ymin = max(0.0, -c1 / 2.0)
    p2 = ymin * ymin + c1 * ymin + c0
    # Q3: cubic with leading coefficient +1; check y = 0 and the stationary points with y > 0
    q2 = -3.0 * (a + b - 1.0)
    q1 = 3.0 * a * a + 3.0 * a * b - 6.0 * a + b * b - 3.0 * b + 2.0
    q0 = -(a ** 3 - 3.0 * a * a + 2.0 * a)
    cand = [0.0]
    disc = 4.0 * q2 * q2 - 12.0 * q1
    if disc >= 0.0:
        r = np.sqrt(disc)
        for yy in ((-2.0 * q2 + r) / 6.0, (-2.0 * q2 - r) / 6.0):
            if yy > 0.0:
                cand.append(yy)
    q3 = min(yy ** 3 + q2 * yy * yy + q1 * yy + q0 for yy in cand)
    d1 = min(-a, L * b)          # both must be >= 0
    return p2, q3, d1


# ------------------------------------------------------------------ vectorized saturation solver
def cubic_real_roots(c2, c1, c0):
    """Real roots of z^3 + c2 z^2 + c1 z + c0 for arrays; returns (roots[n,3], nreal[n])."""
    p = c1 - c2 * c2 / 3.0
    q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    n = c2.size
    roots = np.full((n, 3), np.nan)
    three = disc < 0.0
    if np.any(three):
        pp, qq = p[three], q[three]
        rr = np.sqrt(-pp / 3.0)
        arg = np.clip(3.0 * qq / (2.0 * pp) * np.sqrt(-3.0 / pp), -1.0, 1.0)
        phi = np.arccos(arg)
        for k in range(3):
            roots[three, k] = 2.0 * rr * np.cos(phi / 3.0 - 2.0 * np.pi * k / 3.0) - c2[three] / 3.0
    one = ~three
    if np.any(one):
        qq, dd = q[one], disc[one]
        s = np.sqrt(dd)
        roots[one, 0] = np.cbrt(-qq / 2.0 + s) + np.cbrt(-qq / 2.0 - s) - c2[one] / 3.0
    # Newton polish (two steps) on every real root
    for _ in range(2):
        z = roots
        f = z ** 3 + c2[:, None] * z * z + c1[:, None] * z + c0[:, None]
        fp = 3.0 * z * z + 2.0 * c2[:, None] * z + c1[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(np.isfinite(z) & (np.abs(fp) > 0), f / fp, 0.0)
        roots = z - step
    return roots


class VecCubic:
    """Vectorized SRK/PR saturation solver for one fluid on its temperature grid."""

    def __init__(self, cubic, Tc, pc, omega, T):
        prm = PARAMS[cubic]
        self.cubic, self.Tc, self.pc, self.omega = cubic, float(Tc), float(pc), float(omega)
        self.Oa, self.Ob, self.d1, self.d2, self.Zc = prm["Omega_a"], prm["Omega_b"], prm["d1"], prm["d2"], prm["Zc"]
        self.b = self.Ob * R * self.Tc / self.pc
        self.T = np.asarray(T, dtype=float)
        self.Tr = self.T / self.Tc
        self.ac = self.Oa * (R * self.Tc) ** 2 / self.pc
        # Wilson start, reused as warm start across calls
        self.p_warm = np.clip(self.pc * np.exp(5.372697 * (1.0 + self.omega) * (1.0 - 1.0 / self.Tr)), 1e-3, 0.9995 * self.pc)

    def set_alpha(self, form, theta):
        if form == "soave":
            self.al, self.dal = alpha_soave(self.Tr, theta[0])
        else:
            self.al, self.dal = alpha_twu(self.Tr, *theta)
        self.a = self.ac * self.al
        self.dadT = self.ac * self.dal / self.Tc

    def lnphi(self, Z, A, B):
        return Z - 1.0 - np.log(Z - B) - A / (B * (self.d1 - self.d2)) * np.log((Z + self.d1 * B) / (Z + self.d2 * B))

    def psat(self, tol=1e-11, maxit=60, warm=True):
        T, a = self.T, self.a
        p = self.p_warm.copy() if warm else np.clip(
            self.pc * np.exp(5.372697 * (1.0 + self.omega) * (1.0 - 1.0 / self.Tr)), 1e-3, 0.9995 * self.pc)
        s, q = self.d1 + self.d2, self.d1 * self.d2
        done = np.zeros(T.size, bool)
        Zl = np.full(T.size, np.nan)
        Zv = np.full(T.size, np.nan)
        ok = np.ones(T.size, bool)
        p_lo = np.zeros(T.size)              # bracket: largest pressure known to lie below psat
        p_hi = np.full(T.size, self.pc)      # bracket: smallest pressure known to lie above psat
        for it in range(maxit):
            act = ~done & ok
            if not act.any():
                break
            A = a[act] * p[act] / (R * T[act]) ** 2
            B = self.b * p[act] / (R * T[act])
            c2 = (s - 1.0) * B - 1.0
            c1 = A + q * B * B - s * B * (B + 1.0)
            c0 = -(A * B + q * B * B * (B + 1.0))
            roots = cubic_real_roots(c2, c1, c0)
            valid = np.isfinite(roots) & (roots > (B * (1.0 + 1e-12))[:, None])
            rl = np.where(valid, roots, np.inf).min(axis=1)
            rv = np.where(valid, roots, -np.inf).max(axis=1)
            nval = valid.sum(axis=1)
            idx = np.where(act)[0]
            # no physical root: halve the pressure
            none = nval == 0
            # single root region: nudge as in eos.py
            single = (nval >= 1) & ((rv - rl) < 1e-9 * np.maximum(rv, 1e-12))
            two = (nval >= 1) & ~single
            if none.any():
                jj = idx[none]
                p_hi[jj] = np.minimum(p_hi[jj], p[jj])
                p[jj] = np.where(p_lo[jj] > 0.0, np.sqrt(p_lo[jj] * p[jj]), 0.5 * p[jj])
            if single.any():
                jj = idx[single]
                liq_like = rl[single] < 0.9 * self.Zc
                # liquid-like root: pressure above psat; vapor-like root: pressure below psat
                p_hi[jj] = np.where(liq_like, np.minimum(p_hi[jj], p[jj]), p_hi[jj])
                p_lo[jj] = np.where(liq_like, p_lo[jj], np.maximum(p_lo[jj], p[jj]))
                p_new = np.where(liq_like,
                                 np.where(p_lo[jj] > 0.0, np.sqrt(p_lo[jj] * p[jj]), 0.80 * p[jj]),
                                 np.where(p_hi[jj] < self.pc, np.sqrt(p_hi[jj] * p[jj]), np.minimum(1.25 * p[jj], 0.5 * (p[jj] + self.pc))))
                p[jj] = p_new
            if two.any():
                ii = idx[two]
                lf = self.lnphi(rl[two], A[two], B[two])
                vf = self.lnphi(rv[two], A[two], B[two])
                f = lf - vf                      # ln(phi_l/phi_v); zero at saturation
                ratio = np.exp(f)
                bad = ~np.isfinite(ratio) | (ratio <= 0.0)
                conv = np.abs(ratio - 1.0) < tol
                Zl[ii[conv]] = rl[two][conv]
                Zv[ii[conv]] = rv[two][conv]
                done[ii[conv]] = True
                upd = ~conv & ~bad
                if upd.any():
                    jj = ii[upd]
                    fu = f[upd]
                    pj = p[jj]
                    # bracket update from the sign of f (f > 0: pressure below psat)
                    p_lo[jj] = np.where(fu > 0.0, np.maximum(p_lo[jj], pj), p_lo[jj])
                    p_hi[jj] = np.where(fu > 0.0, p_hi[jj], np.minimum(p_hi[jj], pj))
                    # Newton step in ln p: d(ln phi_l - ln phi_v)/d ln p = Z_l - Z_v
                    dz = rv[two][upd] - rl[two][upd]
                    step = np.where(dz > 1e-14, fu / dz, fu)
                    step = np.clip(step, -5.0, 5.0)
                    p_new = pj * np.exp(step)
                    inside = (p_new > p_lo[jj]) & (p_new < p_hi[jj])
                    p_ss = pj * np.exp(np.clip(fu, -5.0, 5.0))             # successive-substitution step
                    inside_ss = (p_ss > p_lo[jj]) & (p_ss < p_hi[jj])
                    p_bis = np.where(p_lo[jj] > 0.0, np.sqrt(p_lo[jj] * p_hi[jj]), 0.5 * (pj + p_hi[jj]))
                    p[jj] = np.where(inside, p_new, np.where(inside_ss, p_ss, p_bis))
                ok[ii[bad]] = False
            lim = (p < 1e-8) | (p > self.pc * (1.0 + 1e-10))
            ok[lim & ~done] = False
        # looser tolerance for points that ran out of iterations (as eos.py)
        if not done.all():
            act = ~done & ok
            if act.any():
                A = a[act] * p[act] / (R * T[act]) ** 2
                B = self.b * p[act] / (R * T[act])
                c2 = (s - 1.0) * B - 1.0
                c1 = A + q * B * B - s * B * (B + 1.0)
                c0 = -(A * B + q * B * B * (B + 1.0))
                roots = cubic_real_roots(c2, c1, c0)
                valid = np.isfinite(roots) & (roots > (B * (1.0 + 1e-12))[:, None])
                rl = np.where(valid, roots, np.inf).min(axis=1)
                rv = np.where(valid, roots, -np.inf).max(axis=1)
                ratio = np.exp(self.lnphi(rl, A, B) - self.lnphi(rv, A, B))
                conv = np.abs(ratio - 1.0) < 1e-7
                idx = np.where(act)[0]
                Zl[idx[conv]] = rl[conv]
                Zv[idx[conv]] = rv[conv]
                done[idx[conv]] = True
        p_out = np.where(done, p, np.nan)
        if done.all():
            self.p_warm = p.copy()
        return p_out, Zl, Zv

    def h_res(self, p, Z):
        v = Z * R * self.T / p
        log_term = np.log((v + self.d1 * self.b) / (v + self.d2 * self.b))
        return R * self.T * (Z - 1.0) + (self.T * self.dadT - self.a) / (self.b * (self.d1 - self.d2)) * log_term

    def props(self):
        p, Zl, Zv = self.psat()
        h = self.h_res(p, Zv) - self.h_res(p, Zl)
        rho_l = p / (Zl * R * self.T)
        rho_v = p / (Zv * R * self.T)
        return p, h, rho_l, rho_v


# ------------------------------------------------------------------ fitting
def make_objective(vc, form, lnp_ref, kind="ls", eps=1e-7):
    """kind: 'ls' = sum of squared ln-deviations; 'sl1' = smoothed mean |relative deviation| (%);
    'l1' = exact AAD (%) in psat, the statistic reported in the paper."""
    p_ref = np.exp(lnp_ref)

    def F(theta):
        try:
            vc.set_alpha(form, theta)
            p, _, _ = vc.psat()
            if not np.all(np.isfinite(p)):
                p, _, _ = vc.psat(warm=False)
        except FloatingPointError:
            return 1e6
        if not np.all(np.isfinite(p)):
            return 1e6
        if kind == "ls":
            r = np.log(p) - lnp_ref
            return float(np.dot(r, r))
        r = p / p_ref - 1.0
        if kind == "sl1":
            return float(np.mean(np.sqrt(r * r + eps * eps)) * 100.0)
        return float(np.mean(np.abs(r)) * 100.0)
    return F


def fit_soave(vc, lnp_ref, m0, kind="l1"):
    F = make_objective(vc, "soave", lnp_ref, kind=kind)
    # coarse scan then bounded Brent inside the bracket around the scan minimum
    grid = np.linspace(0.1, 2.5, 49)
    vals = [F([m]) for m in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda m: F([m]), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-10, maxiter=500))
    # guard against a non-unimodal bracket: a fine local scan around the result
    mm = np.linspace(lo, hi, 201)
    vv = [F([m]) for m in mm]
    j = int(np.argmin(vv))
    if vv[j] < res.fun:
        res2 = minimize_scalar(lambda m: F([m]), bounds=(mm[max(j - 1, 0)], mm[min(j + 1, 200)]), method="bounded",
                               options=dict(xatol=1e-10, maxiter=500))
        if res2.fun < res.fun:
            res = res2
    return float(res.x), float(res.fun), vals, grid


def feasible(theta, tolc=0.0):
    p2, q3, d1 = consistency_margins(*theta)
    return min(p2, q3, d1) >= -tolc


def fit_twu(vc, lnp_ref, starts, constrained=True):
    """Returns (best, records): best = (theta, AAD_l1, start name, AAD at the LS optimum of the same start)."""
    F_ls = make_objective(vc, "twu", lnp_ref, kind="ls")
    F_sl1 = make_objective(vc, "twu", lnp_ref, kind="sl1")
    F_l1 = make_objective(vc, "twu", lnp_ref, kind="l1")
    cons = []
    if constrained:
        cons = [dict(type="ineq", fun=lambda th: consistency_margins(*th)[0]),
                dict(type="ineq", fun=lambda th: consistency_margins(*th)[1]),
                dict(type="ineq", fun=lambda th: consistency_margins(*th)[2])]
    lo = np.array([b[0] for b in BOUNDS_TWU])
    hi = np.array([b[1] for b in BOUNDS_TWU])

    def pen(th):
        """Exact AAD with a feasibility wall (for the Nelder-Mead polish)."""
        if np.any(th < lo) or np.any(th > hi):
            return 1e6
        if constrained:
            p2, q3, d1 = consistency_margins(*th)
            if min(p2, q3, d1) < 0.0:
                return 1e6
        return F_l1(th)

    records = []
    best = None
    for name, th0 in starts:
        th0 = np.clip(np.asarray(th0, dtype=float), lo, hi)
        if constrained and not feasible(th0, 1e-9):
            for frac in np.linspace(0.0, 1.0, 41):
                cand = th0 * (1 - frac) + np.array([0.3, 0.85, 2.0]) * frac
                if feasible(cand, 1e-9):
                    th0 = cand
                    break
        vc.p_warm = np.clip(vc.pc * np.exp(5.372697 * (1.0 + vc.omega) * (1.0 - 1.0 / vc.Tr)), 1e-3, 0.9995 * vc.pc)
        rec = dict(start=name, L0=th0[0], M0=th0[1], N0=th0[2])
        try:
            if F_ls(th0) >= 1e6:
                rec.update(fun=np.nan, success=False, err="solver fails at the start point")
                records.append(rec)
                continue
            # stage 1: least squares in ln p (smooth)
            r1 = minimize(F_ls, th0, method="SLSQP", bounds=BOUNDS_TWU, constraints=cons,
                          options=dict(maxiter=250, ftol=1e-14, eps=1e-7))
            th = r1.x if (np.isfinite(r1.fun) and r1.fun < 1e6 and (not constrained or feasible(r1.x, 1e-8))) else th0
            rec.update(L_ls=th[0], M_ls=th[1], N_ls=th[2], F_ls=float(F_ls(th)), aad_at_ls=float(F_l1(th)))
            if rec["aad_at_ls"] >= 1e6 or (best is not None and rec["aad_at_ls"] > 5.0 * best[1] + 0.5):
                rec.update(fun=rec["aad_at_ls"], success=False, feasible=False, err="poor least-squares stage; start discarded")
                records.append(rec)
                continue
            # stage 2: smoothed L1 (the paper's statistic), SLSQP with the consistency constraints
            r2 = minimize(F_sl1, th, method="SLSQP", bounds=BOUNDS_TWU, constraints=cons,
                          options=dict(maxiter=200, ftol=1e-13, eps=1e-7))
            if np.isfinite(r2.fun) and (not constrained or feasible(r2.x, 1e-8)) and F_l1(r2.x) <= F_l1(th) + 1e-12:
                th = r2.x
            # stage 3: exact L1 polish by Nelder-Mead with a feasibility wall (small initial simplex)
            r3 = minimize(pen, th, method="Nelder-Mead",
                          options=dict(xatol=1e-8, fatol=1e-9, maxiter=1200, maxfev=1500,
                                       initial_simplex=np.array([th, th + [0.01, 0, 0], th + [0, 0.005, 0], th + [0, 0, 0.02]])))
            if np.isfinite(r3.fun) and r3.fun < F_l1(th) and (not constrained or feasible(r3.x, 0.0)):
                th = r3.x
            fun = float(F_l1(th))
        except Exception as e:  # noqa: BLE001
            rec.update(fun=np.nan, success=False, err=str(e))
            records.append(rec)
            continue
        feas = (not constrained) or feasible(th, 1e-8)
        rec.update(L=th[0], M=th[1], N=th[2], fun=fun, success=True, feasible=feas)
        records.append(rec)
        if feas and np.isfinite(fun) and (best is None or fun < best[1]):
            best = (np.array(th, dtype=float), fun, name, rec["aad_at_ls"])
    return best, records


def main():
    t0 = time.time()
    fluids = list(fl.index)
    if QUICK:
        fluids = ["propane", "methanol", "water", "nitrogen"]
    rng = np.random.default_rng(20260714)
    rows, srows = [], []
    # ---- verification of the vectorized solver against eos.py (committed numbers)
    maxdiff = {"p": 0.0, "h": 0.0, "rho": 0.0, "rhov": 0.0}
    for fluid in fluids:
        g = df[df.fluid == fluid].sort_values("Tr")
        Tc, pc, w = fl.loc[fluid, "Tc_K"], fl.loc[fluid, "pc_MPa"] * 1e6, fl.loc[fluid, "omega"]
        for cubic in ("SRK", "PR"):
            cl = cubic.lower()
            vc = VecCubic(cubic, Tc, pc, w, g.T_K.to_numpy())
            for form, theta, tag in (("soave", [M_SOAVE[(cubic, "s")](w)], f"{cl}_s"),
                                     ("twu", [tf.loc[fluid, f"L_{cubic}"], tf.loc[fluid, f"M_{cubic}"], tf.loc[fluid, f"N_{cubic}"]], f"{cl}_tf")):
                vc.set_alpha(form, theta)
                p, h, rl, rv = vc.props()
                maxdiff["p"] = max(maxdiff["p"], float(np.max(np.abs(p / g[f"p_{tag}_Pa"].to_numpy() - 1.0))))
                maxdiff["h"] = max(maxdiff["h"], float(np.max(np.abs(h / g[f"hvap_{tag}_Jmol"].to_numpy() - 1.0))))
                maxdiff["rho"] = max(maxdiff["rho"], float(np.max(np.abs(rl / g[f"rho_{tag}_molm3"].to_numpy() - 1.0))))
                rv_ref = g.rhov_ref_molm3.to_numpy() * (1.0 + g[f"dev_rhov_{tag}"].to_numpy() / 100.0)
                maxdiff["rhov"] = max(maxdiff["rhov"], float(np.max(np.abs(rv / rv_ref - 1.0))))
    print("vectorized solver vs eos.py (relative):", {k: f"{v:.1e}" for k, v in maxdiff.items()})
    assert max(maxdiff.values()) < 1e-8

    # ---- fits
    for fluid in fluids:
        g = df[df.fluid == fluid].sort_values("Tr")
        Tc, pc, w = fl.loc[fluid, "Tc_K"], fl.loc[fluid, "pc_MPa"] * 1e6, fl.loc[fluid, "omega"]
        lnp_ref = np.log(g.p_ref_Pa.to_numpy())
        for cubic in ("SRK", "PR"):
            cl = cubic.lower()
            vc = VecCubic(cubic, Tc, pc, w, g.T_K.to_numpy())
            rec = dict(fluid=fluid, cubic=cubic, omega=w)
            # (a) Soave, one parameter
            m0 = M_SOAVE[(cubic, "s")](w)
            m_opt, f_opt, vals, grid = fit_soave(vc, lnp_ref, m0)
            m_ls, _, _, _ = fit_soave(vc, lnp_ref, m0, kind="ls")
            rec.update(m_gen=m0, m_fit=m_opt, aad_soave_fit=f_opt, aad_soave_gen=make_objective(vc, "soave", lnp_ref, "l1")([m0]),
                       m_fit_ls=m_ls, aad_soave_at_ls=make_objective(vc, "soave", lnp_ref, "l1")([m_ls]),
                       soave_scan_unimodal=bool(np.sum(np.diff(np.sign(np.diff(vals))) != 0) <= 1))
            # (b) consistent Twu-91, three parameters, multi-start
            gcoef = TWU_CONSISTENT_GENERALIZED[cubic]
            Lg = gcoef["L"][0] + gcoef["L"][1] * w + gcoef["L"][2] * w * w
            Mg = gcoef["M"][0] + gcoef["M"][1] * w + gcoef["M"][2] * w * w
            th_tf = [tf.loc[fluid, f"L_{cubic}"], tf.loc[fluid, f"M_{cubic}"], tf.loc[fluid, f"N_{cubic}"]]
            starts = [("twu-fit [8]", th_tf), ("twu-c generalized", [Lg, Mg, 2.0])]
            n_rand = 2 if QUICK else 4
            for k in range(n_rand):
                for _ in range(200):
                    cand = [rng.uniform(0.05, 1.2), rng.uniform(0.75, 1.0), rng.uniform(0.6, 3.5)]
                    if feasible(cand, 0.0):
                        break
                starts.append((f"random {k + 1}", cand))
            best, recs = fit_twu(vc, lnp_ref, starts, constrained=True)
            for r_ in recs:
                srows.append(dict(fluid=fluid, cubic=cubic, constrained=True, **r_))
            Lf, Mf, Nf = best[0]
            p2, q3, d1 = consistency_margins(Lf, Mf, Nf)
            funs = np.array([r_["fun"] for r_ in recs if r_.get("feasible", False) and np.isfinite(r_["fun"])])
            Ls = np.array([[r_["L"], r_["M"], r_["N"]] for r_ in recs if r_.get("feasible", False) and np.isfinite(r_["fun"])
                           and r_["fun"] <= best[1] + 0.001])
            F_l1 = make_objective(vc, "twu", lnp_ref, "l1")
            rec.update(L_fit=Lf, M_fit=Mf, N_fit=Nf, aad_twu_fit=best[1], best_start=best[2], aad_twu_at_ls=best[3],
                       n_starts=len(recs), n_starts_within_0p001=int(np.sum(funs <= best[1] + 0.001)),
                       n_starts_within_0p01=int(np.sum(funs <= best[1] + 0.01)),
                       spread_L=float(np.ptp(Ls[:, 0])) if len(Ls) else np.nan,
                       spread_M=float(np.ptp(Ls[:, 1])) if len(Ls) else np.nan,
                       spread_N=float(np.ptp(Ls[:, 2])) if len(Ls) else np.nan,
                       margin_P2=p2, margin_Q3=q3, margin_d1=d1,
                       active_P2=bool(p2 < 1e-5), active_Q3=bool(q3 < 1e-5), active_d1=bool(d1 < 1e-5),
                       active_M1=bool(Mf > 1.0 - 1e-5), active_L0=bool(Lf < 1e-5),
                       L_tf=th_tf[0], M_tf=th_tf[1], N_tf=th_tf[2],
                       aad_twu_tf=F_l1(th_tf), aad_twu_gen=F_l1([Lg, Mg, 2.0]), L_gen=Lg, M_gen=Mg)
            # unconstrained diagnostic (two starts: the constrained optimum and the twu-fit set)
            bestu, recsu = fit_twu(vc, lnp_ref, [("constrained optimum", [Lf, Mf, Nf])], constrained=False)
            for r_ in recsu:
                srows.append(dict(fluid=fluid, cubic=cubic, constrained=False, **r_))
            if bestu is not None:
                rec.update(L_unc=bestu[0][0], M_unc=bestu[0][1], N_unc=bestu[0][2], aad_twu_unc=bestu[1],
                           unc_consistent=bool(feasible(bestu[0], 1e-8)))
            # AADs at the optimum (vectorized solver; the final table is recomputed with eos.py)
            for form, theta, lab in (("soave", [m_opt], "sfit"), ("twu", [Lf, Mf, Nf], "tfit")):
                vc.set_alpha(form, theta)
                p, h, rl, rv = vc.props()
                rec[f"psat_aad_{lab}"] = float(np.mean(np.abs(p / g.p_ref_Pa.to_numpy() - 1.0)) * 100)
                rec[f"h_aad_{lab}"] = float(np.mean(np.abs(h / g.hvap_ref_Jmol.to_numpy() - 1.0)) * 100)
            rows.append(rec)
            print(f"{fluid:18s} {cubic:3s} m {m0:.4f}->{m_opt:.4f} psat {rec['aad_soave_gen']:.3f}->{rec['psat_aad_sfit']:.3f}% | Twu "
                  f"L {Lf:.4f} M {Mf:.4f} N {Nf:.4f} psat {rec['aad_twu_tf']:.3f}[8] -> {rec['psat_aad_tfit']:.3f}% (LS-opt {rec['aad_twu_at_ls']:.3f}, unc {rec.get('aad_twu_unc', np.nan):.3f}) "
                  f"act P2 {rec['active_P2']} Q3 {rec['active_Q3']} M1 {rec['active_M1']} starts {rec['n_starts_within_0p001']}/{rec['n_starts']} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "insample_fit.csv" if not QUICK else "insample_fit_quick.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(DATA, "insample_fit_starts.csv" if not QUICK else "insample_fit_starts_quick.csv"), index=False)
    json.dump(dict(solver_check=maxdiff, elapsed_s=time.time() - t0, versions=VERSIONS),
              open(os.path.join(DATA, "insample_fit_meta.json" if not QUICK else "insample_fit_meta_quick.json"), "w"), indent=1)
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
