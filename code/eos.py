"""
Generic two-parameter cubic equation of state (SRK and PR) for pure fluids.

Implements:
  - Soave-Redlich-Kwong (SRK): Soave, Chem. Eng. Sci. 27 (1972) 1197-1203.
  - Peng-Robinson (PR): Peng & Robinson, Ind. Eng. Chem. Fundam. 15 (1976) 59-64.
  - Classical Soave-type alpha function (alpha="soave") and the generalized
    Twu alpha function (alpha="twu"):
      Twu, Coon & Cunningham, Fluid Phase Equilib. 105 (1995) 49-59 (PR) and
      61-69 (RK/SRK). Only the subcritical parameter sets are used; the
      benchmark grid is entirely subcritical (Tr <= 0.985).
  - Analytic residual enthalpy and enthalpy of vaporization.
  - Constant (Peneloux-type) volume translation applied a posteriori, plus a
    numerical (quadrature) verification of the identity
      ln(phi_translated) = ln(phi) - c*p/(R*T),
    from which the invariance of psat, Delta h_vap, and cp under constant
    translation follows (Jaubert et al., Fluid Phase Equilib. 419 (2016) 88-95).

The generic pressure-explicit form is
    p = RT/(v - b) - a(T) / [(v + d1*b)(v + d2*b)]
with (d1, d2) = (1, 0) for SRK and (1+sqrt(2), 1-sqrt(2)) for PR.

All quantities in SI units (Pa, K, m^3/mol, J/mol, J/mol/K).
"""

import numpy as np

R = 8.314462618  # J / (mol K), CODATA 2018

# Subcritical (Tr <= 1) universal constants of the generalized Twu (1995)
# alpha function, alpha = alpha0 + omega*(alpha1 - alpha0) with
# alpha_i = Tr^(N(M-1)) * exp[L*(1 - Tr^(N*M))].
# (L, M, N) for alpha0 and alpha1; values from the original Part 1 (PR) and
# Part 2 (RK) papers.
TWU95_SUBCRITICAL = {
    "SRK": ((0.141599, 0.919422, 2.496441), (0.500315, 0.799457, 3.291790)),
    "PR": ((0.125283, 0.911807, 1.948150), (0.511614, 0.784054, 2.812520)),
}


class CubicEOS:
    """Pure-component cubic EoS with Soave or generalized-Twu alpha function."""

    PARAMS = {
        "SRK": dict(
            Omega_a=0.42748,
            Omega_b=0.08664,
            d1=1.0,
            d2=0.0,
            Zc=1.0 / 3.0,
            m=lambda w: 0.480 + 1.574 * w - 0.176 * w * w,
        ),
        "PR": dict(
            Omega_a=0.45724,
            Omega_b=0.07780,
            d1=1.0 + np.sqrt(2.0),
            d2=1.0 - np.sqrt(2.0),
            Zc=0.3074,
            m=lambda w: 0.37464 + 1.54226 * w - 0.26992 * w * w,
        ),
    }

    def __init__(self, model, Tc, pc, omega, alpha="soave"):
        prm = self.PARAMS[model]
        self.model = model
        self.alpha_model = alpha
        self.Tc = float(Tc)
        self.pc = float(pc)
        self.omega = float(omega)
        self.Oa = prm["Omega_a"]
        self.Ob = prm["Omega_b"]
        self.d1 = prm["d1"]
        self.d2 = prm["d2"]
        self.Zc = prm["Zc"]
        self.m = prm["m"](self.omega)
        self.b = self.Ob * R * self.Tc / self.pc
        if alpha == "twu":
            self.twu0, self.twu1 = TWU95_SUBCRITICAL[model]
        elif alpha != "soave":
            raise ValueError(f"unknown alpha model {alpha!r}")

    # ------------------------------------------------------------------ #
    # alpha(Tr) and d(alpha)/d(Tr)
    def _alpha_soave(self, Tr):
        root = np.sqrt(Tr)
        f = 1.0 + self.m * (1.0 - root)
        return f * f, -self.m * f / root

    @staticmethod
    def _twu_branch(Tr, L, M, N):
        al = Tr ** (N * (M - 1.0)) * np.exp(L * (1.0 - Tr ** (N * M)))
        dlnal_dTr = N * (M - 1.0) / Tr - L * M * N * Tr ** (N * M - 1.0)
        return al, al * dlnal_dTr

    def _alpha_twu(self, Tr):
        a0, d0 = self._twu_branch(Tr, *self.twu0)
        a1, d1 = self._twu_branch(Tr, *self.twu1)
        return a0 + self.omega * (a1 - a0), d0 + self.omega * (d1 - d0)

    def alpha(self, T):
        """Return (alpha, d alpha / d Tr) at temperature T."""
        Tr = T / self.Tc
        if self.alpha_model == "twu":
            return self._alpha_twu(Tr)
        return self._alpha_soave(Tr)

    # ------------------------------------------------------------------ #
    def a(self, T):
        al, _ = self.alpha(T)
        return self.Oa * (R * self.Tc) ** 2 / self.pc * al

    def dadT(self, T):
        _, dal_dTr = self.alpha(T)
        return self.Oa * (R * self.Tc) ** 2 / self.pc * dal_dTr / self.Tc

    def AB(self, T, p):
        A = self.a(T) * p / (R * T) ** 2
        B = self.b * p / (R * T)
        return A, B

    def Z_roots(self, T, p):
        """Real, physical (Z > B) compressibility-factor roots, sorted ascending."""
        A, B = self.AB(T, p)
        s = self.d1 + self.d2
        q = self.d1 * self.d2
        c2 = (s - 1.0) * B - 1.0
        c1 = A + q * B * B - s * B * (B + 1.0)
        c0 = -(A * B + q * B * B * (B + 1.0))
        r = np.roots([1.0, c2, c1, c0])
        r = r[np.abs(r.imag) < 1e-10].real
        r = r[r > B * (1.0 + 1e-12)]
        return np.sort(r), A, B

    def lnphi(self, Z, A, B):
        """Fugacity coefficient of a pure fluid for the generic cubic form."""
        return (
            Z
            - 1.0
            - np.log(Z - B)
            - A / (B * (self.d1 - self.d2)) * np.log((Z + self.d1 * B) / (Z + self.d2 * B))
        )

    # ------------------------------------------------------------------ #
    def psat(self, T, tol=1e-11, maxit=400):
        """
        Saturation pressure at T (< Tc) by successive substitution on the
        fugacity ratio, initialized with the Wilson correlation.

        Returns (p_sat, Z_liq, Z_vap); NaNs on failure.
        """
        if not (0.0 < T < self.Tc * (1.0 - 1e-12)):
            return np.nan, np.nan, np.nan

        # Wilson (1968-type) initial estimate
        p = self.pc * np.exp(5.372697 * (1.0 + self.omega) * (1.0 - self.Tc / T))
        p = min(max(p, 1e-3), 0.9995 * self.pc)

        for _ in range(maxit):
            Z, A, B = self.Z_roots(T, p)
            if Z.size == 0:
                p *= 0.5
                if p < 1e-8:
                    return np.nan, np.nan, np.nan
                continue
            Zl, Zv = Z[0], Z[-1]
            if (Zv - Zl) < 1e-9 * max(Zv, 1e-12):
                # Single-root region: nudge pressure back toward two-phase window.
                if Zl < 0.9 * self.Zc:      # liquid-like root -> p above psat
                    p *= 0.80
                else:                        # vapor-like root  -> p below psat
                    p *= 1.25
                if not (1e-8 < p < self.pc):
                    return np.nan, np.nan, np.nan
                continue
            ratio = np.exp(self.lnphi(Zl, A, B) - self.lnphi(Zv, A, B))
            if not np.isfinite(ratio) or ratio <= 0.0:
                return np.nan, np.nan, np.nan
            if abs(ratio - 1.0) < tol:
                return p, Zl, Zv
            p *= ratio
            if not (1e-8 < p < self.pc * (1.0 + 1e-10)):
                return np.nan, np.nan, np.nan
        # Accept a slightly looser tolerance if the loop ran out near Tc.
        if abs(ratio - 1.0) < 1e-7:
            return p, Zl, Zv
        return np.nan, np.nan, np.nan

    # ------------------------------------------------------------------ #
    def sat_liq_volume(self, T):
        """Saturated liquid molar volume (m^3/mol) at the EoS's own psat(T)."""
        p, Zl, _ = self.psat(T)
        if not np.isfinite(p):
            return np.nan, np.nan
        return Zl * R * T / p, p

    # ------------------------------------------------------------------ #
    def h_res(self, T, p, Z):
        """
        Residual (departure) molar enthalpy h - h_ideal at (T, p) for the
        phase with compressibility factor Z:
          h_res = RT(Z - 1) + [T a'(T) - a(T)] / [b (d1 - d2)]
                  * ln[(v + d1 b)/(v + d2 b)].
        """
        v = Z * R * T / p
        aT = self.a(T)
        daT = self.dadT(T)
        log_term = np.log((v + self.d1 * self.b) / (v + self.d2 * self.b))
        return R * T * (Z - 1.0) + (T * daT - aT) / (self.b * (self.d1 - self.d2)) * log_term

    def hvap(self, T):
        """
        Enthalpy of vaporization at the EoS's own psat(T):
          Delta h_vap = h_res(vapor) - h_res(liquid)
        (the ideal-gas contributions cancel at equal T).
        Returns (hvap [J/mol], psat [Pa]); NaNs on failure.
        """
        p, Zl, Zv = self.psat(T)
        if not np.isfinite(p):
            return np.nan, np.nan
        return self.h_res(T, p, Zv) - self.h_res(T, p, Zl), p

    # ------------------------------------------------------------------ #
    def p_of_v(self, T, v):
        """Pressure from the pressure-explicit form (untranslated)."""
        return R * T / (v - self.b) - self.a(T) / ((v + self.d1 * self.b) * (v + self.d2 * self.b))

    def lnphi_translated_quadrature(self, T, p, Z, c, n_gauss=200):
        """
        Fugacity coefficient of the c-TRANSLATED EoS at (T, p), for the phase
        whose UNTRANSLATED compressibility factor is Z, evaluated by direct
        Gauss-Legendre quadrature of the exact volume-space departure integral
            ln(phi) = Zt - 1 - ln(Zt) + (1/RT) * Int_{vt}^{inf} [p(T,v') - RT/v'] dv',
        where vt = v - c and the translated pressure is p_t(T, v') = p(T, v' + c).

        Used only to verify the analytic identity
            ln(phi_translated) = ln(phi) - c*p/(R*T)
        (and hence the invariance of psat and Delta h_vap under constant
        translation); it plays no role in the benchmark itself.
        """
        v = Z * R * T / p
        vt = v - c
        Zt = p * vt / (R * T)
        # substitute u = 1/v': integral over u in (0, 1/vt]
        u_hi = 1.0 / vt
        nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
        u = 0.5 * u_hi * (nodes + 1.0)
        w = 0.5 * u_hi * weights
        vprime = 1.0 / u
        integrand = (self.p_of_v(T, vprime + c) - R * T * u) / (u * u)
        integral = np.sum(w * integrand)
        return Zt - 1.0 - np.log(Zt) + integral / (R * T)
