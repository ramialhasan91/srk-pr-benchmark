"""
Generic two-parameter cubic equation of state (SRK and PR) for pure fluids,
with an extensible set of alpha functions.

Cubic forms
-----------
  - Soave-Redlich-Kwong (SRK): Soave, Chem. Eng. Sci. 27 (1972) 1197-1203.
  - Peng-Robinson (PR): Peng & Robinson, Ind. Eng. Chem. Fundam. 15 (1976) 59-64.

The generic pressure-explicit form is
    p = RT/(v - b) - a(T) / [(v + d1*b)(v + d2*b)]
with (d1, d2) = (1, 0) for SRK and (1+sqrt(2), 1-sqrt(2)) for PR.

Alpha functions (all analytic in alpha and d(alpha)/dTr)
--------------------------------------------------------
  "s"    Classical Soave form, alpha = [1 + m(1 - sqrt(Tr))]^2, with the
         original m(omega) of each model (Soave 1972 for SRK, Peng-Robinson
         1976 for PR).
  "s19"  Same Soave form with the updated generalized m(omega) of
         Pina-Martinez, Privat, Jaubert & Peng, Fluid Phase Equilib. 485 (2019)
         264-269 (cubic polynomials fitted to 1721 compounds).
  "t95"  Generalized Twu alpha function, alpha = alpha0 + omega*(alpha1 - alpha0),
         alpha_i = Tr^(N_i(M_i-1)) exp[L_i(1 - Tr^(N_i M_i))]:
         Twu, Coon & Cunningham, Fluid Phase Equilib. 105 (1995) 49-59 (PR)
         and 61-69 (RK). Subcritical parameter sets only (grid is Tr <= 0.985).
  "tc"   Generalized *consistent* Twu-91 alpha function of the Nancy group,
         alpha = Tr^(N(M-1)) exp[L(1 - Tr^(N M))] with N = 2 and L, M quadratic
         in omega:
           PR : L = 0.0544 + 0.7536 w + 0.0297 w^2, M = 0.8678 - 0.1785 w + 0.1401 w^2
                (Pina-Martinez, Privat & Jaubert, AIChE J. 68 (2022) e17518,
                 as quoted by Privat & Jaubert, Fluid Phase Equilib. 567 (2023) 113697)
           SRK: L = 0.1359 + 0.7535 w + 0.0611 w^2, M = 0.8787 - 0.2063 w + 0.1709 w^2
                (Pina-Martinez, Le Guennec, Privat, Jaubert & Mathias,
                 J. Chem. Eng. Data 63 (2018) 3980-3988)
  "coq"  Exponential alpha function of Coquelet, Chapoy & Richon (Int. J.
         Thermophys. 25 (2004) 133-158),
         alpha = exp[C1(1 - Tr)] [1 + C2(1 - sqrt(Tr))^2 + C3(1 - sqrt(Tr))^3]^2,
         with the generalized C1..C3(omega) for PR and SRK as compiled by
         Xiao & Yang, ACS Omega 10 (2025) 29021-29036 (Table 2, Eqs. 24-31),
         after Coquelet et al. (2004) and Mahmoodi & Sedigh (2017).
  "ms"   Consistent generalized alpha function of Mahmoodi & Sedigh (Fluid
         Phase Equilib. 436 (2017) 69-84),
         alpha = exp{2 C1 u - (C2 u)^2 + (2/3)(C3 u)^3}, u = 1 - sqrt(Tr),
         with C1..C3(omega) for PR and SRK as compiled by Xiao & Yang (2025),
         Table 2, Eqs. 32-39.
  "tf"   Component-specific consistent Twu-91 parameters (L, M, N) taken from
         the 1800-fluid compilation of Pina-Martinez et al. (2022), Supporting
         Information Table S3; supplied per fluid through `twu_params`. Not
         predictive: used only as a fitted reference.

Also implemented: fugacity coefficients, saturation-pressure solver,
analytic residual enthalpy / enthalpy of vaporization, and a quadrature check
of the volume-translation identity ln(phi_t) = ln(phi) - c p/(RT)
(Jaubert et al., Fluid Phase Equilib. 419 (2016) 88-95).

All quantities in SI units (Pa, K, m^3/mol, J/mol, J/mol/K).
"""

import numpy as np

R = 8.314462618  # J / (mol K), CODATA 2018

# Subcritical universal constants of the generalized Twu (1995) alpha
# function: (L, M, N) for alpha0 and alpha1, from Part 1 (PR) and Part 2 (RK).
TWU95_SUBCRITICAL = {
    "SRK": ((0.141599, 0.919422, 2.496441), (0.500315, 0.799457, 3.291790)),
    "PR": ((0.125283, 0.911807, 1.948150), (0.511614, 0.784054, 2.812520)),
}

# Generalized consistent Twu-91 correlations (N = 2): coefficients of
# L = l0 + l1 w + l2 w^2 and M = m0 + m1 w + m2 w^2.
TWU_CONSISTENT_GENERALIZED = {
    "PR": dict(L=(0.0544, 0.7536, 0.0297), M=(0.8678, -0.1785, 0.1401), N=2.0,
               source="Pina-Martinez et al. 2022 (AIChE J. 68, e17518)"),
    "SRK": dict(L=(0.1359, 0.7535, 0.0611), M=(0.8787, -0.2063, 0.1709), N=2.0,
                source="Pina-Martinez et al. 2018 (J. Chem. Eng. Data 63, 3980)"),
}

# Soave-form m(omega) correlations.
M_SOAVE = {
    ("SRK", "s"): lambda w: 0.480 + 1.574 * w - 0.176 * w * w,                 # Soave 1972
    ("PR", "s"): lambda w: 0.37464 + 1.54226 * w - 0.26992 * w * w,             # Peng-Robinson 1976
    ("SRK", "s19"): lambda w: 0.4810 + 1.5963 * w - 0.2963 * w**2 + 0.1223 * w**3,  # Pina-Martinez 2019
    ("PR", "s19"): lambda w: 0.3919 + 1.4996 * w - 0.2721 * w**2 + 0.1063 * w**3,   # Pina-Martinez 2019
}

# Generalized coefficients of the Coquelet (2004) and Mahmoodi-Sedigh (2017)
# alpha functions, quadratic in omega (C3 of MS is a Gaussian in omega), as
# compiled in Table 2 of Xiao & Yang, ACS Omega 10 (2025) 29021-29036.
COQUELET_GENERALIZED = {
    "PR": dict(C1=(0.40464, 1.3361, -0.07987), C2=(-0.08139, 0.96493, -0.85454), C3=(0.31953, 0.001007, -0.88858)),
    "SRK": dict(C1=(0.53591, 1.4492, -0.13969), C2=(-0.2741, 0.59006, -0.54412), C3=(0.54293, 0.53258, -1.4927)),
}
MS_GENERALIZED = {
    "PR": dict(C1=(0.36818, 1.4801, -0.14407), C2=(0.19422, 1.9061, -0.46577), C3=(1.514, 1.6645, 1.5474)),
    "SRK": dict(C1=(0.47941, 1.673, -0.23356), C2=(0.53827, 2.1529, -0.7143), C3=(1.7482, 1.456, 1.4542)),
}

ALPHA_LABELS = {
    "s": "Soave", "s19": "Soave-19", "t95": "Twu-95", "tc": "Twu-c", "coq": "Coquelet", "ms": "MS", "tf": "Twu-fit",
}


def _quad(c, w):
    return c[0] + c[1] * w + c[2] * w * w


class CubicEOS:
    """Pure-component cubic EoS (SRK or PR) with a selectable alpha function."""

    PARAMS = {
        "SRK": dict(Omega_a=0.42748, Omega_b=0.08664, d1=1.0, d2=0.0, Zc=1.0 / 3.0),
        "PR": dict(Omega_a=0.45724, Omega_b=0.07780, d1=1.0 + np.sqrt(2.0),
                   d2=1.0 - np.sqrt(2.0), Zc=0.3074),
    }

    def __init__(self, model, Tc, pc, omega, alpha="s", twu_params=None):
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
        self.b = self.Ob * R * self.Tc / self.pc
        w = self.omega
        if alpha in ("s", "s19"):
            self.m = M_SOAVE[(model, alpha)](w)
        elif alpha == "t95":
            self.twu0, self.twu1 = TWU95_SUBCRITICAL[model]
        elif alpha == "tc":
            g = TWU_CONSISTENT_GENERALIZED[model]
            L = g["L"][0] + g["L"][1] * w + g["L"][2] * w * w
            M = g["M"][0] + g["M"][1] * w + g["M"][2] * w * w
            self.twu_single = (L, M, g["N"])
        elif alpha == "tf":
            if twu_params is None:
                raise ValueError("alpha='tf' requires twu_params=(L, M, N)")
            self.twu_single = tuple(float(x) for x in twu_params)
        elif alpha == "coq":
            g = COQUELET_GENERALIZED[model]
            self.C = (_quad(g["C1"], w), _quad(g["C2"], w), _quad(g["C3"], w))
        elif alpha == "ms":
            g = MS_GENERALIZED[model]
            a0, w0, sw = g["C3"]
            self.C = (_quad(g["C1"], w), _quad(g["C2"], w), a0 * np.exp(-((w - w0) / sw) ** 4))
        else:
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

    def _alpha_twu95(self, Tr):
        a0, d0 = self._twu_branch(Tr, *self.twu0)
        a1, d1 = self._twu_branch(Tr, *self.twu1)
        return a0 + self.omega * (a1 - a0), d0 + self.omega * (d1 - d0)

    def _alpha_coquelet(self, Tr):
        C1, C2, C3 = self.C
        root = np.sqrt(Tr)
        u = 1.0 - root
        E = np.exp(C1 * (1.0 - Tr))
        B = 1.0 + C2 * u * u + C3 * u ** 3
        dB = (2.0 * C2 * u + 3.0 * C3 * u * u) * (-0.5 / root)
        al = E * B * B
        return al, -C1 * al + E * 2.0 * B * dB

    def _alpha_ms(self, Tr):
        C1, C2, C3 = self.C
        root = np.sqrt(Tr)
        u = 1.0 - root
        f = 2.0 * C1 * u - (C2 * u) ** 2 + (2.0 / 3.0) * (C3 * u) ** 3
        df = (2.0 * C1 - 2.0 * C2 * C2 * u + 2.0 * C3 ** 3 * u * u) * (-0.5 / root)
        al = np.exp(f)
        return al, al * df

    def alpha(self, T):
        """Return (alpha, d alpha / d Tr) at temperature T."""
        Tr = T / self.Tc
        if self.alpha_model in ("s", "s19"):
            return self._alpha_soave(Tr)
        if self.alpha_model == "t95":
            return self._alpha_twu95(Tr)
        if self.alpha_model == "coq":
            return self._alpha_coquelet(Tr)
        if self.alpha_model == "ms":
            return self._alpha_ms(Tr)
        return self._twu_branch(Tr, *self.twu_single)

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
