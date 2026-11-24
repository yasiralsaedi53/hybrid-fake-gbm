"""Explicit finite-difference solver for the Black--Scholes calibration equation (Section 7).

In the variables tau = log t and z = (x - m(t)) / (sigma sqrt t), m(t) = -sigma^2 t / 2, the
conditional probability Pi(tau, z) of the high-volatility regime solves equation (8):

    Pi_tau = L[Pi] + (sigma sqrt(t) / 2) B[Pi] + t [q12 (1 - Pi) - q21 Pi],
    L[Pi]  = G_zz/2 - z G_z + (z/2) Pi_z + (z^2 - 1)(G - Pi)/2,
    B[Pi]  = G_z - Pi_z - z (G - Pi),          G = G(Pi) = b Pi / (a + (b - a) Pi).

The scheme uses central differences on [-Z, Z], zero-flux boundaries and
dtau = cfl * dz^2 / kappa^2.  It starts from the closed-form profile of Theorem 5.1 at t0.
"""
import numpy as np
from .profile import profile, G as Gfun


def _d1(u, dz):
    d = np.empty_like(u)
    d[1:-1] = (u[2:] - u[:-2]) / (2 * dz)
    d[0] = d[-1] = 0.0
    return d


def _d2(u, dz):
    d = np.empty_like(u)
    d[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dz ** 2
    d[0] = 2 * (u[1] - u[0]) / dz ** 2
    d[-1] = 2 * (u[-2] - u[-1]) / dz ** 2
    return d


def solve(f1, f2, sigma, q12=0.0, q21=0.0, pi0=0.5, T=1.0, t0=1e-4, Z=7.0, dz=0.025,
          cfl=0.35, store_t=None, drift=True, switching=True, diagnostics=False):
    """Solve (8) from t0 to T.

    Returns a dict with z, Pi at T, stored profiles {t: Pi} for each t in store_t
    (nearest step at or after t), the closed-form initial profile and, if requested,
    diagnostics (min/max before clipping, max diffusion, Peclet and reaction numbers).
    """
    a, b = f1 ** 2, f2 ** 2
    kappa = f2 / f1
    z = np.arange(-Z, Z + dz / 2, dz)
    Pi0_fun, C = profile(a, b, pi0)
    P = Pi0_fun(z)
    dtau = cfl * dz ** 2 / kappa ** 2
    tau, tauT = np.log(t0), np.log(T)
    store_t = sorted(store_t or [])
    stored = {}
    k = 0
    diag = dict(pmin=1.0, pmax=0.0, diff=0.0, peclet=0.0, react=0.0)
    while tau < tauT - 1e-15:
        h = min(dtau, tauT - tau)
        t = np.exp(tau)
        while k < len(store_t) and store_t[k] <= t:
            stored[store_t[k]] = P.copy(); k += 1
        Gv = Gfun(P, a, b)
        Gz, Gzz, Pz = _d1(Gv, dz), _d2(Gv, dz), _d1(P, dz)
        L = 0.5 * Gzz - z * Gz + 0.5 * z * Pz + 0.5 * (z * z - 1) * (Gv - P)
        rhs = L
        if drift:
            rhs = rhs + 0.5 * sigma * np.sqrt(t) * (Gz - Pz - z * (Gv - P))
        if switching:
            rhs = rhs + t * (q12 * (1 - P) - q21 * P)
        Pn = P + h * rhs
        if diagnostics:
            Gp = a * b / (a + (b - a) * P) ** 2
            D = 0.5 * Gp
            c = -z * Gp + 0.5 * z + (0.5 * sigma * np.sqrt(t) * (Gp - 1) if drift else 0.0)
            diag['pmin'] = min(diag['pmin'], Pn.min()); diag['pmax'] = max(diag['pmax'], Pn.max())
            diag['diff'] = max(diag['diff'], (2 * D * h / dz ** 2).max())
            diag['peclet'] = max(diag['peclet'], (np.abs(c) * dz / D).max())
            diag['react'] = max(diag['react'], h * t * (q12 + q21))
        P = np.clip(Pn, 0.0, 1.0)
        tau += h
    while k < len(store_t):
        stored[store_t[k]] = P.copy(); k += 1
    out = dict(z=z, Pi=P, stored=stored, Pi0=Pi0_fun(z), C=C, a=a, b=b)
    if diagnostics:
        out['diagnostics'] = diag
    return out


def regime_mass(z, P):
    from scipy.stats import norm
    return np.trapezoid(norm.pdf(z) * P, z)
