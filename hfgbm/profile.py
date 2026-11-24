"""Closed-form self-similar profile (Theorem 5.1) and the static-mixture check (Proposition 6.1).

Two regimes with volatility levels f1 < f2, a = f1**2, b = f2**2.  The conditional
probability of the high-volatility regime in the Brownian setting is

    Pi(z) = Phi^{-1}(z**2/2 + C),
    Phi(p) = [b log p - a log(1-p)]/(b-a) - log(a + (b-a) p),

with C fixed by the mass condition  int phi(z) Pi(z) dz = pi0.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import norm


def G(p, a, b):
    """Variance-weighted probability g_2 = b p / (a + (b-a) p)."""
    return b * p / (a + (b - a) * p)


def Gprime(p, a, b):
    return a * b / (a + (b - a) * p) ** 2


def Phi_logit(l, a, b):
    """Phi evaluated at p = 1/(1+exp(-l)), computed stably for large |l|."""
    log_p = -np.logaddexp(0.0, -l)
    log_1mp = -np.logaddexp(0.0, l)
    p = np.exp(log_p)
    return (b * log_p - a * log_1mp) / (b - a) - np.log(a + (b - a) * p)


def Phi(p, a, b):
    p = np.asarray(p, dtype=float)
    return (b * np.log(p) - a * np.log1p(-p)) / (b - a) - np.log(a + (b - a) * p)


def Phi_inv(y, a, b):
    """Inverse of Phi, vectorised; solved in logit space for stability."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    out = np.empty_like(y)
    for i, yi in enumerate(y):
        l = brentq(lambda l: Phi_logit(l, a, b) - yi, -1e4, 1e4, xtol=1e-13)
        out[i] = 1.0 / (1.0 + np.exp(-l))
    return out


def profile(a, b, pi0, zmax=12.0):
    """Return (Pi, C): the profile Pi(z) as a vectorised function and the constant C."""
    def Pi_scalar(z, C):
        return Phi_inv(z * z / 2.0 + C, a, b)[0]

    def mass(C):
        return quad(lambda z: norm.pdf(z) * Pi_scalar(z, C), -zmax, zmax, limit=200)[0] - pi0

    C = brentq(mass, -40.0, 40.0, xtol=1e-12)

    def Pi(z):
        z = np.asarray(z, dtype=float)
        return Phi_inv(z * z / 2.0 + C, a, b).reshape(z.shape)

    return Pi, C


def conditional_probability(Pi, lo, hi):
    """Q(Y0 = 2 | lo <= |X_1| < hi) for the closed-form model (symmetric profile)."""
    num = quad(lambda z: norm.pdf(z) * Pi(z), lo, hi)[0]
    den = quad(norm.pdf, lo, hi)[0]
    return num / den


if __name__ == '__main__':
    f1, f2, pi0 = 0.8, 1.25, 0.5
    a, b = f1 ** 2, f2 ** 2
    Pi, C = profile(a, b, pi0)
    print(f'C = {C:.7f}')
    print(f'Pi(0) = {Pi(0.0):.6f},  Pi(3) = {Pi(3.0):.6f}')
    print(f'Q(Y0=2 | |X1|<0.25)      = {conditional_probability(Pi, 0.0, 0.25):.6f}')
    print(f'Q(Y0=2 | 2<=|X1|<2.5)    = {conditional_probability(Pi, 2.0, 2.5):.6f}')
