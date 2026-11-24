"""Monte Carlo simulation of the calibrated hybrid model (Sections 7--10).

The log-price X_t = log(S_t / S_0) follows

    dX_t = -v_t/2 dt + sqrt(v_t) dW_t,   v_t = sigma^2 f(Y_t)^2 / (a + (b - a) pi(t, X_t)),

where Y is a two-state Markov chain with intensities q12 (low -> high) and q21 (high -> low),
a = f1^2, b = f2^2, and pi(t, x) = Q(Y_t = 2 | X_t = x) is the finite-difference solution of
Section 7.  Every simulation also produces geometric Brownian motion paths driven by the same
normal increments, so that model differences are estimated with small error.
"""
import numpy as np
from .pde import solve


class Calibration:
    """Leverage of the hybrid model, read from the finite-difference solution."""

    def __init__(self, f1, f2, sigma, q12, q21, pi0, T, N, verbose=False, **pde_kw):
        self.f1, self.f2, self.sigma = f1, f2, sigma
        self.q12, self.q21, self.pi0 = q12, q21, pi0
        self.a, self.b = f1 ** 2, f2 ** 2
        self.T, self.N = T, N
        self.dt = T / N
        self.times = self.dt * np.arange(N)
        store = list(self.times[1:])
        sol = solve(f1, f2, sigma, q12, q21, pi0, T=T, store_t=store,
                    switching=(q12 > 0 or q21 > 0), **pde_kw)
        self.z = sol['z']
        self.profiles = [None] + [sol['stored'][t] for t in store]
        self.final = sol['Pi']
        if verbose:
            print(f'  PDE solved: kappa={f2 / f1:.4f}, q=({q12},{q21}), pi0={pi0}')

    def pi(self, n, x):
        """Conditional probability of regime 2 at time t_n given X = x."""
        if n == 0:
            return np.full_like(x, self.pi0)
        t = self.times[n]
        z = (x + 0.5 * self.sigma ** 2 * t) / (self.sigma * np.sqrt(t))
        return np.interp(z, self.z, self.profiles[n])

    def variance(self, n, x, y):
        f2 = np.where(y == 1, self.b, self.a)
        return self.sigma ** 2 * f2 / (self.a + (self.b - self.a) * self.pi(n, x))


def simulate(cal, n_paths, seed=0, chunk=20000, leverage=True, with_regime=False):
    """Yield chunks (X, v, Xg) of hybrid log-prices, hybrid variances and GBM log-prices.

    X and Xg have shape (m, N+1); v has shape (m, N) and holds the variance on each step.
    With leverage=False the same regime driver is used with v = sigma^2 f(Y)^2 / E f(Y_0)^2.
    With with_regime=True the terminal regime (0 or 1) is yielded as a fourth array.
    """
    rng = np.random.default_rng(seed)
    N, dt, sig = cal.N, cal.dt, cal.sigma
    p_up = 1.0 - np.exp(-cal.q12 * dt)
    p_dn = 1.0 - np.exp(-cal.q21 * dt)
    norm_const = cal.a * (1 - cal.pi0) + cal.b * cal.pi0
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        Z = rng.standard_normal((m, N))
        U = rng.random((m, N))
        y = (rng.random(m) < cal.pi0).astype(np.int8)
        X = np.zeros((m, N + 1))
        Xg = np.zeros((m, N + 1))
        v = np.empty((m, N))
        sq = np.sqrt(dt)
        for n in range(N):
            if leverage:
                vn = cal.variance(n, X[:, n], y)
            else:
                vn = sig ** 2 * np.where(y == 1, cal.b, cal.a) / norm_const
            v[:, n] = vn
            X[:, n + 1] = X[:, n] - 0.5 * vn * dt + np.sqrt(vn) * sq * Z[:, n]
            Xg[:, n + 1] = Xg[:, n] - 0.5 * sig ** 2 * dt + sig * sq * Z[:, n]
            switch = np.where(y == 1, U[:, n] < p_dn, U[:, n] < p_up)
            y = np.where(switch, 1 - y, y).astype(np.int8)
        done += m
        yield (X, v, Xg, y) if with_regime else (X, v, Xg)


def rel_diff(h, g):
    """Relative difference E[h]/E[g] - 1 (in %) with a 95% half-width from paired samples."""
    h, g = np.asarray(h), np.asarray(g)
    mg = g.mean()
    r = h.mean() / mg - 1.0
    se = np.std(h - (1 + r) * g, ddof=1) / np.sqrt(len(g)) / mg
    return 100 * r, 196 * se
