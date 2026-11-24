"""Black--Scholes delta hedging of a short call (Section 9, Proposition 9.1)."""
import numpy as np
from . import bs


def hedge_pnl(X, v, dt, sigma, rebalances, S0=100.0, K=100.0, T=1.0):
    """Terminal P&L of a short call sold at the Black--Scholes price and delta hedged.

    X: log-prices on the simulation grid (m, N+1); rebalancing every N / rebalances steps.
    Returns (pnl, floor) where floor is the gamma integral (13) along the same paths.
    """
    N = X.shape[1] - 1
    S = S0 * np.exp(X)
    premium = float(bs.call(S0, K, sigma, T))
    step = N // rebalances
    idx = np.arange(0, N + 1, step)
    if idx[-1] != N:
        idx = np.append(idx, N)
    gains = np.zeros(S.shape[0])
    for i0, i1 in zip(idx[:-1], idx[1:]):
        d = bs.delta(S[:, i0], K, sigma, T - i0 * dt)
        gains += d * (S[:, i1] - S[:, i0])
    pnl = premium + gains - np.maximum(S[:, -1] - K, 0.0)
    t = dt * np.arange(N)
    G = bs.gamma(S[:, :-1], K, sigma, T - t)
    floor = 0.5 * np.sum(G * S[:, :-1] ** 2 * (sigma ** 2 - v) * dt, axis=1)
    return pnl, floor, premium


def summary(pnl, premium):
    """Std. as % of premium, expected shortfall ES_5% (mean of worst 5%), and mean."""
    k = max(1, int(0.05 * len(pnl)))
    es = np.sort(pnl)[:k].mean()
    return 100 * pnl.std(ddof=1) / premium, es, pnl.mean()
