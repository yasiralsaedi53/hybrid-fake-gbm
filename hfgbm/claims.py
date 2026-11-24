"""Payoffs of the claims in Table 4, the Brownian-bridge correction and implied volatilities."""
import numpy as np
from . import bs

S0, K = 100.0, 100.0


def survival_up(X, v, dt, b):
    """Brownian-bridge probability that log-price stays below b between all dates."""
    x0, x1 = X[:, :-1], X[:, 1:]
    below = (x0 < b) & (x1 < b)
    p = np.where(below, np.exp(-2.0 * (b - x0) * (b - x1) / (v * dt)), 1.0)
    return np.prod(np.where(below, 1.0 - p, 0.0), axis=1)


def survival_down(X, v, dt, b):
    x0, x1 = X[:, :-1], X[:, 1:]
    above = (x0 > b) & (x1 > b)
    p = np.where(above, np.exp(-2.0 * (x0 - b) * (x1 - b) / (v * dt)), 1.0)
    return np.prod(np.where(above, 1.0 - p, 0.0), axis=1)


def payoffs(X, v, dt, T=1.0, continuous=False):
    """Discounted payoffs (r = 0) of the claims of Table 4 on paths X with step variances v.

    Barriers are monitored at the simulation dates, or continuously with the Brownian-bridge
    correction if continuous=True.
    """
    S = S0 * np.exp(X)
    N = X.shape[1] - 1
    ST = S[:, -1]
    call = np.maximum(ST - K, 0.0)
    qv = v.sum(axis=1) * dt
    if continuous:
        alive_up = survival_up(X, v, dt, np.log(120.0 / S0))
        alive_dn = survival_down(X, v, dt, np.log(80.0 / S0))
    else:
        alive_up = (S[:, 1:].max(axis=1) < 120.0).astype(float)
        alive_dn = (S[:, 1:].min(axis=1) > 80.0).astype(float)
    return {
        'ATM European call (control)': call,
        'Volatility swap strike': 100 * np.sqrt(qv / T),
        'Variance swap strike': qv / T,
        'Up-and-out call, K=100, B=120': call * alive_up,
        'One-touch, B=80': 100.0 * (1.0 - alive_dn),
        'Forward-start ATM call': np.maximum(ST - S[:, N // 2], 0.0),
        'Down-and-out call, K=100, B=80': call * alive_dn,
        'Arithmetic Asian ATM call': np.maximum(S[:, 1:].mean(axis=1) - K, 0.0),
    }


def implied_vols(ST, strikes, tau, sigma_ref=0.2):
    """Implied volatilities from out-of-the-money options after matching the forward to S0.

    Returns arrays (iv, se) in volatility points; se is the delta-method standard error.
    """
    ST = ST * S0 / ST.mean()
    n = len(ST)
    iv, se = [], []
    for k in strikes:
        if k >= S0:
            pay, kind = np.maximum(ST - k, 0.0), 'call'
        else:
            pay, kind = np.maximum(k - ST, 0.0), 'put'
        price = pay.mean()
        s = bs.implied_vol(price, S0, k, tau, kind)
        iv.append(100 * s)
        se.append(100 * pay.std(ddof=1) / np.sqrt(n) / bs.vega(S0, k, s, tau))
    return np.array(iv), np.array(se)
