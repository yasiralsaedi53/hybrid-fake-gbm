"""Pricing and hedging experiments shared by the scripts (Sections 7--10)."""
import numpy as np
from . import mc, claims, hedging

SIGMA = 0.2
PRICED = ['ATM European call (control)', 'Volatility swap strike', 'Up-and-out call, K=100, B=120',
          'One-touch, B=80', 'Forward-start ATM call', 'Down-and-out call, K=100, B=80',
          'Arithmetic Asian ATM call']


def calibrate(kappa, q12, q21, pi0=0.5, N=500, T=1.0, verbose=True):
    """Two regimes f1 = kappa^(-1/2), f2 = kappa^(1/2), so that f2 / f1 = kappa."""
    return mc.Calibration(kappa ** -0.5, kappa ** 0.5, SIGMA, q12, q21, pi0, T=T, N=N,
                          verbose=verbose)


def pricing(cal, n_paths, seed=0, continuous=False):
    """Relative differences (%) from the Black--Scholes model with 95% half-widths.

    Returns dict claim -> (diff, half_width), the Black--Scholes Monte Carlo prices and the
    variance-swap strike in volatility points.
    """
    H, G = {}, {}
    for X, v, Xg in mc.simulate(cal, n_paths, seed=seed):
        h = claims.payoffs(X, v, cal.dt, continuous=continuous)
        g = claims.payoffs(Xg, np.full_like(v, SIGMA ** 2), cal.dt, continuous=continuous)
        for k in h:
            H.setdefault(k, []).append(h[k]); G.setdefault(k, []).append(g[k])
    H = {k: np.concatenate(x) for k, x in H.items()}
    G = {k: np.concatenate(x) for k, x in G.items()}
    out = {k: mc.rel_diff(H[k], G[k]) for k in PRICED}
    bs_prices = {k: G[k].mean() for k in PRICED}
    var_strike = 100 * np.sqrt(H['Variance swap strike'].mean())
    rv = H['Variance swap strike']
    return dict(diff=out, bs=bs_prices, var_swap_vol=var_strike,
                rv_mean=rv.mean(), rv_std=rv.std(), rv_q=np.quantile(rv, [0.05, 0.95]))


def hedging_study(cal, n_paths, rebalances=(12, 48, 252, 1008), seed=0, chunk=10000, gbm=True):
    """Std. of the hedging error (% of premium), ES_5% at 252 dates, mean, floor and correlation."""
    P = {R: [] for R in rebalances}
    Pg = {R: [] for R in rebalances}
    F = []
    for X, v, Xg in mc.simulate(cal, n_paths, seed=seed, chunk=chunk):
        for R in rebalances:
            pnl, fl, prem = hedging.hedge_pnl(X, v, cal.dt, SIGMA, R)
            P[R].append(pnl)
            if gbm:
                Pg[R].append(hedging.hedge_pnl(Xg, np.full_like(v, SIGMA ** 2), cal.dt, SIGMA, R)[0])
        F.append(fl)
    P = {R: np.concatenate(x) for R, x in P.items()}
    F = np.concatenate(F)
    Rmax = max(rebalances)
    R252 = 252 if 252 in P else Rmax
    res = dict(premium=prem,
               std={R: hedging.summary(P[R], prem)[0] for R in P},
               mean={R: P[R].mean() for R in P},
               es252=hedging.summary(P[R252], prem)[1],
               floor=100 * F.std(ddof=1) / prem,
               corr=np.corrcoef(P[Rmax], F)[0, 1])
    if gbm:
        Pg = {R: np.concatenate(x) for R, x in Pg.items()}
        res['gbm_std'] = {R: hedging.summary(Pg[R], prem)[0] for R in Pg}
        res['gbm_es252'] = hedging.summary(Pg[R252], prem)[1]
    return res
