"""Two-state Markov-switching model for daily log-returns (Section 10).

Gaussian regimes are fitted by the EM algorithm (Hamilton filter, Kim smoother); Student-t
regimes with a common number of degrees of freedom by direct maximisation of the
Hamilton-filter likelihood.  Regime 0 is the low-volatility regime (L), regime 1 the high (H).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

TRADING_DAYS = 252


def hamilton_filter(dens, P, p0):
    """Filtered probabilities and log-likelihood. dens: (n, 2) regime densities."""
    n = dens.shape[0]
    filt = np.empty((n, 2))
    pred = np.empty((n, 2))
    ll = 0.0
    prob = p0
    for t in range(n):
        pred[t] = prob
        joint = prob * dens[t]
        c = joint.sum()
        ll += np.log(c)
        filt[t] = joint / c
        prob = filt[t] @ P
    return filt, pred, ll


def kim_smoother(filt, pred, P):
    n = filt.shape[0]
    sm = np.empty_like(filt)
    xi = np.empty((n - 1, 2, 2))
    sm[-1] = filt[-1]
    for t in range(n - 2, -1, -1):
        ratio = sm[t + 1] / pred[t + 1]
        sm[t] = filt[t] * (P @ ratio)
        xi[t] = filt[t][:, None] * P * ratio[None, :]
    return sm, xi


def stationary(P):
    p01, p10 = P[0, 1], P[1, 0]
    return np.array([p10, p01]) / (p01 + p10)


def fit_gaussian(r, tol=1e-9, max_iter=5000, init=None):
    """EM estimate of a two-state Gaussian Markov-switching model."""
    r = np.asarray(r, dtype=float)
    if init is None:
        s = r.std()
        mu = np.array([r.mean(), r.mean()])
        sd = np.array([0.6 * s, 1.6 * s])
        P = np.array([[0.98, 0.02], [0.05, 0.95]])
    else:
        mu, sd, P = (np.array(x, dtype=float) for x in init)
    p0 = stationary(P)
    ll_old = -np.inf
    for it in range(max_iter):
        dens = norm.pdf(r[:, None], mu[None, :], sd[None, :])
        filt, pred, ll = hamilton_filter(dens, P, p0)
        sm, xi = kim_smoother(filt, pred, P)
        w = sm.sum(axis=0)
        mu = (sm * r[:, None]).sum(axis=0) / w
        sd = np.sqrt((sm * (r[:, None] - mu) ** 2).sum(axis=0) / w)
        P = xi.sum(axis=0) / sm[:-1].sum(axis=0)[:, None]
        p0 = sm[0]
        if ll - ll_old < tol:
            break
        ll_old = ll
    order = np.argsort(sd)
    mu, sd, P = mu[order], sd[order], P[np.ix_(order, order)]
    return dict(mu=mu, sd=sd, P=P, loglik=ll, iterations=it + 1, smoothed=sm[:, order])


def _t_logpdf(x, loc, scale, nu):
    z = (x - loc) / scale
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale)
            - (nu + 1) / 2 * np.log1p(z * z / nu))


def fit_student(r, start=None):
    """Student-t regimes, common nu, by direct maximisation of the Hamilton-filter likelihood.

    Regime standard deviations are scale * sqrt(nu / (nu - 2)); kappa is their ratio.
    """
    r = np.asarray(r, dtype=float)
    if start is None:
        g = fit_gaussian(r)
        start = (g['mu'], g['sd'] * 0.8, g['P'], 6.0)
    mu0, sc0, P0, nu0 = start
    th0 = np.r_[mu0, np.log(sc0), np.log(P0[0, 1] / P0[0, 0]), np.log(P0[1, 0] / P0[1, 1]),
                np.log(nu0 - 2)]

    def unpack(th):
        mu, sc = th[0:2], np.exp(th[2:4])
        p01 = 1 / (1 + np.exp(-th[4])); p10 = 1 / (1 + np.exp(-th[5]))
        P = np.array([[1 - p01, p01], [p10, 1 - p10]])
        nu = 2 + np.exp(th[6])
        return mu, sc, P, nu

    def nll(th):
        mu, sc, P, nu = unpack(th)
        dens = np.exp(_t_logpdf(r[:, None], mu[None, :], sc[None, :], nu))
        return -hamilton_filter(dens, P, stationary(P))[2]

    res = minimize(nll, th0, method='Nelder-Mead',
                   options=dict(maxiter=20000, maxfev=20000, xatol=1e-7, fatol=1e-7))
    res = minimize(nll, res.x, method='BFGS')
    mu, sc, P, nu = unpack(res.x)
    sd = sc * np.sqrt(nu / (nu - 2))
    order = np.argsort(sd)
    return dict(mu=mu[order], sd=sd[order], scale=sc[order], P=P[np.ix_(order, order)], nu=nu,
                loglik=-res.fun)


def generator(P, days=TRADING_DAYS):
    """Annual generator Q = days * log P (closed form for two states)."""
    lam = P[0, 0] + P[1, 1] - 1.0
    if lam <= 0:
        raise ValueError('P is not embeddable (lambda <= 0)')
    return days * np.log(lam) / (lam - 1.0) * (P - np.eye(2))


def summarise(fit, days=TRADING_DAYS):
    """Quantities of Table 6: annualised volatilities, kappa, durations, intensities, pi_H."""
    P = fit['P']
    Q = generator(P, days)
    qLH, qHL = Q[0, 1], Q[1, 0]
    return dict(sigma_L=fit['sd'][0] * np.sqrt(days), sigma_H=fit['sd'][1] * np.sqrt(days),
                kappa=fit['sd'][1] / fit['sd'][0], D_L=1 / (1 - P[0, 0]), D_H=1 / (1 - P[1, 1]),
                q_LH=qLH, q_HL=qHL, pi_H=qLH / (qLH + qHL))


def simulate(mu, sd, P, n, seed=0):
    """Simulate returns from a Gaussian two-state model (used to test the estimator)."""
    rng = np.random.default_rng(seed)
    s = int(rng.random() < stationary(P)[1])
    r = np.empty(n)
    for t in range(n):
        r[t] = mu[s] + sd[s] * rng.standard_normal()
        s = s if rng.random() < P[s, s] else 1 - s
    return r
