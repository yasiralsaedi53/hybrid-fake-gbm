"""Black--Scholes and Bachelier formulas (r = 0)."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1(S, K, sigma, tau):
    return (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))


def call(S, K, sigma, tau):
    S = np.asarray(S, dtype=float)
    tau = np.maximum(tau, 1e-300)
    d1 = _d1(S, K, sigma, tau)
    return S * norm.cdf(d1) - K * norm.cdf(d1 - sigma * np.sqrt(tau))


def put(S, K, sigma, tau):
    return call(S, K, sigma, tau) - S + K


def delta(S, K, sigma, tau):
    tau = np.maximum(tau, 1e-300)
    return norm.cdf(_d1(S, K, sigma, tau))


def gamma(S, K, sigma, tau):
    tau = np.maximum(tau, 1e-300)
    return norm.pdf(_d1(S, K, sigma, tau)) / (S * sigma * np.sqrt(tau))


def vega(S, K, sigma, tau):
    return S * norm.pdf(_d1(S, K, sigma, tau)) * np.sqrt(tau)


def implied_vol(price, S, K, tau, kind='call'):
    """Implied volatility from a call or put price."""
    f = call if kind == 'call' else put
    return brentq(lambda s: f(S, K, s, tau) - price, 1e-4, 5.0, xtol=1e-12)


def up_and_out_call_continuous(S, K, B, sigma, tau):
    """Continuously monitored up-and-out call with r = 0 and K < B.

    With r = 0 the reflection principle gives V(S) = U(S) - (S/B) U(B^2/S), where U is the
    price of the knocked-out payoff (s - K)^+ 1{s < B} without monitoring.
    """
    if S >= B:
        return 0.0

    def U(x):
        d2 = _d1(x, B, sigma, tau) - sigma * np.sqrt(tau)
        return call(x, K, sigma, tau) - call(x, B, sigma, tau) - (B - K) * norm.cdf(d2)

    return float(U(S) - (S / B) * U(B * B / S))


def bachelier_call(S, K, sigma, tau):
    s = sigma * np.sqrt(tau)
    d = (S - K) / s
    return (S - K) * norm.cdf(d) + s * norm.pdf(d)
