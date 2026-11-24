"""Table 1 and Figure 1: the closed-form self-similar fake Brownian motion (Theorem 5.1).

Simulates dX = f(Y0) / sqrt(a + (b-a) Pi(X/sqrt t)) dW with f1 = 0.8, f2 = 1.25, pi0 = 0.5
(200,000 paths, 2,000 Euler steps on [0, 1]) and compares with the theorem and with the
mixture without leverage.
"""
import numpy as np
from scipy import stats
from _common import parser, n_paths, Report, FIGURES
from hfgbm.profile import profile, conditional_probability

F1, F2, PI0 = 0.8, 1.25, 0.5


def simulate(n, steps, seed, chunk=20000):
    a, b = F1 ** 2, F2 ** 2
    Pi, C = profile(a, b, PI0)
    zg = np.linspace(-12, 12, 4801)
    Pg = Pi(zg)
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps
    X1, Y0, RV = [], [], []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        y = rng.random(m) < PI0
        f = np.where(y, F2, F1)
        x = np.zeros(m)
        rv = np.zeros(m)
        for k in range(steps):
            t = k * dt
            p = np.full(m, PI0) if k == 0 else np.interp(x / np.sqrt(t), zg, Pg)
            s2 = f ** 2 / (a + (b - a) * p)
            x += np.sqrt(s2 * dt) * rng.standard_normal(m)
            rv += s2 * dt
        X1.append(x); Y0.append(y); RV.append(rv)
        done += m
    return np.concatenate(X1), np.concatenate(Y0), np.concatenate(RV), Pi, C


def main():
    args = parser(__doc__, 200000).parse_args()
    n = n_paths(args)
    rep = Report('table1_closed_form')
    X, Y, RV, Pi, C = simulate(n, 2000, args.seed)
    a, b = F1 ** 2, F2 ** 2
    mix_var = (1 - PI0) * a + PI0 * b
    mix_kurt = 3 * ((1 - PI0) * a ** 2 + PI0 * b ** 2) / mix_var ** 2 - 3
    rep(f'Table 1. Monte Carlo check of Theorem 5.1 at t = 1 ({n:,} paths, 2,000 steps); C = {C:.4f}')
    rep(f'{"Quantity at t=1":34s} {"Monte Carlo":>12s} {"Theorem 5.1":>12s} {"No leverage":>12s}')
    rep(f'{"Variance of X_1":34s} {X.var():12.3f} {1:12d} {mix_var:12.2f}')
    rep(f'{"Excess kurtosis of X_1":34s} {stats.kurtosis(X):12.3f} {0:12d} {mix_kurt:12.2f}')
    rep(f'{"KS p-value vs N(0,1)":34s} {stats.kstest(X, "norm").pvalue:12.2f} {"-":>12s} {"-":>12s}')
    for lo, hi, lab in [(0, 0.25, 'Q(Y0=2 | |X1|<0.25)'), (2, 2.5, 'Q(Y0=2 | 2<=|X1|<2.5)')]:
        sel = (np.abs(X) >= lo) & (np.abs(X) < hi)
        nolev = f'{PI0:12.1f}' if lo == 0 else f'{"-":>12s}'
        rep(f'{lab:34s} {Y[sel].mean():12.3f} {conditional_probability(Pi, lo, hi):12.3f} {nolev}')
    rep(f'\nRealised variance <X>_1: mean {RV.mean():.3f}, std {RV.std():.2f}, '
        f'90% range [{np.quantile(RV, 0.05):.2f}, {np.quantile(RV, 0.95):.2f}], '
        f'mean of square root {np.sqrt(RV).mean():.3f}')
    z = np.linspace(-3, 3, 25)
    rep('\nFigure 1 data: z, Pi(z)')
    rep(' '.join(f'({zi:g},{p:.4f})' for zi, p in zip(z, Pi(z))))
    rep(f'Pi(0) = {Pi(0.0):.4f}, Pi(3) = {Pi(3.0):.4f}; leverage factor at |z|=3: '
        f'{1 / np.sqrt(a + (b - a) * Pi(3.0)):.2f}')
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        zz = np.linspace(-3, 3, 301)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(zz, Pi(zz), color='navy', lw=1.8, label=r'$\Pi(z)=\Phi^{-1}(z^2/2+C)$')
        ax.axhline(PI0, ls='--', color='gray', label=r'Unconditional $\pi_0=0.5$')
        ax.set_xlabel(r'Standardised move $z=X_t/\sqrt{t}$'); ax.set_ylabel(r'$\Pi(z)$')
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(loc='lower right')
        fig.tight_layout(); fig.savefig(f'{FIGURES}/figure1_profile.png', dpi=200)
    except ImportError:
        pass
    rep.save()


if __name__ == '__main__':
    main()
