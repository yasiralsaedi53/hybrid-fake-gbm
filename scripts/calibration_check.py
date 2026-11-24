"""Section 7.4 and Figure 3: the calibrated model reproduces a flat 20% surface.

Implied volatilities at maturities 0.25, 0.5, 1 and strikes 70..130 for kappa in
{1.25, 1.5625, 2.25} and q in {0, 2} (400,000 paths, 500 steps); conditional regime
probabilities against the finite-difference solution; and the smile of the same driver
without leverage (Figure 3).
"""
import numpy as np
from _common import parser, n_paths, Report, FIGURES
from hfgbm import mc, claims
from hfgbm.experiments import calibrate, SIGMA

STRIKES = [70, 80, 90, 100, 110, 120, 130]


def main():
    args = parser(__doc__, 400000).parse_args()
    n = n_paths(args)
    rep = Report('calibration_check')
    rep(f'Implied volatilities (%) and max |IV - 20| / s.e. ({n:,} paths, 500 steps)')
    for k in (1.25, 1.5625, 2.25):
        for q in (0, 2):
            cal = calibrate(k, q, q, verbose=False)
            idx = {0.25: 125, 0.5: 250, 1.0: 500}
            ST = {T: [] for T in idx}
            XT, YT = [], []
            for X, v, Xg, y in mc.simulate(cal, n, seed=args.seed, with_regime=True):
                for T, i in idx.items():
                    ST[T].append(100 * np.exp(X[:, i]))
                XT.append(X[:, -1]); YT.append(y)
            worst, worst_ratio = 0, 0
            for T in idx:
                iv, se = claims.implied_vols(np.concatenate(ST[T]), STRIKES, T)
                d = np.abs(iv - 100 * SIGMA)
                worst = max(worst, d.max()); worst_ratio = max(worst_ratio, (d / se).max())
            # regime probabilities in z-bins against the PDE
            x, y = np.concatenate(XT), np.concatenate(YT)
            z = (x + 0.5 * SIGMA ** 2) / SIGMA
            bins = np.arange(-2.5, 2.51, 0.5)
            dev = 0
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (z >= lo) & (z < hi)
                zz = np.linspace(lo, hi, 21)
                pde_p = np.interp(zz, cal.z, cal.final).mean()
                dev = max(dev, abs(y[sel].mean() - pde_p))
            rep(f'  kappa={k:<6} q={q}: max |IV-20| = {worst:.2f} pts, max ratio to s.e. = '
                f'{worst_ratio:.2f}, max regime-probability deviation = {dev:.3f}')

    # Figure 3: smile with and without leverage, kappa = 1.5625, q = 0
    k = 1.5625
    cal = calibrate(k, 0, 0, verbose=False)
    K = np.arange(70, 131, 5)
    out = {}
    for lev in (True, False):
        ST = np.concatenate([100 * np.exp(X[:, -1])
                             for X, v, Xg in mc.simulate(cal, n, seed=args.seed, leverage=lev)])
        out[lev] = claims.implied_vols(ST, K, 1.0)[0]
    rep('\nFigure 3 data (T = 1, kappa = 1.5625, q = 0): K, with leverage, same driver without leverage')
    for kk, a, b in zip(K, out[True], out[False]):
        rep(f'  {kk:4d} {a:7.2f} {b:7.2f}')
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(K, out[True], 'o-', color='navy', ms=3, label='Hybrid model (with leverage)')
        ax.plot(K, out[False], 'o-', color='gray', mfc='white', ms=3, label='Same driver (no leverage)')
        ax.set_xlabel('Strike $K$ (spot 100)'); ax.set_ylabel('Implied volatility (%)')
        ax.grid(alpha=0.3); ax.legend(loc='upper right')
        fig.tight_layout(); fig.savefig(f'{FIGURES}/figure3_smile.png', dpi=200)
    except ImportError:
        pass
    rep.save()


if __name__ == '__main__':
    main()
