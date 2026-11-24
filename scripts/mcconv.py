"""Section 7.5: Monte Carlo convergence in the number of Euler steps, and continuous monitoring.

kappa = 2.25 with 100, 200, 500 and 1,000 steps (200,000 paths per run); Brownian-bridge
correction for continuous barrier monitoring.
"""
import numpy as np
from _common import parser, n_paths, Report
from hfgbm import mc, claims, bs
from hfgbm.experiments import calibrate, pricing

UO, OT = 'Up-and-out call, K=100, B=120', 'One-touch, B=80'


def main():
    args = parser(__doc__, 200000).parse_args()
    n = n_paths(args)
    rep = Report('mcconv')
    rep(f'Step-size study, kappa = 2.25 ({n:,} paths per run)')
    rep(f'{"steps":>6s} {"IV80":>7s} {"IV100":>7s} {"IV120":>7s} {"vol swap":>9s} {"UO q=0":>13s} {"OT q=0":>13s} {"UO q=2":>13s}')
    for N in (100, 200, 500, 1000):
        cal = calibrate(2.25, 0, 0, N=N, verbose=False)
        ST, Hs, Gs = [], [], []
        for X, v, Xg in mc.simulate(cal, n, seed=args.seed):
            ST.append(100 * np.exp(X[:, -1]))
            Hs.append(np.sqrt(v.sum(axis=1) * cal.dt))
        iv, _ = claims.implied_vols(np.concatenate(ST), [80, 100, 120], 1.0)
        vs = 100 * np.concatenate(Hs).mean()
        r0 = pricing(cal, n, seed=args.seed)['diff']
        r2 = pricing(calibrate(2.25, 2, 2, N=N, verbose=False), n, seed=args.seed)['diff']
        rep(f'{N:6d} ' + ' '.join(f'{x:7.2f}' for x in iv) + f' {vs:9.2f} '
            + ' '.join(f'{r[c][0]:+7.1f}±{r[c][1]:.1f}' for r, c in [(r0, UO), (r0, OT), (r2, UO)]))

    rep('\nContinuous monitoring (Brownian-bridge correction)')
    cal = calibrate(2.25, 0, 0, N=504, verbose=False)
    G = np.concatenate([claims.payoffs(Xg, np.full_like(v, 0.04), cal.dt, continuous=True)[UO]
                        for X, v, Xg in mc.simulate(cal, n, seed=args.seed)])
    rep(f'  GBM up-and-out: bridge-corrected MC {G.mean():.3f}, closed form '
        f'{bs.up_and_out_call_continuous(100, 100, 120, 0.2, 1.0):.3f}')
    for lab, c in [('kappa=2.25, q=0', cal),
                   ('S&P 500 2000-2025', calibrate(2.776, 2.914, 6.952, pi0=0.295, N=504, verbose=False))]:
        cont = pricing(c, n, seed=args.seed, continuous=True)['diff']
        disc = pricing(c, n, seed=args.seed)['diff']
        rep(f'  {lab:20s} continuous: UO {cont[UO][0]:+.1f}±{cont[UO][1]:.1f}, OT {cont[OT][0]:+.1f}±{cont[OT][1]:.1f};'
            f' discrete (504 dates): UO {disc[UO][0]:+.1f}, OT {disc[OT][0]:+.1f}')
    rep.save()


if __name__ == '__main__':
    main()
