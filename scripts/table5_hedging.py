"""Table 5 and Figure 4: Black--Scholes delta hedging of an ATM call (Section 9).

1,008 Euler steps, 100,000 paths, rebalancing 12, 48, 252 or 1,008 times per year.
"""
from _common import parser, n_paths, Report, FIGURES
from hfgbm.experiments import calibrate, hedging_study

R = (12, 48, 252, 1008)


def main():
    args = parser(__doc__, 100000).parse_args()
    n = n_paths(args)
    rep = Report('table5_hedging')
    rows = {}
    for q in (0, 2):
        for k in (1.25, 1.5625, 2.25):
            rows[(k, q)] = hedging_study(calibrate(k, q, q, N=1008, verbose=False), n,
                                         rebalances=R, seed=args.seed, gbm=(k, q) == (1.25, 0))
    g = rows[(1.25, 0)]
    rep(f'Table 5. Delta-hedging error of a short ATM call (premium {g["premium"]:.3f}), {n:,} paths')
    rep(f'{"Model":22s}{"q":>3s}' + ''.join(f'{r:>7d}' for r in R) + f'{"ES5%@252":>10s}{"Floor":>7s}{"corr":>7s}')
    rep(f'{"GBM":22s}{"-":>3s}' + ''.join(f'{g["gbm_std"][r]:7.1f}' for r in R) + f'{g["gbm_es252"]:10.2f}{0:7d}')
    for (k, q), x in rows.items():
        rep(f'{"kappa=" + str(k):22s}{q:3d}' + ''.join(f'{x["std"][r]:7.1f}' for r in R)
            + f'{x["es252"]:10.2f}{x["floor"]:7.1f}{x["corr"]:7.3f}')
    rep('Largest |mean P&L|: ' + f'{max(abs(m) for x in rows.values() for m in x["mean"].values()):.3f}')
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for k, c, mk in [(2.25, '#1f3b8f', 'o'), (1.5625, '#4f6fc0', 's'), (1.25, '#9fb0e0', '^')]:
            ax.plot(R, [rows[(k, 0)]['std'][r] for r in R], marker=mk, color=c, label=f'Hybrid, $\\kappa={k}$')
        ax.plot(R, [g['gbm_std'][r] for r in R], 'o--', color='gray', mfc='white', label='Geometric Brownian motion')
        ax.set_xscale('log'); ax.set_xticks(R); ax.set_xticklabels(R)
        ax.set_xlabel('Rebalancing dates per year'); ax.set_ylabel('Std. of hedging error (% of premium)')
        ax.set_ylim(0, 75); ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=8, loc='upper center')
        fig.tight_layout(); fig.savefig(f'{FIGURES}/figure4_hedging.png', dpi=200)
    except ImportError:
        pass
    rep.save()


if __name__ == '__main__':
    main()
