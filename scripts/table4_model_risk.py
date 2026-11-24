"""Table 4: model risk under identical vanilla prices (Section 8).

Eight claims, T = 1, under the hybrid model and geometric Brownian motion with the same
random numbers; kappa in {1.25, 1.5625, 2.25}, q in {0, 2}; 400,000 paths, 500 steps.
"""
from _common import parser, n_paths, Report
from hfgbm.experiments import calibrate, pricing, PRICED


def main():
    args = parser(__doc__, 400000).parse_args()
    n = n_paths(args)
    rep = Report('table4_model_risk')
    configs = [(q, k) for q in (0, 2) for k in (1.25, 1.5625, 2.25)]
    res = {}
    for q, k in configs:
        res[(q, k)] = pricing(calibrate(k, q, q, verbose=False), n, seed=args.seed)
    rep(f'Table 4. Relative difference from the Black-Scholes price (%), 95% half-widths ({n:,} paths)')
    rep(f'{"Claim":32s}' + ''.join(f'{"q=%d k=%s" % (q, k):>16s}' for q, k in configs))
    for c in PRICED:
        cells = []
        for cfg in configs:
            d, hw = res[cfg]['diff'][c]
            cells.append(f'{d:+.1f}' if c == 'Volatility swap strike' else f'{d:+.1f}±{hw:.1f}')
        rep(f'{c:32s}' + ''.join(f'{x:>16s}' for x in cells))
    bs = res[(0, 1.25)]['bs']
    rep('\nBlack-Scholes Monte Carlo prices (q = 0 run): ' + ', '.join(f'{bs[c]:.3f}' for c in PRICED))
    rep('Variance swap strike (vol points): ' + ', '.join(f'{res[c]["var_swap_vol"]:.2f}' for c in configs))
    r = res[(0, 2.25)]
    rep(f'Realised variance, kappa = 2.25, q = 0: mean {r["rv_mean"]:.3f}, std {r["rv_std"]:.3f}, '
        f'90% range [{r["rv_q"][0]:.3f}, {r["rv_q"][1]:.3f}]; q = 2: std {res[(2, 2.25)]["rv_std"]:.3f}')
    rep.save()


if __name__ == '__main__':
    main()
