"""Section 10: Tables 6, 7 and 8 and the Student-t robustness check.

Usage:
    python scripts/empirical.py data/SP500_part1.csv data/SP500_part2.csv
    python scripts/empirical.py data/GSPC_yahoo.csv

The S&P 500 data are not included; see data/README.md.
"""
import datetime as dt
import numpy as np
from _common import parser, n_paths, Report
from hfgbm import data, msem
from hfgbm.experiments import calibrate, pricing, hedging_study

SAMPLES = [('2000-2025', 2000, 2025), ('2000-2012', 2000, 2012),
           ('2013-2025', 2013, 2025), ('2015-2025', 2015, 2025)]
CLAIMS = ['Volatility swap strike', 'Up-and-out call, K=100, B=120', 'One-touch, B=80',
          'Forward-start ATM call']


def row(cal, n, seed):
    p = pricing(cal, n, seed=seed)
    h = hedging_study(cal, n, rebalances=(252,), seed=seed, gbm=False)
    return p['diff'], h['std'][252], h['floor']


def fmt(diff, he, fl):
    return (f'{diff[CLAIMS[0]][0]:+6.1f} ' + ' '.join(f'{diff[c][0]:+6.1f}±{diff[c][1]:.1f}' for c in CLAIMS[1:])
            + f' {he:8.1f} {fl:8.1f}')


def main():
    p = parser(__doc__, 200000)
    p.add_argument('csv', nargs='+', help='Investing.com or Yahoo Finance CSV file(s)')
    p.add_argument('--no-mc', action='store_true', help='estimation only (Table 6, Student-t)')
    args = p.parse_args()
    n = n_paths(args)
    rep = Report('empirical')
    dates, closes, common = data.load_closes(*args.csv, start=dt.date(2000, 1, 1), end=dt.date(2025, 12, 31))
    r = data.log_returns(closes)
    rep(f'Data: {dates[0]} to {dates[-1]}, {len(r)} log-returns, annualised volatility '
        f'{100 * r.std() * np.sqrt(252):.1f}%, {common} common dates across files')
    gaps = [(a, b) for a, b in zip(dates[:-1], dates[1:]) if (b - a).days > 5]
    rep(f'Gaps longer than five calendar days: {gaps}')

    rep('\nTable 6. Two-state Gaussian Markov-switching estimates')
    rep(f'{"Sample":10s} {"N":>5s} {"sL":>6s} {"sH":>6s} {"kappa":>6s} {"DL":>5s} {"DH":>5s} {"qLH":>6s} {"qHL":>6s} {"piH":>5s}')
    est, tfits = {}, {}
    for lab, y0, y1 in SAMPLES:
        sel = np.array([y0 <= d.year <= y1 for d in dates])
        rr = data.log_returns(closes[sel])      # returns within the sample only
        fit = msem.fit_gaussian(rr)
        s = msem.summarise(fit)
        est[lab] = s
        rep(f'{lab:10s} {len(rr):5d} {100 * s["sigma_L"]:5.1f}% {100 * s["sigma_H"]:5.1f}% {s["kappa"]:6.2f} '
            f'{s["D_L"]:5.0f} {s["D_H"]:5.0f} {s["q_LH"]:6.2f} {s["q_HL"]:6.2f} {s["pi_H"]:5.2f}')
        P = fit['P']; lam = P[0, 0] + P[1, 1] - 1
        approx = -252 * np.log(np.diag(P))
        exact = np.array([s['q_LH'], s['q_HL']])
        rep(f'{"":10s} lambda = {lam:.4f}; exit-rate approximation differs by '
            f'{100 * np.abs(approx / exact - 1).max():.1f}%')
        tfits[lab] = (rr, fit)

    rep('\nRobustness: Student-t regimes with common degrees of freedom')
    t_est = {}
    for lab, (rr, g) in tfits.items():
        t = msem.fit_student(rr)
        s = msem.summarise(t)
        t_est[lab] = s
        rep(f'  {lab:10s} nu = {t["nu"]:.1f}, kappa = {s["kappa"]:.2f} ({100 * (s["kappa"] / est[lab]["kappa"] - 1):+.0f}%), '
            f'D_L = {s["D_L"]:.0f}, D_H = {s["D_H"]:.0f}, log-likelihood gain = {t["loglik"] - g["loglik"]:.0f}')

    if args.no_mc:
        rep.save(); return

    rep(f'\nTable 7. Differences from Black-Scholes (%), 504 steps, {n:,} paths; hedging error = std of the')
    rep('daily-rebalanced hedge (% of premium); last column = gamma floor')
    rep(f'{"Parameters":38s} {"VolSw":>6s} {"Up-and-out":>10s} {"One-touch":>10s} {"Fwd-start":>10s} {"Hedge":>8s} {"Floor":>8s}')
    rep(f'{"Moderate (kappa=1.5625, q=2)":38s} ' + fmt(*row(calibrate(1.5625, 2, 2, N=504, verbose=False), n, args.seed)))
    for lab in ['2013-2025', '2015-2025', '2000-2025', '2000-2012']:
        s = est[lab]
        cal = calibrate(s['kappa'], s['q_LH'], s['q_HL'], pi0=s['pi_H'], N=504, verbose=False)
        rep(f'{"S&P 500, " + lab:38s} ' + fmt(*row(cal, n, args.seed)))
    rep(f'{"Extreme (kappa=2.25, q=0)":38s} ' + fmt(*row(calibrate(2.25, 0, 0, N=504, verbose=False), n, args.seed)))

    rep('\nStudent-t parameters, up-and-out gap and hedging error')
    for lab in [x[0] for x in SAMPLES]:
        s = t_est[lab]
        cal = calibrate(s['kappa'], s['q_LH'], s['q_HL'], pi0=s['pi_H'], N=504, verbose=False)
        d, he, fl = row(cal, n, args.seed)
        rep(f'  {lab:10s} up-and-out {d[CLAIMS[1]][0]:+.1f}%, hedging error {he:.1f}%')

    rep('\nTable 8. Regime risk premium: q_LH -> theta q_LH, q_HL -> q_HL / theta (full sample;')
    rep('up-and-out and one-touch continuously monitored)')
    s = est['2000-2025']
    for th in (0.75, 1.0, 1.5, 2.0):
        qlh, qhl = th * s['q_LH'], s['q_HL'] / th
        piq = qlh / (qlh + qhl)
        cal = calibrate(s['kappa'], qlh, qhl, pi0=piq, N=504, verbose=False)
        cont = pricing(cal, n, seed=args.seed, continuous=True)['diff']
        disc, he, fl = row(cal, n, args.seed)
        mixed = dict(disc); mixed[CLAIMS[1]] = cont[CLAIMS[1]]; mixed[CLAIMS[2]] = cont[CLAIMS[2]]
        rep(f'  theta={th:<5} piH^Q={piq:.2f} ' + fmt(mixed, he, fl))
    rep.save()


if __name__ == '__main__':
    main()
