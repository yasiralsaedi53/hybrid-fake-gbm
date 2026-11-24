"""Section 7: Table 2 (shifted profile), Table 3 (convergence), scheme checks and Figure 2.

No Monte Carlo is involved; the script runs in a few minutes.
"""
import numpy as np
from _common import Report, FIGURES
from hfgbm.pde import solve, regime_mass
from hfgbm.profile import profile

SIGMA = 0.2


def f(k):
    return k ** -0.5, k ** 0.5


def at(z, P, pts):
    return [float(np.interp(p, z, P)) for p in pts]


def main():
    rep = Report('table2_3_pde')

    # ---- Table 2: unshifted vs shifted small-time profile
    rep('Table 2. Max |pi(t, m(t)+sigma sqrt(t) z) - approximation| over |z| <= 2.8 (q = 0, sigma = 20%)')
    rep(f'{"kappa":>7s} ' + ' '.join(f'{"t=" + str(t) + " uns.":>12s} {"shifted":>9s}' for t in (0.01, 0.1, 1)))
    for k in (1.5625, 2.25, 2.78):
        f1, f2 = f(k)
        Pi0, _ = profile(f1 ** 2, f2 ** 2, 0.5)
        sol = solve(f1, f2, SIGMA, 0, 0, 0.5, T=1.0, store_t=[0.01, 0.1, 1.0], switching=False)
        z = sol['z']; m = np.abs(z) <= 2.8
        row = []
        for t in (0.01, 0.1, 1.0):
            P = sol['stored'][t]
            row.append(np.abs(P - Pi0(z))[m].max())
            row.append(np.abs(P - Pi0(z - 0.5 * SIGMA * np.sqrt(t)))[m].max())
        bound = SIGMA ** 2 * k ** 2 * (k ** 2 - 1) / 32
        rep(f'{k:7.4f} ' + ' '.join(f'{row[2 * i]:12.4f} {row[2 * i + 1]:9.5f}' for i in range(3))
            + f'   Theorem 7.1(3) bound at t=1: {bound:.4f}')
    k = 1.5625; f1, f2 = f(k)
    Pi0, _ = profile(f1 ** 2, f2 ** 2, 0.5)
    sol = solve(f1, f2, SIGMA, 2, 2, 0.5, T=0.01)
    z = sol['z']; m = np.abs(z) <= 2.8
    u = np.abs(sol['Pi'] - Pi0(z))[m].max(); s = np.abs(sol['Pi'] - Pi0(z - 0.1 * 0.1))[m].max()
    rep(f'With q = 2, kappa = 1.5625, t = 0.01: unshifted {u:.4f}, shifted {s:.4f} (ratio {u / s:.2f})')

    # ---- scheme checks
    rep('\nScheme checks')
    sol = solve(f1, f2, SIGMA, 0, 0, 0.5, t0=1e-3, T=1.0, drift=False, switching=False)
    rep(f'  drift and switching removed: max deviation from Pi0 = '
        f'{np.abs(sol["Pi"] - sol["Pi0"]).max():.1e}')
    worst = 0.0
    for k in (1.25, 1.5625, 2.25):
        for q in (0, 2):
            a, b = f(k)
            s2 = solve(a, b, SIGMA, q, q, 0.5, T=1.0, switching=q > 0)
            worst = max(worst, abs(regime_mass(s2['z'], s2['Pi']) - 0.5))
    rep(f'  regime mass error over the six configurations: {worst:.1e}')
    for lab, k, q12, q21, pi0 in [('extreme (2.25, q=0)', 2.25, 0, 0, 0.5),
                                   ('moderate (1.5625, q=2)', 1.5625, 2, 2, 0.5),
                                   ('S&P 500 2000-2025 (2.78)', 2.776, 2.914, 6.952, 0.295)]:
        a, b = f(k)
        d = solve(a, b, SIGMA, q12, q21, pi0, T=1.0, switching=q12 > 0, diagnostics=True)['diagnostics']
        rep(f'  {lab:26s} min {d["pmin"]:.2f}, max {d["pmax"]:.4f}, diffusion no. {d["diff"]:.2f}, '
            f'Peclet {d["peclet"]:.2f}, reaction {d["react"]:.4f}')

    # ---- Table 3: convergence
    rep('\nTable 3. Finite-difference convergence at T = 1, kappa = 1.5625, q = 0 (|z| <= 3)')
    runs = [(0.05, 1e-4), (0.025, 1e-4), (0.0125, 1e-4), (0.025, 1e-3), (0.025, 1e-5)]
    sols = {r: solve(f1, f2, SIGMA, 0, 0, 0.5, T=1.0, dz=r[0], t0=r[1], switching=False) for r in runs}
    zr, Pr = sols[(0.0125, 1e-4)]['z'], sols[(0.0125, 1e-4)]['Pi']
    zz = np.linspace(-3, 3, 241)
    ref = np.interp(zz, zr, Pr)
    rep(f'{"dz":>7s} {"t0":>7s} {"pi(1,.) at z=-2,0,2":>24s} {"max diff":>10s} {"mass err":>10s}')
    for r in runs:
        z, P = sols[r]['z'], sols[r]['Pi']
        d = 'reference' if r == (0.0125, 1e-4) else f'{np.abs(np.interp(zz, z, P) - ref).max():.1e}'
        rep(f'{r[0]:7.4f} {r[1]:7.0e} {", ".join(f"{v:.4f}" for v in at(z, P, (-2, 0, 2))):>24s} '
            f'{d:>10s} {abs(regime_mass(z, P) - 0.5):10.1e}')

    # ---- Figure 2
    s0 = solve(f1, f2, SIGMA, 0, 0, 0.5, T=1.0, switching=False)
    s2 = solve(f1, f2, SIGMA, 2, 2, 0.5, T=1.0)
    z = s0['z']
    rep('\nFigure 2 data at T = 1 (kappa = 1.5625): z, no switching, q = 2, small-time profile')
    for zi in np.arange(-3, 3.01, 0.5):
        rep(f'  {zi:5.1f} {np.interp(zi, z, s0["Pi"]):.4f} {np.interp(zi, z, s2["Pi"]):.4f} {Pi0(zi):.4f}')
    rep(f'Shifted profile at z = -2, +2: {Pi0(-2 - 0.1):.3f}, {Pi0(2 - 0.1):.3f}')
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        m = np.abs(z) <= 3
        ax.plot(z[m], s0['Pi'][m], color='navy', lw=1.8, label='Black–Scholes, no switching')
        ax.plot(z[m], s2['Pi'][m], color='gray', lw=1.8, label='Black–Scholes, switching $q=2$')
        ax.plot(z[m], Pi0(z[m]), 'k--', lw=1, label='Small-time profile (Theorem 5.1)')
        ax.set_xlabel(r'Standardised log-return $z=(x-m(1))/\sigma$'); ax.set_ylabel(r'$\pi(1,x)$')
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=8)
        fig.tight_layout(); fig.savefig(f'{FIGURES}/figure2_bs_profile.png', dpi=200)
    except ImportError:
        pass
    rep.save()


if __name__ == '__main__':
    main()
