"""Fast checks against values reported in the paper (run: python -m pytest tests)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfgbm import bs, profile, pde, msem


def test_closed_form_profile():
    Pi, C = profile.profile(0.64, 1.5625, 0.5)
    assert abs(C + 1.2705) < 1e-4
    assert abs(Pi(0.0) - 0.3859) < 1e-4 and abs(Pi(3.0) - 0.995) < 1e-3
    assert abs(profile.conditional_probability(Pi, 0, 0.25) - 0.388) < 1e-3


def test_black_scholes_values():
    assert abs(bs.call(100, 100, 0.2, 1.0) - 7.966) < 1e-3
    vals = [bs.up_and_out_call_continuous(100, 100, 120, s, 1.0) for s in (0.15, 0.2, 0.25)]
    assert np.allclose(vals, [1.855, 1.105, 0.671], atol=2e-3)


def test_pde_no_switching():
    k = 1.5625
    s = pde.solve(k ** -0.5, k ** 0.5, 0.2, 0, 0, 0.5, T=1.0, switching=False)
    vals = [np.interp(x, s['z'], s['Pi']) for x in (-2, 0, 2)]
    assert np.allclose(vals, [0.8869, 0.3860, 0.8222], atol=2e-4)
    assert abs(pde.regime_mass(s['z'], s['Pi']) - 0.5) < 1e-4


def test_em_recovers_parameters():
    P = np.array([[0.99, 0.01], [0.03, 0.97]])
    r = msem.simulate([0.0005, -0.001], [0.007, 0.02], P, 8000, seed=3)
    f = msem.fit_gaussian(r)
    assert abs(f['sd'][1] / f['sd'][0] - 0.02 / 0.007) < 0.15
    assert np.abs(f['P'] - P).max() < 0.01


def test_generator_closed_form():
    from scipy.linalg import logm
    P = np.array([[0.988, 0.012], [0.028, 0.972]])
    assert np.allclose(msem.generator(P), 252 * logm(P).real)
