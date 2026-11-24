# hybrid-fake-gbm

Code for the paper

> Y. Alsaedi, *Explicit structure and hedging risk in Black–Scholes-consistent regime-switching volatility models* (AIMS Mathematics, 2026).

The code reproduces every numerical result in the paper. This covers:

- the closed-form profile of Theorem 5.1;
- the finite-difference solution of the Black–Scholes calibration equation (Section 7);
- the Monte Carlo calibration, pricing and hedging studies (Sections 7–9);
- the Markov-switching estimation on S&P 500 data (Section 10).

## Installation

```bash
git clone https://github.com/yasiralsaedi53/hybrid-fake-gbm.git
cd hybrid-fake-gbm
pip install -r requirements.txt
```

The code requires Python 3.9 or later, NumPy, SciPy and Matplotlib.

## Package `hfgbm`

| Module | Content | Paper |
|---|---|---|
| `profile.py` | Closed-form profile Π(z) = Φ⁻¹(z²/2 + C) and the constant C | Theorem 5.1, Figure 1 |
| `pde.py` | Explicit finite-difference solver for the calibration equation in (τ, z) | Section 7, equation (8) |
| `mc.py` | Leverage read from the PDE solution; Euler simulation of the hybrid model and of geometric Brownian motion with the same random numbers | Sections 7–10 |
| `claims.py` | Payoffs of the eight claims, Brownian-bridge correction, implied volatilities | Section 8 |
| `hedging.py` | Black–Scholes delta hedge and the gamma floor | Proposition 9.1, Section 9 |
| `bs.py` | Black–Scholes and Bachelier formulas; continuously monitored up-and-out call | |
| `msem.py` | Two-state Markov-switching model: EM (Hamilton filter, Kim smoother), Student-*t* likelihood, generator Q = 252 log P | Section 10 |
| `data.py` | Reader for Investing.com and Yahoo Finance CSV files | Section 10 |
| `experiments.py` | Pricing and hedging studies shared by the scripts | |

## Reproducing the paper

Each script prints its results and writes them to `results/`; figures go to `figures/`. The `results/` and `figures/` folders already contain the output of a full run. To regenerate everything:

```bash
bash run_all.sh data/SP500_part1.csv data/SP500_part2.csv
```

| Script | Reproduces | Paths | Approx. time* |
|---|---|---|---|
| `scripts/table1_closed_form.py` | Table 1, Figure 1 | 200,000 | 1 min |
| `scripts/table2_3_pde.py` | Table 2, Table 3, scheme checks, Figure 2 | – | 3 min |
| `scripts/calibration_check.py` | Section 7.4 (flat surface), Figure 3 | 400,000 | 5 min |
| `scripts/mcconv.py` | Section 7.5 (step size, continuous monitoring) | 200,000 | 7 min |
| `scripts/table4_model_risk.py` | Table 4 | 400,000 | 4 min |
| `scripts/table5_hedging.py` | Table 5, Figure 4 | 100,000 | 6 min |
| `scripts/empirical.py <csv> [<csv> ...]` | Tables 6–8, Student-*t* robustness | 200,000 | 16 min |

\*Measured on one core of a standard cloud machine. Add `--quick` to any Monte Carlo script for a run with one tenth of the paths, or use `--paths N`. `empirical.py --no-mc` performs only the estimation (Table 6 and the Student-*t* fits), which takes about one minute.

The finite-difference scheme (Section 7.3) works as follows:

- central differences on z ∈ [−7, 7] with Δz = 0.025 and zero-flux boundaries;
- time step Δτ = 0.35 Δz²/κ²;
- initial data from the closed-form profile at t₀ = 10⁻⁴.

The Monte Carlo set-up (Section 7.4) works as follows:

- σ = 20%, S₀ = 100, r = 0, T = 1, π₀ = 0.5;
- f₁ = κ^(−1/2) and f₂ = κ^(1/2);
- symmetric intensities q;
- 500 Euler steps;
- the leverage is interpolated linearly from the PDE solution.

Monte Carlo figures agree with the paper to within the reported confidence intervals. They do not agree to the last digit, because the random-number streams differ.

## Data

The S&P 500 closing levels are **not** included in this repository. See [`data/README.md`](data/README.md) for how to download them. The reader accepts:

- Investing.com exports: one or more overlapping files; the reader merges them and checks that they agree;
- Yahoo Finance exports (ticker `^GSPC`).

## Numbering

Theorem, figure and table numbers follow the published version of the paper. Theorem-like statements have a separate counter for each type, for example Theorem 7.1 (Black–Scholes case without switching) and Proposition 9.1 (hedging error).

## Citation

```bibtex
@article{Alsaedi2026,
  author  = {Alsaedi, Yasir},
  title   = {Explicit structure and hedging risk in {Black--Scholes}-consistent regime-switching volatility models},
  journal = {AIMS Mathematics},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).
