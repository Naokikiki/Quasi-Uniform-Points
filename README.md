# Space-Filling Lattice Designs for Computer Experiments

[![arXiv](https://img.shields.io/badge/arXiv-2602.15390-b31b1b.svg)](https://arxiv.org/abs/2602.15390)

This repository contains the code and numerical experiments for the paper:

> **Space-filling lattice designs for computer experiments**  
> Naoki Sakai, Takashi Goda  
> arXiv: [2602.15390](https://doi.org/10.48550/arXiv.2602.15390)

We investigate the construction of space-filling designs for computer experiments, characterized by the quasi-uniformity criterion that integrates covering and separation radii. We propose two construction algorithms based on quasi-Monte Carlo (QMC) lattice point sets:

1. **Explicit Rank-1 Lattice** — generates rank-1 lattice point sets as an approximation of quasi-uniform Kronecker sequences, where the generating vector is determined explicitly.
2. **Korobov Lattice** — employs the LLL basis reduction algorithm to identify the generating parameter that ensures quasi-uniformity.

The numerical study compares the proposed designs against Halton/Sobol' sequences, random points, and a **maximin Latin hypercube design**. Beyond verifying the optimal separation-radius decay, it (i) shows the separation radius controls the **condition number** of the kernel matrix, (ii) confirms the LLL-based Korobov search returns a **near-optimal** generator, and (iii) evaluates approximation quality in Gaussian process regression across a range of test functions.

## Repository Structure

### Package: `lattice_qmc/`

| File | Description |
|------|-------------|
| `korobov.py` | Korobov lattice construction with LLL-based parameter optimization |
| `explicit_rank1.py` | Explicit rank-1 lattice construction via Kronecker sequence approximation |
| `utils.py` | Utility functions (prime generation, separation radius computation, LLL reduction, etc.) |
| `baselines.py` | Maximin Latin hypercube design, general-dimension Genz / pairwise-trigonometric test functions, and the Matérn-5/2 Gram matrix used for the conditioning study |

### Experiments

The numerical experiments are organized to match **Section 4** of the paper.

#### Section 4.1: Empirical Properties of Two Constructions

| Notebook | Section | Description |
|----------|---------|-------------|
| `experiments_04_01_01_parameter_sequences.ipynb` | §4.1.1 | **Construction parameters and the number of points.** Examines how the choice of parameters relates to the number of points in each construction. Generates the point-count tables: comparison of point counts for the explicit construction with different primes *p* and dimensions *d*, and optimal Korobov parameters *a\** for selected primes. |
| `experiments_04_01_02_upper_bound.ipynb` | §4.1.2 | **Bounds on mesh ratio.** Numerically examines how the upper bound *d√d / (λ₁(Λ) λ₁(Λ⊥))* behaves as a function of *N*. Confirms that both constructions maintain bounded mesh ratios across all tested dimensions (*d* = 2, 3, 5, 7). |
| `experiments_04_01_03_separation_radius.ipynb` | §4.1.3 | **Separation radius.** Verifies that both lattice constructions attain the optimal decay rate Θ(*N*⁻¹ᐟᵈ) for the separation radius, while a maximin Latin hypercube design, Halton, Sobol', and random points deviate from this optimal rate. |
| `experiments_04_01_04_condition_number.ipynb` | §4.1.4 | **Conditioning of the kernel matrix.** Plots the spectral condition number of the Matérn-5/2 Gram matrix vs *N* for all six designs (*d* = 2, 3, 5, 7), validating that a larger separation radius yields a better-conditioned kernel matrix. |
| `experiments_04_01_05_lll_optimality.ipynb` | §4.1.5 | **Near-optimality of the LLL-based search.** Compares the LLL surrogate used by the Korobov search against the *exact* shortest primal/dual vectors (computed by congruence enumeration) for every candidate generator in low dimensions, confirming the selected *a\** is essentially optimal. |

#### Section 4.2: Gaussian Process Regression

| Notebook | Section | Description |
|----------|---------|-------------|
| `experiments_04_02_03_gpr_2d.ipynb` | §4.2 | **GPR in 2D.** Evaluates GPR performance using the Franke function, Arctan function, Gaussian peak, and oscillatory function on [0,1]². Uses Gauss–Legendre quadrature (50 nodes/dim) for *L*² error computation. |
| `experiments_04_02_03_gpr_3d.ipynb` | §4.2 | **GPR in 3D.** Evaluates GPR performance using two Genz family functions (Continuous and Gaussian) on [0,1]³. Uses Monte Carlo integration (10,000 samples) for *L*² error computation. |
| `experiments_04_02_03_gpr_5d.ipynb` | §4.2 | **GPR in 5D.** Evaluates GPR performance using the pairwise trigonometric function **and the Genz Continuous / Gaussian functions** on [0,1]⁵. Uses Monte Carlo integration (10,000 samples) for *L*² error computation. |
| `experiments_04_02_03_gpr_7d.ipynb` | §4.2 | **GPR in 7D.** Evaluates GPR performance using the pairwise trigonometric function **and the Genz Continuous / Gaussian functions** on [0,1]⁷. Uses Monte Carlo integration (10,000 samples) for *L*² error computation. |

All GPR experiments compare six types of point sets (Korobov lattice, explicit rank-1 lattice, Halton sequence, Sobol' sequence, maximin Latin hypercube design, and random sampling) using a Matérn kernel (ν = 2.5) with observation noise σ_noise = 0.05, averaged over 100 independent noise realizations.

The Korobov and explicit rank-1 lattices are the only designs that deterministically contain the origin (the *n*=0 point), whereas QMCPy randomizes (scrambles) the Halton/Sobol' sequences by default. To place all designs on an equal footing, the GPR experiments apply a fixed toroidal half-shift `(x + 0.5) mod 1` to the two lattice designs; this leaves their separation/covering radii (hence quasi-uniformity) unchanged.

### Standalone scripts (`revision/`)

Standalone, parallelized equivalents of the experiment notebooks, used to generate the manuscript figures. Each writes its figures into `Figures/` and a results pickle into `revision/`. They are interchangeable with the notebooks above (same constructions and settings).

| Script | Section | Description |
|--------|---------|-------------|
| `revision/run_separation_radius.py` | §4.1.3 | Separation radius vs *N*, now including a **maximin LHS** baseline (`Separation_radius_d=*.png`). |
| `revision/run_condition_number.py` | §4.1.4 | Spectral condition number of the Matérn-5/2 Gram matrix vs *N* for all six designs, validating the separation-radius → conditioning claim (`Condition_number_d=*.png`). |
| `revision/run_lll_optimality.py` | §4.1.5 | Near-optimality study: exact shortest-vector lengths (by congruence enumeration) vs the LLL surrogate used by Algorithm 3.2 (`LLL_optimality.png`). |
| `revision/run_gpr_all.py` | §4.2 | All GPR experiments (all dimensions and test functions, six designs incl. maximin LHS, Genz functions added in *d*=5,7), parallelized over noise trials. |

The maximin LHS and the general-dimension Genz/test functions live in `lattice_qmc/baselines.py`.

Run them with the project virtualenv, e.g. `.venv/bin/python revision/run_condition_number.py`.

### Data

| File | Description |
|------|-------------|
| `data/korobov_optimal_parameters.npy` | Precomputed optimal Korobov parameters *a\** for selected primes and dimensions (generated by `experiments_04_01_01`) |

## Installation

```bash
pip install numpy scipy matplotlib scikit-learn qmcpy joblib
```

> **Note on figures.** Image files (`*.png`, `*.pdf`) and `data/` are git-ignored; the figures are regenerated by running the notebooks (or the `revision/` scripts). The notebooks store their plots inline, so they render directly on GitHub.

## Quick Start

```python
from lattice_qmc import KorobovLattice, ExplicitRank1Lattice

# Korobov lattice with optimal generator
lattice = KorobovLattice(d=3, N=127, generator=102)
points = lattice.points  # shape (127, 3) in [0, 1]^3

# Explicit rank-1 lattice
import numpy as np
p = 2
d = 3
alpha = np.array([p**(j/(d+1)) for j in range(1, d+1)])
lattice = ExplicitRank1Lattice(d=d, m=10, alpha=alpha)
points = lattice.points  # shape (Q, 3) in [0, 1]^3

# Maximin Latin hypercube baseline
from lattice_qmc.baselines import maximin_lhs
lhs_points = maximin_lhs(n=128, d=3, seed=2024)  # shape (128, 3) in [0, 1)^3
```

## Citation

```bibtex
@article{sakai2026space,
  title={Space-filling lattice designs for computer experiments},
  author={Sakai, Naoki and Goda, Takashi},
  journal={arXiv preprint arXiv:2602.15390},
  year={2026}
}
```

## License

MIT License
