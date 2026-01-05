# Quasi-Uniform Lattice Point Sets

This package provides implementations of quasi-uniform lattice point sets with bounded mesh ratios for numerical integration and Gaussian process regression.

## Installation

### From source
```bash
pip install -e .
```

### For Google Colab
```python
!pip install numpy scipy matplotlib qmcpy scikit-learn
```

## Quick Start

```python
from lattice_qmc import KorobovLattice, ExplicitRank1Lattice

# Korobov lattice with prime N points
korobov = KorobovLattice(d=3, N=101)
print(f"Korobov: {korobov.Q} points, separation radius = {korobov.separation_radius():.4f}")

# Explicit rank-1 lattice (Q determined by construction)
explicit = ExplicitRank1Lattice(d=3, m=8)
print(f"Explicit: {explicit.Q} points, separation radius = {explicit.separation_radius():.4f}")
```

## Main Features

1. **Korobov Lattice Selection**: Selects the generator that maximizes the shortest vector in the dual lattice.

2. **Explicit Rank-1 Lattice**: Constructs lattices by approximating Kronecker sequences.

3. **Mesh Ratio Analysis**: Computes upper bounds on mesh ratios.

4. **Comparison Tools**: Compare with random, Halton, and Sobol sequences.

## Citation

```bibtex
@article{your_paper,
  title={Quasi-uniform lattice point sets for numerical integration},
  author={Your Name},
  journal={SIAM Journal on Scientific Computing},
  year={2024}
}
```

## License

MIT License
