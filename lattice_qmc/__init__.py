"""
Quasi-Uniform Lattice Point Sets for Numerical Integration
==========================================================

This package provides implementations of quasi-uniform lattice point sets
with bounded mesh ratios, designed for Gaussian
process regression.

Main classes:
- KorobovLattice: Korobov lattice selection via lattice and dual lattice shortest vector
- ExplicitRank1Lattice: Explicit construction via Kronecker sequence approximation

Reference:
    Space-filling lattice designs for computer experiments

Author: Naoki Sakai, Takashi Goda 
License: MIT
"""

from .korobov import KorobovLattice
from .explicit_rank1 import ExplicitRank1Lattice
from .utils import (
    compute_separation_radius,
    compute_mesh_ratio_upper_bound,
    generate_primes,
)

__version__ = "1.0.0"
__all__ = [
    "KorobovLattice",
    "ExplicitRank1Lattice",
    "compute_separation_radius",
    "compute_mesh_ratio_upper_bound",
    "generate_primes",
]
