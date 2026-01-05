"""
Korobov Lattice Construction with Configurable Selection Criterion
===================================================================

This module implements the Korobov lattice construction, where the generating
vector is selected based on maximizing a configurable criterion:

    - "dual": λ₁(Λ^⊥)  [shortest vector in dual lattice]
    - "primal": λ₁(Λ)  [shortest vector in primal lattice]
    - "product": λ₁(Λ) × λ₁(Λ^⊥)  [product of shortest vectors]

A Korobov lattice with N points in d dimensions is defined by:
    P_N = { ({k/N}, {k*a/N}, {k*a^2/N}, ..., {k*a^{d-1}/N}) : k = 0, 1, ..., N-1 }

where {x} denotes the fractional part and a is the generator.

The generating matrix T for z = (1, a, a², ..., a^{d-1}) mod N is:
    T = [[1/N, 0, ..., 0],
         [a/N, 1, ..., 0],
         [..., ..., ..., ...],
         [a^{d-1}/N, 0, ..., 1]]

The dual lattice basis is computed as T^{-T} (inverse transpose of T).

References
----------
[1] Korobov, N.M. (1959). The approximate computation of multiple integrals.
[2] Sloan, I.H. and Joe, S. (1994). Lattice Methods for Multiple Integration.
"""

import numpy as np
from typing import Optional, Tuple, Literal
from .utils import (
    shortest_vector_length,
    compute_separation_radius_fast,
    compute_mesh_ratio_upper_bound,
)

# Valid criterion types
CriterionType = Literal["dual", "primal", "product"]


class KorobovLattice:
    """
    Korobov lattice with configurable selection criterion.
    
    The construction searches over all valid generators a ∈ {1, ..., N-1}
    and selects the one that maximizes the chosen criterion.
    
    Parameters
    ----------
    d : int
        Dimension of the lattice.
    N : int
        Number of points (should be prime for best results).
    generator : int, optional
        If provided, use this generator directly instead of searching.
    criterion : str, optional
        Selection criterion. One of:
        - "dual": maximize λ₁(Λ^⊥) [shortest vector in dual] (default)
        - "primal": maximize λ₁(Λ) [shortest vector in primal]
        - "product": maximize λ₁(Λ) × λ₁(Λ^⊥) [product]
    verbose : bool, optional
        If True, print progress information (default: False).
    
    Attributes
    ----------
    d : int
        Dimension.
    N : int
        Number of points.
    generator : int
        Selected generator a.
    generating_vector : np.ndarray
        The generating vector (1, a, a^2, ..., a^{d-1}) mod N.
    generating_matrix : np.ndarray
        The generating matrix T of shape (d, d).
    points : np.ndarray
        Generated point set of shape (N, d).
    lambda1_primal : float
        Shortest vector length in primal lattice λ₁(Λ).
    lambda1_dual : float
        Shortest vector length in dual lattice λ₁(Λ^⊥).
    criterion : str
        The criterion used for selection.
    
    Examples
    --------
    >>> lattice = KorobovLattice(d=3, N=101, criterion="product")
    >>> points = lattice.points
    >>> print(f"Generator: {lattice.generator}")
    >>> print(f"λ₁(Λ) × λ₁(Λ^⊥): {lattice.lambda1_primal * lattice.lambda1_dual:.6f}")
    
    >>> lattice_dual = KorobovLattice(d=3, N=101, criterion="dual")
    >>> print(f"λ₁(Λ^⊥): {lattice_dual.lambda1_dual:.4f}")
    """
    
    VALID_CRITERIA = ("dual", "primal", "product")
    
    def __init__(
        self,
        d: int,
        N: int,
        generator: Optional[int] = None,
        criterion: str = "dual",
        verbose: bool = False
    ):
        if criterion not in self.VALID_CRITERIA:
            raise ValueError(
                f"Invalid criterion '{criterion}'. "
                f"Must be one of {self.VALID_CRITERIA}"
            )
        
        self.d = d
        self.N = N
        self.criterion = criterion
        self.verbose = verbose
        
        if generator is not None:
            self.generator = generator
            self.generating_vector = self._compute_generating_vector(generator)
            self.generating_matrix = self._compute_generating_matrix(self.generating_vector)
            self.lambda1_primal = self._compute_primal_lambda1(self.generating_matrix)
            self.lambda1_dual = self._compute_dual_lambda1(self.generating_matrix)
        else:
            result = self._find_best_generator()
            self.generator = result[0]
            self.generating_vector = result[1]
            self.generating_matrix = result[2]
            self.lambda1_primal = result[3]
            self.lambda1_dual = result[4]
        
        self._points = None
        self._det_T = 1.0 / N  # Determinant of generating matrix for rank-1 lattice

    
    def _compute_generating_vector(self, a: int) -> np.ndarray:
        """
        Compute the generating vector (1, a, a^2, ..., a^{d-1}) mod N.
        
        Note: For Korobov lattices, z_1 = 1, so gcd(N, z_1) = 1 always holds.
        
        Parameters
        ----------
        a : int
            The generator.
        
        Returns
        -------
        np.ndarray
            Generating vector of shape (d,).
        """
        z = np.zeros(self.d, dtype=np.int64)
        power = 1
        for j in range(self.d):
            z[j] = power % self.N
            power = (power * a) % self.N
        return z
    
    def _compute_generating_matrix(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the generating matrix T for the rank-1 lattice.
        
        For z = (z_1, z_2, ..., z_d) with gcd(N, z_1) = 1, the generating matrix is:
            T = [[1/N,        0,    ..., 0],
                 [z_1^{-1}*z_2/N, 1,    ..., 0],
                 [...,         ..., ..., ...],
                 [z_1^{-1}*z_d/N, 0,    ..., 1]]
        
        where z_1^{-1} is the multiplicative inverse of z_1 modulo N.
        
        For Korobov lattices, z_1 = 1, so z_1^{-1} = 1 and T simplifies to:
            T = [[1/N,    0,    ..., 0],
                 [z_2/N,  1,    ..., 0],
                 [...,    ..., ..., ...],
                 [z_d/N,  0,    ..., 1]]
        
        Parameters
        ----------
        z : np.ndarray
            Generating vector (z_1, z_2, ..., z_d).
        
        Returns
        -------
        np.ndarray
            Generating matrix T of shape (d, d).
        """
        d = len(z)
        N = self.N
        
        # Compute z_1^{-1} mod N (multiplicative inverse)
        z1 = int(z[0])
        z1_inv = pow(z1, -1, N)  # Python 3.8+ modular inverse
        
        # Build the generating matrix T
        T = np.eye(d, dtype=np.float64)
        T[0, 0] = 1.0 / N
        for i in range(1, d):
            T[i, 0] = (z1_inv * int(z[i]) % N) / N
        
        return T
    
    def _compute_dual_lattice_basis(self, T: np.ndarray) -> np.ndarray:
        """
        Compute the dual lattice basis as T^{-T} (inverse transpose of T).
        
        For a lattice with generating matrix T, the dual lattice has
        generating matrix T^{-T} = (T^{-1})^T.
        
        Parameters
        ----------
        T : np.ndarray
            Generating matrix of shape (d, d).
        
        Returns
        -------
        np.ndarray
            Dual lattice basis matrix of shape (d, d).
        """
        T_inv = np.linalg.inv(T)
        return T_inv.T
    
    def _compute_primal_lambda1(self, T: np.ndarray) -> float:
        """
        Compute the shortest vector length in the primal lattice.
        
        Columns of T are the basis vectors for the primal lattice.
        Uses LLL reduction to find an approximately shortest vector.
        
        Parameters
        ----------
        T : np.ndarray
            Generating matrix (columns = basis vectors).
        
        Returns
        -------
        float
            Length of the shortest vector in primal lattice λ₁(Λ).
        """
        # utils expects rows as basis vectors, so transpose T
        return shortest_vector_length(T.T, use_lll=True)
    
    def _compute_dual_lambda1(self, T: np.ndarray) -> float:
        """
        Compute the shortest vector length in the dual lattice.
        
        The dual lattice basis (columns = basis vectors) is T^{-T}.
        Uses LLL reduction to find an approximately shortest vector.
        
        Parameters
        ----------
        T : np.ndarray
            Generating matrix (columns = basis vectors).
        
        Returns
        -------
        float
            Length of the shortest vector in dual lattice λ₁(Λ^⊥).
        """
        dual_basis = self._compute_dual_lattice_basis(T)
        # utils expects rows as basis vectors, so transpose dual_basis
        return shortest_vector_length(dual_basis.T, use_lll=True)
    
    def _compute_criterion_value(
        self, 
        lambda1_primal: float, 
        lambda1_dual: float
    ) -> float:
        """
        Compute the criterion value based on the selected criterion.
        
        Parameters
        ----------
        lambda1_primal : float
            Shortest vector length in primal lattice λ₁(Λ).
        lambda1_dual : float
            Shortest vector length in dual lattice λ₁(Λ^⊥).
        
        Returns
        -------
        float
            Criterion value (higher is better for all criteria).
        """
        if self.criterion == "dual":
            # λ₁(Λ^⊥)
            return lambda1_dual
        elif self.criterion == "primal":
            # λ₁(Λ)
            return lambda1_primal
        elif self.criterion == "product":
            # λ₁(Λ) × λ₁(Λ^⊥)
            return lambda1_primal * lambda1_dual
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _find_best_generator(self) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
        """
        Find the generator that maximizes the selected criterion.
        
        Returns
        -------
        best_a : int
            Best generator.
        best_z : np.ndarray
            Corresponding generating vector.
        best_T : np.ndarray
            Corresponding generating matrix.
        best_lambda1_primal : float
            Shortest vector length in primal lattice λ₁(Λ).
        best_lambda1_dual : float
            Shortest vector length in dual lattice λ₁(Λ^⊥).
        """
        best_a = 1
        best_z = None
        best_T = None
        best_lambda1_primal = 0.0
        best_lambda1_dual = 0.0
        best_criterion_value = -np.inf
        
        # Search over all generators 1 <= a < N
        # For prime N, all a in {1, ..., N-1} are valid
        candidates = range(1, self.N)
        
        criterion_labels = {
            "dual": "λ₁(Λ^⊥)",
            "primal": "λ₁(Λ)",
            "product": "λ₁(Λ)×λ₁(Λ^⊥)"
        }
        
        if self.verbose:
            print(f"Searching {self.N - 1} candidate generators "
                  f"(criterion: {criterion_labels[self.criterion]})...")
        
        for idx, a in enumerate(candidates):
            z = self._compute_generating_vector(a)
            T = self._compute_generating_matrix(z)
            lambda1_primal = self._compute_primal_lambda1(T)
            lambda1_dual = self._compute_dual_lambda1(T)
            criterion_value = self._compute_criterion_value(lambda1_primal, lambda1_dual)
           
            if criterion_value > best_criterion_value:
                best_criterion_value = criterion_value
                best_lambda1_primal = lambda1_primal
                best_lambda1_dual = lambda1_dual
                best_a = a
                best_z = z.copy()
                best_T = T.copy()
            
            if self.verbose and (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{self.N - 1} generators, "
                      f"best {criterion_labels[self.criterion]} = {best_criterion_value:.6f}")
        
        if self.verbose:
            print(f"Best generator: a = {best_a}")
            print(f"  λ₁(Λ) = {best_lambda1_primal:.6f}")
            print(f"  λ₁(Λ^⊥) = {best_lambda1_dual:.6f}")
            print(f"  λ₁(Λ)×λ₁(Λ^⊥) = {best_lambda1_primal * best_lambda1_dual:.6f}")
        
        return best_a, best_z, best_T, best_lambda1_primal, best_lambda1_dual
    
    @property
    def points(self) -> np.ndarray:
        """
        Generate the lattice point set.
        
        Returns
        -------
        np.ndarray
            Point set of shape (N, d) in [0, 1)^d.
        """
        if self._points is None:
            self._points = np.zeros((self.N, self.d))
            for k in range(self.N):
                self._points[k] = (k * self.generating_vector / self.N) % 1.0
        return self._points
    
    def separation_radius(self) -> float:
        """Compute the separation radius of the point set."""
        return compute_separation_radius_fast(self.points, toroidal=True)
    
    def mesh_ratio_upper_bound(self) -> float:
        """
        Compute the upper bound on mesh ratio using primal and dual lattice.
        
        Uses the bound:
            ρ(Λ) ≤ d√d / (λ₁(Λ) × λ₁(Λ^⊥))
        """
        return compute_mesh_ratio_upper_bound(
            lambda1_primal=self.lambda1_primal,
            lambda1_dual=self.lambda1_dual,
            d=self.d
        )
    
    def info(self) -> dict:
        """Return a dictionary with lattice information."""
        return {
            "type": "Korobov",
            "criterion": self.criterion,
            "dimension": self.d,
            "num_points": self.N,
            "generator": self.generator,
            "generating_vector": self.generating_vector.tolist(),
            "lambda1_primal": self.lambda1_primal,
            "lambda1_dual": self.lambda1_dual,
            "separation_radius": self.separation_radius(),
            "mesh_ratio_upper_bound": self.mesh_ratio_upper_bound(),
        }
    
    def __repr__(self) -> str:
        product = self.lambda1_primal * self.lambda1_dual
        return (f"KorobovLattice(d={self.d}, N={self.N}, criterion='{self.criterion}', "
                f"generator={self.generator}, λ₁(Λ)×λ₁(Λ^⊥)={product:.6f})")