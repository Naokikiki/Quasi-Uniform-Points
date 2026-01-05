"""
Explicit Rank-1 Lattice Construction via Kronecker Sequence Approximation
==========================================================================

This module implements the explicit rank-1 lattice construction based on
approximating the Kronecker sequence nα by a rank-1 lattice.

The construction finds denominators Q that provide good simultaneous
Diophantine approximations to the algebraic vector α, then constructs
a rank-1 lattice with Q points.

The key advantage is that Q is determined by the construction (not freely
chosen), and the resulting lattice has provably bounded mesh ratio.

Reference:
    [Your paper citation here]
"""

import numpy as np
from math import gcd
from functools import reduce
from typing import Optional, Tuple, List
from itertools import product

from .utils import (
    multi_gcd,
    compute_separation_radius_fast,
    compute_mesh_ratio_upper_bound,
    lll_reduce,
    shortest_vector_length,
)


class ExplicitRank1Lattice:
    """
    Explicit rank-1 lattice construction via Kronecker sequence approximation.
    
    This construction approximates the Kronecker sequence {nα} by a rank-1
    lattice. The number of points Q is determined by the construction
    parameters (d, m, α) and cannot be freely chosen.
    
    Parameters
    ----------
    d : int
        Dimension of the lattice.
    m : int
        Number of iterations to find the best approximation denominator.
        Larger m gives more points but better approximation.
    alpha : np.ndarray, optional
        Generating vector for Kronecker sequence. If None, uses the
        default α = (2^{1/(d+1)}, 2^{2/(d+1)}, ..., 2^{d/(d+1)}).
    verbose : bool, optional
        If True, print progress information (default: False).
    
    Attributes
    ----------
    d : int
        Dimension.
    m : int
        Approximation parameter.
    alpha : np.ndarray
        Kronecker generating vector.
    Q : int
        Number of points (determined by construction).
    P : np.ndarray
        Numerator vector of the rank-1 lattice.
    points : np.ndarray
        Generated point set of shape (Q, d).
    Q_list : np.ndarray
        Sequence of denominators found during construction.
    
    Examples
    --------
    >>> lattice = ExplicitRank1Lattice(d=3, m=8)
    >>> print(f"Number of points Q: {lattice.Q}")
    >>> print(f"Numerator vector P: {lattice.P}")
    >>> points = lattice.points
    
    Notes
    -----
    The choice of α = (2^{1/(d+1)}, ..., 2^{d/(d+1)}) ensures that
    {1, α_1, ..., α_d} are linearly independent over Q (as elements
    of an algebraic extension), which is required for the construction.
    """
    
    def __init__(
        self,
        d: int,
        m: int,
        alpha: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        self.d = d
        self.m = m
        self.verbose = verbose
        
        # Default alpha: 2^{j/(d+1)} for j = 1, ..., d
        if alpha is None:
            self.alpha = np.array([2.0 ** (j / (d + 1)) for j in range(1, d + 1)])
        else:
            self.alpha = np.asarray(alpha, dtype=np.float64)
            if len(self.alpha) != d:
                raise ValueError(f"alpha must have length d={d}, got {len(self.alpha)}")
        
        # Find sequence of best approximation denominators
        self.Q_list = self._find_best_denominators()
        self.Q = self.Q_list[-1]
        
        # Find numerator vector P
        self.P = self._find_numerator_vector()
        
        self._points = None
        self._lambda1_primal = None
        self._lambda1_dual = None
    
    def _find_best_denominators(self) -> np.ndarray:
        """
        Find a sequence of denominators Q_i that decrease the max norm.
        
        Starting from Q_0 = 1, we search for Q_{i+1} > Q_i such that:
            max_j |Q_{i+1} * α_j - round(Q_{i+1} * α_j)| < max_j |Q_i * α_j - round(Q_i * α_j)|
        
        Returns
        -------
        np.ndarray
            Array of m+1 denominators Q_0, Q_1, ..., Q_m.
        """
        Q_list = np.zeros(self.m + 1, dtype=np.int64)
        Q_list[0] = 1
        
        for i in range(self.m):
            q_prev = Q_list[i]
            maxnorm_prev = np.max(np.abs(q_prev * self.alpha - np.round(q_prev * self.alpha)))
            
            q_search = q_prev + 1
            while True:
                maxnorm_search = np.max(np.abs(q_search * self.alpha - np.round(q_search * self.alpha)))
                if maxnorm_search < maxnorm_prev:
                    Q_list[i + 1] = q_search
                    if self.verbose:
                        print(f"  Q_{i+1} = {q_search}, max_norm = {maxnorm_search:.6e}")
                    break
                q_search += 1
        
        if self.verbose:
            print(f"Selected Q = {Q_list[-1]}")
        
        return Q_list
    
    def _find_numerator_vector(self) -> np.ndarray:
        """
        Find the numerator vector P for the rank-1 lattice.
        
        The numerator P must satisfy:
        1. |P_j - Q * α_j| is small for each j
        2. gcd(Q, P_1, ..., P_d) = 1
        
        When Q > 2^d, P is unique (round(Q * α)).
        When Q <= 2^d, we perform a local search.
        
        Returns
        -------
        np.ndarray
            Numerator vector P of shape (d,).
        """
        Q = self.Q
        
        if Q > 2**self.d:
            # P is unique
            if self.verbose:
                print(f"Note: Q > 2^d ({Q} > {2**self.d}). P is unique.")
            P = np.round(Q * self.alpha).astype(np.int64)
            
            # Check GCD condition
            if multi_gcd([Q] + list(P)) != 1:
                if self.verbose:
                    print("Warning: The unique P candidate does not satisfy gcd(Q, P) = 1.")
        else:
            # Local search for P
            if self.verbose:
                print(f"Note: Q <= 2^d ({Q} <= {2**self.d}). Performing local search for P.")
            
            epsilon = Q ** (-1.0 / self.d)
            P_candidates = []
            
            for j in range(self.d):
                center = Q * self.alpha[j]
                lb = int(np.ceil(center - epsilon))
                ub = int(np.floor(center + epsilon))
                
                # Generate candidates sorted by distance to center
                cands = list(range(lb, ub + 1))
                if not cands:
                    # Fallback: just use rounded value
                    cands = [int(np.round(center))]
                cands.sort(key=lambda x: abs(x - center))
                P_candidates.append(cands)
            
            # Search for P satisfying coprimality condition
            P = self._find_valid_P(P_candidates, Q)
            
            if P is None:
                if self.verbose:
                    print("Warning: No P satisfies gcd=1 within bounds. "
                          "Falling back to round(Q*alpha).")
                P = np.round(Q * self.alpha).astype(np.int64)
        
        # Reduce P modulo Q
        P = P % Q
        
        if self.verbose:
            print(f"Final Numerator Vector (P): {P}")
            print(f"GCD(Q, P_1, ..., P_d): {multi_gcd([Q] + list(P))}")
        
        return P
    
    def _find_valid_P(
        self,
        candidates: List[List[int]],
        Q: int
    ) -> Optional[np.ndarray]:
        """
        Search combinations of P candidates to find one with gcd(Q, P) = 1.
        
        Parameters
        ----------
        candidates : List[List[int]]
            For each dimension j, a list of candidate values for P_j.
        Q : int
            The denominator.
        
        Returns
        -------
        np.ndarray or None
            Valid P vector, or None if not found.
        """
        # Use itertools.product to iterate through all combinations
        # Candidates are already sorted by distance to center
        for P_tuple in product(*candidates):
            P = np.array(P_tuple, dtype=np.int64)
            if multi_gcd([Q] + list(P)) == 1:
                return P
        
        return None
    
    @property
    def points(self) -> np.ndarray:
        """
        Generate the lattice point set.
        
        Returns
        -------
        np.ndarray
            Point set of shape (Q, d) in [0, 1)^d.
        """
        if self._points is None:
            self._points = np.zeros((self.Q, self.d))
            for k in range(self.Q):
                self._points[k] = (k * self.P / self.Q) % 1.0
        return self._points
    
    @property
    def lambda1_primal(self) -> float:
        """Compute the shortest vector length in the primal lattice."""
        if self._lambda1_primal is None:
            self._lambda1_primal = self._compute_primal_lambda1()
        return self._lambda1_primal
    
    @property
    def lambda1_dual(self) -> float:
        """Compute the shortest vector length in the dual lattice."""
        if self._lambda1_dual is None:
            self._lambda1_dual = self._compute_dual_lambda1()
        return self._lambda1_dual
    
    def _compute_generating_matrix(self) -> np.ndarray:
        """
        Compute the generating matrix B for the rank-1 lattice.
        
        Find j such that gcd(g_j, N) = 1, then construct:
            B = [[1/N,             0,    ..., 0],
                 [g_j^{-1}*g_1/N,  1,    ..., 0],
                 [...,             ..., ..., ...],
                 [g_j^{-1}*g_d/N,  0,    ..., 1]]
        
        where the j-th row has only the 1/N entry (no other components in first column).
        """
        d = self.d
        N = int(self.Q)  # N = Q (number of points)
        g = self.P       # g = P (generating vector)
        
        # Find component j such that gcd(g_j, N) = 1
        pivot_idx = None
        for j in range(d):
            if gcd(int(g[j]), N) == 1:
                pivot_idx = j
                break
        
        B = np.eye(d, dtype=np.float64)
        
        if pivot_idx is not None:
            # Found g_j coprime to N
            gj = int(g[pivot_idx])
            gj_inv = pow(gj, -1, N)
            
            # Reorder: put pivot_idx first, then others
            # Row 0 corresponds to the pivot dimension
            B[0, 0] = 1.0 / N
            
            # Other rows get g_j^{-1} * g_i / N in first column
            row = 1
            for i in range(d):
                if i != pivot_idx:
                    B[row, 0] = (gj_inv * int(g[i]) % N) / N
                    row += 1
        else:
            # Fallback: no component coprime to N (shouldn't happen if gcd(N,g)=1)
            # Use simple structure
            B[0, 0] = 1.0 / N
            for i in range(1, d):
                B[i, 0] = float(g[i]) / N
        
        return B
    
    def _compute_primal_lattice_basis(self) -> np.ndarray:
        """Compute a basis for the primal lattice (the generating matrix T)."""
        return self._compute_generating_matrix()
    
    def _compute_primal_lambda1(self) -> float:
        """Compute shortest vector length in primal lattice."""
        primal_basis = self._compute_primal_lattice_basis()
        # Columns of T are the generating vectors; transpose for utils
        return shortest_vector_length(primal_basis.T, use_lll=True)
    
    def _compute_dual_lattice_basis(self) -> np.ndarray:
        """
        Compute a basis for the dual lattice as T^{-T}.
        """
        T = self._compute_generating_matrix()
        T_inv = np.linalg.inv(T)
        return T_inv.T
    
    def _compute_dual_lambda1(self) -> float:
        """Compute shortest vector length in dual lattice."""
        dual_basis = self._compute_dual_lattice_basis()
        # dual_basis has columns as basis vectors; transpose for utils
        return shortest_vector_length(dual_basis.T, use_lll=True)
    
    def separation_radius(self) -> float:
        """Compute the separation radius of the point set."""
        return compute_separation_radius_fast(self.points, toroidal=True)
    
    def mesh_ratio_upper_bound(self) -> float:
        """
        Compute the upper bound on mesh ratio.
        
        Uses the bound:
            ρ(Λ) ≤ d√d / (λ₁(Λ) × λ₁(Λ^⊥))
        """
        return compute_mesh_ratio_upper_bound(
            lambda1_primal=self.lambda1_primal,
            lambda1_dual=self.lambda1_dual,
            d=self.d
        )
    
    def approximation_error(self) -> float:
        """
        Compute the approximation error max_j |P_j/Q - α_j|.
        
        Returns
        -------
        float
            Maximum approximation error across dimensions.
        """
        return np.max(np.abs(self.P / self.Q - self.alpha))
    
    def info(self) -> dict:
        """Return a dictionary with lattice information."""
        return {
            "type": "ExplicitRank1",
            "dimension": self.d,
            "num_points": self.Q,
            "m_parameter": self.m,
            "alpha": self.alpha.tolist(),
            "numerator_P": self.P.tolist(),
            "lambda1_dual": self.lambda1_dual,
            "separation_radius": self.separation_radius(),
            "approximation_error": self.approximation_error(),
            "Q_sequence": self.Q_list.tolist(),
        }
    
    def __repr__(self) -> str:
        return (f"ExplicitRank1Lattice(d={self.d}, m={self.m}, Q={self.Q}, "
                f"λ₁^⊥={self.lambda1_dual:.4f})")


def find_explicit_lattices_for_range(
    d: int,
    m_min: int,
    m_max: int,
    alpha: Optional[np.ndarray] = None,
    verbose: bool = False
) -> List[ExplicitRank1Lattice]:
    """
    Generate explicit rank-1 lattices for a range of m values.
    
    Parameters
    ----------
    d : int
        Dimension.
    m_min : int
        Minimum m parameter.
    m_max : int
        Maximum m parameter.
    alpha : np.ndarray, optional
        Kronecker generating vector.
    verbose : bool, optional
        Print progress information.
    
    Returns
    -------
    List[ExplicitRank1Lattice]
        List of explicit rank-1 lattices.
    """
    lattices = []
    for m in range(m_min, m_max + 1):
        if verbose:
            print(f"Computing explicit lattice m = {m}")
        lattices.append(ExplicitRank1Lattice(d=d, m=m, alpha=alpha, verbose=False))
    
    return lattices


def get_Q_for_m_range(
    d: int,
    m_min: int,
    m_max: int,
    alpha: Optional[np.ndarray] = None
) -> List[Tuple[int, int]]:
    """
    Get the (m, Q) pairs for a range of m values without full construction.
    
    Useful for planning experiments.
    
    Parameters
    ----------
    d : int
        Dimension.
    m_min : int
        Minimum m parameter.
    m_max : int
        Maximum m parameter.
    alpha : np.ndarray, optional
        Kronecker generating vector.
    
    Returns
    -------
    List[Tuple[int, int]]
        List of (m, Q) pairs.
    """
    if alpha is None:
        alpha = np.array([2.0 ** (j / (d + 1)) for j in range(1, d + 1)])
    
    Q_list = [1]
    results = []
    
    for m in range(1, m_max + 1):
        q_prev = Q_list[-1]
        maxnorm_prev = np.max(np.abs(q_prev * alpha - np.round(q_prev * alpha)))
        
        q_search = q_prev + 1
        while True:
            maxnorm_search = np.max(np.abs(q_search * alpha - np.round(q_search * alpha)))
            if maxnorm_search < maxnorm_prev:
                Q_list.append(q_search)
                break
            q_search += 1
        
        if m >= m_min:
            results.append((m, Q_list[-1]))
    
    return results
