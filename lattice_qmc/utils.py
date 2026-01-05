"""
Utility functions for quasi-uniform lattice point sets.
"""

import numpy as np
from math import gcd
from functools import reduce
from typing import List, Tuple, Optional


def generate_primes(n_max: int, n_min: int = 2) -> List[int]:
    """
    Generate all prime numbers in the range [n_min, n_max] using Sieve of Eratosthenes.
    
    Parameters
    ----------
    n_max : int
        Upper bound for prime search.
    n_min : int, optional
        Lower bound for prime search (default: 2).
    
    Returns
    -------
    List[int]
        List of prime numbers in [n_min, n_max].
    
    Examples
    --------
    >>> generate_primes(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
    >>> generate_primes(20, 10)
    [11, 13, 17, 19]
    """
    if n_max < 2:
        return []
    
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n_max + 1, i):
                sieve[j] = False
    
    return [i for i in range(max(2, n_min), n_max + 1) if sieve[i]]


def multi_gcd(numbers: List[int]) -> int:
    """
    Compute the greatest common divisor of a list of integers.
    
    Parameters
    ----------
    numbers : List[int]
        List of integers.
    
    Returns
    -------
    int
        GCD of all numbers in the list.
    """
    return reduce(gcd, numbers)


def compute_separation_radius(points: np.ndarray, toroidal: bool = False) -> float:
    """
    Compute the separation radius (half of the minimum pairwise distance).
    
    The separation radius is defined as:
        q(P) = (1/2) * min_{i != j} ||x_i - x_j||
    
    For point sets on [0,1]^d with periodic boundary conditions (toroidal),
    we use the toroidal distance.
    
    Parameters
    ----------
    points : np.ndarray
        Point set of shape (n, d).
    toroidal : bool, optional
        If True, use toroidal (periodic) distance on [0,1]^d (default: False).
        Set to True for lattice point sets with periodic structure.
    
    Returns
    -------
    float
        Separation radius of the point set.
    
    Notes
    -----
    Time complexity: O(n^2 * d)
    """
    n = points.shape[0]
    if n < 2:
        return np.inf
    
    min_dist_sq = np.inf
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            if toroidal:
                # Toroidal distance: min(|x|, 1-|x|) for each coordinate
                diff = np.abs(diff)
                diff = np.minimum(diff, 1.0 - diff)
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    return 0.5 * np.sqrt(min_dist_sq)


def compute_separation_radius_fast(points: np.ndarray, toroidal: bool = False) -> float:
    """
    Compute separation radius using vectorized operations.
    
    More memory-intensive but faster for moderate n.
    
    Parameters
    ----------
    points : np.ndarray
        Point set of shape (n, d).
    toroidal : bool, optional
        If True, use toroidal distance (default: False).
        Set to True for lattice point sets with periodic structure.
    
    Returns
    -------
    float
        Separation radius.
    """
    n = points.shape[0]
    if n < 2:
        return np.inf
    
    # For large n, fall back to iterative method to save memory
    if n > 5000:
        return compute_separation_radius(points, toroidal)
    
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    if toroidal:
        diff = np.abs(diff)
        diff = np.minimum(diff, 1.0 - diff)
    
    # Compute squared distances
    dist_sq = np.sum(diff**2, axis=2)
    
    # Set diagonal to inf to exclude self-distances
    np.fill_diagonal(dist_sq, np.inf)
    
    min_dist = np.sqrt(np.min(dist_sq))
    return 0.5 * min_dist


def compute_mesh_ratio_upper_bound(
    lambda1_primal: float,
    lambda1_dual: float,
    d: int
) -> float:
    """
    Compute the upper bound of the mesh ratio for a lattice.
    
    From the theorem:
        ρ(Λ) ≤ d√d / (λ₁(Λ) × λ₁(Λ^⊥))
    
    Parameters
    ----------
    lambda1_primal : float
        Length of shortest vector in the primal lattice λ₁(Λ).
    lambda1_dual : float
        Length of shortest vector in the dual lattice λ₁(Λ^⊥).
    d : int
        Dimension.
    
    Returns
    -------
    float
        Upper bound on mesh ratio.
    """
    return d * np.sqrt(d) / (lambda1_primal * lambda1_dual)


def lll_reduce(basis: np.ndarray, delta: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform LLL lattice basis reduction.
    
    The LLL algorithm finds a reduced basis for a lattice that has
    relatively short and nearly orthogonal vectors.
    
    Parameters
    ----------
    basis : np.ndarray
        Input basis matrix of shape (n, d), where rows are basis vectors.
    delta : float, optional
        Reduction parameter, 0.25 < delta < 1 (default: 0.75).
    
    Returns
    -------
    reduced_basis : np.ndarray
        LLL-reduced basis.
    transform : np.ndarray
        Unimodular transformation matrix.
    
    Notes
    -----
    This is a basic implementation. For production use, consider using
    fpylll or similar specialized libraries.
    """
    B = basis.astype(np.float64).copy()
    n = B.shape[0]
    
    # Gram-Schmidt orthogonalization (stored implicitly via mu coefficients)
    def gram_schmidt(B):
        n = B.shape[0]
        B_star = np.zeros_like(B, dtype=np.float64)
        mu = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            B_star[i] = B[i].copy()
            for j in range(i):
                if np.dot(B_star[j], B_star[j]) > 1e-10:
                    mu[i, j] = np.dot(B[i], B_star[j]) / np.dot(B_star[j], B_star[j])
                    B_star[i] -= mu[i, j] * B_star[j]
        
        return B_star, mu
    
    # Transformation matrix (tracks changes to original basis)
    H = np.eye(n, dtype=np.float64)
    
    k = 1
    while k < n:
        B_star, mu = gram_schmidt(B)
        
        # Size reduction
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = round(mu[k, j])
                B[k] -= q * B[j]
                H[k] -= q * H[j]
                B_star, mu = gram_schmidt(B)
        
        # Lovász condition
        B_star_k_norm_sq = np.dot(B_star[k], B_star[k])
        B_star_km1_norm_sq = np.dot(B_star[k-1], B_star[k-1])
        
        if B_star_k_norm_sq >= (delta - mu[k, k-1]**2) * B_star_km1_norm_sq:
            k += 1
        else:
            # Swap b_k and b_{k-1}
            B[[k, k-1]] = B[[k-1, k]]
            H[[k, k-1]] = H[[k-1, k]]
            k = max(k - 1, 1)
    
    return B, H


def shortest_vector_length(basis: np.ndarray, use_lll: bool = True) -> float:
    """
    Estimate the length of the shortest non-zero vector in a lattice.
    
    Uses LLL reduction to find an approximately shortest vector.
    
    Parameters
    ----------
    basis : np.ndarray
        Lattice basis matrix of shape (n, d).
    use_lll : bool, optional
        If True, use LLL reduction first (default: True).
    
    Returns
    -------
    float
        Length of the (approximately) shortest vector.
    """
    if use_lll:
        reduced_basis, _ = lll_reduce(basis)
    else:
        reduced_basis = basis
    
    # The first vector of an LLL-reduced basis is approximately shortest
    lengths = np.linalg.norm(reduced_basis, axis=1)
    return np.min(lengths)
