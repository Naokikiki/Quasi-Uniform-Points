"""
Baseline designs and test functions for the revision experiments.
===================================================================

This module collects helpers that are shared across the revision
experiments (Section 4 of the paper):

  * ``maximin_lhs``        -- a maximin-optimized Latin hypercube design,
                              added as a space-filling baseline in response
                              to the reviewer request for a comparison
                              against maximin distance-based designs.
  * ``genz_continuous``    -- Genz "continuous" test function for general d.
  * ``genz_gaussian``      -- Genz "Gaussian" test function for general d.
  * ``pairwise_trig``      -- pairwise trigonometric test function.
  * ``matern52_gram``      -- Matern(nu=5/2) Gram matrix and its condition
                              number, used for the numerical-stability study.

The maximin LHS is built by a best-of-many random search over Latin
hypercube designs followed by a Morris--Mitchell style column-swap local
search, both driven by the (ordinary Euclidean) minimum pairwise distance.
Unlike the rank-1 lattices, a Latin hypercube design has no periodic
structure, so its minimum pairwise distance -- and hence its separation
radius -- is measured with the ordinary (non-toroidal) Euclidean metric on
[0,1)^d.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Maximin Latin hypercube design
# ---------------------------------------------------------------------------
def _dist_matrix(points: np.ndarray, toroidal: bool = True) -> np.ndarray:
    """Full pairwise distance matrix (toroidal on [0,1)^d), diagonal = +inf."""
    diff = points[:, None, :] - points[None, :, :]
    if toroidal:
        diff = np.abs(diff)
        diff = np.minimum(diff, 1.0 - diff)
    D = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    return D


def _row_dist(point: np.ndarray, points: np.ndarray, toroidal: bool = True) -> np.ndarray:
    diff = point[None, :] - points
    if toroidal:
        diff = np.abs(diff)
        diff = np.minimum(diff, 1.0 - diff)
    return np.sqrt((diff ** 2).sum(axis=1))


def _random_lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """A single jittered Latin hypercube design in [0,1)^d."""
    pts = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        pts[:, j] = (perm + rng.random(n)) / n
    return pts


def maximin_lhs(
    n: int,
    d: int,
    seed: Optional[int] = 42,
    n_candidates: Optional[int] = None,
    n_swaps: Optional[int] = None,
    toroidal: bool = False,
) -> np.ndarray:
    """
    Generate a maximin-optimized Latin hypercube design with ``n`` points.

    The design maximizes the minimum pairwise distance (the maximin
    criterion). It is obtained by selecting the best of ``n_candidates``
    random Latin hypercube designs, then improving it with a directed
    column-swap local search that repeatedly attacks the current closest
    pair of points (a Morris--Mitchell style search). Each accepted swap
    keeps the design a valid Latin hypercube and strictly increases the
    minimum pairwise distance.

    Parameters
    ----------
    n, d : int
        Number of points and dimension.
    seed : int, optional
        Seed for reproducibility.
    n_candidates, n_swaps : int, optional
        Search-effort parameters; sensible ``n``-dependent defaults are used.
    toroidal : bool
        Whether the maximin criterion uses the toroidal distance. Defaults to
        ``False``: a Latin hypercube design is not periodic, so the ordinary
        Euclidean distance on ``[0,1)^d`` is used.

    Returns
    -------
    np.ndarray
        Maximin LHS point set of shape ``(n, d)`` in ``[0, 1)^d``.
    """
    rng = np.random.default_rng(seed)
    if n < 2:
        return _random_lhs(max(n, 1), d, rng)

    if n_candidates is None:
        n_candidates = int(min(200, max(20, 50_000 // n)))
    if n_swaps is None:
        n_swaps = int(min(4000, max(300, 3_000_000 // n)))

    # Best of several random Latin hypercube designs.
    best_pts, best_D, best_min = None, None, -np.inf
    for _ in range(n_candidates):
        pts = _random_lhs(n, d, rng)
        D = _dist_matrix(pts, toroidal)
        m = D.min()
        if m > best_min:
            best_pts, best_D, best_min = pts, D, m
    pts, D, gmin = best_pts, best_D, best_min

    # Directed local search: break the current closest pair by swapping one
    # of its coordinates with another row in the same column.
    for _ in range(n_swaps):
        flat = int(np.argmin(D))
        i, j = divmod(flat, n)
        m = i if rng.random() < 0.5 else j           # endpoint of closest pair
        c = int(rng.integers(d))
        k = int(rng.integers(n))
        if k == m:
            continue
        # tentative swap of column c between rows m and k
        pts[[m, k], c] = pts[[k, m], c]
        new_m = _row_dist(pts[m], pts, toroidal); new_m[m] = np.inf
        new_k = _row_dist(pts[k], pts, toroidal); new_k[k] = np.inf
        Dt = D.copy()
        Dt[m, :] = new_m; Dt[:, m] = new_m
        Dt[k, :] = new_k; Dt[:, k] = new_k
        Dt[m, k] = new_m[k]; Dt[k, m] = new_m[k]
        np.fill_diagonal(Dt, np.inf)
        ngmin = Dt.min()
        if ngmin > gmin:
            D, gmin = Dt, ngmin                       # keep the swap
        else:
            pts[[m, k], c] = pts[[k, m], c]           # revert
    return pts


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------
def genz_continuous(X: np.ndarray, c: Optional[np.ndarray] = None,
                    u: float = 0.5) -> np.ndarray:
    """
    Genz "continuous" function: f(x) = exp(-sum_i c_i |x_i - u|).

    If ``c`` is None, a dimension-dependent default
    ``c_i = (i + 0.5)`` rescaled to a fixed total is used.
    """
    X = np.atleast_2d(X)
    d = X.shape[1]
    if c is None:
        c = _default_genz_c(d, total=2.0 * d)
    c = np.asarray(c, dtype=float)
    return np.exp(-(np.abs(X - u) * c).sum(axis=1))


def genz_gaussian(X: np.ndarray, c: Optional[np.ndarray] = None,
                  u: float = 0.5) -> np.ndarray:
    """
    Genz "Gaussian" function: f(x) = exp(-sum_i c_i^2 (x_i - u)^2).

    If ``c`` is None, a constant default chosen so that the function decays
    on the scale of the domain is used.
    """
    X = np.atleast_2d(X)
    d = X.shape[1]
    if c is None:
        c = np.full(d, 3.0)
    c = np.asarray(c, dtype=float)
    return np.exp(-((c ** 2) * (X - u) ** 2).sum(axis=1))


def _default_genz_c(d: int, total: float) -> np.ndarray:
    """Smoothly varying positive weights summing to ``total``."""
    base = np.linspace(1.0, 2.0, d)
    return base * (total / base.sum())


def pairwise_trig(X: np.ndarray) -> np.ndarray:
    """Pairwise trigonometric function f(x) = mean_{i<j} sin(pi (x_i + x_j))."""
    X = np.atleast_2d(X)
    n, d = X.shape
    num_pairs = d * (d - 1) // 2
    out = np.zeros(n)
    for i in range(d):
        for j in range(i + 1, d):
            out += np.sin(np.pi * (X[:, i] + X[:, j]))
    return out / num_pairs


# ---------------------------------------------------------------------------
# Gram matrix / numerical stability
# ---------------------------------------------------------------------------
def matern52_gram(points: np.ndarray, length_scale: float = 0.5,
                  variance: float = 1.0) -> np.ndarray:
    """
    Matern(nu=5/2) Gram (kernel) matrix for a point set.

    k(x,y) = sigma^2 (1 + sqrt5 r/l + 5 r^2/(3 l^2)) exp(-sqrt5 r/l),
    with r = ||x - y||_2.
    """
    P = np.atleast_2d(points)
    diff = P[:, None, :] - P[None, :, :]
    r = np.sqrt((diff ** 2).sum(axis=2))
    s5 = np.sqrt(5.0)
    a = s5 * r / length_scale
    return variance * (1.0 + a + (a ** 2) / 3.0) * np.exp(-a)


def gram_condition_number(points: np.ndarray, length_scale: float = 0.5,
                          jitter: float = 0.0) -> float:
    """2-norm condition number of the Matern(5/2) Gram matrix."""
    K = matern52_gram(points, length_scale=length_scale)
    if jitter > 0.0:
        K = K + jitter * np.eye(K.shape[0])
    s = np.linalg.svd(K, compute_uv=False)
    return float(s[0] / s[-1])
