"""
Near-optimality of the LLL-based Korobov search (Algorithm 3.2).

Theorem 3.4 proves the *existence* of a generator a with lambda_1(dual) =
Omega(N^{1/d}); Algorithm 3.2 instead selects a by maximizing an
LLL-computed surrogate of lambda_1(Lambda) * lambda_1(Lambda^perp). This
script closes the "exists vs. LLL finds" gap empirically: for low
dimensions we compute the EXACT shortest vectors of the primal and dual
lattices (by enumerating the defining congruence) for every candidate a,
and compare:

  * per-a: the LLL surrogate vs. the exact length (LLL is an upper bound on
    the true length; in low dimension the two essentially coincide);
  * the generator a* selected by the LLL product criterion vs. the
    generator selected by the exact product criterion -- reported as the
    "efficiency" exact_score(a*_LLL) / max_a exact_score(a) <= 1.

Exact shortest vectors (Korobov, generator a, modulus N):
  dual    Lambda^perp = { h in Z^d : sum_j h_j a^j ≡ 0 (mod N) };
          given (h_1..h_{d-1}), the optimal h_0 is the centered residue of
          -sum_{j>=1} h_j a^j mod N, so lambda_1 is a bounded enumeration.
  primal  Lambda = (z/N) Z + Z^d ; a vector of N*Lambda is
          (c_0, a c_0 + N c_1, ..., a^{d-1} c_0 + N c_{d-1}); for fixed c_0
          each coordinate is minimized by the centered residue of a^j c_0,
          so lambda_1(Lambda) is again a bounded 1-D enumeration over c_0.
"""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice_qmc import KorobovLattice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "Figures")

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.labelsize": 14,
    "legend.fontsize": 10, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 8,
})


def centered_residue(x, N):
    r = np.mod(x, N)
    return np.where(r > N // 2, r - N, r)


def exact_dual_lambda1(a, N, d):
    """Exact lambda_1(Lambda^perp) for the Korobov lattice."""
    powers = np.array([pow(a, j, N) for j in range(d)], dtype=np.int64)  # a^0..a^{d-1}
    B = int(np.ceil(np.sqrt(d) * N ** (1.0 / d))) + 1
    rng = np.arange(-B, B + 1)
    best = float(N)  # vector (N,0,...,0) is in the dual
    # enumerate (h_1,...,h_{d-1})
    grids = np.meshgrid(*([rng] * (d - 1)), indexing="ij")
    H = np.stack([g.ravel() for g in grids], axis=1) if d > 1 else np.zeros((1, 0), int)
    s = (H * powers[1:]).sum(axis=1) if d > 1 else np.zeros(len(H), int)
    h0 = centered_residue(-s, N)
    norm2 = h0 ** 2 + (H ** 2).sum(axis=1)
    # exclude the all-zero vector (h0=0 and H=0)
    zero_mask = (h0 == 0) & (np.all(H == 0, axis=1) if d > 1 else True)
    norm2 = np.where(zero_mask, np.iinfo(np.int64).max, norm2)
    m = np.sqrt(norm2.min())
    return float(min(best, m))


def exact_primal_lambda1(a, N, d):
    """Exact lambda_1(Lambda) for the Korobov lattice."""
    powers = np.array([pow(a, j, N) for j in range(d)], dtype=np.int64)
    C = min(N - 1, int(np.ceil(np.sqrt(d) * N ** ((d - 1.0) / d))) + 1)
    c0 = np.arange(1, C + 1)
    # residues of a^j * c0 for j = 1..d-1
    total = c0.astype(np.float64) ** 2
    for j in range(1, d):
        total = total + centered_residue(powers[j] * c0, N).astype(np.float64) ** 2
    best2 = float(N * N)  # c0 = 0 -> vector (N,0,..) -> N*Lambda length N
    best2 = min(best2, float(total.min()))
    return np.sqrt(best2) / N


PRIMES = [13, 31, 61, 127, 251, 509, 1021]
DIMS = [2, 3, 5]


def study():
    rows = []
    per_a_scatter = None
    for d in DIMS:
        for N in PRIMES:
            if d == 5 and N > 509:
                continue
            ex_d, ex_p, ll_d, ll_p = [], [], [], []
            for a in range(1, N):
                ex_d.append(exact_dual_lambda1(a, N, d))
                ex_p.append(exact_primal_lambda1(a, N, d))
                lat = KorobovLattice(d=d, N=N, generator=a)
                ll_d.append(lat.lambda1_dual)
                ll_p.append(lat.lambda1_primal)
            ex_d = np.array(ex_d); ex_p = np.array(ex_p)
            ll_d = np.array(ll_d); ll_p = np.array(ll_p)

            exact_score = ex_p * ex_d            # exact product criterion
            lll_score = ll_p * ll_d              # what Algorithm 3.2 maximizes
            a_lll = int(np.argmax(lll_score)) + 1
            a_ex = int(np.argmax(exact_score)) + 1
            eff = exact_score[a_lll - 1] / exact_score[a_ex - 1]
            # how often the LLL dual length equals the exact dual length
            agree = float(np.mean(np.isclose(ll_d, ex_d, rtol=1e-6, atol=1e-9)))
            rows.append(dict(d=d, N=N, a_lll=a_lll, a_exact=a_ex, efficiency=eff,
                             dual_match_frac=agree,
                             lll_dual_overshoot=float(np.max(ll_d / ex_d))))
            print(f"d={d} N={N:5d}: a_LLL={a_lll:5d} a_exact={a_ex:5d} "
                  f"efficiency={eff:.4f} dual-match={agree:.2f} "
                  f"max(LLL/exact dual)={np.max(ll_d/ex_d):.3f}")
            if d == 3 and N == 251:
                per_a_scatter = (ex_d.copy(), ll_d.copy(), d, N)
    return rows, per_a_scatter


def make_figure(rows, scatter):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: per-a scatter of exact vs LLL dual length (representative case)
    ax = axes[0]
    ex_d, ll_d, d, N = scatter
    ax.scatter(ex_d, ll_d, s=14, alpha=0.5, color="#1f77b4")
    lim = [0, max(ex_d.max(), ll_d.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1.2, label="$y=x$")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"exact $\lambda_1(\Lambda^\perp)$")
    ax.set_ylabel(r"LLL estimate of $\lambda_1(\Lambda^\perp)$")
    ax.set_title(f"Per-$a$ agreement ($d={d}$, $N={N}$)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    # Right: efficiency of the LLL-selected a* vs N, per dimension
    ax = axes[1]
    colors = {2: "#1f77b4", 3: "#d62728", 5: "#2ca02c"}
    for d in DIMS:
        Ns = [r["N"] for r in rows if r["d"] == d]
        eff = [r["efficiency"] for r in rows if r["d"] == d]
        ax.plot(Ns, eff, marker="o", color=colors[d], label=f"$d={d}$")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0.8, 1.02)
    ax.set_xlabel("Number of points $N$")
    ax.set_ylabel(r"efficiency  $S_{\rm exact}(a^*_{\rm LLL})/\max_a S_{\rm exact}(a)$")
    ax.set_title("Near-optimality of the LLL-selected $a^*$")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(FIGDIR, "LLL_optimality.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


def main():
    rows, scatter = study()
    make_figure(rows, scatter)
    with open(os.path.join(ROOT, "revision", "lll_optimality_results.pkl"), "wb") as f:
        pickle.dump(rows, f)


if __name__ == "__main__":
    main()
