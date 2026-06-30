"""
Condition-number study (numerical-stability validation).

The paper argues that a large separation radius prevents severe
ill-conditioning of the kernel (Gram) matrix used in kernel interpolation
and Gaussian process regression. This script makes that argument explicit:
for each design we form the Matern(nu=5/2) Gram matrix at a fixed length
scale and plot its 2-norm condition number against N, for d = 2, 3, 5, 7.

Designs compared: explicit rank-1, Korobov, Halton, Sobol', random, and
maximin LHS. A fixed length scale is used so that the comparison isolates
the geometric effect of the point distribution; the qualitative ordering is
insensitive to the exact value.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lattice_qmc import KorobovLattice, ExplicitRank1Lattice
from lattice_qmc.baselines import maximin_lhs, matern52_gram
from lattice_qmc.utils import compute_separation_radius_fast
import qmcpy as qp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "Figures")

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.labelsize": 14,
    "legend.fontsize": 10, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 8,
})

KP = np.load(os.path.join(ROOT, "data", "korobov_optimal_parameters.npy"),
             allow_pickle=True).item()
KOR_PRIMES = KP["primes"]
KOR_GEN = KP["optimal_generators"]

LENGTH_SCALE = 0.5          # fixed Matern length scale
POWERS = [16, 32, 64, 128, 256, 512]
N_CAP = 1100                # keep all designs in a comparable N range
EXPL_M = {2: range(3, 12), 3: range(3, 13), 5: range(3, 12), 7: range(3, 9)}

STYLES = {
    "korobov":  {"color": "#1f77b4", "marker": "o", "label": "Korobov (ours)"},
    "explicit": {"color": "#d62728", "marker": "s", "label": "Explicit (ours)"},
    "lhs":      {"color": "#ff7f0e", "marker": "P", "label": "Maximin LHS"},
    "random":   {"color": "#7f7f7f", "marker": "x", "label": "Random"},
    "halton":   {"color": "#2ca02c", "marker": "^", "label": "Halton"},
    "sobol":    {"color": "#9467bd", "marker": "D", "label": "Sobol'"},
}


def cond(points):
    K = matern52_gram(points, length_scale=LENGTH_SCALE)
    s = np.linalg.svd(K, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))


def design(method, n, d):
    if method == "halton":
        return qp.Halton(d, seed=42).gen_samples(n)
    if method == "sobol":
        return qp.Sobol(d, seed=42).gen_samples(n)
    if method == "lhs":
        return maximin_lhs(n, d, seed=2024)
    if method == "random":
        return np.random.default_rng(42).random((n, d))


def run_dim(d):
    res = {m: {"N": [], "cond": [], "q": []} for m in STYLES}
    primes = list(KOR_PRIMES[2:9])
    gens = list(KOR_GEN[d][2:9])
    for N, a in zip(primes, gens):
        if N > N_CAP:
            continue
        P = KorobovLattice(d=d, N=int(N), generator=int(a)).points
        res["korobov"]["N"].append(int(N))
        res["korobov"]["cond"].append(cond(P))
        res["korobov"]["q"].append(compute_separation_radius_fast(P, toroidal=True))
    alpha = np.array([2.0 ** (j / (d + 1)) for j in range(1, d + 1)])
    seen = set()
    for m in EXPL_M[d]:
        lat = ExplicitRank1Lattice(d=d, m=m, alpha=alpha)
        if lat.Q in seen or lat.Q < 3 or lat.Q > N_CAP:
            continue
        seen.add(lat.Q)
        res["explicit"]["N"].append(int(lat.Q))
        res["explicit"]["cond"].append(cond(lat.points))
        res["explicit"]["q"].append(compute_separation_radius_fast(lat.points, toroidal=True))
    for method in ["random", "halton", "sobol", "lhs"]:
        for n in POWERS:
            P = design(method, n, d)
            res[method]["N"].append(n)
            res[method]["cond"].append(cond(P))
            res[method]["q"].append(compute_separation_radius_fast(P, toroidal=True))
    return res


def plot_dim(res, d):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, st in STYLES.items():
        if res[m]["N"]:
            order = np.argsort(res[m]["N"])
            N = np.array(res[m]["N"])[order]
            cv = np.array(res[m]["cond"])[order]
            ax.loglog(N, cv, marker=st["marker"], color=st["color"], label=st["label"])
    ax.set_xlabel("Number of points $N$")
    ax.set_ylabel(r"Condition number $\kappa(K)$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"Condition_number_d={d}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


def main():
    import pickle
    allres = {}
    for d in [2, 3, 5, 7]:
        res = run_dim(d)
        plot_dim(res, d)
        allres[d] = res
        # quick summary near N~256 to show separation/conditioning link
        print(f"--- d={d}: condition number & separation radius at N nearest 256 ---")
        for m in STYLES:
            if res[m]["N"]:
                i = int(np.argmin([abs(n - 256) for n in res[m]["N"]]))
                print(f"  {m:9s} N={res[m]['N'][i]:5d} q={res[m]['q'][i]:.4f} kappa={res[m]['cond'][i]:.3e}")
    with open(os.path.join(ROOT, "revision", "condition_number_results.pkl"), "wb") as f:
        pickle.dump(allres, f)


if __name__ == "__main__":
    main()
