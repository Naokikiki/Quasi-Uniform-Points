"""
Separation-radius comparison, with a maximin LHS baseline added.

Regenerates the separation-radius vs N figures for d = 2, 3, 5, 7, now
including a maximin Latin hypercube design alongside the explicit rank-1
lattice, Korobov lattice, Halton, Sobol' and random points. The reference
line N^{-1/d} marks the optimal (quasi-uniform) decay rate.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lattice_qmc import KorobovLattice, ExplicitRank1Lattice
from lattice_qmc.utils import compute_separation_radius_fast
from lattice_qmc.baselines import maximin_lhs
import qmcpy as qp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "Figures")

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.labelsize": 14,
    "legend.fontsize": 9, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 8,
})

KP = np.load(os.path.join(ROOT, "data", "korobov_optimal_parameters.npy"),
             allow_pickle=True).item()
KOR_PRIMES = KP["primes"]
KOR_GEN = KP["optimal_generators"]

POWERS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
EXPL_M = {2: 7, 3: 12, 5: 10, 7: 5}

STYLES = {
    "korobov":  {"color": "#1f77b4", "marker": "o", "label": "Korobov (ours)"},
    "explicit": {"color": "#d62728", "marker": "s", "label": "Explicit (ours)"},
    "lhs":      {"color": "#ff7f0e", "marker": "P", "label": "Maximin LHS"},
    "random":   {"color": "#7f7f7f", "marker": "x", "label": "Random"},
    "halton":   {"color": "#2ca02c", "marker": "^", "label": "Halton"},
    "sobol":    {"color": "#9467bd", "marker": "D", "label": "Sobol'"},
}


def sep(P):
    return compute_separation_radius_fast(P, toroidal=True)


def run_dim(d):
    res = {m: {"N": [], "q": []} for m in STYLES}
    for idx, N in enumerate(KOR_PRIMES[:9]):
        a = KOR_GEN[d][idx]
        res["korobov"]["N"].append(int(N))
        res["korobov"]["q"].append(sep(KorobovLattice(d=d, N=int(N), generator=int(a)).points))
    alpha = np.array([2.0 ** (j / (d + 1)) for j in range(1, d + 1)])
    seen = set()
    for m in range(EXPL_M[d] + 1):
        lat = ExplicitRank1Lattice(d=d, m=m, alpha=alpha)
        if lat.Q < 3 or lat.Q in seen:
            continue
        seen.add(lat.Q)
        res["explicit"]["N"].append(int(lat.Q))
        res["explicit"]["q"].append(sep(lat.points))
    for n in POWERS:
        res["random"]["N"].append(n)
        res["random"]["q"].append(sep(np.random.default_rng(42).random((n, d))))
        res["halton"]["N"].append(n)
        res["halton"]["q"].append(sep(qp.Halton(d, seed=42).gen_samples(n)))
        res["sobol"]["N"].append(n)
        res["sobol"]["q"].append(sep(qp.Sobol(d, seed=42).gen_samples(n)))
        res["lhs"]["N"].append(n)
        res["lhs"]["q"].append(sep(maximin_lhs(n, d, seed=2024)))
    return res


def plot_dim(res, d):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, st in STYLES.items():
        if res[m]["N"]:
            ax.loglog(res[m]["N"], res[m]["q"], marker=st["marker"],
                      color=st["color"], label=st["label"])
    N_ref = np.array([res["korobov"]["N"][0], res["korobov"]["N"][-1]])
    scale = res["korobov"]["q"][0] * (res["korobov"]["N"][0] ** (1.0 / d))
    ax.loglog(N_ref, scale * N_ref ** (-1.0 / d), "k--", lw=1.5, alpha=0.7,
              label=f"$N^{{-1/{d}}}$ (optimal)")
    ax.set_xlabel("Number of points $N$")
    ax.set_ylabel("Separation radius $q(P)$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"Separation_radius_d={d}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


def decay_rate(N, q):
    return np.polyfit(np.log(N), np.log(q), 1)[0]


def main():
    import pickle
    allres = {}
    for d in [2, 3, 5, 7]:
        res = run_dim(d)
        plot_dim(res, d)
        allres[d] = res
        print(f"--- d={d} decay rates (target {-1/d:.3f}) ---")
        for m in STYLES:
            if res[m]["N"]:
                print(f"  {m:9s} rate={decay_rate(np.array(res[m]['N']), np.array(res[m]['q'])):+.3f}")
    with open(os.path.join(ROOT, "revision", "separation_results.pkl"), "wb") as f:
        pickle.dump(allres, f)


if __name__ == "__main__":
    main()
