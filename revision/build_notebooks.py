"""
Build/patch the experiment notebooks for the revised manuscript.

  * Creates two NEW notebooks:
        experiments_04_01_04_condition_number.ipynb   (Section 4.1.4)
        experiments_04_01_05_lll_optimality.ipynb      (Section 4.1.5)
  * Patches the separation-radius notebook to add a maximin LHS baseline.

The GPR notebooks are patched by build_notebooks_gpr.py.

Notebooks are created with cleared outputs; execute them with the science
env (jupyter nbconvert --execute) to embed figures.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def new_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def write_nb(path, nb):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


def clear_outputs(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


# ---------------------------------------------------------------------------
# Shared header cell
# ---------------------------------------------------------------------------
HEADER = """import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from lattice_qmc import KorobovLattice, ExplicitRank1Lattice
from lattice_qmc.utils import compute_separation_radius_fast
from lattice_qmc.baselines import maximin_lhs, matern52_gram

try:
    import qmcpy as qp
    QMCPY_AVAILABLE = True
except ImportError:
    QMCPY_AVAILABLE = False

plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif', 'axes.labelsize': 14,
    'legend.fontsize': 10, 'figure.figsize': (7, 5), 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'lines.linewidth': 2, 'lines.markersize': 8,
})

# Load optimal Korobov parameters
korobov_params = np.load('data/korobov_optimal_parameters.npy', allow_pickle=True).item()
korobov_primes = korobov_params['primes']
optimal_generators = korobov_params['optimal_generators']

STYLES = {
    'korobov':  {'color': '#1f77b4', 'marker': 'o', 'label': 'Korobov (ours)'},
    'explicit': {'color': '#d62728', 'marker': 's', 'label': 'Explicit (ours)'},
    'lhs':      {'color': '#ff7f0e', 'marker': 'P', 'label': 'Maximin LHS'},
    'random':   {'color': '#7f7f7f', 'marker': 'x', 'label': 'Random'},
    'halton':   {'color': '#2ca02c', 'marker': '^', 'label': 'Halton'},
    'sobol':    {'color': '#9467bd', 'marker': 'D', 'label': "Sobol'"},
}
print('Setup complete.')
"""

GEN = """def design_points(method, n, d):
    if method == 'halton':
        return qp.Halton(d, seed=42).gen_samples(n)
    if method == 'sobol':
        return qp.Sobol(d, seed=42).gen_samples(n)
    if method == 'lhs':
        return maximin_lhs(n, d, seed=2024)
    if method == 'random':
        return np.random.default_rng(42).random((n, d))
    raise ValueError(method)
print('Generators loaded.')
"""


# ===========================================================================
# 1) Condition-number notebook  (Section 4.1.4)
# ===========================================================================
def build_condition():
    cells = [
        md("# Section 4.1.4 — Conditioning of the kernel matrix\n\n"
           "For each design we form the Matérn-5/2 Gram matrix at a fixed "
           "length scale and report its spectral condition number "
           "$\\kappa(K)$ versus $N$, for $d=2,3,5,7$. The condition number is "
           "an essentially monotone decreasing function of the separation "
           "radius, validating the numerical-stability rationale for "
           "quasi-uniform designs."),
        code(HEADER),
        code(GEN),
        code("""LENGTH_SCALE = 0.5
POWERS = [16, 32, 64, 128, 256, 512]
N_CAP = 1100
EXPL_M = {2: range(3, 12), 3: range(3, 13), 5: range(3, 12), 7: range(3, 9)}

def cond(points):
    K = matern52_gram(points, length_scale=LENGTH_SCALE)
    s = np.linalg.svd(K, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))
"""),
        code("""def run_dim(d):
    res = {m: {'N': [], 'cond': [], 'q': []} for m in STYLES}
    for idx, N in enumerate(korobov_primes[2:9]):
        if N > N_CAP:
            continue
        P = KorobovLattice(d=d, N=int(N), generator=int(optimal_generators[d][2:9][idx])).points
        res['korobov']['N'].append(int(N)); res['korobov']['cond'].append(cond(P))
        res['korobov']['q'].append(compute_separation_radius_fast(P, toroidal=True))
    alpha = np.array([2.0 ** (j / (d + 1)) for j in range(1, d + 1)])
    seen = set()
    for m in EXPL_M[d]:
        lat = ExplicitRank1Lattice(d=d, m=m, alpha=alpha)
        if lat.Q in seen or lat.Q < 3 or lat.Q > N_CAP:
            continue
        seen.add(lat.Q)
        res['explicit']['N'].append(int(lat.Q)); res['explicit']['cond'].append(cond(lat.points))
        res['explicit']['q'].append(compute_separation_radius_fast(lat.points, toroidal=True))
    for method in ['random', 'halton', 'sobol', 'lhs']:
        for n in POWERS:
            P = design_points(method, n, d)
            res[method]['N'].append(n); res[method]['cond'].append(cond(P))
            res[method]['q'].append(compute_separation_radius_fast(P, toroidal=True))
    return res
"""),
        code("""def plot_dim(res, d):
    plt.figure(figsize=(7, 5))
    for m, st in STYLES.items():
        if res[m]['N']:
            order = np.argsort(res[m]['N'])
            N = np.array(res[m]['N'])[order]; cv = np.array(res[m]['cond'])[order]
            plt.loglog(N, cv, marker=st['marker'], color=st['color'], label=st['label'])
    plt.xlabel('Number of points $N$'); plt.ylabel(r'Condition number $\\kappa(K)$')
    plt.legend(loc='best', fontsize=9); plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'Condition_number_d={d}.png', dpi=300)
    plt.show()

for d in [2, 3, 5, 7]:
    res = run_dim(d)
    plot_dim(res, d)
    i = int(np.argmin([abs(n - 256) for n in res['korobov']['N']]))
    print(f"d={d}: at N~256, korobov q={res['korobov']['q'][i]:.4f} kappa={res['korobov']['cond'][i]:.2e}")
"""),
    ]
    write_nb(os.path.join(ROOT, "experiments_04_01_04_condition_number.ipynb"), new_notebook(cells))


# ===========================================================================
# 2) LLL near-optimality notebook  (Section 4.1.5)
# ===========================================================================
def build_lll():
    cells = [
        md("# Section 4.1.5 — Near-optimality of the LLL-based search\n\n"
           "Theorem 3.4 proves the *existence* of a good Korobov parameter; "
           "Algorithm 3.2 selects one with an LLL surrogate. Here we compute the "
           "**exact** shortest primal/dual vectors (by enumerating the defining "
           "congruence) for every candidate $a$ in low dimensions, and confirm "
           "that the LLL-selected $a^*$ is essentially optimal."),
        code(HEADER),
        code("""from itertools import product as iproduct

def centered_residue(x, N):
    r = np.mod(x, N)
    return np.where(r > N // 2, r - N, r)

def exact_dual_lambda1(a, N, d):
    powers = np.array([pow(a, j, N) for j in range(d)], dtype=np.int64)
    B = int(np.ceil(np.sqrt(d) * N ** (1.0 / d))) + 1
    rng = np.arange(-B, B + 1)
    best = float(N)
    grids = np.meshgrid(*([rng] * (d - 1)), indexing='ij')
    H = np.stack([g.ravel() for g in grids], axis=1)
    s = (H * powers[1:]).sum(axis=1)
    h0 = centered_residue(-s, N)
    norm2 = h0 ** 2 + (H ** 2).sum(axis=1)
    zero = (h0 == 0) & np.all(H == 0, axis=1)
    norm2 = np.where(zero, np.iinfo(np.int64).max, norm2)
    return float(min(best, np.sqrt(norm2.min())))

def exact_primal_lambda1(a, N, d):
    powers = np.array([pow(a, j, N) for j in range(d)], dtype=np.int64)
    C = min(N - 1, int(np.ceil(np.sqrt(d) * N ** ((d - 1.0) / d))) + 1)
    c0 = np.arange(1, C + 1)
    total = c0.astype(np.float64) ** 2
    for j in range(1, d):
        total = total + centered_residue(powers[j] * c0, N).astype(np.float64) ** 2
    return np.sqrt(min(float(N * N), float(total.min()))) / N
"""),
        code("""PRIMES = [13, 31, 61, 127, 251, 509, 1021]
DIMS = [2, 3, 5]
rows, scatter = [], None
for d in DIMS:
    for N in PRIMES:
        if d == 5 and N > 509:
            continue
        ex_d, ex_p, ll_d, ll_p = [], [], [], []
        for a in range(1, N):
            ex_d.append(exact_dual_lambda1(a, N, d)); ex_p.append(exact_primal_lambda1(a, N, d))
            lat = KorobovLattice(d=d, N=N, generator=a)
            ll_d.append(lat.lambda1_dual); ll_p.append(lat.lambda1_primal)
        ex_d, ex_p = np.array(ex_d), np.array(ex_p)
        ll_d, ll_p = np.array(ll_d), np.array(ll_p)
        exact_score, lll_score = ex_p * ex_d, ll_p * ll_d
        a_lll = int(np.argmax(lll_score)) + 1; a_ex = int(np.argmax(exact_score)) + 1
        eff = exact_score[a_lll - 1] / exact_score[a_ex - 1]
        rows.append(dict(d=d, N=N, efficiency=eff))
        print(f"d={d} N={N:5d}: a_LLL={a_lll:5d} a_exact={a_ex:5d} efficiency={eff:.4f}")
        if d == 3 and N == 251:
            scatter = (ex_d.copy(), ll_d.copy(), d, N)
"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ex_d, ll_d, d0, N0 = scatter
ax = axes[0]
ax.scatter(ex_d, ll_d, s=14, alpha=0.5, color='#1f77b4')
lim = [0, max(ex_d.max(), ll_d.max()) * 1.05]
ax.plot(lim, lim, 'k--', lw=1.2, label='$y=x$')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'exact $\\lambda_1(\\Lambda^\\perp)$')
ax.set_ylabel(r'LLL estimate of $\\lambda_1(\\Lambda^\\perp)$')
ax.set_title(f'Per-$a$ agreement ($d={d0}$, $N={N0}$)')
ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
ax = axes[1]
colors = {2: '#1f77b4', 3: '#d62728', 5: '#2ca02c'}
for d in DIMS:
    Ns = [r['N'] for r in rows if r['d'] == d]
    eff = [r['efficiency'] for r in rows if r['d'] == d]
    ax.plot(Ns, eff, marker='o', color=colors[d], label=f'$d={d}$')
ax.axhline(1.0, color='k', ls=':', lw=1); ax.set_xscale('log'); ax.set_ylim(0.8, 1.02)
ax.set_xlabel('Number of points $N$')
ax.set_ylabel(r'efficiency $S_{\\rm exact}(a^*_{\\rm LLL})/\\max_a S_{\\rm exact}(a)$')
ax.set_title('Near-optimality of the LLL-selected $a^*$')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, which='both')
fig.tight_layout(); fig.savefig('LLL_optimality.png', dpi=300); plt.show()
"""),
    ]
    write_nb(os.path.join(ROOT, "experiments_04_01_05_lll_optimality.ipynb"), new_notebook(cells))


# ===========================================================================
# 3) Patch the separation-radius notebook: add maximin LHS
# ===========================================================================
def patch_separation():
    path = os.path.join(ROOT, "experiments_04_01_03_separation_radius.ipynb")
    nb = json.load(open(path))
    changed = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "'lhs'" in s or "maximin_lhs" in s:
            continue
        # (a) generator cell: add maximin LHS generator
        if "def generate_sobol_points" in s and "print(\"Helper functions loaded.\")" in s:
            s = s.replace(
                "print(\"Helper functions loaded.\")",
                "def generate_maximin_lhs(n: int, d: int) -> np.ndarray:\n"
                "    \"\"\"Generate n maximin-optimized LHS points in [0, 1)^d.\"\"\"\n"
                "    from lattice_qmc.baselines import maximin_lhs\n"
                "    return maximin_lhs(n, d, seed=2024)\n\n\n"
                "print(\"Helper functions loaded.\")")
            changed += 1
        # (b) runner: add 'lhs' to results dict and an LHS loop
        if "def run_separation_radius_experiment" in s:
            s = s.replace(
                "        'sobol': {'N': [], 'sep_radius': []},\n    }",
                "        'sobol': {'N': [], 'sep_radius': []},\n"
                "        'lhs': {'N': [], 'sep_radius': []},\n    }")
            s = s.replace(
                "    \n    return results",
                "    \n    # Maximin LHS\n"
                "    if verbose:\n        print(\"\\nMaximin LHS:\")\n"
                "    for n in powers:\n"
                "        points = generate_maximin_lhs(n, d)\n"
                "        sep_rad = compute_separation_radius_fast(points, toroidal=True)\n"
                "        results['lhs']['N'].append(n)\n"
                "        results['lhs']['sep_radius'].append(sep_rad)\n"
                "        if verbose:\n            print(f\"  N = {n}: q = {sep_rad:.6f}\")\n"
                "    \n    return results")
            changed += 1
        # (c) plot: add 'lhs' style
        if "def plot_separation_radius_comparison" in s:
            s = s.replace(
                "        'explicit': {'color': '#d62728', 'marker': 's', 'label': 'Explicit (ours)'},",
                "        'explicit': {'color': '#d62728', 'marker': 's', 'label': 'Explicit (ours)'},\n"
                "        'lhs': {'color': '#ff7f0e', 'marker': 'P', 'label': 'Maximin LHS'},")
            changed += 1
        c["source"] = s.splitlines(keepends=True)
    clear_outputs(nb)
    write_nb(path, nb)
    print(f"  separation: applied {changed} edits (expected 3)")


if __name__ == "__main__":
    build_condition()
    build_lll()
    patch_separation()
