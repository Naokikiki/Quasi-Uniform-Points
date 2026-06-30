"""
Comprehensive GPR experiment for the revision.

Re-runs the Gaussian process regression study for all dimensions and test
functions, with two changes requested by the reviewers:

  (i)  a maximin Latin hypercube design (LHS) is added as a baseline to
       every comparison;
  (ii) the Genz "continuous" and "Gaussian" test functions are evaluated in
       d = 5 and d = 7 (in addition to the pairwise trigonometric function),
       so that the high-dimensional comparison is not tied to a single
       smooth additive function.

Figures are written into Figures/ with the file names referenced by the
paper. A results pickle is written into revision/ for reproducibility, and
progress is logged to revision/gpr_all.log.

Design choices kept identical to the original notebooks:
  * Matern(nu=5/2) kernel, length-scale and variance fit by marginal
    likelihood (n_restarts_optimizer=3, normalize_y=True, jitter alpha=1e-6);
  * observation noise sigma = 0.05, averaged over 100 noise realizations;
  * Korobov uses the optimal generators from data/korobov_optimal_parameters;
  * explicit rank-1 uses p = 2; L2 error by Gauss-Legendre (d=2) or MC (d>=3).
"""

import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")          # inherited by loky workers on import
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

N_JOBS = int(os.environ.get("GPR_N_JOBS", "8"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from lattice_qmc import KorobovLattice, ExplicitRank1Lattice
from lattice_qmc.baselines import maximin_lhs, genz_continuous, genz_gaussian, pairwise_trig
import qmcpy as qp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "Figures")
REVDIR = os.path.join(ROOT, "revision")
LOG = os.path.join(REVDIR, "gpr_all.log")

NOISE_STD = 0.05
N_TRIALS = int(os.environ.get("GPR_N_TRIALS", "100"))
SMOKE = os.environ.get("GPR_SMOKE", "") == "1"

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.labelsize": 14,
    "legend.fontsize": 10, "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 2, "lines.markersize": 8,
})


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ----------------------------------------------------------------------
# GPR helpers
# ----------------------------------------------------------------------
def fit_gpr(X, y, ls=0.3):
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=ls,
                                          length_scale_bounds=(1e-2, 1e1), nu=2.5)
    g = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                 n_restarts_optimizer=3, normalize_y=True)
    g.fit(X, y)
    return g


def l2_error_gl(gpr, func, d, n_quad=50):
    nodes, w = np.polynomial.legendre.leggauss(n_quad)
    nodes = 0.5 * (nodes + 1); w = 0.5 * w
    X1, X2 = np.meshgrid(nodes, nodes)
    W1, W2 = np.meshgrid(w, w)
    P = np.column_stack([X1.ravel(), X2.ravel()])
    wt = (W1 * W2).ravel()
    err2 = np.sum(wt * (func(P) - gpr.predict(P)) ** 2)
    return np.sqrt(err2)


def l2_error_mc(gpr, func, d, n_mc=10000, seed=12345):
    rng = np.random.default_rng(seed)
    X = rng.random((n_mc, d))
    return np.sqrt(np.mean((func(X) - gpr.predict(X)) ** 2))


def l2_error(gpr, func, d):
    return l2_error_gl(gpr, func, d) if d == 2 else l2_error_mc(gpr, func, d)


# ----------------------------------------------------------------------
# Design point sets
# ----------------------------------------------------------------------
KP = np.load(os.path.join(ROOT, "data", "korobov_optimal_parameters.npy"),
             allow_pickle=True).item()
KOR_PRIMES = KP["primes"]
KOR_GEN = KP["optimal_generators"]

# (korobov index slice, explicit m range, powers) per dimension, matching notebooks
CONFIG = {
    2: (slice(2, 9), range(3, 8),  [16, 32, 64, 128, 256, 512, 1024]),
    3: (slice(2, 9), range(6, 13), [16, 32, 64, 128, 256, 512, 1024]),
    5: (slice(2, 8), range(3, 11), [16, 32, 64, 128, 256, 512]),
    7: (slice(2, 7), range(3, 6),  [16, 32, 64, 128, 256]),
}

_LHS_CACHE = {}


def lhs_points(n, d):
    key = (n, d)
    if key not in _LHS_CACHE:
        _LHS_CACHE[key] = maximin_lhs(n, d, seed=2024)
    return _LHS_CACHE[key]


def design_points(method, n, d, trial):
    if method == "halton":
        return qp.Halton(d, seed=42).gen_samples(n)
    if method == "sobol":
        return qp.Sobol(d, seed=42).gen_samples(n)
    if method == "lhs":
        return lhs_points(n, d)
    if method == "random":
        return np.random.default_rng(42 + trial).random((n, d))
    raise ValueError(method)


# ----------------------------------------------------------------------
# Single trial (module-level so joblib can pickle it)
# ----------------------------------------------------------------------
def _trial_error(X, func, d, ls, trial):
    y0 = func(X)
    yn = y0 + np.random.default_rng(1000 + trial).normal(0, NOISE_STD, y0.shape)
    return l2_error(fit_gpr(X, yn, ls), func, d)


# ----------------------------------------------------------------------
# Experiment runner for one (dimension, function), parallel over trials
# ----------------------------------------------------------------------
def run_one(d, func, fname, ls=0.3):
    kor_idx, m_range, powers = CONFIG[d]
    primes = list(KOR_PRIMES[kor_idx])
    gens = list(KOR_GEN[d][kor_idx])
    res = {m: {"N": [], "mean": [], "std": []}
           for m in ["korobov", "explicit", "random", "halton", "sobol", "lhs"]}

    # Build a flat list of work items: (cfg_key, X, trial)
    items = []
    cfg_order = []  # preserve insertion order of cfg keys

    def add_cfg(method, N, X_or_none):
        cfg_order.append((method, int(N)))
        for t in range(N_TRIALS):
            X = X_or_none if X_or_none is not None else design_points("random", N, d, t)
            items.append(((method, int(N)), X, t))

    for N, a in zip(primes, gens):
        add_cfg("korobov", N, (KorobovLattice(d=d, N=int(N), generator=int(a)).points + 0.5) % 1.0)  # half-shift

    alpha = np.array([2.0 ** (j / (d + 1)) for j in range(1, d + 1)])
    seen_Q = set()
    for m in m_range:
        lat = ExplicitRank1Lattice(d=d, m=m, alpha=alpha)
        if lat.Q in seen_Q or lat.Q < 3:
            continue
        seen_Q.add(lat.Q)
        add_cfg("explicit", lat.Q, (lat.points + 0.5) % 1.0)  # half-shift

    for method in ["halton", "sobol", "lhs"]:
        for n in powers:
            add_cfg(method, n, design_points(method, n, d, 0))
    for n in powers:  # random: X varies per trial
        add_cfg("random", n, None)

    errs = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_trial_error)(X, func, d, ls, t) for (_, X, t) in items)

    # Aggregate by cfg key
    from collections import defaultdict
    bucket = defaultdict(list)
    for (key, _, _), e in zip(items, errs):
        bucket[key].append(e)
    for (method, N) in cfg_order:
        vals = bucket[(method, N)]
        res[method]["N"].append(N)
        res[method]["mean"].append(float(np.mean(vals)))
        res[method]["std"].append(float(np.std(vals)))
    return res


STYLES = {
    "korobov":  {"color": "#1f77b4", "marker": "o", "label": "Korobov (ours)"},
    "explicit": {"color": "#d62728", "marker": "s", "label": "Explicit (ours)"},
    "lhs":      {"color": "#ff7f0e", "marker": "P", "label": "Maximin LHS"},
    "random":   {"color": "#7f7f7f", "marker": "x", "label": "Random"},
    "halton":   {"color": "#2ca02c", "marker": "^", "label": "Halton"},
    "sobol":    {"color": "#9467bd", "marker": "D", "label": "Sobol'"},
}


def plot_res(res, outfile):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, st in STYLES.items():
        if res[m]["N"]:
            ax.errorbar(res[m]["N"], res[m]["mean"], yerr=res[m]["std"],
                        marker=st["marker"], color=st["color"], label=st["label"],
                        capsize=3, capthick=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of points $N$"); ax.set_ylabel("$L^2$ error")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, outfile), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# Test functions
# ----------------------------------------------------------------------
def franke(X):
    x, y = X[:, 0], X[:, 1]
    return (0.75 * np.exp(-((9*x-2)**2 + (9*y-2)**2)/4)
            + 0.75 * np.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
            + 0.5 * np.exp(-((9*x-7)**2 + (9*y-3)**2)/4)
            - 0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2))

def arctan2d(X):
    xt, yt = 2*X[:, 0]-1, 2*X[:, 1]-1
    return np.arctan(2*(xt + 3*yt - 1)) / np.arctan(2*(np.sqrt(10)+1))

def gauss_peak(X):
    xt, yt = 2*X[:, 0]-1, 2*X[:, 1]-1
    return np.exp(-((xt-0.1)**2 + 0.5*yt**2))

def oscillatory(X):
    xt, yt = 2*X[:, 0]-1, 2*X[:, 1]-1
    return np.sin(np.pi*(xt**2 + yt**2))

def genz_cont(X):
    d = X.shape[1]
    return genz_continuous(X, c=np.linspace(1.5, 2.5, d))

def genz_gauss(X):
    d = X.shape[1]
    return genz_gaussian(X, c=np.full(d, 3.0))


JOBS = [
    (2, franke,      "Franke",          "GPR_2d_Franke.png"),
    (2, arctan2d,    "Arctan",          "GPR_2d_Arctan.png"),
    (2, gauss_peak,  "GaussianPeak",    "GPR_2d_Gaussian.png"),
    (2, oscillatory, "Oscillatory",     "GPR_2d_oscillatory.png"),
    (3, genz_cont,   "GenzContinuous",  "GPR_3d_continuous.png"),
    (3, genz_gauss,  "GenzGaussian",    "GPR_3d_Gaussian.png"),
    (5, pairwise_trig, "PairwiseTrig",  "GPR_5d.png"),
    (5, genz_cont,   "GenzContinuous",  "GPR_5d_GenzContinuous.png"),
    (5, genz_gauss,  "GenzGaussian",    "GPR_5d_GenzGaussian.png"),
    (7, pairwise_trig, "PairwiseTrig",  "GPR_7d.png"),
    (7, genz_cont,   "GenzContinuous",  "GPR_7d_GenzContinuous.png"),
    (7, genz_gauss,  "GenzGaussian",    "GPR_7d_GenzGaussian.png"),
]


def main():
    open(LOG, "w").close()
    all_results = {}
    t_start = time.time()
    jobs = JOBS[:2] if SMOKE else JOBS
    for d, func, fname, outfile in jobs:
        t0 = time.time()
        log(f"[{time.strftime('%H:%M:%S')}] START d={d} {fname}")
        res = run_one(d, func, fname)
        plot_res(res, outfile)
        all_results[(d, fname)] = res
        with open(os.path.join(REVDIR, "gpr_all_results.pkl"), "wb") as f:
            pickle.dump(all_results, f)
        log(f"[{time.strftime('%H:%M:%S')}] DONE  d={d} {fname} -> {outfile} "
            f"({time.time()-t0:.0f}s, total {time.time()-t_start:.0f}s)")
    log(f"ALL DONE in {time.time()-t_start:.0f}s")
    # marker
    open(os.path.join(REVDIR, "gpr_all.DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
