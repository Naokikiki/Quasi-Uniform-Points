"""
Patch the four GPR notebooks for the revised manuscript:
  * add a maximin LHS baseline to the generators, runner(s) and plot styles;
  * add Genz Continuous / Gaussian test functions and runs in d=5 and d=7.

Outputs are cleared (the GPR runs are expensive); re-execute the notebooks to
regenerate the inline figures. Figure file names follow each notebook's own
convention (gpr_error_*.png), unchanged.
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GEN_FUNC = '''def generate_maximin_lhs(n: int, d: int) -> np.ndarray:
    """Generate n maximin-optimized LHS points in [0, 1)^d."""
    from lattice_qmc.baselines import maximin_lhs
    return maximin_lhs(n, d, seed=2024)

'''

LHS_BLOCK_MC = '''    # Maximin LHS
    print("\\nMaximin LHS:")
    for n in powers:
        print(f"  N = {n}: ", end="", flush=True)
        X_train = generate_maximin_lhs(n, d)
        y_true = test_func(X_train)
        errors = []
        for trial in range(n_trials):
            np.random.seed(1000 + trial)
            y_noisy = y_true + np.random.normal(0, NOISE_STD, size=y_true.shape)
            gpr = fit_gpr(X_train, y_noisy)
            l2_error = compute_l2_error_mc(gpr, test_func, d=d)
            errors.append(l2_error)
        results['lhs']['N'].append(n)
        results['lhs']['l2_error_mean'].append(np.mean(errors))
        results['lhs']['l2_error_std'].append(np.std(errors))
        print(f"L² = {np.mean(errors):.6e} ± {np.std(errors):.6e}")

'''

LHS_BLOCK_GL = '''    # Maximin LHS
    print("\\nMaximin LHS:")
    for N in powers:
        print(f"  N = {N}: ", end="", flush=True)
        X_train = generate_maximin_lhs(N, d)
        y_true = test_func(X_train)
        errors = []
        for trial in range(n_trials):
            np.random.seed(1000 + trial)
            y_noisy = y_true + np.random.normal(0, NOISE_STD, size=y_true.shape)
            gpr = fit_gpr(X_train, y_noisy)
            l2_error = compute_l2_error(gpr, test_func, d=d)
            errors.append(l2_error)
        results['lhs']['N'].append(N)
        results['lhs']['l2_error_mean'].append(np.mean(errors))
        results['lhs']['l2_error_std'].append(np.std(errors))
        print(f"L² error = {np.mean(errors):.6e} ± {np.std(errors):.6e}")

'''

LHS_STYLE = ("        'explicit': {'color': '#d62728', 'marker': 's', 'label': 'Explicit (ours)'},\n"
             "        'lhs': {'color': '#ff7f0e', 'marker': 'P', 'label': 'Maximin LHS'},")

GENZ_DEFS = '''# --- Genz test functions (added for the revision; evaluated in d=5,7) ---
def genz_continuous(X):
    """Genz continuous: exp(-sum c_i |x_i - 0.5|), c_i equally spaced in [1.5, 2.5]."""
    d = X.shape[1]
    c = np.linspace(1.5, 2.5, d)
    return np.exp(-np.sum(c * np.abs(X - 0.5), axis=1))

def genz_gaussian(X):
    """Genz Gaussian: exp(-sum c_i^2 (x_i - 0.5)^2), c_i = 3.0."""
    d = X.shape[1]
    c = np.full(d, 3.0)
    return np.exp(-np.sum(c**2 * (X - 0.5)**2, axis=1))

print("Genz functions loaded.")
'''


def patch(path, dim):
    nb = json.load(open(path))
    runner_name = None
    n_runner = n_gen = n_plot = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        # generator cell
        if "def generate_sobol_points" in s and "generate_maximin_lhs" not in s:
            s = s.replace('print("Point set generators loaded.")',
                          GEN_FUNC + 'print("Point set generators loaded.")')
            n_gen += 1
        # runner cell(s)
        if "def run_gpr_experiment" in s:
            m = re.search(r"def (run_gpr_experiment\w*)\(", s)
            if m:
                runner_name = m.group(1)
            if "'lhs'" not in s:
                s = s.replace(
                    "        'sobol': {'N': [], 'l2_error_mean': [], 'l2_error_std': []},",
                    "        'sobol': {'N': [], 'l2_error_mean': [], 'l2_error_std': []},\n"
                    "        'lhs': {'N': [], 'l2_error_mean': [], 'l2_error_std': []},")
                block = LHS_BLOCK_GL if "compute_l2_error_mc" not in s else LHS_BLOCK_MC
                # insert before the function's "return results"
                s = s.replace("    return results", block + "    return results", 1)
                n_runner += 1
        # plot cell
        if "def plot_gpr_results" in s and "'lhs'" not in s:
            s = s.replace(
                "        'explicit': {'color': '#d62728', 'marker': 's', 'label': 'Explicit (ours)'},",
                LHS_STYLE)
            n_plot += 1
        c["source"] = s.splitlines(keepends=True)

    # Genz cells for d = 5, 7
    if dim in (5, 7) and runner_name:
        already = any("def genz_continuous" in "".join(c["source"])
                      for c in nb["cells"] if c["cell_type"] == "code")
        if not already:
            cells_new = [
                {"cell_type": "markdown", "metadata": {},
                 "source": ["## Genz test functions (revision)\n\n",
                            "Added for the revised manuscript: the Genz Continuous and "
                            "Genz Gaussian functions are evaluated in this dimension to "
                            "complement the pairwise trigonometric function."]},
                {"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "source": GENZ_DEFS.splitlines(keepends=True)},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": [f"results_genz_cont = {runner_name}(genz_continuous, \"Genz Continuous\")\n",
                            f"plot_gpr_results(results_genz_cont, d={dim}, func_name=\"Genz Continuous\")"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": [f"results_genz_gauss = {runner_name}(genz_gaussian, \"Genz Gaussian\")\n",
                            f"plot_gpr_results(results_genz_gauss, d={dim}, func_name=\"Genz Gaussian\")"]},
            ]
            nb["cells"].extend(cells_new)

    # clear outputs (GPR runs are expensive; user re-executes)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"{os.path.basename(path)}: gen={n_gen} runner={n_runner} plot={n_plot} "
          f"runner_name={runner_name} genz={'yes' if dim in (5,7) else 'n/a'}")


if __name__ == "__main__":
    patch(os.path.join(ROOT, "experiments_04_02_03_gpr_2d.ipynb"), 2)
    patch(os.path.join(ROOT, "experiments_04_02_03_gpr_3d.ipynb"), 3)
    patch(os.path.join(ROOT, "experiments_04_02_03_gpr_5d.ipynb"), 5)
    patch(os.path.join(ROOT, "experiments_04_02_03_gpr_7d.ipynb"), 7)
