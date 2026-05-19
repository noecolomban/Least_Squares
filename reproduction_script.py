import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import traceback
from models.models import LinearModel
from models.initialization import compute_power_x0
from solvers.linear_solvers import LaplaceLinear
from simulations.variance_analysis import compare_different_alphas_variance

try:
    dim = 100
    sigma = 0.1
    exponent = 2
    beta = 1
    T_max = 100000
    optimize = False
    base_lr = 0.001

    model = LinearModel(dim=dim, sigma=sigma, exponent=exponent)
    x0 = compute_power_x0(model, beta=beta)
    solver = LaplaceLinear(model, x0, T_max=T_max, optimize=optimize, base_lr=base_lr)

    list_alphas = np.linspace(1.1, 3.0, 15)
    T_values = [10000, 30000, 100000]
    K = 1

    for T in T_values:
        try:
            laplace, diagonal = compare_different_alphas_variance(solver, list_alphas, T, K)
            print(f"T={T}: Success. laplace min={np.min(laplace):.2e}, max={np.max(laplace):.2e}; diagonal min={np.min(diagonal):.2e}, max={np.max(diagonal):.2e}")
        except Exception:
            print(f"T={T}: Failed")
            traceback.print_exc()

except Exception:
    traceback.print_exc()
    sys.exit(1)
