#%% CONSTANT SCHEDULE

from unittest import result

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0
from scipy.special import zeta
from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    compute_different_sigmas,
)
from src.utils import save_dict_to_json
# %%
dim = 100
sigma = 0.5
exponent = 2 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000]
list_alphas = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

laplace_constant = LaplaceConstant(model, x0)

#%% 
print("Comparing variance trajectories for different alphas...")
laplace_variance, diagonal_variance = laplace_constant.compare_variance_trajectories_different_alphas(T_values, list_alphas, m_constant=Delta)
#%%
laplace_variance_str = {str((alpha, T)): var for (alpha, T), var in laplace_variance.items()}
save_dict_to_json(laplace_variance_str, folder=f"laplace_constant_dim={dim}", filename="variance_trajectories.json")
diagonal_variance_str = {str((alpha, T)): var for (alpha, T), var in diagonal_variance.items()}
save_dict_to_json(diagonal_variance_str, folder=f"laplace_constant_dim={dim}", filename="diagonal_variance_trajectories.json")
# %%
ratios = {key: laplace_variance[key] / diagonal_variance[key] for key in laplace_variance.keys()}

colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))

plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y = zeta(alpha)*(alpha-1) *np.array([ratios[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y, label=f"Variance Ratio (alpha={alpha})", marker='o', color=color)
plt.xscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylim(0, 3)
plt.ylabel("Variance (log scale)")
plt.title("Variance Trajectories for Different Alphas (Laplace vs Diagonal)")
plt.legend()
plt.grid()
plt.savefig("images/CONSTANT_variance_trajectories_comparison.png")
plt.show()
# %%
print(ratios)
# %%
