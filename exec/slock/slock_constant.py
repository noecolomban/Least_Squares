#%% CONSTANT SCHEDULE


import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0
from scipy.special import zeta
from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    SlockConstant,
    Mode,
    compute_different_sigmas,
)
from src.utils import save_dict_to_json
# %%
dim = 1000
sigma = 0.1
exponent = 2 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
list_alphas = [1.5, 2, 2.5]

slock_constant = SlockConstant(model, x0)
mode = Mode.SLOCK

#%% 
print("Comparing variance trajectories for different alphas...")
slock_variance, true_variance, slock_bias, true_bias = (
    slock_constant.compare_biases_variances_trajectories_different_alphas(
        T_values,
        list_alphas, 
        m_exponent=model.exponent, 
        m_constant=Delta, 
        mode=mode)
)
#%%
slock_variance_str = {str((alpha, T)): var for (alpha, T), var in slock_variance.items()}
save_dict_to_json(slock_variance_str, folder=f"slock_constant_dim={dim}", filename="variance_trajectories.json")
true_variance_str = {str((alpha, T)): var for (alpha, T), var in true_variance.items()}
save_dict_to_json(true_variance_str, folder=f"slock_constant_dim={dim}", filename="true_variance_trajectories.json")
# %%
ratios_variance = {key: slock_variance[key] / true_variance[key] for key in slock_variance.keys()}
ratios_bias = {key: slock_bias[key] / true_bias[key] for key in slock_variance.keys()}

colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))

plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y = np.array([ratios_variance[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y, label=f"Variance Ratio (alpha={alpha})", marker='o', color=color)
plt.xscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylim(0, 3)
plt.ylabel("Variance (log scale)")
plt.title("Variance Trajectories for Different Alphas (SLOCK vs Diagonal)")
plt.legend()
plt.grid()
plt.savefig("images/slock/_CONSTANT_variance_trajectories_comparison.png")
plt.show()

#
plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y = np.array([ratios_bias[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y, label=f"Bias Ratio (alpha={alpha})", marker='o', color=color)
plt.xscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylim(0, 3)
plt.ylabel("Bias Ratio")
plt.title("Bias Ratios for Different Alphas (SLOCK vs True Bias from Diagonal Approximation)")
plt.legend()
plt.grid()
plt.savefig("images/slock/_CONSTANT_bias_ratios_comparison.png")
plt.show()
# %%
print(ratios_variance)
print(ratios_bias)
# %%
#Plot all variances (not ratios)
plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    plt.plot(T_values, [slock_variance[(alpha, T)] for T in T_values], label=f"Slock Variance (alpha={alpha})", marker='o', color=color)
    plt.plot(T_values, [true_variance[(alpha, T)] for T in T_values], label=f"True Variance (alpha={alpha})", marker='x', linestyle='--', color=color)
plt.xscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylabel("Value")
plt.title("Variance and Bias for Different Alphas (SLOCK vs True from Diagonal Approximation)")
plt.legend()
plt.grid()
plt.savefig("images/slock/_CONSTANT_variance_bias_comparison.png")
plt.show()
# %%
