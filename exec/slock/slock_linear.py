#%% LINEAR  SCHEDULE


import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0
from scipy.special import zeta
from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    SlockLinear,
    Mode,
    compute_different_sigmas,
)
from src.utils import save_dict_to_json
# %%
dim = 100
sigma = 1
exponent = 2 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]
#T_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
list_alphas = [1.4, 1.9, 2.5]

eta = 0.00001
slock_linear = SlockLinear(model, x0, beta=beta, T_max=max(T_values), optimize=optimize, base_lr=eta)
mode = Mode.SLOCK
#%%

def changing_dim(T, alpha):
    #return int((T/100)**(1/alpha))
    return 100
    #return min(1000, int(100 * (T/10)**(1/alpha)))
    #return min(T, 2000)

#%% 
print("Comparing variance trajectories for different alphas...")
slock_variance, diagonal_variance, slock_bias, diagonal_bias = (
    slock_linear.compare_biases_variances_trajectories_different_alphas(
        T_values,
        list_alphas, 
        m_exponent=model.exponent, 
        m_constant=Delta, 
        changing_dim=changing_dim,
        mode=mode,
        from_file=False)
)
#%%

save_dict_to_json({str((alpha, T)): var for (alpha, T), var in slock_variance.items()}, folder=f"slock_linear_dim={dim}", filename="variance_trajectories.json")
save_dict_to_json({str((alpha, T)): var for (alpha, T), var in diagonal_variance.items()}, folder=f"slock_linear_dim={dim}", filename="true_variance_trajectories.json")
save_dict_to_json({str((alpha, T)): bias for (alpha, T), bias in slock_bias.items()}, folder=f"slock_linear_dim={dim}", filename="bias_trajectories.json")
save_dict_to_json({str((alpha, T)): bias for (alpha, T), bias in diagonal_bias.items()}, folder=f"slock_linear_dim={dim}", filename="true_bias_trajectories.json")

#%%
#Plot trajectories

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y_slock = np.array([slock_variance[(alpha, T)] for T in T_values])
    Y_diagonal = np.array([diagonal_variance[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y_slock, label=f"{mode.value} Variance (alpha={alpha})", marker='o', color=color)
    plt.plot(T_values, Y_diagonal, label=f"True Variance (alpha={alpha})", marker='x', linestyle='--', color=color)
plt.xscale('log')
plt.yscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylabel("Variance (log scale)")
plt.title(f"Variance Trajectories for Different Alphas ({mode.value} vs True) eta={eta}")
plt.legend()
plt.grid()
plt.savefig("images/slock/_LINEAR_variance_trajectories_comparison.pdf")
plt.show()

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y_slock = np.array([slock_bias[(alpha, T)] for T in T_values])
    Y_diagonal = np.array([diagonal_bias[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y_slock, label=f"{mode.value} Bias (alpha={alpha})", marker='o', color=color)
    plt.plot(T_values, Y_diagonal, label=f"True Bias (alpha={alpha})", marker='x', linestyle='--', color=color)
plt.xscale('log')
plt.yscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylabel("Bias (log scale)")
plt.title(f"Bias Trajectories for Different Alphas ({mode.value} vs True) eta={eta}")
plt.legend()
plt.grid()
plt.savefig("images/slock/_LINEAR_bias_trajectories_comparison.pdf")
plt.show()


#%%
ratios_variance = {key: slock_variance[key] / diagonal_variance[key] for key in slock_variance.keys()}
ratios_bias = {key: slock_bias[key] / diagonal_bias[key] for key in slock_variance.keys()}

colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))




plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y = np.array([ratios_variance[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y, label=f"Variance Ratio (alpha={alpha})", marker='o', color=color)
plt.xscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylim(0, 5)
plt.ylabel("Variance (log scale)")
plt.title(f"Variance Trajectories for Different Alphas (SLOCK vs Diagonal) eta={eta}")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/_LINEAR_variance_trajectories_comparison_eta={eta}.pdf")
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
plt.title(f"Bias Ratios for Different Alphas ({mode.value} vs True  ) eta={eta}")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/_LINEAR_bias_ratios_comparison_eta={eta}.pdf")
plt.show()
# %%
print(ratios_variance)
print(ratios_bias)
# %%
#COMPARE ETAS

#ETAS
T = 100000
list_alphas = [1.4, 1.9, 2.5]
etas = np.logspace(-4, -2, 10)
eta_stars = {}
for alpha in list_alphas:
    slock_linear._update_model_for_alpha(alpha)
    eta_star = slock_linear.compute_best_slock_eta(T, m_constant=Delta)
    eta_stars[alpha] = eta_star
print(f"Optimal etas for T={T}: {eta_stars}")
#%%
risks = {}
for alpha in list_alphas:
    risks[alpha] = {}
    slock_linear._update_model_for_alpha(alpha)
    for eta in etas:
        print(f"Computing risk for alpha={alpha}, eta={eta}...")
        slock_linear._setup_for_T(T, base_lr=eta)
        biases, variances = slock_linear.compute_slock_biases_and_variances([T])
        risks[alpha][eta] = biases[T] + variances[T]
# %%
colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))

plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    plt.plot(etas, [risks[alpha][eta] for eta in etas], label=f"Risk (alpha={alpha})", marker='o', color=color)
    plt.axvline(x=eta_stars[alpha], color=color, linestyle='--', label=f"Optimal eta (alpha={alpha})")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("eta (log scale)")
plt.ylabel("Risk (log scale)")
plt.title("Risk for Different Alphas (SLOCK)")
plt.legend()
plt.grid()
plt.savefig("images/slock/_LINEAR_eta_comparison.png")
plt.show()

#%%
save_dict_to_json({str(alpha): risks[alpha] for alpha in list_alphas}, folder=f"slock_linear_dim={dim}", filename="risks_for_different_etas.json")
save_dict_to_json({str(alpha): eta_stars[alpha] for alpha in list_alphas}, folder=f"slock_linear_dim={dim}", filename="optimal_etas.json")

# %%

T = 100000
list_alphas = np.linspace(1.4, 2.5, 10)
etas = np.logspace(-4, -1, 50)
eta_stars__ = {}
for alpha in list_alphas:
    print(f"Computing optimal eta for alpha={alpha}...")
    slock_linear._update_model_for_alpha(alpha)
    eta_star = slock_linear.compute_best_slock_eta(T, m_constant=Delta)
    eta_stars__[alpha] = eta_star
print(f"Optimal etas for T={T}: {eta_stars__}")
print(eta_stars__)


# %%
