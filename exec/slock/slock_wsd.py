#%% WSD  SCHEDULE


import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0
from scipy.special import zeta
from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    SlockLinear,
    SlockWSD,
    Mode,
    compute_different_sigmas,
)
from src.utils import save_dict_to_json
# %%
dim = 100
sigma = 1
exponent = 1.5 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000]
#T_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
list_alphas = [1.1, 1.5, 2.5]
cooldown_len = 0.2  # Cooldown length for WSD schedule

eta = 0.01
slock_wsd = SlockWSD(model, x0, beta=beta, T_max=max(T_values), optimize=optimize, base_lr=eta, cooldown_len=cooldown_len)
mode = Mode.SLOCK

#%%

def changing_dim(T, alpha):
    #return int((T/100)**(1/alpha))
    #return 100
    return min(1000, int(10 * (T/10)**(1/alpha)))
    #return min(T, 2000)

#%% 
print("Comparing variance trajectories for different alphas...")
slock_variance, diagonal_variance, slock_bias, diagonal_bias = (
    slock_wsd.compare_biases_variances_trajectories_different_alphas(
        T_values,
        list_alphas, 
        m_exponent=model.exponent, 
        m_constant=Delta, 
        changing_dim=changing_dim,
        mode=mode,
        from_file=False)
)
#%%

#Plot trajectories

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y_slock = np.array([slock_variance[(alpha, T)] for T in T_values])
    Y_diagonal = np.array([diagonal_variance[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y_slock, label=f"SLOCK Variance (alpha={alpha})", marker='o', color=color)
    plt.plot(T_values, Y_diagonal, label=f"Diagonal Variance (alpha={alpha})", marker='x', linestyle='--', color=color)
plt.xscale('log')
plt.yscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylabel("Variance (log scale)")
plt.title(f"Variance Trajectories for Different Alphas (SLOCK vs Diagonal) eta={eta}")
plt.legend()
plt.grid()
plt.savefig("images/slock/_WSD_variance_trajectories_comparison.pdf")
plt.show()

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y_slock = np.array([slock_bias[(alpha, T)] for T in T_values])
    Y_diagonal = np.array([diagonal_bias[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y_slock, label=f"SLOCK Bias (alpha={alpha})", marker='o', color=color)
    plt.plot(T_values, Y_diagonal, label=f"Diagonal Bias (alpha={alpha})", marker='x', linestyle='--', color=color)
plt.xscale('log')
plt.yscale('log')
plt.xlim(100, max(T_values))
plt.xlabel("T (log scale)")
plt.ylabel("Bias (log scale)")
plt.title(f"Bias Trajectories for Different Alphas (SLOCK vs Diagonal) eta={eta}")
plt.legend()
plt.grid()
plt.savefig("images/slock/_WSD_bias_trajectories_comparison.pdf")
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
plt.savefig(f"images/slock/_WSD_variance_trajectories_comparison_eta={eta}.pdf")
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
plt.title(f"Bias Ratios for Different Alphas (SLOCK vs Laplace) eta={eta}")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/_WSD_bias_ratios_comparison_eta={eta}.pdf")
plt.show()
# %%
print(ratios_variance)
print(ratios_bias)
# %%

#COMPARE COOLDOWN LENGTHS

cooldown_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
eta = 0.001
T_values = [1000, 5000, 10000, 20000, 50000, 100000]
list_alphas = [1.1, 1.5, 2.5]
risks_slock = {}
for cooldown_len in cooldown_list:
    slock_wsd = SlockWSD(model, x0, beta=beta, T_max=max(T_values), optimize=False, base_lr=eta, cooldown_len=cooldown_len)
    mode = Mode.SLOCK
    slock_variance, diagonal_variance, slock_bias, diagonal_bias = (
        slock_wsd.compare_biases_variances_trajectories_different_alphas(
            T_values,
            list_alphas, 
            m_exponent=model.exponent, 
            m_constant=Delta, 
            changing_dim=changing_dim,
            mode=mode,
            from_file=False)
    )
    risks_slock[cooldown_len] = {alpha: [slock_variance[(alpha, T)] + slock_bias[(alpha, T)] for T in T_values] for alpha in list_alphas}
# %% compare
colors = plt.cm.viridis(np.linspace(0, 1, len(T_values)))
for alpha in list_alphas:
    plt.figure(figsize=(12, 8))
    for T in T_values:
        color = colors[T_values.index(T)]
        risk = [risks_slock[cooldown_len][alpha][T_values.index(T)] for cooldown_len in cooldown_list]
        plt.plot(cooldown_list, risk, label=f"B+V (T={T})", marker='o', color=color)
    plt.xscale('linear')
    plt.xlabel("Cooldown Length")
    plt.ylabel("Risk")
    plt.yscale('log')
    plt.title(f"Risk for Different cooldown lengths (SLOCK vs Diagonal) eta={eta}, alpha={alpha}")
    plt.legend()
    plt.grid()
    plt.savefig(f"images/slock/_WSD_risk_cooldown_comparison_eta={eta}_alpha={alpha}.pdf")
    plt.show()

# %%
#COMPARE BEST ETA 
risks_approx = {}
etas = np.logspace(-4, -1, 10)
alpha = 2.2
T = 100000
dim = 100
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=alpha)
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink"]

for eta in etas:
    slock_wsd = SlockWSD(model, x0, beta=beta, T_max=T, optimize=False, base_lr=eta, cooldown_len=cooldown_len)
    mode = Mode.SLOCK
    slock_risk = slock_wsd.compute_slock_approx_risk(T=T,  m_constant=Delta)
    risks_approx[eta] = slock_risk

eta_star = slock_wsd.compute_best_slock_eta(T=T, m_constant=Delta)
print(f"Optimal eta for alpha={alpha}, T={T}, cooldown_len={cooldown_len}: {eta_star}")

plt.figure(figsize=(12, 8))
color = "red"
risk_val = [risks_approx[eta] for eta in etas]
plt.plot(etas, risk_val, label=f"B+V (alpha={alpha}, T={T})", marker='o', color=color)
plt.axvline(x=eta_star, color='black', linestyle='--', label=f"Computed Optimal eta={eta_star:.4f}")
plt.xscale('log')
plt.xlabel("Learning Rate (eta)")
plt.ylabel("Risk")
plt.yscale('log')
plt.title(f"Risk for Different Learning Rates (True vs Approx) alpha={alpha}, T={T}")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/_WSD_risk_optimal_eta_comparison_alpha={alpha}_T={T}.pdf")
plt.show()
# %%
#With TRUE also
etas = np.logspace(-4, -1, 10)
alpha = 2.2
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=alpha)
risks_true = {}
risks_approx = {}
for eta in etas:
    slock_wsd = SlockWSD(model, x0, beta=beta, T_max=T, optimize=False, base_lr=eta, cooldown_len=cooldown_len)
    mode = Mode.SLOCK
    approx_variance, true_variance, approx_bias, true_bias = (
    slock_wsd.compare_biases_variances_trajectories_different_alphas(
        [T],
        [alpha], 
        m_exponent=model.exponent, 
        m_constant=Delta, 
        mode=mode,
        from_file=False)
    )
    risks_true[eta] = true_variance[(alpha, T)] + true_bias[(alpha, T)]
    risks_approx[eta] = approx_variance[(alpha, T)] + approx_bias[(alpha, T)]
#%%
eta_star = slock_wsd.compute_best_slock_eta(T=T, m_constant=Delta)

plt.figure(figsize=(12, 8))
color = "green"
plt.plot(etas, [risks_true[eta] for eta in etas], label=f"True Risk (alpha={alpha}, T={T})", marker='o', color=color)
plt.plot(etas, [risks_approx[eta] for eta in etas], label=f"SLOCK Approx Risk (alpha={alpha}, T={T})", marker='x', color="red")
plt.axvline(x=eta_star, color='black', linestyle='--', label=f"Computed Optimal eta={eta_star:.4f}")
plt.xscale('log')
plt.xlabel("Learning Rate (eta)")
plt.ylabel("Risk")
plt.yscale('log')
plt.title(f"True Risk for Different Learning Rates (True vs Approx) alpha={alpha}, T={T}")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/_WSD_true_risk_comparison_alpha={alpha}_T={T}.pdf")
plt.show()
# %%



#With BEST ETA OPtimized
def changing_dim(T, alpha):
    return 200

with_eta_star = False

cooldown_list = [0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8, 0.9]
T = 100000
list_alphas = [1.2, 1.8, 2.5]
approx_slock = {}
true_slock = {}
for cooldown_len in cooldown_list:
    print(f"COOLDOWN LENGTH: {cooldown_len}")
    slock_wsd = SlockWSD(model, x0, beta=beta, T_max=max(T_values), optimize=False, base_lr=eta, cooldown_len=cooldown_len)
    mode = Mode.SLOCK
    approx_variance, true_variance, approx_bias, true_bias = (
        slock_wsd.compare_biases_variances_trajectories_different_alphas(
            [T],
            list_alphas, 
            m_exponent=model.exponent, 
            m_constant=Delta, 
            changing_dim=changing_dim,
            mode=mode,
            from_file=False,
            with_eta_star=with_eta_star)
        )
    true_slock[cooldown_len] = {alpha: true_variance[(alpha, T)] + true_bias[(alpha, T)]  for alpha in list_alphas}
    approx_slock[cooldown_len] = {alpha: approx_variance[(alpha, T)] + approx_bias[(alpha, T)]  for alpha in list_alphas}

# %% compare
if with_eta_star:
    title = f"Risk optimized eta* (True vs Approx)"
else:
    title = f"Risk for eta={eta} (True vs Approx)"

colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))
plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    risk = [approx_slock[cooldown_len][alpha] for cooldown_len in cooldown_list]
    plt.plot(cooldown_list, risk, label=f"B+V (T={T})", marker='o', color=color)
    plt.xscale('linear')
    plt.xlabel("Cooldown Length")
    plt.ylabel("Risk")
    plt.yscale('log')
    plt.title(f"{title}, alpha={alpha}")
    plt.legend()
    plt.grid()
    if with_eta_star:
        plt.savefig(f"images/slock/_WSD_risk_cooldown_comparison_alpha={alpha}_optimized_eta.pdf")
    else:
        plt.savefig(f"images/slock/_WSD_risk_cooldown_comparison_alpha={alpha}_eta={eta}.pdf")
    plt.show()

# %%




#COMPARE BEST ETAS vs NUMERICAL OPTIMIZATION
alphas = [1+1e-6, 1.001, 1.01, 1.1, 1.5, 2.5, 10, 20 , 50]
best_etas = {}
exact_etas = {}
T= 100000
for alpha in alphas:
    slock_wsd._update_model_for_alpha(alpha)  # Update model for the specific alpha
    slock_wsd._setup_for_T(T, base_lr=0.001)  # Setup for T with a placeholder learning rate
    best_etas[alpha] = slock_wsd.compute_best_slock_eta(T, m_constant=Delta)
    exact_etas[alpha] = slock_wsd.compute_exact_eta(T, m_constant=Delta)
# %%
print("Best SLOCK Etas:", best_etas)
print("Exact Etas (Numerical Optimization):", exact_etas)
plt.figure(figsize=(12, 8))
plt.plot(alphas, [best_etas[alpha] for alpha in alphas], label="Best SLOCK Eta", marker='o')
plt.plot(alphas, [exact_etas[alpha] for alpha in alphas], label="Exact Eta (Numerical Optimization)", marker='x')
plt.xlabel("Alpha")
plt.ylabel("Eta")
plt.title(f"Comparison of Best SLOCK Eta vs Exact Eta (Numerical Optimization); T={T}, Delta={Delta}")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/CONSTANT_BATCH_comparison_best_vs_exact_eta_T={T}_Delta={Delta}.pdf")
plt.show()

ratios = {alpha: best_etas[alpha] / exact_etas[alpha] for alpha in alphas}
plt.figure(figsize=(12, 8))
plt.plot(alphas, [ratios[alpha] for alpha in alphas], label="Ratio of Best SLOCK Eta to Exact Eta", marker='o')
plt.xlabel("Alpha")
plt.ylabel("Ratio")
plt.xscale('log')
plt.title(f"Ratio of Best SLOCK Eta to Exact Eta (Numerical Optimization); T={T}, Delta={Delta}")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/CONSTANT_BATCH_comparison_eta_ratio_T={T}_Delta={Delta}.pdf")
plt.show()
# %%


#FIGURE 11 Fabian
T_list = [400, 1600, 6400, 25600, 102400]
c_list = np.linspace(0.01, 1.0, 100)
colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(T_list)))

results_eta_ratio = {}
for T in T_list:
    results_eta_ratio[T] = {}
    slock_wsd._setup_for_T(T, cooldown_len=1.0)  # Setup for T with a placeholder learning rate
    eta_1 = slock_wsd.compute_best_slock_eta(T=T, m_constant=Delta, c=1.0)  # Compute eta_star for cooldown length 1
    for c in c_list:
        slock_wsd._setup_for_T(T, cooldown_len=c) 
        eta_star = slock_wsd.compute_best_slock_eta(T=T, m_constant=Delta, c=c)
        results_eta_ratio[T][c] = np.log(eta_1 / eta_star)
        print(f"T={T}, cooldown={c}, eta_star={eta_star}, eta_1={eta_1}, ratio={results_eta_ratio[T][c]}")

plt.figure(figsize=(12, 8))
for T in T_list:
    color = colors[T_list.index(T)]
    plt.plot(c_list, [results_eta_ratio[T][c] for c in c_list], label=f"T={T}", marker='o', color=color)
plt.xscale('linear')
plt.xlabel("Cooldown Length (c)")
plt.ylabel("log(eta_star(1) / eta_star(c))")
plt.title(f"Effect of Cooldown Length on Optimal Learning Rate (eta_star) for Different T")
plt.legend()
plt.grid()
plt.savefig(f"images/slock/CONSTANT_BATCH_eta_ratio_vs_cooldown.pdf")
plt.show()


save_dict_to_json(results_eta_ratio, folder=f"figures", filename="eta_ratio_vs_cooldown.json")
# %%
