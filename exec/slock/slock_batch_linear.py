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

def batch_factor_linear(batch, alpha, beta):
    '''So that eta_star is batch_factor_linear * best_slock_eta(T) for the given batch size.'''
    if alpha <2:
        return batch**(alpha/(alpha+beta))
    if alpha>2:
        return batch**((2*alpha)/(3*alpha+2*beta-2))




#%%
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
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
#T_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
list_alphas = [1.01, 1.1, 1.5, 2.5, 10]

eta = 0.001
slock_linear = SlockLinear(model, x0, beta=beta, T_max=max(T_values), optimize=optimize, base_lr=eta)
mode = Mode.SLOCK
#%%
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]
batches = [1, 10, 100]  # Different batch sizes to compare
alpha = 1.1
biases, variances = {}, {}
slock_linear._update_model_for_alpha(alpha)  # Update model for the specific alpha

for batch in batches:
    biases[batch], variances[batch] = {}, {}
    for T in T_values:
        print(f"Computing biases and variances for batch size {batch} and T={T}...")
        eta_star = batch_factor_linear(batch, alpha, beta)*slock_linear.compute_best_slock_eta(T, m_constant=Delta)
        slock_linear._setup_for_T(T, base_lr=eta_star)  # Update schedule for new T
        bias, variance = slock_linear.compute_slock_biases_and_variances([T], batch=batch, K=1)
        biases[batch][T] = bias[T]
        variances[batch][T] = variance[T]

#%%
#%%
#Variance
plt.figure(figsize=(12, 8))
plt.title(f"Slock Biases and Variances for Different Batch Sizes; alpha={alpha}. optiized eta_star for each T")
for batch in batches:
    X = [T*batch for T in T_values]  # Scale T by batch size for plotting
    Y = [biases[batch][T] for T in T_values]
    plt.plot(X, Y, label=f'Bias (Batch={batch})', marker='o')
plt.xlabel("Total Steps (T * Batch Size)")
plt.ylabel("Bias")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
for batch in batches:
    X = [T*batch for T in T_values]  # Scale T by batch size for plotting
    Y = [variances[batch][T] for T in T_values]
    plt.plot(X, Y, label=f'Variance (Batch={batch})', marker='o')
plt.xlabel("Total Steps (T * Batch Size)")
plt.ylabel("Variance")
plt.title(f"Slock Biases and Variances for Different Batch Sizes; alpha={alpha}. optiized eta_star for each T")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_batch={batches}_alpha={alpha}.pdf")
plt.show()
# %%


#Risk
plt.figure(figsize=(12, 8))
for batch in batches:
    X = [T*batch for T in T_values]  # Scale T by batch size for plotting
    Y = [biases[batch][T] + variances[batch][T] for T in T_values]
    plt.plot(X, Y, label=f'Risk (Batch={batch})', marker='o')
plt.xlabel("Total Steps (T * Batch Size)")
plt.ylabel("Risk")
plt.title(f"Slock Risk for Different Batch Sizes; alpha={alpha}. optiized eta_star for each T")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_risk_batch={batches}_alpha={alpha}.pdf")
plt.show()
# %%

# b*T FIXED

bT = 200000

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
list_alphas = [1.01, 1.1, 1.5, 2.5, 10]

eta = 0.01
slock_linear = SlockLinear(model, x0, beta=beta, T_max=max(T_values), optimize=optimize, base_lr=eta)
mode = Mode.SLOCK
#%%
batches = [1, 5, 10, 20, 50, 100]
biases, variances = {}, {}
for batch in batches:
    biases[batch], variances[batch] = {}, {}
    for alpha in list_alphas:
        T = bT // batch  # Adjust T to keep b*T constant
        print(f"Computing for alpha={alpha}, batch size {batch} and T={T} (b*T={bT})...")
        slock_linear._update_model_for_alpha(alpha)  # Update model for the specific alpha
        eta_star = (
            slock_linear.compute_best_slock_eta(T, m_constant=Delta)
            * batch_factor_linear(batch, alpha, beta)
        )
        slock_linear._setup_for_T(T, base_lr=eta_star)  # Update schedule for new T
        bias, variance = slock_linear.compute_slock_biases_and_variances([T], batch=batch, K=1)
        biases[batch][alpha] = bias[T]
        variances[batch][alpha] = variance[T]

#%%

#PlOT bias 

plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    X = [batch for batch in batches]
    Y = [biases[batch][alpha] for batch in batches]
    plt.plot(X, Y, label=f'Bias (alpha={alpha})', marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Bias")
plt.title(f"Slock Bias for Different Batch Sizes; b*T={bT}, eta_star")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_bias_bT={bT}_eta_star_for_each_batch.pdf")
plt.show()
# %%
#Plot variance
plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    X = [batch for batch in batches]
    Y = [variances[batch][alpha] for batch in batches]
    plt.plot(X, Y, label=f'Variance (alpha={alpha})', marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Variance")
plt.title(f"Slock Variance for Different Batch Sizes; b*T={bT}, eta_star")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_variance_bT={bT}_eta_star_for_each_batch.pdf")
plt.show()
# %%
#Plot risk
plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    X = [batch for batch in batches]
    Y = [biases[batch][alpha] + variances[batch][alpha] for batch in batches]
    plt.plot(X, Y, label=f'Risk (alpha={alpha})', marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Risk")
plt.title(f"Slock Risk for Different Batch Sizes; b*T={bT}, eta_star")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_risk_bT={bT}_eta_star_for_each_batch.pdf")
plt.show()
# %% APPROX
list_alphas = [1.4, 1.8, 2.2, 2.6]
batches = np.logspace(0, 4, 20, dtype=int)  # Batch sizes from 1 to 10000
approx_biases, approx_variances = {}, {}
true_biases, true_variances = {}, {}
for batch in batches:
    approx_biases[batch], approx_variances[batch] = {}, {}
    true_biases[batch], true_variances[batch] = {}, {}
    for alpha in list_alphas:
        T = bT // batch  # Adjust T to keep b*T constant
        print(f"Computing for alpha={alpha}, batch size {batch} and T={T} (b*T={bT})...")
        slock_linear._update_model_for_alpha(alpha)  # Update model for the specific alpha
        eta_star = (
            slock_linear.compute_best_slock_eta(T, m_constant=Delta)
            * batch_factor_linear(batch, alpha, beta)
        )
        slock_linear._setup_for_T(T, base_lr=eta_star)  # Update schedule for new T
        bias = slock_linear.compute_slock_approx_bias(T, T, m_constant=Delta, m_exponent=beta, batch=batch)
        variance = slock_linear.compute_slock_approx_variance(T, T, batch=batch)
        approx_biases[batch][alpha] = bias
        approx_variances[batch][alpha] = variance
        true_bias, true_variance = slock_linear.compute_slock_biases_and_variances([T], batch=batch)
        true_biases[batch][alpha], true_variances[batch][alpha] = true_bias[T], true_variance[T]
#%%
batches = np.logspace(0, 4, 20, dtype=int)  # Batch sizes from 1 to 10000
plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    X = [batch for batch in batches]
    Y = [approx_biases[batch][alpha] for batch in batches]
    Y_true = [true_biases[batch][alpha] for batch in batches]
    plt.plot(X, Y, label=f'Approx Bias (alpha={alpha})', marker='o')
    plt.plot(X, Y_true, label=f'True Bias (alpha={alpha})', marker='x', linestyle='--')
plt.xlabel("Batch Size")
plt.ylabel("Approx Bias")
plt.title(f"Slock Approx Bias for Different Batch Sizes; b*T={bT}, eta_star")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_approx_bias_bT={bT}_eta_star_for_each_batch.pdf")
plt.show()

plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    X = [batch for batch in batches]
    Y = [approx_variances[batch][alpha] for batch in batches]
    Y_true = [true_variances[batch][alpha] for batch in batches]
    plt.plot(X, Y, label=f'Approx Variance (alpha={alpha})', marker='o')
    plt.plot(X, Y_true, label=f'True Variance (alpha={alpha})', marker='x', linestyle='--')
plt.xlabel("Batch Size")
plt.ylabel("Approx Variance")
plt.title(f"Slock Approx Variance for Different Batch Sizes; b*T={bT}, eta_star")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_approx_variance_bT={bT}_eta_star_for_each_batch.pdf")
plt.show()

plt.figure(figsize=(12, 8))
for alpha in list_alphas:
    X = [batch for batch in batches]
    Y = [approx_biases[batch][alpha] + approx_variances[batch][alpha] for batch in batches]
    Y_true = [true_biases[batch][alpha] + true_variances[batch][alpha] for batch in batches]
    plt.plot(X, Y, label=f'Approx Risk (alpha={alpha})', marker='o')
    plt.plot(X, Y_true, label=f'True Risk (alpha={alpha})', marker='x', linestyle='--')
plt.xlabel("Batch Size")
plt.ylabel("Approx Risk")
plt.title(f"Slock Approx Risk for Different Batch Sizes; b*T={bT}, eta_star")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(f"images/slock/BATCH_comparison_approx_risk_bT={bT}_eta_star_for_each_batch.pdf")
plt.show()  
# %%
true_biases_to_save = {float(batch): true_biases[batch] for batch in true_biases}
save_dict_to_json(true_biases_to_save, f"slock_linear_dim={dim}", f"BATCH_comparison_bias_bT={bT}_eta_star_for_each_batch.json")
true_variances_to_save = {float(batch): true_variances[batch] for batch in true_variances}
save_dict_to_json(true_variances_to_save, f"slock_linear_dim={dim}", f"BATCH_comparison_variance_bT={bT}_eta_star_for_each_batch.json")
#%%
#%%
bmax = {}
batches_list_max = np.linspace(1, 10000, dtype=int)
for alpha in list_alphas:
    slock_linear._update_model_for_alpha(alpha)  # Update model for the specific alpha
    for batch in batches_list_max:
        T = bT // batch  # Adjust T to keep b*T constant
        bmax[batch] = slock_linear.compute_exact_critical_batch(T, m_constant=Delta)
        if bmax[batch] < batch:
            print(f"Warning: For alpha={alpha}, batch={batch}, T={T}, the critical batch size is {bmax[batch]}, which is less than the current batch size.")
            break


#%%
###
#COMPARE BEST ETAS vs NUMERICAL OPTIMIZATION
alphas = [1+1e-6, 1.001, 1.01, 1.1, 1.5, 2.5, 10, 20 , 50]
best_etas = {}
exact_etas = {}
T= 100000
for alpha in alphas:
    slock_linear._update_model_for_alpha(alpha)  # Update model for the specific alpha
    slock_linear._setup_for_T(T, base_lr=0.001)  # Setup for T with a placeholder learning rate
    best_etas[alpha] = slock_linear.compute_best_slock_eta(T, m_constant=Delta)
    exact_etas[alpha] = slock_linear.compute_exact_eta(T, m_constant=Delta)
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


#STEPS TO RISK


Tmax = 50000
alphas = [1.4, 1.8, 2.2, 2.6]


RISK_TO_HAVE = 1e-5


#%%
batches = np.logspace(2.5, 5, 40, dtype=int)  # Batch sizes from 1 to 1000
Times = np.logspace(2, 5, 200, dtype=int)  # From 10 to 100000
steps_to_risk = {}
for alpha in alphas:
    slock_linear._update_model_for_alpha(alpha)  # Update model for the specific alpha
    steps_to_risk[alpha] = {}
    for batch in batches:
        print(f"Computing for alpha={alpha}, batch size {batch}...")
        #eta_star = batch_factor_linear(batch, alpha, beta)*slock_linear.compute_best_slock_eta(Tmax, m_constant=Delta)
        for T in Times:
            eta_star = batch_factor_linear(batch, alpha, beta)*slock_linear.compute_best_slock_eta(T, m_constant=Delta)
            print(f"Computing for alpha={alpha}, batch size {batch} and T={T}...")
            slock_linear._setup_for_T(T, base_lr=eta_star)  # Update schedule for new T
            bias, variance = slock_linear.compute_slock_biases_and_variances([T], batch=batch)
            total_risk = bias[T] + variance[T]
            print(total_risk)
            if total_risk <= RISK_TO_HAVE:
                steps_to_risk[alpha][batch] = T
                break
        if batch not in steps_to_risk[alpha]:
            steps_to_risk[alpha][batch] = Tmax  # If not found, set to Tmax

#%%
# Compute b_crits
b_crits = {}
for alpha in alphas:
    T_values = steps_to_risk[alpha].values()
    slock_linear._update_model_for_alpha(alpha)  # Update model for the specific alpha
    b_crits[alpha] = {}
    for T in T_values:
        b_crits[alpha][T] = slock_linear.compute_exact_critical_batch(T, m_constant=Delta)

#%%
plt.figure(figsize=(12, 8))
for alpha in alphas:
    color = plt.cm.viridis(alphas.index(alpha) / len(alphas))
    plt.plot(list(steps_to_risk[alpha].keys()), list(steps_to_risk[alpha].values()), marker='o', label =f"alpha={alpha}", color=color)
    plt.plot(list(b_crits[alpha].values()), list(b_crits[alpha].keys()), linestyle="--", marker='.', label=f"alpha={alpha} (critical batch)", color=color)
    plt.xlabel("Batch Size")
    plt.ylabel(f"Steps to Achieve Risk <= {RISK_TO_HAVE}")
    plt.title(f"Steps to Achieve Risk <= {RISK_TO_HAVE} for Different Batch Sizes; alpha={alpha}, Tmax={Tmax}")
    plt.grid()
plt.xlim(200, 100000)
plt.ylim(00, 10000)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
plt.show()


#%%
b_crits = {alpha: {float(b_crits[alpha][T]): int(T) for T in steps_to_risk[alpha].values()} for alpha in alphas}
steps_to_risk = {alpha: {int(batch): float(steps_to_risk[alpha][batch]) for batch in steps_to_risk[alpha]} for alpha in alphas}
# %%
save_dict_to_json(steps_to_risk, f"slock_linear_dim={dim}", f"steps_to_risk.json")
save_dict_to_json(b_crits, f"slock_linear_dim={dim}", f"critical_batches.json")
# %%
