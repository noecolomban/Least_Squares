#%% LINEAR SCHEDULE
# CORRECTED !!

from unittest import result

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    LaplaceConstant, 
    LaplaceLinear,
    compute_different_sigmas,

)

from src.utils import save_dict_to_json, read_dict_from_json, constant_zeta_correction

# %%
dim = 1000
sigma = 0.1
exponent = 3 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 1.2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
K = 1
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
# %%
new_linear_laplace_analysis = LaplaceLinear(model, x0, T_max=max(T_values), optimize=optimize, base_lr=0.001)

#%%
bias, variance = new_linear_laplace_analysis.compute_laplace_approx_biases_and_variances_different_finals(
    T_values=T_values,
    m_exponent=beta,
    m_constant=Delta,
    K=K
)
print("Diagonal approximation...")
diagonal_biases, diagonal_variances = new_linear_laplace_analysis.compute_true_approx_biases_and_variances(T_values=T_values, K=K)
# print("Laplace biases and variances...")
# true_biases, true_variances = new_linear_laplace_analysis.compute_true_biases_and_variances(T_values=T_values, K=K)

#%%
save_dict_to_json(diagonal_variances, folder="laplace_linear", filename=f"diagonal_variances_K={K}.json")
# save_dict_to_json(true_variances, folder="laplace_linear", filename=f"laplace_variances_K={K}.json")
save_dict_to_json(diagonal_biases, folder="laplace_linear", filename=f"diagonal_biases_K={K}.json")
# save_dict_to_json(true_biases, folder="laplace_linear", filename=f"true_biases_K={K}.json")


#%%
plt.figure(figsize=(12, 6))
plt.plot(T_values, bias.values(), label="Laplace Bias", marker='o')
plt.plot(T_values, np.array(list(diagonal_biases.values())), label="Diagonal Bias", marker='o')
#plt.plot(T_values, np.array(list(true_biases.values())), label="True Bias", marker='o')
plt.title(f"Bias components of Laplace approximation vs Diagonal approximation for linear schedule \n alpha={exponent}, beta={beta}, Tmax={max(T_values)}, K=t/T={K}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_bias_linear_schedule_T={max(T_values)}_sigma={sigma}_K={K}_different_finals.pdf")
plt.show()
# %%
# Plotting the variance components
plt.figure(figsize=(12, 6))
plt.plot(T_values, np.array(list(variance.values())), label="Laplace Variance", marker='o')
plt.plot(T_values, np.array(list(diagonal_variances.values())), label="Diagonal Variance", marker='o')
#plt.plot(T_values, np.array(list(true_variances.values())), label="True Variance", marker='o')
#plt.plot(T_values, c*0.002/np.array(T_values)**0.5, label="1/sqrt T", linestyle="dashed", marker='o')
plt.title(f"Variance components of Laplace approximation vs Diagonal approximation for linear schedule \n alpha={exponent}, beta={beta}, Tmax={max(T_values)}, K=t/T={K}")
plt.xscale("log")
plt.yscale("log")
#plt.ylim(2e-7, 1e-6)
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_variance_linear_schedule_T={max(T_values)}_sigma={sigma}_K={K}_different_finals.pdf")
plt.show()
# %%
plt.figure(figsize=(12, 6))
laplace_risk = {T: bias[T] + variance[T] for T in T_values}
diagonal_risk = {T: diagonal_biases[T] + diagonal_variances[T] for T in T_values}
# true_risk = {T: true_biases[T] + true_variances[T] for T in T_values}
plt.plot(T_values, laplace_risk.values(), label="Laplace Risk", marker='o')
plt.plot(T_values, diagonal_risk.values(), label="Diagonal Risk", marker='o')
# plt.plot(T_values, true_risk.values(), label="True Risk", marker='o')
plt.title(f"Total risk of Laplace approximation vs Diagonal approximation for linear schedule \n alpha={exponent}, beta={beta}, Tmax={max(T_values)}, K=t/T={K}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_total_risk_linear_schedule_T={max(T_values)}_sigma={sigma}_K={K}_different_finals.pdf")
plt.show()

#%%
#VARIANCE Ratio


variance_ratios = {T: variance[T]/diagonal_variances[T] for T in T_values}
plt.figure(figsize=(12, 6))
plt.plot(T_values, np.array(list(variance_ratios.values())), label="Variance Ratio (Laplace/Diagonal)", marker='o')
#plt.plot(T_values, np.array(list(variance_ratios.values()))*(2*np.euler_gamma), label="Variance Ratio (Laplace/Diagonal)", marker='o')
plt.title(f"Variance Ratio of Laplace approximation to Diagonal approximation for linear schedule \n alpha={exponent}, beta={beta}, Tmax={max(T_values)}, K=t/T={K}")
plt.xscale("log")
plt.ylim(0, 1)  # Assuming the ratio is less than 1, adjust as needed
#plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_ratio_laplace_diagonal_linear_schedule_T={max(T_values)}_sigma={sigma}_K={K}_different_finals.pdf")
plt.show()
# %%
#COMPARING DIFFERENT ALPHAS

list_alphas = np.linspace(1.1, 3.0, 15)  # Example alpha values to compare
T_values = [10000, 30000, 100000]  # Different T values to compare
colors = ['blue', 'orange', 'green', 'red']  # Colors for different T values

alpha_var, diagonal_var = {}, {}
for T in T_values:
    alpha_var[T], diagonal_var[T] = new_linear_laplace_analysis.compare_different_alphas_variance(T=T, list_alphas=list_alphas, m_constant=Delta, K=K)
# %%
# Plotting the variance for different alphas and T values
from scipy.special import zeta
a = np.array(list_alphas)
constants = np.array([zeta(alpha/2)/(1/(alpha/2-1)) if alpha/2 > 1 else 1 for alpha in a])

plt.figure(figsize=(12, 6))
for i, T in enumerate(T_values):
    color = colors[i]  # Use predefined color for each T
    plt.plot(list_alphas, np.array(list(alpha_var[T].values())), label=f"Laplace Variance T={T}", marker='o', color=color)
    plt.plot(list_alphas, np.array(list(diagonal_var[T].values())), label=f"Diagonal Variance T={T}", marker='o', color=color, linestyle="dashed")
plt.title(f"Variance of Laplace approximation vs Diagonal approximation for linear schedule at K={K} \n for different alpha values, beta={beta}, Delta={Delta}, dim={dim}")
#plt.yscale("log")
plt.xlabel("Alpha")
plt.ylabel("Variance approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_variance_linear_schedule_multipleT_K={K}_different_alphas_dim={dim}.pdf")
plt.show()
# %%
plt.figure(figsize=(12, 6))
for i, T in enumerate(T_values):
    color = colors[i]  # Use predefined color for each T
    ratio = {alpha: alpha_var[T][alpha]/diagonal_var[T][alpha] for alpha in list_alphas}
    plt.plot(list_alphaplt.xlabel("T")
s, np.array(list(ratio.values())), label=f"Variance Ratio (Laplace/Diagonal) T={T}", marker='o', color=color)
plt.title(f"Variance Ratio of Laplace approximation to Diagonal approximation for linear schedule at K={K} \n for different alpha values, beta={beta}, Delta={Delta}, dim={dim}")
plt.xlabel("Alpha")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_ratio_laplace_diagonal_linear_schedule_multipleT_K={K}_different_alphas_dim={dim}.pdf")
plt.show()

plt.figure(figsize=(12, 6))
for i, T in enumerate(T_values):
    color = colors[i]  # Use predefined color for each T
    ratio = {alpha: alpha_var[T][alpha]/diagonal_var[T][alpha] for alpha in list_alphas}
    plt.plot(list_alphas, np.array(list(ratio.values()))*constants, label=f"Variance Ratio (Laplace/Diagonal) T={T}", marker='o', color=color)
plt.title(f"Corrected Variance Ratio of Laplace approximation to Diagonal approximation for linear schedule at K={K} \n for different alpha values, beta={beta}, Delta={Delta}, dim={dim}")
plt.xlabel("Alpha")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid()
plt.savefig(f"images/corrected_variance_ratio_laplace_diagonal_linear_schedule_multipleT_K={K}_different_alphas_dim={dim}.pdf")
plt.show()
# %%
save_dict_to_json(alpha_var, folder="laplace_linear", filename=f"alpha_var_dim={dim}.json")
save_dict_to_json(diagonal_var, folder="laplace_linear", filename=f"diagonal_var_dim={dim}.json")
# %%



#COMPARE DIFFERENT ALPHAS FOR VARIANCE TRAJECTORIES
list_alphas = [1.1, 1.3, 1.7]  # Example alpha values to compare
#list_alphas = [2]  # Example alpha values to compare
#list_alphas = [2.2, 2.8, 3.5,]  # Example alpha values to compare
#dim_text = f"dim=1000"  # Example dimension text for plot titles and filenames
dim_text = f"T**0.7"  # Example dimension text for plot titles and filenames when dimension changes with T

T_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]  # Different T values to compare
#changing_dim = lambda T: 500  # Example of changing dimension with T, adjust as needed
changing_dim = lambda T: T**(0.7)  # Example of changing dimension with T, adjust as needed
all_laplace_variances, all_diagonal_variances = new_linear_laplace_analysis.compare_variance_trajectories_different_alphas(T_values=T_values, list_alphas=list_alphas, changing_dim=changing_dim, K=K)  # Example of changing dimension with T, adjust as needed

#%%
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Colors for different alpha values

plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = colors[i]  # Use predefined color for each alpha
    laplace_variance = [all_laplace_variances[(alpha, T)] for T in T_values]
    diagonal_variance = [all_diagonal_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, laplace_variance, label=f"Laplace Variance trajectory for alpha={alpha:.2f}", marker='o', color=color)
    plt.plot(T_values, diagonal_variance, label=f"Diagonal Variance trajectory for alpha={alpha:.2f}", marker='o', linestyle="dashed", color=color)
plt.xscale("log")
#plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Variance")
plt.title(f"Variance Trajectories for Different Alphas, {dim_text}")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_trajectories_different_alphas_linear_schedule_alpha_max={max(list_alphas)}_{dim_text}.pdf")
plt.show()

plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = colors[i]  # Use predefined color for each alpha
    laplace_variance = [all_laplace_variances[(alpha, T)]*constant_zeta_correction(alpha) for T in T_values]
    diagonal_variance = [all_diagonal_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, laplace_variance, label=f"Corrected Laplace Variance for alpha={alpha:.2f}", marker='o', color=color)
    plt.plot(T_values, diagonal_variance, label=f"Diagonal Variance trajectory for alpha={alpha:.2f}", marker='o', linestyle="dashed", color=color)
plt.xscale("log")
#plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Variance")
plt.title(f"Corrected Variance Trajectories for Different Alphas, {dim_text}")
plt.legend()
plt.grid()
plt.savefig(f"images/corrected_variance_trajectories_different_alphas_linear_schedule_alpha_max={max(list_alphas)}_{dim_text}_Tmax={max(T_values)}.pdf")
plt.show()

# %%
#RATIOS

plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = colors[i]  # Use predefined color for each alpha
    ratio = [all_laplace_variances[(alpha, T)]/all_diagonal_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, ratio, label=f"Variance Ratio (Laplace/Diagonal) for alpha={alpha:.2f}", marker='o', color=color)
plt.xscale("log")
plt.xlabel("T")
plt.ylabel("Variance Ratio")
plt.title(f"Variance Ratio Trajectories for Different Alphas, dim={dim_text}")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_ratio_trajectories_different_alphas_linear_schedule_alpha_max={max(list_alphas)}_{dim_text}_Tmax={max(T_values)}.pdf")
plt.show()

#corrected ratio
plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = colors[i]  # Use predefined color for each alpha
    ratio = [all_laplace_variances[(alpha, T)]*(constant_zeta_correction(alpha))/all_diagonal_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, ratio, label=f"Corrected Variance Ratio (Laplace/Diagonal) for alpha={alpha:.2f}", marker='o', color=color)

plt.xscale("log")
plt.xlabel("T")

plt.ylim(0.7, 1)
plt.yscale("log")
plt.xlim(10000, max(T_values))

plt.ylabel("Corrected Variance Ratio")
plt.title(f"Corrected Variance Ratio Trajectories for Different Alphas, dim={dim_text}")
plt.legend()
plt.grid()
plt.savefig(f"images/corrected_variance_ratio_trajectories_different_alphas_linear_schedule_alpha_max={max(list_alphas)}_{dim_text}_Tmax={max(T_values)}.pdf")
plt.show()

# %%
all_laplace_variances_str = {str(key): value for key, value in all_laplace_variances.items()}
save_dict_to_json(all_laplace_variances_str, folder="laplace_linear", filename=f"LINEAR_all_laplace_variances_{dim_text}_Tmax={max(T_values)}.json")
all_diagonal_variances_str = {str(key): value for key, value in all_diagonal_variances.items()}
save_dict_to_json(all_diagonal_variances_str, folder="laplace_linear", filename=f"LINEAR_all_diagonal_variances_{dim_text}_Tmax={max(T_values)}.json")

# %%
