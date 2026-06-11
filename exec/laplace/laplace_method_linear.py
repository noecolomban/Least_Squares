#%% LINEAR SCHEDULE
# CORRECTED !!


import numpy as np


import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    LaplaceConstant, 
    LaplaceLinear,
    compute_different_sigmas,
    Mode
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
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
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



#COMPARE DIFFERENT ALPHAS FOR VARIANCE TRAJECTORIES
list_alphas = [1.1, 1.3, 1.7]  # Example alpha values to compare
#list_alphas = [2.2, 2.8, 3.5,]  # Example alpha values to compare
#list_alphas = [1.1, 1.3, 1.7, 2.2, 2.8, 3.5]  # Example alpha values to compare

dim_text = f"Ts50**1balpha+0.2"  # Example dimension text for plot titles and filenames when dimension changes with T
#dim_text = ""

T_values = [100, 1000, 2000, 5000, 10000, 20000, 50000]  # Different T values to compare

changing_dim = lambda T, alpha: (T/50)**(1/(alpha+0.2))  # Example function for changing dimension with T and alpha, adjust as needed
#changing_dim = lambda T, alpha: 100  # Example function for changing dimension with T and alpha, adjust as needed

mode = Mode.TRUE  # Choose between Mode.TRUE for exact variance and Mode.DIAGONAL for diagonal approximation

#%%
all_laplace_variances, all_diagonal_variances = new_linear_laplace_analysis.compare_variance_trajectories_different_alphas(
    T_values=T_values, 
    list_alphas=list_alphas, 
    changing_dim=changing_dim,
    K=K,
    mode=mode
  )  
    #Mode.DIAGONAL:  diagonal approximation
    #Mode.TRUE: exact variance

#%%
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Colors for different alpha values

plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = colors[i]  # Use predefined color for each alpha
    laplace_variance = [all_laplace_variances[(alpha, T)] for T in T_values]
    diagonal_variance = [all_diagonal_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, laplace_variance, label=f"Laplace Variance trajectory for alpha={alpha:.2f}", marker='o', color=color)
    plt.plot(T_values, diagonal_variance, label=f"{mode.name} Variance trajectory for alpha={alpha:.2f}", marker='o', linestyle="dashed", color=color)
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
    laplace_variance = [constant_zeta_correction(alpha, d=changing_dim(T, alpha)) *all_laplace_variances[(alpha, T)]  for T in T_values]
    diagonal_variance = [all_diagonal_variances[(alpha, T)]  for T in T_values]
    plt.plot(T_values, laplace_variance, label=f"Corrected Laplace Variance for alpha={alpha:.2f}", marker='o', color=color)
    plt.plot(T_values, diagonal_variance, label=f"{mode.name} Variance trajectory for alpha={alpha:.2f}", marker='o', linestyle="dashed", color=color)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Variance")
plt.title(f"Corrected Variance Trajectories for Different Alphas, {dim_text}")
plt.legend()
plt.grid()
plt.savefig(f"images/corrected_variance_trajectories_different_alphas_linear_schedule_alpha_max={max(list_alphas)}_{dim_text}_Tmax={max(T_values)}.pdf")
plt.show()


# %%
#RATIOS
from scipy.special import zeta

plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = colors[i]  # Use predefined color for each alpha
    ratio = [ all_laplace_variances[(alpha, T)]/all_diagonal_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, ratio, label=f"Variance Ratio (Laplace/{mode.name}) for alpha={alpha:.2f}", marker='o', color=color)

    #plt.plot(T_values, [1/alpha for _ in T_values], color=color, label=f"zeta(alpha/2) for alpha={alpha:.2f}", linestyle="dashed")

plt.xscale("log")
plt.xlim(1000, max(T_values))  # Assuming the ratio is less than 1, adjust as needed
#plt.yscale("log")
plt.ylim(0, 1.3)  # Assuming the ratio is less than 1, adjust as needed

plt.xlabel("T")
plt.ylabel("Variance Ratio")
plt.title(f"Variance Ratio Trajectories for Different Alphas, dim={dim_text}")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_ratio_trajectories_different_alphas_linear_schedule_alpha_max={max(list_alphas)}_{dim_text}_Tmax={max(T_values)}.pdf")
plt.show()


#corrected ratio
#%%

plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = colors[i]  # Use predefined color for each alpha
    ratio = [constant_zeta_correction(alpha, d=changing_dim(T, alpha)) * all_laplace_variances[(alpha, T)]/all_diagonal_variances[(alpha, T)] for T in T_values]
    #if some are non definded, replace them by 0.0
    plt.plot(T_values, ratio, label=f"Corrected Variance Ratio (Laplace/{mode.name}) for alpha={alpha:.2f}", marker='o', color=color)

plt.xscale("log")
plt.xlabel("T")
plt.xlim(1000, max(T_values))  # Assuming the ratio is less than 1, adjust as needed
plt.ylim(0, 1.3)  # Assuming the ratio is less than 1, adjust as needed

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
###########

T_values = [10, 100, 500, 1000, 5000, 10000, 20000]
list_alphas = [1.1, 1.5]

laplace_bias_alpha, diagonal_bias_alpha, laplace_variance_alpha, diagonal_variance_alpha = (
    new_linear_laplace_analysis.compare_different_alphas(
        T_values=T_values, 
        list_alphas=list_alphas,
        m_constant=Delta,
        m_exponent=beta,
        K=1,
        changing_dim=changing_dim,
        mode=Mode.DIAGONAL
    )
)

#%%
ratios_bias = {key: laplace_bias_alpha[key] / diagonal_bias_alpha[key] if diagonal_bias_alpha[key] != 0 else 0 for key in laplace_bias_alpha.keys()}
ratios_variance = {key: laplace_variance_alpha[key] / diagonal_variance_alpha[key] if diagonal_variance_alpha[key] != 0 else 0 for key in laplace_variance_alpha.keys()}
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y_bias = np.array([ratios_bias[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y_bias, label=f"Bias Ratio (alpha={alpha})", marker='o', color=color)
plt.xscale('log')
#plt.yscale('log')
plt.xlim(100, max(T_values))
plt.ylim(0, 2)
plt.xlabel("T (log scale)")
plt.ylabel("Bias Ratio (Laplace Bias / Diagonal Bias)")
plt.title("Bias Ratio Trajectories for Different Alphas (Laplace vs Diagonal) for Linear schedule")
plt.legend()
plt.grid()
plt.savefig(f"images/LINEAR_bias_ratio_trajectories_comparison_dim={dim}_Tmax={max(T_values)}.pdf")
plt.show()

for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y_variance = np.array([ratios_variance[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y_variance, label=f"Variance Ratio (alpha={alpha})", marker='o', color=color)
plt.xscale('log')
plt.xlim(100, max(T_values))  
plt.ylim(0, 2)      
plt.xlabel("T (log scale)")
plt.ylabel("Variance Ratio (Laplace Variance / Diagonal Variance)")
plt.title("Variance Ratio Trajectories for Different Alphas (Laplace vs Diagonal) for Linear schedule")
plt.legend()
plt.grid()
plt.savefig(f"images/LINEAR_variance_ratio_trajectories_comparison_dim={dim}_Tmax={max(T_values)}.pdf")
plt.show()
# %%
