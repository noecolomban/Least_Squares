#%% CONSTANT SCHEDULE
from turtle import color
from unittest import result

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    compute_different_sigmas,
)

# %%
dim = 100
sigma = 0.1
exponent = 2
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)


T = 10000

laplace_analysis = LaplaceConstant(model, x0, T)

#print("Computing True approximation...")
#laplace_result = laplace_analysis.compute_true_approx_risk()
#print("True approximation computed = ", laplace_result)
# %%

#print("Computing Laplace risk approximation...")
#real_risk_approx = laplace_analysis.compute_lagrange_approx_risk(m_constant=Delta, m_exponent=beta)
#print("Laplace risk approximation computed = ", real_risk_approx)
#%%

results_laplace = laplace_analysis.compute_laplace_for_several_ts(
    T, Delta, beta,
    step=100)

results_real_approx = laplace_analysis.compute_real_approx_for_several_ts(
    T,
    step=100)
#%%

plt.plot(list(results_laplace.keys()), list(results_laplace.values()), label="Laplace approximation")
plt.plot(list(results_real_approx.keys()), list(results_real_approx.values()), label="True approximation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.title(f"Constant schedule: \n Laplace risk approximation vs True risk approximation for different T, sigma={sigma}, beta={beta}")
plt.legend()
plt.grid()
plt.savefig("images/laplace_vs_true_approximation_T=100000.pdf")
plt.show()
# %%

results_different_sigmas = compute_different_sigmas(10000, model, x0, Delta, beta, sigmas=[0., 0.1, 0.5, 1.])
risk_dict, real_approx_dict = results_different_sigmas


#%%
for sigma in risk_dict.keys():
    color = np.random.rand(3,)  # Random color for each sigma
    plt.plot(list(risk_dict[sigma].keys()), list(risk_dict[sigma].values()), label=f"Laplace approximation, sigma={sigma}", color=color)
    plt.plot(list(real_approx_dict[sigma].keys()), list(real_approx_dict[sigma].values()), label=f"True approximation, sigma={sigma}", linestyle="dashed", color=color)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.title(rf"Laplace risk approximation vs Diagonal approximation for different T and $\sigma$"+"\n"+rf"$\beta$={beta}")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_true_approximation_different_sigmas_T={T}.pdf")
plt.show()

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

# %%
dim = 100
sigma = 0.1
exponent = 2 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
K = 1
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000,100000, 200000]
# %%
new_linear_laplace_analysis = LaplaceLinear(model, x0, T_max=max(T_values), optimize=optimize, base_lr=0.001)

#%%
bias, variance = new_linear_laplace_analysis.compute_laplace_approx_biases_and_variances_different_finals(
    T_values=T_values,
    m_exponent=beta,
    m_constant=Delta,
    K=K
)
diagonal_biases, diagonal_variances = new_linear_laplace_analysis.compute_true_approx_biases_and_variances(T_values=T_values, K=K)

true_biases, true_variances = new_linear_laplace_analysis.compute_true_biases_and_variances(T_values=T_values, K=K)

#%%
plt.figure(figsize=(12, 6))
plt.plot(T_values, bias.values(), label="Laplace Bias", marker='o')
plt.plot(T_values, np.array(list(diagonal_biases.values())), label="Diagonal Bias", marker='o')
plt.plot(T_values, np.array(list(true_biases.values())), label="True Bias", marker='o')
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
plt.plot(T_values, np.array(list(true_variances.values())), label="True Variance", marker='o')
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
true_risk = {T: true_biases[T] + true_variances[T] for T in T_values}
plt.plot(T_values, laplace_risk.values(), label="Laplace Risk", marker='o')
plt.plot(T_values, diagonal_risk.values(), label="Diagonal Risk", marker='o')
plt.plot(T_values, true_risk.values(), label="True Risk", marker='o')
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
plt.plot(T_values, list(variance_ratios.values()), label="Variance Ratio (Laplace/Diagonal)", marker='o')
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
T_values = [10000, 30000, 100000, 200000]  # Different T values to compare
colors = ['blue', 'orange', 'green', 'red']  # Colors for different T values

alpha_var, diagonal_var = {}, {}
for T in T_values:
    alpha_var[T], diagonal_var[T] = new_linear_laplace_analysis.compare_different_alphas_variance(T=T, list_alphas=list_alphas, m_constant=Delta, K=K)
# %%
# Plotting the variance for different alphas and T values
from scipy.special import zeta
a = np.array(list_alphas)
constants = zeta(a/2)/(2/(a-2))

plt.figure(figsize=(12, 6))
for i, T in enumerate(T_values):
    color = colors[i]  # Use predefined color for each T
    plt.plot(list_alphas, list(alpha_var[T].values()), label=f"Laplace Variance T={T}", marker='o', color=color)
    plt.plot(list_alphas, np.array(list(diagonal_var[T].values())), label=f"Diagonal Variance T={T}", marker='o', color=color, linestyle="dashed")
plt.title(f"Variance of Laplace approximation vs Diagonal approximation for linear schedule at K={K} \n for different alpha values, beta={beta}, Delta={Delta}")
#plt.yscale("log")
plt.xlabel("Alpha")
plt.ylabel("Variance approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_variance_linear_schedule_multipleT_K={K}_different_alphas.pdf")
plt.show()
# %%
plt.figure(figsize=(12, 6))
for i, T in enumerate(T_values):
    color = colors[i]  # Use predefined color for each T
    ratio = {alpha: alpha_var[T][alpha]/diagonal_var[T][alpha] for alpha in list_alphas}
    plt.plot(list_alphas, list(ratio.values()), label=f"Variance Ratio (Laplace/Diagonal) T={T}", marker='o', color=color)
plt.title(f"Variance Ratio of Laplace approximation to Diagonal approximation for linear schedule at K={K} \n for different alpha values, beta={beta}, Delta={Delta}")
plt.xlabel("Alpha")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_ratio_laplace_diagonal_linear_schedule_multipleT_K={K}_different_alphas.pdf")
plt.show()
# %%
plt.figure(figsize=(12, 6))
plt.plot(list_alphas, constants, label="Theoretical constant (zeta(a/2)*a/(a-2))", marker='o', linestyle="dashed")
# %%


#COMPARE DIFFERENT ALPHAS FOR VARIANCE TRAJECTORIES
list_alphas = np.linspace(1.1, 3.0, 5)
T_values = [1000, 5000, 10000, 20000, 30000, 50000, 100000, 200000]
all_laplace_variances, all_diagonal_variances = new_linear_laplace_analysis.compare_variance_trajectories_different_alphas(T_values=T_values, list_alphas=list_alphas, m_constant=Delta, K=K)


#%%
plt.figure(figsize=(12, 6))
for i, alpha in enumerate(list_alphas):
    color = (i / len(list_alphas), 1 - i / len(list_alphas), np.random.rand())  # Color for each alpha
    laplace_variance = [all_laplace_variances[(alpha, T)] for T in T_values]
    diagonal_variance = [all_diagonal_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, laplace_variance, label=f"Laplace Variance trajectory for alpha={alpha:.2f}", marker='o', color=color)
    plt.plot(T_values, diagonal_variance, label=f"Diagonal Variance trajectory for alpha={alpha:.2f}", marker='o', linestyle="dashed", color=color)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Variance")
plt.title("Variance Trajectories for Different Alphas")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_trajectories_different_alphas_linear_schedule_K={K}.pdf")
plt.show()
# %%
