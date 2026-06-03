#%% WSD SCHEDULE

from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    LaplaceWSD,
    compute_different_sigmas,
    Mode
)
from src.utils import save_dict_to_json

# %%
dim = 1000
sigma = 0.1
exponent = 2
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

base_lr = 0.01
T = 10000
cooldown_len = 0.2
#%%

optimize = False
K = 1
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 40000, 60000, 100000, 125000, 150000]
# %%
wsd_laplace_analysis = LaplaceWSD(model, x0, T_max=max(T_values), optimize=optimize, base_lr=base_lr, cooldown_len=cooldown_len)

#%%
bias, variance = wsd_laplace_analysis.compute_laplace_approx_biases_and_variances_different_finals(
    T_values=T_values,
    m_exponent=beta,
    m_constant=Delta,
    K=K
)
diagonal_biases, diagonal_variances = wsd_laplace_analysis.compute_true_approx_biases_and_variances(T_values=T_values, K=K)

#%%

plt.plot(T_values, bias.values(), label="Laplace Bias", marker='o')
plt.plot(T_values, np.array(list(diagonal_biases.values())), label="Diagonal Bias", marker='o')
plt.title(f"Bias components of Laplace approximation vs Diagonal approximation for wsd schedule \n sigma={sigma}, beta={beta}, Tmax={max(T_values)}, K=t/T={K}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_bias_wsd_schedule_T={max(T_values)}_sigma={sigma}_K={K}_different_finals.pdf")
plt.show()
# %%

plt.plot(T_values, np.array(list(variance.values())), label="Laplace Variance", marker='o')
plt.plot(T_values, np.array(list(diagonal_variances.values())), label="Diagonal Variance", marker='o')
#plt.plot(T_values, c*0.002/np.array(T_values)**0.5, label="1/sqrt T", linestyle="dashed", marker='o')
plt.title(f"Variance components of Laplace approximation vs Diagonal approximation for wsd schedule \n sigma={sigma}, beta={beta}, Tmax={max(T_values)}, K=t/T={K}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_variance_wsd_schedule_T={max(T_values)}_sigma={sigma}_K={K}_different_finals.pdf")
plt.show()

# %%

laplace_risk = {T: bias[T] + variance[T] for T in T_values}
diagonal_risk = {T: diagonal_biases[T] + diagonal_variances[T] for T in T_values}

true_risks = wsd_laplace_analysis.compute_true_risks(T_values=T_values)
#%%

true_risks_ = {T: true_risks[T][-1] for T in T_values}  # Assuming the true risk is the same for all T, we take the last value from the computed risks
#%%
plt.plot(T_values, laplace_risk.values(), label="Laplace Risk", marker='o')
plt.plot(T_values, diagonal_risk.values(), label="Diagonal Risk", marker='o')
plt.plot(T_values, true_risks_.values(), label="True Risk", marker='o', linestyle="dotted")
plt.title(f"Total risk of Laplace approximation vs Diagonal approximation for wsd schedule \n sigma={sigma}, beta={beta}, Tmax={max(T_values)}, K=t/T={K}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_diagonal_total_risk_wsd_schedule_T_vs_true={max(T_values)}_sigma={sigma}_K={K}_different_finals.pdf")
plt.show()
# %%


#COMPARING
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]
list_alphas = [1.1, 1.5, 1.8]

changing_dim = lambda T, alpha: (T)**(1/(alpha+0.2))/2  # Example function for changing dimension with T and alpha, adjust as needed

#%%
laplace_variance_alpha, diagonal_variance_alpha = wsd_laplace_analysis.compare_variance_trajectories_different_alphas(
    T_values=T_values, 
    list_alphas=list_alphas,
    m_constant=Delta,
    K=1,
    changing_dim=changing_dim)
# %%

ratios = {key: laplace_variance_alpha[key] / diagonal_variance_alpha[key] if diagonal_variance_alpha[key] != 0 else 0 for key in laplace_variance_alpha.keys()}

colors = plt.cm.viridis(np.linspace(0, 1, len(list_alphas)))
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y = np.array([ratios[(alpha, T)] for T in T_values])
    plt.plot(T_values, Y, label=f"Variance Ratio (alpha={alpha})", marker='o', color=color)
plt.xscale('log')
#plt.yscale('log')
plt.xlim(100, max(T_values))
plt.ylim(0, 2)
plt.xlabel("T (log scale)")
plt.ylabel("Variance (log scale)") 
plt.title("Variance Trajectories for Different Alphas (Laplace vs Diagonal) for WSD schedule")
plt.legend()
plt.grid()
plt.savefig(f"images/WSD_variance_trajectories_comparison_dim={dim}_Tmax={max(T_values)}.pdf")
plt.show()
#
# Plot each variance trajectory separately for better visibility (no ratio)
for alpha in list_alphas:
    color = colors[list_alphas.index(alpha)]
    Y_laplace = [laplace_variance_alpha[(alpha, T)] for T in T_values]
    Y_diagonal = [diagonal_variance_alpha[(alpha, T)] for T in T_values]
    plt.plot(T_values, Y_laplace, label=f"Laplace Variance (alpha={alpha})", marker='o', color=color)
    plt.plot(T_values, Y_diagonal, label=f"Diagonal Variance (alpha={alpha})", marker='x', color=color, linestyle="dashed")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(100, max(T_values))
    plt.xlabel("T (log scale)")
    plt.ylabel("Variance (log scale)") 
    plt.title(f"Variance Trajectory for alpha={alpha} (Laplace vs Diagonal) for WSD schedule")
    plt.legend()
    plt.grid()
plt.savefig(f"images/WSD_variance_trajectory_dim={dim}_Tmax={max(T_values)}.pdf")
plt.show()

# %%
from src.utils import save_dict_to_json
#transform keys into strings for json saving
laplace_variance_alpha_str_keys = {str(key): value for key, value in laplace_variance_alpha.items()}
diagonal_variance_alpha_str_keys = {str(key): value for key, value in diagonal_variance_alpha.items()}
save_dict_to_json(laplace_variance_alpha_str_keys, folder="laplace_wsd", filename=f"laplace_variance_alpha_dim={dim}_Tmax={max(T_values)}.json")
save_dict_to_json(diagonal_variance_alpha_str_keys, folder="laplace_wsd", filename=f"diagonal_variance_alpha_dim={dim}_Tmax={max(T_values)}.json")
# %%
##################
#BIAS and VARIANCE comparison

T_values = [10, 100, 500, 1000, 5000, 10000, 20000]
list_alphas = [1.1, 1.5]

laplace_bias_alpha, diagonal_bias_alpha, laplace_variance_alpha, diagonal_variance_alpha = (
    wsd_laplace_analysis.compare_different_alphas(
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
plt.title("Bias Ratio Trajectories for Different Alphas (Laplace vs Diagonal) for WSD schedule")
plt.legend()
plt.grid()
plt.savefig(f"images/WSD_bias_ratio_trajectories_comparison_dim={dim}_Tmax={max(T_values)}.pdf")
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
plt.title("Variance Ratio Trajectories for Different Alphas (Laplace vs Diagonal) for WSD schedule")
plt.legend()
plt.grid()
plt.savefig(f"images/WSD_variance_ratio_trajectories_comparison_dim={dim}_Tmax={max(T_values)}.pdf")
plt.show()
# %%
