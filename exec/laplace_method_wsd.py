#%% WSD SCHEDULE
import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    LaplaceWSD,
    compute_different_sigmas,
)

# %%
dim = 100
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
