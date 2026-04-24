#%%
from turtle import color
from unittest import result

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    Laplace_constant,  
    compute_laplace_for_several_ts, 
    compute_real_approx_for_several_ts
)

# %%
dim = 100
sigma = 0.
exponent = 2
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)


T = 10000

laplace_analysis = Laplace_constant(model, x0)

#print("Computing True approximation...")
#laplace_result = laplace_analysis.compute_true_approx_risk()
#print("True approximation computed = ", laplace_result)
# %%

#print("Computing Laplace risk approximation...")
#real_risk_approx = laplace_analysis.compute_lagrange_approx_risk(m_constant=Delta, m_exponent=beta)
#print("Laplace risk approximation computed = ", real_risk_approx)
#%%


results_laplace = compute_laplace_for_several_ts([100, 500, 1000, 5000], model, x0, Delta, beta)
#%%

results_real_approx = compute_real_approx_for_several_ts([100, 500, 1000, 5000], model, x0)
# %%

plt.plot(list(results_laplace.keys()), list(results_laplace.values()), label="Laplace approximation")
plt.plot(list(results_real_approx.keys()), list(results_real_approx.values()), label="True approximation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.title(f"Laplace risk approximation vs True risk approximation for different T, sigma={sigma}, beta={beta}")
plt.legend()
plt.grid()
plt.savefig("images/laplace_vs_true_approximation_T=100000.pdf")
plt.show()
# %%

def compute_different_sigmas(T_values, model, x0, Delta, beta, sigmas):
    """Compute Laplace risk approximation for different noise levels."""
    results = {}
    real_approx = {}
    
    for sigma in sigmas:
        print(f"Computing for sigma={sigma}...")
        model.sigma = sigma  # Update the noise level in the model
        laplace_analysis = Laplace_constant(model, x0)  # Re-instantiate to update m0 if needed
        risk = compute_laplace_for_several_ts(T_values, model, x0, Delta, beta)
        real_approx[sigma] = compute_real_approx_for_several_ts(T_values, model, x0)
        results[sigma] = risk
        print(f"Laplace risk approximation for sigma={sigma} computed.")
        
    return results, real_approx

results_different_sigmas = compute_different_sigmas([100, 500, 1000, 5000, 10000, 50000], model, x0, Delta, beta, sigmas=[0., 0.1, 0.5, 1.])
# %%
risk_dict, real_approx_dict = results_different_sigmas

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
plt.savefig("images/laplace_vs_true_approximation_different_sigmas_T=100000.pdf")
plt.show()