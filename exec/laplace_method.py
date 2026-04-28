#%% CONSTANT SCHEDULE
from turtle import color
from unittest import result

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    Laplace_constant,  
    Laplace_linear,
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

laplace_analysis = Laplace_constant(model, x0, T)

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

from turtle import color
from unittest import result

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0

from src.asymptotics import (
    Laplace_constant,  
    Laplace_linear,
    compute_different_sigmas,

)

# %%
dim = 100
sigma = 0.5
exponent = 2
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)


T = 50000

linear_laplace_analysis = Laplace_linear(model, x0, T)
#%%
results_laplace = linear_laplace_analysis.compute_laplace_for_several_ts(
    T, Delta, beta,
    step=10)

results_real_approx = linear_laplace_analysis.compute_real_approx_for_several_ts(
    T,
    step=10)
# %%

print(results_laplace)

# %%
plt.plot(list(results_laplace.keys()), list(results_laplace.values()), label="Laplace approximation")
plt.plot(list(results_real_approx.keys()), list(results_real_approx.values()), label="True approximation")
plt.title(f"Linear schedule: \n Laplace risk approximation vs True risk approximation for different T \n sigma={sigma}, beta={beta}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.legend()
plt.grid()
plt.savefig(f"images/laplace_vs_true_approximation_linear_schedule_T={T}_sigma={sigma}.pdf")
plt.show()
# %%



results_different_sigmas_linear = compute_different_sigmas(10000, model, x0, Delta, beta, sigmas=[0., 0.1, 0.5, 1., 2., 3.], schedule_type="linear")
risk_dict, real_approx_dict = results_different_sigmas_linear


#%%
plt.figure(figsize=(10, 6))
for sigma in risk_dict.keys():
    # Generate a random color for each sigma
    #color = np.random.rand(3,)  # Random color with some transparency
    cmap = plt.get_cmap('tab20')  # Choose a colormap
    color = cmap(np.random.random())
    
    plt.plot(list(risk_dict[sigma].keys()), list(risk_dict[sigma].values()), label=f"Laplace approximation, sigma={sigma}", color=color)
    plt.plot(list(real_approx_dict[sigma].keys()), list(real_approx_dict[sigma].values()), label=f"Diagonal approximation, sigma={sigma}", linestyle="dashed", color=color)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.title(rf"Laplace risk approximation vs Diagonal approximation for different T and $\sigma$"+"\n"+rf"$\beta$={beta}")

# Place the legend outside the plot area
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid()

# Adjust the layout so the legend is not cut off in the saved PDF
plt.tight_layout()

plt.savefig(f"images/laplace_vs_true_approximation_different_sigmas_T={T}_linear.pdf")
plt.show()
# %%
