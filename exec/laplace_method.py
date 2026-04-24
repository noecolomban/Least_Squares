#%%
import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression

from src.asymptotics import (
    Laplace_constant,  
    compute_laplace_for_several_ts, 
    compute_real_approx_for_several_ts
)

# %%
dim = 1000
sigma = 0.1
exponent = 2
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
x0 = diff0 + model.x_star.flatten()

T = 100000

laplace_analysis = Laplace_constant(model, x0, T=T)

print("Computing True approximation...")
laplace_result = laplace_analysis.compute_true_approx_risk()
print("True approximation computed = ", laplace_result)



# %%

print("Computing Laplace risk approximation...")
real_risk_approx = laplace_analysis.compute_lagrange_approx_risk(m_constant=Delta, m_exponent=beta)
print("Laplace risk approximation computed = ", real_risk_approx)
#%%


results_laplace = compute_laplace_for_several_ts([100, 500, 1000, 5000, 10000, 30000], model, x0, Delta, beta)
#%%


results_real_approx = compute_real_approx_for_several_ts([100, 500, 1000, 5000, 10000, 30000], model, x0)
# %%

plt.plot(list(results_laplace.keys()), list(results_laplace.values()), label="Laplace approximation")
plt.plot(list(results_real_approx.keys()), list(results_real_approx.values()), label="True approximation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.title("Laplace risk approximation vs True risk approximation for different T")
plt.legend()
plt.grid()
plt.savefig("images/laplace_vs_true_approximation_T=100000.pdf")
plt.show()
# %%
