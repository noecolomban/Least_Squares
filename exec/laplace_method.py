#%%
import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import LinearRegression, PowerLawRegression
from src.SGD import SGD, NoisyGD
from scheduled import WSDSchedule, ConstantSchedule
from src.new_schedules.polynomial import PolynomialSchedule
import src.utils as utils
from src.visualization import Visualization
import copy
from src.risk_computations import RiskComputations, diff_to_exponents, diff_sgd_vs_approx
from src.utils import save_risk_results
from src.asymptotics import ZTransform_constant, z_transform_several_ts, Laplace_constant

# %%
dim = 100
sigma = 0.1
exponent = 2
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
x0 = diff0 + model.x_star.flatten()

T = 10000

laplace_analysis = Laplace_constant(model, x0, T=T)

print("Computing True approximation...")
laplace_result = laplace_analysis.compute_true_approx_risk()
print("True approximation computed = ", laplace_result)

# %%

print("Computing Laplace risk approximation...")
real_risk_approx = laplace_analysis.compute_lagrange_approx_risk(m_constant=Delta, m_exponent=beta)
print("Laplace risk approximation computed = ", real_risk_approx)
#%%

def compute_laplace_for_several_ts(T_values):
    results = {}
    for T in T_values:
        print(f"Computing Laplace risk approximation for T={T}...")
        laplace_analysis = Laplace_constant(model, x0, T=T)
        laplace_result = laplace_analysis.compute_lagrange_approx_risk(m_constant=Delta, m_exponent=beta)
        results[T] = laplace_result
        print(f"Laplace risk approximation for T={T} computed: {laplace_result}")
    return results

results_laplace = compute_laplace_for_several_ts([100, 500, 1000, 5000, 10000])
#%%

def compute_real_approx_for_several_ts(T_values):
    results = {}
    for T in T_values:
        print(f"Computing True risk approximation for T={T}...")
        laplace_analysis = Laplace_constant(model, x0, T=T)
        laplace_result = laplace_analysis.compute_true_approx_risk()
        results[T] = laplace_result
        print(f"True risk approximation for T={T} computed: {laplace_result}")
    return results

results_real_approx = compute_real_approx_for_several_ts([100, 500, 1000, 5000, 10000])
# %%

plt.plot(list(results_laplace.keys()), list(results_laplace.values()), label="Laplace approximation")
plt.plot(list(results_real_approx.keys()), list(results_real_approx.values()), label="True approximation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk approximation")
plt.title("Laplace risk approximation vs T")
plt.legend()
plt.grid()
plt.show()
# %%
