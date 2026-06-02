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

