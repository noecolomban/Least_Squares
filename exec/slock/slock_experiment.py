#%%
import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0
from scipy.special import zeta
from src.asymptotics import (
    LaplaceConstant,  
    LaplaceLinear,
    SlockLinear,
    SlockConstant,
    Mode,
    compute_different_sigmas,
)
from src.utils import save_dict_to_json
from src.SGD import SGD
from scheduled import ConstantSchedule
# %%
dim = 100
sigma = 1
exponent = 2.1 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
T_values = [10, 100, 500, 1000, 5000, 10000, 20000]
#T_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
list_alphas = [1.1, 1.5, 2.5]

eta = 0.001
mode = Mode.SLOCK
#%%
slock_constant = SlockConstant(model, x0, T_max=max(T_values), optimize=optimize, base_lr=eta, beta=beta)
schedule = ConstantSchedule(steps=max(T_values), base_lr=eta)
sgd = SGD(model, x0, schedule)

#%%
losses = sgd.sample_slock(show=False, n_samples=10)

biases, variances = slock_constant.compute_slock_biases_and_variances(T_values)
slock_risks = {T: biases[T] + variances[T] for T in T_values}

# %%
X = np.arange(max(T_values))
plt.plot(X, losses, label="SGD Loss")
plt.plot(T_values, [slock_risks[T] for T in T_values], marker='o', label='SLOCK Risk', color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss / Risk")
plt.legend()
# %%
