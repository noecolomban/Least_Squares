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
from scheduled import ConstantSchedule, WSDSchedule
# %%

dim = 100
sigma = 0.1
exponent = 1.5 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
T_values = [1, 10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
#T_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
list_alphas = [1.1, 1.5, 2.5]

eta = 0.1
mode = Mode.SLOCK
#%%
slock_constant = SlockConstant(model, x0, T_max=max(T_values), optimize=optimize, base_lr=eta, beta=beta)
schedule = ConstantSchedule(steps=max(T_values), base_lr=eta)
sgd = SGD(model, x0, schedule)

#%%
losses = sgd.sample_slock(show=False, n_samples=50)

biases, variances = slock_constant.compute_slock_biases_and_variances(T_values)
slock_risks = {T: biases[T] + variances[T] for T in T_values}

# %%
X = np.arange(1, max(T_values)+1)
plt.plot(X, losses[0:max(T_values)], label="SGD Loss")
plt.plot(T_values, [slock_risks[T] for T in T_values], marker='o', label='SLOCK Risk', color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlim(1, max(T_values))
plt.xlabel("Epoch")
plt.ylabel("Loss / Risk")
plt.legend()
# %%

to_save = {
    "sgd": {int(T): losses[int(T)-1] for T in np.arange(1, max(T_values)+1)},
    "true": slock_risks
}
save_dict_to_json(to_save, folder=f"slock_experiment_dim={dim}", filename=f"losses_and_risks_alpha={exponent}_beta={beta}_L={eta}_Delta={Delta}_sigma={sigma}.json")

# %%

#LINEAR CASE
schedule = WSDSchedule(steps=max(T_values), cooldown_len=1, base_lr=eta)
sgd = SGD(model, x0, schedule)
slock_linear = SlockLinear(model, x0, T_max=max(T_values), optimize=optimize, base_lr=eta, beta=beta)

biases, variances = slock_linear.compute_slock_biases_and_variances(T_values)
slock_risks = {T: biases[T] + variances[T] for T in T_values}

#%%
losses = {}
for T in T_values:
    schedule = WSDSchedule(steps=T, cooldown_len=1, base_lr=eta)
    sgd = SGD(model, x0, schedule)
    losses[T] = sgd.sample_slock(show=False, n_samples=50)


#%%

plt.figure(figsize=(10, 6))
X = np.arange(1, max(T_values)+1)
for T in T_values:
    plt.plot(X[0:T], losses[T][0:T], label=f"SGD Loss (T={T})")
plt.plot(T_values, [slock_risks[T] for T in T_values], marker='o', label='SLOCK Risk', color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlim(1, max(T_values))
plt.xlabel("Epoch")
plt.ylabel("Loss / Risk")
plt.legend()

# %%
to_save = {
    "sgd": {int(T): {int(t): losses[T][t-1] for t in np.arange(1, T+1)} for T in T_values},
    "true": slock_risks
}
save_dict_to_json(to_save, folder=f"slock_experiment_dim={dim}", filename=f"LINEAR_losses_and_risks_alpha={exponent}_beta={beta}_L={eta}_Delta={Delta}_sigma={sigma}.json")

# %%
