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
# %%
wsd_laplace_analysis = LaplaceWSD(model, x0, T_max=T, optimize=optimize, base_lr=base_lr, cooldown_len=cooldown_len)

#%%

eta, risk_min, risks = wsd_laplace_analysis.optimize_eta(m_constant=1, T=100000, K=K, eta_min=1e-5, eta_max=1, num_points=300)

# %%
plt.plot(risks.keys(), risks.values())
plt.xlabel("Eta")
plt.ylabel("Risk")
plt.title("Risk vs Eta")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.savefig(f"images/eta_optimization_wsd_T={T}_sigma={sigma}_beta={beta}.pdf")
plt.show()
print(f"Optimal eta: {eta}, Minimum Risk: {risk_min}")
# %%
