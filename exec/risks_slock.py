#%%

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import PowerLawRegression, compute_power_x0
from src.SGD import SGD, NoisyGD
from scheduled import WSDSchedule, ConstantSchedule


# %%
dim = 100
sigma = 0.5
exponent = 2 #alpha
model = PowerLawRegression(dim=dim, sigma=sigma, exponent=exponent)

Delta = 1
beta = 2
diff0 = Delta * np.array([1/i**beta for i in range(1, dim+1)])
#beta/2 so that m0i = Delta/i^beta, which is the form we want
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

optimize = False
K = 1
T = 100000

schedule = WSDSchedule(steps=T, base_lr=0.1, cooldown_len=0.2)

sgd = SGD(model, x0, schedule)

#%%

real_risks = sgd.compute_all_theoretical_risks()
slock_risks = sgd.all_slock_risks()

# %%
plt.plot(real_risks, label="Real Risks")
plt.plot(slock_risks, label="Slock Risks")
plt.title(f"Comparison of Real Risks and Slock Risks for SGD with WSD schedule \n sigma={sigma}, beta={beta}, T={T}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Risk")
plt.legend()
plt.grid()
plt.savefig(f"images/real_vs_slock_risks_sgd_wsd_schedule_T={T}_sigma={sigma}_beta={beta}.pdf")
plt.show()
# %%
