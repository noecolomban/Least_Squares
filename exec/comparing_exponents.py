#%%

import numpy as np
import matplotlib.pyplot as plt
from src.least_squares import LinearRegression, PowerLawRegression, compute_power_x0
from src.SGD import SGD, NoisyGD
from scheduled import WSDSchedule, ConstantSchedule
from src.new_schedules.polynomial import PolynomialSchedule
import src.utils as utils
from src.visualization import Visualization
import copy
from src.risk_computations import RiskComputations, diff_to_exponents, diff_sgd_vs_approx
from src.utils import save_risk_results

#%%

T=1000
sigma = 0.1
dim = 100

eta = 0.1
eta_range = np.logspace(-4, 3, 100)
t_values = np.linspace(0, T-1, 10, dtype=int)

model = PowerLawRegression(dim=dim, sigma=sigma, exponent=2)

wsd = WSDSchedule(steps=T, base_lr=eta, cooldown_len=0.2)
constant = ConstantSchedule(steps=T, base_lr=eta)
linear = WSDSchedule(steps=T, base_lr=eta, cooldown_len=1.)

beta = 2
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta)


schedules1 = [wsd, constant, linear]
schedules2 = copy.deepcopy(schedules1)

risks_computations_sgd = RiskComputations(model, x0, schedules=schedules1, schedules_names=["wsd", "constant", "linear"], sgd_class=SGD)
risks_computations_noisy_gd = RiskComputations(model, x0, schedules=schedules2, schedules_names=["wsd", "constant", "linear"], sgd_class=NoisyGD)

#%%

exponents = np.linspace(0, 4, 15)
diff_results = diff_to_exponents(exponents=exponents, relative=True, dim=dim, sigma=sigma, schedules1=schedules1, schedules2=schedules2, eta_range=eta_range, x0=x0) 
diff_approx_vs_sgd = diff_sgd_vs_approx(exponents=exponents, relative=True, dim=dim, sigma=sigma, schedules1=schedules1, schedules2=schedules2, schedules_names=["wsd", "constant", "linear"], eta_range=eta_range, x0=x0)

#%%

visu = Visualization(schedules=schedules1 ,schedules_name=["wsd", "constant", "linear"])
visu.plot_sgd_classes_comparison(
    risks_class1=diff_results, 
    risks_class2=diff_approx_vs_sgd,
    X=exponents, 
    title=rf"Relative Difference in Risk Trajectories for Different Exponents, $T={T}$, $\sigma$={sigma}", 
    xlabel="Exponent", 
    ylabel="Relative Difference in Risk",
    logscale=True,
    label_class1=r"$(R_\text{NoisyGD}[T] - R_\text{SGD}[T]) / R_\text{SGD}[T]$",
    label_class2=r"$(R_\text{Approx}[T] - R_\text{SGD}[T]) / R_\text{SGD}[T]$",
    savefig=True,
    filename=f"Relative_Diff_Risk_Trajectories_Exponents_{T}_sigma_{sigma}_fast.pdf")
# %%
