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
from src.asymptotics import ZTransform_constant, z_transform_several_ts

# %%
T=10000
sigma = 0.01
dim = 1000

eta = 0.1
eta_range = np.logspace(-4, 2, 30)
t_values = np.linspace(0, T-1, 10, dtype=int)

model = PowerLawRegression(dim=dim, sigma=sigma, exponent=2)
constant = ConstantSchedule(steps=T, base_lr=eta)

beta = 0
x0 = np.array([1/i**beta for i in range(1, dim+1)])

schedules1 = [constant]
asymptotics_analysis = ZTransform_constant(model, x0, T=T)

# %%
ztransform, approx = asymptotics_analysis.compute_all_approx_vs_z_transform()

# %%
visu = Visualization(schedules=schedules1 ,schedules_name=["constant"])
visu.plot_sgd_classes_comparison(
    risks_class1=approx, 
    risks_class2=ztransform,
    title=rf"Approximate vs Z-transform risks for {model.__class__.__name__} with {constant.__class__.__name__}", 
    ylabel="Difference in Risk",
    logscale=True,
    label_class1=r"$R_\text{Approx}[T]$",
    label_class2=r"$R_\text{ZTransform}[T]$",
    savefig=True,
    filename=f"Approx_vs_ZTransform_{model.__class__.__name__}_{constant.__class__.__name__}"
    )

#%%
eta_range = np.logspace(-4, 2, 30)
results = z_transform_several_ts(list_T=[100, 1000,  10000, 100000, 200000], sigma=0.1, dim=50, eta_range=eta_range)
# %%
visu = Visualization(schedules=schedules1 ,schedules_name=["constant"])
for T in results:
    print(f"Plotting results for T={T}")
    ztransform, approx = results[T]
    visu.plot_sgd_classes_comparison(
        risks_class1=approx, 
        risks_class2=ztransform,
        title=rf"Approximate vs Z-transform risks for {model.__class__.__name__} with {constant.__class__.__name__} at T={T}", 
        ylabel="Difference in Risk",
        logscale=True,
        label_class1=r"$R_\text{Approx}[T]$",
        label_class2=r"$R_\text{ZTransform}[T]$",
        savefig=True,
        filename=f"Approx_vs_ZTransform_{model.__class__.__name__}_{constant.__class__.__name__}_T_{T}"
    )


# %%
final_risks = {T: (results[T][1]["constant"][-1], results[T][0]["constant"][-1]) for T in results}

plt.plot(list(final_risks.keys()), [final_risks[T][0] for T in final_risks], label="Approx", marker='o')
plt.yscale('log')
plt.xscale('log')
plt.xlabel("T")
plt.ylabel("Final Risk")
plt.axhline(results[T][0]['constant'][-1], color='r', linestyle='--', label="Z-transform limit")
plt.title("Final Risk vs T for Approximate and Z-transform Risks")
plt.savefig(f"images/Final_Risk_vs_T_Approx_vs_ZTransform_{model.__class__.__name__}_{constant.__class__.__name__}.pdf")
plt.legend()
plt.show()
# %%
print(results)