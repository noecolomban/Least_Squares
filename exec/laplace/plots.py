#%%

from src.utils import save_dict_to_json, read_dict_from_json
import matplotlib.pyplot as plt
import numpy as np

file = "LINEAR_all_laplace_variances_0.01T_Tmax=500000.json"
all_laplace_variances = read_dict_from_json("laplace_linear", file)

print(all_laplace_variances)

#%%
T_values = sorted(set(key[1] for key in all_laplace_variances.keys()))
alphas = sorted(set(key[0] for key in all_laplace_variances.keys()))

print(T_values)
print(alphas)
# %%
plt.figure(figsize=(10, 6))
for alpha in alphas:
    variances = [all_laplace_variances[(alpha, T)] for T in T_values]
    plt.plot(T_values, variances, label=f"alpha={alpha}")
plt.xscale("log")
plt.xlabel("T")
plt.ylabel("Laplace Variance Approximation")
plt.title("Laplace Variance Approximation vs T for Different Alphas")
plt.legend()
plt.grid()
# %%
