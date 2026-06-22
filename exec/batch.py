#%%
from matplotlib import figure
import numpy as np
import matplotlib.pyplot as plt
from exec.risks_slock import T
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
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000]
# %%
new_linear_laplace_analysis = LaplaceLinear(model, x0, T_max=max(T_values), optimize=optimize, base_lr=0.001)
#%%

T_values_for_alpha = [100, 1000, 10000, 50000, 100000]
batchs = [1, 10, 50]
alphas = [1.1, 1.5, 1.9]


all_laplace_variances = {}
all_diagonal_variances = {}

for b in batchs:
    new_linear_laplace_analysis._update_model_for_batch(b)
    laplace_variance_alpha = {}
    diagonal_variance_alpha = {}
    all_laplace_variances[b], all_diagonal_variances[b] = new_linear_laplace_analysis.compare_variance_trajectories_different_alphas(
        T_values=T_values,
        list_alphas=alphas,
        m_exponent=beta,
        m_constant=Delta,
        mode=Mode.DIAGONAL
    )
#%%
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
for i, alpha in enumerate(alphas):
    plt.figure(figsize=(10, 6))
    for j, b in enumerate(batchs):
        color = colors[j % len(colors)]  # Use a different color for each alpha
        style = styles[i % len(styles)]  # Use a different style for each batch
        marker = markers[i % len(markers)]  # Use a different marker for each batch
        plt.plot(T_values, [all_laplace_variances[b][(alpha, T)] for T in T_values], label=f"Laplace alpha={alpha}, batch={b}", color=color, linestyle=style)
        plt.plot(T_values, [all_diagonal_variances[b][(alpha, T)] for T in T_values], label=f"Diagonal alpha={alpha}, batch={b}", color=color, linestyle="--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("T")
plt.ylabel("Variance")
plt.title(f"Variance trajectories for batch size {b} and different alphas")
plt.legend()
plt.grid()
plt.savefig(f"images/variance_trajectories_batch_{b}.pdf")
plt.show()
# %%


# %% Plot variance as a function of compute budget (BT)
#LAPLACE ONLY

T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
alphas = [1.1, 1.5, 1.9]
batchs = [1, 10, 50]
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
mode = Mode.LAPLACE_ONLY  # Change to Mode.LAPLACE_ONLY if you want to plot the Laplace variance instead of the diagonal variance

all_laplace_variances = {}
for b in batchs:
    new_linear_laplace_analysis._update_model_for_batch(b)
    laplace_variance_alpha = {}
    diagonal_variance_alpha = {}
    all_laplace_variances[b] = new_linear_laplace_analysis.compare_variance_trajectories_different_alphas(
        T_values=T_values,
        list_alphas=alphas,
        m_exponent=beta,
        m_constant=Delta,
        mode=mode
    )
plt.figure(figsize=(10, 6))

#%%
#PLOT
figure=plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    for j, b in enumerate(batchs):
        color = colors[j % len(colors)]
        style = styles[i % len(styles)]
        marker = markers[i % len(markers)]
        # The x-axis is no longer T, but B * T
        budget_values = [b * T for T in T_values]
        
        # Plot only the Laplace approximation to simplify the chart
        laplace_vars = [all_laplace_variances[b][(alpha, T)] for T in T_values]
        
        plt.plot(budget_values, laplace_vars, 
                 label=f"Laplace: alpha={alpha}, batch={b}", 
                 color=color, 
                 marker=marker, # Add markers to clearly see the measurement points
                 linestyle=style)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Total compute budget (B * T)")
plt.ylabel("Variance")
plt.title(f"Batch Scaling Efficiency (mode={mode.name}) for different alphas and batch sizes")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

# Save the figure with an explicit filename
plt.savefig(f"images/BATCH_variance_iso_budget_{mode}.pdf")
plt.show()
# %%
#BIAS

# %% Bias trajectories vs T for different alphas and batch sizes
T_values = [10, 100, 500, 1000, 5000, 10000, 20000, 50000]
alphas = [1.1, 1.5, 1.9]
batchs = [1, 10, 50]

all_laplace_biases = {}
all_diagonal_biases = {}

for b in batchs:
    new_linear_laplace_analysis._update_model_for_batch(b)
    laplace_bias_alpha = {}
    diagonal_bias_alpha = {}

    current_eta = (
        new_linear_laplace_analysis.schedule.get_base_lr()
        if getattr(new_linear_laplace_analysis, "schedule", None) is not None
        else 0.01
    )

    for T in T_values:
        for alpha in alphas:
            new_linear_laplace_analysis._update_model_for_alpha(alpha)
            new_linear_laplace_analysis._setup_for_T(T, optimize=False, base_lr=current_eta)
            t = T - 1

            laplace_bias_alpha[(alpha, T)] = new_linear_laplace_analysis.compute_laplace_approx_bias(
                T=T,
                t=t,
                m_exponent=beta,
                m_constant=Delta,
            )

            diagonal_bias, _ = new_linear_laplace_analysis.compute_true_approx_biases_and_variances(
                [T],
                K=1,
            )
            diagonal_bias_alpha[(alpha, T)] = diagonal_bias[T]

    all_laplace_biases[b] = laplace_bias_alpha
    all_diagonal_biases[b] = diagonal_bias_alpha

for i, alpha in enumerate(alphas):
    plt.figure(figsize=(10, 6))
    for j, b in enumerate(batchs):
        color = colors[j % len(colors)]
        style = styles[i % len(styles)]
        marker = markers[i % len(markers)]
        plt.plot(
            T_values,
            [all_laplace_biases[b][(alpha, T)] for T in T_values],
            label=f"Laplace alpha={alpha}, batch={b}",
            color=color,
            linestyle=style,
            marker=marker,
        )
        plt.plot(
            T_values,
            [all_diagonal_biases[b][(alpha, T)] for T in T_values],
            label=f"Diagonal alpha={alpha}, batch={b}",
            color=color,
            linestyle="--",
            marker=marker,
            alpha=0.8,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("T")
    plt.ylabel("Bias")
    plt.title(f"Bias trajectories for alpha={alpha} and different batch sizes")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f"images/bias_trajectories_alpha_{alpha}.pdf")
    plt.show()


# %% Plot bias as a function of compute budget (B*T) - LAPLACE ONLY
T_values = [10, 100, 500, 1000, 5000, 10000, 20000]
alphas = [1.1, 1.5, 1.9]
batchs = [1, 10, 50]

all_laplace_biases = {}
for b in batchs:
    new_linear_laplace_analysis._update_model_for_batch(b)
    laplace_bias_alpha = {}

    current_eta = (
        new_linear_laplace_analysis.schedule.get_base_lr()
        if getattr(new_linear_laplace_analysis, "schedule", None) is not None
        else 0.01
    )

    for T in T_values:
        for alpha in alphas:
            new_linear_laplace_analysis._update_model_for_alpha(alpha)
            new_linear_laplace_analysis._setup_for_T(T, optimize=False, base_lr=current_eta)
            t = T - 1

            laplace_bias_alpha[(alpha, T)] = new_linear_laplace_analysis.compute_laplace_approx_bias(
                T=T,
                t=t,
                m_exponent=beta,
                m_constant=Delta,
            )

    all_laplace_biases[b] = laplace_bias_alpha

#%%
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    for j, b in enumerate(batchs):
        color = colors[i % len(colors)]
        style = styles[j % len(styles)]
        marker = markers[j % len(markers)]
        budget_values = [b * T for T in T_values]
        laplace_biases = [all_laplace_biases[b][(alpha, T)] for T in T_values]

        plt.plot(
            budget_values,
            laplace_biases,
            label=f"Laplace: alpha={alpha}, batch={b}",
            color=color,
            marker=marker,
            linestyle=style,
        )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Total compute budget (B * T)")
plt.ylabel("Bias")
plt.title("Batch Scaling Efficiency for bias (Laplace) across alphas and batch sizes")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig(f"images/BATCH_bias_iso_budget_laplace_T={max(T_values)}.pdf")
plt.show()


# %%
