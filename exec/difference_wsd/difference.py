#%%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scheduled import ConstantSchedule, WSDSchedule
from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression, compute_power_x0
#%%

T = 1000
eta = 0.1
cooldown_len = 0.2
output_path = Path("images/wsd_vs_constant_same_eta.png")
risk_output_path = Path("images/risk_wsd_vs_constant_same_eta.png")

wsd = WSDSchedule(steps=T, base_lr=eta, cooldown_len=cooldown_len)
constant = ConstantSchedule(steps=T, base_lr=eta)
iterations = np.arange(1, T + 1)

alpha = 1.9
beta = 2
dim = 1000
sigma = 0.1

model = PowerLawRegression(dim=dim, sigma=sigma, exponent=alpha)
x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta/2)

computations = RiskComputations(model, x0, [wsd, constant], ["wsd", "constant"])


#%%

all_risks = computations.compute_all_theoretical_risks()
risk_wsd = np.asarray(all_risks["wsd"])
risk_constant = np.asarray(all_risks["constant"])

print(f"Final theoretical risk (wsd): {risk_wsd[-1]:.6e}")
print(f"Final theoretical risk (constant): {risk_constant[-1]:.6e}")

#%%

plt.figure(figsize=(8, 4.5))
plt.plot(iterations, wsd.schedule, label=f"wsd (cooldown_len={cooldown_len})", linewidth=2)
plt.plot(iterations, constant.schedule, label="constant", linewidth=2, linestyle="--")
plt.xlabel("iteration")
plt.ylabel("learning rate")
plt.title(f"WSD vs constant pour eta={eta}")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=200)
plt.show()


#%%

plt.figure(figsize=(8, 4.5))
plt.plot(iterations, risk_wsd, label="risk wsd", linewidth=2)
plt.plot(iterations, risk_constant, label="risk constant", linewidth=2, linestyle="--")
plt.xlabel("iteration")
plt.ylabel("theoretical risk")
plt.title(f"Risk trajectory (same eta={eta})")
plt.yscale("log")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

risk_output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(risk_output_path, dpi=200)
plt.show()

# %%

alphas = [1.5, 1.9, 2.3, 2.8]
cooldown_lens = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

results = []

for alpha_test in alphas:
	model_test = PowerLawRegression(dim=dim, sigma=sigma, exponent=alpha_test)
	x0_test = compute_power_x0(dim, model_test.x_star.flatten(), model_test.Q, beta=beta / 2)

	for cooldown_test in cooldown_lens:
		wsd_test = WSDSchedule(steps=T, base_lr=eta, cooldown_len=cooldown_test)
		constant_test = ConstantSchedule(steps=T, base_lr=eta)

		computations_test = RiskComputations(
			model_test,
			x0_test,
			[wsd_test, constant_test],
			["wsd", "constant"],
		)
		risks_test = computations_test.compute_all_theoretical_risks()

		final_wsd = float(np.asarray(risks_test["wsd"])[-1])
		final_constant = float(np.asarray(risks_test["constant"])[-1])
		abs_gap = abs(final_wsd - final_constant)
		rel_gap = abs_gap / max(abs(final_constant), 1e-15)

		results.append(
			{
				"alpha": alpha_test,
				"eta": eta,
				"cooldown_len": cooldown_test,
				"final_risk_wsd": final_wsd,
				"final_risk_constant": final_constant,
				"abs_gap": abs_gap,
				"rel_gap": rel_gap,
			}
		)

print("Meilleur cooldown_len par alpha (eta fixe):")
for alpha_test in alphas:
	rows_alpha = [row for row in results if row["alpha"] == alpha_test]
	best_alpha = min(rows_alpha, key=lambda row: row["rel_gap"])
	print(
		f"alpha={alpha_test:.2f}, eta={eta:.3g}, cooldown={best_alpha['cooldown_len']:.2f}, "
		f"abs_gap={best_alpha['abs_gap']:.6e}, rel_gap={best_alpha['rel_gap']:.6e}"
	)

# %%

relative_gap_output_path = Path("images/risk_relative_gap_vs_cooldown_alpha.png")

plt.figure(figsize=(8.2, 4.8))
for alpha_test in alphas:
	rows_alpha = sorted(
		[row for row in results if row["alpha"] == alpha_test],
		key=lambda row: row["cooldown_len"],
	)
	x_values = [row["cooldown_len"] for row in rows_alpha]
	y_values = [row["rel_gap"] for row in rows_alpha]
	plt.plot(x_values, y_values, marker="o", linewidth=2, label=f"alpha={alpha_test:.2f}")

plt.xlabel("cooldown_len")
plt.ylabel("relative final gap")
plt.title(f"Ecart relatif final vs cooldown_len (eta fixe={eta})")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

relative_gap_output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(relative_gap_output_path, dpi=200)
plt.show()

# %%
#BATCH


batch_sizes = [1, 4, 16, 64]
cooldown_lens_batch = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# Keep global noise level constant when batch changes:
# sigma(B) = sigma_ref / sqrt(B)
sigma_ref = sigma
alpha_batch = alpha

batch_results = []

for batch_size in batch_sizes:
	sigma_batch = sigma_ref / np.sqrt(batch_size)
	model_batch = PowerLawRegression(dim=dim, sigma=sigma_batch, exponent=alpha_batch)
	x0_batch = compute_power_x0(dim, model_batch.x_star.flatten(), model_batch.Q, beta=beta / 2)

	for cooldown_test in cooldown_lens_batch:
		wsd_test = WSDSchedule(steps=T, base_lr=eta, cooldown_len=cooldown_test)
		constant_test = ConstantSchedule(steps=T, base_lr=eta)

		computations_batch = RiskComputations(
			model_batch,
			x0_batch,
			[wsd_test, constant_test],
			["wsd", "constant"],
		)
		risks_batch = computations_batch.compute_all_theoretical_risks()

		final_wsd = float(np.asarray(risks_batch["wsd"])[-1])
		final_constant = float(np.asarray(risks_batch["constant"])[-1])
		abs_gap = abs(final_wsd - final_constant)
		rel_gap = abs_gap / max(abs(final_constant), 1e-15)

		batch_results.append(
			{
				"batch": batch_size,
				"sigma": sigma_batch,
				"alpha": alpha_batch,
				"cooldown_len": cooldown_test,
				"final_risk_wsd": final_wsd,
				"final_risk_constant": final_constant,
				"abs_gap": abs_gap,
				"rel_gap": rel_gap,
			}
		)

print("Meilleur cooldown_len par batch B (eta et alpha fixes):")
for batch_size in batch_sizes:
	rows_batch = [row for row in batch_results if row["batch"] == batch_size]
	best_batch = min(rows_batch, key=lambda row: row["rel_gap"])
	print(
		f"B={batch_size}, sigma={best_batch['sigma']:.4g}, cooldown={best_batch['cooldown_len']:.2f}, "
		f"abs_gap={best_batch['abs_gap']:.6e}, rel_gap={best_batch['rel_gap']:.6e}"
	)


# %%

relative_gap_batch_output_path = Path("images/risk_relative_gap_vs_cooldown_batch.png")

plt.figure(figsize=(8.2, 4.8))
for batch_size in batch_sizes:
	rows_batch = sorted(
		[row for row in batch_results if row["batch"] == batch_size],
		key=lambda row: row["cooldown_len"],
	)
	x_values = [row["cooldown_len"] for row in rows_batch]
	y_values = [row["rel_gap"] for row in rows_batch]
	sigma_batch = rows_batch[0]["sigma"]
	plt.plot(
		x_values,
		y_values,
		marker="o",
		linewidth=2,
		label=f"B={batch_size}, sigma={sigma_batch:.4g}",
	)

plt.xlabel("cooldown_len")
plt.ylabel("relative final gap")
plt.title(
	f"Ecart relatif final vs cooldown_len (eta fixe={eta}, alpha fixe={alpha_batch})"
)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

relative_gap_batch_output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(relative_gap_batch_output_path, dpi=200)
plt.show()

# %%