#%%
"""Numerical verification of the Slock-model SGD risk analysis (slock_analysis.pdf)."""
import numpy as np
from scipy.special import gamma, zeta

rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# Part 1: Monte-Carlo check of Theorem 1 (exact risk recursion)
# ---------------------------------------------------------------
print("=" * 70)
print("Part 1: Theorem 1 (exact risk formula) vs Monte-Carlo Slock SGD")
print("=" * 70)
d, L, alpha, beta, Delta = 30, 1.0, 1.5, 2.0, 1.0
idx_arr = np.arange(1, d + 1.0)
lam = L * idx_arr ** -alpha
trH = lam.sum()
eta, sigma = 0.05, 0.5
xstar = np.sqrt(Delta) * idx_arr ** (-beta / 2)   # so m0_i = Delta i^-beta
T, R = 500, 20000

# theory recursion (Theorem 1, with m0 = diag of E[(x0-x*)(x0-x*)^T])
m = xstar ** 2
q = 1 - 2 * eta * lam + eta ** 2 * trH * lam
risk_th = np.empty(T)
for t in range(T):
    risk_th[t] = 0.5 * lam @ m
    m = q * m + eta ** 2 * sigma ** 2 * lam

# bias-only theory (sigma = 0)
m = xstar ** 2
bias_th = np.empty(T)
for t in range(T):
    bias_th[t] = 0.5 * lam @ m
    m = q * m

# Monte Carlo, x0 = 0  =>  delta0 = -xstar
def run_mc(sig):
    delta = np.tile(-xstar, (R, 1))
    rows = np.arange(R)
    out = np.empty(T)
    for t in range(T):
        out[t] = 0.5 * np.mean((delta ** 2) @ lam)
        idx = rng.choice(d, size=R, p=lam / trH)
        s = rng.choice([-1.0, 1.0], size=R)
        eps = rng.normal(0, sig, size=R)
        delta[rows, idx] = (1 - eta * trH) * delta[rows, idx] \
            + eta * eps * s * np.sqrt(trH)
    return out

risk_mc = run_mc(sigma)
bias_mc = run_mc(0.0)

print(f"{'t':>6} {'risk MC':>12} {'risk Thm1':>12} {'ratio':>8} "
      f"{'bias MC':>12} {'bias Thm1':>12} {'ratio':>8}")
for t in [0, 10, 50, 100, 250, 499]:
    print(f"{t:>6} {risk_mc[t]:>12.5e} {risk_th[t]:>12.5e} "
          f"{risk_mc[t]/risk_th[t]:>8.4f} "
          f"{bias_mc[t]:>12.5e} {bias_th[t]:>12.5e} "
          f"{bias_mc[t]/bias_th[t]:>8.4f}")

# ---------------------------------------------------------------
# Part 2: Theorem 3 bias asymptote (constant schedule) vs exact B_t
# ---------------------------------------------------------------
print()
print("=" * 70)
print("Part 2: Theorem 3 constant-schedule bias asymptote vs exact B_t")
print("=" * 70)
N = 2_000_000
i = np.arange(1, N + 1.0)
lamN = L * i ** -alpha
trZ = L * zeta(alpha)            # Tr(Lambda) of the infinite model
eta = 0.1
etabar = eta - 0.5 * eta ** 2 * trZ
logq = np.log1p(-2 * etabar * lamN)
coef = 0.5 * Delta * L * i ** -(alpha + beta)
expo = (beta - 1) / alpha + 1
pref = (L * Delta / (2 * alpha)) * gamma((beta - 1) / alpha + 1)
print(f"alpha={alpha}, beta={beta}, eta={eta}, etabar={etabar:.5f}")
print(f"{'t':>9} {'B_t exact':>14} {'Thm 3 asymptote':>16} {'ratio':>8}")
for t in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
    B_exact = np.sum(coef * np.exp(t * logq))
    B_asym = pref * (2 * etabar * L * t) ** (-expo)
    print(f"{t:>9} {B_exact:>14.6e} {B_asym:>16.6e} {B_exact/B_asym:>8.4f}")

# ---------------------------------------------------------------
# Part 3: Theorem 4 bias asymptote (linear schedule) vs exact B_T
# ---------------------------------------------------------------
print()
print("=" * 70)
print("Part 3: Theorem 4 linear-schedule bias asymptote vs exact B_T")
print("=" * 70)
N = 200_000
i = np.arange(1, N + 1.0)
lamN = L * i ** -alpha
coef = 0.5 * Delta * L * i ** -(alpha + beta)
print(f"{'T':>7} {'B_T exact':>14} {'Thm 4 asymptote':>16} {'ratio':>8}")
for T_ in [300, 1000, 3000, 10000, 30000]:
    k = np.arange(T_)
    etak = eta * (1 - k / T_)
    etabark = etak - 0.5 * trZ * etak ** 2
    acc = np.zeros(N)
    for c in range(0, T_, 200):                 # chunk over steps
        eb = etabark[c:c + 200]
        acc += np.log1p(-2 * np.outer(eb, lamN)).sum(axis=0)
    B_exact = np.sum(coef * np.exp(acc))
    B_asym = pref * (eta * L * T_ * (1 - eta * trZ / 3)) ** (-expo)
    print(f"{T_:>7} {B_exact:>14.6e} {B_asym:>16.6e} {B_exact/B_asym:>8.4f}")

# %%
