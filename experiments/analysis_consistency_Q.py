"""Cross-batch consistency check of Proposition 1 on the synthetic population Q.
Batch 1 (primary, MINI-25): BRIDGE 475 accepted, 8 harmful. Batch 2 (fresh tasks, MINI-33, HGB): 462 accepted, 7 harmful.
U is the Proposition 1 bound (delta = 0.05). Because U >= S/M by construction, comparing a batch's estimate with its own
bound is not a test; the bound of one batch is therefore compared with the estimate from the other, independent batch."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta
def kl(x, u): 
    x = min(max(x, 1e-15), 1 - 1e-15); return x * np.log(x / u) + (1 - x) * np.log((1 - x) / (1 - u))
def U(S, M, delta=0.05):
    if M == 0: return 1.0
    x = S / M; f = lambda u: M * kl(x, u) - np.log(1 / delta)
    return 1.0 if f(1 - 1e-12) <= 0 else brentq(f, max(x, 1e-12), 1 - 1e-12)
b = {"primary": (8, 475), "fresh": (7, 462)}
for k, (S, M) in b.items(): print(f"{k}: r_hat = {100*S/M:.2f}%  U = {100*U(S, M):.2f}%")
S, M = 15, 937
lo, hi = beta.ppf(0.025, S, M - S + 1), beta.ppf(0.975, S + 1, M - S)
print(f"pooled r_hat = {100*S/M:.2f}%  95% CI (Clopper-Pearson) {100*lo:.2f}-{100*hi:.2f}%")
