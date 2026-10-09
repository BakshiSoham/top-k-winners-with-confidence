"""Feasibility prototype for the megastudy top-k experiment (see experiment_proposal.md).

Gaussian re-run at megastudy-plausible calibration: M = 22 arms, k = 3, 90% intervals,
true effects 2.0 pp down to -0.2 pp, per-arm SE 0.6-1.0 pp.

This is a CALIBRATION, not real data. It predicts what the design-based subsampling
experiment of proposal section 3 should produce on a real megastudy. Run from the repo root:

    python megastudy_prototype.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
if not hasattr(np, "trapz"):           # NumPy >= 2.0 renamed it; the repo still calls trapz
    np.trapz = np.trapezoid

from math import comb, log                                              # noqa: E402
from src import PolyhedralTopKInference, TopKSelectionModel             # noqa: E402

# ---------------------------------------------------------------- calibration
MU_PP = np.array([2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                  0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, -0.2])
M = len(MU_PP)
K = 3
ALPHA = 0.10
B = 1200
P0 = 0.30            # holdout-arm uptake
N_HOLDOUT = 60_000

mu = MU_PP / 100.0
# arm sizes 2,000-6,000 -- a ~10% pilot subsample of a ~700k megastudy
n_arm = np.random.default_rng(7).integers(2000, 6000, M)
se = np.sqrt(P0 * (1 - P0) / n_arm + P0 * (1 - P0) / N_HOLDOUT)
Sigma = np.diag(se ** 2)

EPSILON = log(comb(M, K))
ZCRIT = norm.ppf(1 - ALPHA / 2)
utility = lambda x: np.asarray(x, dtype=float)                          # noqa: E731


# ---------------------------------------------------------------- methods
def standard(t):
    S = tuple(sorted(np.argsort(-t)[:K]))
    return S, {int(j): (t[j] - ZCRIT * se[j], t[j] + ZCRIT * se[j]) for j in S}


def polyhedral(t):
    model = PolyhedralTopKInference(X=t, k=K, H0_mu=np.zeros(M), Sigma=Sigma,
                                   utility_fn=utility, grid_size=500, alpha=ALPHA)
    recs = model.confidence_interval_topk(alpha=ALPHA)
    return (tuple(sorted(int(j) for j in model.selected_set)),
            {int(r["index"]): (float(r["ci_lower"]), float(r["ci_upper"])) for r in recs})


def randomized(t, seed):
    model = TopKSelectionModel(X=t, k=K, H0_mu=np.zeros(M), true_Sigma=Sigma,
                               utility_fn=utility, epsilon=EPSILON, grid_size=500,
                               sel_scale="adaptive")
    S, _ = model.randomized_selected_top_k(X=t, k=K, epsilon=EPSILON,
                                           scale="adaptive", seed=seed)
    S = tuple(sorted(int(j) for j in S))
    out = model.confidence_interval_topk(S_obs=S, Sigma=Sigma, alpha=ALPHA, k=K,
                                         epsilon=EPSILON, grid_size=500, seed=seed + 1,
                                         verbose=False)
    return S, {int(r["idx"]): (float(r["L"]), float(r["U"]))
               for recs in out["per_rank"].values() for r in recs}


# ---------------------------------------------------------------- re-runs
rng = np.random.default_rng(2026)
T = mu + se * rng.standard_normal((B, M))

rows, t0 = [], time.time()
for b in range(B):
    t = T[b]
    for name, (S, ci) in [("standard", standard(t)),
                          ("polyhedral", polyhedral(t)),
                          ("randomized", randomized(t, int(rng.integers(1e9))))]:
        rows += [(name, b, j, t[j], ci[j][0], ci[j][1]) for j in S]
    if (b + 1) % 200 == 0:
        print(f"  {b + 1}/{B}  ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows, columns=["method", "rep", "j", "est", "lo", "hi"])
df["mu"] = mu[df["j"].to_numpy()]
df["covered"] = (df["lo"] <= df["mu"]) & (df["mu"] <= df["hi"])
df["length"] = df["hi"] - df["lo"]

# ---------------------------------------------------------------- marginally valid comparators
# Same selections as `standard` (deterministic top-k), different interval widths.
d = df[df["method"] == "standard"].copy()
d["se"] = se[d["j"].to_numpy()]
c_cal = brentq(lambda c: np.mean(np.abs(d["est"] - d["mu"]) <= c * d["se"]) - (1 - ALPHA),
               1.0, 12.0)
z_fcr = norm.ppf(1 - (ALPHA * K / M) / 2)                  # Benjamini-Yekutieli (2005) FCR
d["cal_cov"] = np.abs(d["est"] - d["mu"]) <= c_cal * d["se"]
d["fcr_cov"] = np.abs(d["est"] - d["mu"]) <= z_fcr * d["se"]

# ---------------------------------------------------------------- output
print(f"\nMarginal coverage over all reported winners (nominal {1 - ALPHA:.2f}):")
for m in ["standard", "polyhedral", "randomized"]:
    g = df[df["method"] == m]
    print(f"  {m:<12s} {g['covered'].mean():.3f}   median length "
          f"{100 * g['length'].median():.2f} pp")
print(f"  {'calibrated':<12s} {d['cal_cov'].mean():.3f}   median length "
      f"{200 * c_cal * d['se'].median():.2f} pp   (c = {c_cal:.2f})")
print(f"  {'FCR-BY':<12s} {d['fcr_cov'].mean():.3f}   median length "
      f"{200 * z_fcr * d['se'].median():.2f} pp   (z = {z_fcr:.2f})")

print(f"\nConditional coverage by arm, arms ordered by TRUE effect (nominal {1 - ALPHA:.2f});"
      "\ncells with fewer than 25 selections are shown as (count).")
cols = ("arm", "mu(pp)", "se(pp)", "sel.freq", "standard", "polyhed", "random",
        "calibr", "FCR-BY", "bias(pp)")
header = "%-4s %7s %7s %9s %9s %8s %8s %8s %8s %9s" % cols
print(header)
print("-" * len(header))
for j in range(M):
    dj = d[d["j"] == j]
    line = "%-4d %7.2f %7.2f %9.3f" % (j, MU_PP[j], 100 * se[j], len(dj) / B)
    for m in ["standard", "polyhedral", "randomized"]:
        g = df[(df["method"] == m) & (df["j"] == j)]
        line += " %8s" % (f"{g['covered'].mean():.3f}" if len(g) >= 25 else f"({len(g)})")
    for col in ["cal_cov", "fcr_cov"]:
        line += " %8s" % (f"{dj[col].mean():.3f}" if len(dj) >= 25 else f"({len(dj)})")
    line += " %9s" % (f"{100 * (dj['est'] - dj['mu']).mean():+.2f}" if len(dj) >= 25 else "-")
    print(line)

# ---------------------------------------------------------------- consequence
top = d.loc[d.groupby("rep")["est"].idxmax()]
overstatement = 100 * (top["est"] - top["mu"]).mean()
print(f"\nReported best arm (standard): mean estimate {100 * top['est'].mean():.2f} pp "
      f"vs mean truth {100 * top['mu'].mean():.2f} pp"
      f"  ->  overstated by {overstatement:.2f} pp "
      f"(x{top['est'].mean() / top['mu'].mean():.2f})")

# Confirmatory two-arm trial sized for 80% power off the naive estimate of the winner.
n_plan = 2 * (ZCRIT + norm.ppf(0.80)) ** 2 * P0 * (1 - P0) / np.maximum(top["est"], 1e-6) ** 2
attained = norm.cdf(np.abs(top["mu"]) / np.sqrt(2 * P0 * (1 - P0) / n_plan) - ZCRIT)
print(f"A confirmatory trial sized for 80% power off that estimate attains median power "
      f"{100 * np.median(attained):.0f}% (mean {100 * attained.mean():.0f}%).")
