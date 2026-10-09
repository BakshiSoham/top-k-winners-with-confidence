"""Winner's-curse figures for the megastudy top-k experiment.

Design-based re-runs: a megastudy is summarised by arm-level counts (N_j, X_j); the
coverage target is the full-sample rate p_j = X_j/N_j, a KNOWN finite-population
parameter. Each re-run draws a smaller megastudy by sampling participants within arm
without replacement (hypergeometric on the counts), then selects the top-k arms and
builds intervals from the pilot alone.

Methods: Standard (naive), Data Splitting (sweep rho), Polyhedral PSI,
Randomized PSI (sweep q, with epsilon = q * log C(M, k)).

    python -m plot.winners_curse_figures --reps 300          # preview (~1 min)
    python -m plot.winners_curse_figures --reps 2000         # full (~7 min)
    python -m plot.winners_curse_figures --reps 300 --replot # redraw only

Swap in a real megastudy by replacing build_population() with the published counts.
"""

import argparse
import os
import sys
import time
import warnings
from math import comb, log

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
if not hasattr(np, "trapz"):           # NumPy >= 2.0 renamed it; the repo still calls trapz
    np.trapz = np.trapezoid

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from scipy.stats import norm                                            # noqa: E402

from src import PolyhedralTopKInference, TopKSelectionModel             # noqa: E402

# ---------------------------------------------------------------- design constants
K = 3
ALPHA = 0.10
PILOT_FRACTION = 0.10
Q_GRID = (0.5, 1.0, 2.0, 4.0)          # randomized PSI: epsilon = q * log C(M, k)
RHO_GRID = (0.3, 0.5, 0.7)             # data splitting: share of pilot used to select
ZCRIT = norm.ppf(1 - ALPHA / 2)
utility = lambda x: np.asarray(x, dtype=float)                          # noqa: E731

# ---------------------------------------------------------------- palette (matches main_figure)
INK, MUTED, GREY, FAINT = "#222222", "#6b6b6b", "#a3a3a3", "#dcdcdc"
C_STD, C_SPLIT, C_POLY, C_OURS = "#3f3f3f", "#2C6FBB", "#2E9E74", "#E0552B"
METHOD_COLOR = {"Standard": C_STD, "Data splitting": C_SPLIT,
                "Polyhedral PSI": C_POLY, "Randomized PSI (ours)": C_OURS}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": "#333333",
    "text.color": INK, "pdf.fonttype": 42, "savefig.dpi": 220,
})


# ---------------------------------------------------------------- the megastudy
def build_population(seed=11):
    """Arm-level counts for a Walmart-scale flu-vaccination megastudy.

    Replace this function with the published (N_j, X_j) counts to run on real data;
    nothing downstream changes.
    """
    rng = np.random.default_rng(seed)
    effects_pp = np.array([1.80, 1.55, 1.40, 1.25, 1.15, 1.05, 0.95, 0.85, 0.75, 0.65,
                           0.55, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
                           0.00, -0.15])
    M = len(effects_pp)
    p_holdout = 0.30
    N_arm = rng.integers(26_000, 34_000, M)
    N_holdout = 60_000
    X_arm = rng.binomial(N_arm, p_holdout + effects_pp / 100.0)
    X_holdout = rng.binomial(N_holdout, p_holdout)
    return dict(M=M, N_arm=N_arm, X_arm=X_arm, N_holdout=N_holdout, X_holdout=X_holdout,
                labels=[f"A{j + 1}" for j in range(M)])


def finite_population_truth(pop):
    """theta_j = p_j - p_0 from the FULL sample. Known exactly; this is the target."""
    return pop["X_arm"] / pop["N_arm"] - pop["X_holdout"] / pop["N_holdout"]


def draw_pilot(pop, fraction, rng):
    """Sample participants within arm without replacement -> a smaller megastudy."""
    n_arm = np.ceil(fraction * pop["N_arm"]).astype(int)
    x_arm = rng.hypergeometric(pop["X_arm"], pop["N_arm"] - pop["X_arm"], n_arm)
    n_hold = int(np.ceil(fraction * pop["N_holdout"]))
    x_hold = rng.hypergeometric(pop["X_holdout"], pop["N_holdout"] - pop["X_holdout"],
                                n_hold)
    return n_arm, x_arm, n_hold, x_hold


def wor_variance(x, n, N):
    """Variance of a rate from a without-replacement sample of n from N."""
    p = np.clip(x / n, 1e-6, 1 - 1e-6)
    return (1.0 - n / N) * p * (1.0 - p) / n * N / np.maximum(N - 1, 1)


def pilot_estimates(pop, n_arm, x_arm, n_hold, x_hold):
    """theta_hat and Sigma_hat. The shared holdout makes Sigma equicorrelated."""
    theta = x_arm / n_arm - x_hold / n_hold
    var_hold = wor_variance(x_hold, n_hold, pop["N_holdout"])
    Sigma = np.diag(wor_variance(x_arm, n_arm, pop["N_arm"])) + var_hold
    return theta, Sigma


# ---------------------------------------------------------------- methods
# Each method returns (S, ci, reported, stat): `reported` is the point estimate the
# analyst would quote, `stat` is the statistic the method ranked on -- these differ for
# data splitting, and ranking on the wrong one re-introduces the selection effect.
def m_standard(theta, Sigma):
    se = np.sqrt(np.diag(Sigma))
    S = tuple(sorted(np.argsort(-theta)[:K]))
    ci = {int(j): (theta[j] - ZCRIT * se[j], theta[j] + ZCRIT * se[j]) for j in S}
    return S, ci, {int(j): float(theta[j]) for j in S}, {int(j): float(theta[j]) for j in S}


def m_polyhedral(theta, Sigma):
    model = PolyhedralTopKInference(X=theta, k=K, H0_mu=np.zeros_like(theta), Sigma=Sigma,
                                    utility_fn=utility, grid_size=500, alpha=ALPHA)
    recs = model.confidence_interval_topk(alpha=ALPHA)
    S = tuple(sorted(int(j) for j in model.selected_set))
    ci = {int(r["index"]): (float(r["ci_lower"]), float(r["ci_upper"])) for r in recs}
    return S, ci, {int(j): float(theta[j]) for j in S}, {int(j): float(theta[j]) for j in S}


def m_randomized(theta, Sigma, q, seed):
    epsilon = q * log(comb(len(theta), K))
    model = TopKSelectionModel(X=theta, k=K, H0_mu=np.zeros_like(theta), true_Sigma=Sigma,
                               utility_fn=utility, epsilon=epsilon, grid_size=500,
                               sel_scale="adaptive")
    S, _ = model.randomized_selected_top_k(X=theta, k=K, epsilon=epsilon,
                                           scale="adaptive", seed=seed)
    S = tuple(sorted(int(j) for j in S))
    out = model.confidence_interval_topk(S_obs=S, Sigma=Sigma, alpha=ALPHA, k=K,
                                         epsilon=epsilon, grid_size=500, seed=seed + 1,
                                         verbose=False)
    # The grid-based bisection can fail to bracket a root, in which case the repo
    # returns L and/or U as None. simulation/Randomized_PSI_Sim.py scores such a
    # record as not-covered with NaN length; we follow that convention and count it.
    ci = {}
    for recs in out["per_rank"].values():
        for r in recs:
            L, U = r.get("L"), r.get("U")
            ok = L is not None and U is not None and np.isfinite(L) and np.isfinite(U)
            ci[int(r["idx"])] = (float(L), float(U)) if ok else (np.nan, np.nan)
    return S, ci, {int(j): float(theta[j]) for j in S}, {int(j): float(theta[j]) for j in S}


def m_data_splitting(pop, n_arm, x_arm, n_hold, x_hold, rho, rng):
    """Split the pilot's participants: select on rho, infer on the rest."""
    nA = np.maximum(np.ceil(rho * n_arm).astype(int), 1)
    xA = rng.hypergeometric(x_arm, n_arm - x_arm, nA)
    nB, xB = n_arm - nA, x_arm - xA
    nhA = max(int(np.ceil(rho * n_hold)), 1)
    xhA = rng.hypergeometric(x_hold, n_hold - x_hold, nhA)
    nhB, xhB = n_hold - nhA, x_hold - xhA

    thetaA = xA / nA - xhA / nhA
    S = tuple(sorted(np.argsort(-thetaA)[:K]))
    thetaB = xB / nB - xhB / nhB
    seB = np.sqrt(wor_variance(xB, nB, pop["N_arm"]) + wor_variance(xhB, nhB, pop["N_holdout"]))
    ci = {int(j): (thetaB[j] - ZCRIT * seB[j], thetaB[j] + ZCRIT * seB[j]) for j in S}
    # Reported value comes from split B; the ranking was done on split A.
    return S, ci, {int(j): float(thetaB[j]) for j in S}, {int(j): float(thetaA[j]) for j in S}


# ---------------------------------------------------------------- experiment
def run(pop, reps, seed=2026):
    truth = finite_population_truth(pop)
    best = set(np.argsort(-truth)[:K])
    oracle_utility = truth[sorted(best)].sum()
    rng = np.random.default_rng(seed)

    rows, t0 = [], time.time()
    for r in range(reps):
        n_arm, x_arm, n_hold, x_hold = draw_pilot(pop, PILOT_FRACTION, rng)
        theta, Sigma = pilot_estimates(pop, n_arm, x_arm, n_hold, x_hold)

        runs = [("Standard", "", *m_standard(theta, Sigma)),
                ("Polyhedral PSI", "", *m_polyhedral(theta, Sigma))]
        for q in Q_GRID:
            runs.append(("Randomized PSI (ours)", f"q={q:g}",
                         *m_randomized(theta, Sigma, q, int(rng.integers(1e9)))))
        for rho in RHO_GRID:
            runs.append(("Data splitting", f"ρ={rho:g}",
                         *m_data_splitting(pop, n_arm, x_arm, n_hold, x_hold, rho, rng)))

        for method, setting, S, ci, reported, stat in runs:
            regret = oracle_utility - truth[list(S)].sum()
            for rank, j in enumerate(sorted(S, key=lambda i: -stat[int(i)])):
                lo, hi = ci[j]
                rows.append((method, setting, r, int(j), rank, reported[int(j)], lo, hi,
                             truth[j], regret, int(j) in best))
        if (r + 1) % max(1, reps // 10) == 0:
            print(f"  {r + 1}/{reps}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows, columns=["method", "setting", "rep", "j", "rank", "reported",
                                     "lo", "hi", "truth", "regret", "truly_top"])
    df["ci_failed"] = ~np.isfinite(df["lo"]) | ~np.isfinite(df["hi"])
    df["covered"] = (~df["ci_failed"]) & (df["lo"] <= df["truth"]) & (df["truth"] <= df["hi"])
    df["length"] = np.where(df["ci_failed"], np.nan, df["hi"] - df["lo"])
    df["bias"] = df["reported"] - df["truth"]
    df["label"] = np.where(df["setting"] == "", df["method"],
                           df["method"] + ", " + df["setting"])
    df.attrs["reps"] = reps
    return df, truth




# ---------------------------------------------------------------- the figure
# One message per panel:
#   a  the winner's curse: picked arms are overstated, the weak ones most;
#   b  so naive intervals fail on exactly those picks, ours do not;
#   c  and ours is the shortest valid interval at a given selection quality.
OURS_Q = "q=1"                          # the paper's default regret budget
RANK_BINS = [(1, 3, "1–3\n(deserved)"), (4, 6, "4–6"),
             (7, 10, "7–10"), (11, 99, "11+")]


def _true_rank(truth):
    r = np.empty(len(truth), dtype=int)
    r[np.argsort(-truth)] = np.arange(1, len(truth) + 1)
    return r


def _panel_title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontweight="bold", fontsize=10.5,
                 pad=10)


def figure_summary(df, truth, path):
    reps = df.attrs["reps"]
    rank = _true_rank(truth)
    df = df.assign(true_rank=rank[df["j"].to_numpy()])
    naive = df[df["method"] == "Standard"]
    ours = df[(df["method"] == "Randomized PSI (ours)") & (df["setting"] == OURS_Q)]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # --- a: the winner's curse
    ax = axes[0]
    g = naive.groupby("j").agg(n=("bias", "size"), bias=("bias", "mean"))
    g = g[g["n"] >= max(15, reps // 25)]
    x = 100 * truth[g.index.to_numpy()]
    y = 100 * g["bias"].to_numpy()
    ax.axhline(0, color=GREY, lw=0.9, zorder=1)
    ax.vlines(x, 0, y, color=C_STD, lw=1.0, alpha=0.35, zorder=2)
    ax.plot(x, y, "o", color=C_STD, ms=6, mec="white", mew=0.8, zorder=3)
    ax.set_xlabel("true effect of the arm (pp)")
    ax.set_ylabel("reported − true, when picked (pp)")
    ax.set_ylim(-0.1, max(y) * 1.25)
    _panel_title(ax, "a", "Picked arms are overstated")

    # --- b: conditional coverage, by how deserved the pick was
    ax = axes[1]
    ax.axhline(1 - ALPHA, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
    xs = np.arange(len(RANK_BINS))
    for d, col, name in [(naive, C_STD, "naive"), (ours, C_OURS, "ours")]:
        cov, err = [], []
        for lo, hi, _ in RANK_BINS:
            c = d[(d["true_rank"] >= lo) & (d["true_rank"] <= hi)]["covered"]
            cov.append(c.mean())
            err.append(1.96 * np.sqrt(c.mean() * (1 - c.mean()) / max(len(c), 1)))
        ax.errorbar(xs, cov, yerr=err, color=col, lw=2, ms=6.5, marker="o",
                    mec="white", mew=0.8, capsize=0, elinewidth=1.1, zorder=3)
        ax.text(xs[-1] + 0.12, cov[-1], name, color=col, fontsize=9.5,
                fontweight="bold", va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[2] for b in RANK_BINS])
    ax.set_xlim(-0.3, len(RANK_BINS) - 0.35)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "90%", "100%"])
    ax.set_xlabel("true rank of the picked arm")
    ax.set_ylabel("coverage when picked")
    _panel_title(ax, "b", "Naive intervals fail on lucky picks")

    # --- c: price of validity
    ax = axes[2]
    pts = {}
    for (method, setting), d in df.groupby(["method", "setting"]):
        pts.setdefault(method, []).append(
            (100 * d.groupby("rep")["regret"].first().mean(), 100 * d["length"].mean()))
    for method in ["Data splitting", "Randomized PSI (ours)"]:
        p = sorted(pts[method])
        ax.plot([a for a, _ in p], [b for _, b in p], "-o", color=METHOD_COLOR[method],
                lw=2, ms=5.5, mec="white", mew=0.8, zorder=3)
    on_scale = [b for m in ("Data splitting", "Randomized PSI (ours)", "Standard")
                for _, b in pts[m]]
    ytop = max(on_scale) * 1.22
    nx, ny = pts["Standard"][0]
    ax.plot(nx, ny, "o", mfc="white", mec=C_STD, mew=1.5, ms=7.5, zorder=4)
    ax.annotate("naive (invalid)", (nx, ny), textcoords="offset points", xytext=(8, -3),
                fontsize=8.5, color=MUTED, va="center")
    px, py = pts["Polyhedral PSI"][0]
    ax.annotate(f"polyhedral: {py:.0f} pp ↑", (px, ytop), textcoords="offset points",
                xytext=(0, -12), fontsize=8.5, color=C_POLY, fontweight="bold")
    p_ds = sorted(pts["Data splitting"])
    p_us = sorted(pts["Randomized PSI (ours)"])
    ax.text(p_ds[-1][0], p_ds[-1][1], "  data splitting", color=C_SPLIT, fontsize=9.5,
            fontweight="bold", va="center")
    ax.text(p_us[-1][0], p_us[-1][1], "  ours", color=C_OURS, fontsize=9.5,
            fontweight="bold", va="top")
    ax.set_ylim(min(on_scale) * 0.75, ytop)
    ax.set_xlabel("regret of the chosen top-3 (pp)   ← better pick")
    ax.set_ylabel("mean interval length (pp)")
    _panel_title(ax, "c", "Ours: shortest valid interval")

    fig.suptitle("Which nudge should we scale?  Top 3 of 22 arms, "
                 f"{reps} re-runs of a 10% megastudy, 90% intervals",
                 x=0.01, ha="left", fontsize=11, color=MUTED, y=1.02)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved runs CSV instead of re-running")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = "preview" if args.reps < 1000 else "full"
    csv = os.path.join(args.outdir, f"megastudy_runs_{tag}.csv")
    pop = build_population()
    truth = finite_population_truth(pop)

    if args.replot:
        df = pd.read_csv(csv)
        df.attrs["reps"] = int(df["rep"].max()) + 1
    else:
        df, truth = run(pop, args.reps)
        df.to_csv(csv, index=False)
    figure_summary(df, truth, os.path.join(args.outdir, f"megastudy_summary_{tag}.png"))
    figure_summary(df, truth, os.path.join(args.outdir, f"megastudy_summary_{tag}.pdf"))
