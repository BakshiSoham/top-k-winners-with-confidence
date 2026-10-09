"""Winner's-curse figures for the megastudy top-k experiment.

Design-based re-runs: a megastudy is summarised by arm-level counts (N_j, X_j); the
coverage target is the full-sample rate p_j = X_j/N_j, a KNOWN finite-population
parameter. Each re-run draws a smaller megastudy by sampling participants within arm
without replacement (hypergeometric on the counts), then selects the top-k arms and
builds intervals from the pilot alone.

Methods: Standard (naive), Data Splitting (sweep rho), Polyhedral PSI,
Randomized PSI (sweep q, with epsilon = q * log C(M, k)).

    python -m plot.winners_curse_figures --reps 150          # preview
    python -m plot.winners_curse_figures --reps 2000         # full

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


# ---------------------------------------------------------------- figure 1
def figure_winners_curse(df, truth, pop, path):
    reps = df.attrs["reps"]
    M = pop["M"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), width_ratios=[1.15, 1])
    order = np.argsort(-truth)

    # --- panel a: one re-run, as the analyst sees it
    rng = np.random.default_rng(5)
    n_arm, x_arm, n_hold, x_hold = draw_pilot(pop, PILOT_FRACTION, rng)
    theta, Sigma = pilot_estimates(pop, n_arm, x_arm, n_hold, x_hold)
    se = np.sqrt(np.diag(Sigma))
    S = m_standard(theta, Sigma)[0]

    ax = axes[0]
    shown = np.argsort(-theta)                     # the ranked list the analyst reads
    truly_top = set(np.argsort(-truth)[:K])
    y = np.arange(M)[::-1]
    for pos, j in zip(y, shown):
        sel = j in S
        col = C_OURS if sel else GREY
        ax.plot([100 * (theta[j] - ZCRIT * se[j]), 100 * (theta[j] + ZCRIT * se[j])],
                [pos, pos], color=col, lw=2.0 if sel else 1.1, solid_capstyle="round",
                zorder=3, alpha=1.0 if sel else 0.6)
        ax.plot(100 * theta[j], pos, "o", color=col, ms=5.0 if sel else 3.0, zorder=4)
        ax.plot(100 * truth[j], pos, "D", color=C_POLY, ms=3.4, zorder=5)
    ax.axvline(0, color=FAINT, lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([pop["labels"][j] for j in shown], fontsize=7.2)
    for tick, j in zip(ax.get_yticklabels(), shown):
        if j in truly_top:
            tick.set_color(C_POLY)
            tick.set_fontweight("bold")
    pad = 0.35
    ax.set_xlim(100 * (theta - ZCRIT * se).min() - pad,
                100 * (theta + ZCRIT * se).max() + pad)
    ax.set_xlabel("effect on vaccination rate (percentage points)")
    ax.set_title("a   One megastudy, as the analyst reads it", loc="left",
                 fontweight="bold", fontsize=10, pad=8)
    hit = len(set(S) & truly_top)
    ax.text(0.985, 0.105, "orange = the top 3 that get reported",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=C_OURS)
    ax.text(0.985, 0.055, "green ◆ and green labels = the truly best 3",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=C_POLY)
    ax.text(0.985, 0.003, f"this re-run reports {hit} of the 3 truly best arms",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=MUTED)

    # --- panel b: selection-induced bias, per arm
    ax = axes[1]
    d = df[df["method"] == "Standard"]
    xs, ys, sz, freq = [], [], [], []
    for j in range(M):
        dj = d[d["j"] == j]
        if len(dj) < max(8, reps // 150):
            continue
        xs.append(100 * truth[j])
        ys.append(100 * dj["reported"].mean())
        freq.append(len(dj) / reps)
        sz.append(12 + 240 * len(dj) / reps)
    gap = [y_ - x_ for x_, y_ in zip(xs, ys)]      # the overstatement itself
    ax.axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    for x_, g_ in zip(xs, gap):
        ax.plot([x_, x_], [0, g_], color=C_STD, lw=0.8, alpha=0.4, zorder=2)
    ax.scatter(xs, gap, s=sz, color=C_STD, alpha=0.55, edgecolor="white",
               linewidth=0.6, zorder=3)
    lo_i, hi_i = int(np.argmin(xs)), int(np.argmax(xs))
    ax.annotate(f"picked only {100 * freq[lo_i]:.0f}% of the time —\n"
                f"and overstated by {gap[lo_i]:.1f} pp when it is",
                (xs[lo_i], gap[lo_i]), textcoords="offset points", xytext=(16, -4),
                fontsize=8, color=MUTED, linespacing=1.35, va="top",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=2, shrinkB=6))
    ax.annotate(f"the truly best arm: picked {100 * freq[hi_i]:.0f}% of the time,\n"
                f"and still overstated by {gap[hi_i]:.1f} pp",
                (xs[hi_i], gap[hi_i]), textcoords="offset points", xytext=(-14, 26),
                fontsize=8, color=MUTED, ha="right", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=2, shrinkB=6))
    ax.set_xlim(min(xs) - 0.25, max(xs) + 0.25)
    ax.set_ylim(-0.12, max(gap) * 1.42)
    ax.text(0.985, 0.012, "honest reporting", transform=ax.transAxes, fontsize=8,
            color=INK, ha="right", va="bottom")
    ax.set_xlabel("true effect of the arm (pp)")
    ax.set_ylabel("overstatement: reported − true (pp)")
    ax.set_title("b   Every arm is overstated, and the weaker it is the worse",
                 loc="left", fontweight="bold", fontsize=10, pad=8)
    ax.text(0.985, 0.955, "bubble size = how often the arm is picked",
            transform=ax.transAxes, fontsize=8.2, color=MUTED, va="top", ha="right")

    fig.suptitle("Which nudge should we scale?   Top-3 arms of a vaccination megastudy",
                 x=0.012, ha="left", fontweight="bold", fontsize=12.5, y=1.045)
    fig.text(0.012, 0.985,
             f"{reps} design-based re-runs: each re-run draws {100 * PILOT_FRACTION:.0f}% of "
             f"participants within arm without replacement; the full-sample rates are the "
             f"ground truth; {int(100 * (1 - ALPHA))}% intervals.",
             ha="left", fontsize=8.6, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- figure 2
def figure_main(df, path):
    reps = df.attrs["reps"]
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.5), width_ratios=[1.05, 1.0, 1.0])

    groups = [("Standard", [""]), ("Data splitting", [f"ρ={r:g}" for r in RHO_GRID]),
              ("Polyhedral PSI", [""]), ("Randomized PSI (ours)",
                                        [f"q={q:g}" for q in Q_GRID])]

    # --- panel a: validity
    ax = axes[0]
    ax.axvspan(1 - ALPHA - 0.012, 1 - ALPHA + 0.012, color=FAINT, zorder=0)
    ax.axvline(1 - ALPHA, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ticks, ticklabels, y = [], [], 0.0
    for method, settings in groups:
        multi = len(settings) > 1
        y -= 0.95 if multi else 0.85
        if multi:                       # header row, then one row per setting
            ax.text(-0.345, y, method, transform=ax.get_yaxis_transform(), fontsize=9,
                    fontweight="bold", color=METHOD_COLOR[method], va="center")
        for s in settings:
            if multi:
                y -= 0.78
            d = df[(df["method"] == method) & (df["setting"] == s)]
            cov = d["covered"].mean()
            mc = np.sqrt(cov * (1 - cov) / len(d))
            hard = d[~d["truly_top"]]["covered"].mean()
            ax.plot([cov - 1.96 * mc, cov + 1.96 * mc], [y, y],
                    color=METHOD_COLOR[method], lw=1.6, solid_capstyle="round", zorder=4)
            ax.plot(cov, y, "o", color=METHOD_COLOR[method], ms=4.6, zorder=5)
            ax.text(1.015, y, f"{100 * hard:.0f}%", transform=ax.get_yaxis_transform(),
                    fontsize=8.2, color=MUTED, va="center")
            ticks.append(y)
            ticklabels.append((s, METHOD_COLOR[method], "normal") if multi
                              else (method, METHOD_COLOR[method], "bold"))
    ax.text(1.015, 0.62, "coverage when\nthe picked arm\nis not truly\ntop-3",
            transform=ax.get_yaxis_transform(), fontsize=7.6, color=MUTED,
            va="center", linespacing=1.45)
    ax.set_yticks(ticks)
    ax.set_yticklabels([t[0] for t in ticklabels], fontsize=8.4)
    for tick, (_, col, weight) in zip(ax.get_yticklabels(), ticklabels):
        tick.set_color(col)
        tick.set_fontweight(weight)
    ax.set_ylim(min(ticks) - 0.7, 0.4)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("coverage of the reported arms' true effects,\n"
                  f"averaged over {reps} re-runs")
    ax.set_title("a   Are the intervals valid?", loc="left", fontweight="bold",
                 fontsize=10.5, pad=10)
    ax.text(1 - ALPHA, 1.005, f"nominal {int(100 * (1 - ALPHA))}%",
            transform=ax.get_xaxis_transform(), fontsize=8.2, color=INK,
            ha="center", va="bottom")

    # --- panel b: precision vs selection quality
    ax = axes[1]
    curves = {}
    for method, settings in groups:
        pts = []
        for s in settings:
            d = df[(df["method"] == method) & (df["setting"] == s)]
            pts.append((100 * d.groupby("rep")["regret"].first().mean(),
                        100 * d["length"].mean(), s))
        curves[method] = pts

    # y-axis is set by the sane methods; polyhedral is drawn off-scale with a label
    on_scale = [p[1] for m, pts in curves.items() if m != "Polyhedral PSI" for p in pts]
    ytop = max(on_scale) * 1.18
    for method, pts in curves.items():
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        col = METHOD_COLOR[method]
        if method == "Standard":
            ax.plot(xs, ys, "o", mfc="white", mec=INK, mew=1.4, ms=8, zorder=6)
            ax.annotate("Standard: shortest, but invalid", (xs[0], ys[0]),
                        textcoords="offset points", xytext=(10, -3), fontsize=8.4,
                        color=INK)
        elif method == "Polyhedral PSI":
            ax.plot(xs[0], ytop * 0.97, "^", color=col, ms=8, zorder=6, clip_on=False)
            ax.annotate(f"Polyhedral PSI: {ys[0]:.0f} pp, off scale",
                        (xs[0], ytop * 0.97), textcoords="offset points",
                        xytext=(9, -2), fontsize=8.4, color=col)
        else:
            ax.plot(xs, ys, "-o", color=col, ms=5, lw=1.5, zorder=5)
        for x_, y_, s in pts:
            if s:
                ax.annotate(s, (x_, y_), textcoords="offset points", xytext=(6, 5),
                            fontsize=7.8, color=col)

    # callout: a setting of ours that dominates data splitting on BOTH axes
    ds = sorted(curves["Data splitting"], key=lambda p: p[0])[0]      # best-regret split
    dominating = [p for p in curves["Randomized PSI (ours)"]
                  if p[0] <= ds[0] and p[1] <= ds[1]]
    if dominating:
        ours = min(dominating, key=lambda p: p[1])
        ax.annotate("", xy=(ours[0], ours[1]), xytext=(ds[0], ds[1]),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                                    shrinkA=5, shrinkB=5), zorder=7)
        ax.text(0.97, 0.955,
                f"{ours[2]} beats every data-splitting\nsetting on both axes: a better pick"
                f"\nand intervals {ds[1] / ours[1]:.1f}× shorter",
                transform=ax.transAxes, fontsize=8.3, color=INK, ha="right", va="top",
                linespacing=1.4)

    ax.set_ylim(min(on_scale) * 0.80, ytop)
    ax.set_xlabel("selection quality: regret of the chosen top-3 (pp)\n"
                  "← better: lower regret, shorter intervals")
    ax.set_ylabel("precision: mean interval length (pp)")
    ax.set_title("b   How precise, at what cost to the pick?", loc="left",
                 fontweight="bold", fontsize=10.5, pad=10)
    for i, (method, _) in enumerate(g for g in groups if g[0] != "Standard"):
        ax.text(0.985, 0.655 - 0.062 * i, method, transform=ax.transAxes, ha="right",
                fontsize=8.8, color=METHOD_COLOR[method], fontweight="bold")

    # --- panel c: the floor you would budget on.
    # All full-data methods quote the same point estimate, so the point estimate cannot
    # separate them. What separates them is the interval's lower endpoint: for a valid
    # 1-alpha two-sided interval it should sit above the truth only alpha/2 of the time.
    # Scored over ALL reported arms -- the guarantee is conditional on the selected SET,
    # so restricting to the rank-1 arm would ask for conditioning the method never claims.
    ax = axes[2]
    show = [("Standard", ""), ("Data splitting", "ρ=0.5"), ("Polyhedral PSI", ""),
            ("Randomized PSI (ours)", "q=1"), ("Randomized PSI (ours)", "q=2")]
    rng = np.random.default_rng(3)
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    floors = {}
    for method, s in show:
        d = df[(df["method"] == method) & (df["setting"] == s) & (~df["ci_failed"])]
        floors[(method, s)] = 100 * (d["lo"] - d["truth"]).to_numpy()
    # Polyhedral intervals have very heavy left tails, so the window comes from the
    # others and the summaries are medians/quartiles rather than means.
    ref = np.concatenate([v for key, v in floors.items() if key[0] != "Polyhedral PSI"])
    xlo = float(np.percentile(ref, 0.5)) * 1.25
    xhi = max(float(np.percentile(ref, 99.9)), 0.0) + 0.45 * (0 - xlo) * 0.25

    for i, (method, s) in enumerate(show):
        y = -i
        col = METHOD_COLOR[method]
        floor = floors[(method, s)]
        vis = floor[(floor >= xlo) & (floor <= xhi)]
        ax.scatter(vis, y + rng.uniform(-0.22, 0.22, len(vis)), s=2.4, color=col,
                   alpha=0.20, linewidths=0, zorder=3)
        q1, med, q3 = np.percentile(floor, [25, 50, 75])
        ax.plot([max(q1, xlo), min(q3, xhi)], [y, y], color=col, lw=2.2,
                solid_capstyle="round", zorder=5)
        ax.plot(med, y, "o", color=col, ms=6.5, zorder=6)
        off = float(np.mean(floor < xlo))
        if off > 0.005:
            ax.annotate(f"{100 * off:.0f}% off scale ←", (xlo, y),
                        textcoords="offset points", xytext=(6, 11), fontsize=7.4,
                        color=col)
        share = float(np.mean(floor > 0))
        ax.text(1.015, y + 0.17, f"{100 * share:.0f}%", transform=ax.get_yaxis_transform(),
                fontsize=8.8, color=col if share > 0.08 else MUTED,
                fontweight="bold", va="center")
        ax.text(1.015, y - 0.2, f"floor beats truth", transform=ax.get_yaxis_transform(),
                fontsize=7.6, color=MUTED, va="center")
        ax.text(-0.02, y, (method if not s else f"{method}, {s}"),
                transform=ax.get_yaxis_transform(), fontsize=8.8, color=col,
                ha="right", va="center",
                fontweight="bold" if "ours" in method or method == "Standard" else "normal")
    ax.set_yticks([]); ax.set_ylim(-len(show) + 0.45, 0.6)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("interval's lower end − true effect, each reported arm (pp)\n"
                  f"dots = {reps} re-runs × {K} reported arms, bar = median and quartiles")
    ax.set_title("c   Would the gain you budget on survive?", loc="left",
                 fontweight="bold", fontsize=10.5, pad=10)
    ax.text(xlo + 0.03 * (xhi - xlo), -len(show) + 0.62, "conservative floor",
            fontsize=8, color=MUTED)
    ax.text(xhi - 0.03 * (xhi - xlo), -len(show) + 0.62,
            "floor over-promises", fontsize=8, color=MUTED, ha="right")
    ax.text(0.5, 1.005, f"a valid {int(100 * (1 - ALPHA))}% interval should over-promise "
            f"{int(100 * ALPHA / 2)}% of the time",
            transform=ax.transAxes, fontsize=8.2, color=MUTED, ha="center")

    fig.suptitle("Randomized selection buys conditional validity without paying for it twice",
                 x=0.008, ha="left", fontweight="bold", fontsize=13, y=1.055)
    fig.text(0.008, 0.995,
             f"Vaccination megastudy, {reps} design-based re-runs of a "
             f"{100 * PILOT_FRACTION:.0f}% participant subsample; top-k = {K} of 22 arms; "
             f"{int(100 * (1 - ALPHA))}% intervals; full-sample rates are ground truth.",
             ha="left", fontsize=8.8, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pop = build_population()
    truth = finite_population_truth(pop)
    print(f"megastudy: {pop['M']} arms, N = {pop['N_arm'].sum() + pop['N_holdout']:,}; "
          f"true effects {100 * truth.max():.2f} to {100 * truth.min():.2f} pp")

    df, truth = run(pop, args.reps)
    tag = "preview" if args.reps < 500 else "full"
    df.to_csv(os.path.join(args.outdir, f"megastudy_runs_{tag}.csv"), index=False)
    figure_winners_curse(df, truth, pop,
                         os.path.join(args.outdir, f"megastudy_curse_{tag}.png"))
    figure_main(df, os.path.join(args.outdir, f"megastudy_main_{tag}.png"))

    print(f"\n{'':<28s} {'cover':>6s} {'length':>7s} {'regret':>7s} {'CI fail':>8s}")
    for lab, d in df.groupby("label"):
        print(f"  {lab:<28s} {d['covered'].mean():6.3f} "
              f"{100 * d['length'].mean():7.2f} "
              f"{100 * d.groupby('rep')['regret'].first().mean():7.3f} "
              f"{d['ci_failed'].mean():8.3f}")
