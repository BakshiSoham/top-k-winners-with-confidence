from pathlib import Path

import numpy as np

from simulation.Combined_Sim import (
    simulate_multiple_parameter_settings_compare_hat,
)

from plot.Plot_Function import (
    plot_four_panel_selection_inference_summary,
)



# ============================================================
# Signal multiplier helpers
# ============================================================

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def sampling_se_standardized_gap(theta, Sigma_hat, k):
    """
    Compute Δ_std = (theta_(k) - theta_(k+1)) / SE(theta_hat_(k) - theta_hat_(k+1)).

    theta: true score / target vector.
    Sigma_hat: covariance matrix of theta_hat, not raw-data covariance.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)

    order = np.argsort(theta)[::-1]
    j_k = int(order[k - 1])
    j_next = int(order[k])

    gap = float(theta[j_k] - theta[j_next])

    var_diff = (
        Sigma_hat[j_k, j_k]
        + Sigma_hat[j_next, j_next]
        - 2.0 * Sigma_hat[j_k, j_next]
    )
    se_diff = float(np.sqrt(max(var_diff, 1e-12)))

    return {
        "delta_gap": gap,
        "delta_std": gap / se_diff,
        "j_k": j_k,
        "j_next": j_next,
        "theta_k": float(theta[j_k]),
        "theta_next": float(theta[j_next]),
        "se_diff": se_diff,
    }

def calibrate_boundary_shift(delta_fn, target, lo=-0.05, hi=2.0,
                             tol=1e-6, max_iter=200):
    """
    Bisection: find the top-k block shift b such that delta_fn(b) == target.
    delta_fn(b) must be the standardized top-k gap delta_std as a function
    of b, monotone increasing on [lo, hi].
    """
    f_lo = delta_fn(lo) - target
    f_hi = delta_fn(hi) - target
    if f_lo * f_hi > 0:
        raise ValueError(
            f"target delta_std={target} not bracketed on [{lo}, {hi}]: "
            f"delta({lo})={f_lo + target:.3f}, delta({hi})={f_hi + target:.3f}"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = delta_fn(mid) - target
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


# Shared Low / High signal targets for the standardized top-k gap
# delta_std = (theta_(k) - theta_(k+1)) / SE(theta_hat_(k) - theta_hat_(k+1)).
# Matched across the Gaussian, Binomial and BTD examples.
#   Low  : delta_std = 0.3  -> selection unreliable (P(correct k/k+1 order) ~ 0.62)
#   High : delta_std = 2.0  -> selection reliable   (P(correct k/k+1 order) ~ 0.98)
# Data splitting selects on half the data, so its effective gap is
# delta_std / sqrt(2) (Low 0.21, High 1.41): its selection quality is
# strictly worse than full-data selection in BOTH settings.
DELTA_STD_TARGETS = [0.3, 2.0]
SIGNAL_LABELS     = ["Weak separation", "Strong separation"]




# ============================================================
# 9.1 Gaussian (M=20, k=3) — shared knobs
#   EPSILON is the log selection budget log(C(M, k)):
#   C(20, 3) = 1140 candidate top-k subsets  ->  log(1140) = 7.04
# ============================================================
EPSILON      = float(np.log(1140))          # = 7.04
EPSILON_LIST = [EPSILON]
METHODS      = ["Standard", "Randomized PSI", "Polyhedral PSI",
                "Data Splitting", "Zoom Correction"]
B            = 2000                        
ALPHA        = 0.05
SEED         = 123
GRID_SIZE    = 500
SPAN         = 6.0
K            = 3

Gaussian_Setting = dict(
    k=K, B=B, epsilon_list=EPSILON_LIST, methods=METHODS, alpha=ALPHA,
    seed=SEED, grid_size=GRID_SIZE, span=SPAN, verbose=True,
    share_same_data_across_methods=True, sigma="known", zoom_sigma_mode="mean",
)


M_g = 20
k = 3
n_obs_gaussian = 50

z_gaussian = np.zeros(M_g, dtype=float)
z_gaussian[0]    =  0.45
z_gaussian[1]    =  0.30
z_gaussian[2]    =  0.00       # high-variance decoy
z_gaussian[3]    =  0.25
z_gaussian[4:8]  =  0.15
z_gaussian[8:20] = -0.20

variances_g = np.ones(M_g, dtype=float)
variances_g[2]   = 10.0
variances_g[3:8] =  2.0
Sigma_g = np.diag(variances_g)

# ============================================================
# Gaussian baseline setting
#   Use b = 0 and condition on selecting index 2.
# ============================================================

b_gaussian = 0.0
mu_gaussian = z_gaussian.copy()

gap_info_gaussian = sampling_se_standardized_gap(
    theta=mu_gaussian,
    Sigma_hat=Sigma_g / float(n_obs_gaussian),
    k=k,
)

print(
    f"Gaussian baseline: b = {b_gaussian:+.4f}, "
    f"gap = {gap_info_gaussian['delta_gap']:.3f}, "
    f"delta_std = {gap_info_gaussian['delta_std']:.3f}"
)

parameter_settings_gaussian = [
    {
        "setting_id": 0,
        "label": "Baseline",
        "setting_label": "Baseline",
        "boundary_shift": float(b_gaussian),
        "delta_std": float(gap_info_gaussian["delta_std"]),
        "delta_gap": float(gap_info_gaussian["delta_gap"]),
        "mu": mu_gaussian,
        "Sigma": Sigma_g,
    }
]

def main():

    gaussian_out = simulate_multiple_parameter_settings_compare_hat(
        parameter_settings=parameter_settings_gaussian,
        n_obs=n_obs_gaussian,
        data_type="gaussian",
        **Gaussian_Setting,
    )

    four_panel_out = plot_four_panel_selection_inference_summary(
        gaussian_out,
        conditional_type="index",
        conditional_index=2,
        layout="1x4",
        epsilon_to_plot=EPSILON,
        x_axis="setting_label",
        hide_xtick_labels=True,
        rotate_xticks=0,
        marginal_coverage_ylim=(0.90, 0.99),
        conditional_coverage_ylim=(0.40, 1.02),
        normalize_regret_by_snoise=True,
        legend_mode="bottom",
        suptitle="",
    )

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figure_dir / "Fig1.pdf"

    four_panel_out["fig"].savefig(
        figure_path,
        bbox_inches="tight",
        dpi=300,
    )

    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()