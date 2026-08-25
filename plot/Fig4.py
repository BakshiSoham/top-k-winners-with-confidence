from pathlib import Path

import numpy as np

from simulation.Combined_Sim import (
    simulate_multiple_parameter_settings_compare_hat,
)

from simulation.data_generator import btd_mu_covariance

from plot.Plot_Function import (
    plot_three_panel_selection_inference_summary,
)

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
# 9.3 Bradley-Terry-Davidson (M=10, k=3) — shared knobs
#   EPSILON = log(C(10, 3)) = log(120) = 4.79
#   GRID_SIZE is larger than the other two examples: the BT log-strength
#   targets are more sensitive to grid resolution (see BTD grid
#   diagnostics in the Debug section).
# ============================================================
EPSILON      = float(np.log(120))           # = 4.79
EPSILON_LIST = [EPSILON]
METHODS      = ["Standard", "Randomized PSI", "Polyhedral PSI",
                "Data Splitting", "Zoom Correction"]
B            = 200
ALPHA        = 0.05
SEED         = 123
GRID_SIZE    = 2000
SPAN         = 6.0
K            = 3

BTD_Setting = dict(
    k=K, B=B, epsilon_list=EPSILON_LIST, methods=METHODS, alpha=ALPHA,
    seed=SEED, grid_size=GRID_SIZE, span=SPAN, verbose=True,
    share_same_data_across_methods=True, sigma="unknown", zoom_sigma_mode="mean",
)


# ============================================================
# 9.3 Bradley-Terry-Davidson — Low / High signal three-panel
#   mu = center(z + b * 1{true top-k}).  nu and n_matches are held
#   FIXED; settings differ ONLY in the top-k block shift b,
#   calibrated against the BTD Fisher-information covariance of
#   mu_hat so the standardized top-k gap hits
#       Low Signal  : delta_std = 0.3   (gap ~ 0.09, selection uncertain)
#       High Signal : delta_std = 2.0   (gap ~ 0.64, selection reliable)
#   exactly, matched to the Gaussian and Binomial examples.
#   Both settings share the same true top-3 set {0, 1, 2}.
#
#   z_bt uses the SAME shape as the Binomial example (separated top-3,
#   bunched tail).  A wide ladder (tail down to -0.9) makes the adaptive
#   randomization scale of the Gumbel top-k mechanism proportional to the
#   cross-sectional score spread, which -- with epsilon = log C(10,3) --
#   costs Randomized PSI more selection accuracy than data splitting's
#   sqrt(2).  Compressing the tail keeps randomized selection strictly
#   better than splitting in BOTH settings
#   (Gaussian-approx MC, P[select true top-3]:
#      Low : full 0.107, randomized 0.091, splitting 0.058
#      High: full 0.877, randomized 0.708, splitting 0.599).
# ============================================================
k = 3
nu_true_bt   = 0.2
n_matches_bt = 10     # keep >= 10: splitting selects on n_matches/2 games per
                      # pair, and the Gaussian approximation behind all CI
                      # methods (incl. Randomized PSI) needs the half-sample
                      # BTD MLE to be stable

def _center(a):
    a = np.asarray(a, dtype=float)
    return a - a.mean()            # BT log-strengths are identified up to a shift

z_bt = np.array([0.60, 0.50, 0.40, 0.30, 0.25, 0.24,0.23, 0.22, 0.21, 0.20], dtype=float)
#z_bt =np.array([0.65, 0.60, 0.55,0.55, 0.55, 0.55, 0.55,0.55, 0.55, 0.55])
M_bt = z_bt.size
top_idx_bt = np.argsort(z_bt)[::-1][:k]          # true top-k = {0, 1, 2}

# Round-robin design: every pair plays n_matches_bt games.
N_bt = n_matches_bt * (np.ones((M_bt, M_bt)) - np.eye(M_bt))

def _mu_bt(b):
    mu = z_bt.copy()
    mu[top_idx_bt] += b
    return _center(mu)

def _delta_bt(b):
    mu = _mu_bt(b)
    Sigma_mu = btd_mu_covariance(N_bt, mu, nu_true_bt)   # Fisher information
    return sampling_se_standardized_gap(theta=mu, Sigma_hat=Sigma_mu, k=k)["delta_std"]

def main():

    parameter_settings_bt = []

    for setting_id, (target, signal_label) in enumerate(
        zip(DELTA_STD_TARGETS, SIGNAL_LABELS)
    ):
        b = calibrate_boundary_shift(
            _delta_bt,
            target,
        )

        mu_b = _mu_bt(b)

        Sigma_mu = btd_mu_covariance(
            N_bt,
            mu_b,
            nu_true_bt,
        )

        gap_info = sampling_se_standardized_gap(
            theta=mu_b,
            Sigma_hat=Sigma_mu,
            k=k,
        )

        print(
            f"BTD {signal_label:17s}: "
            f"b = {b:+.4f}, "
            f"gap = {gap_info['delta_gap']:.3f}, "
            f"delta_std = {gap_info['delta_std']:.3f}"
        )

        parameter_settings_bt.append({
            "setting_id": setting_id,
            "label": signal_label,
            "setting_label": signal_label,
            "boundary_shift": float(b),
            "delta_std": float(gap_info["delta_std"]),
            "delta_gap": float(gap_info["delta_gap"]),
            "mu": mu_b,
            "nu_true": nu_true_bt,
            "n_matches": n_matches_bt,
        })

    bt_out = simulate_multiple_parameter_settings_compare_hat(
        parameter_settings=parameter_settings_bt,
        n_obs=n_matches_bt,
        data_type="bt_davidson",
        **BTD_Setting,
    )

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figure_dir / "Fig4.pdf"

    bt_panels = plot_three_panel_selection_inference_summary(
        bt_out,
        epsilon_to_plot=EPSILON,
        save_pdf=str(figure_path),
        x_axis="setting_label",
        marginal_coverage_ylim=(0.80, 1.02),
        suptitle="",
    )

    print(f"Figure saved to: {figure_path}")

    return {
        "simulation": bt_out,
        "plot": bt_panels,
    }


if __name__ == "__main__":
    main()
