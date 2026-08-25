from pathlib import Path

import numpy as np

from simulation.Combined_Sim import (
    simulate_multiple_parameter_settings_compare_hat,
)

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
# 9.2 Binomial / dosage (M=10, k=3) — shared knobs
#   EPSILON = log(C(10, 3)) = log(120) = 4.79
# ============================================================
EPSILON      = float(np.log(120))           # = 4.79
EPSILON_LIST = [EPSILON]
METHODS      = ["Standard", "Randomized PSI", "Polyhedral PSI",
                "Data Splitting", "Zoom Correction"]
B            = 500
ALPHA        = 0.05
SEED         = 123
GRID_SIZE    = 500
SPAN         = 6.0
K            = 3

Binomial_Setting = dict(
    k=K, B=B, epsilon_list=EPSILON_LIST, methods=METHODS, alpha=ALPHA,
    seed=SEED, grid_size=GRID_SIZE, span=SPAN, verbose=True,
    share_same_data_across_methods=True, sigma="unknown", zoom_sigma_mode="mean", #sigma="unknown"
)


# ============================================================
# 9.2 Binomial / dosage — Low / High signal three-panel
#   Success probabilities p = sigmoid(eta0 + z + b * 1{true top-k}),
#   selection score log(p).  Settings differ ONLY in the top-k block
#   shift b, calibrated so the standardized top-k gap (delta-method
#   SE of log p_hat, Var ~ (1-p)/(m p)) hits
#       Low Signal  : delta_std = 0.3
#       High Signal : delta_std = 2.0
#   exactly, matched to the Gaussian and BTD examples.
#   (Calibrated shifts: b ~ -0.007 for Low, b ~ +0.506 for High;
#    all p stay in [0.28, 0.48], so the Gaussian approximation used
#    by the CI engine remains accurate.)
# ============================================================
M_b = 10
k = 3
m_binom = 100
n_obs_binom = 100
eta0_binom = -1.2
z_binom = np.array([0.60, 0.50, 0.40, 0.30, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],dtype=float)
#z_binom=np.array([0.65, 0.60, 0.55,0.55, 0.55, 0.55, 0.55,0.55, 0.55, 0.55])

top_idx_b = np.argsort(z_binom)[::-1][:k]        # true top-k = {0, 1, 2}

def _p_binom(b):
    eta = eta0_binom + z_binom.copy()
    eta[top_idx_b] += b
    return sigmoid(eta)

def _delta_binom(b):
    p = _p_binom(b)
    Sigma_logp = np.diag((1.0 - p) / (float(m_binom) * p))    # delta-method
    return sampling_se_standardized_gap(
        theta=np.log(p), Sigma_hat=Sigma_logp, k=k,
    )["delta_std"]

def main():

    parameter_settings_binom = []

    for setting_id, (target, signal_label) in enumerate(
        zip(DELTA_STD_TARGETS, SIGNAL_LABELS)
    ):
        b = calibrate_boundary_shift(
            _delta_binom,
            target,
        )

        p_b = _p_binom(b)

        Sigma_logp = np.diag(
            (1.0 - p_b) / (float(m_binom) * p_b)
        )

        gap_info = sampling_se_standardized_gap(
            theta=np.log(p_b),
            Sigma_hat=Sigma_logp,
            k=k,
        )

        print(
            f"Binomial {signal_label:17s}: "
            f"b = {b:+.4f}, "
            f"gap = {gap_info['delta_gap']:.3f}, "
            f"delta_std = {gap_info['delta_std']:.3f}, "
            f"p in [{p_b.min():.3f}, {p_b.max():.3f}]"
        )

        parameter_settings_binom.append({
            "setting_id": setting_id,
            "label": signal_label,
            "setting_label": signal_label,
            "boundary_shift": float(b),
            "delta_std": float(gap_info["delta_std"]),
            "delta_gap": float(gap_info["delta_gap"]),
            "mu": p_b,
            "m": m_binom,
            "statistic": "probability",
            "covariance": "plugin",
        })

    binom_out = simulate_multiple_parameter_settings_compare_hat(
        parameter_settings=parameter_settings_binom,
        n_obs=n_obs_binom,
        data_type="binomial",
        **Binomial_Setting,
    )

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figure_dir / "Fig3.pdf"

    binom_panels = plot_three_panel_selection_inference_summary(
        binom_out,
        epsilon_to_plot=EPSILON,
        save_pdf=str(figure_path),
        x_axis="setting_label",
        marginal_coverage_ylim=(0.80, 1.02),
        suptitle="",
    )

    print(f"Figure saved to: {figure_path}")

    return {
        "simulation": binom_out,
        "plot": binom_panels,
    }


if __name__ == "__main__":
    main()

