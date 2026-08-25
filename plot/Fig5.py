from math import comb, log
from pathlib import Path

import numpy as np

from simulation.Combined_Sim import (
    simulate_multiple_parameter_settings_compare_hat,
)

from simulation.data_generator import (
    prepare_variable_importance_holdout_setting,
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

K = 3
z_vi_base = np.array([0.65, 0.60, 0.55,0.55, 0.55, 0.55, 0.55,0.55, 0.55, 0.55])

#z_vi_base =np.array([0.60, 0.50, 0.40, 0.30, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],dtype=float)



def make_vi_smooth_additive_f(beta):
    beta = np.asarray(beta, dtype=float).copy()
    def f(X):
        G = g_smooth_additive_vi(X)
        return G @ beta
    return f


def make_vi_smooth_additive_m_func(beta):
    beta = np.asarray(beta, dtype=float).copy()
    def m_func(X, cfg=None):
        G = g_smooth_additive_vi(X)
        return G @ beta
    return m_func


def make_vi_smooth_additive_m_minus_func(beta):
    beta = np.asarray(beta, dtype=float).copy()

    def m_minus_func(X, cfg=None):
        X = np.asarray(X, dtype=float)
        G = g_smooth_additive_vi(X)
        m = G @ beta
        n, p = X.shape
        out = np.empty((n, p), dtype=float)
        for j in range(p):
            out[:, j] = m - beta[j] * G[:, j]
        return out
    return m_minus_func


def true_vi_smooth_additive_target(beta, noise_sd=1.0):
    beta = np.asarray(beta, dtype=float)
    raw = beta ** 2
    var_O = float(np.sum(raw) + float(noise_sd) ** 2)
    return raw / var_O


def _I_positive(x):
    return np.where(x > 0.0, 1.0, 0.0)


def make_vi_indicator_f(beta):
    beta = np.asarray(beta, dtype=float).copy()
    def f(X):
        X = np.asarray(X, dtype=float)
        return _I_positive(X) @ beta
    return f


def g_smooth_additive_vi(X):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    G = np.zeros((n, p), dtype=float)
    for j in range(p):
        x = X[:, j]
        pattern = j % 5
        if pattern == 0:
            G[:, j] = np.sqrt(3.0) * x
        elif pattern == 1:
            G[:, j] = np.sqrt(5.0) * 0.5 * (3.0 * x**2 - 1.0)
        elif pattern == 2:
            G[:, j] = np.sqrt(7.0) * 0.5 * (5.0 * x**3 - 3.0 * x)
        elif pattern == 3:
            G[:, j] = np.sqrt(2.0) * np.sin(np.pi * x)
        elif pattern == 4:
            G[:, j] = np.sqrt(2.0) * np.sin(2.0 * np.pi * x)
    return G


def f_vi_smooth_additive(X, beta=z_vi_base):
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1)

    if X.shape[1] != beta.size:
        raise ValueError(
            f"X has p={X.shape[1]} columns, but beta has length {beta.size}."
        )

    G = g_smooth_additive_vi(X)
    return G @ beta


def make_vi_signal_function(beta):
    beta = np.asarray(beta, dtype=float).reshape(-1)

    def f_vi_scaled(X, beta=beta):
        X = np.asarray(X, dtype=float)
        G = g_smooth_additive_vi(X)
        return G @ beta

    return f_vi_scaled


def make_vi_true_m_func(beta):
    beta = np.asarray(beta, dtype=float).reshape(-1).copy()

    def m_func(X):
        X = np.asarray(X, dtype=float)
        G = g_smooth_additive_vi(X)
        return G @ beta

    return m_func


def make_vi_true_m_minus_func(beta):
    beta = np.asarray(beta, dtype=float).reshape(-1).copy()

    def m_minus_func(X):
        X = np.asarray(X, dtype=float)
        G = g_smooth_additive_vi(X)

        # m(X)
        m = G @ beta

        # Column j equals m_{-j}(X_{-j})
        return m[:, None] - G * beta[None, :]

    return m_minus_func


# ============================================================
# Unified Low / High signal calibration
# Same signal-strength definition as Gaussian / Binomial / BTD
# ============================================================

N_OBS_VI = 500
K = 3
SEED = 123
P_VI=10
z_vi_base = np.array([0.65, 0.60, 0.55,0.55, 0.55, 0.55, 0.55,0.55, 0.55, 0.55])

#z_vi_base =np.array([0.60, 0.50, 0.40, 0.30, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],dtype=float)


DELTA_STD_TARGETS = [0.3, 2.0]
SIGNAL_LABELS = ["Low Signal", "High Signal"]
top_idx_vi = np.arange(K)

def _beta_vi(b):
    beta = z_vi_base.copy()
    beta[top_idx_vi] += b
    return beta


def _delta_vi(b):
    beta_s = _beta_vi(b)
    setting = {
        "setting_id": -1,
        "label": "Calibration",
        "setting_label": "Calibration",
        "p": P_VI,
        "f": make_vi_signal_function(beta_s),
        "beta": beta_s,
        "x_dist": "uniform",
        "x_low": -1.0,
        "x_high": 1.0,
        "noise_sd": 1.0,
        "noise_dist": "normal",

        "tree_params": {
            "learner": "spline_gam",
            "learner_params": {
                "n_knots": 8,
                "degree": 3,
                "alpha": 1e-3,
            },
        },

        "vi_statistic": "fixed_nuisance",
        "covariance": "plugin",
    }

    prepared = prepare_variable_importance_holdout_setting(
        setting,
        n_obs=N_OBS_VI,
        holdout_n=50000,
        bootstrap_reps=100,
        bootstrap_n_obs=N_OBS_VI,
        seed=SEED,
        verbose=False,
    )

    gap = sampling_se_standardized_gap(
        theta=np.asarray(prepared["mu"]),
        Sigma_hat=np.asarray(prepared["Sigma"]) / float(N_OBS_VI),
        k=K,
    )

    return gap["delta_std"]
# ------------------------------------------------------------
# Final Low / High settings
# ------------------------------------------------------------
def main():
    parameter_settings_vi_holdout = []
    
    B_LO_VI = z_vi_base[K] - z_vi_base[K - 1] + 1e-4
    
    for setting_id, (target, signal_label) in enumerate(
        zip(DELTA_STD_TARGETS, SIGNAL_LABELS)
    ):
        b = calibrate_boundary_shift(
            _delta_vi,
            target,
            lo=B_LO_VI,
            hi=3.0,
        )
    
        beta_s = _beta_vi(b)
    
        pilot_delta = _delta_vi(b)
    
        print(
            f"{signal_label:11s}: "
            f"b={b:+.4f}, "
            f"target={target:.3f}, "
            f"pilot delta_std={pilot_delta:.3f}"
        )
    
        parameter_settings_vi_holdout.append({
            "setting_id": setting_id,
            "label": signal_label,
            "setting_label": signal_label,
    
            "boundary_shift": float(b),
            "target_delta_std": float(target),
    
            "p": P_VI,
            "f": make_vi_signal_function(beta_s),
            "beta": beta_s.copy(),
    
            "x_dist": "uniform",
            "x_low": -1.0,
            "x_high": 1.0,
            "noise_sd": 1.0,
            "noise_dist": "normal",
    
            "tree_params": {
                "learner": "spline_gam",
                "learner_params": {
                    "n_knots": 8,
                    "degree": 3,
                    "alpha": 1e-3,
                },
            },
    
            "holdout_n": 500_000,
            "bootstrap_reps": 500,
            "bootstrap_n_obs": N_OBS_VI,
    
            "vi_statistic": "fixed_nuisance",
            "covariance": "plugin",
        })
    
    prepared_vi_settings = []
    
    for s, setting_s in enumerate(parameter_settings_vi_holdout):
        prepared_s = prepare_variable_importance_holdout_setting(
            setting_s,
            n_obs=N_OBS_VI,
            holdout_n=setting_s["holdout_n"],
            bootstrap_reps=setting_s["bootstrap_reps"],
            bootstrap_n_obs=setting_s["bootstrap_n_obs"],
            seed=SEED + 10_000 * s,
            verbose=True,
        )
        prepared_setting = dict(prepared_s["setting"])
    
        final_gap = sampling_se_standardized_gap(
            theta=np.asarray(prepared_s["mu"], dtype=float),
            Sigma_hat=(
                np.asarray(prepared_s["Sigma"], dtype=float)
                / float(N_OBS_VI)
            ),
            k=K,
        )
        prepared_setting.update({
            "label": setting_s["label"],
            "setting_label": setting_s["setting_label"],
            "boundary_shift": float(setting_s["boundary_shift"]),
            "target_delta_std": float(setting_s["target_delta_std"]),
            "delta_std": float(final_gap["delta_std"]),
            "delta_gap": float(final_gap["delta_gap"]),
            "vi_statistic": "fixed_nuisance",
            "covariance": "plugin",
        })
        prepared_vi_settings.append(prepared_setting)
    
    
    
    
    
    EPSILON = log(comb(P_VI, K))
    EPSILON_LIST = [EPSILON]
    
    METHODS = [
        "Standard",
        "Randomized PSI",
        "Data Splitting",
        "Polyhedral PSI",
        "Zoom Correction",
    ]
    
    B = 500
    ALPHA = 0.05
    
    GRID_SIZE = 500
    SPAN = 6.0
    
    VI_Setting = dict(
        k=K,
        B=B,
        epsilon_list=EPSILON_LIST,
        methods=METHODS,
        alpha=ALPHA,
        seed=SEED,
        grid_size=GRID_SIZE,
        span=SPAN,
        verbose=True,
        share_same_data_across_methods=True,
        sigma="unknown",
        zoom_sigma_mode="mean",
    )
    
    vi_out_multi_signal = simulate_multiple_parameter_settings_compare_hat(
        parameter_settings=prepared_vi_settings,
        n_obs=N_OBS_VI,
        data_type="variable_importance",
        **VI_Setting,
    )

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    figure_path = figure_dir / "Fig5.pdf"
    
    
    vi_panels = plot_three_panel_selection_inference_summary(
        vi_out_multi_signal,
        epsilon_to_plot=EPSILON,
        save_pdf="VI_Example.pdf",
        x_axis="setting_label",
        marginal_coverage_ylim=(0.8, 1.02),
        suptitle="",
    )


if __name__ == "__main__":
    main()

