import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from .simulation_utils import (
    _add_length_per_rep_outputs,
    _common_se_for_zoom,
    _default_utility_fn_for_data_type,
    _get_rep_stat_cov,
    _make_stat_records,
    _target_mu_from_config,
    _validate_data_type,
    _validate_sigma_mode,
    validate_data_samples,
    _common_se_from_covariance,
)



def stepdown_zoom_lower_radius(sorted_gaps_desc, alpha, sigma):
    Deltas = np.asarray(sorted_gaps_desc, dtype=float)
    m = len(Deltas)
    if m == 0:
        raise ValueError("sorted_gaps_desc cannot be empty.")

    # lower bound
    alpha_hat = alpha

    for k in range(m):
        r_hat_k = norm.isf(alpha_hat / (2 * (m - k)), scale=sigma)
        if Deltas[k] <= 4 * r_hat_k:
            r_l = r_hat_k
            break
        else:
            alpha_hat -= 2 * norm.sf((Deltas[k] - r_hat_k) / 3, scale=sigma)

    return float(r_l)


# =========================================================
# Algorithm 2:
# upper radius for winner
# =========================================================
def stepdown_zoom_upper_radius(sorted_gaps_desc, alpha, sigma):
    Deltas = np.asarray(sorted_gaps_desc, dtype=float)
    m = len(Deltas)

    if m == 0:
        raise ValueError("sorted_gaps_desc cannot be empty.")

    # upper bound
    alpha_hat = alpha

    for k in range(m):
        r_hat_k = norm.isf(alpha_hat / (2 * (m - k)), scale=sigma)

        if Deltas[k] <= 2 * r_hat_k:
            r_u = r_hat_k
            break
        else:
            alpha_hat -= 2 * norm.sf(
                (Deltas[k] + norm.isf(alpha / 2, scale=sigma)) / 3,
                scale=sigma,
            )

    return float(r_u)




def simulate_zoom_stepdown_hat(
    mu,
    Sigma,
    k,
    B,
    n_obs,
    alpha=0.05,
    *,
    utility_fn=None,
    seed=0,
    verbose=True,
    X_samples=None,
    sigma="known",
    zoom_sigma_mode="mean",
    data_type="gaussian",
    data_config=None,
):
    sigma = _validate_sigma_mode(sigma)
    data_type = _validate_data_type(data_type)

    mu_raw = np.asarray(mu, dtype=float).reshape(-1)
    utility_fn = _default_utility_fn_for_data_type(
        data_type=data_type,
        data_config=data_config,
        n_obs=n_obs,
        utility_fn=utility_fn,
    )
    
    target_mu = _target_mu_from_config(
        mu_raw,
        data_type=data_type,
        data_config=data_config,
        n_obs=n_obs,
    )
    M = mu_raw.size

    if data_type == "gaussian":
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.shape != (M, M):
            raise ValueError(f"Sigma must be {(M, M)}, got {Sigma.shape}")
        if n_obs < 2:
            raise ValueError("n_obs must be at least 2")
    else:
        Sigma = np.eye(M) if Sigma is None else np.asarray(Sigma, dtype=float)

    X_samples = validate_data_samples(X_samples, B, n_obs, M, data_type=data_type)

    if not (1 <= k <= M):
        raise ValueError(f"k must be in [1, M], got k={k}, M={M}")
    if B <= 0:
        raise ValueError("B must be positive")

    rng = np.random.default_rng(seed)

    ci_rows = []
    failures = []
    subset_records = []
    time_records = []
    stat_records = []

    for b in range(B):
        t0 = time.time()

        try:
            X_hat, Sigma_hat, sigma2_hat, observed_data, meta = _get_rep_stat_cov(
                b=b,
                rng=rng,
                mu=mu_raw,
                Sigma=Sigma,
                n_obs=n_obs,
                data_type=data_type,
                data_config=data_config,
                data_samples=X_samples,
                sigma=sigma,
            )

            if data_type == "gaussian":
                se_common, sigma2_hat_zoom = _common_se_for_zoom(
                    observed_data,
                    Sigma,
                    n_eff=n_obs,
                    sigma=sigma,
                    zoom_sigma_mode=zoom_sigma_mode,
                )
                sigma2_hat = sigma2_hat_zoom
            else:
                se_common, var_used = _common_se_from_covariance(
                    Sigma_hat,
                    zoom_sigma_mode=zoom_sigma_mode,
                )
                sigma2_hat = float(var_used)

            if se_common <= 0:
                raise ValueError(f"Non-positive common SE encountered: {se_common}")

            # For Zoom, record both the original coordinate-wise Sigma_hat diagonal
            # and the common Zoom variance.
            extra_zoom = dict(meta)
            extra_zoom.update({
                "se_common": float(se_common),
                "var_common": float(se_common ** 2),
                "zoom_sigma_mode": zoom_sigma_mode,
            })
            
            stat_records.extend(
                _make_stat_records(
                    method="Zoom Correction",
                    epsilon=None,
                    rep=b,
                    role="selection",
                    X_hat=X_hat,
                    Sigma_hat=Sigma_hat,
                    sigma2_hat=sigma2_hat,
                    sigma_mode=sigma,
                    data_type=data_type,
                    utility_fn=utility_fn,
                    extra=extra_zoom,
                )
            )

            scores = X_hat.copy()
            if utility_fn is not None:
                scores = np.asarray(utility_fn(X_hat), dtype=float)
                if scores.shape != (M,):
                    raise ValueError("utility_fn(X_hat) must return a vector of length M.")

            selected_idx = np.argsort(scores)[-k:][::-1]
            selected_idx = np.asarray(selected_idx, dtype=int)

            subset_records.append({
                "rep": b,
                "selected_subset": tuple(sorted(int(x) for x in selected_idx)),
            })

            if k == 1:
                idx = int(selected_idx[0])
                xhat = float(X_hat[idx])

                gaps = xhat - X_hat
                gaps_sorted = np.sort(gaps)[::-1]

                r_l = stepdown_zoom_lower_radius(gaps_sorted, alpha=alpha, sigma=se_common)
                r_u = stepdown_zoom_upper_radius(gaps_sorted, alpha=alpha, sigma=se_common)

                L = float(xhat - r_l)
                U = float(xhat + r_u)
                truth = float(target_mu[idx])

                ci_rows.append({
                    "rep": b,
                    "rank": 1,
                    "idx": idx,
                    "L": L,
                    "U": U,
                    "truth": truth,
                    "covered": bool(L <= truth <= U),
                    "length": float(U - L),
                    "radius_lower": float(r_l),
                    "radius_upper": float(r_u),
                    "se_common": float(se_common),
                    "sigma2_hat": float(sigma2_hat),
                    "sigma_mode": sigma,
                    "data_type": data_type,
                    **meta,
                })
            else:
                kth_idx = int(selected_idx[-1])
                x_k = float(X_hat[kth_idx])

                gaps_topk = np.maximum(x_k - X_hat, 0.0)
                gaps_topk_sorted = np.sort(gaps_topk)[::-1]

                r_max = stepdown_zoom_lower_radius(gaps_topk_sorted, alpha=alpha, sigma=se_common)

                r_first = norm.isf(alpha / (2 * M), scale=se_common)
                max_gap_topk = float(gaps_topk_sorted[0])
                first_step_stop = bool(max_gap_topk <= 4 * r_first)
                ratio_to_first = float(r_max / r_first)

                for rank, idx in enumerate(selected_idx, start=1):
                    idx = int(idx)

                    L = float(X_hat[idx] - r_max)
                    U = float(X_hat[idx] + r_max)
                    truth = float(target_mu[idx])

                    ci_rows.append({
                        "rep": b,
                        "rank": int(rank),
                        "idx": idx,
                        "L": L,
                        "U": U,
                        "truth": truth,
                        "covered": bool(L <= truth <= U),
                        "length": float(U - L),
                        "radius_lower": float(r_max),
                        "radius_upper": float(r_max),
                        "se_common": float(se_common),
                        "sigma2_hat": float(sigma2_hat),
                        "sigma_mode": sigma,
                        "r_first": float(r_first),
                        "max_gap_topk": float(max_gap_topk),
                        "first_step_stop": first_step_stop,
                        "ratio_to_first": float(ratio_to_first),
                        "data_type": data_type,
                        **meta,
                    })

            time_records.append({
                "method": "Zoom Correction",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": True,
                "data_type": data_type,
            })

        except Exception as e:
            time_records.append({
                "method": "Zoom Correction",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": False,
                "error_type": type(e).__name__,
                "data_type": data_type,
            })

            failures.append({"rep": b, "error": repr(e), "error_type": type(e).__name__})

            if verbose:
                print(f"\n[zoom_stepdown failure] rep={b}")
                print(type(e).__name__, repr(e))

        if verbose and (b + 1) % max(1, B // 20) == 0:
            print(f"{b+1}/{B}")

    ci_df = pd.DataFrame(ci_rows)
    fail_df = pd.DataFrame(failures)

    coverage_per_rep, length_per_rep_table, length_per_rep = _add_length_per_rep_outputs(
        ci_df,
        B=B,
        k=k,
    )

    return {
        "ci_df": ci_df,
        "failures": fail_df,
        "coverage_per_rep": coverage_per_rep,
        "length_per_rep_table": length_per_rep_table,
        "length_per_rep": length_per_rep,
        "alpha": alpha,
        "k": k,
        "B": B,
        "n_obs": n_obs,
        "sigma": sigma,
        "data_type": data_type,
        "subset_df": pd.DataFrame(subset_records),
        "stat_df": pd.DataFrame(stat_records),
        "time": pd.DataFrame(time_records),
    }
