import time
from statistics import NormalDist

import numpy as np
import pandas as pd

from .simulation_utils import (
    _add_length_per_rep_outputs,
    _default_utility_fn_for_data_type,
    _get_data_splitting_stat_cov,
    _make_stat_records,
    _target_mu_from_config,
    _validate_data_type,
    _validate_sigma_mode,
    validate_data_samples,
)
def simulate_data_splitting_hat(
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
            raise ValueError("n_obs must be at least 2 for data splitting.")
    else:
        Sigma = np.eye(M) if Sigma is None else np.asarray(Sigma, dtype=float)

    X_samples = validate_data_samples(X_samples, B, n_obs, M, data_type=data_type)

    if not (1 <= k <= M):
        raise ValueError(f"k must be in [1, M], got k={k}, M={M}")
    if B <= 0:
        raise ValueError("B must be positive")

    rng = np.random.default_rng(seed)
    zcrit = float(NormalDist().inv_cdf(1 - alpha / 2))

    ci_rows = []
    failures = []
    subset_records = []
    time_records = []
    stat_records = []

    for b in range(B):
        t0 = time.time()

        try:
            ds = _get_data_splitting_stat_cov(
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

            X_hat_sel = ds["X_hat_sel"]
            X_hat_inf = ds["X_hat_inf"]
            Sigma_hat_inf = ds["Sigma_hat_inf"]
            sigma2_hat_inf = ds["sigma2_hat_inf"]

            # Statistic used for selection quality.
            stat_records.extend(
                _make_stat_records(
                    method="Data Splitting",
                    epsilon=None,
                    rep=b,
                    role="selection",
                    X_hat=X_hat_sel,
                    Sigma_hat=ds.get("Sigma_hat_sel", None),  # may be unavailable
                    sigma2_hat=sigma2_hat_inf,
                    sigma_mode=sigma,
                    data_type=data_type,
                    utility_fn=utility_fn,
                    extra={
                        "n_obs_sel": int(ds["n_obs_sel"]),
                        "n_obs_inf": int(ds["n_obs_inf"]),
                    },
                )
            )
            
            # Statistic used for inference intervals.
            stat_records.extend(
                _make_stat_records(
                    method="Data Splitting",
                    epsilon=None,
                    rep=b,
                    role="inference",
                    X_hat=X_hat_inf,
                    Sigma_hat=Sigma_hat_inf,
                    sigma2_hat=sigma2_hat_inf,
                    sigma_mode=sigma,
                    data_type=data_type,
                    utility_fn=utility_fn,
                    extra={
                        "n_obs_sel": int(ds["n_obs_sel"]),
                        "n_obs_inf": int(ds["n_obs_inf"]),
                    },
                )
            )

            scores_sel = (
                X_hat_sel
                if utility_fn is None
                else np.asarray(utility_fn(X_hat_sel), dtype=float)
            )

            if scores_sel.shape[0] != M:
                raise ValueError("utility_fn(X_hat_sel) must return a length-M vector.")

            selected_idx = np.argsort(scores_sel)[-k:][::-1]
            selected_idx = np.asarray(selected_idx, dtype=int)

            subset_records.append({
                "rep": b,
                "selected_subset": tuple(sorted(int(x) for x in selected_idx)),
            })

            for rank, idx in enumerate(selected_idx, start=1):
                idx = int(idx)

                se = float(np.sqrt(max(Sigma_hat_inf[idx, idx], 0.0)))
                L = float(X_hat_inf[idx] - zcrit * se)
                U = float(X_hat_inf[idx] + zcrit * se)
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
                    "X_bar_sel": float(X_hat_sel[idx]),
                    "X_bar_inf": float(X_hat_inf[idx]),
                    "sigma2_hat_inf": float(sigma2_hat_inf),
                    "sigma_mode": sigma,
                    "n_obs_sel": int(ds["n_obs_sel"]),
                    "n_obs_inf": int(ds["n_obs_inf"]),
                    "data_type": data_type,
                })

            time_records.append({
                "method": "Data Splitting",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": True,
                "data_type": data_type,
            })

        except Exception as e:
            time_records.append({
                "method": "Data Splitting",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": False,
                "error_type": type(e).__name__,
                "data_type": data_type,
            })

            failures.append({"rep": b, "error": repr(e), "error_type": type(e).__name__})

            if verbose:
                print(f"\n[data_splitting failure] rep={b}")
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
