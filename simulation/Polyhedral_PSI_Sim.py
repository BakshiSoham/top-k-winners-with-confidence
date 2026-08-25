import time

import numpy as np
import pandas as pd

from src.Polyhedral_PSI import PolyhedralTopKInference

from .simulation_utils import (
    _add_length_per_rep_outputs,
    _default_utility_fn_for_data_type,
    _get_rep_stat_cov,
    _make_stat_records,
    _target_mu_from_config,
    _validate_data_type,
    _validate_sigma_mode,
    validate_data_samples,
)
def simulate_polyhedral_hat(
    mu,
    Sigma,
    k,
    B,
    n_obs,
    alpha=0.05,
    *,
    utility_fn=None,
    grid_size=500,
    seed=0,
    verbose=True,
    selected_subset=None,
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
    else:
        Sigma = np.eye(M) if Sigma is None else np.asarray(Sigma, dtype=float)

    X_samples = validate_data_samples(X_samples, B, n_obs, M, data_type=data_type)

    if not (1 <= k <= M):
        raise ValueError(f"k must be in [1, M], got k={k}, M={M}")
    if B <= 0:
        raise ValueError("B must be positive")

    rng = np.random.default_rng(seed)

    ci_rows = []
    pivot_rows = []
    failures = []
    subset_records = []
    time_records = []
    stat_records = []

    for b in range(B):
        t0 = time.time()

        try:
            X_hat, Sigma_hat, sigma2_hat, _, meta = _get_rep_stat_cov(
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

            stat_records.extend(
                _make_stat_records(
                    method="Polyhedral PSI",
                    epsilon=None,
                    rep=b,
                    role="selection",
                    X_hat=X_hat,
                    Sigma_hat=Sigma_hat,
                    sigma2_hat=sigma2_hat,
                    sigma_mode=sigma,
                    data_type=data_type,
                    utility_fn=utility_fn,
                    extra=meta,
                )
            )

            model = PolyhedralTopKInference(
                X=X_hat,
                k=k,
                H0_mu=target_mu,
                Sigma=Sigma_hat,
                utility_fn=utility_fn,
                grid_size=grid_size,
                alpha=alpha,
                selected_subset=selected_subset,
            )

            res = model.post_selection_inference_on_top_k(alpha=alpha)

            time_records.append({
                "method": "Polyhedral PSI",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": True,
                "data_type": data_type,
            })

            if selected_subset is not None:
                S_obs = tuple(sorted(int(x) for x in selected_subset))
            else:
                S_obs = tuple(sorted(int(x) for x in model.selected_set.tolist()))

            subset_records.append({"rep": b, "selected_subset": S_obs})

            if res is None or len(res) == 0:
                failures.append({"rep": b, "error": "empty_result_from_post_selection_inference"})
                continue

            for rec in res:
                idx = int(rec.index)
                L = float(rec.ci_lower)
                U = float(rec.ci_upper)
                truth = float(target_mu[idx])

                ci_rows.append({
                    "rep": b,
                    "rank": int(rec.rank),
                    "idx": idx,
                    "L": L,
                    "U": U,
                    "truth": truth,
                    "covered": bool(L <= truth <= U),
                    "length": float(U - L),
                    "sigma2_hat": float(sigma2_hat),
                    "sigma_mode": sigma,
                    "data_type": data_type,
                    **meta,
                })

                pivot_rows.append({
                    "rep": b,
                    "rank": int(rec.rank),
                    "idx": idx,
                    "pivot": float(rec.pivot_at_truth),
                    "p_two": float(rec.pvalue_two_sided),
                })

        except Exception as e:
            time_records.append({
                "method": "Polyhedral PSI",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": False,
                "error_type": type(e).__name__,
                "data_type": data_type,
            })

            failures.append({"rep": b, "error": repr(e), "error_type": type(e).__name__})

            if verbose:
                print(f"\n[polyhedral failure] rep={b}")
                print(type(e).__name__, repr(e))

        if verbose and (b + 1) % max(1, B // 20) == 0:
            print(f"{b+1}/{B}")

    ci_df = pd.DataFrame(ci_rows)
    pivot_df = pd.DataFrame(pivot_rows)
    fail_df = pd.DataFrame(failures)

    coverage_per_rep, length_per_rep_table, length_per_rep = _add_length_per_rep_outputs(
        ci_df,
        B=B,
        k=k,
    )

    return {
        "ci_df": ci_df,
        "pivot_df": pivot_df,
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
