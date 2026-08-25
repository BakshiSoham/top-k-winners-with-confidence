import time

import numpy as np
import pandas as pd

from src.Standard import NaiveSubsetInference

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

def simulate_naive_coverage_hat(
    mu,
    Sigma,
    k,
    B,
    n_obs,
    alpha=0.05,
    *,
    utility_fn=None,
    selected_subset=None,
    subset_order="utility",
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
    else:
        # Placeholder only; BT covariance is estimated per replication.
        Sigma = np.eye(M) if Sigma is None else np.asarray(Sigma, dtype=float)

    X_samples = validate_data_samples(X_samples, B, n_obs, M, data_type=data_type)

    if not (1 <= k <= M):
        raise ValueError(f"k must be in [1, M], got k={k}, M={M}")
    if B <= 0:
        raise ValueError("B must be positive")

    rng = np.random.default_rng(seed)

    ci_records = []
    selected_subsets = []
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
                    method="Standard",
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

            naive_model = NaiveSubsetInference(
                X=X_hat,
                k=k,
                H0_mu=target_mu,
                Sigma=Sigma_hat,
                utility_fn=utility_fn,
                alpha=alpha,
                selected_subset=selected_subset,
                subset_order=subset_order,
            )

            res = naive_model.inference_on_subset(alpha=alpha)

            time_records.append({
                "method": "Standard",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": True,
                "data_type": data_type,
            })

            scores_full = (
                X_hat
                if utility_fn is None
                else np.asarray(utility_fn(X_hat), dtype=float)
            )
            
            selected_subsets.append(tuple(sorted(naive_model.selected_set.tolist())))
            subset_records.append({
                "rep": b,
                "selected_subset": tuple(sorted(int(x) for x in naive_model.selected_set.tolist())),
                "score_vec": tuple(float(x) for x in scores_full),
            })

            for rank, rec in enumerate(res, start=1):
                idx = int(rec.index)
                L = float(rec.ci_lower)
                U = float(rec.ci_upper)
                truth = float(target_mu[idx])

                ci_records.append({
                    "rep": b,
                    "rank": int(rank),
                    "index": idx,
                    "ci_lower": L,
                    "ci_upper": U,
                    "truth": truth,
                    "covered": bool(L <= truth <= U),
                    "length": float(U - L),
                    "sigma2_hat": float(sigma2_hat),
                    "sigma_mode": sigma,
                    "data_type": data_type,
                    **meta,
                })

        except Exception as e:
            time_records.append({
                "method": "Standard",
                "epsilon": np.nan,
                "rep": b,
                "time": float(time.time() - t0),
                "success": False,
                "error_type": type(e).__name__,
                "data_type": data_type,
            })
            if verbose:
                print(f"\n[naive failure] rep={b}")
                print(type(e).__name__, repr(e))

        if verbose and (b + 1) % max(1, B // 20) == 0:
            print(f"{b+1}/{B}")

    ci_df = pd.DataFrame(ci_records)
    ci_df_for_rep = ci_df.rename(columns={"index": "idx"}) if len(ci_df) > 0 else pd.DataFrame(columns=["rep", "idx", "covered", "length"])

    coverage_per_rep, length_per_rep_table, length_per_rep = _add_length_per_rep_outputs(
        ci_df_for_rep,
        B=B,
        k=k,
    )

    complete_rep_mask = (
        coverage_per_rep["is_complete"]
        if "is_complete" in coverage_per_rep.columns
        else np.ones(len(coverage_per_rep), dtype=bool)
    )

    coverage_all = (
        float(coverage_per_rep.loc[complete_rep_mask, "coverage_rate"].mean())
        if len(coverage_per_rep) > 0 else np.nan
    )

    avg_length_all = (
        float(coverage_per_rep.loc[complete_rep_mask, "avg_length"].mean())
        if len(coverage_per_rep) > 0 else np.nan
    )

    coverage_per_rank = (
        ci_df.groupby("rank", as_index=False)["covered"]
        .mean()
        .rename(columns={"covered": "coverage_rate"})
        if len(ci_df) > 0 else pd.DataFrame(columns=["rank", "coverage_rate"])
    )

    return {
        "coverage_all": coverage_all,
        "avg_length_all": avg_length_all,
        "coverage_per_rep": coverage_per_rep,
        "length_per_rep_table": length_per_rep_table,
        "length_per_rep": length_per_rep,
        "coverage_per_rank": coverage_per_rank,
        "ci_records": ci_df,
        "selected_subsets": selected_subsets,
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


