import time

import numpy as np
import pandas as pd

from src.Randomized_PSI import TopKSelectionModel

from .simulation_utils import (
    _default_utility_fn_for_data_type,
    _get_rep_stat_cov,
    _make_stat_records,
    _target_mu_from_config,
    _validate_data_type,
    build_rep_coverage_table,
    validate_data_samples,
)
def simulate_topk_hat(
    mu,
    Sigma,
    k,
    B,
    n_obs,
    alpha=0.05,
    *,
    epsilon=3.0,
    grid_size=500,
    utility_fn=None,
    span=4.0,
    density_cutoff=1e-4,
    compute_pivots=False,
    seed=0,
    verbose=True,
    X_samples=None,
    use_known_sigma=True,
    sel_scale="adaptive",   # 'adaptive' or 'none' — must match theory
    data_type="gaussian",
    data_config=None,
):
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

    def _empty_pivot_outputs():
        pivots_per_rank_arr = [np.array([], dtype=float) for _ in range(k)]
        return {
            "pivots_per_rank": pivots_per_rank_arr,
            "pivots_all": np.array([], dtype=float),
            "type1_one_tail_per_rank": [np.nan for _ in range(k)],
            "type1_two_tail_per_rank": [np.nan for _ in range(k)],
            "type1_one_tail_all": np.nan,
            "type1_two_tail_all": np.nan,
        }

    def _run_one_eps(eps: float, *, seed_one: int):
        rng = np.random.default_rng(seed_one)

        ci_records = []
        ci_failures = []
        subset_records = []
        selected_index_records = []
        time_records = []
        stat_records = []

        for b in range(B):
            t0 = time.time()
            success = True
            reason = None

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
                    sigma="known" if use_known_sigma else "unknown",
                )

                stat_records.extend(
                    _make_stat_records(
                        method="Randomized PSI",
                        epsilon=float(eps),
                        rep=b,
                        role="selection",
                        X_hat=X_hat,
                        Sigma_hat=Sigma_hat,
                        sigma2_hat=sigma2_hat,
                        sigma_mode="known" if use_known_sigma else "unknown",
                        data_type=data_type,
                        utility_fn=utility_fn,
                        extra=meta,
                    )
                )

                model = TopKSelectionModel(
                    X=X_hat,
                    k=k,
                    H0_mu=target_mu,
                    true_Sigma=Sigma_hat,
                    utility_fn=utility_fn,
                    epsilon=eps,
                    grid_size=grid_size,
                    sel_scale=sel_scale,
                )

                S_obs, _ = model.randomized_selected_top_k(
                    X=X_hat,
                    k=k,
                    epsilon=eps,
                    scale=sel_scale,
                    seed=int(rng.integers(1_000_000_000)),
                )

                S_obs = tuple(sorted(int(x) for x in S_obs))
                S_obs_set = set(S_obs)

                subset_records.append({
                    "epsilon": float(eps),
                    "rep": b,
                    "selected_subset": S_obs,
                })

                for rank, idx in enumerate(S_obs, start=1):
                    selected_index_records.append({
                        "epsilon": float(eps),
                        "rep": b,
                        "rank": int(rank),
                        "idx": int(idx),
                        "selected_subset": S_obs,
                    })

                try:
                    ci_out = model.confidence_interval_topk(
                        S_obs=S_obs,
                        Sigma=Sigma_hat,
                        alpha=alpha,
                        k=k,
                        epsilon=eps,
                        grid_size=grid_size,
                        seed=int(rng.integers(1_000_000_000)),
                        verbose=False,
                    )

                    S_ci = tuple(sorted(int(x) for x in ci_out.get("S_obs", S_obs)))
                    S_ci_set = set(S_ci)

                    for rank, rec_list in ci_out["per_rank"].items():
                        rank_int = int(rank)

                        for rr in rec_list:
                            idx = int(rr["idx"])
                            L = rr.get("L", None)
                            U = rr.get("U", None)

                            finite_ci = (
                                (L is not None)
                                and (U is not None)
                                and np.isfinite(L)
                                and np.isfinite(U)
                            )

                            truth = float(target_mu[idx])

                            if finite_ci:
                                covered = bool(float(L) <= truth <= float(U))
                                length = float(U - L)
                                L_out = float(L)
                                U_out = float(U)
                            else:
                                covered = False
                                length = np.nan
                                L_out = np.nan
                                U_out = np.nan
                                ci_failures.append({
                                    "epsilon": float(eps),
                                    "rep": b,
                                    "rank": rank_int,
                                    "idx": idx,
                                    "reason": "invalid_interval",
                                })

                            ci_records.append({
                                "epsilon": float(eps),
                                "rep": b,
                                "rank": rank_int,
                                "idx": idx,
                                "selected": idx in S_ci_set,
                                "selected_subset": S_ci,
                                "L": L_out,
                                "U": U_out,
                                "truth": truth,
                                "covered": covered,
                                "length": length,
                                "sigma2_hat": float(sigma2_hat),
                                "data_type": data_type,
                                **meta,
                            })

                except Exception as e:
                    success = False
                    reason = f"ci_error: {repr(e)}"

                    ci_failures.append({
                        "epsilon": float(eps),
                        "rep": b,
                        "rank": None,
                        "idx": None,
                        "reason": reason,
                    })

                    for rank, idx in enumerate(S_obs, start=1):
                        ci_records.append({
                            "epsilon": float(eps),
                            "rep": b,
                            "rank": int(rank),
                            "idx": int(idx),
                            "selected": True,
                            "selected_subset": S_obs,
                            "L": np.nan,
                            "U": np.nan,
                            "truth": float(target_mu[idx]),
                            "covered": False,
                            "length": np.nan,
                            "sigma2_hat": float(sigma2_hat),
                            "data_type": data_type,
                        })

            except Exception as e:
                success = False
                reason = f"rep_error: {repr(e)}"
                ci_failures.append({
                    "epsilon": float(eps),
                    "rep": b,
                    "rank": None,
                    "idx": None,
                    "reason": reason,
                })
                if verbose:
                    print(f"\n[randomized failure] eps={eps}, rep={b}")
                    print(type(e).__name__, repr(e))

            time_records.append({
                "method": "Randomized PSI",
                "epsilon": float(eps),
                "rep": b,
                "time": float(time.time() - t0),
                "success": bool(success),
                "reason": reason,
                "data_type": data_type,
            })

            if verbose and (b + 1) % max(1, B // 50) == 0:
                pct = int((b + 1) / B * 100)
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                print(f"\rProgress: |{bar}| {pct}% done", end="")

        if verbose:
            print()

        ci_df = pd.DataFrame(ci_records)
        fail_df = pd.DataFrame(ci_failures)
        subset_df = pd.DataFrame(subset_records)
        selected_index_df = pd.DataFrame(selected_index_records)
        time_df = pd.DataFrame(time_records)

        if len(ci_df) > 0:
            coverage_by_index_table = (
                ci_df.groupby("idx", as_index=False)
                     .agg(
                         n_selected=("idx", "size"),
                         coverage_rate=("covered", "mean"),
                         avg_length=("length", "mean"),
                     )
                     .sort_values("idx")
                     .reset_index(drop=True)
            )

            coverage_by_rank_table = (
                ci_df.groupby("rank", as_index=False)
                     .agg(
                         coverage_rate=("covered", "mean"),
                         avg_length=("length", "mean"),
                         n=("rank", "size"),
                     )
                     .sort_values("rank")
                     .reset_index(drop=True)
            )

            coverage_by_index = {
                int(row["idx"]): float(row["coverage_rate"])
                for _, row in coverage_by_index_table.iterrows()
            }
            avg_length_by_index = {
                int(row["idx"]): float(row["avg_length"])
                for _, row in coverage_by_index_table.iterrows()
            }

            coverage_per_rep_table = build_rep_coverage_table(ci_df, B=B, k=k)

            if "is_complete" in coverage_per_rep_table.columns:
                complete_rep_mask = coverage_per_rep_table["is_complete"].to_numpy(dtype=bool)
            else:
                complete_rep_mask = np.ones(len(coverage_per_rep_table), dtype=bool)

            coverage_all_selected = (
                float(coverage_per_rep_table.loc[complete_rep_mask, "coverage_rate"].mean())
                if len(coverage_per_rep_table) > 0 and np.any(complete_rep_mask)
                else np.nan
            )
            avg_length_all_selected = (
                float(coverage_per_rep_table.loc[complete_rep_mask, "avg_length"].mean())
                if len(coverage_per_rep_table) > 0 and np.any(complete_rep_mask)
                else np.nan
            )
        else:
            coverage_by_index_table = pd.DataFrame(columns=["idx", "n_selected", "coverage_rate", "avg_length"])
            coverage_by_rank_table = pd.DataFrame(columns=["rank", "coverage_rate", "avg_length", "n"])
            coverage_by_index = {}
            avg_length_by_index = {}
            coverage_per_rep_table = build_rep_coverage_table(
                pd.DataFrame(columns=["rep", "idx", "covered", "length"]),
                B=B,
                k=k,
            )
            coverage_all_selected = np.nan
            avg_length_all_selected = np.nan

        if len(selected_index_df) > 0:
            selection_count = selected_index_df.groupby("idx").size().reindex(range(M), fill_value=0)
            selection_rate_by_index = {int(idx): float(cnt / B) for idx, cnt in selection_count.items()}
        else:
            selection_rate_by_index = {j: 0.0 for j in range(M)}

        pivot_outputs = _empty_pivot_outputs()

        coverage_per_rep_table = coverage_per_rep_table.copy()
        coverage_per_rep_table["epsilon"] = float(eps)

        length_cols = ["epsilon", "rep", "avg_length", "n_intervals"]
        if "is_complete" in coverage_per_rep_table.columns:
            length_cols.append("is_complete")
        length_per_rep_table = coverage_per_rep_table[length_cols].copy()
        length_per_rep = length_per_rep_table["avg_length"].to_numpy(dtype=float)

        return {
            **pivot_outputs,
            "ci_records": ci_df,
            "ci_failures": fail_df,
            "coverage_by_index_table": coverage_by_index_table,
            "coverage_by_rank_table": coverage_by_rank_table,
            "coverage_by_index": coverage_by_index,
            "avg_length_by_index": avg_length_by_index,
            "selection_rate_by_index": selection_rate_by_index,
            "coverage_per_rep_table": coverage_per_rep_table,
            "coverage_all_selected": coverage_all_selected,
            "avg_length_all_selected": avg_length_all_selected,
            "subset_df": subset_df,
            "selected_index_df": selected_index_df,
            "stat_df": pd.DataFrame(stat_records),
            "time": time_df,
            "alpha": alpha,
            "k": k,
            "B": B,
            "epsilon": float(eps),
            "compute_pivots": False,
            "use_known_sigma": bool(use_known_sigma),
            "data_type": data_type,
            "length_per_rep_table": length_per_rep_table,
            "length_per_rep": length_per_rep,
        }

    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_list = [float(eps) for eps in epsilon]
        if len(eps_list) == 0:
            raise ValueError("epsilon list must be non-empty.")

        results = {}
        base_seed = int(seed)
        for j_eps, eps in enumerate(eps_list):
            if verbose:
                print(f"\n=== epsilon = {eps} ({j_eps + 1}/{len(eps_list)}) ===")
            results[eps] = _run_one_eps(eps, seed_one=base_seed + 10_000 * j_eps)
        return results

    return _run_one_eps(float(epsilon), seed_one=int(seed))
