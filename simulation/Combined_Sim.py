import numpy as np
import pandas as pd

from .Standard_Sim import simulate_naive_coverage_hat
from .Randomized_PSI_Sim import simulate_topk_hat
from .Polyhedral_PSI_Sim import simulate_polyhedral_hat
from .Data_Splitting_Sim import simulate_data_splitting_hat
from .Zoom_Correction_Sim import simulate_zoom_stepdown_hat

from .data_generator import (
    _normalize_variable_importance_config,
    estimate_variable_importance_oracle_bootstrap,
    prepare_variable_importance_holdout_setting,
)

from .simulation_utils import (
    _attach_setting_metadata_to_df,
    _default_utility_fn_for_data_type,
    _normalize_binomial_config,
    _normalize_bt_config,
    _prepare_shared_samples,
    _target_mu_from_config,
    _validate_data_type,
    _validate_sigma_mode,
    build_rep_coverage_table,
    build_time_wide_table,
    default_signal_strength_general,
    format_data_splitting_hat_output,
    format_naive_hat_output,
    format_polyhedral_hat_output,
    format_topk_hat_output,
    format_zoom_stepdown_hat_output,
)

from .simulation_utils import (
    _target_mu_for_data_type,
    default_signal_strength_from_mu_sigma,
)

def simulate_three_methods_compare_hat(
    mu,
    Sigma,
    k,
    B,
    n_obs,
    epsilon_list,
    alpha=0.05,
    *,
    methods=("Standard", "Randomized PSI", "Polyhedral PSI", "Data Splitting", "Zoom Correction"),
    utility_fn=None,
    grid_size=500,
    span=4.0,
    density_cutoff=1e-4,
    seed=0,
    verbose=True,
    X_samples=None,
    share_same_data_across_methods=False,
    sel_scale="adaptive",   # passed to simulate_topk_hat
    sigma="known",
    zoom_sigma_mode="mean",
    data_type="gaussian",
    data_config=None,
):
    sigma = _validate_sigma_mode(sigma)
    data_type = _validate_data_type(data_type)
    use_known_sigma = sigma == "known"
    
    utility_fn = _default_utility_fn_for_data_type(
        data_type=data_type,
        data_config=data_config,
        n_obs=n_obs,
        utility_fn=utility_fn,
    )

    epsilon_list = [float(eps) for eps in epsilon_list]

    allowed_methods = {
        "Standard",
        "Randomized PSI",
        "Polyhedral PSI",
        "Data Splitting",
        "Zoom Correction",
    }

    if methods is None:
        methods = ("Standard", "Randomized PSI", "Polyhedral PSI", "Data Splitting", "Zoom Correction")

    methods = [str(m).strip() for m in methods]
    unknown_methods = set(methods) - allowed_methods
    if len(unknown_methods) > 0:
        raise ValueError(f"Unknown methods: {unknown_methods}. Allowed methods are {allowed_methods}.")

    mu = np.asarray(mu, dtype=float).reshape(-1)
    M = mu.size

    if data_type == "gaussian":
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.shape != (M, M):
            raise ValueError(f"Sigma must be {(M, M)}, got {Sigma.shape}")
    else:
        Sigma = np.eye(M) if Sigma is None else np.asarray(Sigma, dtype=float)

    target_mu = _target_mu_from_config(
        mu,
        data_type=data_type,
        data_config=data_config,
        n_obs=n_obs,
    )

    X_samples_shared = _prepare_shared_samples(
        mu=mu,
        Sigma=Sigma,
        B=B,
        n_obs=n_obs,
        seed=seed,
        data_type=data_type,
        data_config=data_config,
        X_samples=X_samples,
        share_same_data_across_methods=share_same_data_across_methods,
    )

    naive_out = None
    topk_outs = None
    poly_out = None
    data_splitting_out = None
    zoom_stepdown_out = None

    df_list = []
    subset_df_list = []
    time_df_list = []
    stat_df_list = []

    if "Standard" in methods:
        if verbose:
            print(f"\n===== Running naive hat | sigma={sigma} | data_type={data_type} =====")
        naive_out = simulate_naive_coverage_hat(
            mu=mu,
            Sigma=Sigma,
            k=k,
            B=B,
            n_obs=n_obs,
            alpha=alpha,
            utility_fn=utility_fn,
            seed=seed,
            verbose=verbose,
            X_samples=X_samples_shared,
            sigma=sigma,
            data_type=data_type,
            data_config=data_config,
        )
        naive_df_base = format_naive_hat_output(naive_out)

        if "subset_df" in naive_out and len(naive_out["subset_df"]) > 0:
            tmp = naive_out["subset_df"].copy()
            tmp["method"] = "Standard"
            naive_subset_list = []
            for eps in epsilon_list:
                tt = tmp.copy()
                tt["epsilon"] = eps
                naive_subset_list.append(tt)
            subset_df_list.append(pd.concat(naive_subset_list, ignore_index=True))

        naive_df_list = []
        for eps in epsilon_list:
            tmp = naive_df_base.copy()
            tmp["epsilon"] = eps
            naive_df_list.append(tmp)
        if len(naive_df_list) > 0:
            naive_df = pd.concat(naive_df_list, ignore_index=True)
            if len(naive_df) > 0:
                df_list.append(naive_df)

        if verbose:
            print("naive rows:", len(naive_out["ci_records"]))
        if "time" in naive_out and len(naive_out["time"]) > 0:
            time_df_list.append(naive_out["time"].copy())

        if "stat_df" in naive_out and len(naive_out["stat_df"]) > 0:
            tmp_stat = naive_out["stat_df"].copy()
            stat_list = []
            for eps in epsilon_list:
                tt = tmp_stat.copy()
                tt["epsilon"] = float(eps)
                stat_list.append(tt)
            stat_df_list.append(pd.concat(stat_list, ignore_index=True))

    if "Randomized PSI" in methods:
        if verbose:
            print(f"\n===== Running randomized selective hat | sigma={sigma} | data_type={data_type} =====")
        topk_outs = simulate_topk_hat(
            mu=mu,
            Sigma=Sigma,
            k=k,
            B=B,
            n_obs=n_obs,
            alpha=alpha,
            epsilon=epsilon_list,
            grid_size=grid_size,
            utility_fn=utility_fn,
            span=span,
            density_cutoff=density_cutoff,
            compute_pivots=False,
            use_known_sigma=use_known_sigma,
            seed=seed + 1000,
            verbose=verbose,
            X_samples=X_samples_shared,
            data_type=data_type,
            data_config=data_config,
        )

        topk_df_list = []
        for eps in epsilon_list:
            topk_df_list.append(format_topk_hat_output(topk_outs[eps], epsilon=eps))

        for eps in epsilon_list:
            if "subset_df" in topk_outs[eps] and len(topk_outs[eps]["subset_df"]) > 0:
                tmp = topk_outs[eps]["subset_df"].copy()
                tmp["method"] = "Randomized PSI"
                tmp["epsilon"] = float(eps)
                subset_df_list.append(tmp)

        if len(topk_df_list) > 0:
            topk_df = pd.concat(topk_df_list, ignore_index=True)
            if len(topk_df) > 0:
                df_list.append(topk_df)

        if verbose:
            for eps in epsilon_list:
                print(f"randomized eps={eps} rows:", len(topk_outs[eps]["ci_records"]))
        for eps in epsilon_list:
            if "time" in topk_outs[eps] and len(topk_outs[eps]["time"]) > 0:
                time_df_list.append(topk_outs[eps]["time"].copy())
            if "stat_df" in topk_outs[eps] and len(topk_outs[eps]["stat_df"]) > 0:
                stat_df_list.append(topk_outs[eps]["stat_df"].copy())

    if "Polyhedral PSI" in methods:
        if verbose:
            print(f"\n===== Running polyhedral hat | sigma={sigma} | data_type={data_type} =====")
        poly_out = simulate_polyhedral_hat(
            mu=mu,
            Sigma=Sigma,
            k=k,
            B=B,
            n_obs=n_obs,
            alpha=alpha,
            utility_fn=utility_fn,
            grid_size=grid_size,
            seed=seed + 2000,
            verbose=verbose,
            X_samples=X_samples_shared,
            sigma=sigma,
            data_type=data_type,
            data_config=data_config,
        )
        poly_df_base = format_polyhedral_hat_output(poly_out)

        if "subset_df" in poly_out and len(poly_out["subset_df"]) > 0:
            tmp = poly_out["subset_df"].copy()
            tmp["method"] = "Polyhedral PSI"
            poly_subset_list = []
            for eps in epsilon_list:
                tt = tmp.copy()
                tt["epsilon"] = eps
                poly_subset_list.append(tt)
            subset_df_list.append(pd.concat(poly_subset_list, ignore_index=True))

        poly_df_list = []
        for eps in epsilon_list:
            tmp = poly_df_base.copy()
            tmp["epsilon"] = eps
            poly_df_list.append(tmp)
        if len(poly_df_list) > 0:
            poly_df = pd.concat(poly_df_list, ignore_index=True)
            if len(poly_df) > 0:
                df_list.append(poly_df)

        if verbose:
            print("polyhedral rows:", len(poly_out["ci_df"]))
            print("polyhedral failures:", len(poly_out["failures"]))
            if len(poly_out["failures"]) > 0:
                print(poly_out["failures"].head())
        if "time" in poly_out and len(poly_out["time"]) > 0:
            time_df_list.append(poly_out["time"].copy())

        

        if "stat_df" in poly_out and len(poly_out["stat_df"]) > 0:
            tmp_stat = poly_out["stat_df"].copy()
            stat_list = []
            for eps in epsilon_list:
                tt = tmp_stat.copy()
                tt["epsilon"] = float(eps)
                stat_list.append(tt)
            stat_df_list.append(pd.concat(stat_list, ignore_index=True))

    if "Data Splitting" in methods:
        if verbose:
            print(f"\n===== Running data splitting hat | sigma={sigma} | data_type={data_type} =====")
        data_splitting_out = simulate_data_splitting_hat(
            mu=mu,
            Sigma=Sigma,
            k=k,
            B=B,
            n_obs=n_obs,
            alpha=alpha,
            utility_fn=utility_fn,
            seed=seed + 3000,
            verbose=verbose,
            X_samples=X_samples_shared,
            sigma=sigma,
            data_type=data_type,
            data_config=data_config,
        )
        ds_df_base = format_data_splitting_hat_output(data_splitting_out)

        if "subset_df" in data_splitting_out and len(data_splitting_out["subset_df"]) > 0:
            tmp = data_splitting_out["subset_df"].copy()
            tmp["method"] = "Data Splitting"
            ds_subset_list = []
            for eps in epsilon_list:
                tt = tmp.copy()
                tt["epsilon"] = eps
                ds_subset_list.append(tt)
            subset_df_list.append(pd.concat(ds_subset_list, ignore_index=True))

        ds_df_list = []
        for eps in epsilon_list:
            tmp = ds_df_base.copy()
            tmp["epsilon"] = eps
            ds_df_list.append(tmp)
        if len(ds_df_list) > 0:
            ds_df = pd.concat(ds_df_list, ignore_index=True)
            if len(ds_df) > 0:
                df_list.append(ds_df)

        if verbose:
            print("data_splitting rows:", len(data_splitting_out["ci_df"]))
            print("data_splitting failures:", len(data_splitting_out["failures"]))
            if len(data_splitting_out["failures"]) > 0:
                print(data_splitting_out["failures"].head())
        if "time" in data_splitting_out and len(data_splitting_out["time"]) > 0:
            time_df_list.append(data_splitting_out["time"].copy())

        if "stat_df" in data_splitting_out and len(data_splitting_out["stat_df"]) > 0:
            tmp_stat = data_splitting_out["stat_df"].copy()
            stat_list = []
            for eps in epsilon_list:
                tt = tmp_stat.copy()
                tt["epsilon"] = float(eps)
                stat_list.append(tt)
            stat_df_list.append(pd.concat(stat_list, ignore_index=True))

    if "Zoom Correction" in methods:
        if verbose:
            print(f"\n===== Running zoom step-down hat | sigma={sigma} | data_type={data_type} =====")
        zoom_stepdown_out = simulate_zoom_stepdown_hat(
            mu=mu,
            Sigma=Sigma,
            k=k,
            B=B,
            n_obs=n_obs,
            alpha=alpha,
            utility_fn=utility_fn,
            seed=seed + 4000,
            verbose=verbose,
            X_samples=X_samples_shared,
            sigma=sigma,
            zoom_sigma_mode=zoom_sigma_mode,
            data_type=data_type,
            data_config=data_config,
        )
        zoom_df_base = format_zoom_stepdown_hat_output(zoom_stepdown_out)

        if "subset_df" in zoom_stepdown_out and len(zoom_stepdown_out["subset_df"]) > 0:
            tmp = zoom_stepdown_out["subset_df"].copy()
            tmp["method"] = "Zoom Correction"
            zoom_subset_list = []
            for eps in epsilon_list:
                tt = tmp.copy()
                tt["epsilon"] = eps
                zoom_subset_list.append(tt)
            subset_df_list.append(pd.concat(zoom_subset_list, ignore_index=True))

        zoom_df_list = []
        for eps in epsilon_list:
            tmp = zoom_df_base.copy()
            tmp["epsilon"] = eps
            zoom_df_list.append(tmp)
        if len(zoom_df_list) > 0:
            zoom_df = pd.concat(zoom_df_list, ignore_index=True)
            if len(zoom_df) > 0:
                df_list.append(zoom_df)

        if verbose:
            print("zoom_stepdown rows:", len(zoom_stepdown_out["ci_df"]))
            print("zoom_stepdown failures:", len(zoom_stepdown_out["failures"]))
            if len(zoom_stepdown_out["failures"]) > 0:
                print(zoom_stepdown_out["failures"].head())
        if "time" in zoom_stepdown_out and len(zoom_stepdown_out["time"]) > 0:
            time_df_list.append(zoom_stepdown_out["time"].copy())

        if "stat_df" in zoom_stepdown_out and len(zoom_stepdown_out["stat_df"]) > 0:
            tmp_stat = zoom_stepdown_out["stat_df"].copy()
            stat_list = []
            for eps in epsilon_list:
                tt = tmp_stat.copy()
                tt["epsilon"] = float(eps)
                stat_list.append(tt)
            stat_df_list.append(pd.concat(stat_list, ignore_index=True))

    output_columns = ["method", "epsilon", "rep", "rank", "idx", "L", "U", "truth", "covered", "length"]
    all_ci_df = pd.DataFrame(columns=output_columns) if len(df_list) == 0 else pd.concat(df_list, ignore_index=True)

    if len(subset_df_list) == 0:
        all_subset_df = pd.DataFrame(columns=["method", "epsilon", "rep", "selected_subset"])
    else:
        all_subset_df = pd.concat(subset_df_list, ignore_index=True)

    if len(stat_df_list) == 0:
        all_stat_df = pd.DataFrame(columns=[
            "method", "epsilon", "rep", "role", "idx",
            "mu_hat", "score_hat", "variance_hat", "se_hat",
            "sigma2_hat", "sigma_mode", "data_type",
        ])
    else:
        all_stat_df = pd.concat(stat_df_list, ignore_index=True)

    all_rep_df = build_rep_coverage_table(all_ci_df, group_cols=["method", "epsilon"], B=B, k=k)

    length_cols = ["method", "epsilon", "rep", "avg_length", "n_intervals"]
    if "is_complete" in all_rep_df.columns:
        length_cols.append("is_complete")
    all_length_rep_df = all_rep_df[length_cols].copy()

    if len(time_df_list) == 0:
        all_time_long = pd.DataFrame(columns=["method", "epsilon", "rep", "time", "success"])
    else:
        all_time_long = pd.concat(time_df_list, ignore_index=True)

    all_time = build_time_wide_table(all_time_long, B=B, methods=methods, epsilon_list=epsilon_list)

    if verbose:
        print("\n===== Final combined table =====")
        print("all_ci_df rows:", len(all_ci_df))
        if len(all_ci_df) > 0:
            print(all_ci_df.groupby(["method", "epsilon"]).size())

    return {
        "naive_out": naive_out,
        "topk_outs": topk_outs,
        "poly_out": poly_out,
        "data_splitting_out": data_splitting_out,
        "zoom_stepdown_out": zoom_stepdown_out,
        "all_ci_df": all_ci_df,
        "all_rep_df": all_rep_df,
        "all_length_rep_df": all_length_rep_df,
        "methods": methods,
        "all_subset_df": all_subset_df,
        "all_stat_df": all_stat_df,
        "X_samples_shared": X_samples_shared,
        "mu": target_mu,
        "mu_raw": mu,
        "Sigma": Sigma,
        "k": k,
        "B": B,
        "n_obs": n_obs,
        "sigma": sigma,
        "use_known_sigma": use_known_sigma,
        "data_type": data_type,
        "data_config": dict(data_config or {}),
        "time": all_time,
        "time_long": all_time_long,
    }


def default_signal_strength_general(
    mu,
    Sigma,
    k,
    *,
    n_obs=None,
    utility_fn=None,
    data_type="gaussian",
    data_config=None,
):
    data_type = _validate_data_type(data_type)
    center_bt = _normalize_bt_config(data_config, n_obs=n_obs)["center_bt"] if data_type == "bt_davidson" else True
    target_mu = _target_mu_for_data_type(mu, data_type=data_type, center_bt=center_bt)

    if Sigma is None:
        # Keep the same metadata keys, but do not pretend to know a standardized gap.
        score_vec = target_mu if utility_fn is None else np.asarray(utility_fn(target_mu), dtype=float).reshape(-1)
        order = np.argsort(score_vec)[::-1]
        topk_idx = order[:k]
        true_topk_subset = tuple(sorted(int(i) for i in topk_idx))
        true_topk_utility = float(np.sum(score_vec[list(true_topk_subset)]))
        topk_gap = float(score_vec[order[k - 1]] - score_vec[order[k]]) if k < len(score_vec) else np.nan
        top1_gap = float(score_vec[order[0]] - score_vec[order[1]]) if len(score_vec) >= 2 else np.nan
        return {
            "signal_strength": np.nan,
            "standardized_topk_gap": np.nan,
            "topk_gap": topk_gap,
            "top1_gap": top1_gap,
            "noise_scale_topk_gap": np.nan,
            "kth_idx": int(order[k - 1]),
            "next_idx": int(order[k]) if k < len(score_vec) else None,
            "true_topk_subset": true_topk_subset,
            "true_topk_utility": true_topk_utility,
        }

    return default_signal_strength_from_mu_sigma(
        target_mu,
        Sigma,
        k,
        n_obs=n_obs,
        utility_fn=utility_fn,
    )


def _merge_setting_keys_for_data_type(setting, base_config, *, data_type):
    dt = _validate_data_type(data_type)
    cfg = dict(base_config or {})
    common_keys = [
        "covariance", "cov_ridge", "split_frac",
    ]
    bt_keys = [
        "nu", "nu_true", "n_matches", "bt_n_matches", "center_bt",
    ]
    binom_keys = [
        "m", "n_trials", "statistic", "clip_eps", "binomial_split_frac",
    ]
    vi_keys = [
        "p", "f", "beta", "beta_signal",
        "m_func", "m_minus_func",
        "x_dist", "rho", "x_mean", "x_scale", "x_low", "x_high",
        "noise_sd", "noise_dist",
        "tree_params", "n_folds", "vi_split_frac", "split_frac",
        "oracle_reps", "oracle_n_obs",
        "oracle_covariance_source", "covariance_source",
        "oracle_cov_ridge", "cov_ridge", "min_var_y",
        "oracle", "phi", "mu", "Sigma",
        "true_vi_score", "normalized_vi_score",
        "fixed_nuisance",
        "vi_statistic",
        "vi_nuisance_mode",   
        "holdout_data",
        "holdout_n",
        "bootstrap_reps",
        "bootstrap_n_obs",
    
        "random_state_offset",
    ]

    keys = list(common_keys)
    if dt == "bt_davidson":
        keys += bt_keys
    elif dt == "binomial":
        keys += binom_keys
    elif dt == "variable_importance":
        keys += vi_keys

    for key in keys:
        if key in setting:
            cfg[key] = setting[key]

    return cfg


def _prepare_variable_importance_setting(
    setting,
    *,
    cfg_s,
    s,
    n_obs,
    seed,
    verbose,
):
    setting = dict(setting)
    cfg_s = dict(cfg_s or {})

    # ============================================================
    # 1. Resolve VI statistic mode
    # ============================================================

    vi_statistic = str(
        cfg_s.get(
            "vi_statistic",
            setting.get(
                "vi_statistic",
                # Backward-compatible default:
                # fixed if fixed_nuisance already exists; otherwise model_cf.
                (
                    "fixed_nuisance"
                    if (
                        cfg_s.get("fixed_nuisance", None) is not None
                        or setting.get("fixed_nuisance", None) is not None
                    )
                    else "model_cf"
                ),
            ),
        )
    ).lower().strip()

    cross_fitting_aliases = {
        "model_cf",
        "cross_fitting",
        "cross_fit",
        "crossfit",
        "model",
        "plugin_statistic",
    }

    fixed_nuisance_aliases = {
        "fixed_nuisance",
        "holdout_fixed",
        "fixed_holdout",
        "independent_holdout",
    }

    if vi_statistic in cross_fitting_aliases:
        vi_mode = "model_cf"
    elif vi_statistic in fixed_nuisance_aliases:
        vi_mode = "fixed_nuisance"
    else:
        raise ValueError(
            "Unknown VI statistic mode. Use either "
            "'model_cf' for cross-fitting or "
            "'fixed_nuisance' for independent-holdout nuisance estimation."
        )

    cfg_s["vi_statistic"] = vi_mode
    # ============================================================
    # 2. Check whether population mu and Sigma were supplied
    # ============================================================
    has_mu = (
        ("mu" in setting and setting["mu"] is not None)
        or ("phi" in setting and setting["phi"] is not None)
    )

    has_sigma = (
        "Sigma" in setting
        and setting["Sigma"] is not None
    )

    if has_mu:
        mu_s = np.asarray(
            setting["mu"]
            if setting.get("mu", None) is not None
            else setting["phi"],
            dtype=float,
        ).reshape(-1)

        cfg_s["p"] = int(mu_s.size)
    else:
        mu_s = None

    Sigma_s = (
        np.asarray(setting["Sigma"], dtype=float)
        if has_sigma
        else None
    )

    # ============================================================
    # 3. Cross-fitting route
    #
    # This reproduces latestcode:
    #   - automatic true-nuisance oracle when mu/Sigma are missing
    #   - model_cf inside every formal simulation replication
    # ============================================================

    if vi_mode == "model_cf":

        # Never accidentally use a fixed nuisance in this mode.
        cfg_s.pop("fixed_nuisance", None)

        oracle = cfg_s.get(
            "oracle",
            setting.get("oracle", None),
        )

        need_oracle = (
            mu_s is None
            or Sigma_s is None
        )

        if need_oracle and oracle is None:
            cfg_norm = _normalize_variable_importance_config(
                cfg_s,
                n_obs=n_obs,
            )

            if verbose:
                print("\n" + "-" * 70)
                print(
                    f"Estimating VI cross-fitting oracle for setting {s}"
                )
                print("vi_statistic = model_cf")
                print(
                    f"oracle_reps = {cfg_norm['oracle_reps']}"
                )
                oracle_n_obs_print = (
                    cfg_norm["oracle_n_obs"]
                    if cfg_norm["oracle_n_obs"] is not None
                    else n_obs
                )
                
                print(f"oracle_n_obs = {oracle_n_obs_print}")
                print(
                    "oracle_covariance_source = "
                    f"{cfg_norm['oracle_covariance_source']}"
                )
                print("-" * 70)
            oracle = estimate_variable_importance_oracle_bootstrap(
                setting=cfg_norm,
                n_obs=n_obs,
                oracle_reps=cfg_norm["oracle_reps"],
                oracle_n_obs=cfg_norm["oracle_n_obs"],
                n_folds=cfg_norm["n_folds"],
                tree_params=cfg_norm.get(
                    "tree_params",
                    None,
                ),
                covariance_source=cfg_norm[
                    "oracle_covariance_source"
                ],
                seed=seed + 100000 * int(s) + 777,
                verbose=verbose,
            )

            cfg_s["oracle"] = oracle

        if mu_s is None:
            if oracle is None or "phi" not in oracle:
                raise ValueError(
                    "Cross-fitting VI requires population mu/phi. "
                    "Provide mu/phi directly or enough DGP information "
                    "for estimate_variable_importance_oracle_bootstrap()."
                )

            mu_s = np.asarray(
                oracle["phi"],
                dtype=float,
            ).reshape(-1)

        if Sigma_s is None:
            if oracle is None or "Sigma_vim" not in oracle:
                raise ValueError(
                    "Cross-fitting VI requires population Sigma_vim. "
                    "Provide Sigma directly or enough DGP information "
                    "for estimate_variable_importance_oracle_bootstrap()."
                )

            Sigma_s = np.asarray(
                oracle["Sigma_vim"],
                dtype=float,
            )

        cfg_s["vi_statistic"] = "model_cf"

    # ============================================================
    # 4. Independent-holdout / fixed-nuisance route
    #
    # Fit nuisance once and reuse it in all formal replications.
    # ============================================================

    else:
        fixed_nuisance = cfg_s.get(
            "fixed_nuisance",
            setting.get("fixed_nuisance", None),
        )

        need_holdout_preparation = (
            fixed_nuisance is None
            or mu_s is None
            or Sigma_s is None
        )

        if need_holdout_preparation:
            cfg_for_holdout = dict(cfg_s)

            # Preserve all setting-level keys, including DGP and learner.
            cfg_for_holdout.update(setting)
            cfg_for_holdout["vi_statistic"] = "fixed_nuisance"

            holdout_n = int(
                cfg_for_holdout.get(
                    "holdout_n",
                    100000,
                )
            )

            bootstrap_reps = int(
                cfg_for_holdout.get(
                    "bootstrap_reps",
                    cfg_for_holdout.get(
                        "oracle_reps",
                        300,
                    ),
                )
            )

            bootstrap_n_obs = cfg_for_holdout.get(
                "bootstrap_n_obs",
                n_obs,
            )

            if verbose:
                print("\n" + "-" * 70)
                print(
                    f"Preparing fixed VI nuisance for setting {s}"
                )
                print("vi_statistic = fixed_nuisance")
                print(f"holdout_n = {holdout_n}")
                print(f"bootstrap_reps = {bootstrap_reps}")
                print(
                    f"bootstrap_n_obs = {bootstrap_n_obs}"
                )
                print("-" * 70)

            prepared = prepare_variable_importance_holdout_setting(
                cfg_for_holdout,
                n_obs=n_obs,
                holdout_n=holdout_n,
                bootstrap_reps=bootstrap_reps,
                bootstrap_n_obs=bootstrap_n_obs,
                seed=seed + 100000 * int(s) + 777,
                verbose=verbose,
            )

            # Use population quantities associated with the fixed nuisance.
            if mu_s is None:
                mu_s = np.asarray(
                    prepared["mu"],
                    dtype=float,
                ).reshape(-1)

            if Sigma_s is None:
                Sigma_s = np.asarray(
                    prepared["Sigma"],
                    dtype=float,
                )

            fixed_nuisance = prepared["fixed_nuisance"]

            # Carry all prepared objects into the formal simulation.
            prepared_cfg = dict(
                prepared.get(
                    "data_config",
                    {},
                )
            )

            cfg_s.update(prepared_cfg)

            cfg_s["oracle"] = prepared.get(
                "oracle",
                cfg_s.get("oracle", None),
            )

            cfg_s["holdout_n"] = int(
                prepared.get(
                    "X_hold",
                    np.empty((holdout_n, 0)),
                ).shape[0]
            )

            cfg_s["bootstrap_reps"] = bootstrap_reps
            cfg_s["bootstrap_n_obs"] = int(
                n_obs
                if bootstrap_n_obs is None
                else bootstrap_n_obs
            )

        if fixed_nuisance is None:
            raise ValueError(
                "Fixed-nuisance VI requires a fitted fixed_nuisance object."
            )

        cfg_s["fixed_nuisance"] = fixed_nuisance
        cfg_s["vi_statistic"] = "fixed_nuisance"

    mu_s = np.asarray(
        mu_s,
        dtype=float,
    ).reshape(-1)

    Sigma_s = np.asarray(
        Sigma_s,
        dtype=float,
    )

    p = int(mu_s.size)

    if Sigma_s.shape != (p, p):
        raise ValueError(
            "For variable_importance data, Sigma must have shape "
            f"{(p, p)}, got {Sigma_s.shape}."
        )

    if not np.all(np.isfinite(mu_s)):
        raise ValueError(
            "Variable Importance population mu contains non-finite values."
        )

    if not np.all(np.isfinite(Sigma_s)):
        raise ValueError(
            "Variable Importance population Sigma contains non-finite values."
        )
    cfg_s["p"] = p
    cfg_s["mu"] = mu_s
    cfg_s["phi"] = mu_s
    cfg_s["Sigma"] = Sigma_s

    # Preserve explicitly supplied truth; otherwise use prepared mu.
    if cfg_s.get("true_vi_score", None) is None:
        cfg_s["true_vi_score"] = mu_s

    if cfg_s.get("normalized_vi_score", None) is None:
        cfg_s["normalized_vi_score"] = mu_s

    return mu_s, Sigma_s, cfg_s



def simulate_multiple_parameter_settings_compare_hat(
    parameter_settings,
    k,
    B,
    n_obs,
    epsilon_list,
    alpha=0.05,
    *,
    methods=("Standard", "Randomized PSI", "Polyhedral PSI", "Data Splitting", "Zoom Correction"),
    utility_fn=None,
    grid_size=500,
    span=4.0,
    density_cutoff=1e-4,
    seed=0,
    verbose=True,
    share_same_data_across_methods=True,
    sel_scale="adaptive",
    sigma="known",
    zoom_sigma_mode="mean",
    signal_strength_fn=None,
    data_type="gaussian",
    data_config=None,
):
    data_type = _validate_data_type(data_type)

    if signal_strength_fn is None:
        signal_strength_fn = default_signal_strength_general

    if not isinstance(parameter_settings, (list, tuple)):
        raise TypeError("parameter_settings must be a list or tuple of dictionaries.")
    if len(parameter_settings) == 0:
        raise ValueError("parameter_settings must be non-empty.")

    M_ref = None
    normalized_settings = []

    for s, setting in enumerate(parameter_settings):
        if not isinstance(setting, dict):
            raise TypeError("Each entry of parameter_settings must be a dictionary.")

        cfg_s = _merge_setting_keys_for_data_type(
            setting,
            data_config,
            data_type=data_type,
        )

        if data_type == "variable_importance":
            mu_s, Sigma_s, cfg_s = _prepare_variable_importance_setting(
                setting,
                cfg_s=cfg_s,
                s=s,
                n_obs=n_obs,
                seed=seed,
                verbose=verbose,
            )

        else:
            if "mu" in setting:
                mu_s = np.asarray(setting["mu"], dtype=float).reshape(-1)
            elif data_type == "binomial" and "p" in setting:
                mu_s = np.asarray(setting["p"], dtype=float).reshape(-1)
            else:
                raise ValueError(
                    "Each setting must contain key 'mu'. For binomial data, key 'p' is also allowed. "
                    "For variable_importance data, provide 'p' plus DGP keys, or provide 'mu'/'phi' and 'Sigma'."
                )

            if data_type == "gaussian" and "Sigma" not in setting:
                raise ValueError("For Gaussian data, each setting must contain key 'Sigma'.")

            M_s_tmp = mu_s.size
            Sigma_s = setting.get("Sigma", None)
            if Sigma_s is not None:
                Sigma_s = np.asarray(Sigma_s, dtype=float)
                if Sigma_s.shape != (M_s_tmp, M_s_tmp):
                    raise ValueError(
                        f"Setting {s}: Sigma must have shape {(M_s_tmp, M_s_tmp)}, got {Sigma_s.shape}."
                    )
            elif data_type == "bt_davidson":
                Sigma_s = None

        M_s = mu_s.size

        if M_ref is None:
            M_ref = M_s
        elif M_s != M_ref:
            raise ValueError(f"All settings must have same dimension. Setting 0 M={M_ref}, setting {s} M={M_s}.")

        setting_id = setting.get("setting_id", s)
        setting_label = setting.get("label", f"setting_{s}")

        normalized_settings.append({
            "setting_id": setting_id,
            "setting_label": setting_label,
            "mu": mu_s,
            "Sigma": Sigma_s,
            "X_samples": setting.get("X_samples", None),
            "data_config": cfg_s,
        })

    setting_outs = []
    settings_rows = []
    all_ci_list = []
    all_rep_list = []
    all_length_rep_list = []
    all_subset_list = []
    all_stat_list = []
    all_time_long_list = []

    for s, setting in enumerate(normalized_settings):
        setting_id = setting["setting_id"]
        setting_label = setting["setting_label"]
        mu_s = setting["mu"]
        Sigma_s = setting["Sigma"]
        X_samples_s = setting["X_samples"]
        cfg_s = setting["data_config"]

        utility_fn_s = _default_utility_fn_for_data_type(
            data_type=data_type,
            data_config=cfg_s,
            n_obs=n_obs,
            utility_fn=utility_fn,
        )

        signal_info = signal_strength_fn(
            mu_s,
            Sigma_s,
            k,
            n_obs=n_obs,
            utility_fn=utility_fn_s,
            data_type=data_type,
            data_config=cfg_s,
        )

        setting_meta = {
            "setting_id": setting_id,
            "setting_label": setting_label,
            "signal_strength": signal_info.get("signal_strength", np.nan),
            "standardized_topk_gap": signal_info.get("standardized_topk_gap", np.nan),
            "topk_gap": signal_info.get("topk_gap", np.nan),
            "top1_gap": signal_info.get("top1_gap", np.nan),
            "noise_scale_topk_gap": signal_info.get("noise_scale_topk_gap", np.nan),
            "kth_idx": signal_info.get("kth_idx", np.nan),
            "next_idx": signal_info.get("next_idx", np.nan),
            "true_topk_utility": signal_info.get("true_topk_utility", np.nan),
            "true_topk_subset": signal_info.get("true_topk_subset", None),
            "data_type": data_type,
        }

        if data_type == "bt_davidson":
            bt_cfg = _normalize_bt_config(cfg_s, n_obs=n_obs)
            setting_meta["nu_true"] = bt_cfg["nu_true"]
            setting_meta["n_matches"] = bt_cfg["n_matches"]

        if data_type == "binomial":
            binom_cfg = _normalize_binomial_config(cfg_s, n_obs=n_obs)
            setting_meta["m"] = binom_cfg["m"]
            setting_meta["statistic"] = binom_cfg["statistic"]
            setting_meta["covariance"] = binom_cfg["covariance"]

        if data_type == "variable_importance":
            vi_cfg = _normalize_variable_importance_config(cfg_s, n_obs=n_obs)
            setting_meta["p"] = vi_cfg["p"]
            setting_meta["n_folds"] = vi_cfg["n_folds"]
            setting_meta["covariance"] = vi_cfg["covariance"]
            setting_meta["oracle_reps"] = vi_cfg["oracle_reps"]
            setting_meta["oracle_n_obs"] = vi_cfg["oracle_n_obs"] if vi_cfg["oracle_n_obs"] is not None else n_obs
            setting_meta["oracle_covariance_source"] = vi_cfg["oracle_covariance_source"]

        settings_rows.append({
            **setting_meta,
            "M": int(mu_s.size),
            "k": int(k),
            "B": int(B),
            "n_obs": int(n_obs),
        })

        if verbose:
            print("\n" + "=" * 70)
            print(f"Running parameter setting {s + 1}/{len(normalized_settings)}")
            print(f"setting_id    = {setting_id}")
            print(f"setting_label = {setting_label}")
            print(f"data_type     = {data_type}")
            print(f"signal_strength = {setting_meta['signal_strength']}")
            print("=" * 70)

        out_s = simulate_three_methods_compare_hat(
            mu=mu_s,
            Sigma=Sigma_s,
            k=k,
            B=B,
            n_obs=n_obs,
            epsilon_list=epsilon_list,
            alpha=alpha,
            methods=methods,
            utility_fn=utility_fn_s,
            grid_size=grid_size,
            span=span,
            density_cutoff=density_cutoff,
            seed=seed + 100_000 * s,
            verbose=verbose,
            X_samples=X_samples_s,
            share_same_data_across_methods=share_same_data_across_methods,
            sigma=sigma,
            sel_scale=sel_scale,
            zoom_sigma_mode=zoom_sigma_mode,
            data_type=data_type,
            data_config=cfg_s,
        )

        out_s["setting_id"] = setting_id
        out_s["setting_label"] = setting_label
        out_s["signal_info"] = signal_info
        out_s["signal_strength"] = setting_meta["signal_strength"]
        setting_outs.append(out_s)

        if "all_ci_df" in out_s and out_s["all_ci_df"] is not None:
            all_ci_list.append(_attach_setting_metadata_to_df(out_s["all_ci_df"], setting_meta))
        if "all_rep_df" in out_s and out_s["all_rep_df"] is not None:
            all_rep_list.append(_attach_setting_metadata_to_df(out_s["all_rep_df"], setting_meta))
        if "all_length_rep_df" in out_s and out_s["all_length_rep_df"] is not None:
            all_length_rep_list.append(_attach_setting_metadata_to_df(out_s["all_length_rep_df"], setting_meta))
        if "all_subset_df" in out_s and out_s["all_subset_df"] is not None:
            all_subset_list.append(_attach_setting_metadata_to_df(out_s["all_subset_df"], setting_meta))
        if "time_long" in out_s and out_s["time_long"] is not None:
            all_time_long_list.append(_attach_setting_metadata_to_df(out_s["time_long"], setting_meta))

        if "all_stat_df" in out_s and out_s["all_stat_df"] is not None:
            all_stat_list.append(_attach_setting_metadata_to_df(out_s["all_stat_df"], setting_meta))

    settings_df = pd.DataFrame(settings_rows)
    all_ci_df = pd.concat(all_ci_list, ignore_index=True) if len(all_ci_list) > 0 else pd.DataFrame()
    all_rep_df = pd.concat(all_rep_list, ignore_index=True) if len(all_rep_list) > 0 else pd.DataFrame()
    all_length_rep_df = pd.concat(all_length_rep_list, ignore_index=True) if len(all_length_rep_list) > 0 else pd.DataFrame()
    all_subset_df = pd.concat(all_subset_list, ignore_index=True) if len(all_subset_list) > 0 else pd.DataFrame()
    all_time_long = pd.concat(all_time_long_list, ignore_index=True) if len(all_time_long_list) > 0 else pd.DataFrame()
    all_stat_df = (
        pd.concat(all_stat_list, ignore_index=True)
        if len(all_stat_list) > 0
        else pd.DataFrame()
    )
    

    combined_out = {
        "setting_outs": setting_outs,
        "settings_df": settings_df,
        "all_ci_df": all_ci_df,
        "all_rep_df": all_rep_df,
        "all_length_rep_df": all_length_rep_df,
        "all_subset_df": all_subset_df,
        "all_stat_df": all_stat_df,
        "time_long": all_time_long,
        "methods": list(methods),
        "epsilon_list": [float(eps) for eps in epsilon_list],
        "k": int(k),
        "B": int(B),
        "n_obs": int(n_obs),
        "M": int(M_ref),
        "alpha": float(alpha),
        "sigma": sigma,
        "zoom_sigma_mode": zoom_sigma_mode,
        "data_type": data_type,
        "data_config": dict(data_config or {}),
        "mu_list": [
            _target_mu_from_config(
                s["mu"],
                data_type=data_type,
                data_config=s["data_config"],
                n_obs=n_obs,
            )
            for s in normalized_settings
        ],
        "mu_raw_list": [s["mu"] for s in normalized_settings],
        "Sigma_list": [s["Sigma"] for s in normalized_settings],
    }

    if data_type == "variable_importance":
        combined_out["oracle_list"] = [
            s["data_config"].get("oracle", None)
            for s in normalized_settings
        ]

    return combined_out



