import numpy as np
import pandas as pd

from scipy.optimize import minimize

from .data_generator import (
    _coerce_variable_importance_sample,
    _default_clip_eps,
    _generate_vi_data_from_cfg,
    _logit,
    _normalize_variable_importance_config,
    _vi_sym_ridge,
    btd_mu_covariance,
    compute_T_D_N,
    estimate_binomial_gaussian_input,
    estimate_variable_importance_statistic,
    generate_binomial_dosage_data,
    generate_variable_importance_data,
    generate_variable_importance_replicates,
    make_binomial_selection_utility,
    mle_btd,
    simulate_bt_davidson,
    true_target_and_oracle_covariance,
)

def build_time_wide_table(time_long, B, *, methods=None, epsilon_list=None):
    if time_long is None or len(time_long) == 0:
        return pd.DataFrame(columns=np.arange(B))

    df = time_long.copy()

    if "epsilon" not in df.columns:
        df["epsilon"] = np.nan

    def make_label(row):
        method = str(row["method"])
        eps = row["epsilon"]

        if method == "randomized":
            return f"randomized_eps={float(eps):g}"

        return method

    df["method_epsilon"] = df.apply(make_label, axis=1)

    wide = (
        df.pivot_table(
            index="method_epsilon",
            columns="rep",
            values="time",
            aggfunc="first",
        )
        .reindex(columns=np.arange(B))
    )

    wide.index.name = "method_epsilon"

    # keep row order consistent with methods and epsilon_list
    if methods is not None:
        ordered_rows = []

        for m in methods:
            if m == "randomized":
                if epsilon_list is not None:
                    for eps in epsilon_list:
                        ordered_rows.append(f"randomized_eps={float(eps):g}")
            else:
                ordered_rows.append(m)

        ordered_rows = [x for x in ordered_rows if x in wide.index]
        wide = wide.reindex(ordered_rows)

    return wide



def _validate_sigma_mode(sigma):
    sigma = str(sigma).lower()
    if sigma not in {"known", "unknown"}:
        raise ValueError("sigma must be either 'known' or 'unknown'.")
    return sigma
    

def _validate_data_type(data_type):
    dt = str(data_type).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "normal": "gaussian",
        "gaussian": "gaussian",
        "bt": "bt_davidson",
        "bradley_terry": "bt_davidson",
        "bradley_terry_davidson": "bt_davidson",
        "btd": "bt_davidson",
        "bt_davidson": "bt_davidson",
        "davidson": "bt_davidson",
        "binom": "binomial",
        "binomial": "binomial",
        "binomial_dosage": "binomial",
        "dosage": "binomial",
        "vi": "variable_importance",
        "vim": "variable_importance",
        "variable_importance": "variable_importance",
        "feature_importance": "variable_importance",
    }
    if dt not in aliases:
        raise ValueError(
            "data_type must be one of {'gaussian', 'bt_davidson', 'binomial', 'variable_importance'} "
            f"or their aliases; got {data_type!r}."
        )
    return aliases[dt]


# ---------------------------------------------------------
# 2) Add these binomial config / target / utility helpers
# ---------------------------------------------------------

def _normalize_binomial_config(data_config, *, n_obs=None):
    cfg = dict(data_config or {})

    m = cfg.get("m", cfg.get("n_trials", None))
    if m is None:
        if n_obs is None:
            raise ValueError(
                "For binomial data, provide data_config={'m': ...} or use n_obs as m."
            )
        m = int(n_obs)
    m = int(m)
    if m <= 0:
        raise ValueError("For binomial data, m must be positive.")

    statistic = str(
        cfg.get("statistic", "log_success")
    ).lower().strip()
    
    if statistic not in {
        "probability",
        "log_success",
        "logit",
    }:
        raise ValueError(
            "For binomial data, statistic must be one of "
            "{'probability', 'log_success', 'logit'}."
        )

    covariance = str(cfg.get("covariance", "plugin"))
    if covariance not in {"plugin", "oracle"}:
        raise ValueError("For binomial data, covariance must be 'plugin' or 'oracle'.")

    split_frac = float(cfg.get("split_frac", cfg.get("binomial_split_frac", 0.5)))
    if not (0.0 < split_frac < 1.0):
        raise ValueError("For binomial data, split_frac must be in (0, 1).")

    return {
        **cfg,
        "m": m,
        "statistic": statistic,
        "covariance": covariance,
        "clip_eps": cfg.get("clip_eps", None),
        "split_frac": split_frac,
        "cov_ridge": float(cfg.get("cov_ridge", 0.0)),
    }


def _target_mu_for_data_type(
    mu,
    *,
    data_type,
    center_bt=True,
    binomial_statistic="log_success",
):
    dt = _validate_data_type(data_type)
    mu = np.asarray(mu, dtype=float).reshape(-1)

    if dt == "bt_davidson" and center_bt:
        return mu - float(np.mean(mu))

    if dt == "binomial":
        p = mu.copy()
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("For binomial data, the input mu/p must be strictly between 0 and 1.")
        if binomial_statistic in {"probability","log_success",}:
            return p
        if binomial_statistic == "logit":
            return _logit(p)
        raise ValueError("binomial_statistic must be 'log_success' or 'logit'.")

    if dt == "variable_importance":
        return mu

    return mu


def _target_mu_from_config(mu, *, data_type, data_config=None, n_obs=None):
    dt = _validate_data_type(data_type)

    if dt == "bt_davidson":
        cfg = _normalize_bt_config(data_config, n_obs=n_obs)
        return _target_mu_for_data_type(
            mu,
            data_type=dt,
            center_bt=cfg["center_bt"],
        )

    if dt == "binomial":
        cfg = _normalize_binomial_config(data_config, n_obs=n_obs)
        return _target_mu_for_data_type(
            mu,
            data_type=dt,
            binomial_statistic=cfg["statistic"],
        )

    return _target_mu_for_data_type(mu, data_type=dt)


def _default_utility_fn_for_data_type(*, data_type, data_config=None, n_obs=None, utility_fn=None):
    if utility_fn is not None:
        return utility_fn

    dt = _validate_data_type(data_type)
    if dt != "binomial":
        return None

    cfg = _normalize_binomial_config(data_config, n_obs=n_obs)
    clip_eps = cfg["clip_eps"]
    if clip_eps is None:
        clip_eps = _default_clip_eps(cfg["m"])

    return make_binomial_selection_utility(
        statistic=cfg["statistic"],
        clip_eps=clip_eps,
    )


# ---------------------------------------------------------
# 3) Add binomial replicate/splitting helpers
# ---------------------------------------------------------

def generate_binomial_replicates(p, m, B, seed=0):
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float).reshape(-1)
    out = []
    for _ in range(int(B)):
        out.append(generate_binomial_dosage_data(p=p, m=int(m), rng=rng))
    return out


def split_binomial_counts(S, m, *, frac=0.5, rng=None):
    """
    Split aggregated Binomial counts S_j out of m trials into selection and
    inference counts using a hypergeometric split, preserving S_sel + S_inf = S.
    """
    if rng is None:
        rng = np.random.default_rng()

    S = np.asarray(S, dtype=int).reshape(-1)
    m = int(m)
    frac = float(frac)

    if m < 2:
        raise ValueError("m must be at least 2 for binomial data splitting.")
    if np.any(S < 0) or np.any(S > m):
        raise ValueError("Binomial counts S must satisfy 0 <= S_j <= m.")

    m_sel = int(np.floor(frac * m))
    m_sel = max(1, min(m - 1, m_sel))
    m_inf = m - m_sel

    S_sel = np.array([
        rng.hypergeometric(
            ngood=int(s),
            nbad=int(m - s),
            nsample=int(m_sel),
        )
        for s in S
    ], dtype=int)
    S_inf = S - S_sel

    return S_sel, S_inf, m_sel, m_inf


# ---------------------------------------------------------
# 4) Replace validate_data_samples with this version
# ---------------------------------------------------------

def validate_data_samples(data_samples, B, n_obs, M, *, data_type, name="X_samples"):
    dt = _validate_data_type(data_type)

    if data_samples is None:
        return None

    if dt == "gaussian":
        return validate_X_samples(data_samples, B, n_obs, M, name=name)

    if dt == "bt_davidson":
        if not isinstance(data_samples, (list, tuple)):
            raise TypeError(f"For BT data, {name} must be a list/tuple of DataFrames.")
        if len(data_samples) != int(B):
            raise ValueError(f"For BT data, {name} must have length B={B}, got {len(data_samples)}.")
        required = {"player_i", "player_j", "W_ij", "W_ji", "D_ij", "n_ij"}
        for b, df in enumerate(data_samples):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"BT sample {b} is not a pandas DataFrame.")
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"BT sample {b} is missing columns: {missing}")
        return list(data_samples)

    if dt == "variable_importance":
        if not isinstance(data_samples, (list, tuple)):
            raise TypeError(
                f"For variable_importance data, {name} must be a list of dicts "
                "with keys 'X' and 'O', or a list of tuples (X, O)."
            )
        if len(data_samples) != int(B):
            raise ValueError(
                f"For variable_importance data, {name} must have length B={B}, got {len(data_samples)}."
            )
        out = []
        for b, obj in enumerate(data_samples):
            X, O = _coerce_variable_importance_sample(obj, p=M, name=f"{name}[{b}]")
            if X.shape[0] != int(n_obs):
                raise ValueError(
                    f"For variable_importance data, {name}[{b}].X must have n_obs={n_obs} rows, got {X.shape[0]}."
                )
            out.append({"X": X, "O": O})
        return out

    arr = np.asarray(data_samples, dtype=object)
    if arr.ndim == 2 and arr.shape == (int(B), int(M)):
        return [np.asarray(arr[b], dtype=int).reshape(-1) for b in range(int(B))]

    if not isinstance(data_samples, (list, tuple)):
        raise TypeError(f"For binomial data, {name} must be an array with shape (B, M) or a list of count vectors.")
    if len(data_samples) != int(B):
        raise ValueError(f"For binomial data, {name} must have length B={B}, got {len(data_samples)}.")

    out = []
    for b, S in enumerate(data_samples):
        S = np.asarray(S, dtype=int).reshape(-1)
        if S.shape[0] != int(M):
            raise ValueError(f"Binomial sample {b} must have length M={M}, got {S.shape[0]}.")
        if np.any(S < 0):
            raise ValueError(f"Binomial sample {b} contains negative counts.")
        out.append(S)
    return out



# =========================================================
# Unified statistic / covariance helpers for all data types
# =========================================================

def _normalize_bt_config(data_config=None, *, n_obs=None):
    """
    Normalize Bradley--Terry--Davidson configuration.

    Required/optional fields:
        nu or nu_true: draw/tie parameter
        n_matches or bt_n_matches: number of matches per pair
        center_bt: whether to center mu
        split_frac: split fraction for data splitting
        cov_ridge: ridge added to covariance
    """
    cfg = dict(data_config or {})

    nu_true = cfg.get("nu_true", cfg.get("nu", 1.0))
    nu_true = float(nu_true)

    if nu_true <= 0:
        raise ValueError("For BT-Davidson data, nu_true/nu must be positive.")

    n_matches = cfg.get("n_matches", cfg.get("bt_n_matches", None))

    if n_matches is None:
        if n_obs is None:
            raise ValueError(
                "For BT-Davidson data, provide n_matches/bt_n_matches "
                "or pass n_obs."
            )
        n_matches = int(n_obs)

    if np.isscalar(n_matches):
        n_matches = int(n_matches)
        if n_matches <= 0:
            raise ValueError("For BT-Davidson data, n_matches must be positive.")
    else:
        n_matches = np.asarray(n_matches, dtype=int)
        if np.any(n_matches < 0):
            raise ValueError("For BT-Davidson data, n_matches matrix cannot have negative entries.")

    center_bt = bool(cfg.get("center_bt", True))

    split_frac = float(cfg.get("split_frac", 0.5))
    if not (0.0 < split_frac < 1.0):
        raise ValueError("For BT-Davidson data, split_frac must be in (0, 1).")

    cov_ridge = float(cfg.get("cov_ridge", 1e-8))

    return {
        **cfg,
        "nu_true": nu_true,
        "nu": nu_true,
        "n_matches": n_matches,
        "bt_n_matches": n_matches,
        "center_bt": center_bt,
        "split_frac": split_frac,
        "cov_ridge": cov_ridge,
    }


def generate_bt_davidson_replicates(mu, nu, n_matches, B, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float).reshape(-1)

    out = []
    for _ in range(int(B)):
        df = simulate_bt_davidson(
            mu=mu,
            nu=nu,
            n_matches=n_matches,
            rng=rng,
        )
        out.append(df)

    return out



import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _resolve_covariance_mode(sigma="known", data_config=None):
    """
    Resolve covariance mode.

    For Gaussian data:
        use sigma directly: "known" or "unknown".

    For non-Gaussian data:
        data_config["covariance"] can override sigma.
        accepted:
            oracle / known      -> known
            plugin / estimated / unknown -> unknown
    """
    sigma = _validate_sigma_mode(sigma)
    cfg = dict(data_config or {})
    cov = str(cfg.get("covariance", sigma)).lower()

    if cov in {"oracle", "known", "true"}:
        return "known"
    if cov in {"plugin", "estimated", "estimate", "unknown"}:
        return "unknown"

    return sigma


def _add_cov_ridge(Sigma, ridge=0.0):
    Sigma = np.asarray(Sigma, dtype=float)
    if ridge is None:
        ridge = 0.0
    ridge = float(ridge)
    if ridge > 0:
        Sigma = Sigma + ridge * np.eye(Sigma.shape[0])
    return Sigma


def _gaussian_rep_stat_cov(
    *,
    b,
    rng,
    mu,
    Sigma,
    n_obs,
    data_samples=None,
    sigma="known",
):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    M = mu.size

    if data_samples is None:
        X = rng.multivariate_normal(mean=mu, cov=Sigma, size=int(n_obs))
    else:
        X = np.asarray(data_samples[b], dtype=float)
        if X.shape != (int(n_obs), M):
            raise ValueError(f"Gaussian sample must have shape {(int(n_obs), M)}, got {X.shape}")

    X_hat = X.mean(axis=0)

    if sigma == "known":
        # covariance of sample mean
        Sigma_hat = Sigma / float(n_obs)
        sigma2_hat = float(np.mean(np.diag(Sigma)))
    else:
        # diagonal plug-in covariance of sample mean
        s2 = X.var(axis=0, ddof=1)
        Sigma_hat = np.diag(s2 / float(n_obs))
        sigma2_hat = float(np.mean(s2))

    meta = {}
    return X_hat, Sigma_hat, sigma2_hat, X, meta


def _binomial_rep_stat_cov(
    *,
    b,
    rng,
    p,
    n_obs,
    data_config=None,
    data_samples=None,
    sigma="known",
):
    cfg = _normalize_binomial_config(data_config, n_obs=n_obs)
    m = int(cfg["m"])
    statistic = cfg.get("statistic", "log_success")
    clip_eps = cfg.get("clip_eps", None)
    cov_ridge = float(cfg.get("cov_ridge", 0.0))
    p = np.asarray(p, dtype=float).reshape(-1)
    M = p.size

    if data_samples is None:
        S = generate_binomial_dosage_data(p, m=m, rng=rng)
    else:
        arr = np.asarray(data_samples)
        if arr.ndim == 2:
            S = np.asarray(arr[b], dtype=int).reshape(-1)# shape: (B, M), already binomial counts
        elif arr.ndim == 3:
            S = np.asarray(arr[b], dtype=int).sum(axis=0).reshape(-1)# shape: (B, m, M), raw Bernoulli trials
        else:
            raise ValueError("Binomial data_samples must have shape (B, M) or (B, m, M).")

        if S.size != M:
            raise ValueError(f"Binomial count vector must have length {M}, got {S.size}")

    est = estimate_binomial_gaussian_input(S, m=m, statistic=statistic, clip_eps=clip_eps)
    X_hat = est["X"]

    if sigma == "known":
        target_mu, Sigma_hat = true_target_and_oracle_covariance(p, m=m, statistic=statistic)
    else:
        target_mu = None
        Sigma_hat = est["Sigma_target"]

    Sigma_hat = _add_cov_ridge(Sigma_hat, ridge=cov_ridge)
    sigma2_hat = float(np.mean(np.diag(Sigma_hat)))

    meta = {
        "m": int(m),
        "statistic": statistic,
        "covariance": "oracle" if sigma == "known" else "plugin",
        "clip_eps": float(est["clip_eps"]),
    }

    return X_hat, Sigma_hat, sigma2_hat, S, meta


def _make_bt_design_N(p, n_matches):
    """
    Construct pairwise match-count matrix N from scalar or matrix n_matches.
    """
    if np.isscalar(n_matches):
        N = np.zeros((p, p), dtype=int)
        for i in range(p):
            for j in range(i + 1, p):
                N[i, j] = int(n_matches)
                N[j, i] = int(n_matches)
        return N

    N = np.asarray(n_matches, dtype=int)
    if N.shape != (p, p):
        raise ValueError(f"n_matches matrix must have shape {(p, p)}, got {N.shape}")
    return N


def _btd_covariance_from_params(N, mu_param, nu_param, cov_ridge=1e-8):
    """
    Fisher-information covariance for BTD ability MLE.
    This uses the same formula as btd_mu_covariance, but with supplied parameters.
    """
    mu_param = np.asarray(mu_param, dtype=float).reshape(-1)
    Sigma_mu = btd_mu_covariance(N=N, mu_hat=mu_param, nu_hat=float(nu_param))
    Sigma_mu = _add_cov_ridge(Sigma_mu, ridge=cov_ridge)
    return Sigma_mu


def _bt_rep_stat_cov(
    *,
    b,
    rng,
    mu,
    n_obs,
    data_config=None,
    data_samples=None,
    sigma="known",
):
    cfg = _normalize_bt_config(data_config, n_obs=n_obs)

    mu = np.asarray(mu, dtype=float).reshape(-1)
    M = mu.size

    center_bt = bool(cfg.get("center_bt", True))
    nu_true = float(cfg.get("nu_true", cfg.get("nu", 1.0)))
    n_matches = cfg.get("n_matches", cfg.get("bt_n_matches", n_obs))
    cov_ridge = float(cfg.get("cov_ridge", 1e-8))

    mu_true = mu.copy()
    if center_bt:
        mu_true = mu_true - mu_true.mean()

    if data_samples is None:
        df = simulate_bt_davidson(
            mu=mu_true,
            nu=nu_true,
            n_matches=n_matches,
            rng=rng,
        )
    else:
        obj = data_samples[b]
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, dict) and "df" in obj:
            df = obj["df"].copy()
        else:
            raise ValueError("BT data_samples must be a list of DataFrames or dicts with key 'df'.")

    T, D_total, N = compute_T_D_N(df, p=M)
    mu_hat, nu_hat = mle_btd(T, N, D_total)

    if center_bt:
        mu_hat = mu_hat - mu_hat.mean()

    X_hat = mu_hat

    if sigma == "known":
        Sigma_hat = _btd_covariance_from_params(
            N=N,
            mu_param=mu_true,
            nu_param=nu_true,
            cov_ridge=cov_ridge,
        )
    else:
        Sigma_hat = btd_mu_covariance(N=N, mu_hat=mu_hat, nu_hat=nu_hat)
        Sigma_hat = _add_cov_ridge(Sigma_hat, ridge=cov_ridge)

    sigma2_hat = float(np.mean(np.diag(Sigma_hat)))

    meta = {
        "nu_true": float(nu_true),
        "nu_hat": float(nu_hat),
        "n_matches": n_matches if np.isscalar(n_matches) else "matrix",
        "covariance": "oracle" if sigma == "known" else "plugin",
    }

    return X_hat, Sigma_hat, sigma2_hat, df, meta


def _variable_importance_rep_stat_cov(
    *,
    b,
    rng,
    phi,
    Sigma_vim,
    n_obs,
    data_config=None,
    data_samples=None,
    sigma="known",
):
    cfg = _normalize_variable_importance_config(data_config, n_obs=n_obs)
    p = int(cfg["p"])

    if data_samples is None:
        X, O = _generate_vi_data_from_cfg(cfg, int(n_obs), rng)
    else:
        X, O = _coerce_variable_importance_sample(
            data_samples[b],
            p=p,
            name=f"VI sample {b}",
        )

    cfg_for_stat = dict(cfg)
    cfg_for_stat["mu"] = np.asarray(phi, dtype=float).reshape(-1)
    cfg_for_stat["phi"] = np.asarray(phi, dtype=float).reshape(-1)
    cfg_for_stat["true_vi_score"] = np.asarray(phi, dtype=float).reshape(-1)

    est = estimate_variable_importance_statistic(
        X,
        O,
        cfg=cfg_for_stat,
        random_state=(
            int(cfg.get("random_state_offset", 0))
            + 1000000
            + 1009 * int(b)
        ),
    )

    X_hat = np.asarray(est["phi_hat"], dtype=float).reshape(-1)

    if sigma == "known":
        if Sigma_vim is None:
            oracle = cfg.get("oracle", None)
            if oracle is None or "Sigma_vim" not in oracle:
                raise ValueError(
                    "For variable_importance with sigma='known', provide "
                    "Sigma or data_config['oracle']['Sigma_vim']."
                )
            Sigma_vim = oracle["Sigma_vim"]

        Sigma_hat = np.asarray(Sigma_vim, dtype=float) / float(n_obs)
        covariance_label = "holdout_bootstrap"

    else:
        Sigma_hat = np.asarray(est["Sigma_hat"], dtype=float)
        covariance_label = "plugin"

    Sigma_hat = _vi_sym_ridge(Sigma_hat, ridge=0.0)
    sigma2_hat = float(np.mean(np.diag(Sigma_hat)))

    if X_hat.size != p:
        raise ValueError(f"VI X_hat has length {X_hat.size}, expected p={p}.")
    if Sigma_hat.shape != (p, p):
        raise ValueError(f"VI Sigma_hat must have shape {(p, p)}, got {Sigma_hat.shape}.")

    meta = {
        "data_type": "variable_importance",
        "covariance": covariance_label,
        "n_folds": int(cfg["n_folds"]),
        "tree_params": cfg.get("tree_params", None),
        "vi_statistic": cfg.get("vi_statistic", None),
        "vi_statistic_used": est.get("vi_statistic_used", None),
        "fixed_nuisance_learner": est.get("fixed_nuisance_learner", None),
    }

    return X_hat, Sigma_hat, sigma2_hat, {"X": X, "O": O}, meta


def _get_rep_stat_cov(
    *,
    b,
    rng,
    mu,
    Sigma,
    n_obs,
    data_type="gaussian",
    data_config=None,
    data_samples=None,
    sigma="known",
):
    data_type = _validate_data_type(data_type)
    cov_mode = _resolve_covariance_mode(sigma=sigma, data_config=data_config)

    if data_type == "gaussian":
        return _gaussian_rep_stat_cov(
            b=b,
            rng=rng,
            mu=mu,
            Sigma=Sigma,
            n_obs=n_obs,
            data_samples=data_samples,
            sigma=cov_mode,
        )

    if data_type == "binomial":
        return _binomial_rep_stat_cov(
            b=b,
            rng=rng,
            p=mu,
            n_obs=n_obs,
            data_config=data_config,
            data_samples=data_samples,
            sigma=cov_mode,
        )

    if data_type == "bt_davidson":
        return _bt_rep_stat_cov(
            b=b,
            rng=rng,
            mu=mu,
            n_obs=n_obs,
            data_config=data_config,
            data_samples=data_samples,
            sigma=cov_mode,
        )

    if data_type == "variable_importance":
        return _variable_importance_rep_stat_cov(
            b=b,
            rng=rng,
            phi=mu,
            Sigma_vim=Sigma,
            n_obs=n_obs,
            data_config=data_config,
            data_samples=data_samples,
            sigma=cov_mode,
        )

    raise ValueError(f"Unsupported data_type: {data_type}")

# ---------------------------------------------------------
# 6) Replace _get_data_splitting_stat_cov with this version
# ---------------------------------------------------------

# =========================================================
# Unified data-splitting helper
# =========================================================

def _split_binomial_counts(S, m_sel, rng):
    """
    Split total binomial counts S into selection and inference counts.
    Conditional on total S, use hypergeometric thinning.
    """
    S = np.asarray(S, dtype=int).reshape(-1)
    m_total = None
    S_sel = np.zeros_like(S)

    # m_total is inferred outside from m_sel + m_inf
    return S_sel


def _get_data_splitting_stat_cov(
    *,
    b,
    rng,
    mu,
    Sigma,
    n_obs,
    data_type="gaussian",
    data_config=None,
    data_samples=None,
    sigma="known",
):
    data_type = _validate_data_type(data_type)
    cov_mode = _resolve_covariance_mode(sigma=sigma, data_config=data_config)

    mu = np.asarray(mu, dtype=float).reshape(-1)
    M = mu.size

    if data_type == "gaussian":
        n_obs_sel = int(n_obs) // 2
        n_obs_inf = int(n_obs) - n_obs_sel
        if n_obs_sel <= 0 or n_obs_inf <= 0:
            raise ValueError("Both selection and inference sample sizes must be positive.")

        Sigma = np.asarray(Sigma, dtype=float)

        if data_samples is None:
            X = rng.multivariate_normal(mean=mu, cov=Sigma, size=int(n_obs))
        else:
            X = np.asarray(data_samples[b], dtype=float)

        X_sel = X[:n_obs_sel]
        X_inf = X[n_obs_sel:]

        X_hat_sel = X_sel.mean(axis=0)
        X_hat_inf = X_inf.mean(axis=0)

        if cov_mode == "known":
            Sigma_hat_inf = Sigma / float(n_obs_inf)
            sigma2_hat_inf = float(np.mean(np.diag(Sigma)))
        else:
            s2_inf = X_inf.var(axis=0, ddof=1)
            Sigma_hat_inf = np.diag(s2_inf / float(n_obs_inf))
            sigma2_hat_inf = float(np.mean(s2_inf))

        return {
            "X_hat_sel": X_hat_sel,
            "X_hat_inf": X_hat_inf,
            "Sigma_hat_inf": Sigma_hat_inf,
            "sigma2_hat_inf": sigma2_hat_inf,
            "n_obs_sel": n_obs_sel,
            "n_obs_inf": n_obs_inf,
        }

    if data_type == "binomial":
        cfg = _normalize_binomial_config(data_config, n_obs=n_obs)
        m = int(cfg["m"])
        statistic = cfg.get("statistic", "log_success")
        clip_eps = cfg.get("clip_eps", None)
        cov_ridge = float(cfg.get("cov_ridge", 0.0))

        split_frac = float(cfg.get("split_frac", cfg.get("binomial_split_frac", 0.5)))
        m_sel = int(np.floor(m * split_frac))
        m_inf = int(m - m_sel)

        if m_sel <= 0 or m_inf <= 0:
            raise ValueError("Both binomial split parts must have positive trial counts.")

        p = mu.copy()

        if data_samples is None:
            S_sel = rng.binomial(n=m_sel, p=p)
            S_inf = rng.binomial(n=m_inf, p=p)
        else:
            arr = np.asarray(data_samples)

            if arr.ndim == 3:
                Y = np.asarray(arr[b], dtype=int)
                S_sel = Y[:m_sel].sum(axis=0)
                S_inf = Y[m_sel:].sum(axis=0)

            elif arr.ndim == 2:
                S_total = np.asarray(arr[b], dtype=int).reshape(-1)
                S_sel = np.zeros(M, dtype=int)
                for j in range(M):
                    S_sel[j] = rng.hypergeometric(
                        ngood=int(S_total[j]),
                        nbad=int(m - S_total[j]),
                        nsample=int(m_sel),
                    )
                S_inf = S_total - S_sel

            else:
                raise ValueError("Binomial data_samples must have shape (B, M) or (B, m, M).")

        est_sel = estimate_binomial_gaussian_input(S_sel, m=m_sel, statistic=statistic, clip_eps=clip_eps)
        est_inf = estimate_binomial_gaussian_input(S_inf, m=m_inf, statistic=statistic, clip_eps=clip_eps)

        X_hat_sel = est_sel["X"]
        X_hat_inf = est_inf["X"]

        if cov_mode == "known":
            _, Sigma_hat_inf = true_target_and_oracle_covariance(p, m=m_inf, statistic=statistic)
        else:
            Sigma_hat_inf = est_inf["Sigma_target"]

        Sigma_hat_inf = _add_cov_ridge(Sigma_hat_inf, ridge=cov_ridge)
        sigma2_hat_inf = float(np.mean(np.diag(Sigma_hat_inf)))

        return {
            "X_hat_sel": X_hat_sel,
            "X_hat_inf": X_hat_inf,
            "Sigma_hat_inf": Sigma_hat_inf,
            "sigma2_hat_inf": sigma2_hat_inf,
            "n_obs_sel": int(m_sel),
            "n_obs_inf": int(m_inf),
        }

    if data_type == "bt_davidson":
        cfg = _normalize_bt_config(data_config, n_obs=n_obs)
        center_bt = bool(cfg.get("center_bt", True))
        nu_true = float(cfg.get("nu_true", cfg.get("nu", 1.0)))
        n_matches = cfg.get("n_matches", cfg.get("bt_n_matches", n_obs))
        cov_ridge = float(cfg.get("cov_ridge", 1e-8))
        split_frac = float(cfg.get("split_frac", 0.5))

        mu_true = mu.copy()
        if center_bt:
            mu_true = mu_true - mu_true.mean()

        if data_samples is None:
            df_full = simulate_bt_davidson(
                mu=mu_true,
                nu=nu_true,
                n_matches=n_matches,
                rng=rng,
            )
        else:
            obj = data_samples[b]
            df_full = obj.copy() if isinstance(obj, pd.DataFrame) else obj["df"].copy()

        sel_rows = []
        inf_rows = []

        for _, row in df_full.iterrows():
            counts = np.array([row["W_ij"], row["W_ji"], row["D_ij"]], dtype=int)
            n_ij = int(counts.sum())
            n_sel = int(np.floor(split_frac * n_ij))

            if n_sel <= 0:
                sel_counts = np.zeros(3, dtype=int)
            elif n_sel >= n_ij:
                sel_counts = counts.copy()
            else:
                sel_counts = rng.multivariate_hypergeometric(counts, n_sel)

            inf_counts = counts - sel_counts

            base = {
                "player_i": int(row["player_i"]),
                "player_j": int(row["player_j"]),
            }

            sel_rows.append({
                **base,
                "W_ij": int(sel_counts[0]),
                "W_ji": int(sel_counts[1]),
                "D_ij": int(sel_counts[2]),
                "n_ij": int(sel_counts.sum()),
            })

            inf_rows.append({
                **base,
                "W_ij": int(inf_counts[0]),
                "W_ji": int(inf_counts[1]),
                "D_ij": int(inf_counts[2]),
                "n_ij": int(inf_counts.sum()),
            })

        df_sel = pd.DataFrame(sel_rows)
        df_inf = pd.DataFrame(inf_rows)

        T_sel, D_sel, N_sel = compute_T_D_N(df_sel, p=M)
        T_inf, D_inf, N_inf = compute_T_D_N(df_inf, p=M)

        mu_hat_sel, nu_hat_sel = mle_btd(T_sel, N_sel, D_sel)
        mu_hat_inf, nu_hat_inf = mle_btd(T_inf, N_inf, D_inf)

        if center_bt:
            mu_hat_sel = mu_hat_sel - mu_hat_sel.mean()
            mu_hat_inf = mu_hat_inf - mu_hat_inf.mean()

        if cov_mode == "known":
            Sigma_hat_inf = _btd_covariance_from_params(
                N=N_inf,
                mu_param=mu_true,
                nu_param=nu_true,
                cov_ridge=cov_ridge,
            )
        else:
            Sigma_hat_inf = btd_mu_covariance(N=N_inf, mu_hat=mu_hat_inf, nu_hat=nu_hat_inf)
            Sigma_hat_inf = _add_cov_ridge(Sigma_hat_inf, ridge=cov_ridge)

        return {
            "X_hat_sel": mu_hat_sel,
            "X_hat_inf": mu_hat_inf,
            "Sigma_hat_inf": Sigma_hat_inf,
            "sigma2_hat_inf": float(np.mean(np.diag(Sigma_hat_inf))),
            "n_obs_sel": int(df_sel["n_ij"].sum()),
            "n_obs_inf": int(df_inf["n_ij"].sum()),
        }

    if data_type == "variable_importance":
        cfg = _normalize_variable_importance_config(data_config, n_obs=n_obs)
        p = int(cfg["p"])

        n_obs_sel = int(np.floor(float(cfg["split_frac"]) * int(n_obs)))
        n_obs_sel = max(2, min(int(n_obs) - 2, n_obs_sel))
        n_obs_inf = int(n_obs) - n_obs_sel

        if data_samples is None:
            X_full, O_full = generate_variable_importance_data(
                int(n_obs),
                p=p,
                f=cfg.get("f", None),
                beta=cfg.get("beta", None),
                x_dist=cfg.get("x_dist", "normal"),
                rho=cfg.get("rho", 0.0),
                x_mean=cfg.get("x_mean", None),
                x_scale=cfg.get("x_scale", 1.0),
                x_low=cfg.get("x_low", -1.0),
                x_high=cfg.get("x_high", 1.0),
                noise_sd=cfg.get("noise_sd", 1.0),
                noise_dist=cfg.get("noise_dist", "normal"),
                rng=rng,
            )
        else:
            X_full, O_full = _coerce_variable_importance_sample(
                data_samples[b],
                p=p,
                name=f"VI sample {b}",
            )

        X_sel, O_sel = X_full[:n_obs_sel], O_full[:n_obs_sel]
        X_inf, O_inf = X_full[n_obs_sel:], O_full[n_obs_sel:]

        cfg_for_stat_sel = dict(cfg)
        cfg_for_stat_sel["mu"] = np.asarray(mu, dtype=float).reshape(-1)
        cfg_for_stat_sel["phi"] = np.asarray(mu, dtype=float).reshape(-1)
        
        if "true_vi_score" not in cfg_for_stat_sel:
            cfg_for_stat_sel["true_vi_score"] = np.asarray(mu, dtype=float).reshape(-1)
        
        cfg_for_stat_sel["n_folds"] = min(cfg["n_folds"], n_obs_sel)
        
        est_sel = estimate_variable_importance_statistic(
            X_sel,
            O_sel,
            cfg=cfg_for_stat_sel,
            random_state=(
                int(cfg.get("random_state_offset", 0))
                + 2000000
                + 1009 * int(b)
                + 1
            ),
        )
        
        
        cfg_for_stat_inf = dict(cfg)
        cfg_for_stat_inf["mu"] = np.asarray(mu, dtype=float).reshape(-1)
        cfg_for_stat_inf["phi"] = np.asarray(mu, dtype=float).reshape(-1)
        
        if "true_vi_score" not in cfg_for_stat_inf:
            cfg_for_stat_inf["true_vi_score"] = np.asarray(mu, dtype=float).reshape(-1)
        
        cfg_for_stat_inf["n_folds"] = min(cfg["n_folds"], n_obs_inf)
        
        est_inf = estimate_variable_importance_statistic(
            X_inf,
            O_inf,
            cfg=cfg_for_stat_inf,
            random_state=(
                int(cfg.get("random_state_offset", 0))
                + 2000000
                + 1009 * int(b)
                + 2
            ),
        )

        X_hat_sel = est_sel["phi_hat"]
        X_hat_inf = est_inf["phi_hat"]

        if cov_mode == "known":
            if Sigma is None:
                oracle = cfg.get("oracle", None)
                if oracle is None or "Sigma_vim" not in oracle:
                    raise ValueError(
                        "For variable_importance data splitting with sigma='known', provide Sigma or data_config['oracle']['Sigma_vim']."
                    )
                Sigma_vim = np.asarray(oracle["Sigma_vim"], dtype=float)
            else:
                Sigma_vim = np.asarray(Sigma, dtype=float)

            Sigma_hat_inf = Sigma_vim / float(n_obs_inf)
            covariance_label = "oracle"
        else:
            Sigma_hat_inf = np.asarray(est_inf["Sigma_hat"], dtype=float)
            covariance_label = "plugin"

        Sigma_hat_inf = _vi_sym_ridge(Sigma_hat_inf, ridge=0.0)

        return {
            "X_hat_sel": X_hat_sel,
            "X_hat_inf": X_hat_inf,
            "Sigma_hat_inf": Sigma_hat_inf,
            "sigma2_hat_inf": float(np.mean(np.diag(Sigma_hat_inf))),
            "n_obs_sel": int(n_obs_sel),
            "n_obs_inf": int(n_obs_inf),
            "covariance": covariance_label,
        }

    raise ValueError(f"Unsupported data_type: {data_type}")


# ---------------------------------------------------------
# 7) Replace _prepare_shared_samples with this version
# ---------------------------------------------------------

def _prepare_shared_samples(
    *,
    mu,
    Sigma,
    B,
    n_obs,
    seed,
    data_type,
    data_config=None,
    X_samples=None,
    share_same_data_across_methods=False,
):
    dt = _validate_data_type(data_type)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    M = mu.size

    if X_samples is not None:
        return validate_data_samples(
            X_samples,
            B,
            n_obs,
            M,
            data_type=dt,
            name="X_samples",
        )

    if not share_same_data_across_methods:
        return None

    if dt == "gaussian":
        return generate_gaussian_replicates(
            mu=mu,
            Sigma=Sigma,
            B=B,
            n_obs=n_obs,
            seed=seed + 999,
        )

    if dt == "bt_davidson":
        cfg = _normalize_bt_config(data_config, n_obs=n_obs)
        return generate_bt_davidson_replicates(
            mu=mu,
            nu=cfg["nu_true"],
            n_matches=cfg["n_matches"],
            B=B,
            seed=seed + 999,
        )

    if dt == "binomial":
        cfg = _normalize_binomial_config(data_config, n_obs=n_obs)
        return generate_binomial_replicates(
            p=mu,
            m=cfg["m"],
            B=B,
            seed=seed + 999,
        )

    if dt == "variable_importance":
        cfg = _normalize_variable_importance_config(data_config, n_obs=n_obs)
        return generate_variable_importance_replicates(
            cfg,
            n_obs=n_obs,
            B=B,
            seed=seed + 999,
        )

    raise ValueError(f"Unsupported data_type: {dt}")


# ---------------------------------------------------------
# 8) Replace default_signal_strength_general with this version
# ---------------------------------------------------------

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

    utility_fn = _default_utility_fn_for_data_type(
        data_type=data_type,
        data_config=data_config,
        n_obs=n_obs,
        utility_fn=utility_fn,
    )

    target_mu = _target_mu_from_config(
        mu,
        data_type=data_type,
        data_config=data_config,
        n_obs=n_obs,
    )

    if Sigma is None:
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


def _add_length_per_rep_outputs(ci_df_for_rep, *, B, k):
    coverage_per_rep_table = build_rep_coverage_table(
        ci_df_for_rep,
        B=B,
        k=k,
    )

    keep_cols = ["rep", "avg_length", "n_intervals"]
    if "is_complete" in coverage_per_rep_table.columns:
        keep_cols.append("is_complete")

    length_per_rep_table = coverage_per_rep_table[keep_cols].copy()
    length_per_rep = length_per_rep_table["avg_length"].to_numpy(dtype=float)

    return coverage_per_rep_table, length_per_rep_table, length_per_rep


def build_rep_coverage_table(
    ci_df,
    *,
    group_cols=None,
    B=None,
    k=None,
):
    if group_cols is None:
        group_cols = []

    if len(ci_df) == 0:
        # An all-failure simulation yields an empty DataFrame with no
        # inferred columns.  Handle that case before schema validation
        # so the original per-replication errors remain in ci_failures.
        out_cols = list(group_cols) + ["rep", "coverage_rate", "avg_length", "n_intervals"]
        if k is not None:
            out_cols.append("is_complete")
        return pd.DataFrame(columns=out_cols)

    needed = {"rep", "covered", "length", "idx"}
    missing = needed - set(ci_df.columns)
    if missing:
        raise ValueError(f"ci_df is missing required columns: {missing}")

    rep_df = (
        ci_df.groupby(list(group_cols) + ["rep"], as_index=False)
             .agg(
                 coverage_rate=("covered", "mean"),
                 avg_length=("length", "mean"),
                 n_intervals=("idx", "size"),
             )
    )

    if B is not None:
        if len(group_cols) == 0:
            full = pd.DataFrame({"rep": np.arange(B)})
            rep_df = full.merge(rep_df, on="rep", how="left")
        else:
            levels = []
            for col in group_cols:
                vals = rep_df[col].dropna().unique().tolist()
                if col == "epsilon":
                    vals = sorted(vals)
                else:
                    vals = sorted(vals, key=str)
                levels.append(vals)

            full_index = pd.MultiIndex.from_product(
                levels + [range(B)],
                names=list(group_cols) + ["rep"]
            )

            rep_df = (
                rep_df.set_index(list(group_cols) + ["rep"])
                      .reindex(full_index)
                      .reset_index()
            )

    if k is not None:
        rep_df["is_complete"] = rep_df["n_intervals"].eq(k)

    return rep_df





def format_data_splitting_hat_output(ds_out, *, epsilon=np.nan):
    df = ds_out["ci_df"].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep", "rank", "idx",
            "L", "U", "truth", "covered", "length"
        ])

    df["method"] = "Data Splitting"
    df["epsilon"] = epsilon

    return df[[
        "method", "epsilon", "rep", "rank", "idx",
        "L", "U", "truth", "covered", "length"
    ]].copy()


def format_naive_hat_output(naive_out, *, epsilon=np.nan):
    df = naive_out["ci_records"].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep", "rank", "idx",
            "L", "U", "truth", "covered", "length"
        ])

    df = df.rename(columns={
        "index": "idx",
        "ci_lower": "L",
        "ci_upper": "U",
    })

    df["method"] = "Standard"
    df["epsilon"] = epsilon

    return df[[
        "method", "epsilon", "rep", "rank", "idx",
        "L", "U", "truth", "covered", "length"
    ]].copy()


def format_topk_hat_output(one_out, *, epsilon):
    df = one_out["ci_records"].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep", "rank", "idx",
            "L", "U", "truth", "covered", "length"
        ])

    df["method"] = "Randomized PSI"
    df["epsilon"] = float(epsilon)

    return df[[
        "method", "epsilon", "rep", "rank", "idx",
        "L", "U", "truth", "covered", "length"
    ]].copy()


def format_polyhedral_hat_output(poly_out, *, epsilon=np.nan):
    df = poly_out["ci_df"].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep", "rank", "idx",
            "L", "U", "truth", "covered", "length"
        ])

    df["method"] = "Polyhedral PSI"
    df["epsilon"] = epsilon

    return df[[
        "method", "epsilon", "rep", "rank", "idx",
        "L", "U", "truth", "covered", "length"
    ]].copy()

def format_zoom_stepdown_hat_output(zoom_out, *, epsilon=np.nan):
    df = zoom_out["ci_df"].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep", "rank", "idx",
            "L", "U", "truth", "covered", "length"
        ])

    df["method"] = "Zoom Correction"
    df["epsilon"] = epsilon

    return df[[
        "method", "epsilon", "rep", "rank", "idx",
        "L", "U", "truth", "covered", "length"
    ]].copy()


def _attach_setting_metadata_to_df(df, setting_meta):
    """
    Add setting-level metadata columns to a dataframe.

    Important:
        If metadata value is a tuple/list/array, we still treat it as
        one object per row, not as a column-length vector.
    """
    df = df.copy()

    for key, val in setting_meta.items():

        if isinstance(val, (list, tuple, np.ndarray)):
            df[key] = pd.Series(
                [val for _ in range(len(df))],
                index=df.index,
                dtype="object",
            )
        else:
            df[key] = val

    return df



def _make_stat_records(
    *,
    method,
    epsilon,
    rep,
    role,
    X_hat,
    Sigma_hat=None,
    sigma2_hat=np.nan,
    sigma_mode=None,
    data_type=None,
    utility_fn=None,
    extra=None,
):
    """
    Record one statistic vector for one method and one replication.

    For VI:
        X_hat = phi_hat

    For Gaussian / binomial / BT:
        X_hat = method-specific Gaussian-style statistic.

    role:
        "selection"  : statistic used for selection quality
        "inference"  : statistic used for CI inference
        "full"       : full-data statistic, usually same as selection/inference
    """
    x = np.asarray(X_hat, dtype=float).reshape(-1)
    M = x.size

    if utility_fn is None:
        score = x.copy()
    else:
        score = np.asarray(utility_fn(x), dtype=float).reshape(-1)

    if score.size != M:
        raise ValueError(
            f"utility_fn(X_hat) must return length {M}, got {score.size}."
        )

    if Sigma_hat is None:
        var_diag = np.full(M, np.nan, dtype=float)
    else:
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        if Sigma_hat.shape != (M, M):
            raise ValueError(
                f"Sigma_hat must have shape {(M, M)}, got {Sigma_hat.shape}."
            )
        var_diag = np.diag(Sigma_hat).astype(float)

    se_diag = np.sqrt(np.maximum(var_diag, 0.0))

    base = {
        "method": method,
        "epsilon": np.nan if epsilon is None else float(epsilon),
        "rep": int(rep),
        "role": str(role),
        "sigma2_hat": float(sigma2_hat) if np.isfinite(sigma2_hat) else np.nan,
        "sigma_mode": sigma_mode,
        "data_type": data_type,
    }

    if extra is not None:
        base.update(extra)

    rows = []
    for j in range(M):
        rows.append({
            **base,
            "idx": int(j),
            "mu_hat": float(x[j]),
            "score_hat": float(score[j]),
            "variance_hat": float(var_diag[j]) if np.isfinite(var_diag[j]) else np.nan,
            "se_hat": float(se_diag[j]) if np.isfinite(se_diag[j]) else np.nan,
        })

    return rows





# ============================================================
# Engine helpers that were referenced but never defined
# ============================================================

def generate_gaussian_replicates(mu, Sigma, B, n_obs, seed=0):
    """Shared Gaussian samples: shape (B, n_obs, M)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    M = mu.size
    out = np.empty((int(B), int(n_obs), M), dtype=float)
    for b in range(int(B)):
        out[b] = rng.multivariate_normal(mean=mu, cov=Sigma, size=int(n_obs))
    return out


def validate_X_samples(data_samples, B, n_obs, M, name="X_samples"):
    arr = np.asarray(data_samples, dtype=float)
    expected = (int(B), int(n_obs), int(M))
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}.")
    return arr


def _common_se_from_covariance(
    Sigma_hat,
    zoom_sigma_mode="mean",
    min_var=1e-12,
):
    """
    Convert a covariance matrix into one common standard error
    for Zoom Correction.

    Sigma_hat must be the covariance matrix of the estimated
    statistic, not the raw-data covariance.
    """
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)

    if (
        Sigma_hat.ndim != 2
        or Sigma_hat.shape[0] != Sigma_hat.shape[1]
    ):
        raise ValueError(
            "Sigma_hat must be a square covariance matrix."
        )

    diag_var = np.diag(Sigma_hat).astype(float)

    if np.any(~np.isfinite(diag_var)):
        raise ValueError(
            "Sigma_hat has non-finite diagonal entries."
        )

    if np.any(diag_var < 0):
        raise ValueError(
            "Sigma_hat has negative diagonal variances."
        )

    mode = str(zoom_sigma_mode).lower()

    if mode in {"mean", "pooled"}:
        var_used = float(np.mean(diag_var))

    elif mode == "max":
        var_used = float(np.max(diag_var))

    elif mode == "min":
        positive = diag_var[diag_var > 0]

        if len(positive) == 0:
            raise ValueError(
                "All diagonal variances are zero."
            )

        var_used = float(np.min(positive))

    elif mode == "original_equal":
        if not np.allclose(
            diag_var,
            diag_var[0],
            rtol=1e-6,
            atol=1e-12,
        ):
            raise ValueError(
                "original_equal requires equal marginal "
                "variances, but Sigma_hat has unequal "
                "diagonal entries."
            )

        var_used = float(diag_var[0])

    else:
        raise ValueError(
            "zoom_sigma_mode must be one of "
            "{'mean', 'pooled', 'max', 'min', "
            "'original_equal'}."
        )

    var_used = max(var_used, float(min_var))
    se_common = float(np.sqrt(var_used))

    return se_common, var_used


def _common_se_for_zoom(observed_data, Sigma, *, n_eff, sigma="known",
                        zoom_sigma_mode="mean"):
    """Gaussian counterpart of _common_se_from_covariance."""
    observed_data = np.asarray(observed_data, dtype=float)
    n_eff = float(n_eff)
    if str(sigma).lower() == "known":
        Sigma_hat = np.asarray(Sigma, dtype=float) / n_eff
    else:
        s2 = observed_data.var(axis=0, ddof=1)
        Sigma_hat = np.diag(s2 / n_eff)
    se_common, var_used = _common_se_from_covariance(
        Sigma_hat, zoom_sigma_mode=zoom_sigma_mode,
    )
    return se_common, var_used


def default_signal_strength_from_mu_sigma(mu, Sigma, k, *, n_obs=None, utility_fn=None):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    M = mu.size
    n = float(n_obs) if (n_obs is not None and n_obs > 0) else 1.0
    if utility_fn is None:
        score_vec = mu.copy()
        deriv = np.ones(M)
    else:
        score_vec = np.asarray(utility_fn(mu), dtype=float).reshape(-1)
        h = 1e-5 * (np.abs(mu) + 1.0)
        f_plus  = np.asarray(utility_fn(mu + h), dtype=float).reshape(-1)
        f_minus = np.asarray(utility_fn(mu - h), dtype=float).reshape(-1)
        deriv = (f_plus - f_minus) / (2.0 * h)
    diag = np.clip(np.diag(Sigma), 0.0, None)
    se_score = np.abs(deriv) * np.sqrt(diag / n)
    order = np.argsort(score_vec)[::-1]
    true_topk_subset  = tuple(sorted(int(i) for i in order[:k]))
    true_topk_utility = float(np.sum(score_vec[list(true_topk_subset)]))
    if k < M:
        kth, nxt = int(order[k - 1]), int(order[k])
        topk_gap = float(score_vec[kth] - score_vec[nxt])
        noise_scale = float(np.sqrt(se_score[kth] ** 2 + se_score[nxt] ** 2))
        standardized = topk_gap / noise_scale if noise_scale > 0 else np.nan
        next_idx = nxt
    else:
        kth, next_idx = int(order[k - 1]), None
        topk_gap = noise_scale = standardized = np.nan
    top1_gap = float(score_vec[order[0]] - score_vec[order[1]]) if M >= 2 else np.nan
    return {
        "signal_strength": standardized,
        "standardized_topk_gap": standardized,
        "topk_gap": topk_gap,
        "top1_gap": top1_gap,
        "noise_scale_topk_gap": noise_scale,
        "kth_idx": kth,
        "next_idx": next_idx,
        "true_topk_subset": true_topk_subset,
        "true_topk_utility": true_topk_utility,
    }


def _common_se_from_covariance(Sigma_hat, zoom_sigma_mode="mean", min_var=1e-12):
    """
    Convert a possibly heteroscedastic covariance matrix into one common SE
    for Zoom Correction.

    Sigma_hat should be the covariance matrix of the statistic X_hat,
    not the raw-data covariance.
    """
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)

    if Sigma_hat.ndim != 2 or Sigma_hat.shape[0] != Sigma_hat.shape[1]:
        raise ValueError("Sigma_hat must be a square covariance matrix.")

    diag_var = np.diag(Sigma_hat).astype(float)

    if np.any(~np.isfinite(diag_var)):
        raise ValueError("Sigma_hat has non-finite diagonal entries.")

    if np.any(diag_var < 0):
        raise ValueError("Sigma_hat has negative diagonal variances.")

    mode = str(zoom_sigma_mode).lower()

    if mode in {"mean", "pooled"}:
        var_used = float(np.mean(diag_var))

    elif mode == "max":
        var_used = float(np.max(diag_var))

    elif mode == "min":
        positive = diag_var[diag_var > 0]
        if len(positive) == 0:
            raise ValueError("All diagonal variances are zero.")
        var_used = float(np.min(positive))

    elif mode == "original_equal":
        if not np.allclose(diag_var, diag_var[0], rtol=1e-6, atol=1e-12):
            raise ValueError(
                "original_equal requires equal marginal variances, "
                "but Sigma_hat has unequal diagonal entries."
            )
        var_used = float(diag_var[0])

    else:
        raise ValueError(
            "zoom_sigma_mode must be one of "
            "{'mean', 'pooled', 'max', 'min', 'original_equal'}."
        )

    var_used = max(var_used, float(min_var))
    se_common = float(np.sqrt(var_used))

    return se_common, var_used
    