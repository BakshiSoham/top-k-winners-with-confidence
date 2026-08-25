import numpy as np
import pandas as pd

from scipy.optimize import minimize

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor

from tqdm.auto import tqdm
###########################
##Binomial
###########################
def _as_1d_float(x, name="x"):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return x


def _clip_prob(x, clip_eps):
    x = np.asarray(x, dtype=float)
    return np.clip(x, clip_eps, 1.0 - clip_eps)


def _logit(x):
    x = np.asarray(x, dtype=float)
    return np.log(x / (1.0 - x))


def _default_clip_eps(m):
    # Continuity-style numerical clipping.
    # This prevents log(0), logit(0), logit(1).
    return 0.5 / float(m)


def generate_binomial_dosage_data(p, m, seed=None, rng=None):
    p = _as_1d_float(p, "p")

    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("All p_j must be strictly between 0 and 1.")

    if m <= 0:
        raise ValueError("m must be positive.")

    if rng is None:
        rng = np.random.default_rng(seed)

    S = rng.binomial(n=int(m), p=p, size=p.shape[0])
    return S.astype(int)



def estimate_binomial_gaussian_input(
    S,
    m,
    statistic="log_success",
    clip_eps=None,
):
    S = np.asarray(S, dtype=float).reshape(-1)

    if clip_eps is None:
        clip_eps = _default_clip_eps(m)

    p_hat_raw = S / float(m)
    p_hat_clip = _clip_prob(p_hat_raw, clip_eps)
    if statistic == "probability":
        # Gaussian statistic used for inference:
        # X = p_hat.
        X = p_hat_raw.copy()

        # Estimated target is also p_hat.
        target_hat = p_hat_raw.copy()

        # Selection is performed directly on p_hat.
        selection_statistic = p_hat_raw.copy()

        # Plug-in covariance for p_hat:
        # Var(p_hat_j) = p_j(1-p_j)/m.
        #
        # We use p_hat_clip inside the variance to avoid a zero
        # estimated variance when S_j is exactly 0 or m.
        var_target = (
            p_hat_clip
            * (1.0 - p_hat_clip)
            / float(m)
        )
        Sigma_target = np.diag(var_target)

        # The selection statistic is also p_hat, so it has
        # the same covariance as the target statistic.
        Sigma_selection_statistic = Sigma_target.copy()

    elif statistic == "log_success":
        X = p_hat_raw.copy()
        target_hat = p_hat_raw.copy()

        # Selection statistic is log(p_hat).
        selection_statistic = np.log(p_hat_clip)

        # Plug-in covariance for p_hat.
        var_target = (
            p_hat_clip
            * (1.0 - p_hat_clip)
            / float(m)
        )
        Sigma_target = np.diag(var_target)

        # Delta-method covariance for log(p_hat).
        var_selection_stat = (
            (1.0 - p_hat_clip)
            / (float(m) * p_hat_clip)
        )
        Sigma_selection_statistic = np.diag(
            var_selection_stat
        )

    elif statistic == "logit":
        # Target is logit(p), so Gaussian variable is logit(p_hat).
        X = _logit(p_hat_clip)
        target_hat = X.copy()
        selection_statistic = X.copy()

        # Delta-method covariance for logit(p_hat):
        # Var(logit(p_hat_j)) approx
        # 1 / {m p_j(1-p_j)}.
        var_target = (
            1.0
            / (
                float(m)
                * p_hat_clip
                * (1.0 - p_hat_clip)
            )
        )
        Sigma_target = np.diag(var_target)
        Sigma_selection_statistic = Sigma_target.copy()

    else:
        raise ValueError(
            "statistic must be one of "
            "{'probability', 'log_success', 'logit'}."
        )

    return {
        "S": S.astype(int),
        "m": int(m),
        "p_hat_raw": p_hat_raw,
        "p_hat_clip": p_hat_clip,
        "selection_statistic": selection_statistic,
        "X": X,
        "target_hat": target_hat,
        "Sigma_target": Sigma_target,
        "Sigma_selection_statistic": Sigma_selection_statistic,
        "clip_eps": clip_eps,
        "statistic": statistic,
    }

def true_target_and_oracle_covariance(
    p,
    m,
    statistic="log_success",
):
    p = _as_1d_float(p, "p")

    if statistic == "probability":
        target_true = p.copy()
        Sigma_oracle = np.diag(
            p * (1.0 - p) / float(m)
        )

    elif statistic == "log_success":
        target_true = p.copy()
        Sigma_oracle = np.diag(
            p * (1.0 - p) / float(m)
        )

    elif statistic == "logit":
        target_true = _logit(p)
        Sigma_oracle = np.diag(
            1.0
            / (
                float(m)
                * p
                * (1.0 - p)
            )
        )

    else:
        raise ValueError(
            "statistic must be one of "
            "{'probability', 'log_success', 'logit'}."
        )

    return target_true, Sigma_oracle



def make_binomial_selection_utility(
    statistic="log_success",
    clip_eps=1e-8,
):
    if statistic == "probability":

        def utility_fn(x):
            x_arr = np.asarray(x, dtype=float)

            if x_arr.ndim == 0:
                return float(x_arr)

            return x_arr
    elif statistic == "log_success":

        def utility_fn(x):
            x_arr = np.asarray(x, dtype=float)
            val = np.log(
                _clip_prob(x_arr, clip_eps)
            )

            if val.ndim == 0:
                return float(val)

            return val
    elif statistic == "logit":

        def utility_fn(x):
            x_arr = np.asarray(x, dtype=float)

            if x_arr.ndim == 0:
                return float(x_arr)

            return x_arr

    else:
        raise ValueError(
            "statistic must be one of "
            "{'probability', 'log_success', 'logit'}."
        )

    return utility_fn



###########################
##BTD
###########################

import numpy as np
import pandas as pd


def simulate_bt_davidson(mu, nu, n_matches, random_state=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(random_state)

    mu = np.asarray(mu, dtype=float).reshape(-1)
    p = len(mu)
    results = []

    for i in range(p):
        for j in range(i + 1, p):
            if np.isscalar(n_matches):
                nij = int(n_matches)
            else:
                nij = int(n_matches[i, j])

            denom = (
                np.exp(mu[i])
                + np.exp(mu[j])
                + 2 * float(nu) * np.exp((mu[i] + mu[j]) / 2)
            )

            p_i_win = np.exp(mu[i]) / denom
            p_j_win = np.exp(mu[j]) / denom
            p_draw = 2 * float(nu) * np.exp((mu[i] + mu[j]) / 2) / denom

            outcomes = rng.multinomial(nij, [p_i_win, p_j_win, p_draw])

            results.append({
                "player_i": i,
                "player_j": j,
                "W_ij": int(outcomes[0]),
                "W_ji": int(outcomes[1]),
                "D_ij": int(outcomes[2]),
                "n_ij": int(nij),
            })

    return pd.DataFrame(results)


def compute_T_D_N(df, p=None):

    if p is None:
        p = int(max(df["player_i"].max(), df["player_j"].max()) + 1)

    T = np.zeros(p)
    N = np.zeros((p, p))
    D_total = 0
    for _, row in df.iterrows():
        i = int(row["player_i"])
        j = int(row["player_j"])
        W_ij = row["W_ij"]
        W_ji = row["W_ji"]
        D_ij = row["D_ij"]
        n_ij = W_ij + W_ji + D_ij
        N[i, j] = n_ij
        N[j, i] = n_ij
        T[i] += W_ij + 0.5 * D_ij
        T[j] += W_ji + 0.5 * D_ij
        D_total += D_ij

    return T, D_total, N



def mle_btd(T, N, D, max_iter=500, lr=0.01, tol=1e-6):
    p = len(T)
    def neg_loglik(theta):
        mu = theta[:p]
        log_nu = theta[p]
        nu = np.exp(log_nu)

        ll = np.dot(T, mu) + D * log_nu

        for i in range(p):
            for j in range(i+1, p):

                Nij = N[i, j]
                if Nij == 0:
                    continue
                ei = np.exp(mu[i])
                ej = np.exp(mu[j])

                denom = ei + ej + 2 * nu * np.sqrt(ei * ej)

                ll -= Nij * np.log(denom)

        return -ll

    theta0 = np.zeros(p + 1)
    res = minimize(neg_loglik, theta0, method="BFGS")
    mu_hat = res.x[:p]
    nu_hat = np.exp(res.x[p])
    mu_hat -= mu_hat.mean()

    return mu_hat, nu_hat


def btd_mu_covariance(N, mu_hat, nu_hat):

    p = len(mu_hat)
    I_mm = np.zeros((p, p))
    I_me = np.zeros(p)
    I_ee = 0.0

    for j in range(p):
        for l in range(j + 1, p):
            n = N[j, l]
            if n == 0:
                continue
            ej = np.exp(mu_hat[j])
            el = np.exp(mu_hat[l])
            sqrt_term = np.sqrt(ej * el)
            Z = ej + el + 2 * nu_hat * sqrt_term
            v = (ej * el + 0.5 * nu_hat * sqrt_term * (ej + el)) / (Z ** 2)
            c = (nu_hat * sqrt_term * (el - ej)) / (Z ** 2)
            d = (2 * nu_hat * sqrt_term * (ej + el)) / (Z ** 2)
            # update I_mu_mu
            I_mm[j, j] += n * v
            I_mm[l, l] += n * v
            I_mm[j, l] -= n * v
            I_mm[l, j] -= n * v
            # update I_mu_eta
            I_me[j] += n * c
            I_me[l] -= n * c

            # update I_eta_eta
            I_ee += n * d

    # Schur complement
    S = I_mm - np.outer(I_me, I_me) / I_ee

    # covariance of mu_hat
    Sigma_mu = np.linalg.pinv(S)


    return Sigma_mu


###########################
##VI
###########################

def _vi_sym_ridge(A, ridge=1e-8):
    A = np.asarray(A, dtype=float)
    A = np.atleast_2d(A)
    A = 0.5 * (A + A.T)
    return A + float(ridge) * np.eye(A.shape[0])

def _vi_ar1_cov(p, rho):
    idx = np.arange(int(p))
    return float(rho) ** np.abs(idx[:, None] - idx[None, :])

def _normalize_vi_nuisance_mode(mode):
    mode = str(mode).lower().strip().replace("-", "_").replace(" ", "_")

    aliases = {
        "cross_fitting": "cross_fitting",
        "crossfit": "cross_fitting",
        "cross_fitted": "cross_fitting",
        "model_cf": "cross_fitting",

        "holdout_fixed": "holdout_fixed",
        "fixed_holdout": "holdout_fixed",
        "fixed_nuisance": "holdout_fixed",
        "independent_holdout": "holdout_fixed",
    }

    if mode not in aliases:
        raise ValueError(
            "vi_nuisance_mode must be either "
            "'cross_fitting' or 'holdout_fixed'."
        )

    return aliases[mode]




def _normalize_variable_importance_config(data_config=None, *, n_obs=None):
    cfg = dict(data_config or {})

    if "p" not in cfg:
        if "beta" in cfg and cfg["beta"] is not None:
            cfg["p"] = len(np.asarray(cfg["beta"], dtype=float).reshape(-1))
        elif "phi" in cfg and cfg["phi"] is not None:
            cfg["p"] = len(np.asarray(cfg["phi"], dtype=float).reshape(-1))
        elif "mu" in cfg and cfg["mu"] is not None:
            cfg["p"] = len(np.asarray(cfg["mu"], dtype=float).reshape(-1))
        elif "oracle" in cfg and isinstance(cfg["oracle"], dict) and "phi" in cfg["oracle"]:
            cfg["p"] = len(np.asarray(cfg["oracle"]["phi"], dtype=float).reshape(-1))
        else:
            raise ValueError(
                "For variable_importance data, provide p, beta, phi/mu, "
                "or data_config['oracle']['phi']."
            )

    p = int(cfg["p"])
    if p <= 0:
        raise ValueError("For variable_importance data, p must be positive.")

    split_frac = float(cfg.get("split_frac", cfg.get("vi_split_frac", 0.5)))
    if not (0.0 < split_frac < 1.0):
        raise ValueError("For variable_importance data, split_frac must be in (0, 1).")

    covariance = str(cfg.get("covariance", "oracle")).lower()
    if covariance not in {"oracle", "known", "true", "plugin", "estimated", "estimate", "unknown"}:
        raise ValueError(
            "For variable_importance data, covariance must be one of "
            "{'oracle','known','true','plugin','estimated','estimate','unknown'}."
        )

    n_folds = int(cfg.get("n_folds", 5))
    if n_folds < 2:
        raise ValueError("For variable_importance data, n_folds must be at least 2.")

    oracle_reps = int(cfg.get("oracle_reps", 300))
    if oracle_reps <= 0:
        raise ValueError("For variable_importance data, oracle_reps must be positive.")

    oracle_n_obs = cfg.get("oracle_n_obs", None)
    if oracle_n_obs is not None:
        oracle_n_obs = int(oracle_n_obs)
        if oracle_n_obs <= 1:
            raise ValueError("For variable_importance data, oracle_n_obs must be larger than 1.")

    oracle_covariance_source = str(
        cfg.get("oracle_covariance_source", cfg.get("covariance_source", "bootstrap"))
    ).lower()
    if oracle_covariance_source not in {"bootstrap", "influence", "average", "avg", "mean"}:
        raise ValueError(
            "oracle_covariance_source must be 'bootstrap', 'influence', or 'average'."
        )

    return {
        **cfg,
        "p": p,
        "x_dist": str(cfg.get("x_dist", "normal")).lower(),
        "rho": float(cfg.get("rho", 0.0)),
        "x_mean": cfg.get("x_mean", None),
        "x_scale": float(cfg.get("x_scale", 1.0)),
        "x_low": float(cfg.get("x_low", -1.0)),
        "x_high": float(cfg.get("x_high", 1.0)),
        "noise_sd": float(cfg.get("noise_sd", 1.0)),
        "noise_dist": str(cfg.get("noise_dist", "normal")).lower(),
        "tree_params": cfg.get("tree_params", None),
        "n_folds": n_folds,
        "split_frac": split_frac,
        "vi_split_frac": split_frac,
        "covariance": covariance,
        "oracle_reps": oracle_reps,
        "oracle_n_obs": oracle_n_obs,
        "oracle_covariance_source": oracle_covariance_source,
        "cov_ridge": float(cfg.get("cov_ridge", 1e-8)),
        "oracle_cov_ridge": float(cfg.get("oracle_cov_ridge", cfg.get("cov_ridge", 1e-8))),
        "min_var_y": float(cfg.get("min_var_y", 1e-12)),
    }





def generate_variable_importance_data(
    n,
    *,
    p,
    f=None,
    beta=None,
    x_dist="normal",
    rho=0.0,
    x_mean=None,
    x_scale=1.0,
    x_low=-1.0,
    x_high=1.0,
    noise_sd=1.0,
    noise_dist="normal",
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    n = int(n)
    p = int(p)
    x_dist = str(x_dist).lower()

    if x_dist in {"normal", "gaussian"}:
        mean = np.zeros(p) if x_mean is None else np.asarray(x_mean, dtype=float).reshape(-1)
        if mean.size != p:
            raise ValueError(f"x_mean must have length p={p}.")
        X = rng.multivariate_normal(
            mean=mean,
            cov=(float(x_scale) ** 2) * _vi_ar1_cov(p, rho),
            size=n,
        )
    elif x_dist in {"uniform", "unif"}:
        X = rng.uniform(float(x_low), float(x_high), size=(n, p))
    else:
        raise ValueError("For variable_importance data, x_dist must be 'normal' or 'uniform'.")

    if f is not None:
        signal = np.asarray(f(X), dtype=float).reshape(-1)
    elif beta is not None:
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.size != p:
            raise ValueError(f"beta must have length p={p}.")
        signal = X @ beta
    else:
        signal = np.zeros(n, dtype=float)
        if p >= 1:
            signal += 1.50 * X[:, 0]
        if p >= 2:
            signal += 1.25 * np.sin(X[:, 1])
        if p >= 3:
            signal += 0.80 * (X[:, 2] ** 2 - np.mean(X[:, 2] ** 2))
        if p >= 5:
            signal += 0.75 * X[:, 3] * X[:, 4]
        if p >= 6:
            signal += 0.50 * np.cos(X[:, 5])

    if signal.size != n:
        raise ValueError("For variable_importance data, f(X) must return a vector of length n.")

    noise_dist = str(noise_dist).lower()
    if noise_dist == "normal":
        eps = rng.normal(0.0, float(noise_sd), size=n)
    elif noise_dist in {"t", "student"}:
        eps = rng.standard_t(df=5, size=n) * float(noise_sd) / np.sqrt(5.0 / 3.0)
    else:
        raise ValueError("For variable_importance data, noise_dist must be 'normal' or 't'.")

    O = signal + eps
    return X, O


def _generate_vi_data_from_cfg(cfg, n, rng):
    cfg = _normalize_variable_importance_config(cfg, n_obs=n)
    return generate_variable_importance_data(
        int(n),
        p=cfg["p"],
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


def fit_vi_holdout_nuisance(
    X_hold,
    O_hold,
    *,
    tree_params=None,
    random_state=0,
):
    X_hold = np.asarray(X_hold, dtype=float)
    O_hold = np.asarray(O_hold, dtype=float).reshape(-1)

    if X_hold.ndim != 2:
        raise ValueError("X_hold must be 2D.")
    if O_hold.size != X_hold.shape[0]:
        raise ValueError("O_hold must have length X_hold.shape[0].")
    n_hold, p = X_hold.shape
    learner, learner_params = _parse_vi_learner_params(tree_params)
    # Full model for m(X)
    m_model = _make_vi_regressor(
        learner,
        learner_params,
        random_state=int(random_state),
    )
    m_model.fit(X_hold, O_hold)

    # Leave-one-variable-out models for m_{-j}(X_{-j})
    m_minus_models = []
    keep_cols = []

    for j in range(p):
        keep = [a for a in range(p) if a != j]
        keep_cols.append(keep)

        model_j = _make_vi_regressor(
            learner,
            learner_params,
            random_state=int(random_state) + 97 * (j + 1),
        )
        model_j.fit(X_hold[:, keep], O_hold)
        m_minus_models.append(model_j)

    return {
        "type": "fitted_holdout_nuisance",
        "learner": learner,
        "learner_params": learner_params,
        "p": int(p),
        "n_holdout_fit": int(n_hold),
        "m_model": m_model,
        "m_minus_models": m_minus_models,
        "keep_cols": keep_cols,
    }


def predict_vi_fixed_nuisance(X, fixed_nuisance):
    """
    Predict m_hat and m_minus_hat using fixed fitted nuisance functions.
    No refitting happens here.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    if fixed_nuisance is None:
        raise ValueError("fixed_nuisance cannot be None.")

    p = X.shape[1]
    p_nuis = int(fixed_nuisance.get("p", p))

    if p != p_nuis:
        raise ValueError(f"X has p={p}, but fixed_nuisance has p={p_nuis}.")

    m_model = fixed_nuisance["m_model"]
    m_minus_models = fixed_nuisance["m_minus_models"]
    keep_cols = fixed_nuisance["keep_cols"]

    m_hat = np.asarray(m_model.predict(X), dtype=float).reshape(-1)

    m_minus_hat = np.empty((X.shape[0], p), dtype=float)

    for j in range(p):
        m_minus_hat[:, j] = np.asarray(
            m_minus_models[j].predict(X[:, keep_cols[j]]),
            dtype=float,
        ).reshape(-1)

    return m_hat, m_minus_hat


def estimate_variable_importance_fixed_nuisance(
    X,
    O,
    *,
    fixed_nuisance,
    phi_for_if=None,
    cov_ridge=1e-8,
    min_var_y=1e-12,
):
    """
    Compute VI statistic using fixed nuisance functions.

    This is the new main statistic for simulation:
        fixed m_hat and fixed m_minus_hat_j
        no cross-fitting
        no refitting inside simulation replicate
    """
    X = np.asarray(X, dtype=float)
    O = np.asarray(O, dtype=float).reshape(-1)

    m_hat, m_minus_hat = predict_vi_fixed_nuisance(X, fixed_nuisance)

    out = estimate_vi_from_nuisance_predictions(
        X,
        O,
        m_hat,
        m_minus_hat,
        phi_for_if=phi_for_if,
        cov_ridge=cov_ridge,
        min_var_y=min_var_y,
    )

    out["estimator"] = "fixed_nuisance"
    out["vi_statistic_used"] = "fixed_nuisance"
    out["fixed_nuisance_learner"] = fixed_nuisance.get("learner", None)
    out["m_hat"] = m_hat
    out["m_minus"] = m_minus_hat

    return out


def estimate_vi_population_from_holdout_bootstrap(
    *,
    X_hold,
    O_hold,
    fixed_nuisance,
    n_obs,
    bootstrap_reps=300,
    bootstrap_n_obs=None,
    seed=12345,
    cov_ridge=1e-8,
    min_var_y=1e-12,
    verbose=True,
):
    X_hold = np.asarray(X_hold, dtype=float)
    O_hold = np.asarray(O_hold, dtype=float).reshape(-1)

    if X_hold.ndim != 2:
        raise ValueError("X_hold must be 2D.")
    if O_hold.size != X_hold.shape[0]:
        raise ValueError("O_hold must have length X_hold.shape[0].")

    n_hold, p = X_hold.shape
    n_boot = int(n_obs if bootstrap_n_obs is None else bootstrap_n_obs)

    if n_boot <= 1:
        raise ValueError("bootstrap_n_obs must be larger than 1.")

    rng = np.random.default_rng(seed)

    phi_boot = np.empty((int(bootstrap_reps), p), dtype=float)

    iterator = range(int(bootstrap_reps))
    if verbose:
        iterator = tqdm(iterator, desc="VI holdout bootstrap")

    for r in iterator:
        idx = rng.integers(0, n_hold, size=n_boot)
        X_b = X_hold[idx]
        O_b = O_hold[idx]

        est_b = estimate_variable_importance_fixed_nuisance(
            X_b,
            O_b,
            fixed_nuisance=fixed_nuisance,
            phi_for_if=None,
            cov_ridge=cov_ridge,
            min_var_y=min_var_y,
        )

        phi_boot[r] = est_b["phi_hat"]

    phi_hat_pop = phi_boot.mean(axis=0)

    if int(bootstrap_reps) >= 2:
        Sigma_vim = float(n_boot) * np.cov(phi_boot, rowvar=False, ddof=1)
    else:
        raise ValueError("bootstrap_reps must be at least 2 to estimate covariance.")

    Sigma_vim = _vi_sym_ridge(Sigma_vim, ridge=cov_ridge)

    return {
        "phi": np.asarray(phi_hat_pop, dtype=float),
        "mu": np.asarray(phi_hat_pop, dtype=float),
        "Sigma_vim": Sigma_vim,
        "Sigma_hat_at_n": Sigma_vim / float(n_obs),
        "phi_boot": phi_boot,
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_n_obs": int(n_boot),
        "n_holdout": int(n_hold),
        "covariance_source": "holdout_bootstrap_fixed_nuisance",
        "estimator": "holdout_fixed_nuisance",
    }


def prepare_variable_importance_holdout_setting(
    setting,
    *,
    n_obs,
    holdout_n=100000,
    bootstrap_reps=300,
    bootstrap_n_obs=None,
    seed=123,
    verbose=True,
):
    cfg = _normalize_variable_importance_config(setting, n_obs=n_obs)
    p = int(cfg["p"])

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Step 1: generate large independent holdout dataset
    # --------------------------------------------------
    if "holdout_data" in cfg and cfg["holdout_data"] is not None:
        X_hold, O_hold = _coerce_variable_importance_sample(
            cfg["holdout_data"],
            p=p,
            name="holdout_data",
        )
    else:
        X_hold, O_hold = _generate_vi_data_from_cfg(
            cfg,
            int(holdout_n),
            rng,
        )

    # --------------------------------------------------
    # Step 2: fit fixed nuisance functions on holdout
    # --------------------------------------------------
    fixed_nuisance = cfg.get("fixed_nuisance", None)

    if fixed_nuisance is None:
        if verbose:
            print("\n" + "-" * 70)
            print("Fitting fixed VI nuisance functions on holdout dataset")
            print(f"holdout_n = {X_hold.shape[0]}")
            print(f"p = {p}")
            print(f"tree_params = {cfg.get('tree_params', None)}")
            print("-" * 70)

        fixed_nuisance = fit_vi_holdout_nuisance(
            X_hold,
            O_hold,
            tree_params=cfg.get("tree_params", None),
            random_state=seed + 1000,
        )

    # --------------------------------------------------
    # Step 3: bootstrap population phi and Sigma from holdout
    # --------------------------------------------------
    oracle = estimate_vi_population_from_holdout_bootstrap(
        X_hold=X_hold,
        O_hold=O_hold,
        fixed_nuisance=fixed_nuisance,
        n_obs=n_obs,
        bootstrap_reps=bootstrap_reps,
        bootstrap_n_obs=bootstrap_n_obs,
        seed=seed + 2000,
        cov_ridge=float(cfg.get("oracle_cov_ridge", cfg.get("cov_ridge", 1e-8))),
        min_var_y=float(cfg.get("min_var_y", 1e-12)),
        verbose=verbose,
    )

    mu_s = np.asarray(oracle["phi"], dtype=float).reshape(-1)
    Sigma_s = np.asarray(oracle["Sigma_vim"], dtype=float)

    if Sigma_s.shape != (p, p):
        raise ValueError(f"Estimated Sigma has shape {Sigma_s.shape}, expected {(p, p)}.")

    # --------------------------------------------------
    # Step 4: build config used by simulation replicates
    # --------------------------------------------------
    cfg_out = dict(cfg)
    cfg_out.update({
        "p": p,
        "mu": mu_s,
        "phi": mu_s,
        "true_vi_score": mu_s,
        "normalized_vi_score": mu_s,
        "Sigma": Sigma_s,
        "oracle": oracle,
        "fixed_nuisance": fixed_nuisance,
        "vi_statistic": "fixed_nuisance",
        "holdout_n": int(X_hold.shape[0]),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_n_obs": int(n_obs if bootstrap_n_obs is None else bootstrap_n_obs),
    })

    prepared_setting = dict(setting)
    prepared_setting.update({
        "p": p,
        "mu": mu_s,
        "phi": mu_s,
        "true_vi_score": mu_s,
        "normalized_vi_score": mu_s,
        "Sigma": Sigma_s,
        "oracle": oracle,
        "fixed_nuisance": fixed_nuisance,
        "vi_statistic": "fixed_nuisance",
        "holdout_n": int(X_hold.shape[0]),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_n_obs": int(n_obs if bootstrap_n_obs is None else bootstrap_n_obs),
    })

    return {
        "mu": mu_s,
        "phi": mu_s,
        "Sigma": Sigma_s,
        "Sigma_vim": Sigma_s,
        "oracle": oracle,
        "fixed_nuisance": fixed_nuisance,
        "X_hold": X_hold,
        "O_hold": O_hold,
        "data_config": cfg_out,
        "setting": prepared_setting,
    }


def estimate_vi_from_nuisance_predictions(
    X,
    O,
    m_hat,
    m_minus_hat,
    *,
    phi_for_if=None,
    cov_ridge=1e-8,
    min_var_y=1e-12,
):
    X = np.asarray(X, dtype=float)
    O = np.asarray(O, dtype=float).reshape(-1)
    m_hat = np.asarray(m_hat, dtype=float).reshape(-1)
    m_minus_hat = np.asarray(m_minus_hat, dtype=float)

    n, p = X.shape

    if O.size != n:
        raise ValueError("O must have length n.")
    if m_hat.size != n:
        raise ValueError("m_hat must have length n.")
    if m_minus_hat.shape != (n, p):
        raise ValueError(f"m_minus_hat must have shape {(n, p)}, got {m_minus_hat.shape}.")

    O_bar = float(np.mean(O))
    var_O = max(float(np.mean((O - O_bar) ** 2)), float(min_var_y))

    delta = m_hat[:, None] - m_minus_hat

    # Uncentered contribution whose expectation is mu_j
    contrib_hat = (
        delta ** 2
        + 2.0 * (O - m_hat)[:, None] * delta
    ) / var_O

    phi_hat = contrib_hat.mean(axis=0)

    # For the IF correction from estimating Var(O), use either
    # the true/oracle phi if available, or phi_hat otherwise.
    if phi_for_if is None:
        phi_plug = phi_hat
    else:
        phi_plug = np.asarray(phi_for_if, dtype=float).reshape(-1)
        if phi_plug.size != p:
            raise ValueError(f"phi_for_if must have length p={p}.")

    if_hat = (
        contrib_hat
        - phi_plug[None, :] * ((O - O_bar)[:, None] ** 2 / var_O)
    )

    Sigma_vim = np.cov(if_hat, rowvar=False, ddof=1)
    Sigma_vim = _vi_sym_ridge(Sigma_vim, ridge=cov_ridge)
    Sigma_hat = Sigma_vim / float(n)

    return {
        "phi_hat": np.asarray(phi_hat, dtype=float),
        "contrib_hat": contrib_hat,
        "if_hat": if_hat,
        "psi_hat": if_hat,  # backward compatibility
        "Sigma_vim": Sigma_vim,
        "Sigma_hat": Sigma_hat,
        "var_O": var_O,
        "n_eff": int(n),
    }



def _vi_call_m_func(m_func, X, cfg):
    """
    Call user-provided true m(X) function robustly.
    Supports either m_func(X) or m_func(X, cfg).
    """
    try:
        out = m_func(X, cfg)
    except TypeError:
        out = m_func(X)
    out = np.asarray(out, dtype=float).reshape(-1)
    if out.size != X.shape[0]:
        raise ValueError(
            f"m_func(X) must return length n={X.shape[0]}, got {out.size}."
        )
    return out


def _vi_call_m_minus_func(m_minus_func, X, cfg):
    """
    Call user-provided true m_minus(X) function robustly.
    Supports either m_minus_func(X) or m_minus_func(X, cfg).

    Must return an n x p matrix whose j-th column is E[O | X_{-j}].
    """
    try:
        out = m_minus_func(X, cfg)
    except TypeError:
        out = m_minus_func(X)
    out = np.asarray(out, dtype=float)
    if out.ndim != 2:
        raise ValueError("m_minus_func(X) must return a 2D array with shape (n, p).")
    if out.shape != X.shape:
        raise ValueError(
            f"m_minus_func(X) must return shape {X.shape}, got {out.shape}."
        )
    return out


def estimate_variable_importance_true_nuisance(
    X,
    O,
    *,
    m_func=None,
    m_minus_func=None,
    beta_signal=None,
    phi_for_if=None,
    cov_ridge=1e-8,
    min_var_y=1e-12,
    cfg=None,
):
    X = np.asarray(X, dtype=float)
    O = np.asarray(O, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    n, p = X.shape
    if O.size != n:
        raise ValueError("O must have length n.")

    cfg = dict(cfg or {})

    if m_func is not None and m_minus_func is not None:
        m = _vi_call_m_func(m_func, X, cfg)
        m_minus = _vi_call_m_minus_func(m_minus_func, X, cfg)

    elif beta_signal is not None:
        beta = np.asarray(beta_signal, dtype=float).reshape(-1)
        if beta.size != p:
            raise ValueError(f"beta_signal must have length p={p}, got {beta.size}.")

        I = (X > 0.0).astype(float)
        m = I @ beta
        m_minus = np.empty((n, p), dtype=float)

        for j in range(p):
            m_minus[:, j] = m - beta[j] * I[:, j] + 0.5 * beta[j]

    else:
        raise ValueError(
            "Provide either (m_func, m_minus_func) or beta_signal."
        )

    out = estimate_vi_from_nuisance_predictions(
        X,
        O,
        m,
        m_minus,
        phi_for_if=phi_for_if,
        cov_ridge=cov_ridge,
        min_var_y=min_var_y,
    )
    out["estimator"] = "true_nuisance"
    return out



def _vi_truth_from_setting(cfg):
    """
    Resolve population truth phi.

    Priority:
        1. analytic/stored true_vi_score
        2. normalized_vi_score
        3. phi
        4. mu
        5. None
    """
    for key in ["true_vi_score", "normalized_vi_score", "phi", "mu"]:
        if key in cfg and cfg[key] is not None:
            return np.asarray(cfg[key], dtype=float).reshape(-1), key
    return None, None


def estimate_variable_importance_oracle_bootstrap(
    *,
    setting,
    n_obs,
    oracle_reps=300,
    oracle_n_obs=None,
    n_folds=5,
    tree_params=None,
    covariance_source="bootstrap",
    seed=12345,
    verbose=True,
):
    """
    Population oracle for VI using true nuisance functions.

    This estimates/checks:
        phi_truth: true population target, analytic if available
        phi_mc: Monte Carlo mean of true-nuisance estimator
        Sigma_vim: Var(D(O, X; eta)), the theoretical IF covariance
    """
    cfg = _normalize_variable_importance_config(setting, n_obs=n_obs)

    p = int(cfg["p"])
    n_oracle = int(n_obs if oracle_n_obs is None else oracle_n_obs)
    rng = np.random.default_rng(seed)

    phi_truth, truth_source = _vi_truth_from_setting(cfg)

    if phi_truth is not None and phi_truth.size != p:
        raise ValueError(f"Truth phi has length {phi_truth.size}, expected p={p}.")

    phi_mat = np.empty((int(oracle_reps), p), dtype=float)
    Sigma_if_list = []

    m_func = cfg.get("m_func", None)
    m_minus_func = cfg.get("m_minus_func", None)
    beta_signal = cfg.get("beta_signal", cfg.get("beta", None))

    if m_func is None or m_minus_func is None:
        if beta_signal is None:
            raise ValueError(
                "True-nuisance VI oracle requires either "
                "setting['m_func'] and setting['m_minus_func'], "
                "or setting['beta_signal']."
            )

    iterator = range(int(oracle_reps))
    if verbose:
        iterator = tqdm(
            iterator,
            desc=f"VI population oracle: {cfg.get('label', cfg.get('setting_label', 'setting'))}"
        )

    for r in iterator:
        X_r, O_r = generate_variable_importance_data(
            n_oracle,
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

        est_r = estimate_variable_importance_true_nuisance(
            X_r,
            O_r,
            m_func=m_func,
            m_minus_func=m_minus_func,
            beta_signal=beta_signal,
            phi_for_if=phi_truth,
            cov_ridge=float(cfg.get("cov_ridge", 1e-8)),
            min_var_y=float(cfg.get("min_var_y", 1e-12)),
            cfg=cfg,
        )

        phi_mat[r] = est_r["phi_hat"]
        Sigma_if_list.append(est_r["Sigma_vim"])

    phi_mc = phi_mat.mean(axis=0)

    # If analytic truth exists, use it for coverage.
    # Otherwise, use MC oracle mean as fallback.
    if phi_truth is None:
        phi_out = phi_mc.copy()
        truth_source = "oracle_mc"
    else:
        phi_out = phi_truth.copy()

    if int(oracle_reps) >= 2:
        Sigma_boot = float(n_oracle) * np.cov(phi_mat, rowvar=False, ddof=1)
    else:
        Sigma_boot = np.mean(np.stack(Sigma_if_list, axis=0), axis=0)

    Sigma_if = np.mean(np.stack(Sigma_if_list, axis=0), axis=0)

    covariance_source = str(covariance_source).lower()

    if covariance_source == "bootstrap":
        Sigma_vim = Sigma_boot
    elif covariance_source == "influence":
        Sigma_vim = Sigma_if
    elif covariance_source in {"average", "avg", "mean"}:
        Sigma_vim = 0.5 * (Sigma_boot + Sigma_if)
    else:
        raise ValueError("covariance_source must be 'bootstrap', 'influence', or 'average'.")

    Sigma_vim = _vi_sym_ridge(
        Sigma_vim,
        ridge=float(cfg.get("oracle_cov_ridge", cfg.get("cov_ridge", 1e-8))),
    )

    return {
        # Used as true parameter in coverage
        "phi": np.asarray(phi_out, dtype=float),

        # Diagnostics
        "phi_truth": np.asarray(phi_out, dtype=float),
        "phi_truth_source": truth_source,
        "phi_mc": np.asarray(phi_mc, dtype=float),
        "phi_boot": phi_mat,

        # Used as known covariance in inference
        "Sigma_vim": Sigma_vim,
        "Sigma_hat_at_n": Sigma_vim / float(n_obs),

        # Diagnostics for Sigma source
        "Sigma_boot": _vi_sym_ridge(
            Sigma_boot,
            ridge=float(cfg.get("cov_ridge", 1e-8)),
        ),
        "Sigma_if": _vi_sym_ridge(
            Sigma_if,
            ridge=float(cfg.get("cov_ridge", 1e-8)),
        ),

        "n_oracle": int(n_oracle),
        "oracle_reps": int(oracle_reps),
        "covariance_source": covariance_source,
        "estimator": "population_true_nuisance",
    }


def _parse_vi_learner_params(tree_params):
    """
    Backward compatible parser.

    Old style:
        tree_params = {"max_depth": 3, "min_samples_leaf": 15}

    New style:
        tree_params = {
            "learner": "extra_trees",
            "learner_params": {...}
        }
    """
    params = dict(tree_params or {})

    learner = params.pop("learner", None)
    if learner is None:
        learner = params.pop("model", None)
    if learner is None:
        learner = params.pop("estimator", None)

    if learner is None:
        learner = "decision_tree"

    nested_params = params.pop("learner_params", None)

    if nested_params is None:
        learner_params = params
    else:
        learner_params = dict(nested_params)
        learner_params.update(params)

    return str(learner).lower(), learner_params


def _make_vi_regressor(learner, params, random_state):
    """
    Construct a regression learner for nuisance estimation.

    Supports tree-based and non-tree learners:
        - decision_tree
        - random_forest
        - extra_trees
        - gradient_boosting
        - hist_gradient_boosting
        - spline_gam
        - polynomial_ridge
        - kernel_ridge
        - projection_spline
        - neural_net
    """
    learner = str(learner).lower()
    params = dict(params or {})

    # =====================================================
    # Tree-based learners
    # =====================================================
    if learner in {"decision_tree", "tree", "dt"}:
        defaults = dict(
            max_depth=None,
            min_samples_leaf=20,
        )
        defaults.update(params)
        defaults["random_state"] = int(random_state)
        return DecisionTreeRegressor(**defaults)

    if learner in {"random_forest", "rf"}:
        defaults = dict(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            max_features=1.0,
            bootstrap=True,
            n_jobs=-1,
        )
        defaults.update(params)
        defaults["random_state"] = int(random_state)
        return RandomForestRegressor(**defaults)

    if learner in {"extra_trees", "extratrees", "et"}:
        defaults = dict(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
        )
        defaults.update(params)
        defaults["random_state"] = int(random_state)
        return ExtraTreesRegressor(**defaults)

    if learner in {"gradient_boosting", "gb", "gbrt"}:
        defaults = dict(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=10,
            subsample=0.8,
        )
        defaults.update(params)
        defaults["random_state"] = int(random_state)
        return GradientBoostingRegressor(**defaults)

    if learner in {"hist_gradient_boosting", "hist_gb", "hgb"}:
        defaults = dict(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
        )
        defaults.update(params)
        defaults["random_state"] = int(random_state)
        return HistGradientBoostingRegressor(**defaults)

    # =====================================================
    # Additive spline / GAM-style learner
    # =====================================================
    if learner in {"spline_gam", "gam", "additive_spline"}:
        n_knots = int(params.pop("n_knots", 6))
        degree = int(params.pop("degree", 3))
        alpha = float(params.pop("alpha", 1e-4))

        return make_pipeline(
            StandardScaler(),
            SplineTransformer(
                n_knots=n_knots,
                degree=degree,
                include_bias=False,
                extrapolation="constant",
            ),
            Ridge(alpha=alpha, fit_intercept=True),
        )

    # =====================================================
    # Polynomial ridge learner
    # =====================================================
    if learner in {"polynomial_ridge", "poly_ridge"}:
        degree = int(params.pop("degree", 2))
        alpha = float(params.pop("alpha", 1e-3))
        interaction_only = bool(params.pop("interaction_only", False))

        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=interaction_only,
            ),
            Ridge(alpha=alpha, fit_intercept=True),
        )

    # =====================================================
    # Kernel ridge approximation
    # =====================================================
    if learner in {"kernel_ridge", "nystroem_ridge", "rbf_ridge"}:
        gamma = float(params.pop("gamma", 1.0))
        n_components = int(params.pop("n_components", 200))
        alpha = float(params.pop("alpha", 1e-3))

        return make_pipeline(
            StandardScaler(),
            Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=n_components,
                random_state=int(random_state),
            ),
            Ridge(alpha=alpha, fit_intercept=True),
        )

    # =====================================================
    # Projection-pursuit-like learner
    # =====================================================
    if learner in {"projection_spline", "ppr", "random_projection_spline"}:
        n_projections = int(params.pop("n_projections", 10))
        n_knots = int(params.pop("n_knots", 6))
        degree = int(params.pop("degree", 3))
        alpha = float(params.pop("alpha", 1e-4))

        return make_pipeline(
            StandardScaler(),
            RandomProjectionSplineFeatures(
                n_projections=n_projections,
                n_knots=n_knots,
                degree=degree,
                random_state=int(random_state),
            ),
            Ridge(alpha=alpha, fit_intercept=True),
        )

    # =====================================================
    # Neural network learner
    # =====================================================
    if learner in {"neural_net", "nn", "mlp"}:
        hidden_layer_sizes = params.pop("hidden_layer_sizes", (32,))
        activation = str(params.pop("activation", "relu"))
        alpha = float(params.pop("alpha", 1e-3))
        learning_rate_init = float(params.pop("learning_rate_init", 1e-3))
        max_iter = int(params.pop("max_iter", 500))
        early_stopping = bool(params.pop("early_stopping", True))

        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=early_stopping,
                n_iter_no_change=20,
                random_state=int(random_state),
            ),
        )

    raise ValueError(
        "Unknown VI learner. Use one of "
        "{'decision_tree', 'random_forest', 'extra_trees', "
        "'gradient_boosting', 'hist_gradient_boosting', "
        "'spline_gam', 'polynomial_ridge', 'kernel_ridge', "
        "'projection_spline', 'neural_net'}."
    )

def estimate_variable_importance_model_cf(
    X,
    O,
    *,
    n_folds=5,
    tree_params=None,
    random_state=0,
    cov_ridge=1e-8,
    min_var_y=1e-12,
):
    """
    Cross-fitted model-based VI estimator.

    This follows Algorithm feature.importance:
        1. fit m and m_{-j} on training folds
        2. predict on held-out fold
        3. compute phi_hat and estimated IF covariance

    The learner can be tree, GAM, kernel, NN, etc.
    """
    X = np.asarray(X, dtype=float)
    O = np.asarray(O, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    n, p = X.shape

    if O.size != n:
        raise ValueError("O must have length n.")

    learner, learner_params = _parse_vi_learner_params(tree_params)

    n_folds_eff = int(min(max(2, int(n_folds)), n))
    kf = KFold(
        n_splits=n_folds_eff,
        shuffle=True,
        random_state=int(random_state),
    )

    m_hat = np.full(n, np.nan)
    m_minus = np.full((n, p), np.nan)

    for fold_id, (tr, te) in enumerate(kf.split(X)):
        base_seed = int(random_state) + 10000 * fold_id

        # Full model m(X)
        model = _make_vi_regressor(
            learner,
            learner_params,
            random_state=base_seed,
        )
        model.fit(X[tr], O[tr])
        m_hat[te] = model.predict(X[te])

        # Leave-one-variable-out models m_{-j}(X_{-j})
        for j in range(p):
            keep = [a for a in range(p) if a != j]

            model_j = _make_vi_regressor(
                learner,
                learner_params,
                random_state=base_seed + 97 * (j + 1),
            )
            model_j.fit(X[tr][:, keep], O[tr])
            m_minus[te, j] = model_j.predict(X[te][:, keep])

    if np.any(~np.isfinite(m_hat)) or np.any(~np.isfinite(m_minus)):
        raise RuntimeError("Non-finite nuisance prediction in VI cross-fitting.")

    out = estimate_vi_from_nuisance_predictions(
        X,
        O,
        m_hat,
        m_minus,
        phi_for_if=None,
        cov_ridge=cov_ridge,
        min_var_y=min_var_y,
    )

    out["estimator"] = "model_cf"
    out["learner"] = learner
    out["m_hat"] = m_hat
    out["m_minus"] = m_minus

    return out


# Backward-compatible alias
estimate_variable_importance_tree_cf = estimate_variable_importance_model_cf

def _coerce_variable_importance_sample(obj, *, p=None, name="VI sample"):
    if isinstance(obj, dict):
        if "X" not in obj or "O" not in obj:
            raise ValueError(f"{name} dict must contain keys 'X' and 'O'.")
        X = np.asarray(obj["X"], dtype=float)
        O = np.asarray(obj["O"], dtype=float).reshape(-1)
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        X = np.asarray(obj[0], dtype=float)
        O = np.asarray(obj[1], dtype=float).reshape(-1)
    else:
        raise TypeError(f"{name} must be a dict with keys X/O or a tuple (X, O).")

    if X.ndim != 2:
        raise ValueError(f"{name}: X must be 2D.")
    if O.size != X.shape[0]:
        raise ValueError(f"{name}: O length must equal X.shape[0].")
    if p is not None and X.shape[1] != int(p):
        raise ValueError(f"{name}: X must have p={p} columns, got {X.shape[1]}.")

    return X, O


def generate_variable_importance_replicates(setting, n_obs, B, seed=0):
    cfg = _normalize_variable_importance_config(setting, n_obs=n_obs)
    rng = np.random.default_rng(seed)
    out = []

    for _ in range(int(B)):
        X, O = generate_variable_importance_data(
            int(n_obs),
            p=cfg["p"],
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
        out.append({"X": X, "O": O})

    return out




def estimate_variable_importance_statistic(
    X,
    O,
    *,
    cfg,
    random_state=0,
):
    """
    Dispatch VI statistic used in each simulation replication.

    Recommended new mode:
        cfg["vi_statistic"] = "fixed_nuisance"

    Legacy modes:
        "true_nuisance"
        "model_cf"
    """
    cfg = _normalize_variable_importance_config(cfg, n_obs=X.shape[0])

    vi_statistic = str(
        cfg.get("vi_statistic", "fixed_nuisance" if cfg.get("fixed_nuisance", None) is not None else "model_cf")
    ).lower()

    phi_truth, _ = _vi_truth_from_setting(cfg)

    # --------------------------------------------------
    # New main path: fixed nuisance from holdout
    # --------------------------------------------------
    if vi_statistic in {"fixed_nuisance", "holdout_fixed", "fixed_holdout"}:
        fixed_nuisance = cfg.get("fixed_nuisance", None)
        if fixed_nuisance is None:
            raise ValueError(
                "cfg['fixed_nuisance'] is required when vi_statistic='fixed_nuisance'."
            )

        out = estimate_variable_importance_fixed_nuisance(
            X,
            O,
            fixed_nuisance=fixed_nuisance,
            phi_for_if=phi_truth,
            cov_ridge=cfg.get("cov_ridge", 1e-8),
            min_var_y=cfg.get("min_var_y", 1e-12),
        )
        out["vi_statistic_used"] = "fixed_nuisance"
        return out

    # --------------------------------------------------
    # Legacy oracle nuisance path
    # --------------------------------------------------
    if vi_statistic in {"true_nuisance", "oracle", "oracle_statistic"}:
        out = estimate_variable_importance_true_nuisance(
            X,
            O,
            m_func=cfg.get("m_func", None),
            m_minus_func=cfg.get("m_minus_func", None),
            beta_signal=cfg.get("beta_signal", cfg.get("beta", None)),
            phi_for_if=phi_truth,
            cov_ridge=cfg.get("cov_ridge", 1e-8),
            min_var_y=cfg.get("min_var_y", 1e-12),
            cfg=cfg,
        )
        out["vi_statistic_used"] = "true_nuisance"
        return out

    # --------------------------------------------------
    # Legacy cross-fitted model path
    # Not recommended for this new holdout experiment.
    # --------------------------------------------------
    if vi_statistic in {"model_cf", "model", "plugin_statistic"}:
        out = estimate_variable_importance_model_cf(
            X,
            O,
            n_folds=cfg["n_folds"],
            tree_params=cfg.get("tree_params", None),
            random_state=random_state,
            cov_ridge=cfg.get("cov_ridge", 1e-8),
            min_var_y=cfg.get("min_var_y", 1e-12),
        )
        out["vi_statistic_used"] = "model_cf"
        return out

    raise ValueError(
        "Unknown vi_statistic. Use 'fixed_nuisance', 'true_nuisance', or 'model_cf'."
    )
