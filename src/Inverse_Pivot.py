from functools import lru_cache

import numpy as np
from scipy.stats import norm

def selective_confidence_interval_bisect_single_safe(
    model,
    target_idx,          
    S_obs,
    u_obs,
    v_obs,              
    u_star,           
    alpha=0.05,
    search_scale=13.0,  
    init_points=13,      
    max_expand=8,        
    expand_mult=2.0,
    tol=1e-6,
    max_iter=120,
    seed=None
):
    rng = np.random.default_rng(seed)
    target_idx = int(np.atleast_1d(target_idx)[0])
    mu0   = float(model.true_mu[target_idx])
    sigma = float(np.sqrt(model.true_Sigma[target_idx, target_idx]))

    @lru_cache(maxsize=None)
    def F_at_theta(theta):
        mu_new = model.true_mu.astype(float).copy()
        mu_new[target_idx] = float(theta)
        dens_dict = model.conditional_density_grid(
            u=u_obs,
            v=v_obs,
            S=S_obs,
            grid_size=model.grid_size,
            true_mu=mu_new,
            true_sigma=model.true_Sigma
        )
        matched = None
        for info in dens_dict.values():
            if int(info["var_index"]) == target_idx:
                matched = info
                break
        if matched is None:
            return np.nan

        grid = np.asarray(matched["grid"], dtype=float)
        dens = np.asarray(matched["density"], dtype=float)

        dens = np.maximum(dens, 0.0)
        Z = np.trapz(dens, grid)
        if not np.isfinite(Z) or Z <= 1e-12:
            dx = np.diff(grid)
            if dx.size and np.allclose(dx, dx[0]):
                Z = dens.sum() * (grid[1] - grid[0])
            if not np.isfinite(Z) or Z <= 1e-12:
                return np.nan
        dens = dens / Z
        F_vals = np.concatenate([[0.0], np.cumsum(0.5*(dens[1:]+dens[:-1])*np.diff(grid))])
        F_vals = F_vals / F_vals[-1]
        if u_star <= grid[0]:
            return 0.0
        if u_star >= grid[-1]:
            return 1.0
        i = np.searchsorted(grid, u_star) - 1
        t = (u_star - grid[i]) / (grid[i+1] - grid[i])
        val = (1-t)*F_vals[i] + t*F_vals[i+1]
        return float(np.clip(val, 0.0, 1.0))

    lo0, hi0 = mu0 - search_scale*sigma, mu0 + search_scale*sigma
    k_in = max(0, int(init_points) - 2)
    if k_in > 0:
        inner = np.linspace(lo0, hi0, k_in)
        theta_grid = np.sort(np.unique(np.concatenate(([lo0, hi0], inner))))
    else:
        theta_grid = np.array([lo0, hi0], dtype=float)

    F_vals = np.array([F_at_theta(th) for th in theta_grid], dtype=float)
    if np.all(np.isnan(F_vals)):
        return (None, None)

    finite_mask = np.isfinite(F_vals)
    if finite_mask.sum() >= 2:
        F_start = F_vals[finite_mask][0]
        F_end   = F_vals[finite_mask][-1]
        increasing = (F_end >= F_start)
    else:
        increasing = True

    tL = (1 - alpha/2) if increasing else (alpha/2)
    tU = (alpha/2)      if increasing else (1 - alpha/2)
    def find_bracket(target, tg, fv):
        for i in range(len(tg) - 1):
            a, b   = tg[i], tg[i+1]
            Fa, Fb = fv[i], fv[i+1]
            if not np.isfinite(Fa) or not np.isfinite(Fb):
                continue
            if (Fa - target) * (Fb - target) < 0:
                return float(a), float(b)
        return None

    def expand_for(target):
        nonlocal theta_grid, F_vals
        for _ in range(max_expand + 1):
            br = find_bracket(target, theta_grid, F_vals)
            if br is not None:
                return br
            span = theta_grid[-1] - theta_grid[0]
            ctr  = 0.5 * (theta_grid[-1] + theta_grid[0])
            half = 0.5 * span * (expand_mult if expand_mult > 1 else 1.8)
            new_lo, new_hi = ctr - half, ctr + half

            theta_grid = np.linspace(new_lo, new_hi, max(len(theta_grid), init_points))
            F_vals = np.array([F_at_theta(th) for th in theta_grid], dtype=float)
            if np.all(np.isnan(F_vals)):
                return None
        return None

    brL = expand_for(tL)
    brU = expand_for(tU)
    def bisect_root(target, a, b):
        Fa, Fb = F_at_theta(a), F_at_theta(b)
        if (not np.isfinite(Fa)) or (not np.isfinite(Fb)) or (Fa - target) * (Fb - target) > 0:
            print("Fail to perform bisection")
            return None
        aa, bb = float(a), float(b)
        for _ in range(max_iter):
            m  = 0.5*(aa + bb)
            Fm = F_at_theta(m)
            if not np.isfinite(Fm):
                m = np.nextafter(m, bb)
                Fm = F_at_theta(m)
                if not np.isfinite(Fm):
                    return None
            if abs(Fm - target) < tol:
                return float(m)
            if (Fa - target)*(Fm - target) < 0:
                bb, Fb = m, Fm
            else:
                aa, Fa = m, Fm
        return float(0.5*(aa + bb))

    

    L = bisect_root(tL, *brL) if brL else None
    U = bisect_root(tU, *brU) if brU else None
    if (L is not None) and (U is not None) and (L > U):
        L, U = U, L
    if L is not None: L = float(np.round(L, 6))
    if U is not None: U = float(np.round(U, 6))

    
    return (L, U)




# ============================================================
# FIX FOR "SI COVERAGE = 0" + FAST RANDOMIZED-SI CI
# ============================================================
#
# Root cause of the original 0 coverage
# ──────────────────────────────────────
# `conditional_density_grid` built its integration grid around the
# OBSERVED value u[j], so when the bisection probed theta values
# outside [u[j]-span*sig, u[j]+span*sig], norm.pdf collapsed to 0,
# the normalisation Z was 0, F_at_theta returned NaN, no bracket
# was found, and CI = (None, None) → covered = False every rep.
#
# Fix
# ───
# (1) Widen the grid to cover BOTH the observed value AND the full
#     bisection search window.
# (2) Vectorised fast CI: selection-probability weights are
#     theta-independent and computed only once on the affine grid.
# (3) Add a `scale` parameter to prob_subset_dp_grid so inference
#     always uses the SAME scaling as selection (default 'adaptive').
#
# epsilon = 1/τ  (inverse temperature).
# tau(t) = tau · snoise(t),   snoise(t) = sqrt(Var_E[s_E(t)] + eps_stab^2)
# eps_stab = scale_floor  (tiny numerical stabiliser, NOT the same as epsilon)
#
# Why SI coverage can still be off for the Gaussian example
# ──────────────────────────────────────────────────────────
# The theory guarantee eps = 2·log C(m,k) is for the UNSCALED
# mechanism  P(S|X) ∝ exp(eps · Σ_{j∈S} X_j)  (scale='none').
# With scale='adaptive' the effective temperature is eps/scale(X),
# which is data-dependent; the same nominal eps does NOT match the
# theoretical threshold.  To run the theory check, set
#   sel_scale='none'  in TopKSelectionModel (and simulate_topk_hat)
#   EPSILON = 2*np.log(math.comb(M, K))          (≈14.08 for m=20,k=3)
# so that selection and inference both use the unscaled model.
# ============================================================

import math as _math

def _esp_logZ_grid(logw, k):
    """log of the k-th elementary symmetric polynomial for each row of logw (G,M)."""
    G, M = logw.shape
    a = np.full((G, k + 1), -np.inf)
    a[:, 0] = 0.0
    for i in range(M):
        lw = logw[:, i]
        for ell in range(min(k, i + 1), 0, -1):
            a[:, ell] = np.logaddexp(a[:, ell], lw + a[:, ell - 1])
    return a[:, k]


def prob_subset_dp_grid(X_grid, S, epsilon, utility_fn, scale_floor=1e-8, scale="adaptive"):
    """
    Vectorised selection probability of subset S for each row of X_grid (G,M).

    Parameters
    ----------
    scale : 'adaptive' (default) or 'none'
        'adaptive' — logit_j = eps/sqrt(item_var) * (s_j - s_bar)
                     (scale-invariant; matches randomized_selected_top_k default)
        'none'     — logit_j = eps * s_j
                     (unscaled Plackett-Luce; matches theory eps=2log C(m,k))
    """
    s = np.asarray(utility_fn(X_grid), dtype=float)          # (G,M)
    if scale == "adaptive":
        k_sub = len(S)
        M_sub = s.shape[1]
        centered = s - s.mean(axis=1, keepdims=True)                          # (G,M)
        item_var         = np.mean(centered ** 2, axis=1)                     # (G,)
        subset_score_var = k_sub * (M_sub - k_sub) / (M_sub - 1) * item_var  # (G,) Var_E[s_E]
        scale_value = np.sqrt(subset_score_var + scale_floor ** 2)            # (G,) snoise
        logw = (float(epsilon) / scale_value)[:, None] * centered
    elif scale == "none":
        logw = float(epsilon) * s
    else:
        raise ValueError(f"scale must be 'adaptive' or 'none', got {scale!r}")
    logZ = _esp_logZ_grid(logw, len(S))
    log_num = logw[:, list(S)].sum(axis=1)
    return np.exp(log_num - logZ)


def selective_confidence_interval_bisect_single_fast(
    model, target_idx, S_obs, u_obs, v_obs, u_star,
    alpha=0.05, search_scale=13.0, init_points=13, max_expand=8,
    expand_mult=2.0, tol=1e-6, max_iter=120, seed=None,
    span=20.0, grid_size=None,
):
    """
    Vectorised selective CI for one selected coordinate.

    Uses model.sel_scale ('adaptive' or 'none') for prob_subset_dp_grid,
    so inference always mirrors the selection mechanism.
    """
    target_idx = int(np.atleast_1d(target_idx)[0])
    S = list(S_obs)
    j = S.index(target_idx)
    Sigma = np.asarray(model.true_Sigma, dtype=float)
    M, k = model.M, len(S)
    if grid_size is None:
        grid_size = model.grid_size
    mu0 = float(model.true_mu[target_idx])
    sigma = float(np.sqrt(Sigma[target_idx, target_idx]))
    sel_scale = getattr(model, 'sel_scale', 'adaptive')

    # affine map X(u_j) = c0 + a_j * u_j
    E_S = np.zeros((M, k))
    for i, idx in enumerate(S):
        E_S[idx, i] = 1.0
    Sigma_S = E_S.T @ Sigma @ E_S
    A = Sigma @ E_S @ np.linalg.inv(Sigma_S)
    a_j = A[:, j]
    u_base = np.asarray(u_obs, dtype=float).copy(); u_base[j] = 0.0
    c0 = A @ u_base + np.asarray(v_obs, dtype=float).reshape(-1)

    sig_j = float(np.sqrt(Sigma_S[j, j]))
    mu_j_obs = float(np.asarray(u_obs, dtype=float)[j])

    # Grid covers both observed value AND bisection search range
    pad = span * sig_j
    grid_lo = min(mu_j_obs - pad, mu0 - search_scale * sigma - pad)
    grid_hi = max(mu_j_obs + pad, mu0 + search_scale * sigma + pad)
    grid = np.linspace(grid_lo, grid_hi, int(grid_size))
    X_grid = c0[None, :] + grid[:, None] * a_j[None, :]       # (G,M)

    # theta-independent selection weights (computed once)
    sel = np.maximum(
        prob_subset_dp_grid(X_grid, S, model.epsilon, model.utility_fn,
                            scale=sel_scale), 0.0,
    )
    dgrid = np.diff(grid)

    def F_at_theta(theta):
        dens = np.maximum(norm.pdf(grid, loc=theta, scale=sig_j) * sel, 0.0)
        Z = np.trapz(dens, grid)
        if not np.isfinite(Z) or Z <= 1e-12:
            # Z→0 when theta is far below the selection region.
            # All remaining conditional mass is below u_star → CDF = 1.0.
            return 1.0
        dens = dens / Z
        F = np.concatenate(
            [[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * dgrid)]
        )
        F = F / F[-1]
        if u_star <= grid[0]:
            return 0.0
        if u_star >= grid[-1]:
            return 1.0
        i = np.searchsorted(grid, u_star) - 1
        t = (u_star - grid[i]) / (grid[i + 1] - grid[i])
        return float(np.clip((1 - t) * F[i] + t * F[i + 1], 0.0, 1.0))

    lo0, hi0 = mu0 - search_scale * sigma, mu0 + search_scale * sigma
    k_in = max(0, int(init_points) - 2)
    theta_grid = np.sort(np.unique(np.concatenate(
        ([lo0, hi0], np.linspace(lo0, hi0, k_in) if k_in > 0 else [])
    )))
    F_vals = np.array([F_at_theta(t) for t in theta_grid])
    if np.all(np.isnan(F_vals)):
        return (None, None)
    fin = np.isfinite(F_vals)
    increasing = (F_vals[fin][-1] >= F_vals[fin][0]) if fin.sum() >= 2 else True
    tL = (1 - alpha / 2) if increasing else (alpha / 2)
    tU = (alpha / 2) if increasing else (1 - alpha / 2)

    def find_bracket(target, tg, fv):
        for i in range(len(tg) - 1):
            Fa, Fb = fv[i], fv[i + 1]
            if np.isfinite(Fa) and np.isfinite(Fb) and (Fa - target) * (Fb - target) < 0:
                return float(tg[i]), float(tg[i + 1])
        return None

    def expand_for(target):
        nonlocal theta_grid, F_vals
        for _ in range(max_expand + 1):
            br = find_bracket(target, theta_grid, F_vals)
            if br is not None:
                return br
            sp = theta_grid[-1] - theta_grid[0]
            ctr = 0.5 * (theta_grid[-1] + theta_grid[0])
            half = 0.5 * sp * (expand_mult if expand_mult > 1 else 1.8)
            theta_grid = np.linspace(ctr - half, ctr + half, max(len(theta_grid), init_points))
            F_vals = np.array([F_at_theta(t) for t in theta_grid])
            if np.all(np.isnan(F_vals)):
                return None
        return None

    brL, brU = expand_for(tL), expand_for(tU)

    def bisect_root(target, a, b):
        Fa, Fb = F_at_theta(a), F_at_theta(b)
        if (not np.isfinite(Fa)) or (not np.isfinite(Fb)) or (Fa - target) * (Fb - target) > 0:
            return None
        aa, bb = float(a), float(b)
        for _ in range(max_iter):
            m = 0.5 * (aa + bb); Fm = F_at_theta(m)
            if not np.isfinite(Fm):
                return None
            if abs(Fm - target) < tol:
                return float(m)
            if (Fa - target) * (Fm - target) < 0:
                bb, Fb = m, Fm
            else:
                aa, Fa = m, Fm
        return float(0.5 * (aa + bb))

    L = bisect_root(tL, *brL) if brL else None
    U = bisect_root(tU, *brU) if brU else None
    if (L is not None) and (U is not None) and (L > U):
        L, U = U, L
    if L is not None: L = float(np.round(L, 6))
    if U is not None: U = float(np.round(U, 6))
    return (L, U)


# ── Patch conditional_density_grid to also use model.sel_scale ─────────────

def _conditional_density_grid_fixed(
    self, u, v, S,
    grid_size=None, span=20.0,
    true_mu=None, true_sigma=None, density_cutoff=1e-6,
    theta_search_lo=None, theta_search_hi=None,
):
    """
    Widened integration grid (covers observed AND search window) +
    respects model.sel_scale for prob_subset_dp.
    """
    epsilon   = self.epsilon
    sel_scale = getattr(self, 'sel_scale', 'adaptive')
    mu    = self.true_mu    if true_mu    is None else true_mu
    Sigma = self.true_Sigma if true_sigma is None else true_sigma
    if grid_size is None:
        grid_size = self.grid_size
    k_ = len(S)
    M_ = self.M
    E_S = np.zeros((M_, k_))
    for i, idx in enumerate(S):
        E_S[idx, i] = 1.0
    Sigma_S     = E_S.T @ Sigma @ E_S
    Sigma_S_inv = np.linalg.inv(Sigma_S)
    mu_S = np.asarray(mu[list(S)], dtype=float)
    results_dict = {}
    for j in range(k_):
        mu_j_obs   = u[j]
        sig_j      = np.sqrt(Sigma_S[j, j])
        sigma_full = float(np.sqrt(Sigma[S[j], S[j]]))
        mu0        = float(self.true_mu[S[j]])
        t_lo = (mu0 - 13.0 * sigma_full) if theta_search_lo is None else theta_search_lo
        t_hi = (mu0 + 13.0 * sigma_full) if theta_search_hi is None else theta_search_hi
        pad     = span * sig_j
        grid_lo = min(mu_j_obs - pad, t_lo - pad)
        grid_hi = max(mu_j_obs + pad, t_hi + pad)
        grid_j  = np.linspace(grid_lo, grid_hi, int(grid_size))
        dens_vals = np.zeros_like(grid_j)
        for i_g, u_val in enumerate(grid_j):
            u_vec    = u.copy(); u_vec[j] = u_val
            X        = Sigma @ E_S @ Sigma_S_inv @ u_vec + v.T
            phi_val  = norm.pdf(u_val, loc=mu_S[j], scale=sig_j)
            dens_vals[i_g] = phi_val * self.prob_subset_dp(X, S, epsilon,
                                                           scale=sel_scale)
        Z = np.trapz(dens_vals, grid_j) + 1e-12
        if Z <= 0:
            raise ValueError("Integral of density is non-positive.")
        dens_vals /= Z
        results_dict[f"top{j + 1}"] = {
            "grid": grid_j, "density": dens_vals, "var_index": S[j],
        }
    return results_dict








