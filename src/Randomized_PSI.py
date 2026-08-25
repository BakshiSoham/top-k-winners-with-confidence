import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

from .Inverse_Pivot import (
    selective_confidence_interval_bisect_single_fast,
    _conditional_density_grid_fixed,
)

def cdf_from_density(grid_u, density):
    grid_u = np.asarray(grid_u, dtype=float)
    density = np.asarray(density, dtype=float)
    dx = np.diff(grid_u)
    area_segments = 0.5 * (density[:-1] + density[1:]) * dx
    cum_area = np.concatenate([[0.0], np.cumsum(area_segments)])
    total_area = cum_area[-1]+0.000000001
    if total_area <= 0:
        raise ValueError("Integral of Density is Non-positive")
    F_vals = cum_area / total_area
    cdf_spline = interp1d(grid_u, F_vals, kind="cubic",fill_value="extrapolate",bounds_error=False)
    def F(x):
        return np.clip(cdf_spline(x), 0.0, 1.0)
    return F, (grid_u, F_vals)

class TopKSelectionModel:
    def __init__(self, X: np.ndarray, k: int, H0_mu,true_Sigma,utility_fn=None,epsilon=1.,grid_size=1000, sel_scale='adaptive'):
        self.X = X
        self.M = len(X)
        self.k = k
        self.utility_fn = utility_fn or self.default_utility
        #self.utilities = self.compute_utilities()
        self.selected_indices = self.select_top_k()
        self.epsilon=epsilon
        self.true_mu=H0_mu
        self.true_Sigma=true_Sigma 
        self.grid_size=grid_size
        self.sel_scale=sel_scale  # 'adaptive' or 'none' — must match randomized_selected_top_k(scale=)

    def default_utility(self,X: np.ndarray) -> float:
        col_means = X
        return col_means


    def select_top_k(self, X=None):
        if X is None:
            X = self.X
        X = np.asarray(X, dtype=float).reshape(-1)   # (p,)
    
        utilities = self.utility_fn(X)               # should return (p,)
        utilities = np.asarray(utilities, dtype=float).reshape(-1)
    
        return np.argsort(utilities)[-self.k:][::-1]

    def top_k_additive_utility(self,S,X=None):
        if X is None:
            X=self.X
        return self.utility_fn(X[list(S)]).sum()

    def get_projection(self, index_set, X=None, Sigma=None):
        if X is None:
            X = self.X
        if Sigma is None:
            Sigma = self.true_Sigma
    
        k = self.k
    
        X = np.asarray(X, dtype=float).reshape(-1, 1) 
    
        E_S = np.zeros((self.M, k))
        for i, idx in enumerate(index_set):
            E_S[idx, i] = 1.0
    
        Sigma_S = E_S.T @ Sigma @ E_S                  # (k,k)
        Sigma_inv_proj = Sigma @ E_S @ np.linalg.inv(Sigma_S)   # (p,k)
        X_S = E_S.T @ X                                # (k,1)
        proj = Sigma_inv_proj @ X_S                    # (p,1)
        orth = X - proj                                # (p,1)
    
        return proj, orth

    def randomized_selected_top_k(
        self,
        X=None,
        k=None,
        epsilon=None,
        method="gibbs_dp",
        scale="adaptive",
        seed=None,
        return_log_prob=False,
        scale_floor=1e-8,
    ):
        if X is None:
            X = self.X
        if k is None:
            k = self.k
        if epsilon is None:
            epsilon = self.epsilon
    
        X = np.asarray(X, dtype=float).reshape(-1)
        M = len(X)
    
        scores = np.asarray(self.utility_fn(X), dtype=float).reshape(-1)
    
        if scores.shape[0] != M:
            raise ValueError("utility_fn(X) must return a length-M vector.")
    
        # --------------------------------------------------
        # scale_value
        # --------------------------------------------------
        if scale == "adaptive":
            s_bar = float(np.mean(scores))
            centered_scores = scores - s_bar
    
            if M <= 1:
                subset_score_var = 0.0
            else:
                item_var = float(np.mean(centered_scores ** 2))
                subset_score_var = float(k * (M - k) / (M - 1) * item_var)
    
            # snoise(t)^2 = Var_E[s_E(t)] + eps_stab^2
            # Var_E[s_E] = k(M-k)/(M-1) * item_var  (subset-score variance)
            scale_value = float(np.sqrt(subset_score_var + scale_floor ** 2))
    
            # Subtracting the mean does not change selection probs
            scores_for_logits = centered_scores
    
        elif scale == "mean":
            scale_value = np.mean(np.abs(scores)) + 1e-12
            scores_for_logits = scores
    
        elif scale == "range":
            scale_value = np.ptp(scores) + 1e-12
            scores_for_logits = scores
    
        elif scale == "none":
            scale_value = 1.0
            scores_for_logits = scores
    
        else:
            raise ValueError("scale must be one of {'adaptive', 'mean', 'range', 'none'}.")
    
        logits = ((float(epsilon) / scale_value) * scores_for_logits).reshape(-1)
    
        rng = np.random.default_rng(seed)
    
        if method == "gibbs_dp":
            # Exact sampler for the paper's fixed-size Gibbs rule.
            # L[i, m] is the log partition function for choosing m
            # items from the first i item-level logits.
            if not (0 <= k <= M):
                raise ValueError(f"k must lie in [0, {M}], got {k}.")

            L = np.full((M + 1, k + 1), -np.inf, dtype=float)
            L[:, 0] = 0.0
            for i in range(1, M + 1):
                for m_take in range(1, min(i, k) + 1):
                    L[i, m_take] = np.logaddexp(
                        L[i - 1, m_take],
                        logits[i - 1] + L[i - 1, m_take - 1],
                    )

            selected = []
            m_left = k
            for i in range(M, 0, -1):
                if m_left == 0:
                    break
                log_q = (
                    logits[i - 1]
                    + L[i - 1, m_left - 1]
                    - L[i, m_left]
                )
                q_i = float(np.clip(np.exp(min(0.0, log_q)), 0.0, 1.0))
                if rng.random() < q_i:
                    selected.append(i - 1)
                    m_left -= 1

            if m_left != 0:
                raise RuntimeError("Exact Gibbs sampler failed to select k items.")
            selected_subset = tuple(sorted(selected))
        else:
            raise NotImplementedError("Only method='gibbs_dp' is implemented.")

        # Previous implementation (disabled): independent itemwise
        # Gumbels followed by top-k induce a Plackett--Luce set law,
        # not the fixed-size Gibbs law used by prob_subset_dp().
        # g = rng.gumbel(size=M)
        # y = logits + g
        # idx = np.argpartition(y, -k)[-k:]
        # selected_subset = tuple(sorted(int(i) for i in idx))
    
        return selected_subset, None

    def prob_subset_dp(
        self,
        X,
        S,
        epsilon,
        *,
        scale="adaptive",
        return_log=False,
        scale_floor=1e-8,
    ):
        if X is None:
            X = self.X
    
        X = np.asarray(X, dtype=float).reshape(-1)
        S = tuple(sorted(int(i) for i in S))
        k = len(S)
    
        s = np.asarray(self.utility_fn(X), dtype=float).reshape(-1)
        M = s.size
    
        if s.shape[0] != X.shape[0]:
            raise ValueError("utility_fn(X) must return a vector with the same length as X.")
    
        if not (0 <= k <= M):
            raise ValueError(f"Invalid subset size k={k} for M={M}.")
    
        # --------------------------------------------------
        # scale_value
        # --------------------------------------------------
        if scale == "adaptive":
            s_bar = float(np.mean(s))
            centered_s = s - s_bar
    
            if M <= 1:
                subset_score_var = 0.0
            else:
                item_var = float(np.mean(centered_s ** 2))
                subset_score_var = float(k * (M - k) / (M - 1) * item_var)
    
            # snoise(t)^2 = Var_E[s_E(t)] + eps_stab^2  (matches theory, §2)
            scale_value = float(np.sqrt(subset_score_var + scale_floor ** 2))
    
            # Centered scores: s_E - s_bar = sum_{j in E}(X_j - X_bar)
            s_for_logw = centered_s
    
        elif scale == "mean":
            scale_value = max(1e-12, float(k) * float(np.mean(s)))
            s_for_logw = s
    
        elif scale == "range":
            scale_value = max(
                1e-12,
                float(k) * (float(np.max(s)) - float(np.min(s)))
            )
            s_for_logw = s
    
        elif scale == "none":
            scale_value = 1.0
            s_for_logw = s
    
        else:
            raise ValueError("scale must be one of {'adaptive', 'mean', 'range', 'none'}.")
    
        logw = (float(epsilon) / scale_value) * s_for_logw
    
        log_num = float(np.sum(logw[list(S)]))
    
        a = np.full(k + 1, -np.inf, dtype=float)
        a[0] = 0.0
    
        for i in range(M):
            lw = float(logw[i])
            upper = min(k, i + 1)
            for ell in range(upper, 0, -1):
                a[ell] = np.logaddexp(a[ell], lw + a[ell - 1])
    
        logZ = float(a[k])
        logp = log_num - logZ
    
        return logp if return_log else float(np.exp(logp))

    def direct_selected_top_k(self, X=None, k=None):
        if X is None:
            X = self.X
        if k is None:
            k = self.k
        u = self.utility_fn(X)
        return tuple(np.argsort(-u)[:k])

    def conditional_density_grid(
        self,u, v, S,
        grid_size=None,
        span=20.0,
        true_mu=None,
        true_sigma=None,
        density_cutoff=1e-6
    ):
        epsilon = self.epsilon
        mu = self.true_mu if true_mu is None else true_mu
        Sigma = self.true_Sigma if true_sigma is None else true_sigma
        if grid_size is None:
            grid_size = self.grid_size
        k = len(S)
        M = self.M
        E_S = np.zeros((M, k))
        for i, idx in enumerate(S):
            E_S[idx, i] = 1.0
        Sigma_S = E_S.T @ Sigma @ E_S
        Sigma_S_inv = np.linalg.inv(Sigma_S)
        mu_S = np.asarray(mu[list(S)], dtype=float)
        results_dict = {}
        for j in range(k):
            mu_j = mu_S[j]
            mu_j_cen=u[j]
            sig_j = np.sqrt(Sigma_S[j, j])
            grid_j = np.linspace(mu_j_cen - span * sig_j, mu_j_cen + span * sig_j, grid_size)
            dens_vals = np.zeros_like(grid_j)
    
            for i, u_val in enumerate(grid_j):
                u_vec = u.copy()
                u_vec[j] = u_val  
                #X = self.utility_fn(Sigma @ E_S @ Sigma_S_inv @ u_vec + v.T)  
                X = Sigma @ E_S @ Sigma_S_inv @ u_vec + v.T
                phi_val = norm.pdf(u_val, loc=mu_j, scale=sig_j)
                dens_vals[i] = phi_val * self.prob_subset_dp(X, S, epsilon)

            #print(dens_vals)
            Z = np.trapz(dens_vals, grid_j)+0.00000001
            if Z <= 0:
                raise ValueError("Integral of density is non-positive.")
            dens_vals /= Z    
            results_dict[f"top{j+1}"] = {
                "grid": grid_j,
                "density": dens_vals,
                "var_index": S[j]
            }
        return results_dict


        
    def post_selection_inference_on_top_k(self,H0_mu=None, Sigma=None, alpha=0.05,
                    k=None, epsilon=None, grid_size=None, tail="two_tails",
                    span=4.0, density_cutoff=1e-4, S=None):
        if H0_mu is None: H0_mu = self.true_mu
        if Sigma is None: Sigma = self.true_Sigma
        if k is None: k = self.k
        if epsilon is None: epsilon = self.epsilon
        if grid_size is None: grid_size = self.grid_size 
        X=self.X
        pivots_per_rank = [[] for _ in range(k)]
        p_right_per_rank = [[] for _ in range(k)]
        p_two_per_rank = [[] for _ in range(k)]
        varidx_per_rank = [[] for _ in range(k)] 
        details_per_sim = []
        S_list = []
        #S, _ = self.randomized_selected_top_k(X=X, k=k, epsilon=epsilon)
        # S,_=self.randomized_selected_top_k(X=X, k=k, epsilon=epsilon)
        S = S
        _, v = self.get_projection(S, X, Sigma=Sigma) 
        u=X[S]
        dens_dict = self.conditional_density_grid(u=u,
            v=v, S=S, grid_size=grid_size, span=span,
            true_mu=H0_mu, true_sigma=Sigma, density_cutoff=density_cutoff
        )
        sim_records = []
        for r in range(1, k + 1):
            key = f"top{r}"
            if key not in dens_dict:
                continue  
            info = dens_dict[key]
            grid_j, dens_j, var_idx = info["grid"], info["density"], info["var_index"]
            F_j, _ = cdf_from_density(grid_j, dens_j)
            u_star_j = X[ var_idx]
            pivot_j = float(F_j(u_star_j).item())
            p_right_j = pivot_j
            p_two_j = 2.0 * min(pivot_j, 1.0 - pivot_j)
            pivots_per_rank[r - 1].append(pivot_j)
            p_right_per_rank[r - 1].append(p_right_j)
            p_two_per_rank[r - 1].append(p_two_j)
            varidx_per_rank[r - 1].append(var_idx)
            sim_records.append({
                "rank": r,
                "var_index": var_idx,
                "u_star": float(u_star_j.item()),
                "pivot": float(pivot_j),
                "p_right": float(p_right_j),
                "p_two": float(p_two_j),
                "grid": grid_j,
                "density": dens_j
            })   
        return sim_records   
        
    def confidence_interval_topk(
        self,
        S_obs=None,
        Sigma=None,
        alpha=0.05,
        k=None,
        epsilon=None,
        grid_size=None,
        ci_func=None,
        seed=None,
        verbose=True,
    ):
        if ci_func is None:
            ci_func = selective_confidence_interval_bisect_single_fast
        rng = np.random.default_rng(seed)
        if Sigma is None: Sigma = self.true_Sigma
        if k is None: k = self.k
        if epsilon is None: epsilon = self.epsilon
        if grid_size is None: grid_size = self.grid_size
        if ci_func is None:
            raise ValueError("Must provide ci_func, e.g. selective_confidence_interval_bisect_fast")

        per_rank_records = {r: [] for r in range(1, k + 1)}
        X = self.X
        if S_obs is None:
            S_obs, _ = self.randomized_selected_top_k(X=X, k=k, epsilon=epsilon)
        else:
            S_obs=S_obs
        S_obs = list(S_obs)
        _, v_obs = self.get_projection(S_obs, X,  Sigma=Sigma)
        utils = [self.utility_fn(X[ idx]) for idx in S_obs]
        order = np.argsort(utils)[::-1]
        S_sorted = [S_obs[i] for i in order]
        u_obs=X[S_obs]
        for rank, idx in enumerate(S_sorted, start=1):
            u_star = float(X[ idx])
            L, U = ci_func(self,[idx],  S_obs, u_obs,v_obs,u_star,alpha=alpha,seed=int(rng.integers(1e9)))
            per_rank_records[rank].append({ "idx": idx,"L": L,"U": U, })
        return { "S_obs": S_obs,"per_rank": per_rank_records,}

    def plot_conditional_density(
        self,
        S,
        v,
        theta,
        sigma,
        u=None,
        grid_size=None,
        span=5.0,
        n_samples=5000,
        bins=35,
        seed=123
    ):
        import numpy as np
        import matplotlib.pyplot as plt
    
        if grid_size is None:
            grid_size = self.grid_size
    
        S = list(S)
        if u is None:
            u = np.asarray(self.X[S], dtype=float)
        else:
            u = np.asarray(u, dtype=float)
    
        dens_dict = self.conditional_density_grid(
            u=u,
            v=v,
            S=S,
            grid_size=grid_size,
            span=span,
            true_mu=theta,
            true_sigma=sigma
        )
    
        plt.figure(figsize=(11, 5))
        rng = np.random.default_rng(seed)
    
        top_keys = sorted(
            [k for k in dens_dict.keys() if k.startswith("top")],
            key=lambda x: int(x.replace("top", ""))
        )
    
        for j, key in enumerate(top_keys, start=1):
            info = dens_dict[key]
            grid = np.asarray(info["grid"], dtype=float)
            density = np.asarray(info["density"], dtype=float)
            var_idx = info["var_index"]
    
            dx = np.diff(grid)
            mass = 0.5 * (density[:-1] + density[1:]) * dx
            mass = np.clip(mass, 0.0, None)
    
            if mass.sum() <= 0:
                continue
            mass = mass / mass.sum()
    
            local_rng = np.random.default_rng(seed + j)
            idx = local_rng.choice(len(dx), size=n_samples, p=mass)
            samples = grid[idx] + local_rng.random(n_samples) * (grid[idx + 1] - grid[idx])

            plt.hist(
                samples,
                bins=bins,
                density=True,
                alpha=0.25,
                edgecolor="white",
                label=f"Winner-{j} histogram"
            )
            plt.plot(
                grid,
                density,
                linewidth=2.2,
                label=f"Winner-{j} density"
            )
            plt.axvline(
                self.X[var_idx],
                linestyle="--",
                linewidth=1.6,
                label=f"Winner-{j} observed"
            )
    
        plt.xlabel("u")
        plt.ylabel("Density")
        plt.title("Conditional Densities for Selected Winners")
        plt.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5)
        )
        plt.tight_layout(rect=[0, 0, 0.72, 1])
        plt.show()
     
          
    
TopKSelectionModel.conditional_density_grid = (
    _conditional_density_grid_fixed
)


    
    
