import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import log_ndtr

@dataclass
class SelectiveInferenceResult:
    rank: int
    index: int
    observed_value: float
    observed_target_mean: float
    trunc_lower: float
    trunc_upper: float
    sigma: float
    pivot_at_truth: float
    pvalue_two_sided: float
    ci_lower: float
    ci_upper: float
    covered: bool
    length: float


class PolyhedralTopKInference:
    def __init__(
        self,
        X: np.ndarray,
        k: int,
        H0_mu: np.ndarray,
        Sigma: np.ndarray,
        utility_fn=None,
        grid_size: int = 500,
        alpha: float = 0.05,
        selected_subset=None,
    ):
        self.X = np.asarray(X, dtype=float)
        self.k = int(k)
        self.mu = np.asarray(H0_mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.grid_size = int(grid_size)
        self.alpha = float(alpha)

        if utility_fn is None:
            self.utility_fn = lambda x: x
        else:
            self.utility_fn = utility_fn

        if self.X.ndim != 1:
            raise ValueError("X must be a 1D array.")
        if self.mu.ndim != 1:
            raise ValueError("H0_mu must be a 1D array.")
        if self.Sigma.ndim != 2:
            raise ValueError("Sigma must be a 2D array.")
        if len(self.X) != len(self.mu):
            raise ValueError("X and H0_mu must have the same length.")
        if self.Sigma.shape != (len(self.X), len(self.X)):
            raise ValueError("Sigma must be p x p.")
        if not (1 <= self.k < len(self.X)):
            raise ValueError("Need 1 <= k < p.")

        self.p = len(self.X)

        self.utility = np.asarray(self.utility_fn(self.X), dtype=float)
        if self.utility.shape != self.X.shape:
            raise ValueError("utility_fn must return a vector of same shape as X.")

        self.order_desc = np.argsort(self.utility)[::-1]

        if selected_subset is None:
            self.selected_set = self._top_k_indices(self.utility, self.k)
        else:
            selected_subset = np.asarray(selected_subset, dtype=int)
        
            if selected_subset.ndim != 1:
                raise ValueError("selected_subset must be a 1D array-like of indices.")
            if len(selected_subset) != self.k:
                raise ValueError(f"selected_subset must have length k={self.k}.")
            if len(np.unique(selected_subset)) != self.k:
                raise ValueError("selected_subset contains duplicated indices.")
            if np.any(selected_subset < 0) or np.any(selected_subset >= self.p):
                raise ValueError("selected_subset contains invalid indices.")
        
            self.selected_set = selected_subset.copy()
        
        self.not_selected_set = np.array(
            [j for j in range(self.p) if j not in self.selected_set],
            dtype=int
        )
        
        # still define t_kplus1 from the observed utility ranking
        self.t_kplus1 = float(self.utility[self.order_desc[self.k]])

    @staticmethod
    def _top_k_indices(x: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(x)[::-1][:k]

    def _ranked_selected_from_subset(self) -> np.ndarray:
        return self.selected_set[np.argsort(self.utility[self.selected_set])[::-1]]

    def _sigma_j(self, j0: int) -> float:
        val = float(self.Sigma[j0, j0])
        if val <= 0:
            raise ValueError(f"Sigma[{j0},{j0}] must be positive.")
        return np.sqrt(val)

    def _t_perp(self, j0: int) -> np.ndarray:
        idx_other = np.array([i for i in range(self.p) if i != j0], dtype=int)
        sigma_jj = self.Sigma[j0, j0]
        return self.X[idx_other] - self.Sigma[idx_other, j0] * self.X[j0] / sigma_jj

    def truncation_bounds(self, j0: int, tol: float = 1e-10):
        j0 = int(j0)
    
        if j0 not in self.selected_set:
            raise ValueError(f"j0={j0} is not in the observed selected set.")
    
        sigma_jj = float(self.Sigma[j0, j0])
        if sigma_jj <= 0:
            raise ValueError(f"Sigma[{j0},{j0}] must be positive.")
    
        t_obs = float(self.X[j0])
    
        E = np.asarray(self.selected_set, dtype=int)
        G = np.asarray(
            [i for i in range(self.p) if i not in set(E.tolist())],
            dtype=int
        )
        beta = self.Sigma[:, j0] / sigma_jj
        beta[j0] = 1.0
        z = self.X - beta * t_obs
        z[j0] = 0.0
    
        lower = -np.inf
        upper = np.inf
        for e in E:
            for l in G:
                # X_e(t) - X_l(t) = (z_e - z_l) + (beta_e - beta_l) t >= 0
                a = float(beta[e] - beta[l])
                c = float(z[e] - z[l])
    
                if abs(a) <= tol:
                    # inequality does not depend on t
                    if c < -tol:
                        raise RuntimeError(
                            f"Observed selection event is inconsistent for constraint "
                            f"e={e}, l={l}: c={c}."
                        )
                    continue
    
                bound = -c / a
    
                if a > 0:
                    # t >= bound
                    lower = max(lower, bound)
                else:
                    # t <= bound
                    upper = min(upper, bound)
    
        # Numerical protection only.
        # The observed t should satisfy lower <= t_obs <= upper.
        eps = 1e-8 * max(1.0, abs(t_obs), self._sigma_j(j0))
    
        if np.isfinite(lower) and t_obs < lower - 100 * eps:
            raise RuntimeError(
                f"Observed value violates lower truncation bound for j0={j0}: "
                f"x={t_obs}, lower={lower}, upper={upper}."
            )
    
        if np.isfinite(upper) and t_obs > upper + 100 * eps:
            raise RuntimeError(
                f"Observed value violates upper truncation bound for j0={j0}: "
                f"x={t_obs}, lower={lower}, upper={upper}."
            )
    
        # Avoid exact boundary sticking caused by floating point error.
        if np.isfinite(lower) and abs(t_obs - lower) <= eps:
            lower = t_obs - eps
    
        if np.isfinite(upper) and abs(t_obs - upper) <= eps:
            upper = t_obs + eps
    
        if np.isfinite(lower) and np.isfinite(upper) and lower >= upper:
            raise RuntimeError(
                f"Invalid truncation interval for j0={j0}: "
                f"lower={lower}, upper={upper}, x={t_obs}."
            )
    
        return float(lower), float(upper)

    # Keep a single decorator: double-wrapping made this method non-callable.
    @staticmethod
    def _logdiffexp(log_x, log_y):
        """
        Stable calculation of
    
            log(exp(log_x) - exp(log_y))
    
        assuming log_x >= log_y.
        """
        log_x = float(log_x)
        log_y = float(log_y)
    
        if np.isneginf(log_y):
            return log_x
    
        if log_y > log_x:
            if log_y - log_x < 1e-12:
                return -np.inf
    
            raise ValueError(
                "Expected log_x >= log_y in _logdiffexp, "
                f"got log_x={log_x}, log_y={log_y}."
            )
        if log_x == log_y:
            return -np.inf
        return log_x + np.log1p(-np.exp(log_y - log_x))


    @classmethod
    def _log_normal_interval_prob(cls, lower, upper):
        lower = float(lower)
        upper = float(upper)
        if lower >= upper:
            return -np.inf
        if np.isneginf(lower) and np.isposinf(upper):
            return 0.0
        if np.isneginf(lower):
            return float(log_ndtr(upper))
        if np.isposinf(upper):
            return float(log_ndtr(-lower))
        if upper <= 0.0:
            log_upper = float(log_ndtr(upper))
            log_lower = float(log_ndtr(lower))
    
            return cls._logdiffexp(
                log_upper,
                log_lower,
            )
        if lower >= 0.0:
            log_sf_lower = float(log_ndtr(-lower))
            log_sf_upper = float(log_ndtr(-upper))
    
            return cls._logdiffexp(
                log_sf_lower,
                log_sf_upper,
            )
        probability = (
            float(norm.cdf(upper))
            - float(norm.cdf(lower))
        )
        if probability <= 0.0:
            return -np.inf
        return float(np.log(probability))
    
    
    @classmethod
    def _trunc_cdf(cls, x, mu, sigma, a, b):
        x = float(x)
        mu = float(mu)
        sigma = float(sigma)
        a = float(a)
        b = float(b)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                f"sigma must be positive and finite; got {sigma}."
            )
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        za = (
            (a - mu) / sigma
            if np.isfinite(a)
            else -np.inf
        )
        zx = (x - mu) / sigma
        zb = (
            (b - mu) / sigma
            if np.isfinite(b)
            else np.inf
        )
        log_left = cls._log_normal_interval_prob(
            za,
            zx,
        )
        log_right = cls._log_normal_interval_prob(
            zx,
            zb,
        )
        log_denom = float(
            np.logaddexp(log_left, log_right)
        )
        if not np.isfinite(log_denom):
            raise RuntimeError(
                f"x={x}, mu={mu}, sigma={sigma}, "
                f"a={a}, b={b}, "
                f"za={za}, zx={zx}, zb={zb}."
            )
        if log_left <= log_right:
            value = np.exp(log_left - log_denom)
        else:
            right_probability = np.exp(
                log_right - log_denom
            )
            value = 1.0 - right_probability
        return float(np.clip(value, 0.0, 1.0))

    def pivot(self, j0: int, mu0: float = None) -> float:
        if mu0 is None:
            mu0 = float(self.mu[j0])
        t_j0 = float(self.X[j0])
        sigma = self._sigma_j(j0)
        v_minus, v_plus = self.truncation_bounds(j0)
        return self._trunc_cdf(t_j0, mu0, sigma, v_minus, v_plus)

    def pvalue(self, j0: int, mu0: float = None, two_sided: bool = True) -> float:
        piv = self.pivot(j0, mu0)
        if two_sided:
            return float(min(1.0, 2.0 * min(piv, 1.0 - piv)))
        return float(1.0 - piv)

    def confidence_interval(
        self,
        j0: int,
        alpha: float = None,
        search_radius: float = 16.0,
        max_expand: int = 10,
    ):
        if alpha is None:
            alpha = self.alpha
        t_j0 = float(self.X[j0])
        sigma = self._sigma_j(j0)
        v_minus, v_plus = self.truncation_bounds(j0)
        def F(theta):
            return self._trunc_cdf(t_j0, theta, sigma, v_minus, v_plus)
        def f_lower(theta):
            return F(theta) - (1.0 - alpha / 2.0)
        def f_upper(theta):
            return F(theta) - (alpha / 2.0)
        center = t_j0
        left = center - search_radius * sigma
        right = center + search_radius * sigma
        def bracket_root(fun, left, right):
            fl = fun(left)
            fr = fun(right)
            n_expand = 0
            while fl * fr > 0 and n_expand < max_expand:
                width = right - left
                left -= 0.75 * width
                right += 0.75 * width
                fl = fun(left)
                fr = fun(right)
                n_expand += 1
            if fl * fr > 0:
                return None
            return left, right
        br1 = bracket_root(f_lower, left, right)
        br2 = bracket_root(f_upper, left, right)
        if br1 is None or br2 is None:
            raise RuntimeError(
                f"Could not bracket CI root for j0={j0}. "
                f"Observed x={t_j0:.6f}, trunc=({v_minus:.6f}, {v_plus})."
            )
        ci_lower = brentq(f_lower, br1[0], br1[1])
        ci_upper = brentq(f_upper, br2[0], br2[1])
        return float(ci_lower), float(ci_upper)
    
    def post_selection_inference_on_top_k(self, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        results = []
        ranked_selected = self._ranked_selected_from_subset()
        for rank, j0 in enumerate(ranked_selected, start=1):
            v_minus, v_plus = self.truncation_bounds(int(j0))
            sigma = self._sigma_j(int(j0))
            piv = self.pivot(int(j0), self.mu[int(j0)])
            pval = self.pvalue(int(j0), self.mu[int(j0)], two_sided=True)
            ci_l, ci_u = self.confidence_interval(int(j0), alpha=alpha)
            truth = float(self.mu[int(j0)])
            results.append(
                SelectiveInferenceResult(
                    rank=rank,
                    index=int(j0),
                    observed_value=float(self.X[int(j0)]),
                    observed_target_mean=truth,
                    trunc_lower=float(v_minus),
                    trunc_upper=float(v_plus),
                    sigma=float(sigma),
                    pivot_at_truth=float(piv),
                    pvalue_two_sided=float(pval),
                    ci_lower=float(ci_l),
                    ci_upper=float(ci_u),
                    covered=bool(ci_l <= truth <= ci_u),
                    length=float(ci_u - ci_l),
                )
            )
        return results

    def confidence_interval_topk(self, alpha: float = None):
        if alpha is None:
            alpha = self.alpha

        out = []
        ranked_selected = self._ranked_selected_from_subset()
        for rank, j0 in enumerate(ranked_selected, start=1):
            ci_l, ci_u = self.confidence_interval(int(j0), alpha=alpha)
            out.append({
                "rank": rank,
                "index": int(j0),
                "ci_lower": float(ci_l),
                "ci_upper": float(ci_u),
                "truth": float(self.mu[int(j0)]),
                "covered": bool(ci_l <= self.mu[int(j0)] <= ci_u),
                "length": float(ci_u - ci_l),
            })
        return out

    def plot_density(self, j0: int, mu0: float = None, num_grid: int = 500):
        if mu0 is None:
            mu0 = float(self.mu[j0])

        sigma = self._sigma_j(j0)
        v_minus, v_plus = self.truncation_bounds(j0)
        t_j0 = float(self.X[j0])

        left = v_minus if np.isfinite(v_minus) else (mu0 - 4 * sigma)
        right = v_plus if np.isfinite(v_plus) else (max(t_j0, mu0) + 4 * sigma)

        xs = np.linspace(left, right, num_grid)
        za = (v_minus - mu0) / sigma if np.isfinite(v_minus) else -np.inf
        zb = (v_plus - mu0) / sigma if np.isfinite(v_plus) else np.inf
        denom = norm.cdf(zb) - norm.cdf(za)

        dens = norm.pdf((xs - mu0) / sigma) / sigma / denom
        dens[(xs < v_minus) | (xs > v_plus)] = 0.0

        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, dens, lw=2)
        plt.axvline(t_j0, ls="--", lw=1.5, label=f"observed T[{j0}]")
        plt.axvline(v_minus, ls=":", lw=1.5, label="trunc lower")
        if np.isfinite(v_plus):
            plt.axvline(v_plus, ls=":", lw=1.5, label="trunc upper")
        plt.xlabel("t")
        plt.ylabel("density")
        plt.title(f"Selective truncated-normal density for coordinate {j0}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_density_topk(self, num_grid: int = 500):
        ranked_selected = self._ranked_selected_from_subset()
        for rank, j0 in enumerate(ranked_selected, start=1):
            print(f"rank={rank}, index={j0}")
            self.plot_density(int(j0), num_grid=num_grid)

    def plot_pivot_curve(self, j0: int, theta_min: float = None, theta_max: float = None, num_grid: int = 300):
        t_j0 = float(self.X[j0])
        sigma = self._sigma_j(j0)

        if theta_min is None:
            theta_min = t_j0 - 5 * sigma
        if theta_max is None:
            theta_max = t_j0 + 5 * sigma

        thetas = np.linspace(theta_min, theta_max, num_grid)
        vals = np.array([self.pivot(j0, mu0=th) for th in thetas])

        plt.figure(figsize=(7, 4.5))
        plt.plot(thetas, vals, lw=2)
        plt.axhline(self.alpha / 2, ls="--", lw=1.5, label=r"$\alpha/2$")
        plt.axhline(1 - self.alpha / 2, ls="--", lw=1.5, label=r"$1-\alpha/2$")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$F_{j_0}(\theta)$")
        plt.title(f"Pivot curve for coordinate {j0}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary(self, alpha: float = None):
        results = self.post_selection_inference_on_top_k(alpha=alpha)
        print(f"Observed selected set: {self.selected_set.tolist()}")
        print(f"Observed (k+1)-th largest value: {self.t_kplus1:.6f}")
        print("-" * 90)
        for r in results:
            print(f"rank               : {r.rank}")
            print(f"index              : {r.index}")
            print(f"observed value     : {r.observed_value:.6f}")
            print(f"truth              : {r.observed_target_mean:.6f}")
            print(f"truncation         : [{r.trunc_lower:.6f}, {r.trunc_upper}]")
            print(f"sigma              : {r.sigma:.6f}")
            print(f"pivot at truth     : {r.pivot_at_truth:.6f}")
            print(f"two-sided p-value  : {r.pvalue_two_sided:.6f}")
            print(f"CI                 : [{r.ci_lower:.6f}, {r.ci_upper:.6f}]")
            print(f"covered            : {r.covered}")
            print(f"length             : {r.length:.6f}")
            print("-" * 90)
