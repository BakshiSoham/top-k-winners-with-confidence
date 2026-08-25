from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
@dataclass
class NaiveInferenceResultUnordered:
    index: int
    observed_value: float
    observed_target_mean: float
    sigma: float
    pivot_at_truth: float
    pvalue_two_sided: float
    ci_lower: float
    ci_upper: float
    covered: bool
    length: float


class NaiveSubsetInference:
    def __init__(
        self,
        X: np.ndarray,
        k: int,
        H0_mu: np.ndarray,
        Sigma: np.ndarray,
        utility_fn=None,
        alpha: float = 0.05,
        selected_subset=None,
        subset_order: str = "sorted_index",   # "sorted_index", "input", "utility"
    ):
        self.X = np.asarray(X, dtype=float)
        self.k = int(k)
        self.mu = np.asarray(H0_mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.alpha = float(alpha)
        self.subset_order = subset_order

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
        if not (1 <= self.k <= len(self.X)):
            raise ValueError("Need 1 <= k <= p.")

        self.p = len(self.X)

        self.utility = np.asarray(self.utility_fn(self.X), dtype=float)
        if self.utility.shape != self.X.shape:
            raise ValueError("utility_fn must return a vector of same shape as X.")

        if selected_subset is None:
            # observed top-k subset
            selected_subset = self._top_k_indices(self.utility, self.k)
            self._input_subset_was_given = False
        else:
            selected_subset = np.asarray(selected_subset, dtype=int)
            self._input_subset_was_given = True

            if selected_subset.ndim != 1:
                raise ValueError("selected_subset must be a 1D array-like of indices.")
            if len(selected_subset) != self.k:
                raise ValueError(f"selected_subset must have length k={self.k}.")
            if len(np.unique(selected_subset)) != self.k:
                raise ValueError("selected_subset contains duplicated indices.")
            if np.any(selected_subset < 0) or np.any(selected_subset >= self.p):
                raise ValueError("selected_subset contains invalid indices.")

        self.selected_set = np.asarray(selected_subset, dtype=int).copy()
        self.not_selected_set = np.array(
            [j for j in range(self.p) if j not in self.selected_set],
            dtype=int
        )

    @staticmethod
    def _top_k_indices(x: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(x)[::-1][:k]

    def _ordered_subset_for_output(self) -> np.ndarray:
        """
        This only controls display/output order, not inferential meaning.
        No 'rank' interpretation should be attached.
        """
        if self.subset_order == "input":
            return self.selected_set.copy()

        if self.subset_order == "utility":
            return self.selected_set[np.argsort(self.utility[self.selected_set])[::-1]]

        if self.subset_order == "sorted_index":
            return np.sort(self.selected_set)

        raise ValueError("subset_order must be one of: 'sorted_index', 'input', 'utility'.")

    def _sigma_j(self, j0: int) -> float:
        val = float(self.Sigma[j0, j0])
        if val < 0:
            raise ValueError(f"Sigma[{j0},{j0}] must be nonnegative.")
        return np.sqrt(val)

    def pivot(self, j0: int, mu0: float = None) -> float:
        if mu0 is None:
            mu0 = float(self.mu[j0])

        sigma = self._sigma_j(j0)
        xj = float(self.X[j0])

        if sigma == 0:
            return float(1.0 if xj >= mu0 else 0.0)

        z = (xj - mu0) / sigma
        return float(norm.cdf(z))

    def pvalue(self, j0: int, mu0: float = None, two_sided: bool = True) -> float:
        piv = self.pivot(j0, mu0)
        if two_sided:
            return float(min(1.0, 2.0 * min(piv, 1.0 - piv)))
        return float(1.0 - piv)

    def confidence_interval(self, j0: int, alpha: float = None):
        if alpha is None:
            alpha = self.alpha

        xj = float(self.X[j0])
        sigma = self._sigma_j(j0)
        zcrit = float(norm.ppf(1.0 - alpha / 2.0))

        ci_lower = xj - zcrit * sigma
        ci_upper = xj + zcrit * sigma
        return float(ci_lower), float(ci_upper)

    def inference_on_subset(self, alpha: float = None):
        if alpha is None:
            alpha = self.alpha

        results = []
        subset_for_output = self._ordered_subset_for_output()

        for j0 in subset_for_output:
            j0 = int(j0)
            sigma = self._sigma_j(j0)
            piv = self.pivot(j0, self.mu[j0])
            pval = self.pvalue(j0, self.mu[j0], two_sided=True)
            ci_l, ci_u = self.confidence_interval(j0, alpha=alpha)
            truth = float(self.mu[j0])

            results.append(
                NaiveInferenceResultUnordered(
                    index=j0,
                    observed_value=float(self.X[j0]),
                    observed_target_mean=truth,
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

    def confidence_interval_subset(self, alpha: float = None):
        if alpha is None:
            alpha = self.alpha

        out = []
        subset_for_output = self._ordered_subset_for_output()

        for j0 in subset_for_output:
            j0 = int(j0)
            ci_l, ci_u = self.confidence_interval(j0, alpha=alpha)
            out.append({
                "index": j0,
                "ci_lower": float(ci_l),
                "ci_upper": float(ci_u),
                "truth": float(self.mu[j0]),
                "covered": bool(ci_l <= self.mu[j0] <= ci_u),
                "length": float(ci_u - ci_l),
            })
        return out

    def summary(self, alpha: float = None):
        results = self.inference_on_subset(alpha=alpha)
        print(f"Selected subset (unordered): {sorted(self.selected_set.tolist())}")
        print("-" * 90)
        for r in results:
            print(f"index              : {r.index}")
            print(f"observed value     : {r.observed_value:.6f}")
            print(f"truth              : {r.observed_target_mean:.6f}")
            print(f"sigma              : {r.sigma:.6f}")
            print(f"pivot at truth     : {r.pivot_at_truth:.6f}")
            print(f"two-sided p-value  : {r.pvalue_two_sided:.6f}")
            print(f"CI                 : [{r.ci_lower:.6f}, {r.ci_upper:.6f}]")
            print(f"covered            : {r.covered}")
            print(f"length             : {r.length:.6f}")
            print("-" * 90)

    def plot_density(self, j0: int, mu0: float = None, num_grid: int = 500):
        if mu0 is None:
            mu0 = float(self.mu[j0])

        sigma = self._sigma_j(j0)
        xj = float(self.X[j0])

        if sigma == 0:
            raise ValueError(f"Sigma[{j0},{j0}] is zero, cannot plot Gaussian density.")

        xs = np.linspace(mu0 - 4 * sigma, mu0 + 4 * sigma, num_grid)
        dens = norm.pdf((xs - mu0) / sigma) / sigma

        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, dens, lw=2)
        plt.axvline(xj, ls="--", lw=1.5, label=f"observed X[{j0}]")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title(f"Naive Gaussian density for coordinate {j0}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_density_subset(self, num_grid: int = 500):
        subset_for_output = self._ordered_subset_for_output()
        for j0 in subset_for_output:
            print(f"index={int(j0)}")
            self.plot_density(int(j0), num_grid=num_grid)
