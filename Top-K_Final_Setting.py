from scipy.stats import norm
from scipy.integrate import quad
from itertools import combinations
from scipy.stats import multivariate_normal
import numpy as np
from typing import Callable, List
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from itertools import combinations
from scipy.stats import norm, uniform
from tqdm import tqdm
from collections import Counter
import seaborn as sns
from scipy.stats import kstest
from tqdm import tqdm
import sys
import time
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d,PchipInterpolator
from matplotlib import gridspec
from PIL import Image
import os


def cdf_from_density(grid_u, density):

    grid_u = np.asarray(grid_u, dtype=float)
    density = np.asarray(density, dtype=float)
    
    dx = np.diff(grid_u)
    area_segments = 0.5 * (density[:-1] + density[1:]) * dx
    cum_area = np.concatenate([[0.0], np.cumsum(area_segments)])
    
    total_area = cum_area[-1]
    if total_area <= 0:
        raise ValueError("Integral of Density is Non-positive")
    F_vals = cum_area / total_area
    
    cdf_spline = interp1d(grid_u, F_vals, kind="cubic",fill_value="extrapolate",bounds_error=False)
    
    def F(x):
        return np.clip(cdf_spline(x), 0.0, 1.0)
    
    return F, (grid_u, F_vals)

class TopKSelectionModel:
    def __init__(self, X: np.ndarray, k: int, true_mu,true_Sigma,utility_fn=None,epsilon=1.,grid_size=1000):
        self.X = X
        self.n, self.M = X.shape
        self.k = k
        self.utility_fn = utility_fn or self.default_utility
        self.utilities = self.compute_utilities()
        self.selected_indices = self.select_top_k()
        self.epsilon=epsilon
        self.true_mu=true_mu
        self.true_Sigma=true_Sigma 
        self.grid_size=grid_size
        
    def default_utility(self,X: np.ndarray) -> float:
        col_means = X.mean(axis=0)   
        return col_means.sum()

    def compute_utilities(self):
        return np.array([self.utility_fn(self.X[:, m]) for m in range(self.M)])

    def select_top_k(self, X=None):
        if X is None:
            X = self.X
        utilities = np.array([self.utility_fn(X[:, m]) for m in range(X.shape[1])])
        return np.argsort(utilities)[-self.k:][::-1]

    def top_k_additive_utility(self,S,X=None):
        if X is None:
            X=self.X
        return self.utility_fn(X[:,list(S)]).sum()

    
            
    def get_projection(self, index_set,X=None, mu=None, Sigma=None):
        if X is None:
            X=self.X
        if Sigma is None:
            Sigma = self.true_Sigma
        if mu is None:
            mu =self.true_mu
        k=self.k
        E_S = np.zeros((self.M, k))
        for i, idx in enumerate(index_set):
            E_S[idx, i] = 1.0
        Sigma_S = E_S.T @ Sigma @ E_S
        Sigma_inv_proj = Sigma @ E_S @ np.linalg.inv(Sigma_S) 
        X_S=E_S.T @ X.T 
        proj=Sigma_inv_proj @ X_S
        orth = X.T - proj 
        return proj, orth

    def randomized_selected_top_k(self,X=None, k=None,epsilon=None):
        if epsilon is None:
            epsilon=self.epsilon
        else:
            epsilon=epsilon
        if X is None:
            X=self.X
        if k is None:
            k=self.k
        n, M = X.shape
        all_subsets = list(combinations(range(M), k))
        weights = []
        for S in all_subsets:        
            X_S = X[:,list(S)]
            u_S = self.utility_fn(X_S)
            weight = np.exp(epsilon * u_S)
            weights.append(weight)
        weights = np.array(weights)
        probs = weights / weights.sum()
        idx = np.random.choice(len(all_subsets), p=probs)
        selected_subset = all_subsets[idx]
        prob_dict = {tuple(sorted(S)): p for S, p in zip(all_subsets, probs)}       
        return selected_subset, prob_dict

    def conditional_density(self, u,v,S,epsilon=None,true_mu=None,true_sigma=None):
        if np.isscalar(u):
            u = np.array([u])
        if epsilon is None:
            epsilon=self.epsilon
            
        k = self.k
        mu=self.true_mu
        Sigma=self.true_Sigma
        M=self.M
        S_sorted = tuple(sorted(S))
        n = self.n
        E_S_fixed = np.zeros((self.M, k))
        for i, idx in enumerate(S):
            E_S_fixed[idx, i] = 1.
        Sigma_S = E_S_fixed.T @ Sigma @ E_S_fixed
        mu_S = mu[list(S)]
        X=Sigma @ E_S_fixed @ np.linalg.inv(Sigma_S) @ u +v.T     
        weights = []
        log_weights=[]
        all_subsets = list(combinations(range(M), k))
        for SS in all_subsets:        
            u_S = self.top_k_additive_utility(SS,X)
            weight = np.exp(epsilon * u_S)
            weights.append(weight)
        weights = np.array(weights)
        p_S = weights / weights.sum()
        phi_val = multivariate_normal.pdf(u, mean=mu_S, cov=Sigma_S)
        idx = all_subsets.index(S_sorted)
        density_val = phi_val * p_S[idx]
        return  float(density_val)

    def conditional_density_grid(self, v,S,grid_size=None,true_mu=None,true_sigma=None):
        epsilon=self.epsilon
        k = self.k
        mu=self.true_mu
        Sigma=self.true_Sigma
        M=self.M
        if grid_size is None:
            grid_size=self.grid_size
        n = self.n
        grid_points = np.linspace(-20, 20, grid_size)
        all_subsets = list(combinations(range(M), k))
        density_vals = []
        for point in grid_points:
            u = np.array([point/k]*k) 
            density_vals.append(self.conditional_density(u,v,S,epsilon))
        density_vals = np.array(density_vals)
        Z_trapz = np.trapz(density_vals, grid_points)
        density_vals = density_vals / Z_trapz

        cutoff_value=10e-4

        mask = density_vals > cutoff_value
        grid_points = grid_points[mask]
        density_vals = density_vals[mask]

        if len(density_vals) > 1:
            Z_final = np.trapz(density_vals, grid_points)
            density_vals /= Z_final
        
        return grid_points, density_vals 
        
    def true_winner_selection_probability(self, epsilon_list, B=1000):
        true_subset = tuple(sorted(self.select_top_k()))
        selection_probs = []
        for eps in epsilon_list:
            count = 0
            for _ in range(B):
                selected_subset, _ = self.randomized_selected_top_k(epsilon=eps)
                if tuple(sorted(selected_subset)) == true_subset:
                    count += 1
            selection_probs.append(count / B)
        return epsilon_list, selection_probs  

    def simulation_test(self,mu=None,Sigma=None,B=1000,alpha=0.05,k=None,epsilon=None,grid_size=None,tail="two_tails"):
        if mu is None:
            mu=self.true_mu
        if Sigma is None:
            Sigma=self.true_Sigma
        if k is None:
            k=self.k
        if epsilon is None:
            epsilon=self.epsilon
        if grid_size is None:
            grid_size=self.grid_size
            
        Ps=[]
        pivots=[]
        us=[]
        Ss=[]

        for i in range(B):
            X=np.random.multivariate_normal(mean=mu,cov=Sigma,size=1)
            S,_=self.randomized_selected_top_k(X=X, k=k, epsilon=epsilon)
            S = list(S)
            u_star=self.top_k_additive_utility(S,X)
            _, v = self.get_projection(S,X,mu=mu, Sigma=Sigma)


            
            grid_points, density_vals =self.conditional_density_grid(v,S,grid_size=grid_size,true_mu=mu,true_sigma=mu)
            F, (ug, Fg) = cdf_from_density(grid_points, density_vals)
            pivot=F(u_star)
            pivots.append(pivot)
            us.append(u_star)
            Ss.append(S)
            percent = int((i + 1) / B * 100)
            bar = '#' * (percent // 2) + '-' * (50 - percent // 2)
            sys.stdout.write(
                f'\rProgress: |{bar}| {percent}% done (epsilon={epsilon:.2f})'
            )
            sys.stdout.flush()

        pivots=np.array(pivots)
        p_two =  2.0 * np.minimum(pivots, 1.0 - pivots)
        p_one = pivots
        reject_rate_two = np.mean((pivots >= 0.975) | (pivots <= 0.025))
        reject_rate_one = np.mean(p_one <= alpha)
        return {"pivot": pivots, "p_right": p_one, "p_two": p_two, "S": Ss, "u_star": us,"reject_rate_one_tail":reject_rate_one,"reject_rate_two_tail":reject_rate_two}
    
    def plot_density_stem(self, grid_points, density_vals, title="Conditional density (stem plot)"):
        plt.figure(figsize=(8, 5))
        plt.stem(grid_points, density_vals, basefmt=" ")
        plt.xlabel("u")
        plt.ylabel("Density")
        plt.title(title)
        plt.tight_layout()
        plt.show()        

def plot_ecdf_vs_uniform(pivots, alpha=0.05, title="ECDF of pivots vs Uniform(0,1)"):
    pivots = np.sort(np.asarray(pivots))
    n = pivots.size
    y = np.arange(1, n+1) / n
    t = np.linspace(0, 1, 501)

    eps = np.sqrt(np.log(2/alpha) / (2*n))
    upper = np.clip(t + eps, 0, 1)
    lower = np.clip(t - eps, 0, 1)

    plt.figure()
    plt.step(pivots, y, where="post", label="ECDF of pivots")
    plt.plot([0,1],[0,1],"--",label="y = x (Uniform CDF)")
    plt.plot(t, upper, ":", label=f"DKW band (1-{alpha:.2f})")
    plt.plot(t, lower, ":")

    stat, pval = kstest(pivots, 'uniform')
    plt.title(title + f"\nKS stat={stat:.4f}, p={pval:.3g}")
    plt.xlabel("t"); plt.ylabel("F_n(t)")
    plt.legend(); plt.tight_layout(); plt.show()
    print(f"KS stat={stat:.4f}, p-value={pval:.3g}")



