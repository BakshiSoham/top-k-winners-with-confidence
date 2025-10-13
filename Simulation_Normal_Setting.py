#np.random.seed(42)
# ===== Initialization =====
H0_mu = np.array([1,1,2,1,1])
H0_Sigma = np.eye(5)
X = np.random.multivariate_normal(mean=H0_mu, cov=H0_Sigma, size=1)
model = TopKSelectionModel(X, k=1, true_mu=H0_mu, true_Sigma=H0_Sigma, epsilon=20)

S, _ = model.randomized_selected_top_k()
_, v = model.get_projection(S)
grid_points, density_vals = model.conditional_density_grid(v, S, grid_size=1500)
F, (ug, Fg) = cdf_from_density(grid_points, density_vals)

# ===== Conditional Density under different temperature parameters =====
def conditional_density_grid_eps(model, v, S, grid_size=400, epsilon=1.0, span=4.0):
    mu_s = model.true_mu[list(S)][0]
    umin, umax = mu_s - span, mu_s + span
    grid_points = np.linspace(umin, umax, grid_size)
    dens = [model.conditional_density(np.array([p]), v, S, epsilon=epsilon) for p in grid_points]
    dens = np.asarray(dens)
    dens /= np.trapz(dens, grid_points)
    return grid_points, dens

eps_list = [0.2, 1, 5, 20, 30]
colors = ["C0","C1","C2","C3","C4"]
densities = [conditional_density_grid_eps(model, v, S, grid_size=600, epsilon=eps, span=4.0)
             for eps in eps_list]

# ===== Layout =====
fig = plt.figure(figsize=(11,6))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2.5], height_ratios=[1, 1],
                       wspace=0.15, hspace=0.05)

# --- Figures ---
ax_pdf = plt.subplot(gs[0,0])
markerline, stemlines, baseline = ax_pdf.stem(grid_points, density_vals, linefmt='C0-', markerfmt='C0o', basefmt=" ")
plt.setp(stemlines, linewidth=0.8)
ax_pdf.axvline(model.true_mu[list(S)][0], color='gray', ls=':', lw=1)
ax_pdf.set_title("PDF: Conditional density (stem)")
ax_pdf.set_xlabel('u'); ax_pdf.set_ylabel('Density') 
ax_pdf.set_ylim(0, 1.2)
ax_pdf.set_aspect(2.5)     

ax_cdf = plt.subplot(gs[1,0])
markerline, stemlines, baseline = ax_cdf.stem(ug, Fg, linefmt='C1-', markerfmt='C1o', basefmt=" ")
plt.setp(stemlines, linewidth=0.8)
ax_cdf.axvline(model.true_mu[list(S)][0], color='gray', ls=':', lw=1)
ax_cdf.set_title("CDF: From conditional density (stem)")
ax_cdf.set_xlabel('u'); ax_cdf.set_ylabel('CDF')
ax_cdf.set_ylim(0, 1.2)
ax_cdf.set_aspect(2.5)

ax_eps = plt.subplot(gs[:,1])
for (gp, dv), eps, c in zip(densities, eps_list, colors):
    ax_eps.plot(gp, dv, color=c, lw=1.8, label=fr'$\epsilon={eps}$')
ax_eps.axvline(model.true_mu[list(S)][0], color='gray', ls=':', lw=1)
ax_eps.set_title(r'Conditional density vs $\epsilon$')
ax_eps.set_xlabel('u'); ax_eps.set_ylabel('Density')
ax_eps.legend(frameon=False)
ax_eps.set_ylim(bottom=0)
ax_eps.set_aspect(5.5)

plt.tight_layout()
plt.show()


def simulate_topk_multi_H0(
    H0_mu_list, H0_Sigma,
    epsilon_grid, 
    k=1, B_reject=300, N_repeat=20,
    grid_size=1000, save_prefix="simulation"
):
    """
    Run Top-K PSI simulations for multiple H0_mu configurations across a grid of epsilon values.
    Draw combined scree plot for selection probabilities, and save (not show) Type-I error and ECDF plots.

    Parameters
    ----------
    H0_mu_list : list[np.ndarray]
        List of mean vectors under H0.
    H0_Sigma : np.ndarray
        Covariance matrix under H0 (shared for all settings).
    epsilon_grid : np.ndarray
        Array of epsilon values to test.
    k : int, default=1
        Number of top-K selections.
    B_reject : int, default=300
        Monte Carlo repetitions per run.
    N_repeat : int, default=20
        Number of repetitions per epsilon (for boxplot variability).
    grid_size : int, default=1000
        Grid size for conditional density.
    save_prefix : str, default="simulation"
        Prefix for saving output plots.

    Returns
    -------
    all_results : dict
        Dictionary mapping each H0_mu index to its result dict.
    """

    np.random.seed(42)
    sns.set(style="whitegrid", font_scale=1.2, rc={"axes.facecolor": "white", "figure.dpi": 150})
    all_results = {}

    # Prepare color palette for multiple curves
    colors = sns.color_palette("tab10", n_colors=len(H0_mu_list))

    plt.figure(figsize=(8, 5))
    plt.title("True Top-K Winner Selection Probability (Scree Plot)", fontsize=13, weight="bold")
    plt.xlabel("Epsilon")
    plt.ylabel("Selection Probability of True Winner")
    plt.grid(True, linestyle='--', alpha=0.6)

    # =========================================================
    # Loop over each H0_mu
    # =========================================================
    for idx, H0_mu in enumerate(H0_mu_list):
        signal_strength = np.sort(H0_mu)[::-1][0] - np.sort(H0_mu)[::-1][1]
        label_str = f"Δμ={signal_strength:.1f}"
        print(f"\n>>> Running for H0_mu #{idx+1} with signal strength = {signal_strength:.1f}")

        X = np.random.multivariate_normal(mean=H0_mu, cov=H0_Sigma, size=1)
        model = TopKSelectionModel(X=X, k=k, true_mu=H0_mu, true_Sigma=H0_Sigma, grid_size=grid_size)

        # === 1, True Top-K selection probability curve
        eps_list, true_probs = model.true_winner_selection_probability(epsilon_grid, 500)
        plt.plot(eps_list, true_probs, marker='o', lw=1.8, label=label_str, color=colors[idx])

        # === 2, Type-I Error simulation
        records = []
        all_pivots = {}
        for eps in tqdm(epsilon_grid, desc=f"Simulating Type-I error (μ index {idx+1})"):
            reject_vals = []
            pivot_samples = []
            for rep in range(N_repeat):
                model = TopKSelectionModel(X=X, k=k, true_mu=H0_mu, true_Sigma=H0_Sigma, 
                                           epsilon=eps, grid_size=grid_size)
                res = model.simulation_test(mu=H0_mu, Sigma=H0_Sigma, 
                                            B=B_reject, alpha=0.05, tail="two_tails")
                reject_vals.append(res["reject_rate_two_tail"])
                pivot_samples.extend(res["pivot"])

            for val in reject_vals:
                records.append({"epsilon": float(eps), "typeI_error": val})
            all_pivots[float(eps)] = np.array(pivot_samples)

        df = pd.DataFrame(records)

        # === Saved Type I Error boxplot
        fig1, ax1 = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df, x="epsilon", y="typeI_error", color=colors[idx], width=0.6, fliersize=2, linewidth=1.1, ax=ax1)
        ax1.axhline(0.05, color="red", linestyle="--", lw=1.2, label="Nominal α=0.05")
        ax1.set_xlabel("Epsilon", fontsize=12)
        ax1.set_ylabel("Type I Error Rate", fontsize=12)
        ax1.set_title(f"Type-I Error Distribution (Δμ={signal_strength:.1f})", fontsize=13, weight="bold")
        ax1.legend(loc="upper right", frameon=True)
        fig1.tight_layout()
        fig1.savefig(f"{save_prefix}_type1_error_boxplot_signal{signal_strength:.1f}.png", dpi=300)
        plt.close(fig1)

        # === Saved ECDF of pivots
        fig2, ax2 = plt.subplots(figsize=(7,6))
        for eps, pivots in all_pivots.items():
            pivots = np.sort(pivots)
            n = len(pivots)
            y = np.arange(1, n+1) / n
            ax2.step(pivots, y, where="post", label=f"ε={eps:.1f}")
        ax2.plot([0,1],[0,1],"--",color="black",label="y=x (Uniform CDF)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("ECDF(t)")
        ax2.set_title(f"ECDF of Pivots (Δμ={signal_strength:.1f})", fontsize=13, weight="bold")
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle='--', alpha=0.6)
        fig2.tight_layout()
        fig2.savefig(f"{save_prefix}_ecdf_pivots_signal{signal_strength:.1f}.png", dpi=300)
        plt.close(fig2)

        
        all_results[signal_strength] = {
            "mu": H0_mu,
            "epsilon_grid": epsilon_grid,
            "true_probs": true_probs,
            "typeI_error_df": df,
            "all_pivots": all_pivots
        }
    plt.legend(title="Signal Strength Δμ", frameon=True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_selection_probability_scree.png", dpi=300)
    plt.show()

    return all_results

H0_mu_list = [
    np.array([0, 1, 0, 0, 0]),
    np.array([0, 2, 0, 0, 0]),
    np.array([0, 3, 0, 0, 0]),
    np.array([0, 4, 0, 0, 0]),
    np.array([0, 5, 0, 0, 0]),
    np.array([0, 9, 0, 0, 0])
]

results = simulate_topk_multi_H0(
    H0_mu_list=H0_mu_list,
    H0_Sigma=np.eye(5),
    epsilon_grid=np.array([0,0.1,0.5,1.,3., 5.,10]),
    k=1,
    B_reject=20,
    N_repeat=5,
    grid_size=150,
    save_prefix="signal_strength_study"
)


def combine_plots_to_grid(image_prefix, pattern, output_name, ncols=3, nrows=2, figsize=(12, 8)):
    """
    Combine multiple saved PNGs into a grid layout.

    Parameters
    ----------
    image_prefix : str
        File prefix (e.g., 'signal_strength_study').
    pattern : str
        Pattern part of filename to match (e.g., 'ecdf_pivots_signal' or 'type1_error_boxplot_signal').
    output_name : str
        Output filename for the combined figure.
    ncols, nrows : int
        Grid layout.
    figsize : tuple
        Figure size (in inches).
    """

    all_files = sorted([f for f in os.listdir(".") if f.startswith(image_prefix) and pattern in f])
    if len(all_files) == 0:
        print(f"[Warning] No files found matching pattern '{pattern}' with prefix '{image_prefix}'")
        return

    print(f" Found {len(all_files)} images to combine for pattern '{pattern}'")

    images = [Image.open(f) for f in all_files]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, img, fname in zip(axes, images, all_files):
        ax.imshow(img)
        ax.set_title(fname.split(pattern)[-1].replace(".png", ""), fontsize=11, pad=4)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.suptitle(f"{pattern.replace('_', ' ').title()} (Signal Strength Comparison)", fontsize=15, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_name, dpi=300)
    plt.show()

    print(f" Combined figure saved to '{output_name}'")

def plot_combined_type1_boxplot(all_results, save_path="combined_type1_error_multi_signal.png"):
    """
    Combine Type-I Error distributions for multiple signal strengths into one comparative boxplot.

    Parameters
    ----------
    all_results : dict
        Output dictionary from simulate_topk_multi_H0()
    save_path : str
        File name to save the combined figure
    """

    dfs = []
    for signal_strength, res in all_results.items():
        df = res["typeI_error_df"].copy()
        df["signal_strength"] = signal_strength
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    sns.set(style="whitegrid", font_scale=1.4, rc={"axes.facecolor": "white", "figure.dpi": 150})
    palette = sns.color_palette("Set2", n_colors=len(all_results))

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_all,
        x="epsilon",
        y="typeI_error",
        hue="signal_strength",
        palette=palette,
        width=0.7,
        fliersize=2.5,
        linewidth=1.1
    )

    plt.axhline(0.05, color="red", linestyle="--", lw=1.2, label="Nominal α=0.05")

    plt.xlabel("Epsilon (Randomization Strength)", fontsize=13)
    plt.ylabel("Type I Error Rate", fontsize=13)
    plt.title("Type-I Error Distribution across Epsilon and Signal Strengths", fontsize=15, weight="bold")
    plt.legend(title="Signal Strength Δμ", frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f" Combined multi-signal boxplot saved to {save_path}")


combine_plots_to_grid(
    image_prefix="signal_strength_study",
    pattern="type1_error_boxplot_signal",
    output_name="combined_type1_error_2x3.png",
    ncols=3, nrows=2
)

combine_plots_to_grid(
    image_prefix="signal_strength_study",
    pattern="ecdf_pivots_signal",
    output_name="combined_ecdf_2x3.png",
    ncols=3, nrows=2
)

plot_combined_type1_boxplot(results, save_path="combined_type1_error_multi_signal.png")


