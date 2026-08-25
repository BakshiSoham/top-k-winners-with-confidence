def infer_method_order(
    compare_out,
    preferred_order=("Standard", "Randomized PSI", "Polyhedral PSI", "Data Splitting", "Zoom Correction")
):
    df = compare_out["all_ci_df"].copy()

    if df.empty or "method" not in df.columns:
        return []

    existing_methods = df["method"].dropna().unique().tolist()

    ordered = [m for m in preferred_order if m in existing_methods]
    extra = [m for m in existing_methods if m not in ordered]

    return ordered + extra


def get_rep_level_coverage_df(compare_out, complete_only=False):

    if "all_rep_df" in compare_out and compare_out["all_rep_df"] is not None:
        rep_df = compare_out["all_rep_df"].copy()

    else:
        if "all_ci_df" not in compare_out:
            raise ValueError("compare_out must contain 'all_ci_df' or 'all_rep_df'.")

        df = compare_out["all_ci_df"].copy()

        if df.empty:
            return pd.DataFrame(columns=[
                "method", "epsilon", "rep",
                "coverage_rate", "avg_length", "n_intervals", "is_complete"
            ])

        required_cols = ["method", "epsilon", "rep", "covered", "length", "idx"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"all_ci_df is missing columns: {missing_cols}")

        # This is the key point:
        # group by rep first, so each simulation contributes only one avg_length.
        rep_df = (
            df.groupby(["method", "epsilon", "rep"], as_index=False)
              .agg(
                  coverage_rate=("covered", "mean"),
                  avg_length=("length", "mean"),
                  n_intervals=("idx", "size"),
              )
              .sort_values(["epsilon", "method", "rep"])
              .reset_index(drop=True)
        )

        k = compare_out.get("k", None)
        if k is not None:
            rep_df["is_complete"] = rep_df["n_intervals"].eq(int(k))
        else:
            rep_df["is_complete"] = True

    required_rep_cols = ["method", "epsilon", "rep", "coverage_rate", "avg_length"]
    missing_rep_cols = [col for col in required_rep_cols if col not in rep_df.columns]
    if len(missing_rep_cols) > 0:
        raise ValueError(f"rep-level dataframe is missing columns: {missing_rep_cols}")

    if complete_only and "is_complete" in rep_df.columns:
        rep_df = rep_df[rep_df["is_complete"]].copy()

    return rep_df.reset_index(drop=True)



def get_index_level_coverage_df(compare_out):
    df = compare_out["all_ci_df"].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "method", "epsilon", "idx",
            "coverage_rate", "n_selected", "avg_length"
        ])

    required_cols = ["method", "epsilon", "idx", "covered", "length"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"all_ci_df is missing columns: {missing_cols}")

    out = (
        df.groupby(["method", "epsilon", "idx"], as_index=False)
          .agg(
              coverage_rate=("covered", "mean"),
              n_selected=("idx", "size"),
              avg_length=("length", "mean"),
          )
          .sort_values(["epsilon", "method", "idx"])
          .reset_index(drop=True)
    )
    return out


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D





def plot_compare_length_boxplots_gaussian(
    compare_out,
    *,
    method_order=None,
    method_labels=None,
    colors=None,
    figsize=(10, 5),
    ylim=None,
    box_width=0.22,
    show_points=False,
    complete_only=False,
    errorbar_type="se",   # "sd" or "se"

    # New options for multi_out
    x_axis="setting_label",      # "setting_label", "setting_id", or "signal_strength"
    hide_xtick_labels=False,
    epsilon_to_plot=None,
    rotate_xticks=0,
):

    len_df = get_rep_level_coverage_df(
        compare_out,
        complete_only=complete_only,
    )

    if len_df.empty:
        print("No length data available to plot.")
        return len_df

    # --------------------------------------------------
    # Optional epsilon filtering
    # --------------------------------------------------
    if "epsilon" in len_df.columns:
        eps_values = sorted(len_df["epsilon"].dropna().astype(float).unique())

        if epsilon_to_plot is not None:
            len_df = len_df[
                np.isclose(len_df["epsilon"].astype(float), float(epsilon_to_plot))
            ].copy()
        else:
            if len(eps_values) > 1:
                raise ValueError(
                    "Multiple epsilon values found. "
                    "Since the x-axis is now parameter setting, please specify "
                    "epsilon_to_plot, e.g. epsilon_to_plot=10."
                )

    if method_order is None:
        method_order = infer_method_order(compare_out)

    method_order = list(method_order)

    if len(method_order) == 0 or len_df.empty:
        print("No length data available to plot.")
        return len_df

    required_cols = ["method", "rep", "avg_length", x_axis]
    missing_cols = [col for col in required_cols if col not in len_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            f"rep-level dataframe is missing columns: {missing_cols}. "
            f"For multi_out, make sure all_rep_df contains '{x_axis}'."
        )

    if method_labels is None:
        method_labels = {
            "Standard": "Standard",
            "Randomized PSI": "Randomized PSI",
            "Polyhedral PSI": "Polyhedral PSI",
            "Data Splitting": "Data Splitting",
            "Zoom Correction": "Zoom Correction",
        }

    if colors is None:
        colors = {
            "Standard": "#7A7A7A",
            "Randomized PSI": "#1F77B4",
            "Polyhedral PSI": "#9467BD",
            "Data Splitting": "#2CA02C",
            "Zoom Correction": "#E6550D",
        }

    # --------------------------------------------------
    # x-axis values: parameter settings, not epsilon
    # --------------------------------------------------
    if x_axis == "setting_label" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_label"].tolist()
        x_values = [x for x in setting_order if x in len_df[x_axis].unique()]
        extra = [x for x in len_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra

    elif x_axis == "setting_id" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_id"].tolist()
        x_values = [x for x in setting_order if x in len_df[x_axis].unique()]
        extra = [x for x in len_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra

    else:
        x_values = len_df[x_axis].dropna().unique().tolist()
        try:
            x_values = sorted(x_values)
        except Exception:
            x_values = list(x_values)

    fig, ax = plt.subplots(figsize=figsize)
    base_positions = np.arange(len(x_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    rng = np.random.default_rng(123)
    plotted_methods = []
    summary_rows = []
    all_x_positions = []

    for j, method in enumerate(method_order):
        color = colors.get(method, "#999999")

        x_positions = []
        mean_values = []
        error_values = []

        method_has_data = False

        for i, x_val in enumerate(x_values):
            vals = len_df.loc[
                (len_df["method"] == method) &
                (len_df[x_axis] == x_val),
                "avg_length"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            method_has_data = True
            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "sd":
                err_val = sd_val
            elif errorbar_type == "se":
                err_val = se_val
            else:
                raise ValueError("errorbar_type must be either 'sd' or 'se'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_rows.append({
                x_axis: x_val,
                "method": method,
                "mean_length": mean_val,
                "sd_length": sd_val,
                "se_length": se_val,
                "n_reps": len(vals),
                "min_length": float(np.min(vals)),
                "max_length": float(np.max(vals)),
            })

            if "epsilon" in len_df.columns:
                summary_rows[-1]["epsilon"] = (
                    float(epsilon_to_plot)
                    if epsilon_to_plot is not None
                    else float(len_df["epsilon"].dropna().iloc[0])
                )

            if show_points:
                x = pos + rng.uniform(-0.018, 0.018, size=len(vals))
                ax.plot(
                    x,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.8,
                )

        if method_has_data:
            plotted_methods.append(method)

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt="o",
                color=color,
                linewidth=1.8,
                markersize=5,
                capsize=4,
                elinewidth=1.3,
                label=method_labels.get(method, method),
            )

    if len(all_x_positions) > 0:
        ax.set_xlim(min(all_x_positions) - 0.15, max(all_x_positions) + 0.15)

    ax.set_xticks(base_positions)

    if hide_xtick_labels:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xticklabels([str(x) for x in x_values], rotation=rotate_xticks)
        ax.set_xlabel("Parameter setting")

    ax.set_ylabel("Average confidence interval length")

    if errorbar_type == "sd":
        ax.set_title("Gaussian data: confidence interval length, mean ± SD across simulations")
    else:
        ax.set_title("Gaussian data: confidence interval length, mean ± SE across simulations")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            color=colors.get(m, "#999999"),
            marker="o",
            linewidth=1.8,
            label=method_labels.get(m, m),
        )
        for m in plotted_methods
    ]

    ax.legend(handles=legend_handles, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_compare_coverage_boxplots_gaussian(
    compare_out,
    *,
    method_order=None,
    method_labels=None,
    colors=None,
    nominal=0.95,
    figsize=(10, 5),
    ylim=(0.7, 1.02),
    box_width=0.22,
    show_points=True,
    complete_only=False,
    errorbar_type="sd",   # "sd" or "se"

    # New options for multi_out
    x_axis="setting_label",      # "setting_label", "setting_id", or "signal_strength"
    hide_xtick_labels=False,
    epsilon_to_plot=None,
    rotate_xticks=0,
):

    cov_df = get_rep_level_coverage_df(compare_out, complete_only=complete_only)

    if cov_df.empty:
        print("No coverage data available to plot.")
        return cov_df

    # --------------------------------------------------
    # Optional epsilon filtering
    # --------------------------------------------------
    if "epsilon" in cov_df.columns:
        eps_values = sorted(cov_df["epsilon"].dropna().astype(float).unique())

        if epsilon_to_plot is not None:
            cov_df = cov_df[
                np.isclose(cov_df["epsilon"].astype(float), float(epsilon_to_plot))
            ].copy()
        else:
            if len(eps_values) > 1:
                raise ValueError(
                    "Multiple epsilon values found. "
                    "Since the x-axis is now parameter setting, please specify "
                    "epsilon_to_plot, e.g. epsilon_to_plot=10."
                )

    if method_order is None:
        method_order = infer_method_order(compare_out)

    method_order = list(method_order)

    if len(method_order) == 0 or cov_df.empty:
        print("No coverage data available to plot.")
        return cov_df

    required_cols = ["method", "rep", "coverage_rate", x_axis]
    missing_cols = [col for col in required_cols if col not in cov_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            f"rep-level dataframe is missing columns: {missing_cols}. "
            f"For multi_out, make sure all_rep_df contains '{x_axis}'."
        )

    if method_labels is None:
        method_labels = {
            "Standard": "Standard",
            "Randomized PSI": "Randomized PSI",
            "Polyhedral PSI": "Polyhedral PSI",
            "Data Splitting": "Data Splitting",
            "Zoom Correction": "Zoom Correction",
        }

    if colors is None:
        colors = {
            "Standard": "#7A7A7A",
            "Randomized PSI": "#1F77B4",
            "Polyhedral PSI": "#9467BD",
            "Data Splitting": "#2CA02C",
            "Zoom Correction": "#E6550D",
        }

    # --------------------------------------------------
    # x-axis values: parameter settings, not epsilon
    # --------------------------------------------------
    if x_axis == "setting_label" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_label"].tolist()
        x_values = [x for x in setting_order if x in cov_df[x_axis].unique()]
        extra = [x for x in cov_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra

    elif x_axis == "setting_id" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_id"].tolist()
        x_values = [x for x in setting_order if x in cov_df[x_axis].unique()]
        extra = [x for x in cov_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra

    else:
        x_values = cov_df[x_axis].dropna().unique().tolist()
        try:
            x_values = sorted(x_values)
        except Exception:
            x_values = list(x_values)

    fig, ax = plt.subplots(figsize=figsize)
    base_positions = np.arange(len(x_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    rng = np.random.default_rng(123)
    plotted_methods = []
    summary_rows = []
    all_x_positions = []

    for j, method in enumerate(method_order):
        color = colors.get(method, "#999999")

        x_positions = []
        mean_values = []
        error_values = []

        method_has_data = False

        for i, x_val in enumerate(x_values):
            vals = cov_df.loc[
                (cov_df["method"] == method) &
                (cov_df[x_axis] == x_val),
                "coverage_rate"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            method_has_data = True
            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "sd":
                err_val = sd_val
            elif errorbar_type == "se":
                err_val = se_val
            else:
                raise ValueError("errorbar_type must be either 'sd' or 'se'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_rows.append({
                x_axis: x_val,
                "method": method,
                "mean_coverage": mean_val,
                "sd_coverage": sd_val,
                "se_coverage": se_val,
                "n_reps": len(vals),
                "min_coverage": float(np.min(vals)),
                "max_coverage": float(np.max(vals)),
            })

            if "epsilon" in cov_df.columns:
                summary_rows[-1]["epsilon"] = (
                    float(epsilon_to_plot)
                    if epsilon_to_plot is not None
                    else float(cov_df["epsilon"].dropna().iloc[0])
                )

            if show_points:
                x = pos + rng.uniform(-0.018, 0.018, size=len(vals))
                ax.plot(
                    x,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.8,
                )

        if method_has_data:
            plotted_methods.append(method)

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt="o",
                color=color,
                linewidth=1.8,
                markersize=5,
                capsize=4,
                elinewidth=1.3,
                label=method_labels.get(method, method),
            )

    ax.axhline(
        nominal,
        linestyle="--",
        linewidth=1.3,
        color="black",
    )

    if len(all_x_positions) > 0:
        ax.set_xlim(min(all_x_positions) - 0.15, max(all_x_positions) + 0.15)

    ax.set_xticks(base_positions)

    if hide_xtick_labels:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xticklabels([str(x) for x in x_values], rotation=rotate_xticks)
        ax.set_xlabel("Parameter setting")

    ax.set_ylabel("Coverage rate")

    if errorbar_type == "sd":
        ax.set_title("Gaussian data: coverage rate, mean ± SD across simulations")
    else:
        ax.set_title("Gaussian data: coverage rate, mean ± SE across simulations")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            color=colors.get(m, "#999999"),
            marker="o",
            linewidth=1.8,
            label=method_labels.get(m, m),
        )
        for m in plotted_methods
    ]

    legend_handles.append(
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            label=f"Nominal = {nominal:.2f}",
        )
    )

    ax.legend(handles=legend_handles, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_compare_coverage_and_length_boxplots_gaussian(
    compare_out,
    *,
    method_order=None,
    method_labels=None,
    colors=None,
    nominal=0.95,
    coverage_figsize=(10, 5),
    length_figsize=(10, 5),
    coverage_ylim=(0.7, 1.02),
    length_ylim=None,
    box_width=0.22,
    show_points_coverage=True,
    show_points_length=False,
    complete_only=False,
    errorbar_type="se",

    # New options for multi_out
    x_axis="setting_label",
    hide_xtick_labels=False,
    epsilon_to_plot=None,
    rotate_xticks=0,
):

    if method_order is None:
        method_order = infer_method_order(compare_out)

    coverage_summary_df = plot_compare_coverage_boxplots_gaussian(
        compare_out,
        method_order=method_order,
        method_labels=method_labels,
        colors=colors,
        nominal=nominal,
        figsize=coverage_figsize,
        ylim=coverage_ylim,
        box_width=box_width,
        show_points=show_points_coverage,
        complete_only=complete_only,
        errorbar_type=errorbar_type,
        x_axis=x_axis,
        hide_xtick_labels=hide_xtick_labels,
        epsilon_to_plot=epsilon_to_plot,
        rotate_xticks=rotate_xticks,
    )

    length_summary_df = plot_compare_length_boxplots_gaussian(
        compare_out,
        method_order=method_order,
        method_labels=method_labels,
        colors=colors,
        figsize=length_figsize,
        ylim=length_ylim,
        box_width=box_width,
        show_points=show_points_length,
        complete_only=complete_only,
        errorbar_type=errorbar_type,
        x_axis=x_axis,
        hide_xtick_labels=hide_xtick_labels,
        epsilon_to_plot=epsilon_to_plot,
        rotate_xticks=rotate_xticks,
    )

    return {
        "coverage_summary": coverage_summary_df,
        "length_summary": length_summary_df,
    }



import ast
from matplotlib.lines import Line2D

import ast
from matplotlib.lines import Line2D



def _get_setting_group_cols(df):
    """
    For multi_out, use setting_id to separate parameter settings.
    For single compare_out, return [].
    """
    if "setting_id" in df.columns:
        return ["setting_id"]
    if "setting_label" in df.columns:
        return ["setting_label"]
    return []



def _canonical_subset(x):

    if x is None:
        return None

    if isinstance(x, float) and np.isnan(x):
        return None

    if isinstance(x, str):
        s = x.strip()
        if s in {"", "None", "nan", "NaN"}:
            return None
        try:
            x = ast.literal_eval(s)
        except Exception:
            s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            x = [v.strip() for v in s.split(",") if v.strip() != ""]

    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        vals = [int(v) for v in list(x)]
    else:
        vals = [int(x)]

    return tuple(sorted(vals))


def build_modal_subset_table(compare_out, *, max_rank=None):
    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_subset_df'.")

    subset_df = compare_out["all_subset_df"].copy()

    required_cols = {"method", "epsilon", "rep", "selected_subset"}
    missing = required_cols - set(subset_df.columns)
    if missing:
        raise ValueError(f"all_subset_df is missing columns: {missing}")

    if subset_df.empty:
        return pd.DataFrame(columns=[
            "method", "epsilon", "subset_rank", "modal_subset",
            "n_conditional_reps", "n_total_reps", "conditional_event_rate"
        ])

    subset_df["selected_subset"] = subset_df["selected_subset"].apply(_canonical_subset)

    rows = []

    for (method, eps), g in subset_df.groupby(["method", "epsilon"]):
        g = g.dropna(subset=["selected_subset"]).copy()

        if g.empty:
            continue

        counts = g["selected_subset"].value_counts()
        n_total = int(len(g))

        if max_rank is None:
            n_keep = len(counts)
        else:
            n_keep = min(int(max_rank), len(counts))

        for rank_idx in range(n_keep):
            selected_subset = counts.index[rank_idx]
            n_selected = int(counts.iloc[rank_idx])

            rows.append({
                "method": method,
                "epsilon": float(eps),
                "subset_rank": rank_idx + 1,
                "modal_subset": selected_subset,
                "n_conditional_reps": n_selected,
                "n_total_reps": n_total,
                "conditional_event_rate": n_selected / n_total,
            })

    modal_table = (
        pd.DataFrame(rows)
          .sort_values(["epsilon", "method", "subset_rank"])
          .reset_index(drop=True)
    )

    return modal_table


def _build_rep_coverage_table_for_conditional(
    all_ci_df,
    *,
    k=None,
    complete_only=False,
):


    df = all_ci_df.copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep",
            "coverage_rate", "avg_length",
            "n_intervals", "k", "is_complete"
        ])

    required_cols = {"method", "epsilon", "rep", "covered", "length"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"all_ci_df is missing columns: {missing}")

    if k is None:
        k = int(df.groupby(["method", "epsilon", "rep"]).size().max())

    rep_df = (
        df.groupby(["method", "epsilon", "rep"], as_index=False)
          .agg(
              n_covered=("covered", "sum"),
              n_intervals=("covered", "size"),
              avg_length=("length", "mean"),
          )
          .sort_values(["epsilon", "method", "rep"])
          .reset_index(drop=True)
    )

    rep_df["k"] = int(k)
    rep_df["coverage_rate"] = rep_df["n_covered"] / rep_df["k"]
    rep_df["is_complete"] = rep_df["n_intervals"] == rep_df["k"]

    if complete_only:
        rep_df = rep_df[rep_df["is_complete"]].copy()

    return rep_df.reset_index(drop=True)


def make_conditional_compare_out_by_modal_subset(
    compare_out,
    *,
    subset_rank=1,
    selected_subset=None,
    min_conditional_reps=1,
    complete_only=True,
    verbose=True,
):
    if "all_ci_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_ci_df'.")
    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_subset_df'.")

    all_ci_df = compare_out["all_ci_df"].copy()
    all_subset_df = compare_out["all_subset_df"].copy()

    required_ci_cols = {"method", "epsilon", "rep", "idx", "covered", "length"}
    missing_ci = required_ci_cols - set(all_ci_df.columns)
    if missing_ci:
        raise ValueError(f"all_ci_df is missing columns: {missing_ci}")

    required_subset_cols = {"method", "epsilon", "rep", "selected_subset"}
    missing_subset = required_subset_cols - set(all_subset_df.columns)
    if missing_subset:
        raise ValueError(f"all_subset_df is missing columns: {missing_subset}")

    k = int(
        compare_out.get(
            "k",
            all_ci_df.groupby(["method", "epsilon", "rep"]).size().max()
        )
    )

    all_subset_df["selected_subset"] = all_subset_df["selected_subset"].apply(_canonical_subset)

    full_modal_table = build_modal_subset_table(compare_out)

    if full_modal_table.empty:
        raise ValueError("No selected subsets found. Check compare_out['all_subset_df'].")

    # ============================================================
    # Mode 1: original behavior, condition on subset_rank
    # ============================================================
    if selected_subset is None:
        if int(subset_rank) < 1:
            raise ValueError("subset_rank must be >= 1.")

        subset_rank = int(subset_rank)
        condition_mode = "modal_rank"
        target_subset = None

        target_table = full_modal_table[
            full_modal_table["subset_rank"] == subset_rank
        ].copy()

        if target_table.empty:
            raise ValueError(
                f"No subset_rank={subset_rank} found. "
                "This usually means each method/epsilon group has fewer distinct selected subsets."
            )

        target_table = target_table[
            target_table["n_conditional_reps"] >= int(min_conditional_reps)
        ].copy()

        if target_table.empty:
            raise ValueError(
                f"No method/epsilon group has subset_rank={subset_rank} "
                f"with at least min_conditional_reps={min_conditional_reps}. "
                "Try lowering min_conditional_reps."
            )

    # ============================================================
    # Mode 2: new behavior, condition on user-specified subset
    # ============================================================
    else:
        condition_mode = "specified_subset"
        target_subset = _canonical_subset(selected_subset)

        if target_subset is None:
            raise ValueError("selected_subset cannot be None after canonicalization.")

        if len(target_subset) != k:
            raise ValueError(
                f"selected_subset has length {len(target_subset)}, but k={k}. "
                "For top-k conditional coverage, selected_subset should have exactly k indices."
            )

        # Count total reps per method/epsilon.
        group_total = (
            all_subset_df
            .groupby(["method", "epsilon"], as_index=False)
            .agg(n_total_reps=("rep", "nunique"))
        )

        # Count how many times the specified subset appears.
        selected_count = (
            all_subset_df[all_subset_df["selected_subset"] == target_subset]
            .groupby(["method", "epsilon"], as_index=False)
            .agg(n_conditional_reps=("rep", "nunique"))
        )

        target_table = group_total.merge(
            selected_count,
            on=["method", "epsilon"],
            how="left",
        )

        target_table["n_conditional_reps"] = (
            target_table["n_conditional_reps"].fillna(0).astype(int)
        )

        target_table["conditional_event_rate"] = (
            target_table["n_conditional_reps"] / target_table["n_total_reps"]
        )

        target_table["subset_rank"] = np.nan
        target_table["modal_subset"] = [target_subset] * len(target_table)

        target_table = target_table[
            target_table["n_conditional_reps"] >= int(min_conditional_reps)
        ].copy()

        if target_table.empty:
            raise ValueError(
                f"The specified subset {target_subset} was not selected at least "
                f"min_conditional_reps={min_conditional_reps} times for any method/epsilon group. "
                "Try lowering min_conditional_reps or choose a more frequently selected subset."
            )

    # Attach target subset to every subset record.
    subset_with_target = all_subset_df.merge(
        target_table[["method", "epsilon", "subset_rank", "modal_subset"]],
        on=["method", "epsilon"],
        how="inner",
    )

    # Keep reps where selected_subset equals the target subset.
    conditional_subset_df = subset_with_target[
        subset_with_target["selected_subset"] == subset_with_target["modal_subset"]
    ].copy()

    conditional_keys = (
        conditional_subset_df[["method", "epsilon", "rep"]]
        .drop_duplicates()
        .copy()
    )

    conditional_ci_df = all_ci_df.merge(
        conditional_keys,
        on=["method", "epsilon", "rep"],
        how="inner",
    )

    conditional_rep_df = _build_rep_coverage_table_for_conditional(
        conditional_ci_df,
        k=k,
        complete_only=False,
    )

    if complete_only:
        complete_keys = (
            conditional_rep_df[conditional_rep_df["is_complete"]]
            [["method", "epsilon", "rep"]]
            .drop_duplicates()
        )

        conditional_ci_df = conditional_ci_df.merge(
            complete_keys,
            on=["method", "epsilon", "rep"],
            how="inner",
        )

        conditional_subset_df = conditional_subset_df.merge(
            complete_keys,
            on=["method", "epsilon", "rep"],
            how="inner",
        )

        conditional_rep_df = _build_rep_coverage_table_for_conditional(
            conditional_ci_df,
            k=k,
            complete_only=False,
        )

        conditional_keys = complete_keys.copy()

    # Refresh used counts after complete_only filtering.
    if not conditional_subset_df.empty:
        final_counts = (
            conditional_subset_df
            .groupby(["method", "epsilon"], as_index=False)
            .agg(n_used_reps=("rep", "nunique"))
        )

        target_table = target_table.merge(
            final_counts,
            on=["method", "epsilon"],
            how="left",
        )

        target_table["n_used_reps"] = target_table["n_used_reps"].fillna(0).astype(int)
    else:
        target_table["n_used_reps"] = 0

    conditional_out = dict(compare_out)
    conditional_out["all_ci_df"] = conditional_ci_df.reset_index(drop=True)
    conditional_out["all_subset_df"] = conditional_subset_df.reset_index(drop=True)
    conditional_out["all_rep_df"] = conditional_rep_df.reset_index(drop=True)

    conditional_out["all_subset_frequency_table"] = full_modal_table.reset_index(drop=True)
    conditional_out["modal_subset_table"] = target_table.reset_index(drop=True)
    conditional_out["conditional_keys"] = conditional_keys.reset_index(drop=True)

    conditional_out["conditional_mode"] = condition_mode
    conditional_out["conditional_subset_rank"] = subset_rank if selected_subset is None else np.nan
    conditional_out["conditional_selected_subset"] = target_subset

    if verbose:
        if condition_mode == "modal_rank":
            print(f"\n===== Conditional event: subset_rank = {subset_rank} =====")
        else:
            print(f"\n===== Conditional event: selected_subset = {target_subset} =====")

        print("\n===== Selected subset table used for conditioning =====")
        print(conditional_out["modal_subset_table"])

        print("\n===== Conditional all_ci_df rows =====")
        if len(conditional_ci_df) > 0:
            print(conditional_ci_df.groupby(["method", "epsilon"]).size())
        else:
            print("No conditional CI rows.")

        print("\n===== Conditional rep-level coverage rows =====")
        if len(conditional_rep_df) > 0:
            print(conditional_rep_df.groupby(["method", "epsilon"]).size())
        else:
            print("No conditional rep-level rows.")

    return conditional_out


def _infer_method_order_for_conditional(compare_out, method_order=None):
    if method_order is not None:
        return list(method_order)

    preferred_order = [
        "Standard",
        "Randomized PSI",
        "Polyhedral PSI",
        "Data Splitting",
        "Zoom Correction",
    ]

    if "all_ci_df" in compare_out:
        df = compare_out["all_ci_df"].copy()
    elif "all_rep_df" in compare_out:
        df = compare_out["all_rep_df"].copy()
    else:
        return []

    if df.empty or "method" not in df.columns:
        return []

    existing_methods = df["method"].dropna().unique().tolist()
    ordered = [m for m in preferred_order if m in existing_methods]
    extra = [m for m in existing_methods if m not in ordered]

    return ordered + extra


def _default_method_labels():
    return {
        "Standard": "Standard",
        "Randomized PSI": "Randomized PSI",
        "Polyhedral PSI": "Polyhedral PSI",
        "Data Splitting": "Data Splitting",
        "Zoom Correction": "Zoom Correction",
    }


def _default_method_colors():
    return {
        "Standard": "#7A7A7A",          # gray
        "Randomized PSI": "#1F77B4",   # blue
        "Polyhedral PSI": "#9467BD",   # purple
        "Data Splitting": "#2CA02C",   # green
        "Zoom Correction": "#E6550D",  # red/orange
    }


def plot_conditional_coverage_errorbars_gaussian(
    conditional_out,
    *,
    method_order=None,
    method_labels=None,
    colors=None,
    nominal=0.95,
    figsize=(6.5, 4.2),
    ylim=(0.7, 1.02),
    box_width=0.22,
    show_points=True,
    errorbar_type="se",
    compact_x=True,
    x_margin=0.10,
):

    if "all_rep_df" not in conditional_out:
        raise ValueError("conditional_out must contain 'all_rep_df'.")

    rep_df = conditional_out["all_rep_df"].copy()

    if rep_df.empty:
        print("No conditional rep-level coverage data available to plot.")
        return pd.DataFrame()

    required_cols = {"method", "epsilon", "rep", "coverage_rate"}
    missing = required_cols - set(rep_df.columns)
    if missing:
        raise ValueError(f"conditional_out['all_rep_df'] is missing columns: {missing}")

    method_order = _infer_method_order_for_conditional(conditional_out, method_order)

    if method_labels is None:
        method_labels = _default_method_labels()

    if colors is None:
        colors = _default_method_colors()

    eps_values = sorted(rep_df["epsilon"].dropna().unique())
    base_positions = np.arange(len(eps_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(123)
    plotted_methods = []
    all_x_positions = []
    summary_rows = []

    condition_mode = conditional_out.get("conditional_mode", "modal_rank")
    condition_subset = conditional_out.get("conditional_selected_subset", None)
    condition_rank = conditional_out.get("conditional_subset_rank", np.nan)

    for j, method in enumerate(method_order):
        color = colors.get(method, "#999999")

        x_positions = []
        mean_values = []
        error_values = []

        method_has_data = False

        for i, eps in enumerate(eps_values):
            vals = rep_df.loc[
                (rep_df["method"] == method) & (rep_df["epsilon"] == eps),
                "coverage_rate"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            method_has_data = True
            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "se":
                err_val = se_val
            elif errorbar_type == "sd":
                err_val = sd_val
            else:
                raise ValueError("errorbar_type must be either 'se' or 'sd'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_rows.append({
                "method": method,
                "epsilon": eps,
                "condition_mode": condition_mode,
                "subset_rank": condition_rank,
                "selected_subset": condition_subset,
                "mean_conditional_coverage": mean_val,
                "sd_conditional_coverage": sd_val,
                "se_conditional_coverage": se_val,
                "n_conditional_reps": len(vals),
                "min_conditional_coverage": float(np.min(vals)),
                "max_conditional_coverage": float(np.max(vals)),
            })

            if show_points:
                x = pos + rng.uniform(-0.012, 0.012, size=len(vals))
                ax.plot(
                    x,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.5,
                )

        if method_has_data:
            plotted_methods.append(method)

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt="o-",
                color=color,
                linewidth=1.5,
                markersize=4.5,
                capsize=3,
                elinewidth=1.1,
                label=method_labels.get(method, method),
            )

    ax.axhline(
        nominal,
        linestyle="--",
        linewidth=1.3,
        color="black",
    )

    if compact_x and len(all_x_positions) > 0:
        xmin = min(all_x_positions)
        xmax = max(all_x_positions)
        ax.set_xlim(xmin - x_margin, xmax + x_margin)

    ax.set_xticks(base_positions)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("Conditional coverage rate")

    if condition_mode == "specified_subset":
        condition_label = f"specified subset {condition_subset}"
    else:
        condition_label = f"subset rank {condition_rank}"

    if errorbar_type == "se":
        ax.set_title(f"Conditional coverage, {condition_label}: mean ± SE")
    else:
        ax.set_title(f"Conditional coverage, {condition_label}: mean ± SD")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            color=colors.get(m, "#999999"),
            marker="o",
            linewidth=1.5,
            label=method_labels.get(m, m),
        )
        for m in plotted_methods
    ]

    legend_handles.append(
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            label=f"Nominal = {nominal:.2f}",
        )
    )

    ax.legend(handles=legend_handles, frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_conditional_length_errorbars_gaussian(
    conditional_out,
    *,
    method_order=None,
    method_labels=None,
    colors=None,
    figsize=(6.5, 4.2),
    ylim=None,
    box_width=0.22,
    show_points=False,
    errorbar_type="se",
    compact_x=True,
    x_margin=0.10,
):


    if "all_rep_df" not in conditional_out:
        raise ValueError("conditional_out must contain 'all_rep_df'.")

    rep_df = conditional_out["all_rep_df"].copy()

    if rep_df.empty:
        print("No conditional rep-level length data available to plot.")
        return pd.DataFrame()

    required_cols = {"method", "epsilon", "rep", "avg_length"}
    missing = required_cols - set(rep_df.columns)
    if missing:
        raise ValueError(f"conditional_out['all_rep_df'] is missing columns: {missing}")

    method_order = _infer_method_order_for_conditional(conditional_out, method_order)

    if method_labels is None:
        method_labels = _default_method_labels()

    if colors is None:
        colors = _default_method_colors()

    eps_values = sorted(rep_df["epsilon"].dropna().unique())
    base_positions = np.arange(len(eps_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(123)
    plotted_methods = []
    all_x_positions = []
    summary_rows = []

    condition_mode = conditional_out.get("conditional_mode", "modal_rank")
    condition_subset = conditional_out.get("conditional_selected_subset", None)
    condition_rank = conditional_out.get("conditional_subset_rank", np.nan)

    for j, method in enumerate(method_order):
        color = colors.get(method, "#999999")

        x_positions = []
        mean_values = []
        error_values = []

        method_has_data = False

        for i, eps in enumerate(eps_values):
            vals = rep_df.loc[
                (rep_df["method"] == method) & (rep_df["epsilon"] == eps),
                "avg_length"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            method_has_data = True
            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "se":
                err_val = se_val
            elif errorbar_type == "sd":
                err_val = sd_val
            else:
                raise ValueError("errorbar_type must be either 'se' or 'sd'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_rows.append({
                "method": method,
                "epsilon": eps,
                "condition_mode": condition_mode,
                "subset_rank": condition_rank,
                "selected_subset": condition_subset,
                "mean_conditional_length": mean_val,
                "sd_conditional_length": sd_val,
                "se_conditional_length": se_val,
                "n_conditional_reps": len(vals),
                "min_conditional_length": float(np.min(vals)),
                "max_conditional_length": float(np.max(vals)),
            })

            if show_points:
                x = pos + rng.uniform(-0.012, 0.012, size=len(vals))
                ax.plot(
                    x,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.5,
                )

        if method_has_data:
            plotted_methods.append(method)

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt="o-",
                color=color,
                linewidth=1.5,
                markersize=4.5,
                capsize=3,
                elinewidth=1.1,
                label=method_labels.get(method, method),
            )

    if compact_x and len(all_x_positions) > 0:
        xmin = min(all_x_positions)
        xmax = max(all_x_positions)
        ax.set_xlim(xmin - x_margin, xmax + x_margin)

    ax.set_xticks(base_positions)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("Conditional average CI length")

    if condition_mode == "specified_subset":
        condition_label = f"specified subset {condition_subset}"
    else:
        condition_label = f"subset rank {condition_rank}"

    if errorbar_type == "se":
        ax.set_title(f"Conditional CI length, {condition_label}: mean ± SE")
    else:
        ax.set_title(f"Conditional CI length, {condition_label}: mean ± SD")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            color=colors.get(m, "#999999"),
            marker="o",
            linewidth=1.5,
            label=method_labels.get(m, m),
        )
        for m in plotted_methods
    ]

    ax.legend(handles=legend_handles, frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_conditional_modal_subset_coverage_and_length_gaussian(
    compare_out,
    *,
    subset_rank=1,
    selected_subset=None,
    min_conditional_reps=1,
    complete_only=True,
    method_order=None,
    method_labels=None,
    colors=None,
    nominal=0.95,
    coverage_figsize=(6.5, 4.2),
    length_figsize=(6.5, 4.2),
    coverage_ylim=(0.7, 1.02),
    length_ylim=None,
    box_width=0.22,
    show_points_coverage=True,
    show_points_length=False,
    errorbar_type="se",
    verbose=True,
):


    conditional_out = make_conditional_compare_out_by_modal_subset(
        compare_out,
        subset_rank=subset_rank,
        selected_subset=selected_subset,
        min_conditional_reps=min_conditional_reps,
        complete_only=complete_only,
        verbose=verbose,
    )

    coverage_summary_df = plot_conditional_coverage_errorbars_gaussian(
        conditional_out,
        method_order=method_order,
        method_labels=method_labels,
        colors=colors,
        nominal=nominal,
        figsize=coverage_figsize,
        ylim=coverage_ylim,
        box_width=box_width,
        show_points=show_points_coverage,
        errorbar_type=errorbar_type,
    )

    conditional_out["conditional_coverage_summary_df"] = coverage_summary_df

    plot_compare_length_boxplots_gaussian(
        conditional_out,
        method_order=method_order,
        method_labels=method_labels,
        colors=colors,
        figsize=length_figsize,
        ylim=length_ylim,
        box_width=box_width,
        show_points=show_points_length,
    )

    return conditional_out




def _canonical_subset(x):

    if x is None:
        return None

    if isinstance(x, float) and np.isnan(x):
        return None

    if isinstance(x, str):
        s = x.strip()
        if s in {"", "None", "nan", "NaN"}:
            return None
        try:
            x = ast.literal_eval(s)
        except Exception:
            s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            x = [v.strip() for v in s.split(",") if v.strip() != ""]

    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        vals = [int(v) for v in list(x)]
    else:
        vals = [int(x)]

    return tuple(sorted(vals))


def build_modal_subset_table(compare_out, *, max_rank=None):
    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_subset_df'.")

    subset_df = compare_out["all_subset_df"].copy()

    required_cols = {"method", "epsilon", "rep", "selected_subset"}
    missing = required_cols - set(subset_df.columns)
    if missing:
        raise ValueError(f"all_subset_df is missing columns: {missing}")

    setting_group_cols = _get_setting_group_cols(subset_df)
    group_cols = setting_group_cols + ["method", "epsilon"]

    if subset_df.empty:
        return pd.DataFrame(columns=[
            *setting_group_cols,
            "setting_label",
            "signal_strength",
            "method",
            "epsilon",
            "subset_rank",
            "modal_subset",
            "n_conditional_reps",
            "n_total_reps",
            "conditional_event_rate",
        ])

    subset_df["selected_subset"] = subset_df["selected_subset"].apply(_canonical_subset)

    rows = []

    for group_key, g in subset_df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_cols, group_key))

        g = g.dropna(subset=["selected_subset"]).copy()

        if g.empty:
            continue

        counts = g["selected_subset"].value_counts()
        n_total = int(g["rep"].nunique())

        if max_rank is None:
            n_keep = len(counts)
        else:
            n_keep = min(int(max_rank), len(counts))

        for rank_idx in range(n_keep):
            selected_subset = counts.index[rank_idx]
            n_selected = int(counts.iloc[rank_idx])

            row = {
                **group_info,
                "subset_rank": rank_idx + 1,
                "modal_subset": selected_subset,
                "n_conditional_reps": n_selected,
                "n_total_reps": n_total,
                "conditional_event_rate": n_selected / n_total if n_total > 0 else np.nan,
            }

            if "setting_label" in g.columns:
                row["setting_label"] = g["setting_label"].iloc[0]

            if "signal_strength" in g.columns:
                row["signal_strength"] = g["signal_strength"].iloc[0]

            rows.append(row)

    sort_cols = setting_group_cols + ["epsilon", "method", "subset_rank"]

    modal_table = (
        pd.DataFrame(rows)
          .sort_values(sort_cols)
          .reset_index(drop=True)
    )

    return modal_table




def _build_rep_coverage_table_for_conditional(
    all_ci_df,
    *,
    k=None,
    complete_only=False,
):
    df = all_ci_df.copy()

    setting_group_cols = _get_setting_group_cols(df)
    group_cols = setting_group_cols + ["method", "epsilon", "rep"]

    if df.empty:
        return pd.DataFrame(columns=[
            *setting_group_cols,
            "setting_label",
            "signal_strength",
            "method",
            "epsilon",
            "rep",
            "coverage_rate",
            "avg_length",
            "n_intervals",
            "k",
            "is_complete",
        ])

    required_cols = {"method", "epsilon", "rep", "covered", "length"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"all_ci_df is missing columns: {missing}")

    if k is None:
        k = int(df.groupby(group_cols).size().max())

    agg_dict = {
        "n_covered": ("covered", "sum"),
        "n_intervals": ("covered", "size"),
        "avg_length": ("length", "mean"),
    }

    if "setting_label" in df.columns:
        agg_dict["setting_label"] = ("setting_label", "first")

    if "signal_strength" in df.columns:
        agg_dict["signal_strength"] = ("signal_strength", "first")

    rep_df = (
        df.groupby(group_cols, as_index=False)
          .agg(**agg_dict)
          .sort_values(setting_group_cols + ["epsilon", "method", "rep"])
          .reset_index(drop=True)
    )

    rep_df["k"] = int(k)
    rep_df["coverage_rate"] = rep_df["n_covered"] / rep_df["k"]
    rep_df["is_complete"] = rep_df["n_intervals"] == rep_df["k"]

    if complete_only:
        rep_df = rep_df[rep_df["is_complete"]].copy()

    return rep_df.reset_index(drop=True)




def make_conditional_compare_out_by_modal_subset(
    compare_out,
    *,
    subset_rank=1,
    selected_subset=None,
    min_conditional_reps=1,
    complete_only=True,
    verbose=True,
):
    if "all_ci_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_ci_df'.")
    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_subset_df'.")

    all_ci_df = compare_out["all_ci_df"].copy()
    all_subset_df = compare_out["all_subset_df"].copy()

    required_ci_cols = {"method", "epsilon", "rep", "idx", "covered", "length"}
    missing_ci = required_ci_cols - set(all_ci_df.columns)
    if missing_ci:
        raise ValueError(f"all_ci_df is missing columns: {missing_ci}")

    required_subset_cols = {"method", "epsilon", "rep", "selected_subset"}
    missing_subset = required_subset_cols - set(all_subset_df.columns)
    if missing_subset:
        raise ValueError(f"all_subset_df is missing columns: {missing_subset}")

    k = int(
        compare_out.get(
            "k",
            all_ci_df.groupby(["method", "epsilon", "rep"]).size().max()
        )
    )

    all_subset_df["selected_subset"] = all_subset_df["selected_subset"].apply(_canonical_subset)

    setting_group_cols = _get_setting_group_cols(all_subset_df)
    group_cols = setting_group_cols + ["method", "epsilon"]

    full_modal_table = build_modal_subset_table(compare_out)

    if full_modal_table.empty:
        raise ValueError("No selected subsets found. Check compare_out['all_subset_df'].")

    # ============================================================
    # Mode 1: condition on subset rank within each parameter setting
    # ============================================================
    if selected_subset is None:
        if int(subset_rank) < 1:
            raise ValueError("subset_rank must be >= 1.")

        subset_rank = int(subset_rank)
        condition_mode = "modal_rank"
        target_subset = None

        target_table = full_modal_table[
            full_modal_table["subset_rank"] == subset_rank
        ].copy()

        if target_table.empty:
            raise ValueError(
                f"No subset_rank={subset_rank} found."
            )

        target_table = target_table[
            target_table["n_conditional_reps"] >= int(min_conditional_reps)
        ].copy()

        if target_table.empty:
            raise ValueError(
                f"No setting/method/epsilon group has subset_rank={subset_rank} "
                f"with at least min_conditional_reps={min_conditional_reps}."
            )

    # ============================================================
    # Mode 2: condition on user-specified subset
    # ============================================================
    else:
        condition_mode = "specified_subset"
        target_subset = _canonical_subset(selected_subset)

        if target_subset is None:
            raise ValueError("selected_subset cannot be None after canonicalization.")

        if len(target_subset) != k:
            raise ValueError(
                f"selected_subset has length {len(target_subset)}, but k={k}."
            )

        group_total = (
            all_subset_df
            .groupby(group_cols, as_index=False)
            .agg(n_total_reps=("rep", "nunique"))
        )

        selected_count = (
            all_subset_df[all_subset_df["selected_subset"] == target_subset]
            .groupby(group_cols, as_index=False)
            .agg(n_conditional_reps=("rep", "nunique"))
        )

        target_table = group_total.merge(
            selected_count,
            on=group_cols,
            how="left",
        )

        target_table["n_conditional_reps"] = (
            target_table["n_conditional_reps"].fillna(0).astype(int)
        )

        target_table["conditional_event_rate"] = (
            target_table["n_conditional_reps"] / target_table["n_total_reps"]
        )

        target_table["subset_rank"] = np.nan
        target_table["modal_subset"] = [target_subset] * len(target_table)

        if "setting_label" in all_subset_df.columns:
            label_df = (
                all_subset_df[group_cols + ["setting_label"]]
                .drop_duplicates(subset=group_cols)
            )
            target_table = target_table.merge(label_df, on=group_cols, how="left")

        if "signal_strength" in all_subset_df.columns:
            sig_df = (
                all_subset_df[group_cols + ["signal_strength"]]
                .drop_duplicates(subset=group_cols)
            )
            target_table = target_table.merge(sig_df, on=group_cols, how="left")

        target_table = target_table[
            target_table["n_conditional_reps"] >= int(min_conditional_reps)
        ].copy()

        if target_table.empty:
            raise ValueError(
                f"The specified subset {target_subset} was not selected at least "
                f"min_conditional_reps={min_conditional_reps} times for any setting/method/epsilon group."
            )

    # Attach target subset to every subset record.
    subset_with_target = all_subset_df.merge(
        target_table[group_cols + ["subset_rank", "modal_subset"]],
        on=group_cols,
        how="inner",
    )

    conditional_subset_df = subset_with_target[
        subset_with_target["selected_subset"] == subset_with_target["modal_subset"]
    ].copy()

    conditional_key_cols = setting_group_cols + ["method", "epsilon", "rep"]

    conditional_keys = (
        conditional_subset_df[conditional_key_cols]
        .drop_duplicates()
        .copy()
    )

    conditional_ci_df = all_ci_df.merge(
        conditional_keys,
        on=conditional_key_cols,
        how="inner",
    )

    conditional_rep_df = _build_rep_coverage_table_for_conditional(
        conditional_ci_df,
        k=k,
        complete_only=False,
    )

    if complete_only:
        complete_keys = (
            conditional_rep_df[conditional_rep_df["is_complete"]]
            [conditional_key_cols]
            .drop_duplicates()
        )

        conditional_ci_df = conditional_ci_df.merge(
            complete_keys,
            on=conditional_key_cols,
            how="inner",
        )

        conditional_subset_df = conditional_subset_df.merge(
            complete_keys,
            on=conditional_key_cols,
            how="inner",
        )

        conditional_rep_df = _build_rep_coverage_table_for_conditional(
            conditional_ci_df,
            k=k,
            complete_only=False,
        )

        conditional_keys = complete_keys.copy()

    if not conditional_subset_df.empty:
        final_counts = (
            conditional_subset_df
            .groupby(group_cols, as_index=False)
            .agg(n_used_reps=("rep", "nunique"))
        )

        target_table = target_table.merge(
            final_counts,
            on=group_cols,
            how="left",
        )

        target_table["n_used_reps"] = target_table["n_used_reps"].fillna(0).astype(int)
    else:
        target_table["n_used_reps"] = 0

    conditional_out = dict(compare_out)
    conditional_out["all_ci_df"] = conditional_ci_df.reset_index(drop=True)
    conditional_out["all_subset_df"] = conditional_subset_df.reset_index(drop=True)
    conditional_out["all_rep_df"] = conditional_rep_df.reset_index(drop=True)

    conditional_out["all_subset_frequency_table"] = full_modal_table.reset_index(drop=True)
    conditional_out["modal_subset_table"] = target_table.reset_index(drop=True)
    conditional_out["conditional_keys"] = conditional_keys.reset_index(drop=True)

    conditional_out["conditional_mode"] = condition_mode
    conditional_out["conditional_subset_rank"] = subset_rank if selected_subset is None else np.nan
    conditional_out["conditional_selected_subset"] = target_subset
    conditional_out["setting_group_cols"] = setting_group_cols

    if verbose:
        if condition_mode == "modal_rank":
            print(f"\n===== Conditional event: subset_rank = {subset_rank} within each parameter setting =====")
        else:
            print(f"\n===== Conditional event: selected_subset = {target_subset} within each parameter setting =====")

        print("\n===== Selected subset table used for conditioning =====")
        print(conditional_out["modal_subset_table"])

        print("\n===== Conditional all_ci_df rows =====")
        if len(conditional_ci_df) > 0:
            print(conditional_ci_df.groupby(group_cols).size())
        else:
            print("No conditional CI rows.")

        print("\n===== Conditional rep-level coverage rows =====")
        if len(conditional_rep_df) > 0:
            print(conditional_rep_df.groupby(group_cols).size())
        else:
            print("No conditional rep-level rows.")

    return conditional_out




def _infer_method_order_for_conditional(compare_out, method_order=None):
    if method_order is not None:
        return list(method_order)

    preferred_order = [
        "Standard",
        "Randomized PSI",
        "Polyhedral PSI",
        "Data Splitting",
        "Zoom Correction",
    ]

    if "all_ci_df" in compare_out:
        df = compare_out["all_ci_df"].copy()
    elif "all_rep_df" in compare_out:
        df = compare_out["all_rep_df"].copy()
    else:
        return []

    if df.empty or "method" not in df.columns:
        return []

    existing_methods = df["method"].dropna().unique().tolist()
    ordered = [m for m in preferred_order if m in existing_methods]
    extra = [m for m in existing_methods if m not in ordered]

    return ordered + extra


def _default_method_labels():
    return {
        "Standard": "Standard",
        "Randomized PSI": "Randomized PSI",
        "Polyhedral PSI": "Polyhedral PSI",
        "Data Splitting": "Data Splitting",
        "Zoom Correction": "Zoom Correction",
    }


def _default_method_colors():
    return {
        "Standard": "#7A7A7A",          # gray
        "Randomized PSI": "#1F77B4",   # blue
        "Polyhedral PSI": "#9467BD",   # purple
        "Data Splitting": "#2CA02C",   # green
        "Zoom Correction": "#E6550D",  # red/orange
    }


def plot_conditional_coverage_errorbars_gaussian(
    conditional_out,
    *,
    method_order=None,
    method_labels=None,
    colors=None,
    nominal=0.95,
    figsize=(6.5, 4.2),
    ylim=(0.7, 1.02),
    box_width=0.22,
    show_points=True,
    errorbar_type="se",
    compact_x=True,
    x_margin=0.10,

    # New options
    x_axis="setting_label",
    hide_xtick_labels=False,
    epsilon_to_plot=None,
    rotate_xticks=0,
):
    if "all_rep_df" not in conditional_out:
        raise ValueError("conditional_out must contain 'all_rep_df'.")

    rep_df = conditional_out["all_rep_df"].copy()

    if rep_df.empty:
        print("No conditional rep-level coverage data available to plot.")
        return pd.DataFrame()

    if epsilon_to_plot is not None:
        rep_df = rep_df[
            np.isclose(rep_df["epsilon"].astype(float), float(epsilon_to_plot))
        ].copy()
    else:
        eps_values = sorted(rep_df["epsilon"].dropna().astype(float).unique())
        if len(eps_values) > 1:
            raise ValueError(
                "Multiple epsilon values found. Since x-axis is parameter setting, "
                "please specify epsilon_to_plot, e.g. epsilon_to_plot=10."
            )

    required_cols = {"method", "epsilon", "rep", "coverage_rate", x_axis}
    missing = required_cols - set(rep_df.columns)
    if missing:
        raise ValueError(f"conditional_out['all_rep_df'] is missing columns: {missing}")

    method_order = _infer_method_order_for_conditional(conditional_out, method_order)

    if method_labels is None:
        method_labels = _default_method_labels()

    if colors is None:
        colors = _default_method_colors()

    if x_axis == "setting_label" and "settings_df" in conditional_out:
        setting_order = conditional_out["settings_df"]["setting_label"].tolist()
        x_values = [x for x in setting_order if x in rep_df[x_axis].unique()]
        extra = [x for x in rep_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra
    elif x_axis == "setting_id" and "settings_df" in conditional_out:
        setting_order = conditional_out["settings_df"]["setting_id"].tolist()
        x_values = [x for x in setting_order if x in rep_df[x_axis].unique()]
        extra = [x for x in rep_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra
    else:
        x_values = rep_df[x_axis].dropna().unique().tolist()
        try:
            x_values = sorted(x_values)
        except Exception:
            x_values = list(x_values)

    base_positions = np.arange(len(x_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(123)
    plotted_methods = []
    all_x_positions = []
    summary_rows = []

    condition_mode = conditional_out.get("conditional_mode", "modal_rank")
    condition_subset = conditional_out.get("conditional_selected_subset", None)
    condition_rank = conditional_out.get("conditional_subset_rank", np.nan)

    for j, method in enumerate(method_order):
        color = colors.get(method, "#999999")

        x_positions = []
        mean_values = []
        error_values = []

        method_has_data = False

        for i, x_val in enumerate(x_values):
            vals = rep_df.loc[
                (rep_df["method"] == method) &
                (rep_df[x_axis] == x_val),
                "coverage_rate"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            method_has_data = True
            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "se":
                err_val = se_val
            elif errorbar_type == "sd":
                err_val = sd_val
            else:
                raise ValueError("errorbar_type must be either 'se' or 'sd'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_rows.append({
                x_axis: x_val,
                "method": method,
                "epsilon": (
                    float(epsilon_to_plot)
                    if epsilon_to_plot is not None
                    else float(rep_df["epsilon"].dropna().iloc[0])
                ),
                "condition_mode": condition_mode,
                "subset_rank": condition_rank,
                "selected_subset": condition_subset,
                "mean_conditional_coverage": mean_val,
                "sd_conditional_coverage": sd_val,
                "se_conditional_coverage": se_val,
                "n_conditional_reps": len(vals),
                "min_conditional_coverage": float(np.min(vals)),
                "max_conditional_coverage": float(np.max(vals)),
            })

            if show_points:
                x = pos + rng.uniform(-0.012, 0.012, size=len(vals))
                ax.plot(
                    x,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.5,
                )

        if method_has_data:
            plotted_methods.append(method)

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt="o",
                color=color,
                linewidth=1.5,
                markersize=4.5,
                capsize=3,
                elinewidth=1.1,
                label=method_labels.get(method, method),
            )

    ax.axhline(
        nominal,
        linestyle="--",
        linewidth=1.3,
        color="black",
    )

    if compact_x and len(all_x_positions) > 0:
        ax.set_xlim(min(all_x_positions) - x_margin, max(all_x_positions) + x_margin)

    ax.set_xticks(base_positions)

    if hide_xtick_labels:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xticklabels([str(x) for x in x_values], rotation=rotate_xticks)
        ax.set_xlabel("Parameter setting")

    ax.set_ylabel("Conditional coverage rate")

    if errorbar_type == "se":
        ax.set_title("Conditional coverage rate, mean ± SE")
    else:
        ax.set_title("Conditional coverage rate, mean ± SD")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            color=colors.get(m, "#999999"),
            marker="o",
            linewidth=1.5,
            label=method_labels.get(m, m),
        )
        for m in plotted_methods
    ]

    legend_handles.append(
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            label=f"Nominal = {nominal:.2f}",
        )
    )

    ax.legend(handles=legend_handles, frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(summary_rows)


def plot_conditional_length_errorbars_gaussian(
    conditional_out,
    *,
    method_order=None,
    method_labels=None,
    colors=None,
    figsize=(6.5, 4.2),
    ylim=None,
    box_width=0.22,
    show_points=False,
    errorbar_type="se",
    compact_x=True,
    x_margin=0.10,

    # New options
    x_axis="setting_label",
    hide_xtick_labels=False,
    epsilon_to_plot=None,
    rotate_xticks=0,
):
    if "all_rep_df" not in conditional_out:
        raise ValueError("conditional_out must contain 'all_rep_df'.")

    rep_df = conditional_out["all_rep_df"].copy()

    if rep_df.empty:
        print("No conditional rep-level length data available to plot.")
        return pd.DataFrame()

    if epsilon_to_plot is not None:
        rep_df = rep_df[
            np.isclose(rep_df["epsilon"].astype(float), float(epsilon_to_plot))
        ].copy()
    else:
        eps_values = sorted(rep_df["epsilon"].dropna().astype(float).unique())
        if len(eps_values) > 1:
            raise ValueError(
                "Multiple epsilon values found. Since x-axis is parameter setting, "
                "please specify epsilon_to_plot, e.g. epsilon_to_plot=10."
            )

    required_cols = {"method", "epsilon", "rep", "avg_length", x_axis}
    missing = required_cols - set(rep_df.columns)
    if missing:
        raise ValueError(f"conditional_out['all_rep_df'] is missing columns: {missing}")

    method_order = _infer_method_order_for_conditional(conditional_out, method_order)

    if method_labels is None:
        method_labels = _default_method_labels()

    if colors is None:
        colors = _default_method_colors()

    if x_axis == "setting_label" and "settings_df" in conditional_out:
        setting_order = conditional_out["settings_df"]["setting_label"].tolist()
        x_values = [x for x in setting_order if x in rep_df[x_axis].unique()]
        extra = [x for x in rep_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra
    elif x_axis == "setting_id" and "settings_df" in conditional_out:
        setting_order = conditional_out["settings_df"]["setting_id"].tolist()
        x_values = [x for x in setting_order if x in rep_df[x_axis].unique()]
        extra = [x for x in rep_df[x_axis].dropna().unique().tolist() if x not in x_values]
        x_values = x_values + extra
    else:
        x_values = rep_df[x_axis].dropna().unique().tolist()
        try:
            x_values = sorted(x_values)
        except Exception:
            x_values = list(x_values)

    base_positions = np.arange(len(x_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(123)
    plotted_methods = []
    all_x_positions = []
    summary_rows = []

    condition_mode = conditional_out.get("conditional_mode", "modal_rank")
    condition_subset = conditional_out.get("conditional_selected_subset", None)
    condition_rank = conditional_out.get("conditional_subset_rank", np.nan)

    for j, method in enumerate(method_order):
        color = colors.get(method, "#999999")

        x_positions = []
        mean_values = []
        error_values = []

        method_has_data = False

        for i, x_val in enumerate(x_values):
            vals = rep_df.loc[
                (rep_df["method"] == method) &
                (rep_df[x_axis] == x_val),
                "avg_length"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            method_has_data = True
            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "se":
                err_val = se_val
            elif errorbar_type == "sd":
                err_val = sd_val
            else:
                raise ValueError("errorbar_type must be either 'se' or 'sd'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_rows.append({
                x_axis: x_val,
                "method": method,
                "epsilon": (
                    float(epsilon_to_plot)
                    if epsilon_to_plot is not None
                    else float(rep_df["epsilon"].dropna().iloc[0])
                ),
                "condition_mode": condition_mode,
                "subset_rank": condition_rank,
                "selected_subset": condition_subset,
                "mean_conditional_length": mean_val,
                "sd_conditional_length": sd_val,
                "se_conditional_length": se_val,
                "n_conditional_reps": len(vals),
                "min_conditional_length": float(np.min(vals)),
                "max_conditional_length": float(np.max(vals)),
            })

            if show_points:
                x = pos + rng.uniform(-0.012, 0.012, size=len(vals))
                ax.plot(
                    x,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.5,
                )

        if method_has_data:
            plotted_methods.append(method)

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt="o",
                color=color,
                linewidth=1.5,
                markersize=4.5,
                capsize=3,
                elinewidth=1.1,
                label=method_labels.get(method, method),
            )

    if compact_x and len(all_x_positions) > 0:
        ax.set_xlim(min(all_x_positions) - x_margin, max(all_x_positions) + x_margin)

    ax.set_xticks(base_positions)

    if hide_xtick_labels:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xticklabels([str(x) for x in x_values], rotation=rotate_xticks)
        ax.set_xlabel("Parameter setting")

    ax.set_ylabel("Conditional average CI length")

    if errorbar_type == "se":
        ax.set_title("Conditional CI length, mean ± SE")
    else:
        ax.set_title("Conditional CI length, mean ± SD")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            color=colors.get(m, "#999999"),
            marker="o",
            linewidth=1.5,
            label=method_labels.get(m, m),
        )
        for m in plotted_methods
    ]

    ax.legend(handles=legend_handles, frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(summary_rows)




def plot_conditional_modal_subset_coverage_and_length_gaussian(
    compare_out,
    *,
    subset_rank=1,
    selected_subset=None,
    min_conditional_reps=1,
    complete_only=True,
    method_order=None,
    method_labels=None,
    colors=None,
    nominal=0.95,
    coverage_figsize=(6.5, 4.2),
    length_figsize=(6.5, 4.2),
    coverage_ylim=(0.7, 1.02),
    length_ylim=None,
    box_width=0.22,
    show_points_coverage=True,
    show_points_length=False,
    errorbar_type="se",
    verbose=True,

    # New options
    x_axis="setting_label",
    hide_xtick_labels=False,
    epsilon_to_plot=None,
    rotate_xticks=0,
):
    conditional_out = make_conditional_compare_out_by_modal_subset(
        compare_out,
        subset_rank=subset_rank,
        selected_subset=selected_subset,
        min_conditional_reps=min_conditional_reps,
        complete_only=complete_only,
        verbose=verbose,
    )

    coverage_summary_df = plot_conditional_coverage_errorbars_gaussian(
        conditional_out,
        method_order=method_order,
        method_labels=method_labels,
        colors=colors,
        nominal=nominal,
        figsize=coverage_figsize,
        ylim=coverage_ylim,
        box_width=box_width,
        show_points=show_points_coverage,
        errorbar_type=errorbar_type,
        x_axis=x_axis,
        hide_xtick_labels=hide_xtick_labels,
        epsilon_to_plot=epsilon_to_plot,
        rotate_xticks=rotate_xticks,
    )

    length_summary_df = plot_conditional_length_errorbars_gaussian(
        conditional_out,
        method_order=method_order,
        method_labels=method_labels,
        colors=colors,
        figsize=length_figsize,
        ylim=length_ylim,
        box_width=box_width,
        show_points=show_points_length,
        errorbar_type=errorbar_type,
        x_axis=x_axis,
        hide_xtick_labels=hide_xtick_labels,
        epsilon_to_plot=epsilon_to_plot,
        rotate_xticks=rotate_xticks,
    )

    conditional_out["conditional_coverage_summary_df"] = coverage_summary_df
    conditional_out["conditional_length_summary_df"] = length_summary_df

    return conditional_out


import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _as_index_tuple(x):
    if x is None:
        return tuple()

    if isinstance(x, float) and np.isnan(x):
        return tuple()

    if isinstance(x, tuple):
        return tuple(int(i) for i in x)

    if isinstance(x, list):
        return tuple(int(i) for i in x)

    if isinstance(x, np.ndarray):
        return tuple(int(i) for i in x.tolist())

    if isinstance(x, str):
        s = x.strip()

        if s in {"", "None", "nan", "NaN"}:
            return tuple()

        try:
            parsed = ast.literal_eval(s)
            return _as_index_tuple(parsed)
        except Exception:
            x_clean = (
                s.replace("(", "")
                 .replace(")", "")
                 .replace("[", "")
                 .replace("]", "")
            )
            if len(x_clean) == 0:
                return tuple()
            return tuple(int(i.strip()) for i in x_clean.split(",") if i.strip() != "")

    raise TypeError(f"Cannot parse selected_subset entry of type {type(x)}")


def _selection_rule_from_method(method):
    method = str(method)

    if method in ["Standard", "Polyhedral PSI", "Zoom Correction"]:
        return "Full-data argmax"

    if method == "Data Splitting":
        return "Half-data argmax"

    if method == "Randomized PSI":
        return "Randomized PSI"

    return method


def build_selection_utility_df_from_compare(
    compare_out,
    *,
    utility_fn=None,
    collapse_selection_rules=True,
    deduplicate=True,
    epsilon_to_plot=None,
):


    if "setting_outs" in compare_out:
        setting_outs = compare_out["setting_outs"]
    else:
        setting_outs = [compare_out]

    rows = []

    for setting_pos, out_s in enumerate(setting_outs):

        if "X_samples_shared" not in out_s or out_s["X_samples_shared"] is None:
            raise ValueError(
                "This empirical regret requires X_samples_shared. "
                "Please run simulation with share_same_data_across_methods=True."
            )

        if "all_subset_df" not in out_s:
            raise ValueError("Each compare_out must contain all_subset_df.")

        if "k" not in out_s:
            raise ValueError("Each compare_out must contain k.")

        X_samples = np.asarray(out_s["X_samples_shared"], dtype=float)
        subset_df = out_s["all_subset_df"].copy()
        k = int(out_s["k"])

        if X_samples.ndim != 3:
            raise ValueError(
                "X_samples_shared must have shape (B, n_obs, M). "
                f"Got shape {X_samples.shape}."
            )

        B, n_obs, M = X_samples.shape

        setting_id = out_s.get("setting_id", setting_pos)
        setting_label = out_s.get("setting_label", f"setting_{setting_pos}")
        signal_strength = out_s.get("signal_strength", np.nan)

        required_cols = ["method", "epsilon", "rep", "selected_subset"]
        missing_cols = [c for c in required_cols if c not in subset_df.columns]
        if missing_cols:
            raise ValueError(f"all_subset_df is missing columns: {missing_cols}")

        if epsilon_to_plot is not None:
            subset_df = subset_df[
                np.isclose(subset_df["epsilon"].astype(float), float(epsilon_to_plot))
            ].copy()

        if subset_df.empty:
            continue

        # --------------------------------------------------
        # Loop through each selected subset from the simulation.
        # This uses the actual selected subset for each method.
        # --------------------------------------------------
        for _, row in subset_df.iterrows():

            method = str(row["method"])
            selection_rule = _selection_rule_from_method(method)

            eps = float(row["epsilon"])
            rep = int(row["rep"])

            if rep < 0 or rep >= B:
                continue

            # Fixed statistic for this simulation replication.
            T_b = X_samples[rep].mean(axis=0)

            if utility_fn is None:
                score_vec = np.asarray(T_b, dtype=float).reshape(-1)
            else:
                score_vec = np.asarray(utility_fn(T_b), dtype=float).reshape(-1)

            if score_vec.shape[0] != M:
                raise ValueError("utility_fn(T_b) must return a vector of length M.")

            # Standard/full-data top-k utility: s^*(T_b)
            topk_idx = np.argsort(score_vec)[-k:]
            standard_subset = tuple(sorted(int(i) for i in topk_idx))
            standard_utility = float(np.sum(score_vec[list(standard_subset)]))

            # --------------------------------------------------
            # Full-data argmax methods should have regret 0.
            # We set it exactly to 0 to reflect the intended comparison group.
            # --------------------------------------------------
            if selection_rule == "Full-data argmax":
                selected_subset = standard_subset
                selected_utility = standard_utility
                regret = 0.0

            else:
                selected_subset = _as_index_tuple(row["selected_subset"])

                if len(selected_subset) != k:
                    continue

                selected_utility = float(np.sum(score_vec[list(selected_subset)]))
                regret = float(standard_utility - selected_utility)

                # Numerical safety.
                if regret < 0 and regret > -1e-10:
                    regret = 0.0

            rows.append({
                "setting_id": setting_id,
                "setting_label": setting_label,
                "signal_strength": signal_strength,
                "method": method,
                "selection_rule": selection_rule,
                "plot_group": selection_rule,
                "epsilon": eps,
                "rep": rep,
                "standard_subset": standard_subset,
                "selected_subset": selected_subset,
                "standard_utility": standard_utility,
                "selected_utility": selected_utility,
                "regret": regret,
            })

    score_df = pd.DataFrame(rows)

    if score_df.empty:
        return score_df

    if deduplicate:
        # Collapse Standard / Polyhedral PSI / Zoom Correction into one
        # Full-data argmax point per setting, epsilon, rep.
        score_df = (
            score_df.drop_duplicates(
                subset=["setting_id", "epsilon", "rep", "plot_group"]
            )
            .reset_index(drop=True)
        )

    return score_df


def build_selection_utility_df_from_compare(
    compare_out,
    *,
    utility_fn=None,
    collapse_selection_rules=True,
    deduplicate=True,
    epsilon_to_plot=None,
):
    """
    Robust selection-quality/regret builder for multi-setting output.

    Important:
        For multi-setting compare_out, this function uses compare_out["mu_list"]
        setting by setting, instead of using a single compare_out["mu"].
    """

    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_subset_df'.")

    subset_df = compare_out["all_subset_df"].copy()

    if subset_df.empty:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep", "selected_subset",
            "selected_utility", "standard_utility", "regret",
            "plot_group",
        ])

    # Filter epsilon early.
    if epsilon_to_plot is not None and "epsilon" in subset_df.columns:
        eps_num = pd.to_numeric(subset_df["epsilon"], errors="coerce")
        subset_df = subset_df[np.isclose(eps_num, float(epsilon_to_plot))].copy()

    if subset_df.empty:
        return pd.DataFrame(columns=[
            "method", "epsilon", "rep", "selected_subset",
            "selected_utility", "standard_utility", "regret",
            "plot_group",
        ])

    # Drop duplicated records if needed.
    if deduplicate:
        dedup_cols = [
            c for c in [
                "setting_id", "setting_label", "method",
                "epsilon", "rep", "selected_subset"
            ]
            if c in subset_df.columns
        ]
        subset_df = subset_df.drop_duplicates(subset=dedup_cols).copy()

    # --------------------------------------------------
    # Build setting-specific score vectors.
    # --------------------------------------------------
    setting_score = {}

    if "mu_list" in compare_out and "settings_df" in compare_out:
        settings_df = compare_out["settings_df"].copy()

        for ii, mu_s in enumerate(compare_out["mu_list"]):
            mu_s = np.asarray(mu_s, dtype=float).reshape(-1)

            if utility_fn is None:
                score_vec = mu_s.copy()
            else:
                score_vec = np.asarray(utility_fn(mu_s), dtype=float).reshape(-1)

            sid = settings_df.iloc[ii]["setting_id"]
            setting_score[sid] = score_vec

    elif "mu" in compare_out:
        mu_s = np.asarray(compare_out["mu"], dtype=float).reshape(-1)

        if utility_fn is None:
            score_vec = mu_s.copy()
        else:
            score_vec = np.asarray(utility_fn(mu_s), dtype=float).reshape(-1)

        setting_score[None] = score_vec

    else:
        raise ValueError("compare_out must contain either 'mu_list' or 'mu'.")

    k = int(compare_out["k"])

    rows = []

    for _, row in subset_df.iterrows():
        selected_subset = tuple(int(x) for x in row["selected_subset"])

        if len(selected_subset) != k:
            continue

        sid = row["setting_id"] if "setting_id" in row.index else None

        if sid not in setting_score:
            continue

        score_vec = setting_score[sid]
        M = len(score_vec)

        # Protect against stale/mismatched subset rows.
        if max(selected_subset) >= M or min(selected_subset) < 0:
            print(
                f"[skip invalid subset] setting_id={sid}, "
                f"M={M}, selected_subset={selected_subset}"
            )
            continue

        order = np.argsort(score_vec)[::-1]
        true_topk_subset = tuple(sorted(int(i) for i in order[:k]))

        standard_utility = float(np.sum(score_vec[list(true_topk_subset)]))
        selected_utility = float(np.sum(score_vec[list(selected_subset)]))
        regret = float(standard_utility - selected_utility)

        if abs(regret) < 1e-12:
            regret = 0.0

        method = row["method"]

        if collapse_selection_rules:
            if method in {"Standard", "Polyhedral PSI", "Zoom Correction"}:
                plot_group = "Full-data argmax"
            elif method == "Data Splitting":
                plot_group = "Half-data argmax"
            elif method == "Randomized PSI":
                plot_group = "Randomized PSI"
            else:
                plot_group = method
        else:
            plot_group = method

        out_row = {
            "method": method,
            "epsilon": row["epsilon"] if "epsilon" in row.index else np.nan,
            "rep": row["rep"],
            "selected_subset": selected_subset,
            "true_topk_subset": true_topk_subset,
            "selected_utility": selected_utility,
            "standard_utility": standard_utility,
            "regret": regret,
            "plot_group": plot_group,
        }

        for col in [
            "setting_id",
            "setting_label",
            "signal_strength",
            "standardized_topk_gap",
            "topk_gap",
            "top1_gap",
            "data_type",
            "nu_true",
            "n_matches",
        ]:
            if col in row.index:
                out_row[col] = row[col]

        rows.append(out_row)

    return pd.DataFrame(rows)


import ast
import numpy as np
import pandas as pd

def _parse_subset_for_regret(x):
    if isinstance(x, tuple):
        return tuple(int(i) for i in x)
    if isinstance(x, list):
        return tuple(int(i) for i in x)
    if isinstance(x, np.ndarray):
        return tuple(int(i) for i in x.tolist())
    if isinstance(x, str):
        return tuple(int(i) for i in ast.literal_eval(x))
    raise TypeError(f"Cannot parse selected_subset={x!r}")


def build_selection_utility_df_from_compare(
    compare_out,
    *,
    utility_fn=None,
    collapse_selection_rules=True,
    deduplicate=True,
    epsilon_to_plot=None,
):
    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain all_subset_df.")
    if "all_stat_df" not in compare_out:
        raise ValueError("compare_out must contain all_stat_df.")

    subset_df = compare_out["all_subset_df"].copy()
    stat_df = compare_out["all_stat_df"].copy()

    if subset_df.empty or stat_df.empty:
        return pd.DataFrame()

    if epsilon_to_plot is not None and "epsilon" in subset_df.columns:
        eps_num = pd.to_numeric(subset_df["epsilon"], errors="coerce")
        subset_df = subset_df[np.isclose(eps_num, float(epsilon_to_plot))].copy()

    k = int(compare_out["k"])

    # Use Standard full-data statistic as observed reference T_b.
    ref_df = stat_df[
        (stat_df["method"] == "Standard")
        & (stat_df["role"] == "selection")
    ].copy()

    if ref_df.empty:
        raise ValueError("Cannot find Standard full-data statistic in all_stat_df.")

    group_cols = ["rep"]
    if "setting_id" in ref_df.columns and "setting_id" in subset_df.columns:
        group_cols = ["setting_id", "rep"]
    elif "setting_label" in ref_df.columns and "setting_label" in subset_df.columns:
        group_cols = ["setting_label", "rep"]

    ref_score = {}

    for key, g in ref_df.groupby(group_cols, dropna=False):
        g = g.sort_values("idx")
        score_vec = g["score_hat"].to_numpy(dtype=float)

        if utility_fn is not None:
            score_vec = np.asarray(utility_fn(score_vec), dtype=float).reshape(-1)

        if not isinstance(key, tuple):
            key = (key,)

        ref_score[key] = score_vec

    rows = []

    for _, row in subset_df.iterrows():
        rep = int(row["rep"])

        key_vals = []
        if "setting_id" in group_cols:
            key_vals.append(row["setting_id"])
        elif "setting_label" in group_cols:
            key_vals.append(row["setting_label"])
        key_vals.append(rep)

        key = tuple(key_vals)

        if key not in ref_score:
            continue

        score_vec = ref_score[key]
        order = np.argsort(score_vec)[::-1]
        standard_subset = tuple(sorted(int(i) for i in order[:k]))
        standard_utility = float(np.sum(score_vec[list(standard_subset)]))

        method = row["method"]

        if collapse_selection_rules:
            if method in {"Standard", "Polyhedral PSI", "Zoom Correction"}:
                plot_group = "Full-data argmax"
            elif method == "Data Splitting":
                plot_group = "Half-data argmax"
            elif method == "Randomized PSI":
                plot_group = "Randomized PSI"
            else:
                plot_group = method
        else:
            plot_group = method

        if plot_group == "Full-data argmax":
            selected_subset = standard_subset
            selected_utility = standard_utility
            regret = 0.0
        else:
            selected_subset = _parse_subset_for_regret(row["selected_subset"])
            selected_utility = float(np.sum(score_vec[list(selected_subset)]))
            regret = float(standard_utility - selected_utility)
            if abs(regret) < 1e-12:
                regret = 0.0

        out_row = {
            "method": method,
            "epsilon": row["epsilon"] if "epsilon" in row.index else np.nan,
            "rep": rep,
            "selected_subset": selected_subset,
            "standard_subset": standard_subset,
            "selected_utility": selected_utility,
            "standard_utility": standard_utility,
            "regret": regret,
            "plot_group": plot_group,
        }

        for col in ["setting_id", "setting_label", "signal_strength", "data_type"]:
            if col in row.index:
                out_row[col] = row[col]

        rows.append(out_row)

    regret_df = pd.DataFrame(rows)

    if regret_df.empty:
        return regret_df

    if deduplicate:
        dedup_cols = [
            c for c in ["setting_id", "setting_label", "epsilon", "rep", "plot_group"]
            if c in regret_df.columns
        ]
        regret_df = regret_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    return regret_df




def plot_selection_utility_boxplots_from_compare(
    compare_out,
    *,
    utility_fn=None,
    collapse_selection_rules=True,
    deduplicate=True,
    epsilon_to_plot=None,
    method_order=None,
    labels=None,
    colors=None,
    figsize=(10, 5),
    box_width=0.20,
    ylim=None,
    show_points=False,
    title="Selection quality: empirical realized regret",
    show_zero_line=True,
    errorbar_type="se",   # "sd" or "se"
    x_axis="setting_label",
    hide_xtick_labels=False,
    rotate_xticks=0,
):

    score_df = build_selection_utility_df_from_compare(
        compare_out,
        utility_fn=utility_fn,
        collapse_selection_rules=collapse_selection_rules,
        deduplicate=deduplicate,
        epsilon_to_plot=epsilon_to_plot,
    )

    if score_df.empty:
        print("No selection regret data available.")
        return score_df

    eps_values = sorted(score_df["epsilon"].dropna().astype(float).unique())

    if epsilon_to_plot is None and len(eps_values) > 1:
        raise ValueError(
            "Multiple epsilon values found. Since x-axis is parameter setting, "
            "please specify epsilon_to_plot, e.g. epsilon_to_plot=10."
        )

    if method_order is None:
        method_order = [
            "Full-data argmax",
            "Half-data argmax",
            "Randomized PSI",
        ]

    method_order = [m for m in method_order if m in score_df["plot_group"].unique()]

    if labels is None:
        labels = {
            "Full-data argmax": "Full-data argmax",
            "Half-data argmax": "Half-data argmax",
            "Randomized PSI": "Randomized PSI",
        }

    if colors is None:
        colors = {
            "Full-data argmax": "#7A7A7A",
            "Half-data argmax": "#2CA02C",
            "Randomized PSI": "#1F77B4",
        }

    if x_axis not in score_df.columns:
        raise ValueError(
            f"x_axis='{x_axis}' is not in score_df. "
            f"Available columns are {list(score_df.columns)}."
        )

    # Preserve parameter-setting order from multi_out["settings_df"].
    if x_axis == "setting_label" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_label"].tolist()
        x_values = [x for x in setting_order if x in score_df[x_axis].unique()]
        extra = [
            x for x in score_df[x_axis].dropna().unique().tolist()
            if x not in x_values
        ]
        x_values = x_values + extra

    elif x_axis == "setting_id" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_id"].tolist()
        x_values = [x for x in setting_order if x in score_df[x_axis].unique()]
        extra = [
            x for x in score_df[x_axis].dropna().unique().tolist()
            if x not in x_values
        ]
        x_values = x_values + extra

    else:
        x_values = score_df[x_axis].dropna().unique().tolist()
        try:
            x_values = sorted(x_values)
        except Exception:
            x_values = list(x_values)

    base_positions = np.arange(len(x_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(123)

    plotted_groups = []
    summary_rows = []
    all_x_positions = []

    for j, group in enumerate(method_order):
        color = colors.get(group, "#999999")

        x_positions = []
        mean_values = []
        error_values = []
        has_data = False

        for i, x_val in enumerate(x_values):
            vals = score_df.loc[
                (score_df["plot_group"] == group) &
                (score_df[x_axis] == x_val),
                "regret"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            has_data = True
            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "sd":
                err_val = sd_val
            elif errorbar_type == "se":
                err_val = se_val
            else:
                raise ValueError("errorbar_type must be either 'sd' or 'se'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_row = {
                x_axis: x_val,
                "plot_group": group,
                "mean_regret": mean_val,
                "sd_regret": sd_val,
                "se_regret": se_val,
                "n_reps": len(vals),
                "min_regret": float(np.min(vals)),
                "max_regret": float(np.max(vals)),
            }

            if epsilon_to_plot is not None:
                summary_row["epsilon"] = float(epsilon_to_plot)
            elif len(eps_values) == 1:
                summary_row["epsilon"] = float(eps_values[0])

            summary_rows.append(summary_row)

            if show_points:
                x = pos + rng.uniform(-0.018, 0.018, size=len(vals))
                ax.plot(
                    x,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.8,
                )

        if has_data:
            plotted_groups.append(group)

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt="o",
                color=color,
                linewidth=1.8,
                markersize=5,
                capsize=4,
                elinewidth=1.3,
                label=labels.get(group, group),
            )

    if show_zero_line:
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="Zero regret",
        )

    if len(all_x_positions) > 0:
        ax.set_xlim(min(all_x_positions) - 0.15, max(all_x_positions) + 0.15)

    ax.set_xticks(base_positions)

    if hide_xtick_labels:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xticklabels([str(x) for x in x_values], rotation=rotate_xticks)
        ax.set_xlabel("Parameter setting")

    ax.set_ylabel(
        r"Empirical regret $s^*(T)-s_{\widehat E}(T)$"
    )

    if errorbar_type == "sd":
        ax.set_title(f"{title}: mean ± SD across simulations")
    else:
        ax.set_title(f"{title}: mean ± SE across simulations")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0],
            color=colors.get(g, "#999999"),
            marker="o",
            linewidth=1.8,
            label=labels.get(g, g),
        )
        for g in plotted_groups
    ]

    if show_zero_line:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=1.3,
                label="Zero regret",
            )
        )

    ax.legend(handles=legend_handles, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_rows)

    return {
        "regret_df": score_df,
        "summary_df": summary_df,
    }





def _as_index_tuple_safe(x):
    if isinstance(x, tuple):
        return tuple(int(i) for i in x)
    if isinstance(x, list):
        return tuple(int(i) for i in x)
    if isinstance(x, np.ndarray):
        return tuple(int(i) for i in x.tolist())

    if isinstance(x, str):
        x_clean = x.strip().replace("(", "").replace(")", "")
        if len(x_clean) == 0:
            return tuple()
        return tuple(int(i.strip()) for i in x_clean.split(",") if i.strip() != "")

    raise TypeError(f"Cannot parse selected_subset entry of type {type(x)}")


def _selection_rule_from_method_safe(method):
    method = str(method)

    if method in ["naive", "polyhedral", "zoom_stepdown"]:
        return "full_argmax"

    if method == "data_splitting":
        return "split_argmax"

    if method == "randomized":
        return "randomized"

    return method


def compute_true_winner_set_from_compare(
    compare_out,
    *,
    utility_fn=None,
    true_set_mode="topk",
    tol=1e-12,
):
    if "mu" not in compare_out:
        raise ValueError("compare_out must contain 'mu'.")
    if "k" not in compare_out:
        raise ValueError("compare_out must contain 'k'.")

    mu = np.asarray(compare_out["mu"], dtype=float).reshape(-1)
    k = int(compare_out["k"])

    if utility_fn is None:
        score = mu.copy()
    else:
        score = np.asarray(utility_fn(mu), dtype=float).reshape(-1)

    if score.shape[0] != mu.shape[0]:
        raise ValueError("utility_fn(mu) must return a vector with the same length as mu.")

    if true_set_mode == "topk":
        idx = np.argsort(score)[-k:]
        E_star = tuple(sorted(int(i) for i in idx))

    elif true_set_mode == "max":
        max_score = float(np.max(score))
        idx = np.where(np.isclose(score, max_score, atol=tol, rtol=0.0))[0]
        E_star = tuple(sorted(int(i) for i in idx))

    elif true_set_mode == "topk_with_ties":
        sorted_score = np.sort(score)[::-1]
        kth_score = float(sorted_score[k - 1])
        idx = np.where(score >= kth_score - tol)[0]
        E_star = tuple(sorted(int(i) for i in idx))

    else:
        raise ValueError("true_set_mode must be 'topk', 'max', or 'topk_with_ties'.")

    return E_star, score


def build_common_elements_df_from_compare(
    compare_out,
    *,
    utility_fn=None,
    true_set_mode="topk",
    collapse_selection_rules=True,
    deduplicate=True,
):

    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_subset_df'.")

    subset_df = compare_out["all_subset_df"].copy()

    if subset_df.empty:
        raise ValueError("compare_out['all_subset_df'] is empty.")

    required_cols = ["method", "epsilon", "rep", "selected_subset"]
    missing = [c for c in required_cols if c not in subset_df.columns]
    if missing:
        raise ValueError(f"all_subset_df is missing columns: {missing}")

    E_star, true_score = compute_true_winner_set_from_compare(
        compare_out,
        utility_fn=utility_fn,
        true_set_mode=true_set_mode,
    )

    E_star_set = set(E_star)

    rows = []
    for _, row in subset_df.iterrows():
        method = str(row["method"])
        eps = float(row["epsilon"])
        rep = int(row["rep"])
        S_hat = _as_index_tuple_safe(row["selected_subset"])
        S_hat_set = set(S_hat)

        n_common = len(E_star_set & S_hat_set)

        rows.append({
            "method": method,
            "selection_rule": _selection_rule_from_method_safe(method),
            "epsilon": eps,
            "rep": rep,
            "selected_subset": S_hat,
            "true_subset": E_star,
            "n_common": int(n_common),
            "common_rate": float(n_common / len(E_star_set)) if len(E_star_set) > 0 else np.nan,
        })

    out = pd.DataFrame(rows)

    if collapse_selection_rules:
        out["plot_group"] = out["selection_rule"]
    else:
        out["plot_group"] = out["method"]

    if deduplicate:
        # For full_argmax, naive/polyhedral/zoom are the same selection rule.
        # This avoids triple-counting the same selected subset.
        out = (
            out.drop_duplicates(
                subset=["plot_group", "epsilon", "rep", "selected_subset"]
            )
            .reset_index(drop=True)
        )

    return out

def plot_common_elements_selection_quality(
    compare_out,
    *,
    utility_fn=None,
    true_set_mode="topk",
    collapse_selection_rules=True,
    deduplicate=True,
    method_order=None,
    labels=None,
    colors=None,
    figsize=(8, 5),
    box_width=0.50,
    show_points=False,
    ylim=None,
    title=None,
    errorbar_type="sd",   # "sd" or "se"
    return_summary=False,
):
    """
    Plot selection quality by overlap with true winners using error-bar style.

    If there is only one epsilon value:
        x-axis = selection method.
    If there are multiple epsilon values:
        x-axis = epsilon, with one line per method.

    Error bars are mean ± SD or mean ± SE across simulations.
    """

    df = build_common_elements_df_from_compare(
        compare_out,
        utility_fn=utility_fn,
        true_set_mode=true_set_mode,
        collapse_selection_rules=collapse_selection_rules,
        deduplicate=deduplicate,
    )

    if df.empty:
        print("No data available to plot.")
        return df

    if labels is None:
        labels = {
            "full_argmax": "Full-data\nargmax",
            "split_argmax": "Half-data\nargmax",
            "randomized": "Randomized",
            "naive": "Naive",
            "polyhedral": "Polyhedral",
            "zoom_stepdown": "Zoom",
            "data_splitting": "Data splitting",
        }

    if colors is None:
        colors = {
            "full_argmax": "#4C78A8",
            "split_argmax": "#B279A2",
            "randomized": "#F58518",
            "naive": "#4C78A8",
            "polyhedral": "#54A24B",
            "zoom_stepdown": "#E45756",
            "data_splitting": "#B279A2",
        }

    if method_order is None:
        if collapse_selection_rules:
            method_order = ["full_argmax", "split_argmax", "randomized"]
        else:
            method_order = [
                "naive",
                "polyhedral",
                "zoom_stepdown",
                "data_splitting",
                "randomized",
            ]

    method_order = [m for m in method_order if m in df["plot_group"].unique()]

    if len(method_order) == 0:
        print("No matching methods available to plot.")
        return df

    eps_values = sorted(df["epsilon"].dropna().unique())
    use_epsilon_axis = len(eps_values) > 1

    E_star = df["true_subset"].iloc[0]
    max_common = len(E_star)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(123)
    summary_rows = []
    plotted_groups = []

    # =========================================================
    # Case 1: multiple epsilon values
    # x-axis = epsilon, one error-bar line per method
    # =========================================================
    if use_epsilon_axis:
        base_positions = np.arange(len(eps_values))

        if len(method_order) == 1:
            offsets = np.array([0.0])
        else:
            spacing_scale = 0.015
            offsets = np.linspace(
                -box_width * spacing_scale * (len(method_order) - 1) / 2,
                box_width * spacing_scale * (len(method_order) - 1) / 2,
                len(method_order),
            )

        for j, group in enumerate(method_order):
            color = colors.get(group, "#999999")

            x_positions = []
            mean_values = []
            error_values = []
            has_data = False

            for i, eps in enumerate(eps_values):
                vals = df.loc[
                    (df["plot_group"] == group) &
                    (df["epsilon"] == eps),
                    "n_common"
                ].dropna().to_numpy()

                if len(vals) == 0:
                    continue

                has_data = True
                pos = base_positions[i] + offsets[j]

                mean_val = float(np.mean(vals))
                sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

                if errorbar_type == "sd":
                    err_val = sd_val
                elif errorbar_type == "se":
                    err_val = se_val
                else:
                    raise ValueError("errorbar_type must be either 'sd' or 'se'.")

                x_positions.append(pos)
                mean_values.append(mean_val)
                error_values.append(err_val)

                summary_rows.append({
                    "plot_group": group,
                    "epsilon": eps,
                    "mean_n_common": mean_val,
                    "sd_n_common": sd_val,
                    "se_n_common": se_val,
                    "n_reps": len(vals),
                    "min_n_common": float(np.min(vals)),
                    "max_n_common": float(np.max(vals)),
                })

                if show_points:
                    x = pos + rng.uniform(-0.018, 0.018, size=len(vals))
                    ax.plot(
                        x,
                        vals,
                        "o",
                        color=color,
                        alpha=0.20,
                        markersize=2.8,
                    )

            if has_data:
                plotted_groups.append(group)

                ax.errorbar(
                    x_positions,
                    mean_values,
                    yerr=error_values,
                    fmt="o-",
                    color=color,
                    linewidth=1.8,
                    markersize=5,
                    capsize=4,
                    elinewidth=1.3,
                    label=labels.get(group, group).replace("\n", " "),
                )

        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(eps) for eps in eps_values])
        ax.set_xlabel(r"$\epsilon$")

    # =========================================================
    # Case 2: one epsilon value
    # x-axis = method, one error bar per method
    # =========================================================
    else:
        positions = np.arange(1, len(method_order) + 1)
        xlabels = []

        for pos, group in zip(positions, method_order):
            color = colors.get(group, "#999999")

            vals = df.loc[
                df["plot_group"] == group,
                "n_common"
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            plotted_groups.append(group)
            xlabels.append(labels.get(group, group))

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            if errorbar_type == "sd":
                err_val = sd_val
            elif errorbar_type == "se":
                err_val = se_val
            else:
                raise ValueError("errorbar_type must be either 'sd' or 'se'.")

            summary_rows.append({
                "plot_group": group,
                "epsilon": eps_values[0] if len(eps_values) > 0 else np.nan,
                "mean_n_common": mean_val,
                "sd_n_common": sd_val,
                "se_n_common": se_val,
                "n_reps": len(vals),
                "min_n_common": float(np.min(vals)),
                "max_n_common": float(np.max(vals)),
            })

            if show_points:
                jitter = rng.uniform(-0.06, 0.06, size=len(vals))
                ax.plot(
                    pos + jitter,
                    vals,
                    "o",
                    color=color,
                    alpha=0.20,
                    markersize=2.8,
                )

            ax.errorbar(
                [pos],
                [mean_val],
                yerr=[err_val],
                fmt="o",
                color=color,
                linewidth=1.8,
                markersize=6,
                capsize=5,
                elinewidth=1.4,
                label=labels.get(group, group).replace("\n", " "),
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Selection method")

    # --------------------------------------------------
    # Perfect overlap reference line
    # --------------------------------------------------
    ax.axhline(
        max_common,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Perfect overlap = {max_common}",
    )

    ax.set_ylabel(r"Number of common elements $|E^* \cap \widehat E|$")

    if title is None:
        title = r"Selection quality by overlap with true winners"

    if errorbar_type == "sd":
        ax.set_title(f"{title}: mean ± SD across simulations")
    else:
        ax.set_title(f"{title}: mean ± SE across simulations")

    if ylim is None:
        ax.set_ylim(-0.1, max_common + 0.5)
    else:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=colors.get(g, "#999999"),
            marker="o",
            linewidth=1.8,
            label=labels.get(g, g).replace("\n", " "),
        )
        for g in plotted_groups
    ]

    legend_handles.append(
        plt.Line2D(
            [0], [0],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Perfect overlap = {max_common}",
        )
    )

    ax.legend(handles=legend_handles, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_rows)

    print("True winner set E*:", E_star)
    print(
        summary_df.groupby("plot_group")[
            ["mean_n_common", "sd_n_common", "se_n_common", "n_reps"]
        ].first()
    )

    if return_summary:
        return {
            "overlap_df": df,
            "summary_df": summary_df,
        }

    return df



import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _canonical_subset_safe(S):
    """
    Robust canonical subset converter.
    Works for tuple/list/np.ndarray and string like '(2, 3, 7)'.
    """
    if S is None:
        return None

    try:
        if pd.isna(S):
            return None
    except Exception:
        pass

    if isinstance(S, str):
        try:
            S = ast.literal_eval(S)
        except Exception:
            return None

    try:
        return tuple(sorted(int(x) for x in S))
    except Exception:
        return None


def _ensure_setting_columns_for_index_plot(df, compare_out):
    """
    Single-setting compare_out may not have setting_id/setting_label.
    Multi-setting compare_out should already have them.
    """
    df = df.copy()

    if "setting_id" not in df.columns:
        df["setting_id"] = 0

    if "setting_label" not in df.columns:
        if "settings_df" in compare_out and len(compare_out["settings_df"]) == 1:
            df["setting_label"] = str(compare_out["settings_df"]["setting_label"].iloc[0])
        else:
            df["setting_label"] = "setting_0"

    return df


def _filter_epsilon_for_index_plot(df, *, epsilon_to_plot=None, name="dataframe"):
    """
    For multi-setting figure, x-axis is parameter setting, so plot one epsilon at a time.
    """
    df = df.copy()

    if "epsilon" not in df.columns:
        return df

    eps_num = pd.to_numeric(df["epsilon"], errors="coerce")
    eps_values = sorted(eps_num.dropna().astype(float).unique())

    if epsilon_to_plot is not None:
        df = df[np.isclose(eps_num, float(epsilon_to_plot))].copy()
        return df

    if len(eps_values) > 1:
        raise ValueError(
            f"{name} contains multiple epsilon values: {eps_values}. "
            "Since x-axis is parameter setting, please specify epsilon_to_plot, "
            "for example epsilon_to_plot=5."
        )

    return df


def _get_x_values_for_index_plot(summary_df, compare_out, x_axis):
    """
    Preserve setting order from compare_out['settings_df'] when possible.
    """
    if x_axis not in summary_df.columns:
        raise ValueError(
            f"x_axis='{x_axis}' is not in summary_df. "
            f"Available columns are {list(summary_df.columns)}."
        )

    if "settings_df" in compare_out and x_axis in compare_out["settings_df"].columns:
        setting_order = compare_out["settings_df"][x_axis].tolist()
        existing = summary_df[x_axis].dropna().unique().tolist()
        x_values = [x for x in setting_order if x in existing]
        extra = [x for x in existing if x not in x_values]
        return x_values + extra

    x_values = summary_df[x_axis].dropna().unique().tolist()

    try:
        return sorted(x_values)
    except Exception:
        return x_values


def summarize_conditional_index_coverage_gaussian(
    compare_out,
    j,
    *,
    epsilon_to_plot=None,
    min_selected=1,
    x_axis="setting_label",
    verbose=True,
):
    """
    Conditional coverage for a fixed selected index j.

    Multi-setting version.

    For each setting/method/epsilon group, computes

        coverage(j | j selected)
        = (# times CI for j covers truth) / (# times j is selected).

    Denominator:
        compare_out["all_subset_df"]

    Numerator:
        compare_out["all_ci_df"]

    If j is selected but no CI row is produced, this counts as not covered.
    """

    if "all_ci_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_ci_df'.")
    if "all_subset_df" not in compare_out:
        raise ValueError("compare_out must contain 'all_subset_df'.")

    j = int(j)

    all_ci_df = compare_out["all_ci_df"].copy()
    all_subset_df = compare_out["all_subset_df"].copy()

    all_ci_df = _ensure_setting_columns_for_index_plot(all_ci_df, compare_out)
    all_subset_df = _ensure_setting_columns_for_index_plot(all_subset_df, compare_out)

    required_ci_cols = {
        "setting_id", "setting_label",
        "method", "epsilon", "rep", "idx", "covered", "length",
    }
    missing_ci = required_ci_cols - set(all_ci_df.columns)
    if missing_ci:
        raise ValueError(f"all_ci_df is missing columns: {missing_ci}")

    required_subset_cols = {
        "setting_id", "setting_label",
        "method", "epsilon", "rep", "selected_subset",
    }
    missing_subset = required_subset_cols - set(all_subset_df.columns)
    if missing_subset:
        raise ValueError(f"all_subset_df is missing columns: {missing_subset}")

    all_ci_df = _filter_epsilon_for_index_plot(
        all_ci_df,
        epsilon_to_plot=epsilon_to_plot,
        name="all_ci_df",
    )
    all_subset_df = _filter_epsilon_for_index_plot(
        all_subset_df,
        epsilon_to_plot=epsilon_to_plot,
        name="all_subset_df",
    )

    if all_subset_df.empty:
        return pd.DataFrame(columns=[
            "setting_id", "setting_label", "method", "epsilon",
            "n_selected", "n_covered", "n_ci_rows",
            "conditional_coverage", "avg_length", "sd_length",
            "coverage_se", "coverage_sd", "length_se", "length_sd",
        ])

    all_subset_df["selected_subset"] = all_subset_df["selected_subset"].apply(
        _canonical_subset_safe
    )

    all_subset_df["j_selected"] = all_subset_df["selected_subset"].apply(
        lambda S: False if S is None else (j in S)
    )

    selected_df = all_subset_df[all_subset_df["j_selected"]].copy()

    if selected_df.empty:
        if verbose:
            print(f"Index j={j} was never selected after epsilon filtering.")
        return pd.DataFrame(columns=[
            "setting_id", "setting_label", "method", "epsilon",
            "n_selected", "n_covered", "n_ci_rows",
            "conditional_coverage", "avg_length", "sd_length",
            "coverage_se", "coverage_sd", "length_se", "length_sd",
        ])

    group_cols = ["setting_id", "setting_label", "method", "epsilon"]

    # Denominator: number of replications where j was selected.
    denom_df = (
        selected_df
        .groupby(group_cols, as_index=False)
        .agg(n_selected=("rep", "nunique"))
    )

    # Numerator: CI rows for selected index j.
    j_ci_df = all_ci_df[all_ci_df["idx"].astype(int) == j].copy()

    selected_keys = selected_df[group_cols + ["rep"]].drop_duplicates()

    j_ci_selected_df = j_ci_df.merge(
        selected_keys,
        on=group_cols + ["rep"],
        how="inner",
    )

    numer_df = (
        j_ci_selected_df
        .groupby(group_cols, as_index=False)
        .agg(
            n_covered=("covered", "sum"),
            n_ci_rows=("covered", "size"),
            avg_length=("length", "mean"),
            sd_length=("length", "std"),
        )
    )

    out = denom_df.merge(
        numer_df,
        on=group_cols,
        how="left",
    )

    out["n_covered"] = out["n_covered"].fillna(0).astype(int)
    out["n_ci_rows"] = out["n_ci_rows"].fillna(0).astype(int)

    out["conditional_coverage"] = out["n_covered"] / out["n_selected"]
    out["ci_missing_count"] = out["n_selected"] - out["n_ci_rows"]
    out["ci_success_rate_given_selected"] = out["n_ci_rows"] / out["n_selected"]

    # Error bars.
    out["coverage_se"] = np.sqrt(
        out["conditional_coverage"]
        * (1.0 - out["conditional_coverage"])
        / out["n_selected"].clip(lower=1)
    )

    out["coverage_sd"] = np.sqrt(
        out["conditional_coverage"]
        * (1.0 - out["conditional_coverage"])
    )

    out["length_se"] = (
        out["sd_length"]
        / np.sqrt(out["n_ci_rows"].clip(lower=1))
    )
    out["length_sd"] = out["sd_length"]

    out = out[out["n_selected"] >= int(min_selected)].copy()

    sort_cols = []
    if x_axis in out.columns:
        sort_cols.append(x_axis)
    sort_cols += ["epsilon", "method"]

    out = out.sort_values(sort_cols).reset_index(drop=True)

    if verbose:
        print(f"\n===== Conditional coverage for selected index j = {j} =====")
        if epsilon_to_plot is not None:
            print(f"epsilon_to_plot = {epsilon_to_plot}")
        print(out)

    return out


def plot_conditional_index_coverage_and_length_gaussian(
    compare_out,
    j,
    *,
    epsilon_to_plot=None,
    x_axis="setting_label",
    min_selected=1,
    method_order=None,
    method_labels=None,
    colors=None,
    marker_map=None,
    nominal=0.95,
    coverage_figsize=(7.2, 4.2),
    length_figsize=(7.2, 4.2),
    coverage_ylim=(0.7, 1.02),
    length_ylim=None,
    box_width=0.22,
    errorbar_type="se",   # "se", "sd", or None
    connect_lines=False,
    hide_xtick_labels=False,
    rotate_xticks=30,
    reference_line_color="#D62728",
    verbose=True,
):
    """
    Plot conditional coverage and average CI length for a fixed selected index j.

    Multi-setting version:
        - x-axis is parameter setting, e.g. setting_label.
        - epsilon_to_plot fixes one epsilon value.
        - default method_order excludes Standard, matching the main figure.
    """

    summary_df = summarize_conditional_index_coverage_gaussian(
        compare_out,
        j=j,
        epsilon_to_plot=epsilon_to_plot,
        min_selected=min_selected,
        x_axis=x_axis,
        verbose=verbose,
    )

    if summary_df.empty:
        print(f"No method/setting group selected j={j} at least {min_selected} times.")
        return summary_df

    if method_order is None:
        method_order = [
            "Randomized PSI",
            "Polyhedral PSI",
            "Data Splitting",
            "Zoom Correction",
        ]
    else:
        method_order = [m for m in method_order if m in summary_df["method"].unique()]

    if method_labels is None:
        method_labels = {
            "Standard": "Standard",
            "Randomized PSI": "Randomized PSI",
            "Polyhedral PSI": "Polyhedral PSI",
            "Data Splitting": "Data Splitting",
            "Zoom Correction": "Zoom Correction",
        }

    if colors is None:
        colors = {
            "Standard": "#7A7A7A",
            "Randomized PSI": "#1F77B4",
            "Polyhedral PSI": "#9467BD",
            "Data Splitting": "#2CA02C",
            "Zoom Correction": "#E6550D",
        }

    if marker_map is None:
        marker_map = {
            "Standard": "s",
            "Randomized PSI": "^",
            "Polyhedral PSI": "s",
            "Data Splitting": "^",
            "Zoom Correction": "s",
        }

    if errorbar_type not in ["se", "sd", None]:
        raise ValueError("errorbar_type must be 'se', 'sd', or None.")

    x_values = _get_x_values_for_index_plot(
        summary_df,
        compare_out,
        x_axis=x_axis,
    )

    base_positions = np.arange(len(x_values))

    if len(method_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(method_order) - 1) / 2,
            box_width * spacing_scale * (len(method_order) - 1) / 2,
            len(method_order),
        )

    def _draw_panel(ax, *, y_col, se_col, sd_col, title, nominal_line=None):
        plotted_methods = []

        for jj, method in enumerate(method_order):
            color = colors.get(method, "#999999")
            marker = marker_map.get(method, "o")

            x_positions = []
            y_values = []
            y_errors = []

            for ii, x_val in enumerate(x_values):
                row = summary_df[
                    (summary_df["method"] == method)
                    & (summary_df[x_axis] == x_val)
                ]

                if row.empty:
                    continue

                y_val = row[y_col].iloc[0]
                if pd.isna(y_val):
                    continue

                if errorbar_type == "se":
                    err = row[se_col].iloc[0]
                elif errorbar_type == "sd":
                    err = row[sd_col].iloc[0]
                else:
                    err = 0.0

                if pd.isna(err):
                    err = 0.0

                x_positions.append(base_positions[ii] + offsets[jj])
                y_values.append(float(y_val))
                y_errors.append(float(err))

            if len(y_values) == 0:
                continue

            plotted_methods.append(method)

            if errorbar_type is None:
                ax.plot(
                    x_positions,
                    y_values,
                    linestyle="-" if connect_lines else "None",
                    marker=marker,
                    color=color,
                    linewidth=1.5,
                    markersize=8.0,
                    label=method_labels.get(method, method),
                )
            else:
                fmt = f"{marker}-" if connect_lines else marker
                ax.errorbar(
                    x_positions,
                    y_values,
                    yerr=y_errors,
                    fmt=fmt,
                    color=color,
                    linewidth=1.5,
                    markersize=8.0,
                    capsize=4,
                    elinewidth=1.2,
                    label=method_labels.get(method, method),
                )

        if nominal_line is not None:
            ax.axhline(
                nominal_line,
                linestyle="--",
                linewidth=1.3,
                color=reference_line_color,
                label=f"Nominal = {nominal_line:.2f}",
            )

        ax.set_xticks(base_positions)

        if hide_xtick_labels:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xticklabels([str(x) for x in x_values], rotation=rotate_xticks)
            ax.set_xlabel("Parameter setting")

        # Following Snigdha's requested style:
        # y-axis label removed; metric is in the panel title.
        ax.set_ylabel("")
        ax.set_title(title)

        ax.grid(axis="y", alpha=0.25)

        return plotted_methods

    # ---------------------------
    # Coverage plot
    # ---------------------------
    fig_cov, ax_cov = plt.subplots(figsize=coverage_figsize)

    if errorbar_type == "se":
        coverage_title = rf"Conditional coverage rate for selected index $j={j}$, mean $\pm$ SE"
    elif errorbar_type == "sd":
        coverage_title = rf"Conditional coverage rate for selected index $j={j}$, mean $\pm$ SD"
    else:
        coverage_title = rf"Conditional coverage rate for selected index $j={j}$"

    _draw_panel(
        ax_cov,
        y_col="conditional_coverage",
        se_col="coverage_se",
        sd_col="coverage_sd",
        title=coverage_title,
        nominal_line=nominal,
    )

    if coverage_ylim is not None:
        ax_cov.set_ylim(*coverage_ylim)

    ax_cov.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Length plot
    # ---------------------------
    fig_len, ax_len = plt.subplots(figsize=length_figsize)

    if errorbar_type == "se":
        length_title = rf"Average CI length conditional on selecting $j={j}$, mean $\pm$ SE"
    elif errorbar_type == "sd":
        length_title = rf"Average CI length conditional on selecting $j={j}$, mean $\pm$ SD"
    else:
        length_title = rf"Average CI length conditional on selecting $j={j}$"

    _draw_panel(
        ax_len,
        y_col="avg_length",
        se_col="length_se",
        sd_col="length_sd",
        title=length_title,
        nominal_line=None,
    )

    if length_ylim is not None:
        ax_len.set_ylim(*length_ylim)

    ax_len.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.show()

    return summary_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _filter_epsilon_for_parameter_setting_axis(df, *, epsilon_to_plot=None, name="dataframe"):
    """
    Since x-axis is parameter setting, we should plot one epsilon at a time.
    """
    df = df.copy()

    if "epsilon" not in df.columns:
        return df

    eps_values = sorted(df["epsilon"].dropna().astype(float).unique())

    if epsilon_to_plot is not None:
        df = df[np.isclose(df["epsilon"].astype(float), float(epsilon_to_plot))].copy()
        return df

    if len(eps_values) > 1:
        raise ValueError(
            f"{name} contains multiple epsilon values: {eps_values}. "
            "Since x-axis is parameter setting, please specify epsilon_to_plot, "
            "e.g. epsilon_to_plot=10."
        )

    return df


def _get_x_values_for_parameter_settings(df, compare_out, x_axis):
    """
    Preserve the order from compare_out['settings_df'] when possible.
    """

    if x_axis not in df.columns:
        raise ValueError(
            f"x_axis='{x_axis}' is not in dataframe. "
            f"Available columns are {list(df.columns)}."
        )

    if x_axis == "setting_label" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_label"].tolist()
        x_values = [x for x in setting_order if x in df[x_axis].dropna().unique()]
        extra = [
            x for x in df[x_axis].dropna().unique().tolist()
            if x not in x_values
        ]
        return x_values + extra

    if x_axis == "setting_id" and "settings_df" in compare_out:
        setting_order = compare_out["settings_df"]["setting_id"].tolist()
        x_values = [x for x in setting_order if x in df[x_axis].dropna().unique()]
        extra = [
            x for x in df[x_axis].dropna().unique().tolist()
            if x not in x_values
        ]
        return x_values + extra

    x_values = df[x_axis].dropna().unique().tolist()

    try:
        return sorted(x_values)
    except Exception:
        return list(x_values)


from itertools import combinations


def _compute_snoise_from_score_vec(score_vec, k, *, snoise_ridge=1e-12):
    """
    Compute

        s_noise(t) =
        sqrt( mean_E (s_E(t) - s_bar(t))^2 + snoise_ridge^2 )

    where s_E(t) = sum_{j in E} score_vec[j].
    """

    score_vec = np.asarray(score_vec, dtype=float).reshape(-1)
    M = score_vec.size
    k = int(k)

    if not (1 <= k <= M):
        raise ValueError(f"k must be in [1, M], got k={k}, M={M}.")

    subset_scores = np.array(
        [
            np.sum(score_vec[list(E)])
            for E in combinations(range(M), k)
        ],
        dtype=float,
    )

    s_bar = float(np.mean(subset_scores))

    snoise = float(
        np.sqrt(
            np.mean((subset_scores - s_bar) ** 2)
            + float(snoise_ridge) ** 2
        )
    )

    return snoise


def _attach_snoise_to_regret_df(
    regret_df,
    compare_out,
    *,
    utility_fn=None,
    snoise_ridge=1e-12,
):
    """
    Attach s_noise(T) to regret_df using stored all_stat_df.

    No recomputation. No tree refitting.
    """
    regret_df = regret_df.copy()

    if regret_df.empty:
        regret_df["snoise"] = np.nan
        return regret_df

    if "all_stat_df" not in compare_out or compare_out["all_stat_df"] is None:
        raise ValueError(
            "compare_out must contain all_stat_df. "
            "Please record stat_df during simulation."
        )

    stat_df = compare_out["all_stat_df"].copy()

    if stat_df.empty:
        raise ValueError("compare_out['all_stat_df'] is empty.")

    required_cols = {"method", "epsilon", "rep", "role", "idx", "score_hat"}
    missing = required_cols - set(stat_df.columns)
    if missing:
        raise ValueError(f"all_stat_df is missing columns: {missing}")

    # For selection quality, use the statistic used for selection.
    stat_df = stat_df[stat_df["role"] == "selection"].copy()

    # Match epsilon.
    if "epsilon" in regret_df.columns and "epsilon" in stat_df.columns:
        regret_eps = pd.to_numeric(regret_df["epsilon"], errors="coerce")
        stat_df["epsilon"] = pd.to_numeric(stat_df["epsilon"], errors="coerce")

    # Build grouping columns.
    merge_cols = ["method", "epsilon", "rep"]

    if "setting_id" in regret_df.columns and "setting_id" in stat_df.columns:
        merge_cols = ["setting_id"] + merge_cols
    elif "setting_label" in regret_df.columns and "setting_label" in stat_df.columns:
        merge_cols = ["setting_label"] + merge_cols

    k = int(compare_out["k"])

    snoise_rows = []

    for key, g in stat_df.groupby(merge_cols, dropna=False):
        g = g.sort_values("idx")
        score_vec = g["score_hat"].to_numpy(dtype=float)

        snoise_b = _compute_snoise_from_score_vec(
            score_vec,
            k,
            snoise_ridge=snoise_ridge,
        )

        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(merge_cols, key)}
        row["snoise"] = float(snoise_b)
        snoise_rows.append(row)

    snoise_df = pd.DataFrame(snoise_rows)

    regret_df = regret_df.merge(
        snoise_df[merge_cols + ["snoise"]],
        on=merge_cols,
        how="left",
    )

    if regret_df["snoise"].isna().any():
        missing_n = int(regret_df["snoise"].isna().sum())
        raise ValueError(
            f"snoise merge failed for {missing_n} rows. "
            "Check method / epsilon / rep / setting alignment."
        )

    return regret_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _plot_grouped_errorbar_panel(
    ax,
    df,
    *,
    y_col,
    group_col,
    x_axis,
    x_values,
    group_order,
    group_labels,
    colors,
    ylabel,
    title,
    nominal=None,
    ylim=None,
    box_width=0.20,
    errorbar_type="se",
    show_points=False,
    hide_xtick_labels=False,
    rotate_xticks=0,
    show_xlabel=True,
    connect_lines=False,
    marker_map=None,
    add_nominal_to_legend=True,
    nominal_label=None,
    reference_line_color="black",
):
    """
    Internal helper: plot mean +/- SE or SD for one panel.

    Main changes:
        1. marker_map controls marker shape by group.
        2. add_nominal_to_legend controls whether reference line appears in legend.
    """

    if marker_map is None:
        marker_map = {}

    if df.empty:
        ax.set_title(title)
        ax.text(
            0.5, 0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return pd.DataFrame(), [], []

    if y_col not in df.columns:
        raise ValueError(f"'{y_col}' is not in dataframe columns.")

    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' is not in dataframe columns.")

    group_order = [g for g in group_order if g in df[group_col].dropna().unique()]

    base_positions = np.arange(len(x_values))

    if len(group_order) == 1:
        offsets = np.array([0.0])
    else:
        spacing_scale = 0.45
        offsets = np.linspace(
            -box_width * spacing_scale * (len(group_order) - 1) / 2,
            box_width * spacing_scale * (len(group_order) - 1) / 2,
            len(group_order),
        )

    rng = np.random.default_rng(123)

    summary_rows = []
    plotted_groups = []
    all_x_positions = []

    for j, group in enumerate(group_order):
        color = colors.get(group, "#999999")
        marker = marker_map.get(group, "o")

        x_positions = []
        mean_values = []
        error_values = []

        has_data = False

        for i, x_val in enumerate(x_values):
            vals = df.loc[
                (df[group_col] == group) &
                (df[x_axis] == x_val),
                y_col
            ].dropna().to_numpy()

            if len(vals) == 0:
                continue

            has_data = True

            pos = base_positions[i] + offsets[j]
            all_x_positions.append(pos)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = float(sd_val / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

            if errorbar_type == "se":
                err_val = se_val
            elif errorbar_type == "sd":
                err_val = sd_val
            else:
                raise ValueError("errorbar_type must be either 'se' or 'sd'.")

            x_positions.append(pos)
            mean_values.append(mean_val)
            error_values.append(err_val)

            summary_rows.append({
                x_axis: x_val,
                group_col: group,
                "mean": mean_val,
                "sd": sd_val,
                "se": se_val,
                "n_reps": int(len(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })

            if "epsilon" in df.columns:
                eps_unique = sorted(df["epsilon"].dropna().astype(float).unique())
                if len(eps_unique) == 1:
                    summary_rows[-1]["epsilon"] = float(eps_unique[0])

            if show_points:
                x_jitter = pos + rng.uniform(-0.018, 0.018, size=len(vals))
                ax.plot(
                    x_jitter,
                    vals,
                    linestyle="None",
                    marker=marker,
                    color=color,
                    alpha=0.18,
                    markersize=2.8,
                )

        if has_data:
            plotted_groups.append(group)

            fmt = f"{marker}-" if connect_lines else marker

            ax.errorbar(
                x_positions,
                mean_values,
                yerr=error_values,
                fmt=fmt,
                color=color,
                linewidth=1.6,
                markersize=5.5,
                capsize=5,
                elinewidth=1.6, #1.2
                label=group_labels.get(group, group),
            )

    if nominal is not None:
        ax.axhline(
            nominal,
            linestyle="--",
            linewidth=1.2,
            color=reference_line_color,
        )

    if len(all_x_positions) > 0:
        ax.set_xlim(min(all_x_positions) - 0.15, max(all_x_positions) + 0.15)

    ax.set_xticks(base_positions)

    if hide_xtick_labels:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xticklabels([str(x) for x in x_values], rotation=rotate_xticks)
        ax.set_xlabel("Parameter setting" if show_xlabel else "")

    #ax.set_ylabel(ylabel)
    ax.set_ylabel("")
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25)

    handles = [
        Line2D(
            [0], [0],
            color=colors.get(g, "#999999"),
            marker=marker_map.get(g, "o"),
            linewidth=1.6,
            linestyle="-" if connect_lines else "None",
            label=group_labels.get(g, g),
        )
        for g in plotted_groups
    ]

    labels = [group_labels.get(g, g) for g in plotted_groups]

    if nominal is not None and add_nominal_to_legend:
        if nominal_label is None:
            nominal_label = f"Nominal = {nominal:.2f}"

        handles.append(
            Line2D(
                [0], [0],
                color=reference_line_color,
                linestyle="--",
                linewidth=1.2,
                label=nominal_label,
            )
        )
        labels.append(nominal_label)

    return pd.DataFrame(summary_rows), handles, labels


def plot_four_panel_selection_inference_summary(
    compare_out,
    *,
    layout="1x4",   # "1x4" or "2x2"

    # epsilon / x-axis
    epsilon_to_plot=None,
    x_axis="setting_label",
    hide_xtick_labels=False,
    rotate_xticks=0,

    # conditional coverage event
    subset_rank=1,
    selected_subset=None,
    min_conditional_reps=1,
    complete_only=True,
    verbose=False,
    conditional_type="subset",   # "subset" or "index"
    conditional_index=None,

    # methods
    ci_method_order=None,
    regret_method_order=None,
    method_labels=None,
    regret_labels=None,
    colors=None,
    regret_colors=None,
    marker_map=None,
    regret_marker_map=None,

    # style
    nominal=0.95,
    errorbar_type="se",
    box_width=0.20,
    show_points_regret=False,
    show_points_marginal_coverage=False,
    show_points_conditional_coverage=False,
    show_points_length=False,
    connect_lines=False,

    # y limits
    regret_ylim=None,
    marginal_coverage_ylim=(0.0, 1.02),
    conditional_coverage_ylim=(0.0, 1.02),
    length_ylim=None,

    # figure
    figsize=None,
    legend_mode="bottom",   # "right", "bottom", "each", or "none"
    legend_ncol=None,
    suptitle=None,

    # standardized noise
    normalize_regret_by_snoise=True,
    snoise_ridge=1e-12,

    # font sizes
    title_fontsize=22,
    tick_fontsize=19,
    axis_label_fontsize=20,
    legend_fontsize=19,
    suptitle_fontsize=22,
):
    if layout not in {"1x4", "2x2"}:
        raise ValueError("layout must be either '1x4' or '2x2'.")

    # --------------------------------------------------
    # 1. Method order
    # --------------------------------------------------
    if ci_method_order is None:
        ci_method_order = [
            "Randomized PSI",
            "Polyhedral PSI",
            "Data Splitting",
            "Zoom Correction",
        ]
    else:
        ci_method_order = [m for m in ci_method_order if m != "Standard"]

    if regret_method_order is None:
        regret_method_order = [
            "Full-data argmax",
            "Half-data argmax",
            "Randomized PSI",
        ]

    # --------------------------------------------------
    # 2. Labels
    # --------------------------------------------------
    if method_labels is None:
        method_labels = {
            "Standard": "Standard",
            "Randomized PSI": "Randomized PSI",
            "Polyhedral PSI": "Polyhedral PSI",
            "Data Splitting": "Data Splitting",
            "Zoom Correction": "Zoom Correction",
        }

    if regret_labels is None:
        regret_labels = {
            "Full-data argmax": "Standard",
            "Half-data argmax": "Data Splitting",
            "Randomized PSI": "Randomized PSI",
        }

    # --------------------------------------------------
    # 3. Colors
    # --------------------------------------------------
    if colors is None:
        colors = {
            "Standard": "#7A7A7A",
            "Randomized PSI": "#1F77B4",
            "Polyhedral PSI": "#9467BD",
            "Data Splitting": "#2CA02C",
            "Zoom Correction": "#E6550D",
        }

    if regret_colors is None:
        regret_colors = {
            "Full-data argmax": "#7A7A7A",
            "Half-data argmax": "#2CA02C",
            "Randomized PSI": "#1F77B4",
        }

    # --------------------------------------------------
    # 4. Markers
    # --------------------------------------------------
    if marker_map is None:
        marker_map = {
            "Standard": "s",
            "Randomized PSI": "^",
            "Polyhedral PSI": "s",
            "Data Splitting": "^",
            "Zoom Correction": "s",
        }

    if regret_marker_map is None:
        regret_marker_map = {
            "Full-data argmax": "s",
            "Half-data argmax": "^",
            "Randomized PSI": "^",
        }

    # --------------------------------------------------
    # 5. Build regret / selection-quality data
    # --------------------------------------------------
    regret_df = build_selection_utility_df_from_compare(
        compare_out,
        epsilon_to_plot=epsilon_to_plot,
        deduplicate=True,
    )

    regret_df = _filter_epsilon_for_parameter_setting_axis(
        regret_df,
        epsilon_to_plot=epsilon_to_plot,
        name="regret_df",
    )

    # Full-data argmax is the reference, so its regret is exactly zero.
    regret_df.loc[
        regret_df["plot_group"] == "Full-data argmax",
        "regret"
    ] = 0.0

    regret_is_normalized = False

    if normalize_regret_by_snoise:
        try:
            regret_df = _attach_snoise_to_regret_df(
                regret_df,
                compare_out,
                utility_fn=None,
                snoise_ridge=snoise_ridge,
            )

            regret_df["raw_regret"] = regret_df["regret"]
            regret_df["regret"] = regret_df["raw_regret"] / regret_df["snoise"]

            regret_df.loc[
                regret_df["plot_group"] == "Full-data argmax",
                "regret"
            ] = 0.0

            regret_is_normalized = True

        except (ValueError, KeyError) as e:
            print(
                f"[4-panel] snoise normalization unavailable ({e}); "
                f"showing raw selection-quality regret."
            )

    # --------------------------------------------------
    # 6. Build marginal coverage / marginal length data
    # --------------------------------------------------
    marginal_df = get_rep_level_coverage_df(
        compare_out,
        complete_only=complete_only,
    )

    marginal_df = _filter_epsilon_for_parameter_setting_axis(
        marginal_df,
        epsilon_to_plot=epsilon_to_plot,
        name="marginal_df",
    )

    marginal_df = marginal_df[
        marginal_df["method"].isin(ci_method_order)
    ].copy()

    # Length panel only: remove Polyhedral PSI
    length_df = marginal_df[
        marginal_df["method"] != "Polyhedral PSI"
    ].copy()

    # --------------------------------------------------
    # 7. Build conditional coverage data
    # --------------------------------------------------
    if conditional_type == "subset":

        conditional_out = make_conditional_compare_out_by_modal_subset(
            compare_out,
            subset_rank=subset_rank,
            selected_subset=selected_subset,
            min_conditional_reps=min_conditional_reps,
            complete_only=complete_only,
            verbose=verbose,
        )
    
        conditional_df = conditional_out["all_rep_df"].copy()
    
    elif conditional_type == "index":
    
        conditional_df = summarize_conditional_index_coverage_gaussian(
            compare_out,
            j=conditional_index,
            epsilon_to_plot=epsilon_to_plot,
            x_axis=x_axis,
            min_selected=min_conditional_reps,
            verbose=verbose,
        )
    
        conditional_df = conditional_df.rename(
            columns={
                "conditional_coverage": "coverage_rate",
            }
        )
        conditional_out = {
            "all_rep_df": conditional_df.copy(),
            "conditional_type": "index",
            "conditional_index": int(conditional_index),
        }
    
    else:
        raise ValueError(...)
    
        conditional_df = _filter_epsilon_for_parameter_setting_axis(
            conditional_df,
            epsilon_to_plot=epsilon_to_plot,
            name="conditional_df",
        )
    
        conditional_df = conditional_df[
            conditional_df["method"].isin(ci_method_order)
        ].copy()

    # --------------------------------------------------
    # 8. X-axis values
    # --------------------------------------------------
    x_values = _get_x_values_for_parameter_settings(
        marginal_df,
        compare_out,
        x_axis,
    )

    # --------------------------------------------------
    # 9. Figure layout
    # Panel order:
    #   1. Selection quality
    #   2. Marginal coverage rate
    #   3. Average interval width
    #   4. Conditional coverage rate
    # --------------------------------------------------
    if figsize is None:
        if layout == "1x4":
            figsize = (20, 5)
        else:
            figsize = (11.5, 8.5)

    if layout == "1x4":
        fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=False)
        axes = np.asarray(axes).reshape(-1)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
        axes = np.asarray(axes).reshape(-1)

    all_handles = []
    all_labels = []

    def _collect_legend(handles, labels):
        for h, lab in zip(handles, labels):
            if lab not in all_labels:
                all_handles.append(h)
                all_labels.append(lab)

    def _style_axis(ax):
        ax.title.set_fontsize(title_fontsize)
        ax.title.set_fontweight("normal")

        ax.tick_params(axis="both", labelsize=tick_fontsize)

        ax.xaxis.label.set_size(axis_label_fontsize)
        ax.yaxis.label.set_size(axis_label_fontsize)
        ax.xaxis.label.set_fontweight("normal")
        ax.yaxis.label.set_fontweight("normal")

        for tick in ax.get_xticklabels():
            tick.set_fontsize(tick_fontsize)
            tick.set_fontweight("normal")

        for tick in ax.get_yticklabels():
            tick.set_fontsize(tick_fontsize)
            tick.set_fontweight("normal")

    # --------------------------------------------------
    # Panel 1: Selection quality / regret
    # --------------------------------------------------
    regret_summary, handles, labels_ = _plot_grouped_errorbar_panel(
        axes[0],
        regret_df,
        y_col="regret",
        group_col="plot_group",
        x_axis=x_axis,
        x_values=x_values,
        group_order=regret_method_order,
        group_labels=regret_labels,
        colors=regret_colors,
        ylabel="",
        title="(a) Selection quality",
        nominal=0,
        ylim=regret_ylim,
        box_width=box_width,
        errorbar_type=errorbar_type,
        show_points=show_points_regret,
        hide_xtick_labels=hide_xtick_labels,
        rotate_xticks=rotate_xticks,
        show_xlabel=True,
        connect_lines=connect_lines,
        marker_map=regret_marker_map,
        add_nominal_to_legend=False,
        reference_line_color="#666666",
    )
    _collect_legend(handles, labels_)

    # --------------------------------------------------
    # Panel 2: Marginal coverage
    # --------------------------------------------------
    marginal_coverage_summary, handles, labels_ = _plot_grouped_errorbar_panel(
        axes[1],
        marginal_df,
        y_col="coverage_rate",
        group_col="method",
        x_axis=x_axis,
        x_values=x_values,
        group_order=ci_method_order,
        group_labels=method_labels,
        colors=colors,
        ylabel="",
        title="(b) Marginal coverage",
        nominal=nominal,
        ylim=marginal_coverage_ylim,
        box_width=box_width,
        errorbar_type=errorbar_type,
        show_points=show_points_marginal_coverage,
        hide_xtick_labels=hide_xtick_labels,
        rotate_xticks=rotate_xticks,
        show_xlabel=True,
        connect_lines=connect_lines,
        marker_map=marker_map,
        add_nominal_to_legend=True,
        nominal_label=f"Nominal = {nominal:.2f}",
        reference_line_color="#D62728",
    )
    _collect_legend(handles, labels_)

    # --------------------------------------------------
    # Panel 3: Average interval width
    # --------------------------------------------------


    for setting in x_values:

        df_s = marginal_df[
            marginal_df[x_axis] == setting
        ]
    
        poly = df_s.loc[
            df_s["method"] == "Polyhedral PSI",
            "avg_length",
        ].mean()
    
        rand = df_s.loc[
            df_s["method"] == "Randomized PSI",
            "avg_length",
        ].mean()
    
        print(
            f"{setting}: Polyhedral intervals were "
            f"{100 * (poly / rand - 1):.1f}% longer "
            f"than the proposed Randomized PSI intervals."
        )
    
    length_summary, handles, labels_ = _plot_grouped_errorbar_panel(
        axes[2],
        length_df,
        y_col="avg_length",
        group_col="method",
        x_axis=x_axis,
        x_values=x_values,
        group_order=ci_method_order,
        group_labels=method_labels,
        colors=colors,
        ylabel="",
        title="(c) Interval length",
        nominal=None,
        ylim=length_ylim,
        box_width=box_width,
        errorbar_type=errorbar_type,
        show_points=show_points_length,
        hide_xtick_labels=hide_xtick_labels,
        rotate_xticks=rotate_xticks,
        show_xlabel=True,
        connect_lines=connect_lines,
        marker_map=marker_map,
        add_nominal_to_legend=False,
    )
    _collect_legend(handles, labels_)

    # --------------------------------------------------
    # Panel 4: Conditional coverage
    # --------------------------------------------------
    conditional_coverage_summary, handles, labels_ = _plot_grouped_errorbar_panel(
        axes[3],
        conditional_df,
        y_col="coverage_rate",
        group_col="method",
        x_axis=x_axis,
        x_values=x_values,
        group_order=ci_method_order,
        group_labels=method_labels,
        colors=colors,
        ylabel="",
        title="(d) Conditional coverage",
        nominal=nominal,
        ylim=conditional_coverage_ylim,
        box_width=box_width,
        errorbar_type=errorbar_type,
        show_points=show_points_conditional_coverage,
        hide_xtick_labels=hide_xtick_labels,
        rotate_xticks=rotate_xticks,
        show_xlabel=True,
        connect_lines=connect_lines,
        marker_map=marker_map,
        add_nominal_to_legend=True,
        nominal_label=f"Nominal = {nominal:.2f}",
        reference_line_color="#D62728",
    )
    _collect_legend(handles, labels_)

    # --------------------------------------------------
    # Font styling
    # --------------------------------------------------
    for ax in axes:
        _style_axis(ax)

    # --------------------------------------------------
    # Legend handling
    # --------------------------------------------------
    if legend_mode == "each":
        for ax in axes:
            ax.legend(
                frameon=False,
                fontsize=legend_fontsize,
                loc="best",
            )

    elif legend_mode == "right":
        fig.legend(
            all_handles,
            all_labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=legend_fontsize,
        )

    elif legend_mode == "bottom":
        if legend_ncol is None:
            legend_ncol = min(len(all_labels), 6)

        fig.legend(
            all_handles,
            all_labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            columnspacing=1.4,
            handletextpad=0.5,
        )

    elif legend_mode == "none":
        pass

    else:
        raise ValueError(
            "legend_mode must be 'right', 'bottom', 'each', or 'none'."
        )

    # --------------------------------------------------
    # Suptitle and layout
    # --------------------------------------------------
    if suptitle is not None:
        fig.suptitle(
            suptitle,
            y=1.06,
            fontsize=suptitle_fontsize,
            fontweight="normal",
        )

    if legend_mode == "bottom":
        plt.tight_layout(rect=(0, 0.13, 1, 1))
    elif legend_mode == "right":
        plt.tight_layout(rect=(0, 0, 0.98, 1))
    else:
        plt.tight_layout()

    fig.savefig(
        "example1.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()

    return {
        "fig": fig,
        "axes": axes,

        "regret_df": regret_df,
        "marginal_df": marginal_df,
        "conditional_out": conditional_out,
        "conditional_df": conditional_df,

        "regret_summary": regret_summary,
        "marginal_coverage_summary": marginal_coverage_summary,
        "conditional_coverage_summary": conditional_coverage_summary,
        "length_summary": length_summary,

        "regret_is_normalized": regret_is_normalized,
    }




def plot_three_panel_selection_inference_summary(
    compare_out, *,
    epsilon_to_plot=None,
    save_pdf=None,
    x_axis="setting_label",
    hide_xtick_labels=False,
    rotate_xticks=0,
    complete_only=True,
    nominal=0.95,
    errorbar_type="se",
    box_width=0.20,
    connect_lines=False,
    regret_ylim=None,
    marginal_coverage_ylim=(0., 1.02),
    length_ylim=None,
    figsize=(15, 5),
    legend_mode="bottom",
    legend_ncol=None,
    suptitle=None,
    normalize_regret_by_snoise=True,
    snoise_ridge=1e-12,
    ci_method_order=None,
    regret_method_order=None,
    method_labels=None,
    regret_labels=None,
    colors=None,
    regret_colors=None,
    marker_map=None,
    regret_marker_map=None,
    title_fontsize=18,
    tick_fontsize=15,
    axis_label_fontsize=12,
    legend_fontsize=15,
    suptitle_fontsize=18,
):
    if ci_method_order is None:
        ci_method_order = ["Randomized PSI", "Polyhedral PSI",
                           "Data Splitting", "Zoom Correction"]
    else:
        ci_method_order = [m for m in ci_method_order if m != "Standard"]

    if regret_method_order is None:
        regret_method_order = ["Full-data argmax", "Half-data argmax", "Randomized PSI"]

    if method_labels is None:
        method_labels = {m: m for m in
                         ["Standard", "Randomized PSI", "Polyhedral PSI",
                          "Data Splitting", "Zoom Correction"]}
    if regret_labels is None:
        regret_labels = {"Full-data argmax": "Standard",
                         "Half-data argmax": "Data Splitting",
                         "Randomized PSI": "Randomized PSI"}

    if colors is None:
        colors = {"Standard": "#7A7A7A",
                  "Randomized PSI": "#1F77B4",
                  "Polyhedral PSI": "#9467BD",
                  "Data Splitting": "#2CA02C",
                  "Zoom Correction": "#E6550D"}

    if regret_colors is None:
        regret_colors = {"Full-data argmax": "#7A7A7A",
                         "Half-data argmax": "#2CA02C",
                         "Randomized PSI": "#1F77B4"}

    if marker_map is None:
        marker_map = {"Standard": "s",
                      "Randomized PSI": "^",
                      "Polyhedral PSI": "s",
                      "Data Splitting": "^",
                      "Zoom Correction": "s"}

    if regret_marker_map is None:
        regret_marker_map = {"Full-data argmax": "s",
                             "Half-data argmax": "^",
                             "Randomized PSI": "^"}

    # ---- Panel 1: selection quality / regret -------------------------------
    regret_df = build_selection_utility_df_from_compare(
        compare_out, epsilon_to_plot=epsilon_to_plot, deduplicate=True,
    )
    regret_df = _filter_epsilon_for_parameter_setting_axis(
        regret_df, epsilon_to_plot=epsilon_to_plot, name="regret_df",
    )

    regret_df.loc[regret_df["plot_group"] == "Full-data argmax", "regret"] = 0.0

    regret_is_normalized = False
    if normalize_regret_by_snoise:
        try:
            regret_df = _attach_snoise_to_regret_df(
                regret_df,
                compare_out,
                utility_fn=None,
                snoise_ridge=snoise_ridge,
            )
            regret_df["raw_regret"] = regret_df["regret"]
            regret_df["regret"] = regret_df["raw_regret"] / regret_df["snoise"]
            regret_df.loc[regret_df["plot_group"] == "Full-data argmax", "regret"] = 0.0
            regret_is_normalized = True
        except (ValueError, KeyError) as e:
            print(
                f"[3-panel] snoise normalization unavailable ({e}); "
                f"showing raw selection-quality regret."
            )

    # ---- Panels 2 & 3: marginal coverage + length --------------------------
    marginal_df = get_rep_level_coverage_df(compare_out, complete_only=complete_only)
    marginal_df = _filter_epsilon_for_parameter_setting_axis(
        marginal_df, epsilon_to_plot=epsilon_to_plot, name="marginal_df",
    )
    marginal_df = marginal_df[marginal_df["method"].isin(ci_method_order)].copy()
    length_df = marginal_df[
        marginal_df["method"] != "Polyhedral PSI"
    ].copy()
    x_values = _get_x_values_for_parameter_settings(marginal_df, compare_out, x_axis)
    x_label_map = {
        "Low Signal": "Weak Separation",
        "High Signal": "Strong Separation",
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=False)
    axes = np.asarray(axes).reshape(-1)

    all_handles, all_labels = [], []

    def _collect(handles, labels):
        for h, lab in zip(handles, labels):
            if lab not in all_labels:
                all_handles.append(h)
                all_labels.append(lab)

    def _style_axis(ax):
        ax.title.set_fontsize(title_fontsize)
        ax.title.set_fontweight("normal")
    
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    
        ax.xaxis.label.set_size(axis_label_fontsize)
        ax.yaxis.label.set_size(axis_label_fontsize)
        ax.xaxis.label.set_fontweight("normal")
        ax.yaxis.label.set_fontweight("normal")
    
        for tick in ax.get_xticklabels():
            tick.set_fontsize(tick_fontsize)
            tick.set_fontweight("normal")
    
        for tick in ax.get_yticklabels():
            tick.set_fontsize(tick_fontsize)
            tick.set_fontweight("normal")

    # Panel 1
    regret_summary, h, l = _plot_grouped_errorbar_panel(
        axes[0],
        regret_df,
        y_col="regret",
        group_col="plot_group",
        x_axis=x_axis,
        x_values=x_values,
        group_order=regret_method_order,
        group_labels=regret_labels,
        colors=regret_colors,
        ylabel=("Regret / sd(scores)" if regret_is_normalized else "Regret (raw)"),
        title=("Selection quality" if regret_is_normalized
                else "Selection quality"),
        nominal=0,
        ylim=regret_ylim,
        box_width=box_width,
        errorbar_type=errorbar_type,
        show_points=False,
        hide_xtick_labels=hide_xtick_labels,
        rotate_xticks=rotate_xticks,
        show_xlabel=True,
        connect_lines=connect_lines,
        marker_map=regret_marker_map,
        add_nominal_to_legend=False,
        reference_line_color="#666666",
    )
    _collect(h, l)

    # Panel 2
    marginal_coverage_summary, h, l = _plot_grouped_errorbar_panel(
        axes[1],
        marginal_df,
        y_col="coverage_rate",
        group_col="method",
        x_axis=x_axis,
        x_values=x_values,
        group_order=ci_method_order,
        group_labels=method_labels,
        colors=colors,
        ylabel="",
        title="Marginal coverage",
        nominal=nominal,
        ylim=marginal_coverage_ylim,
        box_width=box_width,
        errorbar_type=errorbar_type,
        show_points=False,
        hide_xtick_labels=hide_xtick_labels,
        rotate_xticks=rotate_xticks,
        show_xlabel=True,
        connect_lines=connect_lines,
        marker_map=marker_map,
        add_nominal_to_legend=True,
        nominal_label=f"Nominal = {nominal:.2f}",
        reference_line_color="#D62728",
    )
    _collect(h, l)

    # Panel 3

    for setting in x_values:

        df_s = marginal_df[
            marginal_df[x_axis] == setting
        ]
    
        poly = df_s.loc[
            df_s["method"] == "Polyhedral PSI",
            "avg_length",
        ].mean()
    
        rand = df_s.loc[
            df_s["method"] == "Randomized PSI",
            "avg_length",
        ].mean()
    
        print(
            f"{setting}: Polyhedral intervals were "
            f"{100 * (poly / rand - 1):.1f}% longer "
            f"than the proposed Randomized PSI intervals."
        )
    
    length_summary, h, l = _plot_grouped_errorbar_panel(
        axes[2],
        length_df,
        y_col="avg_length",
        group_col="method",
        x_axis=x_axis,
        x_values=x_values,
        group_order=ci_method_order,
        group_labels=method_labels,
        colors=colors,
        ylabel="",
        title="Interval length",
        nominal=None,
        ylim=length_ylim,
        box_width=box_width,
        errorbar_type=errorbar_type,
        show_points=False,
        hide_xtick_labels=hide_xtick_labels,
        rotate_xticks=rotate_xticks,
        show_xlabel=True,
        connect_lines=connect_lines,
        marker_map=marker_map,
        add_nominal_to_legend=False,
    )
    _collect(h, l)

    x_label_map = {
        "Low Signal": "Weak Separation",
        "High Signal": "Strong Separation",
    }

    for ax in axes:
        current_labels = [
            tick.get_text()
            for tick in ax.get_xticklabels()
        ]
        ax.set_xticklabels([
            x_label_map.get(label, label)
            for label in current_labels
        ])


    # ---- Font styling -------------------------------------------------------
    for ax in axes:
        _style_axis(ax)

    # ---- Legend placement --------------------------------------------------
    if legend_mode == "each":
        for ax in axes:
            ax.legend(frameon=False, fontsize=legend_fontsize, loc="best")

    elif legend_mode == "right":
        fig.legend(
            all_handles,
            all_labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=legend_fontsize,
        )

    elif legend_mode == "bottom":
        if legend_ncol is None:
            legend_ncol = min(len(all_labels), 6)

        fig.legend(
            all_handles,
            all_labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            columnspacing=1.4,
            handletextpad=0.5,
        )

    elif legend_mode != "none":
        raise ValueError("legend_mode must be 'right', 'bottom', 'each', or 'none'.")

    if suptitle is not None:
        fig.suptitle(
            suptitle,
            y=1.06,
            fontsize=suptitle_fontsize,
            fontweight="normal",
        )

    if legend_mode == "bottom":
        plt.tight_layout(rect=(0, 0.13, 1, 1))
    elif legend_mode == "right":
        plt.tight_layout(rect=(0, 0, 0.98, 1))
    else:
        plt.tight_layout()

    if save_pdf is not None:
        fig.savefig(
            save_pdf,
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

    return {
        "fig": fig,
        "axes": axes,
        "regret_df": regret_df,
        "marginal_df": marginal_df,
        "regret_summary": regret_summary,
        "marginal_coverage_summary": marginal_coverage_summary,
        "length_summary": length_summary,
        "regret_is_normalized": regret_is_normalized,
    }


