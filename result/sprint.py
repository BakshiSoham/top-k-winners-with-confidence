from pathlib import Path
from math import comb
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.Randomized_PSI import TopKSelectionModel
from src.Polyhedral_PSI import PolyhedralTopKInference


# ============================================================
# Paths
# ============================================================

RESULT_DIR = Path(__file__).resolve().parent
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

baseline_path = RESULT_DIR / "baseline.csv"
outcomes_path = RESULT_DIR / "outcomes.csv"


def save_table(table, filename, *, index=False, show=True):
    """Save a result table and optionally print it in Terminal."""
    output_path = TABLE_DIR / filename
    table.to_csv(output_path, index=index)

    print(f"\nSaved table: {output_path}")

    if show:
        print(table.to_string(index=index))

    return output_path


baseline = pd.read_csv(baseline_path)
outcomes = pd.read_csv(outcomes_path)


if baseline["MASKID"].duplicated().any():
    raise ValueError("baseline.csv contains duplicated MASKID values.")

if outcomes["MASKID"].duplicated().any():
    raise ValueError("outcomes.csv contains duplicated MASKID values.")

sprint = baseline.merge(
    outcomes,
    on="MASKID",
    how="inner",
    validate="one_to_one",
)

print("Baseline shape:", baseline.shape)
print("Outcomes shape:", outcomes.shape)
print("Merged shape:", sprint.shape)
print("Matched participants:", sprint["MASKID"].nunique())

if len(sprint) != 9361:
    raise ValueError(
        f"Expected 9,361 matched participants, obtained {len(sprint):,}."
    )


analysis_vars = [
    # Subject ID
    "MASKID",

    # Treatment
    "INTENSIVE",

    # Demographics
    "AGE",
    "FEMALE",
    "RACE4",

    # History
    "N_AGENTS",
    "SMOKE_3CAT",
    "ASPIRIN",
    "SUB_CLINICALCVD",
    "SUB_SUBCLINICALCVD",
    "STATIN",

    # Laboratory / baseline measurements
    "SBP",
    "DBP",
    "EGFR",
    "SCREAT",
    "CHR",
    "GLUR",
    "HDL",
    "TRR",
    "UMALCR",
    "BMI",

    # Primary outcome
    "EVENT_PRIMARY",
    "T_PRIMARY",
]

sprint_analysis = sprint[analysis_vars].copy()


# ============================================================
# 1. Construct age subgroups and fixed 3-year binary outcome
# ============================================================


sprint_analysis = sprint_analysis.copy()
numeric_columns = [
    "INTENSIVE",
    "AGE",
    "EVENT_PRIMARY",
    "T_PRIMARY",
    "FEMALE",
    "N_AGENTS",
    "SMOKE_3CAT",
    "ASPIRIN",
    "SUB_CLINICALCVD",
    "SUB_SUBCLINICALCVD",
    "STATIN",
    "SBP",
    "DBP",
    "EGFR",
    "SCREAT",
    "CHR",
    "GLUR",
    "HDL",
    "TRR",
    "UMALCR",
    "BMI",
]

for column in numeric_columns:
    sprint_analysis[column] = pd.to_numeric(sprint_analysis[column],errors="coerce",)


# ------------------------------------------------------------
# 4 mutually exclusive age groups
# ------------------------------------------------------------
age_bins = [49, 64, 69, 74, np.inf]
age_labels = ["50–64","65–69","70–74","75+",]


sprint_analysis["AGE_GROUP"] = pd.cut(
    sprint_analysis["AGE"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True,
    right=True,
)

age_group_sizes = (
    sprint_analysis["AGE_GROUP"]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("age_group")
    .reset_index(name="n")
)

save_table(
    age_group_sizes,
    "01_age_group_sizes.csv",
)


# ------------------------------------------------------------
# Fixed 3-year binary primary outcome
# ------------------------------------------------------------

#FOLLOWUP_DAYS = 3 * 365   # 1095 days
FOLLOWUP_DAYS = 1000
event_within_followup = (
    (sprint_analysis["EVENT_PRIMARY"] == 1)
    & (sprint_analysis["T_PRIMARY"] <= FOLLOWUP_DAYS)
)
# Outcome is known to be event-free at FOLLOWUP_DAYS:
#   (1) no event and observed at least FOLLOWUP_DAYS
#   (2) event occurred after FOLLOWUP_DAYS
event_free_at_followup = (
    (sprint_analysis["T_PRIMARY"] > FOLLOWUP_DAYS)
    | (
        (sprint_analysis["EVENT_PRIMARY"] == 0)
        & (sprint_analysis["T_PRIMARY"] >= FOLLOWUP_DAYS)
    )
)

sprint_analysis["Y_PRIMARY_3Y"] = np.nan
sprint_analysis.loc[event_within_followup, "Y_PRIMARY_3Y"] = 1
sprint_analysis.loc[event_free_at_followup, "Y_PRIMARY_3Y"] = 0
sprint_complete = sprint_analysis.dropna(subset=["MASKID","INTENSIVE","AGE_GROUP","Y_PRIMARY_3Y",]).copy()
sprint_complete["INTENSIVE"] = sprint_complete["INTENSIVE"].astype(int)
print("Original sample size:", len(sprint_analysis))
print("Usable fixed-3-year sample size:", len(sprint_complete))
outcome_summary = (
    sprint_complete
    .groupby(
        ["AGE_GROUP", "INTENSIVE"],
        observed=False,
    )
    .agg(
        n=("MASKID", "size"),
        events=("Y_PRIMARY_3Y", "sum"),
        event_rate=("Y_PRIMARY_3Y", "mean"),
    )
    .reset_index()
)

save_table(
    outcome_summary,
    "02_outcome_summary.csv",
)


# ============================================================
# 2. Define favorable binary outcome
# Section 6.1: outcome 1 = favorable outcome
# ============================================================

sprint_complete = sprint_complete.copy()

# Y_PRIMARY_3Y:
#   1 = primary event occurred
#   0 = event-free
#
# SUCCESS:
#   1 = event-free (favorable)
#   0 = primary event occurred
sprint_complete["SUCCESS"] = (1 - sprint_complete["Y_PRIMARY_3Y"]).astype(int)
success_distribution = (
    sprint_complete["SUCCESS"]
    .value_counts()
    .sort_index()
    .rename_axis("success")
    .reset_index(name="n")
)

save_table(
    success_distribution,
    "03_success_distribution.csv",
)

# ============================================================
# 3. Covariates for adjustment
# ============================================================

continuous_covariates = [
    "AGE",
    "N_AGENTS",
    "SBP",
    "DBP",
    "EGFR",
    "SCREAT",
    "CHR",
    "GLUR",
    "HDL",
    "TRR",
    "UMALCR",
    "BMI",
]

binary_covariates = [
    "FEMALE",
    "ASPIRIN",
    "SUB_CLINICALCVD",
    "SUB_SUBCLINICALCVD",
    "STATIN",
]

categorical_covariates = [
    "RACE4",
    "SMOKE_3CAT",
]


# ============================================================
# 4. Covariate preprocessing
# ============================================================

def prepare_sprint_covariates(
    data,
    *,
    reference_columns=None,
):
    dat = data.copy()

    # Continuous: median imputation
    for col in continuous_covariates:
        dat[col] = pd.to_numeric(dat[col],errors="coerce",)
        dat[col] = dat[col].fillna(dat[col].median())

    # Binary: mode imputation
    for col in binary_covariates:
        dat[col] = pd.to_numeric(dat[col],errors="coerce",)
        mode_value = dat[col].mode(dropna=True).iloc[0]
        dat[col] = dat[col].fillna(mode_value)

    # Categorical: mode + dummy variables
    for col in categorical_covariates:
        dat[col] = dat[col].astype("string")
        mode_value = dat[col].mode(dropna=True).iloc[0]
        dat[col] = dat[col].fillna(mode_value)

    X_num = dat[continuous_covariates+ binary_covariates].astype(float)
    X_cat = pd.get_dummies(
        dat[categorical_covariates],
        prefix=categorical_covariates,
        drop_first=True,
        dtype=float,
    )

    X_cov = pd.concat([X_num.reset_index(drop=True),X_cat.reset_index(drop=True),],axis=1,)
    if reference_columns is not None:
        X_cov = X_cov.reindex(columns=reference_columns,fill_value=0.0,)

    return (dat.reset_index(drop=True),X_cov.reset_index(drop=True),)


# ============================================================
# 5. Adjusted success probabilities
# ============================================================

import statsmodels.api as sm


def estimate_sprint_adjusted_binary_effects(
    data,
    *,
    age_labels=age_labels,
    compute_variance=True,
):

    dat, X_cov = prepare_sprint_covariates(data)
    theta_hat = []
    variance_hat = []
    rows = []

    for age_group in age_labels:
        mask = (dat["AGE_GROUP"].astype(str)== str(age_group))
        group_data = (dat.loc[mask].reset_index(drop=True))
        group_cov = (X_cov.loc[mask].reset_index(drop=True))
        if len(group_data) == 0:
            raise ValueError(f"No observations in {age_group}")
        if (group_data["INTENSIVE"].nunique()< 2):
            raise ValueError(f"{age_group} does not contain both treatment arms.")
        y = group_data["SUCCESS"].astype(float).to_numpy()
        A = group_data["INTENSIVE"].astype(float).to_numpy()

        # ----------------------------------------------------
        # Logistic regression:
        #
        # logit P(SUCCESS=1 | A, X)
        # =
        # beta_0 + beta_A A + gamma^T X
        # ----------------------------------------------------
        design = pd.concat(
            [pd.Series(1.0,index=group_data.index,name="intercept",),pd.Series(A,index=group_data.index,name="INTENSIVE",),group_cov,],
            axis=1,
        )

        #fit = sm.GLM(y,design,family=sm.families.Binomial(),).fit(cov_type="HC0")
        fit = sm.GLM(y,design,family=sm.families.Binomial(),).fit()
        beta_hat = np.asarray(fit.params,dtype=float,)
        #if compute_variance:
            #robust_fit = sm.GLM(y,design,family=sm.families.Binomial(),).fit(cov_type="HC0")
            #V_beta = np.asarray(robust_fit.cov_params(),dtype=float,)
        #else:
            #V_beta = None
        V_beta = np.asarray(fit.cov_params(),dtype=float,)
        X1 = design.copy()
        X1["INTENSIVE"] = 1.0
        eta1 = (X1.to_numpy()@ beta_hat)
        p1 = 1.0 / (1.0 + np.exp(-eta1))
        pi1_hat = float(np.mean(p1))
        X0 = design.copy()
        X0["INTENSIVE"] = 0.0
        eta0 = (X0.to_numpy()@ beta_hat)
        p0 = 1.0 / (1.0 + np.exp(-eta0))
        pi0_hat = float(np.mean(p0))

        # ----------------------------------------------------
        # Subgroup-specific treatment effect
        # ----------------------------------------------------

        theta_j = (pi1_hat - pi0_hat)
        n1 = int((group_data["INTENSIVE"] == 1).sum())
        n0 = int((group_data["INTENSIVE"] == 0).sum())

        # ----------------------------------------------------
        # Delta-method variance
        # ----------------------------------------------------
        
        X1_np = X1.to_numpy(dtype=float)
        X0_np = X0.to_numpy(dtype=float)
        grad_pi1 = np.mean((p1 * (1.0 - p1))[:, None]* X1_np,axis=0,)
        grad_pi0 = np.mean((p0 * (1.0 - p0))[:, None]* X0_np,axis=0,)
        var_pi1 = float(grad_pi1@ V_beta@ grad_pi1)
        var_pi0 = float(grad_pi0@ V_beta@ grad_pi0)
        cov_pi1_pi0 = float(grad_pi1@ V_beta@ grad_pi0)
        var_theta_j = float(var_pi1+ var_pi0- 2.0 * cov_pi1_pi0)

        var_theta_j = max(var_theta_j,0.0,)
        theta_hat.append(theta_j)
        variance_hat.append(var_theta_j)
        rows.append({
            "age_group":str(age_group),
            "n_total":len(group_data),
            "n_intensive":n1,
            "n_standard":n0,
            "adjusted_pi_1":pi1_hat,
            "adjusted_pi_0":pi0_hat,
            "theta_hat":theta_j,
            "var_pi_1":var_pi1,
            "var_pi_0":var_pi0,
            "var_theta":var_theta_j,
            "standard_error":np.sqrt(var_theta_j),
        })

    theta_hat = np.asarray(theta_hat,dtype=float,)
    variance_hat = np.asarray(variance_hat,dtype=float,)
    Sigma_hat = np.diag(variance_hat)
    effect_table = pd.DataFrame(rows)
    effect_table["rank"] = (effect_table["theta_hat"].rank(ascending=False,method="first",).astype(int))
    effect_table = effect_table.sort_values("rank").reset_index(drop=True)

    

    return {"theta_hat":theta_hat,"Sigma_hat":Sigma_hat,"effect_table":effect_table,}



# ============================================================
# 6. Full-data estimates
# ============================================================

full_fit = estimate_sprint_adjusted_binary_effects(sprint_complete)
theta_hat_sprint = full_fit["theta_hat"]
Sigma_hat_sprint = full_fit["Sigma_hat"]
print("Adjusted subgroup treatment effects:")

adjusted_effect_table = full_fit["effect_table"].copy()

save_table(
    adjusted_effect_table,
    "04_adjusted_effects.csv",
)

print("\ntheta_hat:")
print(theta_hat_sprint)

covariance_table = pd.DataFrame(
    Sigma_hat_sprint,
    index=age_labels,
    columns=age_labels,
)
covariance_table.index.name = "age_group"

save_table(
    covariance_table,
    "05_covariance_matrix.csv",
    index=True,
)



# ============================================================
# 7. Utility
# ============================================================

def sprint_utility(x):
    return np.asarray(x,dtype=float,)


# ============================================================
# 8. Randomized PSI
# ============================================================

def run_randomized_psi_sprint(
    theta_hat,
    Sigma_hat,
    *,
    subgroup_labels,
    k=2,
    epsilon=None,
    alpha=0.05,
    grid_size=500,
    seed=123,
):
    theta_hat = np.asarray(theta_hat,dtype=float,)
    Sigma_hat = np.asarray(Sigma_hat,dtype=float,)
    if epsilon is None:
        from math import comb
        epsilon = np.log(comb(len(theta_hat),k,))
    model = TopKSelectionModel(
        X=theta_hat,
        k=k,
        H0_mu=np.zeros_like(theta_hat),
        true_Sigma=Sigma_hat,
        utility_fn=sprint_utility,
        epsilon=float(epsilon),
        grid_size=int(grid_size),
        sel_scale="adaptive",
    )

    selected_subset, _ = (
        model.randomized_selected_top_k(
            X=theta_hat,
            k=k,
            epsilon=float(epsilon),
            scale="adaptive",
            seed=int(seed),
        )
    )

    selected_subset = tuple(sorted(int(j)for j in selected_subset))
    ci_output = (
        model.confidence_interval_topk(
            S_obs=selected_subset,
            Sigma=Sigma_hat,
            alpha=alpha,
            k=k,
            epsilon=float(epsilon),
            grid_size=int(grid_size),
            seed=int(seed) + 1,
            verbose=False,
        )
    )

    rows = []
    for rank, record_list in (
        ci_output["per_rank"].items()
    ):
        for record in record_list:
            j = int(record["idx"])
            lower = float(record["L"])
            upper = float(record["U"])
            rows.append({
                "method":"Randomized PSI",
                "rank":int(rank),
                "subgroup_index":j,
                "age_group":subgroup_labels[j],
                "estimate":float(theta_hat[j]),
                "lower":lower,
                "upper":upper,
                "length":upper - lower,
                "selected_top2":tuple(subgroup_labels[j0]for j0 in selected_subset),
            })

    return pd.DataFrame(rows)




# ============================================================
# 9. Polyhedral PSI
# ============================================================

def run_polyhedral_psi_sprint(
    theta_hat,
    Sigma_hat,
    *,
    subgroup_labels,
    k=2,
    alpha=0.05,
    grid_size=500,
):
    theta_hat = np.asarray(theta_hat,dtype=float,)
    Sigma_hat = np.asarray(Sigma_hat,dtype=float,)

    model = PolyhedralTopKInference(
        X=theta_hat,
        k=k,
        H0_mu=np.zeros_like(theta_hat),
        Sigma=Sigma_hat,
        utility_fn=sprint_utility,
        grid_size=int(grid_size),
        alpha=alpha,
    )

    ci_records = (model.confidence_interval_topk(alpha=alpha))
    selected_subset = tuple(int(j)for j in model.selected_set)
    rows = []
    for record in ci_records:
        j = int(record["index"])
        lower = float(record["ci_lower"])
        upper = float(record["ci_upper"])
        rows.append({
            "method":"Polyhedral PSI",
            "rank":int(record["rank"]),
            "subgroup_index":j,
            "age_group":subgroup_labels[j],
            "estimate":float(theta_hat[j]),
            "lower":lower,
            "upper":upper,
            "length":upper - lower,
            "selected_top2":tuple(subgroup_labels[j0]for j0 in selected_subset),
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. Data Splitting
# ============================================================

from statistics import NormalDist


def run_data_splitting_sprint(
    data,
    *,
    subgroup_labels,
    k=2,
    alpha=0.05,
    seed=123,
    selection_frac=0.5,
):
    rng = np.random.default_rng(seed)
    shuffled_indices = (rng.permutation(len(data)))
    if not (0.0 < selection_frac < 1.0):
        raise ValueError("selection_frac must be between 0 and 1.")
    
    n_selection = int(np.floor(len(data) * selection_frac))
    selection_indices = (shuffled_indices[:n_selection])
    inference_indices = (shuffled_indices[n_selection:])
    selection_data = data.iloc[selection_indices].copy().reset_index(drop=True)
    inference_data = data.iloc[inference_indices].copy().reset_index(drop=True)

    # --------------------------------------------------------
    # Re-estimate everything separately
    # --------------------------------------------------------
    selection_fit = estimate_sprint_adjusted_binary_effects(selection_data)
    inference_fit = estimate_sprint_adjusted_binary_effects(inference_data)
    theta_selection = selection_fit["theta_hat"]
    theta_inference = inference_fit["theta_hat"]
    Sigma_inference = inference_fit["Sigma_hat"]
    selected_ranked = (np.argsort(theta_selection)[::-1][:k])

    z_value = NormalDist().inv_cdf(1.0- alpha / 2.0)
    selection_pct = int(round(selection_frac * 100))
    inference_pct = int(round((1.0 - selection_frac) * 100))
    method_name = (f"Data Splitting "f"({selection_pct}/{inference_pct})")
    rows = []

    for rank, j in enumerate(
        selected_ranked,
        start=1,
    ):
        j = int(j)
        estimate = float(theta_inference[j])
        standard_error = float(
            np.sqrt(max(Sigma_inference[j, j],0.0,)))
        lower = (estimate- z_value* standard_error)
        upper = (estimate+ z_value* standard_error)
        rows.append({
            "method":method_name,
            "rank":rank,
            "subgroup_index":j,
            "age_group":subgroup_labels[j],
            "estimate":estimate,
            "lower":float(lower),
            "upper":float(upper),
            "length":float(upper - lower),
            "selected_top2":tuple(subgroup_labels[int(j0)]for j0 in selected_ranked),
            "selection_frac":selection_frac,
            "selection_half_estimate":float(theta_selection[j]),
        })

    return pd.DataFrame(rows)



# ============================================================
# 11. Run all methods
# ============================================================

from math import comb

P_SPRINT = len(age_labels)
K_SPRINT = 2
ALPHA_SPRINT = 0.1
seed=123
Q_SPRINT = 1.0

EPSILON_SPRINT = Q_SPRINT* np.log(comb(P_SPRINT,K_SPRINT,))

print(
    "p =", P_SPRINT,
    "k =", K_SPRINT,
    "epsilon =", EPSILON_SPRINT,
)


randomized_result = (
    run_randomized_psi_sprint(
        theta_hat_sprint,
        Sigma_hat_sprint,
        subgroup_labels=age_labels,
        k=K_SPRINT,
        epsilon=EPSILON_SPRINT,
        alpha=ALPHA_SPRINT,
        grid_size=500,
        seed=seed,
    )
)

polyhedral_result = (
    run_polyhedral_psi_sprint(
        theta_hat_sprint,
        Sigma_hat_sprint,
        subgroup_labels=age_labels,
        k=K_SPRINT,
        alpha=ALPHA_SPRINT,
        grid_size=500,
    )
)

# ============================================================
# Data Splitting: 30/70, 50/50, 70/30
# ============================================================

data_splitting_30_70 = (
    run_data_splitting_sprint(
        sprint_complete,
        subgroup_labels=age_labels,
        k=K_SPRINT,
        alpha=ALPHA_SPRINT,
        seed=seed,
        selection_frac=0.3,
    )
)


data_splitting_50_50 = (
    run_data_splitting_sprint(
        sprint_complete,
        subgroup_labels=age_labels,
        k=K_SPRINT,
        alpha=ALPHA_SPRINT,
        seed=seed,
        selection_frac=0.5,
    )
)

data_splitting_70_30 = (
    run_data_splitting_sprint(
        sprint_complete,
        subgroup_labels=age_labels,
        k=K_SPRINT,
        alpha=ALPHA_SPRINT,
        seed=seed,
        selection_frac=0.7,
    )
)

# data_splitting_90_10 = (
#     run_data_splitting_sprint(
#         sprint_complete,
#         subgroup_labels=age_labels,
#         k=K_SPRINT,
#         alpha=ALPHA_SPRINT,
#         seed=seed,
#         selection_frac=0.9,
#     )
# )

sprint_result = pd.concat(
    [
        randomized_result,
        polyhedral_result,
        data_splitting_30_70,
        data_splitting_50_50,
        data_splitting_70_30,
        # data_splitting_90_10,
    ],
    ignore_index=True,
)

sprint_result = sprint_result.sort_values(["method","rank",]).reset_index(drop=True)
save_table(
    sprint_result,
    "06_all_method_results.csv",
)


# ============================================================
# 12. Final table
# ============================================================

sprint_table = (
    sprint_result[
        [
            "method",
            "rank",
            "age_group",
            "estimate",
            "lower",
            "upper",
            "length",
            "selected_top2",
        ]
    ]
    .copy()
)

sprint_table["90% CI"] = (
    sprint_table.apply(
        lambda row:
            f"({row['lower']:.4f}, "
            f"{row['upper']:.4f})",
        axis=1,
    )
)

sprint_table["estimate"] = (sprint_table["estimate"].round(4))
sprint_table["length"] = (sprint_table["length"].round(4))

sprint_table = (
    sprint_table.rename(
        columns={
            "method":"Method",
            "rank":"Rank",
            "age_group":"Selected subgroup",
            "estimate":"Point estimate",
            "length":"Interval length",
            "selected_top2":"Selected Top-2",
        }
    )
)

sprint_table = sprint_table[
    [
        "Method",
        "Rank",
        "Selected subgroup",
        "Selected Top-2",
        "Point estimate",
        "90% CI",
        "Interval length",
    ]
]
save_table(
    sprint_table,
    "07_final_table.csv",
)


# ============================================================
# 13. Stability analysis
#
# Significant =
#   subgroup is selected
#   AND its 90% CI does not contain 0
# ============================================================



def ci_excludes_zero(lower, upper):
    return (lower > 0.0) or (upper < 0.0)


# ============================================================
# 14. Repeat Randomized PSI and Data Splitting over seeds
# ============================================================

def run_sprint_stability_analysis(
    *,
    sprint_complete,
    theta_hat,
    Sigma_hat,
    subgroup_labels,
    n_reps=100,
    k=2,
    alpha=0.10,
    epsilon=None,
    grid_size=500,
    split_fracs=(0.3,0.5, 0.7, 0.9),
    seed_start=0,
    verbose=True,
):

    if epsilon is None:
        from math import comb
        epsilon = np.log(comb(len(subgroup_labels),k,))

    all_records = []
    failures = []
    seeds = range(seed_start,seed_start + n_reps,)

    for rep, seed in enumerate(seeds):
        if verbose and (rep % 10 == 0 or rep == n_reps - 1):
            print(f"Stability repetition "f"{rep + 1}/{n_reps}")

        # ====================================================
        # A. Randomized PSI
        # ====================================================

        try:
            random_result = (
                run_randomized_psi_sprint(
                    theta_hat,
                    Sigma_hat,
                    subgroup_labels=subgroup_labels,
                    k=k,
                    epsilon=epsilon,
                    alpha=alpha,
                    grid_size=grid_size,
                    seed=seed,
                )
            )

            selected_randomized = set(random_result["age_group"].astype(str))

            for subgroup in subgroup_labels:
                subgroup = str(subgroup)
                subset = random_result[random_result["age_group"].astype(str)== subgroup]
                selected = (subgroup in selected_randomized)

                if (selected and len(subset) > 0):
                    lower = float(subset.iloc[0]["lower"])
                    upper = float(subset.iloc[0]["upper"])
                    significant = (ci_excludes_zero(lower,upper,))

                else:
                    lower = np.nan
                    upper = np.nan
                    significant = False

                all_records.append({
                    "rep":rep,
                    "seed":seed,
                    "method":"Randomized PSI",
                    "subgroup":subgroup,
                    "selected":bool(selected),
                    "significant":bool(significant),
                    "selected_and_significant":bool(selected and significant),
                    "lower":lower,
                    "upper":upper,
                })

        except Exception as err:

            failures.append({
                "rep":rep,
                "seed":seed,
                "method":"Randomized PSI",
                "error":repr(err),
            })

        # ====================================================
        # B. Data Splitting
        # ====================================================

        for frac in split_fracs:

            method_name = (
                f"Data Splitting "
                f"({int(round(100 * frac))}/"
                f"{int(round(100 * (1-frac)))})"
            )

            try:

                split_result = (
                    run_data_splitting_sprint(
                        sprint_complete,
                        subgroup_labels=
                            subgroup_labels,
                        k=k,
                        alpha=alpha,
                        seed=seed,
                        selection_frac=frac,
                    )
                )

                selected_split = set(split_result["age_group"].astype(str))

                for subgroup in subgroup_labels:
                    subgroup = str(subgroup)
                    subset = split_result[split_result["age_group"].astype(str)== subgroup]

                    selected = (subgroup in selected_split)

                    if (selected and len(subset) > 0):
                        lower = float(subset.iloc[0]["lower"])
                        upper = float(subset.iloc[0]["upper"])
                        significant = ci_excludes_zero(lower,upper,)

                    else:
                        lower = np.nan
                        upper = np.nan
                        significant = False

                    all_records.append({
                        "rep":rep,
                        "seed":seed,
                        "method":method_name,
                        "subgroup":subgroup,
                        "selected":bool(selected),
                        "significant":bool(significant),
                        "selected_and_significant":bool(selected and significant),
                        "lower":lower,
                        "upper":upper,
                    })

            except Exception as err:

                failures.append({
                    "rep":rep,
                    "seed":seed,
                    "method":method_name,
                    "error":repr(err),
                })

    records_df = pd.DataFrame(all_records)
    failures_df = pd.DataFrame(failures)

    return {
        "records":records_df,
        "failures":failures_df,
    }



# ============================================================
# 15. Run stability analysis
# ============================================================

N_STABILITY = 100

stability_out = (
    run_sprint_stability_analysis(
        sprint_complete=sprint_complete,
        theta_hat=theta_hat_sprint,
        Sigma_hat=Sigma_hat_sprint,
        subgroup_labels=age_labels,
        n_reps=N_STABILITY,
        k=K_SPRINT,
        alpha=0.1,
        epsilon=EPSILON_SPRINT,
        grid_size=500,
        split_fracs=(0.3,0.5,0.7,),
        seed_start=0,
        verbose=True,
    )
)

stability_records = stability_out["records"]
stability_failures = stability_out["failures"]

print("Number of failures:", len(stability_failures))

save_table(
    stability_records,
    "08_stability_records.csv",
    show=False,
)

if stability_failures.empty:
    stability_failures = pd.DataFrame(
        columns=["rep", "seed", "method", "error"]
    )

save_table(
    stability_failures,
    "09_stability_failures.csv",
)



# ============================================================
# 16. Count selected + significant
# ============================================================

stability_summary = (
    stability_records
    .groupby(
        [
            "method",
            "subgroup",
        ],
        as_index=False,
    )
    .agg(
        n_selected=("selected","sum",),
        n_significant=("significant","sum",),
        n_selected_and_significant=("selected_and_significant","sum",),
        n_runs=("seed","nunique",),
    )
)

stability_summary["selected_and_significant_rate"] = stability_summary["n_selected_and_significant"]/stability_summary["n_runs"]
save_table(
    stability_summary,
    "10_stability_summary.csv",
)

# ============================================================
# 17. Stability bar plot
# ============================================================



method_order = [
    "Randomized PSI",
    "Data Splitting (30/70)",
    "Data Splitting (50/50)",
    "Data Splitting (70/30)",
#    "Data Splitting (90/10)",
]

subgroup_order = [str(x) for x in age_labels]


plot_df = (
    stability_summary
    .pivot(
        index="subgroup",
        columns="method",
        values="n_selected_and_significant",
    )
    .reindex(
        index=subgroup_order,
        columns=method_order,
    )
    .fillna(0)
)

save_table(
    plot_df.reset_index(),
    "11_stability_plot_data.csv",
)


ax = plot_df.plot(kind="bar",figsize=(10, 6),width=0.8,)
ax.set_xlabel("Age Subgroup",fontsize=12,)
ax.set_ylabel("Number of Times Selected and Significant",fontsize=12,)
ax.set_title(("SPRINT Stability Analysis "f"({N_STABILITY} Random Seeds, 90% CI)"),fontsize=14,)
ax.legend(title="Method",)
ax.tick_params(axis="x",rotation=0,)

plt.tight_layout()

figure_pdf = FIGURE_DIR / "sprint_stability.pdf"
figure_png = FIGURE_DIR / "sprint_stability.png"

plt.savefig(
    figure_pdf,
    bbox_inches="tight",
)

plt.savefig(
    figure_png,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved figure: {figure_pdf}")
print(f"Saved figure: {figure_png}")






