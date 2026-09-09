from src.data_utils import get_feature_lists
import sigfig
import warnings
import pandas as pd
import numpy as np

from scipy.stats import fisher_exact, mannwhitneyu
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def add_dummy_rows(*_, df, col, result_list, header_ORs, header_p):
    """
    Add empty placeholder rows for categorical feature instances to match summary table structure.
    """
    entries = sorted(df[col].unique())
    for entry in entries:
        entry_name = f"{col.upper()} {entry}"
        dummy_row = {
            "Feature": entry_name,
            header_ORs: "",
            header_p: "",
        }
        result_list.append(dummy_row)
    return result_list


def format_p_val(p_val):
    """
    Format p-value as '<0.0001' if <0.0001, otherwise round.
    """
    if p_val < 0.0001:
        return "<0.0001"
    elif p_val <= 0.05:
        return str(sigfig.round(p_val, sigfigs=2))
    else:
        return str(round(p_val, 1))


def get_analysis_df(*_, df, outcome_data, data_type, fish_dict):
    """
    Generate statistical analysis table with p-values, odds ratios, and 95% CIs for all features.

    Uses appropriate tests: .
        ORs: Contingency table for binary, logistic regression for others
        P-vals: Fisher's/Chi-square for binary, Mann-Whitney U for numerical, logistic regression for categorical
    """
    if _ != tuple():
        raise ValueError("This function does not accept positional arguments!")
    ## Set func globals
    feature_lists = get_feature_lists(df)
    binary_cols = feature_lists["Binary"]
    numerical_cols = feature_lists["Numerical"]
    nominal_cols = feature_lists["Nominal"]
    ordinal_cols = feature_lists["Ordinal"]
    header_ORs = "Odds Ratios (95% CI)"
    header_p = "P-Value"
    ##Analysis (p-vals + ORs)
    result_list = []
    full_df = pd.concat([df, outcome_data], axis=1)
    for col in df.columns:  # Loop through all columns
        if col == "ORN":
            result_list.append(
                {"Feature": f"{col.upper()}", header_ORs: "---", header_p: "---"}
            )
            result_list = add_dummy_rows(
                df=full_df,
                col=col,
                result_list=result_list,
                header_ORs=header_ORs,
                header_p=header_p,
            )
            continue
        if col in binary_cols:
            # because small sample size, don't even report ORs if any cell count < 5
            # also, only use fishers exact
            contingency_table = pd.crosstab(df[col], outcome_data)
            _, p_value = fisher_exact(contingency_table)
            p_value = format_p_val(p_value)
            if contingency_table.values.min() < 5:
                # Sparse table — OR is unstable or unbounded; report counts only.
                odds_conf = "UNSTABLE"
            else:
                result = odds_ratio(contingency_table, kind="conditional")
                or_estimate = result.statistic
                ci_low, ci_high = result.confidence_interval(confidence_level=0.95)
                odds_conf = f"{or_estimate:.2f} ({ci_low:.2f}, {ci_high:.2f})"

            result_list.append(
                {"Feature": col.upper(), header_ORs: odds_conf, header_p: p_value}
            )
            ##Append empty rows just to ensure index matches with summary
            result_list = add_dummy_rows(
                df=full_df,
                col=col,
                result_list=result_list,
                header_ORs=header_ORs,
                header_p=header_p,
            )
        elif col in numerical_cols:
            ### Mann-Whitney U test for p-vals ###
            group1 = full_df[full_df["ORN"] == 0][col]
            group2 = full_df[full_df["ORN"] == 1][col]
            _, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            p_value = format_p_val(p_value)
            ### Log Regresion for ORs and CIs ###
            X = sm.add_constant(full_df[col])
            y = full_df["ORN"].values
            model = sm.Logit(y, X).fit(disp=0)
            or_estimate = np.exp(model.params[col])
            conf_int = model.conf_int().loc[col]
            ci_lower = np.exp(conf_int[0])
            ci_upper = np.exp(conf_int[1])
            logit_p = model.pvalues[col]
            logit_p = format_p_val(logit_p)
            odds_conf = f"{or_estimate:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"
            list_adds = [
                {
                    "Feature": col.upper(),
                    header_ORs: odds_conf,
                    header_p: p_value,
                },
                # Extra row to match summary df
                {
                    "Feature": col.upper() + ", Avg ± (SD) -- [25%, 50%, 75%]",
                    header_ORs: "",
                    header_p: "",
                },
                # Extra row to match summary df
                {
                    "Feature": col.upper() + " Missing (% missing)",
                    header_ORs: "",
                    header_p: "",
                },
            ]
            result_list.extend(list_adds)
        elif col in nominal_cols + ordinal_cols:
            entries = sorted(full_df[col].unique())
            result_list.append({"Feature": col.upper(), header_ORs: "", header_p: ""})
            y = full_df["ORN"].values
            ### One-hot encode, excluding entry with highest frequency as a reference-> run log regression for p-val and ORs (CI)
            # List of possible entries in a given column, sorted from low to HIGH
            entries = sorted(full_df[col].unique())
            # Create subset onehot-encoded temporary df
            temp_df = full_df[[col]]
            temp_df = pd.get_dummies(
                temp_df, columns=[col], drop_first=False, dtype=int
            )
            # If nominal, drop highest freq entry and make that reference
            if col in nominal_cols:
                val_counts = full_df[col].value_counts()
                max_freq_idx = val_counts.idxmax()  # Entry with the highest frequency
                drop_col = f"{col}_{max_freq_idx}"
            # If ordinal, drop the entry with the lowest value and make that reference
            else:
                # Ensure no perfect sep in reference instance
                ct = pd.crosstab(df[col], y)
                zero_entries = ct[ct[1] == 0].index.tolist()
                for i in range(len(entries)):
                    drop_col = f"{col}_{entries[i]}"
                    if entries[i] not in zero_entries:
                        break
            temp_df.drop(drop_col, axis=1, inplace=True)
            X = sm.add_constant(temp_df)
            ## Identify problematic entries
            ct = pd.crosstab(df[col], y)
            zero_entries = ct[ct[1] == 0].index.tolist()
            model = None
            or_estimates = None
            conf_ints = None
            p_values = None
            try:  # Try to run Logit
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    model = sm.Logit(y, X).fit(disp=0)
                    or_estimates = np.exp(model.params)
                    conf_ints = model.conf_int()
                    p_values = model.pvalues
            except (
                ConvergenceWarning,
                RuntimeWarning,
                ValueError,
                KeyError,
                OverflowError,
            ) as e:  # Except any issues with logit
                # Log the specific issue
                print(f"Failed to fit model for {col}: {type(e).__name__}: {e}")
                # Mark all entries as unstable
                for entry in entries:
                    entry_name = f"{col}_{entry}"
                    result_list.append(
                        {
                            "Feature": f"{col.upper()} {entry}",
                            header_ORs: "UNSTABLE",
                            header_p: "UNSTABLE",
                        }
                    )
                continue  # Skip to next column
            if model is not None:
                for entry in entries:  # Loop through each possible entry of the feature
                    entry_name = f"{col}_{entry}"  # Reformat to allow for indexing ex) 1.0 --> SITE_1.0
                    if entry_name == drop_col:
                        result_list.append(
                            {
                                "Feature": f"{col.upper()} {entry}",
                                header_ORs: "Reference",
                                header_p: "Reference",
                            }
                        )
                    # For BOTH: If not designated reference, get stat vals
                    else:
                        p_val_specific = format_p_val(p_values[entry_name])
                        or_estimate = or_estimates.loc[entry_name]
                        ci_lower = np.exp(conf_ints.loc[entry_name, 0])
                        ci_upper = np.exp(conf_ints.loc[entry_name, 1])
                        odds_conf = (
                            f"{or_estimate:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"
                        )
                        result_list.append(
                            {
                                "Feature": f"{col.upper()} {entry}",
                                header_ORs: odds_conf,
                                header_p: p_val_specific,
                            }
                        )

    results_df = pd.DataFrame(result_list)
    return results_df.set_index("Feature")


def generate_summary_column(
    df_impute,
    og_df,
    outcome,
    data_type,
    all_categories,
    feature_dict,
):
    """
    Generate single summary statistics column showing counts/percentages (categorical) or mean±SD with quartiles (numerical).
    """
    binary_cols = feature_dict["Binary"]
    numerical_cols = feature_dict["Numerical"]
    nominal_cols = feature_dict["Nominal"]
    ordinal_cols = feature_dict["Ordinal"]

    total_entries = len(df_impute)
    header = f"{outcome}-{data_type}  (n={total_entries})"
    summary_list = []
    for col in df_impute:  # Loop through all columns
        summary_list.append({"Feature": f"{col.upper()}", header: ""})
        if col in binary_cols + nominal_cols + ordinal_cols:  # Categorical
            # Get counts and percentages, add blank row for variable name
            counts = df_impute[col].value_counts()
            percentages = np.round(
                df_impute[col].value_counts(normalize=True) * 100, decimals=1
            )
            # For each instance in feature, add counts/percentages to df
            for entry in all_categories[col]:
                summary_list.append(
                    {
                        "Feature": f"{col.upper()} {entry}",
                        # Get value count, if not existent, replace with 0
                        header: f"{counts.get(entry, 0)} ({percentages.get(entry, 0.0)})",
                    }
                )
        elif col in numerical_cols:  # Numerical
            # Get mean, stdev, quantiles
            avg = np.mean(df_impute[col])
            std = np.std(df_impute[col])
            quantiles = np.round(
                df_impute[col].quantile([0.25, 0.5, 0.75]).values.tolist(), 3
            )
            summary_list.append(
                {
                    "Feature": col.upper() + ", Avg ± (SD) -- [25%, 50%, 75%]",
                    header: f"{avg:.1f} ± {std:.1f} -- {quantiles}",
                }
            )
            n_missing = og_df[col].isna().sum()
            pct_missing = np.round(n_missing / total_entries * 100, 1)
            summary_list.append(
                {
                    "Feature": f"{col.upper()} Missing (% missing)",
                    header: f"{n_missing} ({pct_missing})",
                }
            )
        else:
            print(col)
            raise ValueError
    return pd.DataFrame(summary_list)


def generate_summary_table(
    *_,
    X_df_final,
    X_df_og,
    outcome_data,
    data_type,
    all_categories,
    feature_dict,
):
    """
    Create complete summary table with three columns: Total, Negative outcome, and Positive outcome values.
    """
    ## Concat
    concat_df = pd.concat([X_df_final, outcome_data], axis=1)
    og_concat = pd.concat(
        [X_df_og, outcome_data], axis=1
    )  # This is used to get NA counts for numerical vars
    non_outcome_df = concat_df[concat_df["ORN"] == 0].drop("ORN", axis=1).copy()
    non_outcome_df_og = og_concat[og_concat["ORN"] == 0].drop("ORN", axis=1).copy()
    outcome_df = concat_df[concat_df["ORN"] == 1].drop("ORN", axis=1).copy()
    outcome_df_og = og_concat[og_concat["ORN"] == 1].drop("ORN", axis=1).copy()
    # Get total, neg-outcome, and pos-outcome summary tables
    total_col = generate_summary_column(
        X_df_final,
        X_df_og,
        data_type,
        "Total",
        all_categories,
        feature_dict,
    )
    non_outcome_col = generate_summary_column(
        non_outcome_df,
        non_outcome_df_og,
        data_type,
        "Negative",
        all_categories,
        feature_dict,
    )
    outcome_col = generate_summary_column(
        outcome_df,
        outcome_df_og,
        data_type,
        "Positive",
        all_categories,
        feature_dict,
    )

    ## Combine by feature column
    total_col = total_col.set_index("Feature")

    non_outcome_col = non_outcome_col.set_index("Feature")

    outcome_col = outcome_col.set_index("Feature")

    combined_df = pd.concat([total_col, non_outcome_col, outcome_col], axis=1)

    return combined_df
