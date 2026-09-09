import pandas as pd
import numpy as np


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
