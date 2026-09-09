import warnings
import copy
import logging
from src.config import SEED

warnings.filterwarnings("ignore", category=UserWarning)
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline

from shap.utils._legacy import DenseData


def as_frame(bg, columns):
    """Unwrap a shap.kmeans DenseData summary into a DataFrame, replicating
    rows by cluster weight so the background keeps its original density."""
    if not isinstance(bg, DenseData):
        return bg
    df = pd.DataFrame(bg.data, columns=columns)
    w = getattr(bg, "weights", None)
    if w is None:
        return df
    counts = np.maximum(1, np.round(np.asarray(w) * 100).astype(int))
    return df.loc[np.repeat(df.index.values, counts)].reset_index(drop=True)


def get_ohe_cols(df):
    """
    Parse one-hot encoded column names to extract original feature names and their categories.

    Returns dict mapping feature prefixes (e.g., 'SITE') to lists of category values (e.g., ['1.0', '2.0']).
    """
    ohe_dict = {}
    for col in df.columns:
        col_split = col.split("_")
        if len(col_split) == 1:
            continue
        else:
            col_name = col_split[0]
            instance_name = col_split[1]
        if col_name in ohe_dict.keys():
            ohe_dict[col_name].append(instance_name)
        else:
            ohe_dict[col_name] = [instance_name]
    return ohe_dict


######## Combine one-hot-encoded #######
def combine_encoded(shap_values, name, mask, return_original=True):
    """
    Aggregate SHAP values from one-hot encoded columns into a single combined feature importance.

    Sums SHAP values across encoded columns and reconstructs categorical data for display.

    Implementation Details
    ----------------------
    Adapted from: https://gist.github.com/peterdhansen/ca87cc1bfbc4c092f0872a3bfe3204b2
    """
    mask = np.array(mask)
    mask_col_names = np.array(shap_values.feature_names, dtype="object")[mask]
    sv_name = shap.Explanation(
        shap_values.values[:, mask],
        feature_names=list(mask_col_names),
        data=shap_values.data[:, mask],
        base_values=shap_values.base_values,
        display_data=shap_values.display_data,
        instance_names=shap_values.instance_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )

    # Use argmax to find which category is active (most robust for one-hot data)
    # This finds the index of the maximum value in each row
    new_data = np.argmax(sv_name.data, axis=1).astype(int)

    # Safety check: clip to valid range
    new_data = np.clip(new_data, 0, len(mask_col_names) - 1)

    svdata = np.concatenate(
        [shap_values.data[:, ~mask], new_data.reshape(-1, 1)], axis=1
    )

    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]

    svdisplay_data = np.concatenate(
        [svdd, mask_col_names[new_data].reshape(-1, 1)], axis=1
    )

    # Handle multi-class (3D) vs binary/regression (2D) SHAP arrays
    if len(shap_values.values.shape) == 3:  # Multi-class case
        new_values = sv_name.values.sum(axis=1, keepdims=True)  # type: ignore
        svvalues = np.concatenate([shap_values.values[:, ~mask, :], new_values], axis=1)
    else:  # Binary/regression case
        new_values = sv_name.values.sum(axis=1)  # type: ignore
        svvalues = np.concatenate(
            [shap_values.values[:, ~mask], new_values.reshape(-1, 1)], axis=1
        )

    svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

    sv = shap.Explanation(
        svvalues,
        base_values=shap_values.base_values,
        data=svdata,
        display_data=svdisplay_data,
        instance_names=shap_values.instance_names,
        feature_names=svfeature_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )
    if return_original:
        return sv, sv_name
    else:
        return sv


def get_vals_to_plot(shap_vals):
    """
    Convert 3D SHAP arrays (multi-class/multi-output) to 2D format for plotting.

    Extracts positive class for binary classification or handles single-output cases.
    """
    if len(shap_vals.values.shape) == 3:  # 3D array
        if shap_vals.values.shape[2] == 1:  # Binary classification with single output
            # DNN
            shap_vals_to_plot = shap_vals[:, :, 0]
        elif shap_vals.values.shape[2] >= 2:  # Binary with two outputs or multi-class
            shap_vals_to_plot = shap_vals[:, :, 1]  # Use positive class
        else:
            shap_vals_to_plot = shap_vals.mean(axis=2)  # Fallback
    else:  # 2D array
        # LightGBM, SVC, KNN, Stack, LR-Nomogram
        shap_vals_to_plot = shap_vals
    return shap_vals_to_plot


def generate_MAV(shap_vals, feat_order, model_name, result_path=None):
    """
    Calculate mean absolute SHAP values (MASV) and relative percentages for feature importance ranking.

    Optionally exports results to Excel with custom feature ordering.
    """
    feat_names = shap_vals.feature_names
    try:
        assert set(feat_names) == set(feat_order)
    except AssertionError:
        print(set(feat_names) - set(feat_order))
        print(set(feat_order) - set(feat_names))
        raise AssertionError("Feature names and feature order do not match")
    shap_to_plot = get_vals_to_plot(shap_vals)
    shap_df = pd.DataFrame(shap_to_plot.values, columns=feat_names)
    absolute_mean_shap = shap_df.abs().mean().reset_index()
    # Get absolute avg + relative abs avg
    absolute_mean_shap = shap_df.abs().mean().reset_index()
    absolute_mean_shap.columns = ["Feature", "MASV"]
    sum_vals = absolute_mean_shap["MASV"].sum()
    if sum_vals == 0:
        raise Exception("All generated SHAP values are 0. Exiting...")
    absolute_mean_shap["Relative_ MASV"] = np.round(
        (100 * absolute_mean_shap["MASV"] / absolute_mean_shap["MASV"].sum()), 2
    )
    # Ensure logic makes sense
    assert np.isclose(
        absolute_mean_shap["Relative_ MASV"].sum(), 100, atol=0.1
    ), f"Sum is instead {absolute_mean_shap['Relative_ MASV'].sum()}"
    # Reorder
    absolute_mean_shap["Feature"] = absolute_mean_shap["Feature"].astype(str)

    absolute_mean_shap_reordered = (
        absolute_mean_shap.set_index("Feature").loc[feat_order].reset_index()
    )
    ######################## Display + Export ########################
    if result_path:
        result_path.parent.mkdir(exist_ok=True, parents=True)
        with pd.ExcelWriter(
            result_path,
            engine="openpyxl",
            mode="a" if result_path.exists() else "w",
        ) as writer:
            absolute_mean_shap_reordered.to_excel(
                writer, sheet_name=model_name, index=True
            )


def get_shap_single_model(
    *_,
    model,
    model_name,
    feat_order,
    explanation_vals,
    background_vals,
    result_path=None,
):
    """
    Compute SHAP values for a single model using appropriate explainer (Tree/Linear/Kernel).

    Handles pipelines with samplers, combines one-hot features, and exports MASV table.
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    print(f"Starting SHAP for model: {model_name}")
    ######################## GET RAW SHAP VALUES ########################
    feat_names_raw = explanation_vals.columns.tolist()
    is_pipeline = isinstance(model, (ImbPipeline, SkPipeline))

    ######################## GET RAW SHAP VALUES ########################
    if model_name == "xgb":
        # XGBoost in a pipeline with sampler
        if is_pipeline:
            print(f"\t Using kernel...")

            # Use full pipeline with KernelExplainer
            def predict_fn(X):
                return model.predict_proba(X)[:, 1]

            explainer = shap.KernelExplainer(
                model=predict_fn,
                data=background_vals,
            )
            shap_raw = explainer(explanation_vals)
        else:
            print("\tUsing tree...")
            # Regular TreeExplainer
            explainer = shap.TreeExplainer(
                model=model,
                data=as_frame(background_vals, feat_names_raw),
                feature_perturbation="interventional",
                model_output="probability",
                feature_names=feat_names_raw,
            )
            shap_raw = explainer(explanation_vals)

    elif model_name == "lgbm":
        explainer = shap.TreeExplainer(
            model=model,
            data=as_frame(background_vals, feat_names_raw),
            feature_perturbation="interventional",
            model_output="probability",
            feature_names=feat_names_raw,
        )
        shap_raw = explainer(explanation_vals)

    elif model_name == "lr":
        explainer = shap.LinearExplainer(
            model,
            as_frame(background_vals, feat_names_raw),
            feature_names=feat_names_raw,
            seed=SEED,
        )
        shap_raw = explainer(explanation_vals)

    elif model_name == "knn":
        # KNN in pipeline with sampler
        def predict_fn(X):
            return model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(
            model=predict_fn,
            data=background_vals,
        )
        shap_raw = explainer(explanation_vals)

    elif model_name in ["nn", "stack", "svc"]:
        print(f"[{model_name}] Using KernelExplainer")
        explainer = shap.KernelExplainer(
            model=model.predict_proba,
            data=background_vals,
            feature_names=feat_names_raw,
        )
        shap_raw = explainer(explanation_vals)
    else:
        raise ValueError(
            f"Expected model_name to be one of ['xgb', 'lgbm', 'nn', 'lr', 'stack', 'knn', 'svc']; got {model_name} instead!"
        )

    print("SHAP values calculated")
    ######################## Deal with one-hot encoded ########################
    ohe_dict = get_ohe_cols(explanation_vals)
    ohe_cols = ohe_dict.keys()

    raw_feat_order = []
    for col in feat_order:
        if col in ohe_cols:
            for sub_col in ohe_dict[col]:
                raw_feat_order.append(f"{col}_{sub_col}")
        else:
            raw_feat_order.append(col)

    shap_old = copy.deepcopy(shap_raw)
    for col_name in ohe_cols:
        # Create mask for exact matches: COLNAME_*
        mask = [
            feat_name.startswith(f"{col_name}_") for feat_name in shap_old.feature_names
        ]

        # Skip if no columns match (already combined or doesn't exist)
        if not any(mask):
            logging.warning(f"No features found for {col_name}, skipping")
            continue

        shap_combined, _ = combine_encoded(shap_old, col_name, mask)
        shap_old = shap_combined

    print("OHE features combined")
    ######################## Generate + export MAV table ########################
    # generate_MAV(shap_raw, raw_feat_order, model_name=model_name, result_path=raw_path)
    generate_MAV(
        shap_combined, feat_order, model_name=model_name, result_path=result_path
    )
    print("Computation complete!")


################################## PERMUTATION ##################################
def plot_perm(
    model_name,
    model,
    X,
    y,
    n_repeats=300,
    result_dir=None,
    show_output=False,
    scoring="roc_auc",
):
    """
    Calculate and plot permutation feature importance using box-and-whisker and horizontal bar charts.

    Measures decrease in model scoring (auroc by default) when each feature is randomly shuffled.
    """
    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        n_repeats=n_repeats,
        random_state=SEED,
        n_jobs=4,
        scoring=scoring,
    )
    forest_importances = pd.Series(
        result.importances_mean, index=X.columns  # type: ignore
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    forest_importances.plot.barh(xerr=result.importances_std, ax=ax)  # type: ignore
    ax.set_title(
        f"Permutation Feature Importance for {model_name} ({n_repeats} iterations)"
    )
    ax.set_xlabel("Decrease in mean AUROC")
    fig.tight_layout()
    if result_dir:
        result_path = result_dir / f"{model_name}.pdf"
        if result_path.exists():
            result_path.unlink()
        result_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(result_path, bbox_inches="tight")
    if show_output:
        plt.show()
    plt.close()
