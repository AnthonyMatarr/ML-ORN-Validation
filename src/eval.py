from src.config import SEED
import warnings
from pathlib import Path
import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from MLstatkit import Bootstrapping

BIN_NAMES = ["Low", "Medium", "High"]
N_BINS = len(BIN_NAMES)


# Custom encoder to handle NumPy when exporting results json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):  # type: ignore
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_bin_row(metrics):
    """
    Extract and format bin metrics
    """
    # Extract n and percentage
    n_perc = metrics["n_perc"]
    n = n_perc["n"]
    perc_cohort = n_perc["perc"]

    # Extract percentage of all positives
    perc_all_pos = metrics["perc_all_pos"]
    n_pos = perc_all_pos["n"]
    perc_pos = perc_all_pos["perc"]

    # Extract event rate with CIs
    event_dict = metrics["event_rate_w_CIs"]
    event_rate = event_dict["event_rate"]
    ci_lower = event_dict.get("lower_CI", "N/A")
    ci_upper = event_dict.get("upper_CI", "N/A")

    # Format event rate string
    ci_str = f"({ci_lower:.2%}, {ci_upper:.2%})"

    # Extract lift
    lift = metrics["lift"]

    # Extract thresholds and mean output
    thresholds = metrics["thresholds"]
    mean_output = metrics["mean_model_output"]

    # Build row
    row = {
        "N (% of tot cohort)": f"{int(n)} ({float(perc_cohort):.2%})",
        "N pos (% of All Positives)": f"{int(n_pos)} ({float(perc_pos):.2%})",
        "Event Rate (95% CI)": f"{event_rate:.2%} {ci_str}",
        "Lift": f"{float(lift):.2f}" if not np.isnan(lift) else np.nan,
        "Thresholds": thresholds,
        "Mean Model Output": (
            f"{float(mean_output):.2%}" if not np.isnan(mean_output) else np.nan
        ),
    }
    return row


########################################## Helper Functions ##########################################
def get_bin_metrics(y_true, y_proba, thresholds, bin_report_dict, n_bootstraps):
    """
    Assumes global bin names
    """
    thresholds = np.asarray(thresholds, dtype=float).flatten()
    bin_indices = np.digitize(y_proba, thresholds, right=False)  # 0,1,...,n_bins-1
    tot_n = len(y_true)
    tot_n_pos = np.sum(y_true)
    tot_event_rate = tot_n_pos / tot_n
    for b in range(N_BINS):
        ## Get labels + probs of allocated to this bin
        mask = bin_indices == b
        n = mask.sum()
        in_bin_labels = y_true[mask]
        in_bin_probs = y_proba[mask]
        # ================= Populate bin dict =================
        bin_name = BIN_NAMES[b]
        bin_report_dict[bin_name] = {}
        ## Total patients in bin (% of test cohort)
        bin_report_dict[bin_name]["n_perc"] = {"n": n, "perc": n / tot_n}
        ## % of all pos patients in this bin
        in_bin_n_pos = np.sum(in_bin_labels)
        # n_perc_all_pos
        n_perc_pos = in_bin_n_pos / tot_n_pos if tot_n_pos > 0 else np.nan
        bin_report_dict[bin_name]["perc_all_pos"] = {
            "n": in_bin_n_pos,
            "perc": n_perc_pos,
        }
        ## Event rate w/ CIs
        # Check if bin has both classes (required for bootstrap)
        n_unique_classes = len(np.unique(in_bin_labels))

        if n > 0 and n_unique_classes > 1:
            try:
                event_rate_boot, ci_lower, ci_upper = Bootstrapping(
                    in_bin_labels,
                    in_bin_probs,  # this not used but need to pass
                    metric_str="event_rate",
                    n_bootstraps=n_bootstraps,
                    random_state=SEED,
                    show_progress=False,
                )
            except RuntimeError:
                # Fallback if bootstrap fails
                event_rate_boot = in_bin_labels.mean()
                ci_lower = np.nan
                ci_upper = np.nan
        else:
            # Not enough data or only one class
            event_rate_boot = in_bin_labels.mean() if n > 0 else np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        event_rate = in_bin_labels.mean()
        bin_report_dict[bin_name]["event_rate_w_CIs"] = {
            "event_rate": event_rate,
            "lower_CI": ci_lower,
            "upper_CI": ci_upper,
            "event_rate_boot": event_rate_boot,
        }
        ## Lift
        bin_report_dict[bin_name]["lift"] = event_rate / tot_event_rate
        ## thresholds
        if b == 0:
            threshold_str = f"[0%, {thresholds[0]:.2%})"
        elif b == N_BINS - 1:
            threshold_str = f"[{thresholds[-1]:.2%}, 100%]"
        else:
            threshold_str = f"[{thresholds[b-1]:.2%}, {thresholds[b]:.2%})"
        bin_report_dict[bin_name]["thresholds"] = threshold_str
        ## mean model output
        bin_report_dict[bin_name]["mean_model_output"] = in_bin_probs.mean()
    return bin_report_dict


def plot_risk_bar_dot(bin_report_dict, ax=None, y_max=1.0):
    """
    Create risk stratification plot with bar graph showing observed event rates and overlaid mean predictions per bin.
    """
    ## Label bins w/ thresholds
    event_rates = []
    bins_labels = []
    mean_preds = []
    counts = []
    for i in range(N_BINS):
        bin_name = BIN_NAMES[i]
        cur_dict = bin_report_dict[bin_name]
        threshold_str = cur_dict["thresholds"]
        bins_labels.append(f"{bin_name}\n{threshold_str}")
        event_rates.append(cur_dict["event_rate_w_CIs"]["event_rate"])
        mean_preds.append(cur_dict["mean_model_output"])
        counts.append(cur_dict["n_perc"]["n"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(N_BINS), event_rates, color="C0", alpha=0.7, label="Event Rate")
    ax.plot(range(N_BINS), mean_preds, "o-", color="C1", label="Avg. Predicted Risk")
    ax.set_xticks(range(N_BINS))
    ax.set_xticklabels(bins_labels, rotation=0)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max + (y_max / 10), 5))
    ax.set_ylabel("Fraction With Outcome / Mean Prediction")
    ax.set_xlabel("Risk Bin")
    ax.legend()

    # n=XXXX at bottom of bar
    for i, n in enumerate(counts):
        ax.text(
            i,
            0.0,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="k",
        )
    plt.tight_layout()
    return ax


def get_cm(
    model_name,
    data_type,
    y_true,
    y_pred,
    show_output=False,
    results_path=None,
):
    """
    Generate confusion matrix and (optionally) export it

    Parameters
    ---------
    model_name: str
        Specify name of model used to generate y_pred
    data_type: str
        Specify subset of df
        Usually one of [train, val, test]
    y_true: numpy.ndarray
        Array containing true binary outcome/target labels
    y_pred: numpy.ndarray
        Array containing predicted binary outcome/target labels
    show_output: Optional boolean; defaults to False
        Boolean flag specifying whether to display generated confusion matrix
    results_path: Optional pathlib.Path; defaults to None
        Path to directory where results are stored
        If left None, will not write to memory
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title(f"{model_name}: {data_type}")
    if results_path:
        cm_path = results_path / "figures" / "CM" / f"{model_name}_{data_type}_CM.pdf"
        if cm_path.exists():
            warnings.warn(f"Over-writing confusion matrix at path: {cm_path}")
            cm_path.unlink()
        cm_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(cm_path, bbox_inches="tight")
    if show_output:
        plt.show()
    else:
        plt.close()


def get_discrimination_str(
    *_,
    y_true,
    y_proba,
    metric_str,
    threshold,
    n_bootstraps=5000,
    random_state=SEED,
    bin_thresholds=None,
    show_progress=False,
):
    """
    Calculate a value and 95% CI for a given metric using MLStakit.Bootstrapping

    Parameters
    ----------
    y_true: numpy.ndarray
        True binary class labels
    y_proba: numpy.ndarray
        Continues predicted probabilities
    metric_str: str
        Specify the metric type to get values for
    threshold: float
        Threshold value to use for converting probabilities into hard labels
    n_bootstraps: Optional int; defaults to 5000
        Number of iterations to run bootstrap method for
    random_state: Optional int; defaults to SEED from src.config
        Controls determinism

    Returns
    -------
    final_str: String of format
        '<metric_val> (<ci_lower>, <ci_upper>)'
    Raises
    ------
    ValueError:
        -If positional arguments are passed
        -If an unaccepted str type is passed. Must be one of:
            'f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'average_precision', 'pr_auc', 'ici', 'brier'
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if metric_str == "ici":
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            bin_thresholds=bin_thresholds,
            show_progress=show_progress,
        )
    else:
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            show_progress=show_progress,
        )
    final_str = f"{metric_val:.3f} ({ci_lower:.3f}, {ci_upper:.3f})"
    return final_str


def plot_ROC(
    y_true, y_proba, data_type, n_bootstraps=5000, seed=SEED, show_progress=False
):
    """
    Plot ROC curve, get AUROC w/ CIs, determine threshold for hard predictions
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc, lower_CI, upper_CI = Bootstrapping(
        y_true,
        y_proba,
        random_state=seed,
        metric_str="roc_auc",
        n_bootstraps=n_bootstraps,
        show_progress=show_progress,
    )
    auc_string = f"{auc:.3f} ({lower_CI:.3f}-{upper_CI:.3f})"
    model_score = f"AUROC = {auc_string}"
    plt.plot(fpr, tpr, lw=4, label=f"{data_type} {model_score}")
    return auc_string


########################################## Main function ##########################################
def evaluate_models(
    *_,
    model_dict,
    X,
    y,
    threshold_dict,
    bin_import_dir,
    data_type="test",
    results_path=None,
    n_bootstraps=5000,
    show_cm=False,
    show_roc=False,
    show_cal=False,
    show_progress=False,
):
    """
    Comprehensive model evaluation: ROC/AUROC, optimal thresholds, discrimination metrics, risk stratification, and calibration.

    Returns nested dict with metrics across all models.

    Params
    ------
    model_dict: dict{str: <model>}
        Maps model name to trained model
    X: pandas df
    y: pandas Series
    threshold_dict: dict{str:float}
        Maps model name to selected threshold for binary
    bin_import_dir: str or Pathlib.path
        Path to directory holding thresholds for risk stratification
    data_type: str
        Specify what kind of data (ex. train, val, test)

    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    CLASS_REPORT_DICT = {}
    BIN_REPORT_DICT = {}
    ## For each model
    for model_name, model in model_dict.items():
        BIN_REPORT_DICT[model_name] = {}
        print(f"Model: {model_name}...")
        # ================== ADD TO CLASS REPORT ===================
        y_proba = model.predict_proba(X)[:, 1]
        #################################################################################################################
        ############################################## Risk Bins ########################################################
        #################################################################################################################
        # ================== GET BIN THRESHOLDS ===================
        bins_path = bin_import_dir / f"{model_name}.npz"
        bin_thresholds = np.load(bins_path)["thresholds"]
        # ================== Populate bin report ===================
        BIN_REPORT_DICT[model_name] = get_bin_metrics(
            y_true=y,
            y_proba=y_proba,
            thresholds=bin_thresholds,
            bin_report_dict=BIN_REPORT_DICT[model_name],
            n_bootstraps=n_bootstraps,
        )
        # ================== PLOT RISK BARS ===================
        ax = plot_risk_bar_dot(BIN_REPORT_DICT[model_name], y_max=1.0)
        plt.title(f"{model_name} {data_type} Risk Stratification")
        if results_path:
            bin_plot_path = results_path / "figures" / "risk_bins" / f"{model_name}.pdf"
            if bin_plot_path.exists():
                print(f"Over-writing bin plot at path {bin_plot_path}")
            bin_plot_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(bin_plot_path, bbox_inches="tight")
        if show_cal:
            plt.show()
        else:
            plt.close()

        #################################################################################################################
        ########################################### AUROC + binary thresholds ###########################################
        #################################################################################################################
        print(f"\t Dealing with AUROC...")
        # ================== Add to class report ===================
        plt.figure(figsize=(12, 8))
        plt.plot(
            [0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier"
        )
        roc_str = plot_ROC(
            y,
            y_proba,
            data_type,
            n_bootstraps=n_bootstraps,
            show_progress=show_progress,
        )
        # ================== ADD TO CLASS REPORT ===================
        binary_threshold = threshold_dict[model_name]
        # ================== Add to class report ===================
        CLASS_REPORT_DICT[model_name] = {
            "AUROC (95% CI)": roc_str,
            "Threshold": round(binary_threshold, 3),
        }
        # ================== PLOT ===================
        plt.xlim([0.0, 1.0])  # type: ignore
        plt.ylim([0.0, 1.05])  # type: ignore
        plt.xlabel("False Positive Rate", fontsize=21, fontweight=550)
        plt.ylabel("True Positive Rate", fontsize=21, fontweight=550)
        plt.tick_params(axis="both", which="major", labelsize=15)
        plt.title(f"{model_name} ROC", fontweight="semibold", fontsize=25)
        plt.legend(loc="lower right", prop={"size": 19, "weight": 550})
        if results_path:
            roc_path = Path(results_path) / "figures" / "ROC" / f"{model_name}_ROC.pdf"
            if roc_path.exists():
                warnings.warn(f"Over-writing roc-curve at path {roc_path}")
                roc_path.unlink()
            roc_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(roc_path, bbox_inches="tight")
        if show_roc:
            plt.show()
        else:
            plt.close()
        #################################################################################################################
        ########################################### Get discrimination metrics ##########################################
        #################################################################################################################
        print(f"\t Getting discrimination metrics...")
        # ================== Get predictions ===================
        y_pred = (y_proba >= binary_threshold).astype(int)
        # ================== Confusion Matrices ===================
        get_cm(
            model_name,
            data_type,
            y,
            y_pred,
            show_cm,
            results_path=results_path,
        )
        # ================== Get accuracy, recall, precision, brier, ici ===================
        metrics_strs = ["f1", "accuracy", "recall", "precision", "brier", "ici"]
        for metric_str in metrics_strs:
            ## Only need bin thresholds for ICI
            if metric_str == "ici":
                bin_thresholds_for_ici = bin_thresholds
            else:
                bin_thresholds_for_ici = None
            ## Train
            CLASS_REPORT_DICT[model_name][metric_str] = get_discrimination_str(
                y_true=y,
                y_proba=y_proba,
                metric_str=metric_str,
                threshold=binary_threshold,
                n_bootstraps=n_bootstraps,
                random_state=SEED,
                bin_thresholds=bin_thresholds_for_ici,
                show_progress=show_progress,
            )
    return CLASS_REPORT_DICT, BIN_REPORT_DICT
