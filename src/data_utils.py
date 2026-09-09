from src.config import BASE_PATH
import joblib
import pandas as pd
from src.nn_model import load_nn_clf

## xgboost,LightGBM loading conflicts
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


def get_feature_lists(df):
    """
    Classify each col in a df as numerical, nominal, ordinal, or binary.
    """
    num_cols = []
    nominal_cols = []
    ordinal_cols = ["STAGE", "OSTEOTOMY"]
    binary_cols = []
    for col in df:
        if col in ordinal_cols:
            continue
        len_entries = len(df[col].unique())
        if len_entries > 10:
            num_cols.append(col)
        elif len_entries > 2:
            nominal_cols.append(col)
        else:  # binary
            binary_cols.append(col)
    ## Add 1 for target var
    assert (
        len(num_cols) + len(nominal_cols) + len(ordinal_cols) + len(binary_cols)
        == df.shape[1]
    )
    return {
        "Numerical": num_cols,
        "Ordinal": ordinal_cols,
        "Nominal": nominal_cols,
        "Binary": binary_cols,
    }


def get_data(is_nomo, file_dir=BASE_PATH / "data" / "processed"):
    """
    Get X/y train, and testing (internal validation) data
    """
    if is_nomo:
        data_dict = {
            "X": pd.read_parquet(file_dir / "nomo" / "X.parquet"),
            "y": pd.read_excel(file_dir / "nomo" / "y.xlsx", index_col=0),
        }
    else:
        data_dict = {
            "X": pd.read_parquet(file_dir / "base" / "X.parquet"),
            "y": pd.read_excel(file_dir / "base" / "y.xlsx", index_col=0),
        }
    return data_dict


def get_models(model_prefix_list, file_dir=BASE_PATH / "models/trained"):
    """
    Get all models from memory
    """
    model_dict = {}
    X_shape = get_data(is_nomo=False)["X"].shape[1]
    for model_name in model_prefix_list:
        if model_name == "nn":
            model = load_nn_clf(
                data_path=file_dir / "nn.pt",
                in_dim=X_shape,
                device="cpu",
            )
        else:
            model = joblib.load(file_dir / f"{model_name}.joblib")
        model_dict[model_name] = model
    return model_dict
