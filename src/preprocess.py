import numpy as np
import pandas as pd
from shutil import rmtree


def transform_data(X_df, pipeline):
    feature_names = pipeline.get_feature_names_out()
    hopkins_transformed = np.array(pipeline.transform(X_df))
    hopkins_transformed = pd.DataFrame(hopkins_transformed, columns=feature_names)
    hopkins_transformed = remove_prefix(hopkins_transformed)
    for col in hopkins_transformed.columns:
        try:
            hopkins_transformed[col] = pd.to_numeric(hopkins_transformed[col])
        except Exception as e:
            print(f"Column {col} failed: {e}")
    return hopkins_transformed


def remove_prefix(df):
    X = df.copy()
    X.columns = X.columns.str.replace(r"^\w+__", "", regex=True)
    return X


def export_data(X, y, exp_path):
    if exp_path.exists():
        rmtree(exp_path)
    exp_path.mkdir(exist_ok=False, parents=True)
    X.to_parquet(exp_path / "X.parquet")
    y.to_excel(exp_path / "y.xlsx")
