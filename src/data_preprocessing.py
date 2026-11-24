"""
Module: data_preprocessing.py
Handles loading, inspecting, cleaning, and preprocessing pipeline construction
for the Census Income dataset without data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define default features
DEFAULT_NUMERIC_FEATURES: List[str] = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

DEFAULT_CATEGORICAL_FEATURES: List[str] = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Sensitive attributes for fairness analysis
SENSITIVE_FEATURES: List[str] = ["sex", "race"]

TARGET_COL: str = "income"
POSITIVE_CLASS: str = ">50K"
NEGATIVE_CLASS: str = "<=50K"


def resolve_data_path(custom_path: Optional[str] = None) -> Path:
    """Resolve the path to Adult.csv, checking common project locations."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    candidates = [
        Path("Adult.csv"),
        Path("dataset/Adult.csv"),
        Path("../Adult.csv"),
        Path("../dataset/Adult.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find Adult.csv in default locations.")


def inspect_raw_data(filepath: Optional[str] = None) -> Dict:
    """Inspect raw dataset properties and return descriptive metadata."""
    path = resolve_data_path(filepath)
    df_raw = pd.read_csv(path)

    stats = {
        "raw_shape": df_raw.shape,
        "columns": df_raw.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df_raw.dtypes.items()},
        "null_counts": df_raw.isnull().sum().to_dict(),
        "question_mark_counts": {},
        "raw_target_distribution": df_raw[TARGET_COL].value_counts().to_dict()
        if TARGET_COL in df_raw.columns
        else {},
        "raw_duplicates": int(df_raw.duplicated().sum()),
    }

    for col in df_raw.columns:
        if df_raw[col].dtype == object or str(df_raw[col].dtype).startswith("str"):
            qm = (df_raw[col].astype(str).str.strip() == "?").sum()
            if qm > 0:
                stats["question_mark_counts"][col] = int(qm)

    return stats


def clean_dataset(
    df: pd.DataFrame, drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean raw dataset according to TASK specifications:
    1. Strip leading/trailing whitespace in string columns.
    2. Convert '?' symbols to NaN.
    3. Normalize target variable by stripping trailing '.' (ensuring only '<=50K' and '>50K').
    4. Assert and validate target classes.
    5. Check and optionally remove identical duplicate rows.
    """
    df_cleaned = df.copy()
    report: Dict = {}

    initial_rows = len(df_cleaned)
    report["initial_rows"] = initial_rows

    # 1 & 2: String stripping and replacing '?' with NaN
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == object or str(df_cleaned[col].dtype).startswith("str"):
            # Strip whitespace
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            # Replace '?' or 'nan' / 'None' with actual NaN
            df_cleaned.loc[df_cleaned[col].isin(["?", "nan", "None"]), col] = np.nan

    # 3 & 4: Target normalization
    if TARGET_COL in df_cleaned.columns:
        df_cleaned[TARGET_COL] = df_cleaned[TARGET_COL].astype(str).str.rstrip(".")
        unique_targets = set(df_cleaned[TARGET_COL].dropna().unique())
        expected_targets = {NEGATIVE_CLASS, POSITIVE_CLASS}
        if not unique_targets.issubset(expected_targets):
            unexpected = unique_targets - expected_targets
            raise ValueError(
                f"Unexpected target labels found: {unexpected}. Expected only {expected_targets}."
            )
        report["target_distribution_before_dedup"] = (
            df_cleaned[TARGET_COL].value_counts().to_dict()
        )

    # 5: Identical duplicate rows
    duplicates_count = int(df_cleaned.duplicated().sum())
    report["duplicates_found"] = duplicates_count
    if drop_duplicates and duplicates_count > 0:
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
        report["dropped_duplicates"] = duplicates_count
        report["reason_drop_duplicates"] = (
            "Exact duplicate records across all columns were removed to avoid artificial "
            "overweighting and potential data leakage across cross-validation folds."
        )
    else:
        report["dropped_duplicates"] = 0

    if TARGET_COL in df_cleaned.columns:
        report["final_target_distribution"] = (
            df_cleaned[TARGET_COL].value_counts().to_dict()
        )

    report["final_rows"] = len(df_cleaned)
    return df_cleaned, report


def load_and_clean_data(
    filepath: Optional[str] = None, drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Load Adult.csv and apply standard cleaning routines."""
    path = resolve_data_path(filepath)
    df_raw = pd.read_csv(path)
    df_clean, report = clean_dataset(df_raw, drop_duplicates=drop_duplicates)
    return df_clean, report


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_fnlwgt: bool = True,
    drop_education: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets using stratified sampling on target.
    Prevents data leakage by performing split prior to feature imputation and scaling.

    Note on feature selection:
    - education is 1:1 redundant with education-num. education-num is kept as ordinal numeric.
    - fnlwgt is the Census population sampling weight, not an individual characteristic.
      It has zero correlation with income (r = -0.006) and adds noise. It is dropped by default.
    """
    df_work = df.copy()

    drop_cols = [TARGET_COL]
    if drop_fnlwgt and "fnlwgt" in df_work.columns:
        drop_cols.append("fnlwgt")
    if drop_education and "education" in df_work.columns:
        drop_cols.append("education")

    X = df_work.drop(columns=drop_cols)
    y = df_work[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that handles:
    - Numeric features: Median imputation + (optional) Standard scaling.
    - Categorical features: Most frequent imputation + One-Hot Encoding (handle_unknown='ignore').
    """
    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def create_full_pipeline(
    classifier,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> Pipeline:
    """Wrap preprocessor and classifier into a single robust pipeline."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


if __name__ == "__main__":
    print("Executing data_preprocessing verification...")
    raw_stats = inspect_raw_data()
    print("Raw Data Stats:", raw_stats["raw_shape"])
    print("Null Counts:", raw_stats["null_counts"])
    print("Question Mark Counts:", raw_stats["question_mark_counts"])
    print("Raw Target:", raw_stats["raw_target_distribution"])
    print("Raw Duplicates:", raw_stats["raw_duplicates"])

    df_cleaned, report = load_and_clean_data()
    print("\nCleaned Data Stats:")
    print("Initial rows:", report["initial_rows"])
    print("Duplicates dropped:", report["dropped_duplicates"])
    print("Final rows:", report["final_rows"])
    print("Target distribution:", report["target_distribution"])

    X_train, X_test, y_train, y_test = split_data(df_cleaned)
    print("\nSplit Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("y_train distribution:\n", y_train.value_counts(normalize=True))
    print("y_test distribution:\n", y_test.value_counts(normalize=True))
