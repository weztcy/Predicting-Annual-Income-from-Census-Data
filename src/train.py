"""
Module: train.py
Orchestrates model building, cross-validation on training data, hyperparameter tuning,
imbalance handling evaluation, and saving the winning model pipeline.
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COL,
    create_full_pipeline,
    load_and_clean_data,
    split_data,
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_candidate_models() -> Dict[str, object]:
    """Define candidate model configurations for comparison."""
    return {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Default)": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree (Default)": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
        "Decision Tree (Balanced)": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42
        ),
        "Random Forest (Default)": RandomForestClassifier(
            n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "HistGradientBoosting (Default)": HistGradientBoostingClassifier(
            random_state=42
        ),
        "HistGradientBoosting (Balanced)": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42
        ),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive classification metrics treating '>50K' as positive class."""
    y_binary = (y_true == POSITIVE_CLASS).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    if y_proba is not None and len(np.unique(y_proba)) > 1:
        roc_auc = roc_auc_score(y_binary, y_proba)
        pr_auc = average_precision_score(y_binary, y_proba)
    else:
        roc_auc = 0.5
        pr_auc = y_binary.mean()

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Positive": f1_pos,
        "F1_Macro": f1_macro,
        "Balanced_Accuracy": bal_acc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


def perform_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Perform 5-fold Stratified Cross-Validation on training data for all candidate models.
    Returns summary DataFrame and out-of-fold prediction dictionaries.
    """
    candidates = get_candidate_models()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results_list = []
    oof_results = {}

    print(f"\n--- Starting {cv_splits}-Fold Cross-Validation on Training Data (N = {len(X_train):,}) ---")

    for name, clf in candidates.items():
        print(f"  Evaluating candidate: {name}...")
        pipeline = create_full_pipeline(clf)

        # Out-of-fold predictions
        y_pred_oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=-1)

        if hasattr(clf, "predict_proba"):
            y_proba_oof_matrix = cross_val_predict(
                pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
            )
            # Find positive class index
            pos_idx = 1
            y_proba_oof = y_proba_oof_matrix[:, pos_idx]
        else:
            y_proba_oof = None

        metrics = evaluate_predictions(y_train, y_pred_oof, y_proba_oof)
        row = {"Model": name, **metrics}
        results_list.append(row)

        oof_results[name] = {
            "y_pred": y_pred_oof,
            "y_proba": y_proba_oof,
            "metrics": metrics,
        }

    cv_df = pd.DataFrame(results_list).sort_values(by="PR_AUC", ascending=False).reset_index(drop=True)
    cv_df.to_csv(REPORTS_DIR / "cv_model_comparison.csv", index=False)
    print(f"\nCV results saved to {REPORTS_DIR / 'cv_model_comparison.csv'}")
    return cv_df, oof_results


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[object, Dict]:
    """
    Perform hyperparameter tuning for HistGradientBoostingClassifier using GridSearchCV
    on training data folds only.
    """
    print("\n--- Tuning HistGradientBoosting Hyperparameters via GridSearchCV ---")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    base_clf = HistGradientBoostingClassifier(random_state=random_state)
    pipeline = create_full_pipeline(base_clf)

    param_grid = {
        "classifier__learning_rate": [0.08, 0.1, 0.12],
        "classifier__max_iter": [100, 150],
        "classifier__max_leaf_nodes": [31, 45],
        "classifier__l2_regularization": [0.0, 1.0],
    }

    scorer = make_scorer(
        average_precision_score,
        response_method="predict_proba",
        pos_label=POSITIVE_CLASS,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scorer,  # Optimizing PR-AUC for minority positive class
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best CV PR-AUC Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    return grid_search.best_estimator_, {
        "best_cv_pr_auc": float(best_score),
        "best_params": {k.replace("classifier__", ""): v for k, v in best_params.items()},
    }


def train_and_save_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_pipeline,
    best_params_info: Dict,
) -> Path:
    """Fit winning pipeline on full training data and serialize to disk."""
    print("\nFitting final winning pipeline on full training set...")
    best_pipeline.fit(X_train, y_train)

    save_path = MODELS_DIR / "best_income_pipeline.joblib"
    joblib.dump(best_pipeline, save_path)
    print(f"Saved best pipeline model to: {save_path}")

    # Save hyperparameters and training metadata
    meta_path = MODELS_DIR / "best_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(best_params_info, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")

    return save_path


def train_no_sensitive_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict,
    random_state: int = 42,
) -> object:
    """
    Train a comparison model excluding sensitive attributes (sex, race)
    to audit potential bias and examine proxy effects.
    """
    print("\nTraining comparison model without sensitive attributes (sex, race)...")
    non_sensitive_cat_features = [
        col for col in DEFAULT_CATEGORICAL_FEATURES if col not in ["sex", "race"]
    ]

    hgb_params = {k: v for k, v in best_params.items()}
    clf = HistGradientBoostingClassifier(random_state=random_state, **hgb_params)

    pipe_no_sens = create_full_pipeline(
        clf,
        numeric_features=DEFAULT_NUMERIC_FEATURES,
        categorical_features=non_sensitive_cat_features,
    )

    # Note: drop sex and race from X_train passed to this pipeline
    X_train_no_sens = X_train.drop(columns=["sex", "race"])
    pipe_no_sens.fit(X_train_no_sens, y_train)
    return pipe_no_sens


if __name__ == "__main__":
    df, clean_report = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    cv_df, oof_results = perform_cross_validation(X_train, y_train)
    print("\n--- Cross-Validation Summary Table ---")
    print(cv_df.to_string(index=False))

    best_pipeline, tuning_info = tune_best_model(X_train, y_train)
    save_path = train_and_save_pipeline(X_train, y_train, best_pipeline, tuning_info)
