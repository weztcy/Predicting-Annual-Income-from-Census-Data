"""
Module: evaluate.py
Evaluates models on the untouched test set, generates test performance metrics,
conducts the fairness audit across sex and race, runs permutation importance,
and generates the remaining mandatory evaluation figures (Figures 9 - 14).
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_preprocessing import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    POSITIVE_CLASS,
    create_full_pipeline,
    load_and_clean_data,
    split_data,
)
from src.train import (
    get_candidate_models,
    train_no_sensitive_pipeline,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_fairness_analysis,
    plot_feature_importance,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")


def evaluate_model_on_test(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Tuple[Dict[str, float], np.ndarray, Optional[np.ndarray]]:
    """Compute test set metrics for a single fitted model."""
    y_pred = model.predict(X_test)
    y_binary = (y_test == POSITIVE_CLASS).astype(int)

    if hasattr(model, "predict_proba"):
        y_proba_mat = model.predict_proba(X_test)
        # Class 1 is '>50K'
        pos_idx = 1
        y_proba = y_proba_mat[:, pos_idx]
        roc_auc = roc_auc_score(y_binary, y_proba)
        pr_auc = average_precision_score(y_binary, y_proba)
    else:
        y_proba = None
        roc_auc = 0.5
        pr_auc = y_binary.mean()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    f1_pos = f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Positive": f1_pos,
        "F1_Macro": f1_macro,
        "Balanced_Accuracy": bal_acc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }
    return metrics, y_pred, y_proba


def evaluate_all_candidates_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_pipeline,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Fit each candidate model on X_train and evaluate once on X_test for fair test comparison.
    Also includes the best tuned pipeline.
    """
    candidates = get_candidate_models()
    test_results = []
    models_pred_dict = {}

    print("\n--- Evaluating Models on Test Set (N = {:,}) ---".format(len(X_test)))

    # Evaluate best tuned pipeline first
    print("  Evaluating: Best Tuned HistGradientBoosting...")
    best_metrics, best_pred, best_proba = evaluate_model_on_test(
        best_pipeline, X_test, y_test, "Best Tuned HistGB"
    )
    test_results.append(best_metrics)
    models_pred_dict["Best Tuned HistGB"] = {
        "y_true": y_test,
        "y_pred": best_pred,
        "y_score": best_proba,
        "roc_auc": best_metrics["ROC_AUC"],
        "pr_auc": best_metrics["PR_AUC"],
    }

    for name, clf in candidates.items():
        print(f"  Fitting & Evaluating candidate on test set: {name}...")
        pipeline = create_full_pipeline(clf)
        pipeline.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model_on_test(pipeline, X_test, y_test, name)
        test_results.append(metrics)

        if y_proba is not None:
            models_pred_dict[name] = {
                "y_true": y_test,
                "y_pred": y_pred,
                "y_score": y_proba,
                "roc_auc": metrics["ROC_AUC"],
                "pr_auc": metrics["PR_AUC"],
            }

    test_comparison_df = pd.DataFrame(test_results).sort_values(by="PR_AUC", ascending=False).reset_index(drop=True)
    test_comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"\nFinal Test Model Comparison saved to: {REPORTS_DIR / 'model_comparison.csv'}")

    return test_comparison_df, models_pred_dict


def compute_fairness_audit(
    model_full,
    model_no_sens,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Perform fairness and bias audit across 'sex' and 'race' groups.
    Computes:
    - Sample size (N)
    - Selection Rate (Proportion predicted >50K)
    - Precision
    - Recall (True Positive Rate)
    - False Positive Rate (FPR)
    - F1 Score
    - Accuracy
    Compares the full model vs the model trained without sex and race features.
    """
    print("\n--- Conducting Fairness & Demographic Disparity Audit ---")
    y_pred_full = model_full.predict(X_test)
    X_test_no_sens = X_test.drop(columns=["sex", "race"])
    y_pred_no_sens = model_no_sens.predict(X_test_no_sens)

    audit_records = []

    def audit_subgroup(group_type: str, group_val: str, mask: pd.Series, model_variant: str, y_pred: np.ndarray):
        sub_y_true = y_test[mask]
        sub_y_pred = y_pred[mask]
        n_samples = len(sub_y_true)

        if n_samples == 0:
            return

        y_true_bin = (sub_y_true == POSITIVE_CLASS).astype(int)
        y_pred_bin = (sub_y_pred == POSITIVE_CLASS).astype(int)

        sel_rate = y_pred_bin.mean()
        acc = accuracy_score(sub_y_true, sub_y_pred)
        prec = precision_score(sub_y_true, sub_y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        rec = recall_score(sub_y_true, sub_y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        f1 = f1_score(sub_y_true, sub_y_pred, pos_label=POSITIVE_CLASS, zero_division=0)

        # FPR: FP / (FP + TN)
        neg_mask = (y_true_bin == 0)
        fpr = y_pred_bin[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

        audit_records.append({
            "Model_Variant": model_variant,
            "Group_Type": group_type,
            "Subgroup": group_val,
            "Sample_Size": n_samples,
            "Selection_Rate": sel_rate,
            "Precision": prec,
            "Recall": rec,
            "FPR": fpr,
            "F1": f1,
            "Accuracy": acc,
        })

    # Evaluate Sex subgroups
    for sex in sorted(X_test["sex"].unique()):
        mask = (X_test["sex"] == sex)
        audit_subgroup("Sex", sex, mask, "Full Model (With Sex & Race)", y_pred_full)
        audit_subgroup("Sex", sex, mask, "Ablation Model (Without Sex & Race)", y_pred_no_sens)

    # Evaluate Race subgroups
    for race in sorted(X_test["race"].unique()):
        mask = (X_test["race"] == race)
        audit_subgroup("Race", race, mask, "Full Model (With Sex & Race)", y_pred_full)
        audit_subgroup("Race", race, mask, "Ablation Model (Without Sex & Race)", y_pred_no_sens)

    fairness_df = pd.DataFrame(audit_records)
    fairness_df.to_csv(REPORTS_DIR / "fairness_report.csv", index=False)
    print(f"Fairness report saved to: {REPORTS_DIR / 'fairness_report.csv'}")
    return fairness_df


def compute_permutation_importance(
    best_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate permutation feature importance on test set using PR-AUC (average precision)."""
    print("\n--- Computing Permutation Feature Importance on Test Set ---")
    def pr_auc_scorer(estimator, X, y):
        y_bin = (y == POSITIVE_CLASS).astype(int)
        proba = estimator.predict_proba(X)[:, 1]
        return average_precision_score(y_bin, proba)

    perm_res = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring=pr_auc_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm_res.importances_mean,
        "std": perm_res.importances_std,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print(f"Feature importance saved to: {REPORTS_DIR / 'feature_importance.csv'}")
    return importance_df


def save_sample_predictions(
    best_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_samples: int = 25,
    random_state: int = 42,
) -> Path:
    """Save sample predictions showing inputs, actual target, prediction, and positive probability."""
    sample_indices = X_test.sample(n=n_samples, random_state=random_state).index

    sample_X = X_test.loc[sample_indices].copy()
    sample_y = y_test.loc[sample_indices]

    pred_classes = best_pipeline.predict(sample_X)
    pred_probas = best_pipeline.predict_proba(sample_X)[:, 1]

    results_df = sample_X.copy()
    results_df["actual_income"] = sample_y
    results_df["predicted_income"] = pred_classes
    results_df["predicted_probability_>50K"] = pred_probas.round(4)
    results_df["prediction_correct"] = (results_df["actual_income"] == results_df["predicted_income"])

    output_path = REPORTS_DIR / "sample_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Sample test predictions saved to: {output_path}")
    return output_path


def run_full_evaluation():
    """Main execution entry point for complete model evaluation."""
    df, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Load best model pipeline
    best_model_path = MODELS_DIR / "best_income_pipeline.joblib"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {best_model_path}. Run train.py first.")

    best_pipeline = joblib.load(best_model_path)
    print(f"Loaded best pipeline from {best_model_path}")

    # 1. Candidate models evaluation on test set
    test_comp_df, models_pred_dict = evaluate_all_candidates_test(
        X_train, y_train, X_test, y_test, best_pipeline
    )

    # 2. Train ablation model without sensitive features for fairness comparison
    import json
    meta_path = MODELS_DIR / "best_model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            best_params = meta.get("best_params", {})
    else:
        best_params = {}

    model_no_sens = train_no_sensitive_pipeline(X_train, y_train, best_params)

    # 3. Compute fairness audit
    fairness_df = compute_fairness_audit(best_pipeline, model_no_sens, X_test, y_test)

    # 4. Compute Permutation Feature Importance
    importance_df = compute_permutation_importance(best_pipeline, X_test, y_test)

    # 5. Save Sample Predictions
    save_sample_predictions(best_pipeline, X_test, y_test)

    # 6. Generate Figures 9 - 14
    print("\n--- Generating Evaluation Figures (9 - 14) ---")
    print("Generating Fig 9 (Model Comparison)...")
    plot_model_comparison(test_comp_df)

    print("Generating Fig 10 (Best Model Confusion Matrix)...")
    best_y_pred = models_pred_dict["Best Tuned HistGB"]["y_pred"]
    plot_confusion_matrix(y_test, best_y_pred, model_name="Best Tuned HistGB")

    print("Generating Fig 11 (ROC Curves)...")
    # Select key models for cleaner ROC plot
    key_models = [
        "Best Tuned HistGB",
        "Random Forest (Default)",
        "Random Forest (Balanced)",
        "Logistic Regression (Default)",
        "Logistic Regression (Balanced)",
        "Decision Tree (Default)",
    ]
    roc_dict = {k: models_pred_dict[k] for k in key_models if k in models_pred_dict}
    plot_roc_curves(roc_dict)

    print("Generating Fig 12 (Precision-Recall Curves)...")
    pr_dict = {k: models_pred_dict[k] for k in key_models if k in models_pred_dict}
    plot_pr_curves(pr_dict)

    print("Generating Fig 13 (Feature Importance)...")
    plot_feature_importance(importance_df, top_n=12)

    print("Generating Fig 14 (Fairness Analysis)...")
    # Use full model records for figure 14
    fairness_plot_df = fairness_df[fairness_df["Model_Variant"] == "Full Model (With Sex & Race)"].copy()
    plot_fairness_analysis(fairness_plot_df)

    print("\n=== All Evaluation Steps and Figures 9-14 Completed Successfully! ===")


if __name__ == "__main__":
    run_full_evaluation()
