"""
Module: visualization.py
Generates and saves the 14 mandatory figures required by TASK.md.
All plots are rendered with high resolution (>= 150 DPI), clear annotations,
proper labels, legends, and clean styling.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)

# Set global publication-ready style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Income Class Distribution
# --------------------------------------------------------------------------
def plot_income_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 1: Class distribution of the target variable income."""
    if output_path is None:
        output_path = FIGURES_DIR / "01_income_class_distribution.png"

    counts = df["income"].value_counts()
    props = df["income"].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    colors = ["#2b5c8f", "#d95f02"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)

    for bar, prop in zip(bars, props):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts.values) * 0.02,
            f"{int(height):,} ({prop:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Distribution of Annual Income Target Class (N = {:,})".format(len(df)), fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Income Category", fontsize=11)
    ax.set_ylabel("Number of Observations", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 2. Missing Values Per Feature
# --------------------------------------------------------------------------
def plot_missing_values(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 2: Missing values per feature."""
    if output_path is None:
        output_path = FIGURES_DIR / "02_missing_values_per_feature.png"

    # Evaluate nulls and '?' in string columns
    missing_dict = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            qm_count = (df[col].astype(str).str.strip() == "?").sum()
        else:
            qm_count = 0
        total_missing = null_count + qm_count
        missing_dict[col] = total_missing

    missing_series = pd.Series(missing_dict).sort_values(ascending=True)
    pct_series = (missing_series / len(df)) * 100

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    colors = ["#d95f02" if val > 0 else "#2b5c8f" for val in missing_series.values]
    bars = ax.barh(missing_series.index, missing_series.values, color=colors, edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, pct_series.values):
        width = bar.get_width()
        if width > 0:
            ax.text(
                width + len(df) * 0.005,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(width):,} ({pct:.2f}%)",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#b30000",
            )
        else:
            ax.text(
                len(df) * 0.005,
                bar.get_y() + bar.get_height() / 2.0,
                "0 (0.0%)",
                ha="left",
                va="center",
                fontsize=8,
                color="#555555",
            )

    ax.set_title("Missing Values per Feature (including '?' Symbols, N = {:,})".format(len(df)), fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Number of Missing / Unknown Values", fontsize=11)
    ax.set_ylabel("Dataset Features", fontsize=11)
    ax.set_xlim(0, max(missing_series.values) * 1.25 if max(missing_series.values) > 0 else 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 3. Age Distribution by Income Class
# --------------------------------------------------------------------------
def plot_age_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 3: Age distribution separated by income class."""
    if output_path is None:
        output_path = FIGURES_DIR / "03_age_distribution_by_income.png"

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}, dpi=200
    )

    palette = {"<=50K": "#2b5c8f", ">50K": "#d95f02"}

    # Boxplot on top
    sns.boxplot(
        data=df,
        x="age",
        y="income",
        hue="income",
        palette=palette,
        ax=ax_box,
        orient="h",
        legend=False,
    )
    ax_box.set(xlabel="", ylabel="")
    ax_box.set_title("Age Distribution by Income Class", fontsize=12, pad=10, fontweight="bold")

    # KDE / Histogram on bottom
    sns.histplot(
        data=df,
        x="age",
        hue="income",
        palette=palette,
        element="step",
        stat="density",
        common_norm=False,
        bins=35,
        ax=ax_hist,
        alpha=0.4,
    )

    med_low = df[df["income"] == "<=50K"]["age"].median()
    med_high = df[df["income"] == ">50K"]["age"].median()

    ax_hist.axvline(med_low, color="#2b5c8f", linestyle="--", linewidth=1.5, label=f"Median <=50K ({med_low:.0f} yrs)")
    ax_hist.axvline(med_high, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Median >50K ({med_high:.0f} yrs)")

    ax_hist.set_xlabel("Age (Years)", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.legend(title="Income Class", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 4. Education Level by Income Class
# --------------------------------------------------------------------------
def plot_education_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 4: Educational attainment by income class."""
    if output_path is None:
        output_path = FIGURES_DIR / "04_education_level_by_income.png"

    # Order education levels by education-num
    edu_order = (
        df.groupby("education")["education-num"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    ct = pd.crosstab(df["education"], df["income"])[["<=50K", ">50K"]].reindex(edu_order)
    ct_prop = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax2 = ax.twiny()

    # Stacked horizontal bar chart of percentages
    ct_prop.plot(
        kind="barh",
        stacked=True,
        color=["#2b5c8f", "#d95f02"],
        ax=ax,
        edgecolor="black",
        alpha=0.85,
    )

    for i, (idx, row) in enumerate(ct_prop.iterrows()):
        pct_high = row[">50K"]
        total_n = ct.loc[idx].sum()
        ax.text(
            101,
            i,
            f">50K: {pct_high:.1f}% (n={total_n:,})",
            va="center",
            fontsize=8.5,
            fontweight="bold" if pct_high > 40 else "normal",
        )

    ax.set_title("Proportion of Income Classes across Education Levels", fontsize=12, pad=25, fontweight="bold")
    ax.set_xlabel("Percentage within Education Level (%)", fontsize=11)
    ax.set_ylabel("Education Level (Ordered from Low to High)", fontsize=11)
    ax.set_xlim(0, 140)
    ax.legend(title="Income", loc="lower right", frameon=True)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 5. Occupation by Income Class
# --------------------------------------------------------------------------
def plot_occupation_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 5: Occupation distribution split by income class."""
    if output_path is None:
        output_path = FIGURES_DIR / "05_occupation_by_income.png"

    # Handle missing / unknown occupation
    df_plot = df.copy()
    df_plot["occupation"] = df_plot["occupation"].fillna("Missing/Unknown")

    ct = pd.crosstab(df_plot["occupation"], df_plot["income"])[["<=50K", ">50K"]]
    # Sort by proportion of >50K
    ct["pct_high"] = (ct[">50K"] / ct.sum(axis=1)) * 100
    ct = ct.sort_values(by="pct_high", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    ct[["<=50K", ">50K"]].plot(
        kind="barh",
        stacked=True,
        color=["#2b5c8f", "#d95f02"],
        ax=ax,
        edgecolor="black",
        alpha=0.85,
    )

    for i, (idx, row) in enumerate(ct.iterrows()):
        pct_high = row["pct_high"]
        total_n = row["<=50K"] + row[">50K"]
        ax.text(
            total_n + max(ct["<=50K"] + ct[">50K"]) * 0.01,
            i,
            f">50K: {pct_high:.1f}%",
            va="center",
            fontsize=8.5,
            fontweight="bold" if pct_high > 40 else "normal",
        )

    ax.set_title("Occupation Categories and Income Distribution (Sorted by % >50K)", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Number of Individuals", fontsize=11)
    ax.set_ylabel("Occupation", fontsize=11)
    ax.set_xlim(0, max(ct["<=50K"] + ct[">50K"]) * 1.22)
    ax.legend(title="Income", loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 6. Hours Per Week by Income Class
# --------------------------------------------------------------------------
def plot_hours_per_week(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 6: Working hours per week by income class."""
    if output_path is None:
        output_path = FIGURES_DIR / "06_hours_per_week_by_income.png"

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}, dpi=200
    )

    palette = {"<=50K": "#2b5c8f", ">50K": "#d95f02"}

    sns.boxplot(
        data=df,
        x="hours-per-week",
        y="income",
        hue="income",
        palette=palette,
        ax=ax_box,
        orient="h",
        legend=False,
    )
    ax_box.set(xlabel="", ylabel="")
    ax_box.set_title("Hours Worked per Week by Income Class", fontsize=12, pad=10, fontweight="bold")

    sns.histplot(
        data=df,
        x="hours-per-week",
        hue="income",
        palette=palette,
        element="step",
        stat="density",
        common_norm=False,
        bins=30,
        ax=ax_hist,
        alpha=0.4,
    )

    med_low = df[df["income"] == "<=50K"]["hours-per-week"].median()
    med_high = df[df["income"] == ">50K"]["hours-per-week"].median()

    ax_hist.axvline(40, color="gray", linestyle=":", linewidth=1.2, label="Standard (40 hrs/wk)")
    ax_hist.axvline(med_high, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Median >50K ({med_high:.0f} hrs)")

    ax_hist.set_xlabel("Hours Worked per Week", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.legend(title="Income Class", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 7. Capital Gain and Loss Distribution
# --------------------------------------------------------------------------
def plot_capital_gain_loss(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 7: Distribution of capital-gain and capital-loss, showing zero-inflation and top-coding."""
    if output_path is None:
        output_path = FIGURES_DIR / "07_capital_gain_loss_distribution.png"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=200)

    # 1. Overall Capital Gain
    axes[0, 0].hist(df["capital-gain"], bins=30, color="#2b5c8f", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title(
        f"Capital Gain (Full, Zero = {(df['capital-gain'] == 0).mean()*100:.1f}%)",
        fontsize=10,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("USD ($)")
    axes[0, 0].set_ylabel("Count")

    # 2. Non-zero Capital Gain (Log Scale)
    nonzero_gain = df[df["capital-gain"] > 0]["capital-gain"]
    axes[0, 1].hist(nonzero_gain, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title(
        f"Capital Gain (Non-Zero Only, n = {len(nonzero_gain):,})",
        fontsize=10,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("USD ($) [Note: 99,999 is top-coded limit]")
    axes[0, 1].set_ylabel("Count")

    # 3. Overall Capital Loss
    axes[1, 0].hist(df["capital-loss"], bins=30, color="#d95f02", edgecolor="black", alpha=0.8)
    axes[1, 0].set_title(
        f"Capital Loss (Full, Zero = {(df['capital-loss'] == 0).mean()*100:.1f}%)",
        fontsize=10,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("USD ($)")
    axes[1, 0].set_ylabel("Count")

    # 4. Non-zero Capital Loss
    nonzero_loss = df[df["capital-loss"] > 0]["capital-loss"]
    axes[1, 1].hist(nonzero_loss, bins=30, color="#d95f02", edgecolor="black", alpha=0.8)
    axes[1, 1].set_title(
        f"Capital Loss (Non-Zero Only, n = {len(nonzero_loss):,})",
        fontsize=10,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("USD ($)")
    axes[1, 1].set_ylabel("Count")

    plt.suptitle("Distributions of Capital Gain and Capital Loss (Severe Zero-Inflation & Top-Coding)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 8. Numeric Correlation Heatmap
# --------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 8: Pearson correlation heatmap among numeric features and income."""
    if output_path is None:
        output_path = FIGURES_DIR / "08_numeric_correlation_heatmap.png"

    df_corr = df.copy()
    df_corr["income_positive"] = (df_corr["income"] == ">50K").astype(int)

    num_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "income_positive"]
    if "fnlwgt" in df_corr.columns:
        num_cols.insert(1, "fnlwgt")

    corr = df_corr[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Correlation Heatmap of Numeric Features with Income (>50K = 1)", fontsize=12, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 9. Model Metrics Comparison
# --------------------------------------------------------------------------
def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Fig 9: Grouped bar chart comparing key metrics across all evaluated models."""
    if output_path is None:
        output_path = FIGURES_DIR / "09_model_metrics_comparison.png"

    metrics_to_plot = ["Accuracy", "F1_Positive", "Balanced_Accuracy", "ROC_AUC", "PR_AUC"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    df_melt = comparison_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    sns.barplot(
        data=df_melt,
        x="Model",
        y="Score",
        hue="Metric",
        ax=ax,
        palette="tab10",
        edgecolor="black",
        alpha=0.9,
    )

    ax.set_title("Performance Comparison Across Evaluated Models", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xlabel("Model Name", fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="Evaluation Metric", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 10. Best Model Confusion Matrix
# --------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
    output_path: Optional[Path] = None,
) -> Path:
    """Fig 10: Confusion matrix heatmap of the best performing model."""
    if output_path is None:
        output_path = FIGURES_DIR / "10_best_model_confusion_matrix.png"

    cm = confusion_matrix(y_true, y_pred, labels=["<=50K", ">50K"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted <=50K", "Predicted >50K"],
        yticklabels=["Actual <=50K", "Actual >50K"],
        ax=ax,
        linewidths=1,
        linecolor="black",
    )

    ax.set_title(f"Confusion Matrix on Test Set ({model_name})", fontsize=12, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 11. ROC Curves
# --------------------------------------------------------------------------
def plot_roc_curves(
    models_dict: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Fig 11: ROC curves comparing multiple models.
    models_dict structure: {model_name: {'y_true': y_true, 'y_score': y_score, 'roc_auc': auc_val}}
    """
    if output_path is None:
        output_path = FIGURES_DIR / "11_roc_curves.png"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.500)")

    colors = sns.color_palette("tab10", len(models_dict))

    for idx, (name, res) in enumerate(models_dict.items()):
        y_true = res["y_true"]
        y_score = res["y_score"]
        # Convert to binary
        y_binary = (y_true == ">50K").astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = res.get("roc_auc", auc(fpr, tpr))
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[idx], linewidth=1.8)

    ax.set_title("Receiver Operating Characteristic (ROC) Curves - Test Set", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 12. Precision-Recall Curves
# --------------------------------------------------------------------------
def plot_pr_curves(
    models_dict: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Fig 12: Precision-Recall curves comparing multiple models.
    models_dict structure: {model_name: {'y_true': y_true, 'y_score': y_score, 'pr_auc': pr_auc_val}}
    """
    if output_path is None:
        output_path = FIGURES_DIR / "12_precision_recall_curves.png"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # Calculate baseline prevalence
    first_key = list(models_dict.keys())[0]
    first_y = models_dict[first_key]["y_true"]
    y_binary_sample = (first_y == ">50K").astype(int)
    baseline_prev = y_binary_sample.mean()

    ax.axhline(
        baseline_prev,
        color="gray",
        linestyle="--",
        label=f"Baseline Prevalence ({baseline_prev:.3f})",
    )

    colors = sns.color_palette("tab10", len(models_dict))

    for idx, (name, res) in enumerate(models_dict.items()):
        y_true = res["y_true"]
        y_score = res["y_score"]
        y_binary = (y_true == ">50K").astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, y_score)
        ap = res.get("pr_auc", average_precision_score(y_binary, y_score))
        ax.plot(recall, precision, label=f"{name} (PR-AUC / AP = {ap:.3f})", color=colors[idx], linewidth=1.8)

    ax.set_title("Precision-Recall (PR) Curves - Test Set (Positive Class: >50K)", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 13. Feature Importance
# --------------------------------------------------------------------------
def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Top Permutation Feature Importances (Best Model)",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Fig 13: Feature importance or Permutation importance bar chart.
    importance_df requires columns ['feature', 'importance'] and optionally 'std'.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "13_feature_importance.png"

    df_top = importance_df.sort_values(by="importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    bars = ax.barh(
        df_top["feature"],
        df_top["importance"],
        xerr=df_top["std"] if "std" in df_top.columns else None,
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.85,
        capsize=3,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(df_top["importance"]) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.4f}",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Metric (Permutation Importance)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_xlim(0, max(df_top["importance"]) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------
# 14. Fairness by Sex and Race
# --------------------------------------------------------------------------
def plot_fairness_analysis(
    fairness_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Fig 14: Subgroup performance and fairness metrics comparison across sex and race.
    fairness_df columns: ['Group_Type', 'Subgroup', 'Sample_Size', 'Selection_Rate', 'Precision', 'Recall', 'FPR', 'F1']
    """
    if output_path is None:
        output_path = FIGURES_DIR / "14_fairness_by_sex_and_race.png"

    fig, (ax_sex, ax_race) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=200)

    # 1. Sex subgroups
    sex_df = fairness_df[fairness_df["Group_Type"] == "Sex"].copy()
    metrics = ["Selection_Rate", "Recall", "FPR", "Precision", "F1"]
    sex_melt = sex_df.melt(
        id_vars=["Subgroup"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    sns.barplot(
        data=sex_melt,
        x="Subgroup",
        y="Score",
        hue="Metric",
        ax=ax_sex,
        palette="tab10",
        edgecolor="black",
        alpha=0.85,
    )
    ax_sex.set_title("Subgroup Performance by Sex", fontsize=11, pad=10, fontweight="bold")
    ax_sex.set_ylabel("Metric Value / Rate", fontsize=11)
    ax_sex.set_xlabel("Sex", fontsize=11)
    ax_sex.set_ylim(0, 1.05)
    ax_sex.legend(title="Metric", loc="upper right", frameon=True)

    # Add sample size labels
    for idx, row in sex_df.iterrows():
        ax_sex.text(
            row["Subgroup"],
            0.03,
            f"n={int(row['Sample_Size']):,}",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # 2. Race subgroups
    race_df = fairness_df[fairness_df["Group_Type"] == "Race"].copy()
    race_melt = race_df.melt(
        id_vars=["Subgroup"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    sns.barplot(
        data=race_melt,
        x="Subgroup",
        y="Score",
        hue="Metric",
        ax=ax_race,
        palette="tab10",
        edgecolor="black",
        alpha=0.85,
    )
    ax_race.set_title("Subgroup Performance by Race", fontsize=11, pad=10, fontweight="bold")
    ax_race.set_xlabel("Race", fontsize=11)
    plt.setp(ax_race.get_xticklabels(), rotation=20, ha="right")
    ax_race.legend_.remove()  # Shared legend on ax_sex

    for idx, row in race_df.iterrows():
        ax_race.text(
            row["Subgroup"],
            0.03,
            f"n={int(row['Sample_Size']):,}",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    plt.suptitle("Fairness Audit: Subgroup Disparity Across Sex and Race (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
