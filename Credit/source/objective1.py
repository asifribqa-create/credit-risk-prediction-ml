"""
Objective 1 -- Default Prediction (Classification)
===================================================
Trains a Random Forest Classifier to predict whether a credit card holder
will default on their next payment using 22 behavioural features from the
Taiwan Credit dataset.

Outputs (saved to output/):
  - confusion_matrix.png
  - roc_curve.png
  - feature_importances.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


# ---------------------------------------------------------------------------
# Helper -- IQR-based outlier capping
# ---------------------------------------------------------------------------
def _cap_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Cap values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# Main callable function
# ---------------------------------------------------------------------------
def run_classification(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    End-to-end classification pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Taiwan Credit data (must include all required columns).
    output_dir : str
        Directory path where plots will be saved.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame (useful downstream for Objectives 2 & 3).
    """
    os.makedirs(output_dir, exist_ok=True)
    data = df.copy()

    # -- 1. Data Cleaning ----------------------------------------------------
    # Map invalid EDUCATION codes (0, 5, 6) -> 4 ("Others")
    data["EDUCATION"] = data["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Map invalid MARRIAGE code (0) -> 3 ("Others")
    data["MARRIAGE"] = data["MARRIAGE"].replace({0: 3})

    # Cap outliers for all BILL_AMT columns using IQR method
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    for col in bill_cols:
        data[col] = _cap_outliers_iqr(data[col])

    # -- 2. Feature / Target Split -------------------------------------------
    feature_cols = (
        ["LIMIT_BAL"]
        + [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)]
    )
    X = data[feature_cols]
    y = data["class"]

    # -- 3. Stratified Train / Test Split (70/30) ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # -- 4. Feature Scaling --------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -- 5. Model Training ---------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    # -- 6. Predictions & Evaluation -----------------------------------------
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("  OBJECTIVE 1 -- Default Prediction Results")
    print("=" * 60)
    print("  Accuracy Score  : {:.4f}".format(acc))
    print("  ROC-AUC Score   : {:.4f}".format(roc))
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    print("=" * 60)

    # -- 7. Visualisations ---------------------------------------------------
    _plot_confusion_matrix(y_test, y_pred, output_dir)
    _plot_roc_curve(y_test, y_proba, output_dir)
    _plot_feature_importances(rf, feature_cols, output_dir)

    return data  # preprocessed df for downstream objectives


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(y_true, y_pred, output_dir):
    """Generate and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix - Random Forest", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  [OK] Saved: {}".format(path))


def _plot_roc_curve(y_true, y_proba, output_dir):
    """Generate and save the ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label="ROC Curve (AUC = {:.4f})".format(roc_auc))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve - Random Forest", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  [OK] Saved: {}".format(path))


def _plot_feature_importances(model, feature_names, output_dir):
    """Generate and save a horizontal bar chart for top-10 feature importances."""
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top10 = feat_imp.tail(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top10)))
    ax.barh(top10.index, top10.values, color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_title("Top 10 Feature Importances - Random Forest", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importances.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  [OK] Saved: {}".format(path))
