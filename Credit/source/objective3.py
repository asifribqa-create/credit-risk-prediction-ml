"""
Objective 3 — Data Warehouse + OLAP Cube (with Data Visualization)
====================================================================
Creates human-readable dimension columns, bins AGE and LIMIT_BAL, 
performs a multi-dimensional GroupBy OLAP operation, and automatically
generates a multi-panel breakdown chart of metrics.

Output (saved to output/):
  • olap_cube_report.csv
  • olap_cube_chart.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Background execution layer to prevent process hanging
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Dimension mapping helpers
# ---------------------------------------------------------------------------
def _map_gender(code):
    mapping = {1: "Male", 2: "Female"}
    return mapping.get(code, "Unknown")


def _map_education(code):
    mapping = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
    return mapping.get(code, "Unknown")


def _map_marriage(code):
    mapping = {1: "Married", 2: "Single", 3: "Others"}
    return mapping.get(code, "Unknown")


# ---------------------------------------------------------------------------
# Main callable function
# ---------------------------------------------------------------------------
def run_olap(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Build dimension columns and produce a multi-dimensional OLAP cube report and chart.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed Taiwan Credit data.
    output_dir : str
        Directory path where the OLAP report and graph will be saved.

    Returns
    -------
    pd.DataFrame
        The OLAP cube summary table.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = df.copy()

    # ── 1. Create Dimension Columns ─────────────────────────────────────

    # AGE → AGE_GROUP
    bins_age = [0, 25, 35, 45, 55, np.inf]
    labels_age = ["Under 25", "25-35", "35-45", "45-55", "Above 55"]
    data["AGE_GROUP"] = pd.cut(data["AGE"], bins=bins_age, labels=labels_age, right=False)

    # LIMIT_BAL → CREDIT_TIER (quantile-based, 3 tiers)
    data["CREDIT_TIER"] = pd.qcut(
        data["LIMIT_BAL"], q=3, labels=["Low", "Medium", "High"]
    )

    # SEX → GENDER
    data["GENDER"] = data["SEX"].apply(_map_gender)

    # EDUCATION → EDUCATION_LEVEL
    data["EDUCATION_LEVEL"] = data["EDUCATION"].apply(_map_education)

    # MARRIAGE → MARITAL_STATUS
    data["MARITAL_STATUS"] = data["MARRIAGE"].apply(_map_marriage)

    # ── 2. Define dimensions and measures ───────────────────────────────
    dimensions = ["AGE_GROUP", "CREDIT_TIER", "GENDER", "EDUCATION_LEVEL", "MARITAL_STATUS"]

    # ── 3. Multi-dimensional OLAP GroupBy ───────────────────────────────
    olap_cube = (
        data.groupby(dimensions, observed=False)
        .agg(
            default_rate=("class", "mean"),
            average_bill=("BILL_AMT1", "mean"),
            average_payment=("PAY_AMT1", "mean"),
            customer_count=("class", "count"),
        )
        .reset_index()
        .round(4)
    )

    # ── 4. Export CSV Report ────────────────────────────────────────────
    path = os.path.join(output_dir, "olap_cube_report.csv")
    olap_cube.to_csv(path, index=False)

    print("\n" + "=" * 60)
    print("  OBJECTIVE 3 — OLAP Cube Report")
    print("=" * 60)
    print(f"  Dimensions  : {dimensions}")
    print(f"  Total Cells : {len(olap_cube)}")
    print(f"  Non-Empty   : {(olap_cube['customer_count'] > 0).sum()}")
    print("-" * 60)
    print(olap_cube.head(15).to_string(index=False))
    print("  ...")
    print("-" * 60)
    print("  [OK] Saved: {}".format(path))

    # ── 5. NEW VISUALIZATION LAYER: MULTIDIMENSIONAL OLAP CHART ───────────
    print("📊 [VISUAL] Generating Multi-dimensional OLAP Cube Facet Chart...")
    
    # Filter rows with actual customers to keep plot metrics clear and noise-free
    plot_data = olap_cube[olap_cube["customer_count"] > 0].copy()
    
    # Set up seaborn aesthetic standard styling
    sns.set_theme(style="whitegrid")
    
    # Generate multi-panel categorical plot split by Credit Tiers
    g = sns.catplot(
        data=plot_data,
        x="AGE_GROUP",
        y="default_rate",
        hue="GENDER",
        col="CREDIT_TIER",
        kind="bar",
        palette="muted",
        height=5,
        aspect=0.9,
        errorbar=None
    )
    
    # Fine-tune layouts labels and structural alignment
    g.set_axis_labels("Age Groups", "Average Default Rate")
    g.set_titles("Credit Tier: {col_name}")
    g.set_xticklabels(rotation=30, ha="right")
    
    # Set clean title layout overhead
    plt.subplots_adjust(top=0.82)
    g.fig.suptitle("OLAP Multidimensional Risk Analysis\n(Age Group vs Default Rate across Credit Tiers)", 
                    fontsize=14, fontweight="bold")
    
    # Save chart image layer cleanly inside output folder
    chart_path = os.path.join(output_dir, "olap_cube_chart.png")
    g.savefig(chart_path, dpi=300)
    plt.close()
    
    print("  [OK] Saved OLAP Visualization Graph: {}".format(chart_path))
    print("=" * 60)

    return olap_cube