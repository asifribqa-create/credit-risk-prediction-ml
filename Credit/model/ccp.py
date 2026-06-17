"""
CCP - Complete CRISP Pipeline Orchestrator (Standard Production Mode)
========================================================================
Master controller that sequentially executes data warehouse ETL processing,
queries dataset keeping native precise case mapping, and executes downstream pipelines.

File Location: model/ccp.py
Usage:
    python model/ccp.py
"""

import os
import sys
import time
import io
import sqlite3
import pandas as pd

# Force UTF-8 stdout so box-drawing and emoji chars render perfectly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup - ensure cross-directory imports work regardless of CWD
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Core resource targets mapping
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "TaiwanData.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "Warehouse_Credit_Risk.db")  
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Import modular operational source layers safely from source directory
from source.database_setup import build_star_schema
from source.objective1 import run_classification
from source.objective2 import run_clustering
from source.objective3 import run_olap


def _banner(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def main():
    pipeline_start = time.time()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prevent Duplicates: Wipe previous Sqlite file securely
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print("🗑️  [SYSTEM] Old database file deleted to enforce clean data types.")
        except Exception:
            pass

    _banner("CRISP-DM MULTI-OBJECTIVE PIPELINE WITH DATA WAREHOUSE")
    print(f"  Project Root     : {PROJECT_ROOT}")
    print(f"  Target Database  : {DB_PATH}")
    print(f"  Output Directory : {OUTPUT_DIR}")

    # ── PHASE 0: DATABASE PIPELINE & ETL EXECUTION ──────────────────────────
    _banner("PHASE 0 -- Data Warehouse Extraction, Transformation & Load")
    build_star_schema(DATA_PATH, DB_PATH)

    # ── PHASE 1: DATA SOURCE EXTRACTION VIA SQL JOIN ────────────────________
    print("\n📥 Extracting Dataset via Data Warehouse Join Query...")
    conn = sqlite3.connect(DB_PATH)
    
    # CASE RESOLVED QUERY: 
    # 'class' column is kept lowercase for Objective 1.
    # 'Age' is aliased as 'AGE' for Objective 3.
    query = """
    SELECT 
        f.Customer_ID as ID, f.LIMIT_BAL, f.Default_Status as class,
        f.PAY_0, f.PAY_2, f.PAY_3, f.PAY_4, f.PAY_5, f.PAY_6,
        f.BILL_AMT1, f.BILL_AMT2, f.BILL_AMT3, f.BILL_AMT4, f.BILL_AMT5, f.BILL_AMT6,
        f.PAY_AMT1, f.PAY_AMT2, f.PAY_AMT3, f.PAY_AMT4, f.PAY_AMT5, f.PAY_AMT6,
        d.Age as AGE,
        CASE WHEN c.Gender = 'Male' THEN 1 ELSE 2 END as SEX,
        CASE 
            WHEN c.Education_Level = 'Graduate School' THEN 1 
            WHEN c.Education_Level = 'University' THEN 2 
            WHEN c.Education_Level = 'High School' THEN 3 
            ELSE 4 
        END as EDUCATION,
        CASE 
            WHEN c.Marital_Status = 'Married' THEN 1 
            WHEN c.Marital_Status = 'Single' THEN 2 
            ELSE 3 
        END as MARRIAGE
    FROM Fact_Credit_Risk f
    JOIN Dim_Customer c ON f.Customer_ID = c.Customer_ID
    JOIN Dim_Demographics d ON f.Demographic_ID = d.Demographic_ID;
    """
    
    df_raw = pd.read_sql_query(query, conn)
    conn.close()

    print(f"  [OK] Successfully retrieved {len(df_raw)} safe numeric rows via SQL Engine.")

    # Segregate features matrix slice safely
    df_ml_input = df_raw.copy()
    if "ID" in df_ml_input.columns:
        df_ml_input.drop(columns=["ID"], inplace=True)

    # ── PHASE 2: OBJECTIVE 1 MODEL RUNNING ──────────────────────────────────
    _banner("PHASE 1 -- Objective 1: Default Prediction (Classification)")
    df_clean = run_classification(df_ml_input, OUTPUT_DIR)
    
    # Re-attach target identity sequence back using local alignment matching
    df_clean["ID"] = df_raw["ID"].values

    # ── PHASE 3: OBJECTIVE 2 SEGMENTATION RUNNING ───────────────────────────
    _banner("PHASE 2 -- Objective 2: Risk Segmentation (Clustering)")
    df_segmented = run_clustering(df_clean, OUTPUT_DIR)

    # ── PHASE 4: OBJECTIVE 3 MULTIDIMENSIONAL SUMMARY RUNNING ───────────────
    _banner("PHASE 3 -- Objective 3: Multi-dimensional OLAP Cube")
    run_olap(df_segmented, OUTPUT_DIR)

    # ── PIPELINE TERMINATION METRICS REPORT ──────────────────────────────────
    _banner("PIPELINE RUN COMPLETED SUCCESSFULLY")
    total = time.time() - pipeline_start
    print("  Total End-to-End Execution Time : {:.2f}s".format(total))
    print("  Database Warehouse Location     : {}".format(DB_PATH))
    print("  Output Reports Location        : {}".format(OUTPUT_DIR))
    print("="*70)


if __name__ == "__main__":
    main()