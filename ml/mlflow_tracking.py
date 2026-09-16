"""
MLflow tracking for Churn Prediction and Customer Segmentation.

What this script does:
    1. Loads features from SQLite
    2. Trains multiple models (LogReg, RandomForest, XGBoost)
    3. Logs params, metrics, and models to MLflow
    4. Saves the best model

Run:
    python -m ml.mlflow_tracking

Then:
    mlflow ui --port 5000
    → open http://localhost:5000
"""

import os
import sqlite3
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = "database/ecommerce.db"
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "churn_prediction"

# Try importing XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_features():
    """Build customer-level features from SQLite."""
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            customer_unique_id,
            recency,
            frequency,
            monetary,
            R_score,
            F_score,
            M_score,
            CASE
                WHEN monetary > 0 THEN ROUND(monetary / (frequency + 1), 2)
                ELSE 0
            END AS avg_order_value,
            CASE
                WHEN frequency > 0 THEN ROUND(monetary / frequency, 2)
                ELSE 0
            END AS monetary_per_order
        FROM customer_segments
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Churn = recency > 180 days
    df["churn"] = (df["recency"] > 180).astype(int)
    return df


# ------------------------------------------------------------
# MLflow setup
# ------------------------------------------------------------
def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[INFO] MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"[INFO] Experiment: {EXPERIMENT_NAME}")


# ------------------------------------------------------------
# Model training + logging
# ------------------------------------------------------------
def train_and_log(model, name, X_train, X_test, y_train, y_test, needs_scaling=False):
    """Train a model and log everything to MLflow."""

    with mlflow.start_run(run_name=name):
        print(f"\n[RUN] {name}")

        # Optional scaling
        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Log params
        mlflow.log_param("model_type", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("needs_scaling", needs_scaling)
        mlflow.log_param("n_features", X_train.shape[1])

        # Log model-specific params
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)
        if hasattr(model, "C"):
            mlflow.log_param("C", model.C)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            print(f"   {k:12s}: {v:.4f}")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        return metrics


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=" * 60)
    print("MLflow Tracking — Churn Prediction")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading features from SQLite...")
    df = load_features()
    print(f"   Rows: {len(df):,}")
    print(f"   Churn rate: {df['churn'].mean() * 100:.2f}%")

    # 2. Prepare data
    print("\n[2/4] Preparing train/test split...")
    feature_cols = [
        "frequency",
        "monetary",
        "R_score",
        "F_score",
        "M_score",
        "avg_order_value",
        "monetary_per_order",
    ]

    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    # 3. Setup MLflow
    print("\n[3/4] Setting up MLflow...")
    setup_mlflow()

    # 4. Train models
    print("\n[4/4] Training models...")

    results = {}

    # Logistic Regression
    results["LogisticRegression"] = train_and_log(
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "LogisticRegression",
        X_train, X_test, y_train, y_test,
        needs_scaling=True,
    )

    # Random Forest
    results["RandomForest"] = train_and_log(
        RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            class_weight="balanced", random_state=42,
        ),
        "RandomForest",
        X_train, X_test, y_train, y_test,
        needs_scaling=False,
    )

    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        results["XGBoost"] = train_and_log(
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="logloss",
            ),
            "XGBoost",
            X_train, X_test, y_train, y_test,
            needs_scaling=False,
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T.round(4)
    print(summary.to_string())

    best_model = summary["roc_auc"].idxmax()
    print(f"\n[BEST] Model: {best_model} (ROC-AUC = {summary.loc[best_model, 'roc_auc']:.4f})")

    print("\n[SUCCESS] MLflow tracking complete!")
    print("\nNext step:")
    print("   mlflow ui --port 5000")
    print("   → open http://localhost:5000")


if __name__ == "__main__":
    main()