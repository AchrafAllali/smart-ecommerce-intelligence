"""
Customer Churn Prediction using supervised Machine Learning.

Approach:
    1. Define churn: customer inactive for > CHURN_DAYS (default 180).
    2. Build features: RFM + behavioral (returns, payment, delivery, etc.)
    3. Train models: Logistic Regression, Random Forest, XGBoost (optional).
    4. Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC.
    5. Save best model + predictions to SQLite.
    6. Visualize feature importance + confusion matrix.
"""

# ------------------------------------------------------------
# Fix joblib CPU detection on Windows
# ------------------------------------------------------------
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import sqlite3
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = "database/ecommerce.db"
OUT_DIR = "dashboard"
MODEL_DIR = "ml/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

CHURN_DAYS = 180          # customer considered churned if recency > 180 days
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# Helpers
# ============================================================
def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def save(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"   [SAVED] {path}")


# ============================================================
# 1. Load data
# ============================================================
section("[1] Loading data")

conn = sqlite3.connect(DB_PATH)
fact_sales = pd.read_sql_query("SELECT * FROM fact_sales", conn)
reviews = pd.read_sql_query("SELECT * FROM reviews", conn)
conn.close()

fact_sales["order_purchase_timestamp"] = pd.to_datetime(
    fact_sales["order_purchase_timestamp"], errors="coerce"
)
print(f"   fact_sales: {fact_sales.shape}")
print(f"   reviews:    {reviews.shape}")


# ============================================================
# 2. Build customer-level features
# ============================================================
section("[2] Building customer features")

snapshot_date = fact_sales["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
print(f"   Snapshot date: {snapshot_date.date()}")

# Aggregate per customer
cust = (
    fact_sales.groupby("customer_unique_id")
    .agg(
        recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("total_amount", "sum"),
        avg_order_value=("total_amount", "mean"),
        total_freight=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
        n_products=("product_id", "nunique"),
        n_categories=("product_category_name_english", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_payment_methods=("payment_type", "nunique"),
        avg_installments=("payment_installments", "mean"),
    )
    .reset_index()
)

# Customer lifetime (days)
cust["lifetime_days"] = (cust["last_purchase"] - cust["first_purchase"]).dt.days
cust["purchase_frequency"] = cust["frequency"] / (cust["lifetime_days"] + 1)
cust["monetary_per_order"] = cust["monetary"] / cust["frequency"]

# Reviews: avg score + number of reviews
review_agg = (
    reviews.groupby("review_id")  # placeholder; reviews are by order
    .size()
    .reset_index(name="n")
)
# Link reviews to customers via order_id
reviews_per_customer = (
    fact_sales[["order_id", "customer_unique_id"]]
    .drop_duplicates()
    .merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
    .groupby("customer_unique_id")
    .agg(avg_review_score=("review_score", "mean"),
         n_reviews=("review_score", "count"))
    .reset_index()
)
cust = cust.merge(reviews_per_customer, on="customer_unique_id", how="left")
cust["avg_review_score"] = cust["avg_review_score"].fillna(cust["avg_review_score"].median())

print(f"   Customer features: {cust.shape}")
print(f"   Columns: {list(cust.columns)}")


# ============================================================
# 3. Define churn label
# ============================================================
section("[3] Defining churn label")

cust["churn"] = (cust["recency"] > CHURN_DAYS).astype(int)

churn_rate = cust["churn"].mean() * 100
print(f"   Churn threshold: {CHURN_DAYS} days")
print(f"   Churn rate: {churn_rate:.2f}%")
print(f"   Churned:     {cust['churn'].sum():,}")
print(f"   Not churned: {(1 - cust['churn']).sum():,}")


# ============================================================
# 4. Prepare features / target
# ============================================================
section("[4] Preparing features / target")

feature_cols = [
    "frequency",
    "monetary",
    "avg_order_value",
    "total_freight",
    "n_items",
    "n_products",
    "n_categories",
    "n_sellers",
    "n_payment_methods",
    "avg_installments",
    "lifetime_days",
    "purchase_frequency",
    "monetary_per_order",
    "avg_review_score",
    "n_reviews",
]

X = cust[feature_cols].copy()
y = cust["churn"].copy()

# Handle missing / infinite
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"   X: {X.shape}")
print(f"   y: {y.shape}  |  churn rate = {y.mean()*100:.2f}%")

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")

# Scale (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. Train models
# ============================================================
section("[5] Training models")

results = {}

# --- 5.1 Logistic Regression ---
print("\n   [5.1] Logistic Regression")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

results["LogisticRegression"] = {
    "model": lr,
    "y_pred": y_pred_lr,
    "y_proba": y_proba_lr,
    "accuracy": accuracy_score(y_test, y_pred_lr),
    "precision": precision_score(y_test, y_pred_lr),
    "recall": recall_score(y_test, y_pred_lr),
    "f1": f1_score(y_test, y_pred_lr),
    "roc_auc": roc_auc_score(y_test, y_proba_lr),
}

# --- 5.2 Random Forest ---
print("   [5.2] Random Forest")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

results["RandomForest"] = {
    "model": rf,
    "y_pred": y_pred_rf,
    "y_proba": y_proba_rf,
    "accuracy": accuracy_score(y_test, y_pred_rf),
    "precision": precision_score(y_test, y_pred_rf),
    "recall": recall_score(y_test, y_pred_rf),
    "f1": f1_score(y_test, y_pred_rf),
    "roc_auc": roc_auc_score(y_test, y_proba_rf),
}

# --- 5.3 XGBoost (optional) ---
try:
    from xgboost import XGBClassifier
    print("   [5.3] XGBoost")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

    results["XGBoost"] = {
        "model": xgb,
        "y_pred": y_pred_xgb,
        "y_proba": y_proba_xgb,
        "accuracy": accuracy_score(y_test, y_pred_xgb),
        "precision": precision_score(y_test, y_pred_xgb),
        "recall": recall_score(y_test, y_pred_xgb),
        "f1": f1_score(y_test, y_pred_xgb),
        "roc_auc": roc_auc_score(y_test, y_proba_xgb),
    }
except ImportError:
    print("   [5.3] XGBoost not installed (skipping)")


# ============================================================
# 6. Compare models
# ============================================================
section("[6] Model comparison")

comparison = pd.DataFrame({
    name: {
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
        "roc_auc": r["roc_auc"],
    }
    for name, r in results.items()
}).T.round(4)

print(comparison.to_string())

# Best model by ROC-AUC
best_name = comparison["roc_auc"].idxmax()
best_result = results[best_name]
print(f"\n   --> Best model: {best_name} (ROC-AUC = {best_result['roc_auc']:.4f})")


# ============================================================
# 7. Visualizations
# ============================================================
section("[7] Visualizations")

# 7.1 Confusion matrices
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
if n_models == 1:
    axes = [axes]
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
save(fig, "churn_confusion_matrices.png")

# 7.2 ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "churn_roc_curves.png")

# 7.3 Feature importance (Random Forest)
fig, ax = plt.subplots(figsize=(10, 8))
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
importances.plot.barh(ax=ax, color="steelblue")
ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
save(fig, "churn_feature_importance.png")

# 7.4 Probability distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, r) in zip(axes, list(results.items())[:2]):
    sns.histplot(
        r["y_proba"][y_test == 0], bins=30, alpha=0.5,
        label="Not churned", ax=ax, color="green", stat="density"
    )
    sns.histplot(
        r["y_proba"][y_test == 1], bins=30, alpha=0.5,
        label="Churned", ax=ax, color="red", stat="density"
    )
    ax.set_title(f"{name} - Churn Probability Distribution", fontweight="bold")
    ax.set_xlabel("Churn Probability")
    ax.legend()
plt.tight_layout()
save(fig, "churn_probability_distribution.png")


# ============================================================
# 8. Save best model + predictions
# ============================================================
section("[8] Saving best model & predictions")

# Save model
model_path = os.path.join(MODEL_DIR, f"churn_{best_name}.pkl")
joblib.dump(best_result["model"], model_path)
joblib.dump(scaler, os.path.join(MODEL_DIR, "churn_scaler.pkl"))
print(f"   [OK] Model saved: {model_path}")

# Predict churn probability for ALL customers
if best_name == "LogisticRegression":
    X_all_scaled = scaler.transform(X.fillna(X.median()))
    cust["churn_probability"] = best_result["model"].predict_proba(X_all_scaled)[:, 1]
else:
    cust["churn_probability"] = best_result["model"].predict_proba(
        X.fillna(X.median())
    )[:, 1]

cust["churn_prediction"] = (cust["churn_probability"] > 0.5).astype(int)

# Keep useful columns
output = cust[[
    "customer_unique_id",
    "recency",
    "frequency",
    "monetary",
    "lifetime_days",
    "churn",
    "churn_probability",
    "churn_prediction",
]].copy()

conn = sqlite3.connect(DB_PATH)
output.to_sql("customer_churn", conn, if_exists="replace", index=False)
comparison.to_sql("churn_model_comparison", conn, if_exists="replace")
conn.close()

print(f"   [OK] customer_churn: {output.shape}")
print(f"   [OK] churn_model_comparison: {comparison.shape}")


# ============================================================
# 9. Summary
# ============================================================
section("[9] SUMMARY")

print("\n   Model comparison:")
print(comparison.to_string())

print(f"\n   Best model: {best_name}")
print(f"   ROC-AUC:    {best_result['roc_auc']:.4f}")
print(f"   F1 score:   {best_result['f1']:.4f}")

print(f"\n   Churn rate (actual):  {cust['churn'].mean()*100:.2f}%")
print(f"   Churn rate (pred):    {cust['churn_prediction'].mean()*100:.2f}%")

print("\n   Top 5 features (Random Forest):")
top_features = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(5)
for feat, imp in top_features.items():
    print(f"   - {feat:20s}: {imp:.4f}")

print("\n   [SUCCESS] Churn prediction complete!")