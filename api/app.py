"""
Flask REST API for Smart E-Commerce Intelligence Platform.

Endpoints:
    GET  /                         -> API info
    GET  /api/health               -> health check
    GET  /api/kpis                 -> global KPIs
    GET  /api/monthly-sales        -> monthly revenue + orders
    GET  /api/top-categories       -> top categories
    GET  /api/top-cities           -> top cities
    GET  /api/segments             -> segment summary
    GET  /api/segment/<customer_id>-> segment of a given customer
    GET  /api/churn/<customer_id>  -> churn probability for customer
    GET  /api/churn/top-risk       -> top 20 churn-risk customers
    GET  /api/recommendations/<customer_id>  -> top-N recommendations
    GET  /api/popular-products     -> top popular products
"""

import os
import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = "database/ecommerce.db"
MODEL_DIR = "ml/models"

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

# ------------------------------------------------------------
# Load model artifacts at startup
# ------------------------------------------------------------
print("[INFO] Loading model artifacts...")

try:
    customer_products = joblib.load(os.path.join(MODEL_DIR, "customer_products.pkl"))
    customer_categories = joblib.load(os.path.join(MODEL_DIR, "customer_categories.pkl"))
    top_per_category = joblib.load(os.path.join(MODEL_DIR, "top_per_category.pkl"))
    popular_products_df = joblib.load(os.path.join(MODEL_DIR, "popular_products.pkl"))
    print("[OK] Recommendation artifacts loaded")
except Exception as e:
    print(f"[WARNING] Could not load recommendation artifacts: {e}")
    customer_products = {}
    customer_categories = {}
    top_per_category = {}
    popular_products_df = pd.DataFrame()

try:
    churn_model = joblib.load(os.path.join(MODEL_DIR, "churn_XGBoost.pkl"))
    churn_scaler = joblib.load(os.path.join(MODEL_DIR, "churn_scaler.pkl"))
    print("[OK] Churn model loaded")
except Exception as e:
    print(f"[WARNING] Could not load churn model: {e}")
    churn_model = None
    churn_scaler = None


# ------------------------------------------------------------
# DB helper
# ------------------------------------------------------------
def query_db(sql, params=None):
    """Execute a SQL query and return a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if params:
            df = pd.read_sql_query(sql, conn, params=params)
        else:
            df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


def df_to_records(df):
    """Convert DataFrame to JSON-serializable list of dicts."""
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)
    return records


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def root():
    return jsonify({
        "name": "Smart E-Commerce Intelligence API",
        "version": "1.0",
        "endpoints": [
            "/api/health",
            "/api/kpis",
            "/api/monthly-sales",
            "/api/top-categories",
            "/api/top-cities",
            "/api/segments",
            "/api/segment/<customer_id>",
            "/api/churn/<customer_id>",
            "/api/churn/top-risk",
            "/api/recommendations/<customer_id>",
            "/api/popular-products",
        ],
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "db": os.path.exists(DB_PATH)})


# ------------------------------------------------------------
# 1. Global KPIs
# ------------------------------------------------------------
@app.route("/api/kpis")
def kpis():
    df = query_db("""
        SELECT
            ROUND(SUM(total_amount), 2) AS total_revenue,
            COUNT(DISTINCT order_id)    AS total_orders,
            COUNT(DISTINCT customer_unique_id) AS total_customers,
            ROUND(SUM(total_amount) / COUNT(DISTINCT order_id), 2) AS avg_order_value
        FROM fact_sales
    """)
    return jsonify(df_to_records(df)[0])


# ------------------------------------------------------------
# 2. Monthly sales
# ------------------------------------------------------------
@app.route("/api/monthly-sales")
def monthly_sales():
    df = query_db("""
        SELECT
            year_month,
            COUNT(DISTINCT order_id)    AS orders,
            ROUND(SUM(total_amount), 2) AS revenue
        FROM fact_sales
        GROUP BY year_month
        ORDER BY year_month
    """)
    return jsonify(df_to_records(df))


# ------------------------------------------------------------
# 3. Top categories
# ------------------------------------------------------------
@app.route("/api/top-categories")
def top_categories():
    limit = int(request.args.get("limit", 10))
    df = query_db("""
        SELECT
            product_category_name_english AS category,
            COUNT(DISTINCT order_id)      AS orders,
            ROUND(SUM(total_amount), 2)   AS revenue
        FROM fact_sales
        GROUP BY product_category_name_english
        ORDER BY revenue DESC
        LIMIT ?
    """, params=[limit])
    return jsonify(df_to_records(df))


# ------------------------------------------------------------
# 4. Top cities
# ------------------------------------------------------------
@app.route("/api/top-cities")
def top_cities():
    limit = int(request.args.get("limit", 10))
    df = query_db("""
        SELECT
            customer_city,
            customer_state,
            COUNT(DISTINCT order_id)    AS orders,
            ROUND(SUM(total_amount), 2) AS revenue
        FROM fact_sales
        GROUP BY customer_city, customer_state
        ORDER BY revenue DESC
        LIMIT ?
    """, params=[limit])
    return jsonify(df_to_records(df))


# ------------------------------------------------------------
# 5. Segments summary
# ------------------------------------------------------------
@app.route("/api/segments")
def segments():
    df = query_db("""
        SELECT
            segment,
            COUNT(*)                      AS n_customers,
            ROUND(AVG(recency), 2)        AS avg_recency,
            ROUND(AVG(frequency), 2)      AS avg_frequency,
            ROUND(AVG(monetary), 2)       AS avg_monetary
        FROM customer_segments
        GROUP BY segment
        ORDER BY avg_monetary DESC
    """)
    return jsonify(df_to_records(df))


# ------------------------------------------------------------
# 6. Segment of a specific customer
# ------------------------------------------------------------
@app.route("/api/segment/<customer_id>")
def segment_of_customer(customer_id):
    df = query_db("""
        SELECT customer_unique_id, segment, recency, frequency, monetary
        FROM customer_segments
        WHERE customer_unique_id = ?
    """, params=[customer_id])
    if df.empty:
        return jsonify({"error": "Customer not found"}), 404
    return jsonify(df_to_records(df)[0])


# ------------------------------------------------------------
# 7. Churn probability for a customer
# ------------------------------------------------------------
@app.route("/api/churn/<customer_id>")
def churn_of_customer(customer_id):
    df = query_db("""
        SELECT customer_unique_id, recency, frequency, monetary,
               churn_probability, churn_prediction
        FROM customer_churn
        WHERE customer_unique_id = ?
    """, params=[customer_id])
    if df.empty:
        return jsonify({"error": "Customer not found"}), 404
    return jsonify(df_to_records(df)[0])


# ------------------------------------------------------------
# 8. Top 20 churn-risk customers
# ------------------------------------------------------------
@app.route("/api/churn/top-risk")
def churn_top_risk():
    limit = int(request.args.get("limit", 20))
    df = query_db("""
        SELECT customer_unique_id, recency, frequency, monetary,
               churn_probability
        FROM customer_churn
        WHERE churn_prediction = 1
        ORDER BY churn_probability DESC
        LIMIT ?
    """, params=[limit])
    return jsonify(df_to_records(df))


# ------------------------------------------------------------
# 9. Recommendations for a customer
# ------------------------------------------------------------
@app.route("/api/recommendations/<customer_id>")
def recommendations(customer_id):
    top_n = int(request.args.get("top_n", 10))

    # Check DB first
    df = query_db("""
        SELECT product_id, rank, score
        FROM recommendations
        WHERE customer_unique_id = ?
        ORDER BY rank
        LIMIT ?
    """, params=[customer_id, top_n])

    if df.empty:
        # Fallback: return popular products
        popular = popular_products_df.head(top_n)[["product_id", "category"]]
        return jsonify({
            "customer_id": customer_id,
            "fallback": True,
            "recommendations": df_to_records(popular),
        })

    return jsonify({
        "customer_id": customer_id,
        "fallback": False,
        "recommendations": df_to_records(df),
    })


# ------------------------------------------------------------
# 10. Popular products
# ------------------------------------------------------------
@app.route("/api/popular-products")
def popular_products():
    limit = int(request.args.get("limit", 20))
    df = query_db("""
        SELECT product_id, category, n_purchases, n_customers
        FROM popular_products
        ORDER BY n_customers DESC
        LIMIT ?
    """, params=[limit])
    return jsonify(df_to_records(df))


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)