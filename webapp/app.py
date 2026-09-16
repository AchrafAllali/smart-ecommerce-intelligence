"""
Flask Web Application for Smart E-Commerce Intelligence Platform.

Pages:
    /home                 -> Landing page (project overview)
    /                     -> Overview (KPIs + monthly trend)
    /sales                -> Sales analytics
    /segments             -> Customer segments
    /churn                -> Churn prediction
    /recommendations      -> Recommendations
    /assistant            -> AI Assistant (LLM + SQL)
    /explorer             -> Data explorer
"""

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = "database/ecommerce.db"
DASHBOARD_DIR = "dashboard"

app = Flask(__name__, template_folder="templates", static_folder="static")


# ------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------
def query_db(sql, params=None):
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
            if isinstance(v, np.integer):
                rec[k] = int(v)
            elif isinstance(v, np.floating):
                rec[k] = float(v)
            elif isinstance(v, np.bool_):
                rec[k] = bool(v)
    return records


# ============================================================
# ROUTES
# ============================================================

# ------------------------------------------------------------
# Home / Landing page
# ------------------------------------------------------------
@app.route("/home")
def home():
    return render_template("home.html", active="home")


# ------------------------------------------------------------
# Overview (KPIs)
# ------------------------------------------------------------
@app.route("/")
def index():
    kpis = query_db("""
        SELECT
            ROUND(SUM(total_amount), 2) AS total_revenue,
            COUNT(DISTINCT order_id)    AS total_orders,
            COUNT(DISTINCT customer_unique_id) AS total_customers,
            ROUND(SUM(total_amount) / COUNT(DISTINCT order_id), 2) AS avg_order_value
        FROM fact_sales
    """).iloc[0].to_dict()

    monthly = query_db("""
        SELECT year_month,
               COUNT(DISTINCT order_id)    AS orders,
               ROUND(SUM(total_amount), 2) AS revenue
        FROM fact_sales
        WHERE year_month >= '2017-01'
        GROUP BY year_month
        ORDER BY year_month
    """)

    return render_template(
        "index.html",
        kpis=kpis,
        monthly=df_to_records(monthly),
        active="overview",
    )


# ------------------------------------------------------------
# Sales analytics
# ------------------------------------------------------------
@app.route("/sales")
def sales():
    top_cats = query_db("""
        SELECT product_category_name_english AS category,
               COUNT(DISTINCT order_id) AS orders,
               ROUND(SUM(total_amount), 2) AS revenue
        FROM fact_sales
        GROUP BY product_category_name_english
        ORDER BY revenue DESC
        LIMIT 10
    """)

    top_cities = query_db("""
        SELECT customer_city || ' (' || customer_state || ')' AS city,
               COUNT(DISTINCT order_id) AS orders,
               ROUND(SUM(total_amount), 2) AS revenue
        FROM fact_sales
        GROUP BY customer_city, customer_state
        ORDER BY revenue DESC
        LIMIT 10
    """)

    pay = query_db("""
        SELECT payment_type, COUNT(*) AS transactions,
               ROUND(SUM(payment_value), 2) AS total
        FROM payments
        GROUP BY payment_type
        ORDER BY total DESC
    """)

    reviews = query_db("""
        SELECT review_score, COUNT(*) AS total
        FROM reviews
        GROUP BY review_score
        ORDER BY review_score
    """)

    return render_template(
        "sales.html",
        top_cats=df_to_records(top_cats),
        top_cities=df_to_records(top_cities),
        payments=df_to_records(pay),
        reviews=df_to_records(reviews),
        active="sales",
    )


# ------------------------------------------------------------
# Customer segments
# ------------------------------------------------------------
@app.route("/segments")
def segments():
    segments = query_db("""
        SELECT segment,
               COUNT(*) AS n_customers,
               ROUND(AVG(recency), 2) AS avg_recency,
               ROUND(AVG(frequency), 2) AS avg_frequency,
               ROUND(AVG(monetary), 2) AS avg_monetary
        FROM customer_segments
        GROUP BY segment
        ORDER BY avg_monetary DESC
    """)

    customer_id = request.args.get("customer_id", "").strip()
    lookup = None
    if customer_id:
        result = query_db("""
            SELECT customer_unique_id, segment, recency, frequency, monetary
            FROM customer_segments
            WHERE customer_unique_id = ?
        """, params=[customer_id])
        if not result.empty:
            lookup = df_to_records(result)[0]

    return render_template(
        "segments.html",
        segments=df_to_records(segments),
        lookup=lookup,
        customer_id=customer_id,
        active="segments",
    )


# ------------------------------------------------------------
# Churn prediction
# ------------------------------------------------------------
@app.route("/churn")
def churn():
    stats = query_db("""
        SELECT
            COUNT(*) AS total,
            SUM(churn) AS churned,
            ROUND(AVG(churn_probability), 4) AS avg_prob
        FROM customer_churn
    """).iloc[0].to_dict()

    top_risk = query_db("""
        SELECT customer_unique_id, recency, frequency, monetary, churn_probability
        FROM customer_churn
        WHERE churn_prediction = 1
        ORDER BY churn_probability DESC
        LIMIT 20
    """)

    customer_id = request.args.get("customer_id", "").strip()
    lookup = None
    if customer_id:
        result = query_db("""
            SELECT customer_unique_id, recency, frequency, monetary,
                   churn_probability, churn_prediction
            FROM customer_churn
            WHERE customer_unique_id = ?
        """, params=[customer_id])
        if not result.empty:
            lookup = df_to_records(result)[0]

    return render_template(
        "churn.html",
        stats=stats,
        top_risk=df_to_records(top_risk),
        lookup=lookup,
        customer_id=customer_id,
        active="churn",
    )


# ------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------
@app.route("/recommendations")
def recommendations():
    customer_id = request.args.get("customer_id", "").strip()
    top_n = int(request.args.get("top_n", 10))

    recos = None
    fallback = False
    if customer_id:
        result = query_db("""
            SELECT product_id, rank, score
            FROM recommendations
            WHERE customer_unique_id = ?
            ORDER BY rank
            LIMIT ?
        """, params=[customer_id, top_n])
        if result.empty:
            fallback = True
            result = query_db("""
                SELECT product_id, category, n_customers
                FROM popular_products
                ORDER BY n_customers DESC
                LIMIT ?
            """, params=[top_n])
        recos = df_to_records(result)

    popular = query_db("""
        SELECT product_id, category, n_purchases, n_customers
        FROM popular_products
        ORDER BY n_customers DESC
        LIMIT 10
    """)

    return render_template(
        "recommendations.html",
        customer_id=customer_id,
        top_n=top_n,
        recos=recos,
        fallback=fallback,
        popular=df_to_records(popular),
        active="recommendations",
    )


# ------------------------------------------------------------
# AI Assistant
# ------------------------------------------------------------
@app.route("/assistant")
def assistant_page():
    return render_template("assistant.html", active="assistant")


@app.route("/api/assistant", methods=["POST"])
def assistant_api():
    from rag.sql_agent import ask_safe

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"success": False, "answer": "No question provided"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"success": False, "answer": "Empty question"}), 400

    result = ask_safe(question)
    return jsonify(result)


# ------------------------------------------------------------
# Data explorer
# ------------------------------------------------------------
@app.route("/explorer")
def explorer():
    tables = query_db("""
        SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
    """)["name"].tolist()

    selected = request.args.get("table", tables[0] if tables else "")
    limit = int(request.args.get("limit", 100))

    df = None
    if selected:
        df = query_db(f"SELECT * FROM {selected} LIMIT {limit}")

    return render_template(
        "explorer.html",
        tables=tables,
        selected=selected,
        limit=limit,
        data=df_to_records(df) if df is not None else [],
        columns=list(df.columns) if df is not None else [],
        active="explorer",
    )


# ------------------------------------------------------------
# Serve generated PNGs from dashboard/
# ------------------------------------------------------------
@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(os.path.abspath(DASHBOARD_DIR), filename)


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)