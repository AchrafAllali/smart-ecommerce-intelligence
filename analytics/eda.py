"""
Exploratory Data Analysis (EDA) for Smart E-Commerce Intelligence Platform.
Generates visualizations and saves them to dashboard/.
"""

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Config
DB_PATH = "database/ecommerce.db"
OUT_DIR = "dashboard"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11


# ============================================================
# Helpers
# ============================================================
def load_table(conn, name):
    return pd.read_sql_query(f"SELECT * FROM {name}", conn)


def save(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"   [SAVED] {path}")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. Load data
# ============================================================
section("[1] Loading data from SQLite")

conn = sqlite3.connect(DB_PATH)

fact_sales = load_table(conn, "fact_sales")
orders = load_table(conn, "orders")
customers = load_table(conn, "dim_customers")
products = load_table(conn, "dim_products")
reviews = load_table(conn, "reviews")
payments = load_table(conn, "payments")

print(f"   fact_sales: {fact_sales.shape}")
print(f"   orders:     {orders.shape}")
print(f"   customers:  {customers.shape}")
print(f"   products:   {products.shape}")
print(f"   reviews:    {reviews.shape}")
print(f"   payments:   {payments.shape}")

# Parse dates
date_cols = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in date_cols:
    if col in fact_sales.columns:
        fact_sales[col] = pd.to_datetime(fact_sales[col], errors="coerce")
    if col in orders.columns:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")


# ============================================================
# 2. Missing values
# ============================================================
section("[2] Missing values in fact_sales")

missing = fact_sales.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if len(missing) == 0:
    print("   No missing values.")
else:
    print(missing.to_string())


# ============================================================
# 3. Monthly revenue & orders trend
# ============================================================
section("[3] Monthly revenue & orders trend")

monthly = (
    fact_sales.groupby("year_month")
    .agg(revenue=("total_amount", "sum"), orders=("order_id", "nunique"))
    .reset_index()
)
monthly = monthly[monthly["year_month"] >= "2017-01"]

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.bar(monthly["year_month"], monthly["revenue"], color="steelblue", alpha=0.7, label="Revenue")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue (BRL)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

ax2 = ax1.twinx()
ax2.plot(monthly["year_month"], monthly["orders"], color="red", marker="o", linewidth=2, label="Orders")
ax2.set_ylabel("Orders", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Monthly Revenue & Orders (2017-2018)", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "monthly_trend.png")


# ============================================================
# 4. Top 10 categories by revenue
# ============================================================
section("[4] Top 10 categories by revenue")

top_cat = (
    fact_sales.groupby("product_category_name_english")
    .agg(revenue=("total_amount", "sum"), orders=("order_id", "nunique"))
    .sort_values("revenue", ascending=False)
    .head(10)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_cat, x="revenue", y="product_category_name_english", palette="viridis", ax=ax)
ax.set_title("Top 10 Product Categories by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("Revenue (BRL)")
ax.set_ylabel("")
plt.tight_layout()
save(fig, "top_categories.png")


# ============================================================
# 5. Top 10 cities by revenue
# ============================================================
section("[5] Top 10 cities by revenue")

top_cities = (
    fact_sales.groupby(["customer_city", "customer_state"])
    .agg(revenue=("total_amount", "sum"))
    .sort_values("revenue", ascending=False)
    .head(10)
    .reset_index()
)
top_cities["label"] = top_cities["customer_city"] + " (" + top_cities["customer_state"] + ")"

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_cities, x="revenue", y="label", palette="magma", ax=ax)
ax.set_title("Top 10 Cities by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("Revenue (BRL)")
ax.set_ylabel("")
plt.tight_layout()
save(fig, "top_cities.png")


# ============================================================
# 6. Review score distribution
# ============================================================
section("[6] Review score distribution")

review_counts = reviews["review_score"].value_counts().sort_index()
review_pct = (review_counts / review_counts.sum() * 100).round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(review_counts.index, review_counts.values, color="coral")
axes[0].set_title("Review Score Distribution", fontweight="bold")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Count")

axes[1].pie(review_pct.values, labels=review_pct.index, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Review Score Share", fontweight="bold")
plt.tight_layout()
save(fig, "review_distribution.png")


# ============================================================
# 7. Payment methods
# ============================================================
section("[7] Payment methods")

pay_counts = (
    payments.groupby("payment_type")
    .agg(transactions=("order_id", "count"), value=("payment_value", "sum"))
    .sort_values("value", ascending=False)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=pay_counts, x="payment_type", y="value", palette="Set2", ax=axes[0])
axes[0].set_title("Payment Methods by Value", fontweight="bold")
axes[0].set_ylabel("Total Value (BRL)")
axes[0].tick_params(axis="x", rotation=30)

sns.barplot(data=pay_counts, x="payment_type", y="transactions", palette="Set3", ax=axes[1])
axes[1].set_title("Payment Methods by Count", fontweight="bold")
axes[1].set_ylabel("Transactions")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
save(fig, "payment_methods.png")


# ============================================================
# 8. Delivery time distribution
# ============================================================
section("[8] Delivery time distribution")

delivery = orders.dropna(subset=["order_delivered_customer_date"]).copy()
delivery["delivery_days"] = (
    delivery["order_delivered_customer_date"] -
    delivery["order_purchase_timestamp"]
).dt.days

fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(delivery["delivery_days"], bins=50, kde=True, color="teal", ax=ax)
ax.axvline(delivery["delivery_days"].mean(), color="red", linestyle="--",
           label=f"Mean: {delivery['delivery_days'].mean():.1f} days")
ax.set_title("Delivery Time Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Days")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
save(fig, "delivery_time.png")

print(f"   Mean:   {delivery['delivery_days'].mean():.2f} days")
print(f"   Median: {delivery['delivery_days'].median():.2f} days")
print(f"   Max:    {delivery['delivery_days'].max():.2f} days")


# ============================================================
# 9. RFM distributions
# ============================================================
section("[9] RFM distributions")

snapshot_date = fact_sales["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = (
    fact_sales.groupby("customer_unique_id")
    .agg(
        recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("total_amount", "sum"),
    )
    .reset_index()
)

print(f"   RFM shape: {rfm.shape}")
print(f"   Recency  (days): mean={rfm['recency'].mean():.0f}, median={rfm['recency'].median():.0f}")
print(f"   Frequency:       mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
print(f"   Monetary (BRL):  mean={rfm['monetary'].mean():.2f}, median={rfm['monetary'].median():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, ["recency", "frequency", "monetary"]):
    sns.histplot(rfm[col], bins=50, kde=True, color="steelblue", ax=ax)
    ax.set_title(f"{col.capitalize()} distribution", fontweight="bold")
plt.tight_layout()
save(fig, "rfm_distribution.png")


# ============================================================
# 10. Orders by day of week
# ============================================================
section("[10] Orders by day of week")

dow = (
    fact_sales.groupby("day_of_week")
    .agg(orders=("order_id", "nunique"))
    .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=dow.index, y=dow["orders"], palette="coolwarm", ax=ax)
ax.set_title("Orders by Day of Week", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Orders")
plt.tight_layout()
save(fig, "orders_by_dow.png")


# ============================================================
# 11. Heatmap: month × top categories
# ============================================================
section("[11] Heatmap: category × month")

pivot = fact_sales.pivot_table(
    index="product_category_name_english",
    columns="month",
    values="total_amount",
    aggfunc="sum",
).fillna(0)

top_10_cat = fact_sales.groupby("product_category_name_english")["total_amount"].sum().nlargest(10).index
pivot = pivot.loc[top_10_cat]

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, ax=ax)
ax.set_title("Top 10 Categories × Month Revenue Heatmap", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("")
plt.tight_layout()
save(fig, "category_month_heatmap.png")


# ============================================================
# 12. Key insights summary
# ============================================================
section("[12] KEY INSIGHTS")

total_revenue = fact_sales["total_amount"].sum()
total_orders = fact_sales["order_id"].nunique()
total_customers = fact_sales["customer_unique_id"].nunique()
aov = total_revenue / total_orders

best_month = monthly.loc[monthly["revenue"].idxmax(), "year_month"]
best_month_rev = monthly["revenue"].max()

print(f"   1.  Total Revenue:       {total_revenue:,.2f} BRL")
print(f"   2.  Total Orders:        {total_orders:,}")
print(f"   3.  Total Customers:     {total_customers:,}")
print(f"   4.  Avg Order Value:     {aov:.2f} BRL")
print(f"   5.  Top Category:        {top_cat.iloc[0]['product_category_name_english']}")
print(f"   6.  Top City:            {top_cities.iloc[0]['customer_city']} ({top_cities.iloc[0]['customer_state']})")
print(f"   7.  Best Month:          {best_month} ({best_month_rev:,.2f} BRL)")
print(f"   8.  Avg Delivery:        {delivery['delivery_days'].mean():.2f} days")
print(f"   9.  5-star Reviews:      {(reviews['review_score'] == 5).mean() * 100:.2f}%")
print(f"   10. Credit Card Share:   {(payments['payment_type'] == 'credit_card').mean() * 100:.2f}%")


# ============================================================
# Done
# ============================================================
conn.close()
print("\n" + "=" * 60)
print("[SUCCESS] EDA complete! Check dashboard/ for images.")
print("=" * 60)