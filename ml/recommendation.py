"""
Product Recommendation System for Olist (hybrid approach).

Approach:
    1. Category-based: recommend top products from categories the customer
       has already bought from.
    2. Popularity-based: rank products by number of unique customers.
    3. Exclude products already purchased.
    4. Fallback: global top popular products for cold-start.

Why not SVD / Item-CF?
    Olist matrix is 99.96% empty with very few ratings per user.
    SVD gives meaningless scores; item-CF gives all-zero similarities.
    A hybrid category+popularity approach is far more robust.
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

TOP_N = 10
MIN_PRODUCT_PURCHASES = 10   # only recommend products with >= 10 purchases


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
fact_sales = pd.read_sql_query(
    """SELECT order_id, customer_unique_id, product_id,
              product_category_name_english
       FROM fact_sales""",
    conn,
)
conn.close()

print(f"   fact_sales: {fact_sales.shape}")


# ============================================================
# 2. Build product popularity
# ============================================================
section("[2] Building product popularity")

product_stats = (
    fact_sales.groupby("product_id")
    .agg(
        n_purchases=("order_id", "count"),
        n_customers=("customer_unique_id", "nunique"),
        category=("product_category_name_english", "first"),
    )
    .reset_index()
)

# Keep only popular products
popular_products = product_stats[
    product_stats["n_purchases"] >= MIN_PRODUCT_PURCHASES
].copy()

print(f"   Total products:      {len(product_stats):,}")
print(f"   Popular products:    {len(popular_products):,}  (>= {MIN_PRODUCT_PURCHASES} purchases)")

# Rank by n_customers
popular_products = popular_products.sort_values(
    ["n_customers", "n_purchases"], ascending=False
).reset_index(drop=True)
popular_products["popularity_rank"] = popular_products.index + 1

print("\n   Top 10 popular products:")
print(popular_products.head(10)[
    ["product_id", "category", "n_customers", "n_purchases"]
].to_string(index=False))


# ============================================================
# 3. Category-level popularity
# ============================================================
section("[3] Category-level popularity")

category_stats = (
    fact_sales.groupby("product_category_name_english")
    .agg(
        n_purchases=("order_id", "count"),
        n_customers=("customer_unique_id", "nunique"),
    )
    .sort_values("n_customers", ascending=False)
    .reset_index()
)

print("   Top 10 categories:")
print(category_stats.head(10).to_string(index=False))


# ============================================================
# 4. Customer history
# ============================================================
section("[4] Building customer history")

customer_history = (
    fact_sales.groupby("customer_unique_id")
    .agg(
        n_orders=("order_id", "nunique"),
        n_products=("product_id", "nunique"),
        n_categories=("product_category_name_english", "nunique"),
    )
    .reset_index()
)

# Customers with >= 1 purchase
active_customers = customer_history["customer_unique_id"].tolist()
print(f"   Total customers: {len(active_customers):,}")


# ============================================================
# 5. Recommendation function (hybrid)
# ============================================================
section("[5] Recommendation function (hybrid)")

# Precompute: what each customer already bought
customer_products = (
    fact_sales.groupby("customer_unique_id")["product_id"]
    .apply(set)
    .to_dict()
)

# Precompute: what each customer's favorite categories are
customer_categories = (
    fact_sales.groupby("customer_unique_id")["product_category_name_english"]
    .apply(lambda x: list(x.value_counts().index))
    .to_dict()
)

# Precompute: top products per category
top_per_category = (
    popular_products
    .sort_values("n_customers", ascending=False)
    .groupby("category")
    .head(20)
    .groupby("category")["product_id"]
    .apply(list)
    .to_dict()
)


def recommend_for_user(customer_id, top_n=TOP_N):
    """
    Hybrid recommendation:
        1. Get customer's favorite categories.
        2. Recommend top popular products from those categories.
        3. Exclude already purchased.
        4. If not enough, fill with global popular.
    """
    bought = customer_products.get(customer_id, set())
    fav_cats = customer_categories.get(customer_id, [])

    candidates = []

    # Step 1: Top products from customer's favorite categories
    for cat in fav_cats:
        for prod in top_per_category.get(cat, []):
            if prod not in bought and prod not in candidates:
                candidates.append(prod)

    # Step 2: Fill with global popular
    if len(candidates) < top_n:
        for prod in popular_products["product_id"].tolist():
            if prod not in bought and prod not in candidates:
                candidates.append(prod)
            if len(candidates) >= top_n:
                break

    return candidates[:top_n]


# Test on sample user
sample_user = active_customers[0]
sample_bought = customer_products.get(sample_user, set())
sample_cats = customer_categories.get(sample_user, [])
sample_recos = recommend_for_user(sample_user, top_n=TOP_N)

cat_map = popular_products.set_index("product_id")["category"].to_dict()

print(f"   Sample user:        {sample_user}")
print(f"   Products bought:    {len(sample_bought)}")
print(f"   Favorite categories: {sample_cats[:5]}")
print(f"\n   Top {TOP_N} recommendations:")
for i, prod in enumerate(sample_recos, 1):
    cat = cat_map.get(prod, "unknown")
    print(f"   {i:2d}. {prod[:30]}... | {cat}")


# ============================================================
# 6. Evaluation: category hit rate
# ============================================================
section("[6] Evaluating (Category Hit Rate@10)")

def evaluate_category_hit_rate(n_users=500, k=10):
    """
    For each user, hold out 1 product, then check if the top-k
    recommendations contain a product from the SAME category.
    """
    rng = np.random.default_rng(42)
    sampled = rng.choice(active_customers, size=min(n_users, len(active_customers)), replace=False)

    hits = 0
    total = 0

    for user in sampled:
        bought = customer_products.get(user, set())
        if len(bought) < 2:
            continue

        # Hold out 1 product
        bought_list = list(bought)
        holdout = rng.choice(bought_list, 1)[0]
        holdout_cat = fact_sales[fact_sales["product_id"] == holdout]["product_category_name_english"].iloc[0]

        # Temporary remove from history
        train_bought = bought - {holdout}
        customer_products[user] = train_bought

        # Recommend
        recos = recommend_for_user(user, top_n=k)

        # Restore
        customer_products[user] = bought

        # Check if any recommendation is in the same category as holdout
        reco_cats = [cat_map.get(p, "unknown") for p in recos]
        if holdout_cat in reco_cats:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


hit_rate_cat = evaluate_category_hit_rate(n_users=500, k=10)
print(f"   Category Hit Rate@10: {hit_rate_cat*100:.2f}%")
print(f"   (random baseline ≈ {100/category_stats.shape[0]:.2f}%)")


# ============================================================
# 7. Generate recommendations for ALL users
# ============================================================
section("[7] Generating recommendations for all users")

all_recos = []
for user in active_customers:
    recos = recommend_for_user(user, top_n=5)
    for rank, prod in enumerate(recos, 1):
        all_recos.append({
            "customer_unique_id": user,
            "product_id": prod,
            "rank": rank,
            "score": float(10 - rank),  # simple decreasing score
        })

recos_df = pd.DataFrame(all_recos)
print(f"   Generated {len(recos_df):,} recommendations for {recos_df['customer_unique_id'].nunique():,} users")


# ============================================================
# 8. Save to SQLite
# ============================================================
section("[8] Saving to SQLite")

conn = sqlite3.connect(DB_PATH)
recos_df.to_sql("recommendations", conn, if_exists="replace", index=False)
popular_products.to_sql("popular_products", conn, if_exists="replace", index=False)
category_stats.to_sql("category_stats", conn, if_exists="replace", index=False)
conn.close()

print(f"   [OK] recommendations: {recos_df.shape}")
print(f"   [OK] popular_products: {popular_products.shape}")
print(f"   [OK] category_stats: {category_stats.shape}")


# ============================================================
# 9. Visualizations
# ============================================================
section("[9] Visualizations")

# 9.1 Top categories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top10_cats = category_stats.head(10)
sns.barplot(data=top10_cats, x="n_purchases", y="product_category_name_english",
            ax=axes[0], palette="viridis")
axes[0].set_title("Top 10 Categories by Purchases", fontweight="bold")
axes[0].set_xlabel("Purchases")

# 9.2 Top popular products (by n_customers)
top10_prods = popular_products.head(10).copy()
top10_prods["short_id"] = top10_prods["product_id"].str[:10] + "..."
sns.barplot(data=top10_prods, x="n_customers", y="short_id",
            ax=axes[1], palette="magma")
axes[1].set_title("Top 10 Popular Products", fontweight="bold")
axes[1].set_xlabel("Unique customers")
plt.tight_layout()
save(fig, "recommendation_overview.png")

# 9.3 Recommendation category distribution (sample)
sample_recs = recos_df.sample(min(5000, len(recos_df)), random_state=42)
sample_recs = sample_recs.merge(
    popular_products[["product_id", "category"]],
    on="product_id", how="left"
)
fig, ax = plt.subplots(figsize=(12, 6))
cat_dist = sample_recs["category"].value_counts().head(15)
sns.barplot(x=cat_dist.values, y=cat_dist.index, ax=ax, palette="coolwarm")
ax.set_title("Top Categories in Recommendations (sample)", fontweight="bold")
ax.set_xlabel("Count")
plt.tight_layout()
save(fig, "recommendation_categories.png")


# ============================================================
# 10. Save model artifacts
# ============================================================
section("[10] Saving model artifacts")

joblib.dump(customer_products, os.path.join(MODEL_DIR, "customer_products.pkl"))
joblib.dump(customer_categories, os.path.join(MODEL_DIR, "customer_categories.pkl"))
joblib.dump(top_per_category, os.path.join(MODEL_DIR, "top_per_category.pkl"))
joblib.dump(popular_products, os.path.join(MODEL_DIR, "popular_products.pkl"))

print(f"   [OK] Model artifacts saved in {MODEL_DIR}/")


# ============================================================
# 11. Summary
# ============================================================
section("[11] SUMMARY")

print(f"\n   Method:              Hybrid (Category + Popularity)")
print(f"   Total products:      {len(product_stats):,}")
print(f"   Popular products:    {len(popular_products):,}")
print(f"   Total categories:    {len(category_stats):,}")
print(f"   Total customers:     {len(active_customers):,}")
print(f"   Recommendations:     {len(recos_df):,}")
print(f"   Category Hit Rate:   {hit_rate_cat*100:.2f}%")

print("\n   [SUCCESS] Recommendation system complete!")