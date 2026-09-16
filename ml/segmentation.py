"""
Customer Segmentation using RFM analysis + K-Means clustering.

Segments (business labels):
    - VIP       : high monetary, frequent buyers
    - Loyal     : recent, good monetary
    - New       : recent, low monetary
    - At Risk   : old, good monetary (used to buy)
    - Lost      : old, low monetary
"""

# ------------------------------------------------------------
# Fix joblib CPU detection on Windows
# ------------------------------------------------------------
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = "database/ecommerce.db"
OUT_DIR = "dashboard"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

FORCED_K = 5  # business-driven choice


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
# 1. Load fact_sales
# ============================================================
section("[1] Loading data")

conn = sqlite3.connect(DB_PATH)
fact_sales = pd.read_sql_query("SELECT * FROM fact_sales", conn)
conn.close()

fact_sales["order_purchase_timestamp"] = pd.to_datetime(
    fact_sales["order_purchase_timestamp"], errors="coerce"
)
print(f"   fact_sales: {fact_sales.shape}")


# ============================================================
# 2. Build RFM table
# ============================================================
section("[2] Building RFM table")

snapshot_date = fact_sales["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
print(f"   Snapshot date: {snapshot_date.date()}")

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
print(rfm.describe().round(2).to_string())


# ============================================================
# 3. RFM Scoring (quartiles 1-4)
# ============================================================
section("[3] RFM Scoring (quartiles 1-4)")

rfm["R_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(
    rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]
).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

rfm["RFM_score"] = (
    rfm["R_score"].astype(str)
    + rfm["F_score"].astype(str)
    + rfm["M_score"].astype(str)
)

print("   Score distribution:")
print(f"   R_score: {rfm['R_score'].value_counts().sort_index().to_dict()}")
print(f"   F_score: {rfm['F_score'].value_counts().sort_index().to_dict()}")
print(f"   M_score: {rfm['M_score'].value_counts().sort_index().to_dict()}")


# ============================================================
# 4. Find optimal K (Elbow + Silhouette)
# ============================================================
section("[4] Finding optimal K (Elbow + Silhouette)")

features = ["R_score", "F_score", "M_score"]
X = rfm[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

SAMPLE_SIZE = 5000
rng = np.random.default_rng(42)
sample_idx = rng.choice(
    len(X_scaled),
    size=min(SAMPLE_SIZE, len(X_scaled)),
    replace=False,
)
X_sample = X_scaled[sample_idx]

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_full = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    score = silhouette_score(X_sample, labels_full[sample_idx])
    silhouettes.append(score)
    print(f"   k={k}: inertia={km.inertia_:.0f}, silhouette={score:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_range), inertias, marker="o", color="steelblue")
axes[0].axvline(FORCED_K, color="red", linestyle="--", label=f"Forced k={FORCED_K}")
axes[0].set_title("Elbow Method", fontweight="bold")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(list(K_range), silhouettes, marker="o", color="coral")
axes[1].axvline(FORCED_K, color="red", linestyle="--", label=f"Forced k={FORCED_K}")
axes[1].set_title("Silhouette Score (sample n=5000)", fontweight="bold")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Score")
axes[1].legend()
plt.tight_layout()
save(fig, "segmentation_optimal_k.png")

best_k = list(K_range)[int(np.argmax(silhouettes))]
print(f"\n   --> Best k by silhouette: {best_k}")
print(f"   --> Using FORCED_K = {FORCED_K} (business-driven)")


# ============================================================
# 5. Fit final K-Means
# ============================================================
section(f"[5] Fitting K-Means with k={FORCED_K}")

km_final = KMeans(n_clusters=FORCED_K, random_state=42, n_init=10)
rfm["cluster"] = km_final.fit_predict(X_scaled)

profile = (
    rfm.groupby("cluster")
    .agg(
        n_customers=("customer_unique_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        R_score=("R_score", "mean"),
        F_score=("F_score", "mean"),
        M_score=("M_score", "mean"),
    )
    .round(2)
)
profile["pct"] = (profile["n_customers"] / profile["n_customers"].sum() * 100).round(2)
print(profile.to_string())


# ============================================================
# 6. Smart business labeling
# ============================================================
section("[6] Assigning business labels")

r_med = profile["recency"].median()
m_med = profile["monetary"].median()


def smart_label(row):
    """Assign a business label based on RFM signature."""
    r = row["recency"]
    f = row["frequency"]
    m = row["monetary"]

    # VIP: high monetary + frequent
    if m >= m_med * 1.3 and f >= 1.05:
        return "VIP"
    # Loyal: recent + good monetary
    if r <= r_med and m >= m_med:
        return "Loyal"
    # New: recent + low monetary
    if r <= r_med and m < m_med:
        return "New"
    # At Risk: old + good monetary
    if r > r_med and m >= m_med:
        return "At Risk"
    # Lost: old + low monetary
    return "Lost"


profile["label"] = profile.apply(smart_label, axis=1)

# Ensure labels are unique
seen = {}
unique_labels = []
for lbl in profile["label"]:
    if lbl in seen:
        seen[lbl] += 1
        unique_labels.append(f"{lbl}_{seen[lbl]}")
    else:
        seen[lbl] = 1
        unique_labels.append(lbl)
profile["label"] = unique_labels

cluster_to_label = profile["label"].to_dict()
rfm["segment"] = rfm["cluster"].map(cluster_to_label)

print("\n   Segment distribution:")
seg_dist = rfm["segment"].value_counts()
for seg, count in seg_dist.items():
    pct = count / len(rfm) * 100
    print(f"   - {seg:12s}: {count:6,} ({pct:5.2f}%)")


# ============================================================
# 7. Visualizations
# ============================================================
section("[7] Visualizations")

# 7.1 Pie + Box
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
seg_dist.plot.pie(autopct="%1.1f%%", ax=axes[0], cmap="Set2")
axes[0].set_title("Customer Segments Distribution", fontweight="bold")
axes[0].set_ylabel("")

sns.boxplot(data=rfm, x="segment", y="monetary", ax=axes[1], palette="Set3")
axes[1].set_title("Monetary Value by Segment", fontweight="bold")
axes[1].set_yscale("log")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
save(fig, "segmentation_distribution.png")

# 7.2 Scatter
fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(
    data=rfm.sample(min(10000, len(rfm)), random_state=42),
    x="recency",
    y="frequency",
    hue="segment",
    palette="Set1",
    alpha=0.6,
    ax=ax,
)
ax.set_title("Customer Segments: Recency vs Frequency", fontweight="bold")
ax.set_xlabel("Recency (days)")
ax.set_ylabel("Frequency (orders)")
plt.tight_layout()
save(fig, "segmentation_scatter.png")

# 7.3 Profile bars
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["recency", "frequency", "monetary"]):
    sns.barplot(data=profile, x="label", y=col, ax=ax, palette="viridis")
    ax.set_title(f"Avg {col.capitalize()} by Segment", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
save(fig, "segmentation_profile.png")


# ============================================================
# 8. Save to SQLite
# ============================================================
section("[8] Saving results to SQLite")

conn = sqlite3.connect(DB_PATH)
rfm.to_sql("customer_segments", conn, if_exists="replace", index=False)
profile.to_sql("segment_profile", conn, if_exists="replace")
conn.close()

print(f"   [OK] customer_segments: {rfm.shape}")
print(f"   [OK] segment_profile:   {profile.shape}")


# ============================================================
# 9. Summary
# ============================================================
section("[9] SUMMARY")

print("\n   Segment profiles:")
print(
    profile[["label", "n_customers", "pct", "recency", "frequency", "monetary"]]
    .sort_values("monetary", ascending=False)
    .to_string()
)

print("\n   [SUCCESS] Segmentation complete!")
print(f"   Best k = {FORCED_K}")
print(f"   Segments: {sorted(rfm['segment'].unique())}")