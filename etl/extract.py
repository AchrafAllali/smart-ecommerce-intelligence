import pandas as pd
import os

RAW_PATH = "data/raw/"

# Olist dataset file names
FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_raw_data():
    """Read all Olist CSV files from data/raw/."""
    data = {}
    for name, filename in FILES.items():
        path = os.path.join(RAW_PATH, filename)
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue
        data[name] = pd.read_csv(path)
        print(f"[OK] {name}: {data[name].shape}")
    return data


if __name__ == "__main__":
    data = load_raw_data()
    print(f"\n[INFO] Loaded {len(data)} files successfully")