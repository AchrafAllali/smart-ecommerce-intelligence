import pandas as pd


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime."""
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    return orders


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """Fill missing product categories."""
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    return products


def add_category_translation(products: pd.DataFrame,
                              translation: pd.DataFrame) -> pd.DataFrame:
    """Add English category name."""
    products = products.merge(
        translation,
        on="product_category_name",
        how="left"
    )
    products["product_category_name_english"] = (
        products["product_category_name_english"].fillna("unknown")
    )
    return products


def build_fact_sales(orders, order_items, products, customers, payments):
    """Build the central fact_sales table."""

    # 1. Merge order_items with orders
    df = order_items.merge(orders, on="order_id", how="left")

    # 2. Merge with products
    df = df.merge(
        products[[
            "product_id",
            "product_category_name",
            "product_category_name_english",
        ]],
        on="product_id",
        how="left",
    )

    # 3. Merge with customers
    df = df.merge(
        customers[[
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
        ]],
        on="customer_id",
        how="left",
    )

    # 4. Compute total amount
    df["total_amount"] = df["price"] + df["freight_value"]

    # 5. Add date features
    df["year"] = df["order_purchase_timestamp"].dt.year
    df["month"] = df["order_purchase_timestamp"].dt.month
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["year_month"] = (
        df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )
    df["day_of_week"] = df["order_purchase_timestamp"].dt.day_name()

    # 6. Add payment info
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum"),
        )
        .reset_index()
    )
    df = df.merge(payments_agg, on="order_id", how="left")

    return df


def build_dim_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Customer dimension table."""
    return customers[[
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
    ]].drop_duplicates()


def build_dim_products(products: pd.DataFrame) -> pd.DataFrame:
    """Product dimension table."""
    return products[[
        "product_id",
        "product_category_name",
        "product_category_name_english",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]].drop_duplicates()


def build_dim_sellers(sellers: pd.DataFrame) -> pd.DataFrame:
    """Seller dimension table."""
    return sellers[[
        "seller_id",
        "seller_city",
        "seller_state",
    ]].drop_duplicates()


def transform_all(data: dict) -> dict:
    """Apply all transformations and return a dict ready for loading."""
    print("[INFO] Starting transformations...")

    # Cleaning
    orders = clean_orders(data["orders"])
    products = clean_products(data["products"])
    products = add_category_translation(products, data["category_translation"])

    # Build fact_sales
    fact_sales = build_fact_sales(
        orders=orders,
        order_items=data["order_items"],
        products=products,
        customers=data["customers"],
        payments=data["payments"],
    )

    # Build dimensions
    dim_customers = build_dim_customers(data["customers"])
    dim_products = build_dim_products(products)
    dim_sellers = build_dim_sellers(data["sellers"])

    transformed = {
        "fact_sales": fact_sales,
        "dim_customers": dim_customers,
        "dim_products": dim_products,
        "dim_sellers": dim_sellers,
        "orders": orders,
        "reviews": data["reviews"],
        "payments": data["payments"],
        "geolocation": data["geolocation"],
    }

    print("[INFO] Tables created:")
    for name, df in transformed.items():
        print(f"   - {name}: {df.shape}")

    return transformed