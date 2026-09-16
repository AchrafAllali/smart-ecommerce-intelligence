"""
Shared pytest fixtures.
"""

import os
import sqlite3
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB_PATH = "database/ecommerce.db"


@pytest.fixture(scope="session")
def db_path():
    """Path to SQLite database."""
    return DB_PATH


@pytest.fixture(scope="session")
def db_exists():
    """True if database exists."""
    return os.path.exists(DB_PATH)


@pytest.fixture(scope="session")
def db_conn():
    """SQLite connection (session-scoped)."""
    if not os.path.exists(DB_PATH):
        pytest.skip("Database not found. Run ETL first.")
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def expected_tables():
    """Tables expected in the database."""
    return [
        "fact_sales",
        "dim_customers",
        "dim_products",
        "dim_sellers",
        "orders",
        "reviews",
        "payments",
        "customer_segments",
        "customer_churn",
        "popular_products",
        "recommendations",
    ]