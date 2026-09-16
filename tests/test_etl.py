"""
Tests for ETL pipeline and database structure.
"""

import os
import sqlite3

import pandas as pd
import pytest


class TestETLDatabase:
    """Test the SQLite database built by the ETL."""

    def test_database_exists(self, db_path):
        """Database file must exist."""
        assert os.path.exists(db_path), (
            f"Database {db_path} not found. Run 'python main.py' first."
        )

    def test_database_not_empty(self, db_path):
        """Database must have a non-zero size."""
        assert os.path.getsize(db_path) > 1024, "Database is empty"

    def test_expected_tables_exist(self, db_conn, expected_tables):
        """All expected tables must be present."""
        cursor = db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        for table in expected_tables:
            assert table in tables, f"Missing table: {table}"

    def test_fact_sales_not_empty(self, db_conn):
        """fact_sales must contain data."""
        df = pd.read_sql_query("SELECT COUNT(*) as n FROM fact_sales", db_conn)
        assert df["n"][0] > 0, "fact_sales is empty"

    def test_fact_sales_has_required_columns(self, db_conn):
        """fact_sales must have required columns."""
        df = pd.read_sql_query("SELECT * FROM fact_sales LIMIT 1", db_conn)
        required = [
            "order_id",
            "customer_unique_id",
            "product_id",
            "total_amount",
            "order_purchase_timestamp",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_total_revenue_positive(self, db_conn):
        """Total revenue must be positive."""
        df = pd.read_sql_query(
            "SELECT SUM(total_amount) AS total FROM fact_sales", db_conn
        )
        assert df["total"][0] > 0

    def test_no_null_totals(self, db_conn):
        """total_amount must not have NULL values."""
        df = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM fact_sales WHERE total_amount IS NULL",
            db_conn,
        )
        assert df["n"][0] == 0, "Found NULL values in total_amount"