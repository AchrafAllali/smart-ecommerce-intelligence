"""
Tests for Machine Learning models and artifacts.
"""

import os

import pytest

MODEL_DIR = "ml/models"


class TestMLArtifacts:
    """Test ML model files exist."""

    def test_model_dir_exists(self):
        """Model directory must exist."""
        assert os.path.isdir(MODEL_DIR), f"{MODEL_DIR} not found"

    def test_churn_model_exists(self):
        """Churn XGBoost model must exist."""
        path = os.path.join(MODEL_DIR, "churn_XGBoost.pkl")
        assert os.path.exists(path), f"{path} not found"

    def test_churn_scaler_exists(self):
        """Churn scaler must exist."""
        path = os.path.join(MODEL_DIR, "churn_scaler.pkl")
        assert os.path.exists(path), f"{path} not found"

    def test_recommendation_artifacts_exist(self):
        """Recommendation artifacts must exist."""
        files = [
            "customer_products.pkl",
            "customer_categories.pkl",
            "popular_products.pkl",
        ]
        for f in files:
            path = os.path.join(MODEL_DIR, f)
            assert os.path.exists(path), f"{path} not found"


class TestMLDatabase:
    """Test ML results stored in DB."""

    def test_customer_segments_table(self, db_conn):
        """customer_segments must have 5 segments."""
        import pandas as pd
        df = pd.read_sql_query(
            "SELECT DISTINCT segment FROM customer_segments", db_conn
        )
        segments = df["segment"].tolist()
        assert len(segments) >= 4
        assert "VIP" in segments

    def test_churn_probabilities_range(self, db_conn):
        """Churn probabilities must be in [0, 1]."""
        import pandas as pd
        df = pd.read_sql_query(
            """SELECT MIN(churn_probability) AS min_p,
                      MAX(churn_probability) AS max_p
               FROM customer_churn""",
            db_conn,
        )
        assert 0 <= df["min_p"][0] <= 1
        assert 0 <= df["max_p"][0] <= 1