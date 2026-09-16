"""
Tests for RAG (NL→SQL) pipeline — no LLM calls.
"""

import re

import pytest


class TestIntentClassification:
    """Test intent detection logic (no LLM)."""

    def test_sql_prompt_exists(self):
        """SQL prompt should be defined."""
        try:
            from rag.sql_agent import SQL_PROMPT
            assert "{schema}" in SQL_PROMPT
            assert "{question}" in SQL_PROMPT
        except ImportError:
            pytest.skip("rag.sql_agent not importable")

    def test_execute_sql_simple(self):
        """Execute a simple SQL query."""
        try:
            from rag.sql_agent import execute_sql
        except ImportError:
            pytest.skip("rag.sql_agent not importable")

        df = execute_sql("SELECT 1 AS x")
        assert len(df) == 1
        assert df["x"][0] == 1

    def test_execute_sql_from_fact_sales(self):
        """Execute a query on the actual DB."""
        try:
            from rag.sql_agent import execute_sql
        except ImportError:
            pytest.skip("rag.sql_agent not importable")

        df = execute_sql(
            "SELECT ROUND(SUM(total_amount), 2) AS total FROM fact_sales"
        )
        assert len(df) == 1
        assert df["total"][0] > 0