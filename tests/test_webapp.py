"""
Tests for the Web UI (Flask).
"""

import pytest

try:
    from webapp.app import app as web_app
    WEBAPP_AVAILABLE = True
except Exception:
    WEBAPP_AVAILABLE = False


@pytest.mark.skipif(not WEBAPP_AVAILABLE, reason="Webapp not importable")
class TestWebUI:
    """Test Web UI pages."""

    @pytest.fixture
    def client(self):
        web_app.config["TESTING"] = True
        with web_app.test_client() as c:
            yield c

    def test_home(self, client):
        """GET /home returns 200."""
        r = client.get("/home")
        assert r.status_code == 200
        assert b"Smart E-Commerce" in r.data

    def test_overview(self, client):
        """GET / returns KPIs page."""
        r = client.get("/")
        assert r.status_code == 200
        assert b"Global KPIs" in r.data or b"Revenue" in r.data

    def test_sales_page(self, client):
        """GET /sales returns 200."""
        r = client.get("/sales")
        assert r.status_code == 200

    def test_segments_page(self, client):
        """GET /segments returns 200."""
        r = client.get("/segments")
        assert r.status_code == 200
        assert b"Customer Segments" in r.data

    def test_churn_page(self, client):
        """GET /churn returns 200."""
        r = client.get("/churn")
        assert r.status_code == 200

    def test_recommendations_page(self, client):
        """GET /recommendations returns 200."""
        r = client.get("/recommendations")
        assert r.status_code == 200

    def test_assistant_page(self, client):
        """GET /assistant returns 200."""
        r = client.get("/assistant")
        assert r.status_code == 200
        assert b"AI" in r.data or b"Assistant" in r.data

    def test_explorer_page(self, client):
        """GET /explorer returns 200."""
        r = client.get("/explorer")
        assert r.status_code == 200