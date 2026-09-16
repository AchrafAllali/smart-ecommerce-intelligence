"""
Tests for the REST API (Flask).
"""

import pytest

# Try to import the API app
try:
    from api.app import app as api_app
    API_AVAILABLE = True
except Exception:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="API app not importable")
class TestRESTAPI:
    """Test REST API endpoints."""

    @pytest.fixture
    def client(self):
        api_app.config["TESTING"] = True
        with api_app.test_client() as c:
            yield c

    def test_root(self, client):
        """GET / returns API info."""
        r = client.get("/")
        assert r.status_code == 200
        data = r.get_json()
        assert "name" in data
        assert "Smart E-Commerce" in data["name"]

    def test_health(self, client):
        """GET /api/health returns ok."""
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"

    def test_kpis(self, client):
        """GET /api/kpis returns KPIs."""
        r = client.get("/api/kpis")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_revenue" in data
        assert "total_orders" in data
        assert "total_customers" in data
        assert data["total_revenue"] > 0

    def test_monthly_sales(self, client):
        """GET /api/monthly-sales returns list."""
        r = client.get("/api/monthly-sales")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "year_month" in data[0]
        assert "revenue" in data[0]

    def test_top_categories(self, client):
        """GET /api/top-categories returns top 10."""
        r = client.get("/api/top-categories?limit=5")
        assert r.status_code == 200
        data = r.get_json()
        assert len(data) <= 5
        assert "category" in data[0]

    def test_top_cities(self, client):
        """GET /api/top-cities works."""
        r = client.get("/api/top-cities?limit=5")
        assert r.status_code == 200
        data = r.get_json()
        assert len(data) <= 5

    def test_segments(self, client):
        """GET /api/segments returns 5 segments."""
        r = client.get("/api/segments")
        assert r.status_code == 200
        data = r.get_json()
        assert len(data) == 5
        segments = [s["segment"] for s in data]
        assert "VIP" in segments
        assert "Loyal" in segments

    def test_segment_not_found(self, client):
        """GET /api/segment/<invalid> returns 404."""
        r = client.get("/api/segment/nonexistent_id_xyz")
        assert r.status_code == 404

    def test_churn_top_risk(self, client):
        """GET /api/churn/top-risk returns list."""
        r = client.get("/api/churn/top-risk?limit=5")
        assert r.status_code == 200
        data = r.get_json()
        assert len(data) <= 5
        assert "churn_probability" in data[0]

    def test_recommendations_fallback(self, client):
        """GET /api/recommendations/<invalid> returns fallback."""
        r = client.get("/api/recommendations/nonexistent_id_xyz")
        assert r.status_code == 200
        data = r.get_json()
        assert data["fallback"] is True

    def test_popular_products(self, client):
        """GET /api/popular-products works."""
        r = client.get("/api/popular-products?limit=5")
        assert r.status_code == 200
        data = r.get_json()
        assert len(data) <= 5