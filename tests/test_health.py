"""
Tests for health check endpoints.
"""


def test_health_returns_200(client):
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_payload(client):
    data = response = client.get("/api/health").json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_health_ready(client):
    response = client.get("/api/health/ready")
    assert response.status_code in (200, 503)  # 503 if DB not connected


def test_health_live(client):
    response = client.get("/api/health/live")
    assert response.status_code == 200
