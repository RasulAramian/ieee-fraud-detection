from fastapi.testclient import TestClient
import pytest

from api.main import app


def test_health_endpoint():
    """Test the API health check response."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


def test_predict_endpoint_success():
    """Test successful inference request and response structure."""
    payload = {
        "TransactionID": 2987000,
        "TransactionDT": 86400,
        "TransactionAmt": 68.5,
        "ProductCD": "W",
        "card1": 13926,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert isinstance(data["is_fraud"], bool)
        assert isinstance(data["fraud_probability"], float)
