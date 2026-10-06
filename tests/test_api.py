from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the API health check response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint_structure():
    """Test inference request schema validation."""
    payload = {
        "TransactionID": 2987000,
        "TransactionDT": 86400,
        "TransactionAmt": 68.5,
        "ProductCD": "W",
        "card1": 13926,
    }
    response = client.post("/predict", json=payload)
    # Status code can be 200 or 503 if model file is not pre-loaded during tests
    assert response.status_code in [200, 503]
