from fastapi.testclient import TestClient
from api.main import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

def test_predict_endpoint_valid():
    payload = {
        "TransactionID": 2987000,
        "TransactionDT": 86400,
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card1": 14290,
        "card2": 360.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 315.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com"
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "TransactionID" in data
        assert "is_fraud" in data
        assert "fraud_probability" in data
        
        prob = data["fraud_probability"]
        assert 0.0 <= prob <= 1.0
        assert isinstance(data["is_fraud"], bool)
