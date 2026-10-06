from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint_valid():
    payload = {
        "TransactionDT": 86400,
        "TransactionAmt": 150.0,
        "card1": 14290,
        "card2": 111.0,
        "card3": 150.0,
        "card5": 226.0,
        "addr1": 315.0,
        "addr2": 87.0,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "M1": "T",
        "M2": "T",
        "M3": "T",
        "M4": "M0",
        "M5": "T",
        "M6": "T",
        "M7": "T",
        "M8": "T",
        "M9": "T"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    
    # Verify probability bounds
    prob = data["fraud_probability"]
    assert 0.0 <= prob <= 1.0
    
    # Verify binary prediction type
    assert isinstance(data["is_fraud"], int)
    assert data["is_fraud"] in [0, 1]
