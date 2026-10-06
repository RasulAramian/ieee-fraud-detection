# IEEE-CIS Fraud Detection - Production Pipeline

A modular, production-ready machine learning pipeline and REST API for the **IEEE-CIS Fraud Detection** competition dataset. Designed with clean software engineering practices, memory-efficient out-of-core processing, and robust compliance controls.

## 🚀 Key Features

- **Out-of-Core Processing**: A memory-optimized 3-pass pipeline capable of handling large-scale tabular datasets without RAM overflow.
- **Robust Feature Engineering**: Automated statistical aggregations and transformations (src/features/).
- **High-Performance Modeling**: Implements LightGBM and XGBoost trained on validated baseline splits (achieving strong baseline AUC performance).
- **FastAPI Inference Service**: Real-time transaction scoring via a high-performance REST API with strict Pydantic request/response validation.
- **Dockerized Environment**: Fully containerized setup (Python 3.12-slim) with optimized layers and security guardrails (.dockerignore protecting raw data leakage while preserving inference metadata).
- **Comprehensive Unit Testing**: Rigorous test suites covering feature logic and API endpoints using pytest.

## 📁 Project Structure

```text
ieee-fraud-detection/
├── api/
│   ├── main.py            # FastAPI application & endpoints (/health, /predict)
│   └── schemas.py         # Pydantic validation schemas
├── src/
│   ├── data/
│   │   └── loader.py      # Chunked data loading utilities
│   ├── features/
│   │   ├── build_features.py  # Pass 1: Statistical aggregations
│   │   └── features.py        # Pass 2: Feature transformation pipeline
│   ├── models/
│   │   ├── train.py       # LightGBM / XGBoost training script
│   │   └── predict.py     # Chunked batch inference script
│   └── utils/
│       ├── logger.py      # Centralized logging utilities
│       └── memory.py      # Memory optimization utilities
├── tests/
│   ├── test_api.py        # Unit tests for FastAPI endpoints
│   ├── test_data.py       # Unit tests for data loading logic
│   └── test_features.py   # Unit tests for feature engineering logic
├── notebooks/             # Exploratory data analysis & baseline notebooks
├── Dockerfile             # Container definition
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── README.md              # Project documentation
```
## 🛠 Installation & Setup

### 1. Clone the Repository
git clone https://github.com/rasularamian/ieee-fraud-detection.git
cd ieee-fraud-detection

### 2. Create and Activate Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

## 🧪 Running Tests

To verify the integrity of the feature engineering logic and API routing, run pytest:
pytest tests/

## 🐳 Docker Deployment

### Build the Docker Image
docker build -t ieee-fraud-detection:v1 .

### Run the Container
docker run -d --name fraud_app -p 8000:8000 ieee-fraud-detection:v1

Once running, access the interactive API documentation (Swagger UI) at:
http://localhost:8000/docs

## 🔒 Compliance & Security

- **Data Privacy**: Raw data files (data/raw/) are strictly excluded via .dockerignore to prevent unauthorized inclusion in build artifacts.
- **Metadata Preservation**: Only necessary schema JSONs (data/processed/*.json) are packaged for live inference.

## 👨‍💻 Author
**Rasul Aramian**
