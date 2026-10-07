# IEEE-CIS Fraud Detection

A production-oriented machine learning pipeline and REST API for detecting fraudulent transactions using the IEEE-CIS Fraud Detection dataset. The project covers the complete ML lifecycle: Raw Data -> Feature Engineering -> Time-Aware Validation -> Incremental Training -> Model Persistence -> FastAPI -> Docker -> Render.

---

## Architecture

```text
                        IEEE-CIS Dataset
                               │
                               ▼
                    Chunked Data Loading
                               │
                               ▼
                      Feature Engineering
                               │
                               ▼
                     Time-Aware Validation
                               │
                               ▼
                    Incremental LightGBM
                               │
                               ▼
                   Model + Feature Artifacts
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
         Batch Prediction                  FastAPI
                                              │
                                              ▼
                                           Docker
                                              │
                                              ▼
                                            Render
```

---

## Key Features

- **Memory-Efficient Data Processing**: Transaction data is processed in chunks using `pandas` to prevent RAM exhaustion on large tabular data.
- **Time-Aware Validation**: Uses a strict temporal split to simulate real-world production settings and prevent data leakage.
- **Consistent Feature Transformation**: Centralizes feature engineering inside `FeatureTransformer` to eliminate train-serve skew.
- **Incremental LightGBM Training**: Trains models across chunks using LightGBM's continued boosting capability (`init_model`).
- **Optimized Classification Threshold**: Tunes the decision threshold using validation F1-score rather than defaulting to 0.5.
- **REST API**: Provides real-time transaction scoring via FastAPI and Pydantic validation.
- **Containerized Deployment**: Packaged with Docker and deployed as a live web service on Render.
- **Automated Testing**: Verified using `pytest` for API health, endpoints, and transformation logic.

---

## Tech Stack

- **Core**: Python 3.12, Pandas, NumPy
- **Machine Learning**: LightGBM, Scikit-Learn
- **API & Validation**: FastAPI, Pydantic, Uvicorn
- **DevOps & Deployment**: Docker, Render, Git/GitHub
- **Testing**: Pytest

---

## Memory-Efficient Processing

The IEEE-CIS dataset creates significant memory pressure when loaded entirely into RAM. Instead of processing the complete transaction table at once, the pipeline uses chunked loading:

```python
for chunk in pd.read_csv("train_transaction.csv", chunksize=chunk_size):
    # Process and transform chunk iteratively
```

This approach allows feature engineering, training, and inference to operate efficiently under constrained resources.

---

## Model Performance

### Kaggle Competition Results
The following scores were obtained from separate competition submissions during model iteration:

| Pipeline Iteration | Public AUC | Private AUC |
| :--- | :---: | :---: |
| Initial LightGBM Baseline | 0.9080 | 0.8853 |
| Feature-Engineered Baseline | 0.9116 | 0.8889 |
| Out-of-Core Incremental Baseline | **0.9144** | **0.8906** |

### Local Validation
A time-aware hold-out validation strategy was implemented, achieving a local validation **ROC-AUC of 0.9003**.

### Classification Threshold
Because fraud detection is highly imbalanced, the default 0.5 decision threshold was optimized on the validation set, resulting in an optimal threshold of **0.3505**.

---

## API Usage

### Start the Service Locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Health Check
```bash
curl http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction Endpoint (`POST /predict`)
Send a JSON payload containing transaction attributes to receive a fraud probability and classification decision.

---

## Deployment

The FastAPI inference service is containerized with Docker and deployed as a live web service on Render:

- **Live Service URL**: `https://ieee-fraud-detection-dsjf.onrender.com`
- **Endpoints**:
  - `GET /health`: Service health and model artifact status.
  - `POST /predict`: Real-time transaction scoring.

---

## Docker

The Docker image is built using a lightweight Python base image:
- Base image: `python:3.12-slim`
- Excludes raw training datasets via `.dockerignore`
- Installs necessary system dependencies (e.g., `libgomp1` for LightGBM)

---

## Testing

Run the test suite using `pytest`:
```bash
pytest tests/
```

The test suite covers API health checks, prediction behavior, response validation, and feature transformation logic.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RasulAramian/ieee-fraud-detection.git
   cd ieee-fraud-detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset
The IEEE-CIS Fraud Detection dataset is available through Kaggle. Place the downloaded CSV files under:
```text
data/
└── raw/
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    └── test_identity.csv
```
Note that the raw dataset is excluded from version control via `.dockerignore` and `.gitignore`.

---

## Project Structure

```text
ieee-fraud-detection/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
├── models/
│   ├── artifacts.pkl
│   └── lgb_model.txt
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/
│   ├── test_api.py
│   └── test_features.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Author

**Rasul Aramian**
