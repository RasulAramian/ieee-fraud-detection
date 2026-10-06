printf '%s\n' \
"# IEEE-CIS Fraud Detection - Production Pipeline" \
"" \
"A modular, production-ready machine learning pipeline and REST API for the **IEEE-CIS Fraud Detection** competition dataset. Designed with clean software engineering practices, memory-efficient out-of-core processing, and robust deployment guardrails." \
"" \
"## 🚀 Key Features" \
"" \
"- **Out-of-Core Processing**: A memory-optimized 3-pass pipeline capable of handling large-scale tabular datasets without RAM overflow." \
"- **Robust Feature Engineering**: Automated statistical aggregations and transformations (src/features/)." \
"- **Production Modeling**: Built around a high-performance **LightGBM** engine trained on validated temporal splits (with XGBoost explored during experimentation)." \
"- **FastAPI Inference Service**: Real-time transaction scoring via a high-performance REST API with strict Pydantic request/response validation." \
"- **Dockerized Environment**: Fully containerized setup (Python 3.12-slim) with optimized layers and security guardrails (.dockerignore protecting raw data leakage while preserving inference metadata)." \
"- **Automated Testing**: Pytest-based test suites covering API endpoints and core pipeline components." \
"" \
"## 📁 Project Structure" \
"" \
'```text' \
"ieee-fraud-detection/" \
"├── api/" \
"│   ├── main.py            # FastAPI application & endpoints (/health, /predict)" \
"│   └── schemas.py         # Pydantic validation schemas" \
"├── src/" \
"│   ├── data/" \
"│   │   └── loader.py      # Chunked data loading utilities" \
"│   ├── features/" \
"│   │   ├── build_features.py  # Pass 1: Statistical aggregations" \
"│   │   └── features.py        # Pass 2: Feature transformation pipeline" \
"│   ├── models/" \
"│   │   ├── train.py       # LightGBM training script" \
"│   │   └── predict.py     # Chunked batch inference script" \
"│   └── utils/" \
"│       ├── logger.py      # Centralized logging utilities" \
"│       └── memory.py      # Memory optimization utilities" \
"├── tests/" \
"│   ├── test_api.py        # Unit tests for FastAPI endpoints" \
"│   └── test_features.py   # Unit tests for feature engineering logic" \
"├── notebooks/             # Exploratory data analysis & baseline notebooks" \
"├── Dockerfile             # Container definition" \
"├── requirements.txt       # Production dependencies" \
"├── requirements-dev.txt   # Development dependencies" \
"└── README.md              # Project documentation" \
'```' \
"" \
"## 📊 Model Performance & Results" \
"" \
"The out-of-core training pipeline was evaluated on the IEEE-CIS competition splits, achieving competitive performance against standard full-memory approaches while maintaining a strict memory footprint." \
"" \
"### Leaderboard & Validation Summary" \
"| Pipeline Iteration | Public AUC | Private AUC | Key Highlights |" \
"| :--- | :---: | :---: | :--- |" \
"| **Initial LightGBM Baseline** | 0.9080 | 0.8853 | Basic feature encoding |" \
"| **Feature-Engineered Baseline** | 0.9116 | 0.8889 | Added frequency encodings & time features |" \
"| **Out-of-Core Incremental Baseline** | **0.9144** | **0.8906** | 2-Pass Chunked LGBM + Full Feature Alignment |" \
"" \
"- **Validation Strategy**: A strict time-aware hold-out split was used rather than a random split. Fraud patterns evolve over time, so evaluating on a later time window provides a realistic estimate of generalization to future transactions (achieving **0.9003 ROC-AUC**)." \
"- **Decision Threshold**: The default classification threshold of 0.5 was optimized to **0.3505** based on validation-set F1 performance, reflecting the highly imbalanced, asymmetric nature of fraud detection." \
"" \
"## ⚙️️ Production Model" \
"" \
"- **Inference Engine**: LightGBM." \
"- **Execution Strategy**: Trained via an out-of-core incremental strategy to prevent memory overflow." \
"- **Artifact Alignment**: The FastAPI service loads the model alongside feature-engineering schema artifacts to precisely reproduce training-time transformations during inference." \
"" \
"## 📂 Dataset Setup" \
"" \
"To run the pipeline locally, download the IEEE-CIS Fraud Detection dataset from Kaggle and place the raw CSV files into the \`data/raw/\` directory with the following structure:" \
"" \
'```text' \
"data/" \
"└── raw/" \
"    ├── train_transaction.csv" \
"    ├── train_identity.csv" \
"    ├── test_transaction.csv" \
"    └── test_identity.csv" \
'```' \
"" \
"## 🛠 Installation & Setup" \
"" \
"### 1. Clone the Repository" \
'```bash' \
"git clone [https://github.com/RasulAramian/ieee-fraud-detection.git](https://github.com/RasulAramian/ieee-fraud-detection.git)" \
"cd ieee-fraud-detection" \
'```' \
"" \
"### 2. Create and Activate Virtual Environment" \
'```bash' \
"python -m venv .venv" \
"source .venv/bin/activate  # On Windows: .venv\Scripts\activate" \
'```' \
"" \
"### 3. Install Dependencies" \
'```bash' \
"pip install -r requirements.txt" \
'```' \
"" \
"## 🧪 Running Tests" \
"" \
"To verify the integrity of the feature engineering logic and API routing, run pytest:" \
'```bash' \
"pytest tests/" \
'```' \
"" \
"## 🐳 Docker Deployment & Security" \
"" \
"### Build the Docker Image" \
'```bash' \
"docker build -t ieee-fraud-detection:v1 ." \
'```' \
"" \
"### Run the Container" \
'```bash' \
"docker run -d --name fraud_app -p 8000:8000 ieee-fraud-detection:v1" \
'```' \
"" \
"- **Security Considerations**: Raw training data (\`data/raw/\`) is strictly excluded from the Docker build context via \`.dockerignore\` to keep large datasets and sensitive artifacts out of container images." \
"" \
"## 👨‍💻 Author" \
"**Rasul Aramian**" \
> README.md
