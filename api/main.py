import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import lightgbm as lgb
import pandas as pd

from api.schemas import (
    BatchTransactionInput,
    PredictionOutput,
    TransactionInput,
)
from src.features.features import FeatureTransformer
from src.utils.logger import get_logger

logger = get_logger("fastapi_service")

# Global state containers
model_container = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to pre-load model and metadata artifacts on service startup."""
    logger.info("Initializing FastAPI service & loading model artifacts...")

    model_path = "models/lgb_model.txt"
    artifacts_dir = "data/processed"

    if not os.path.exists(model_path):
        logger.warning(
            f"Model file not found at {model_path}. API will run in fallback state."
        )
    else:
        booster = lgb.Booster(model_file=model_path)
        model_container["booster"] = booster
        model_container["features"] = booster.feature_name()

    # Load artifacts if available
    try:
        with open(os.path.join(artifacts_dir, "card1_mean.json")) as f:
            c1_m = {float(k): v for k, v in json.load(f).items()}
        with open(os.path.join(artifacts_dir, "card1_std.json")) as f:
            c1_s = {float(k): v for k, v in json.load(f).items()}
        with open(os.path.join(artifacts_dir, "global_freq.json")) as f:
            g_freq = json.load(f)
        with open(os.path.join(artifacts_dir, "cat_mappings.json")) as f:
            c_maps = json.load(f)

        model_container["transformer"] = FeatureTransformer(
            c1_m, c1_s, g_freq, c_maps
        )
        logger.info("Artifacts successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading metadata artifacts: {e}")

    yield
    logger.info("Shutting down FastAPI service...")
    model_container.clear()


app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="Production-grade real-time inference endpoint for financial fraud detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for container readiness probes."""
    is_ready = "booster" in model_container
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_single(transaction: TransactionInput):
    """Inference endpoint for real-time transaction processing."""
    if "booster" not in model_container:
        raise HTTPException(
            status_code=503, detail="Model is not loaded or ready."
        )

    # Convert request to pandas DataFrame
    data_dict = transaction.model_dump()
    df = pd.DataFrame([data_dict])

    # Transform features
    transformer = model_container.get("transformer")
    if transformer:
        df = transformer.transform(df)

    # Align features with model expected input schema
    required_features = model_container["features"]
    for col in required_features:
        if col not in df.columns:
            df[col] = 0

    X = df[required_features]
    prob = float(model_container["booster"].predict(X)[0])

    return PredictionOutput(
        TransactionID=transaction.TransactionID,
        fraud_probability=round(prob, 5),
        is_fraud=prob >= 0.3505,  # Validated optimal threshold from hold-out validation
    )
