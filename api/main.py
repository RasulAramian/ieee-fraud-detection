import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import lightgbm as lgb
import pandas as pd

from api.schemas import (
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
    stats_path = "data/processed/global_stats.json"

    try:
        # Load LightGBM model
        if os.path.exists(model_path):
            model_container["model"] = lgb.Booster(model_file=model_path)
            logger.info("LightGBM model loaded successfully.")
        else:
            logger.warning(f"Model file not found at {model_path}.")

        # Initialize and load FeatureTransformer stats if available
        transformer = FeatureTransformer()
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                transformer.global_stats = json.load(f)
            logger.info("Feature transformer stats loaded successfully.")
        else:
            logger.warning(f"Global stats file not found at {stats_path}. Running with base transformer.")
            
        model_container["transformer"] = transformer

    except Exception as e:
        logger.error(f"Failed to load model artifacts during startup: {str(e)}")

    yield
    model_container.clear()
    logger.info("FastAPI service shutdown complete.")


# FastAPI application instance
app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="Production-ready inference service for credit card fraud detection",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service and model status."""
    model_loaded = "model" in model_container and "transformer" in model_container
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    """Score an incoming transaction for fraud probability."""
    if "model" not in model_container or "transformer" not in model_container:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Service is in fallback state.",
        )
    
    try:
        transformer = model_container["transformer"]
        model = model_container["model"]
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([transaction.model_dump()])
        
        # Apply transformation
        processed_data = transformer.transform(input_data)
        
        # Predict probability
        prob = float(model.predict(processed_data)[0])
        
        # Default threshold optimization value (0.3505)
        is_fraud = prob >= 0.3505
        
        return {
            "TransactionID": transaction.TransactionID,
            "fraud_probability": prob,
            "is_fraud": is_fraud,
        }
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
