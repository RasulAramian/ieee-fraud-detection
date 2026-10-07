import json
import os
import pickle
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

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(BASE_DIR, "models", "lgb_model.txt")
    artifacts_path = os.path.join(BASE_DIR, "models", "artifacts.pkl")

    try:
        # Load LightGBM model booster
        if os.path.exists(model_path):
            model_container["model"] = lgb.Booster(model_file=model_path)
            logger.info("LightGBM model loaded successfully.")
        else:
            logger.warning(f"Model file not found at {model_path}.")

        # Load feature engineering artifacts consistently from artifacts.pkl
        if os.path.exists(artifacts_path):
            with open(artifacts_path, "rb") as f:
                artifacts = pickle.load(f)
            
            if isinstance(artifacts, FeatureTransformer):
                transformer = artifacts
            else:
                transformer = FeatureTransformer(
                    card1_mean=artifacts.get("card1_mean", {}),
                    card1_std=artifacts.get("card1_std", {}),
                    global_freq=artifacts.get("global_freq", {}),
                    cat_mappings=artifacts.get("cat_mappings", {}),
                )
            model_container["transformer"] = transformer
            logger.info("FeatureTransformer initialized from artifacts.pkl successfully.")
        else:
            logger.warning(f"Artifacts file not found at {artifacts_path}.")

    except Exception as e:
        logger.error(f"Failed to load model artifacts during startup: {str(e)}")

    yield
    model_container.clear()
    logger.info("FastAPI service shutdown complete.")


# FastAPI application instance
app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="Production-oriented inference service for credit card fraud detection",
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
    """Score an incoming transaction for fraud probability with strict feature alignment."""
    if "model" not in model_container or "transformer" not in model_container:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Service is in fallback state.",
        )
    
    try:
        transformer = model_container["transformer"]
        model = model_container["model"]
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([transaction.model_dump()])
        
        # Apply transformation
        processed_data = transformer.transform(input_data)
        
        # Strict feature alignment and categorical mapping expected by LightGBM booster
        required_features = model.feature_name()
        for col in required_features:
            if col not in processed_data.columns:
                processed_data[col] = 0
                
        X = processed_data[required_features].copy()
        
        # Convert all object/string types to category cleanly without raising mismatch
        for col in X.select_dtypes(include=["object", "string"]).columns:
            X[col] = X[col].astype("category")
        
        # Predict probability
        prob = float(model.predict(X)[0])
        
        # Optimized decision threshold
        is_fraud = prob >= 0.3505
        
        return {
            "TransactionID": transaction.TransactionID,
            "fraud_probability": prob,
            "is_fraud": is_fraud,
        }
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
