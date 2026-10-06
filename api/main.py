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

    # Use absolute paths based on the project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("DEBUG BASE_DIR:", BASE_DIR)
    
    model_path = os.path.join(BASE_DIR, "models", "lgb_model.txt")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    print("DEBUG model path:", model_path)
    print("DEBUG model exists:", os.path.exists(model_path))

    try:
        # Load LightGBM model
        if os.path.exists(model_path):
            model_container["model"] = lgb.Booster(model_file=model_path)
            logger.info("LightGBM model loaded successfully.")
        else:
            logger.warning(f"Model file not found at {model_path}.")

        # Load actual feature engineering artifacts produced by build_features.py
        card1_mean_path = os.path.join(processed_dir, "card1_mean.json")
        card1_std_path = os.path.join(processed_dir, "card1_std.json")
        global_freq_path = os.path.join(processed_dir, "global_freq.json")
        cat_mappings_path = os.path.join(processed_dir, "cat_mappings.json")

        card1_mean, card1_std, global_freq, cat_mappings = {}, {}, {}, {}

        if os.path.exists(card1_mean_path):
            with open(card1_mean_path, "r") as f:
                card1_mean = json.load(f)
        if os.path.exists(card1_std_path):
            with open(card1_std_path, "r") as f:
                card1_std = json.load(f)
        if os.path.exists(global_freq_path):
            with open(global_freq_path, "r") as f:
                global_freq = json.load(f)
        if os.path.exists(cat_mappings_path):
            with open(cat_mappings_path, "r") as f:
                cat_mappings = json.load(f)

        # Initialize FeatureTransformer with actual artifacts
        transformer = FeatureTransformer(
            card1_mean=card1_mean,
            card1_std=card1_std,
            global_freq=global_freq,
            cat_mappings=cat_mappings,
        )
        model_container["transformer"] = transformer
        logger.info("FeatureTransformer initialized with project artifacts successfully.")

    except Exception as e:
        logger.error(f"Failed to load model artifacts during startup: {str(e)}")
        print("DEBUG EXCEPTION DURING LOAD:", str(e))

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
    """Score an incoming transaction for fraud probability with strict feature alignment."""
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
