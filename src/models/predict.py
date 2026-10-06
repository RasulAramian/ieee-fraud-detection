import logging
from pathlib import Path
import pickle
import pandas as pd
import lightgbm as lgb
from src.data.loader import ChunkedDataLoader
from src.features.features import FeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_batch_inference(
    data_dir: str = "data/raw",
    model_path: str = "models/lgb_model.txt",
    artifacts_path: str = "models/artifacts.pkl",
    output_path: str = "data/predictions.csv",
    split: str = "test",
    chunk_size: int = 50000
):
    """Run batch inference on chunked data using the trained LightGBM model and saved artifacts."""
    logger.info(f"Loading feature transformer artifacts from {artifacts_path}...")
    if not Path(artifacts_path).exists():
        raise FileNotFoundError(f"Missing artifacts file at {artifacts_path}. Please run training first.")
        
    with open(artifacts_path, "rb") as f:
        transformer = pickle.load(f)

    logger.info(f"Loading model from {model_path}...")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Missing model file at {model_path}. Please run training first.")
        
    model = lgb.Booster(model_file=model_path)
    expected_features = model.feature_name()

    logger.info("Initializing data loader...")
    loader = ChunkedDataLoader(data_dir=data_dir, chunk_size=chunk_size)

    predictions = []
    
    logger.info(f"Starting chunked inference stream for split: {split}...")
    for chunk in loader.stream_transactions(split=split):
        processed_chunk = transformer.transform(chunk)
        
        # Align features with what the model expects during training
        for col in expected_features:
            if col not in processed_chunk.columns:
                processed_chunk[col] = 0
        
        processed_chunk = processed_chunk[expected_features]
        
        preds = model.predict(processed_chunk)
        
        if "TransactionID" in chunk.columns:
            res_df = pd.DataFrame({
                "TransactionID": chunk["TransactionID"],
                "fraud_probability": preds
            })
        else:
            res_df = pd.DataFrame({
                "fraud_probability": preds
            })
            
        predictions.append(res_df)

    final_preds = pd.concat(predictions, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_preds.to_csv(output_path, index=False)
    logger.info(f"Batch inference complete. Saved {len(final_preds)} predictions to {output_path}")

if __name__ == "__main__":
    run_batch_inference()
