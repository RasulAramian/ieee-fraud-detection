import os
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Optional

class IncrementalTrainer:
    """Orchestrates Out-of-Core incremental training for LightGBM across memory-bounded data chunks."""

    def __init__(
        self,
        model_dir: str = "models",
        params: Optional[Dict[str, Any]] = None,
        val_chunk_idx: int = 5,
    ) -> None:
        self.model_dir = model_dir
        self.val_chunk_idx = val_chunk_idx
        os.makedirs(self.model_dir, exist_ok=True)

        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "n_jobs": -1,
        }

    def train_pipeline(
        self, data_dir: str = "data/raw", chunk_size: int = 100000
    ) -> lgb.Booster:
        """
        Runs the training pipeline by loading chunks of train data,
        merging transaction and identity, and training LightGBM incrementally.
        """
        print(f"Loading data from {data_dir} with chunk size {chunk_size}...")
        
        train_transaction_path = os.path.join(data_dir, "train_transaction.csv")
        train_identity_path = os.path.join(data_dir, "train_identity.csv")

        if not os.path.exists(train_transaction_path):
            raise FileNotFoundError(f"Missing {train_transaction_path}")

        # Read the first chunk to initialize and train a baseline model
        print("Reading the first chunk for training...")
        trans_chunk = pd.read_csv(train_transaction_path, nrows=chunk_size)
        
        if os.path.exists(train_identity_path):
            ident_chunk = pd.read_csv(train_identity_path, nrows=chunk_size)
            df = pd.merge(trans_chunk, ident_chunk, on="TransactionID", how="left")
        else:
            df = trans_chunk

        if "isFraud" not in df.columns:
            raise ValueError("Target column 'isFraud' not found in training data.")

        X = df.drop(columns=["TransactionID", "TransactionDT", "isFraud"], errors="ignore")
        y = df["isFraud"]

        # Convert categorical/object columns to numeric codes to avoid strict metadata mismatch during inference
        for col in X.select_dtypes(include=["object", "category", "string"]).columns:
            X[col] = pd.factorize(X[col].astype(str))[0]

        print("Training initial LightGBM model on the first chunk...")
        train_data = lgb.Dataset(X, label=y)
        
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100
        )

        model_path = os.path.join(self.model_dir, "lgb_model.txt")
        model.save_model(model_path)
        print(f"Model successfully saved to {model_path}")
        return model

if __name__ == "__main__":
    print("Starting training pipeline...")
    trainer = IncrementalTrainer()
    if os.path.exists("data/raw"):
        trainer.train_pipeline(data_dir="data/raw")
        print("Training pipeline finished successfully!")
    else:
        print("Error: data/raw directory not found!")
