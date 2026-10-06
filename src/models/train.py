import os
from typing import Dict, List, Tuple
import lightgbm as lgb
import pandas as pd
from src.data.loader import ChunkedDataLoader
from src.features.build_features import FeatureStatsBuilder
from src.features.features import FeatureTransformer
from src.utils.logger import get_logger

logger = get_logger("train_model")


class IncrementalTrainer:
    """Orchestrates Out-of-Core incremental training for LightGBM across memory-bounded data chunks."""

    def __init__(
        self,
        model_dir: str = "models",
        params: Dict = None,
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
        """Runs the complete 2-Pass pipeline: Pass 1 Stats extraction & Pass 2 Chunked training."""
        loader = ChunkedDataLoader(data_dir=data_dir, chunk_size=chunk_size)

        # Pass 1: Fit and extract global stats
        builder = FeatureStatsBuilder()
        c1_mean, c1_std, global_freq, cat_mappings = builder.fit(loader)
        transformer = FeatureTransformer(
            c1_mean, c1_std, global_freq, cat_mappings
        )

        booster = None
        val_data = None

        logger.info(
            "Starting Pass 2: Incremental Training across data chunks..."
        )

        for chunk_idx, chunk in enumerate(
            loader.stream_transactions("train", merge_identity=True)
        ):
            transformed_chunk = transformer.transform(chunk)

            # Hold-out chunk for validation (e.g., Chunk 6)
            if chunk_idx == self.val_chunk_idx:
                logger.info(
                    f"Holding out Chunk {chunk_idx + 1} strictly for Validation..."
                )
                val_data = transformed_chunk
                continue

            logger.info(
                f"Training on Chunk {chunk_idx + 1} (Size: {len(transformed_chunk)})..."
            )

            X_train = transformed_chunk.drop(
                columns=["isFraud", "TransactionID"], errors="ignore"
            )
            y_train = transformed_chunk["isFraud"]

            train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

            booster = lgb.train(
                self.params,
                train_ds,
                num_boost_round=100,
                init_model=booster,
                keep_training_booster=True,
            )

        # Save trained booster model
        model_path = os.path.join(self.model_dir, "lgb_model.txt")
        booster.save_model(model_path)
        logger.info(
            f"Training finished successfully. Saved model to {model_path}"
        )

        if val_data is not None:
            self._evaluate_validation(booster, val_data)

        return booster

    def _evaluate_validation(
        self, booster: lgb.Booster, val_df: pd.DataFrame
    ) -> None:
        """Evaluates model performance on the held-out validation chunk."""
        from sklearn.metrics import roc_auc_score

        X_val = val_df.drop(
            columns=["isFraud", "TransactionID"], errors="ignore"
        )
        y_val = val_df["isFraud"]

        preds = booster.predict(X_val)
        auc_score = roc_auc_score(y_val, preds)
        logger.info(f"--- Hold-out Validation AUC Score: {auc_score:.5f} ---")


if __name__ == "__main__":
    trainer = IncrementalTrainer()
    trainer.train_pipeline()
