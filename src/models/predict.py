import os
from typing import Dict
import lightgbm as lgb
import pandas as pd
from src.data.loader import ChunkedDataLoader
from src.features.features import FeatureTransformer
from src.utils.logger import get_logger

logger = get_logger("predict_model")


class BatchInferenceEngine:
    """Generates predictions for test datasets in streaming chunks using trained LightGBM models."""

    def __init__(
        self,
        model_path: str = "models/lgb_model.txt",
        artifacts_dir: str = "data/processed",
    ) -> None:
        self.model_path = model_path
        self.artifacts_dir = artifacts_dir
        self.booster = lgb.Booster(model_file=self.model_path)
        self.model_features = self.booster.feature_name()

    def generate_predictions(
        self,
        transformer: FeatureTransformer,
        data_dir: str = "data/raw",
        chunk_size: int = 100000,
        output_filename: str = "submission_out_of_core_baseline.csv",
    ) -> pd.DataFrame:
        """Streams test transactions, applies transformations, and collects predictions."""
        loader = ChunkedDataLoader(data_dir=data_dir, chunk_size=chunk_size)
        submission_list = []

        logger.info(
            "Starting Pass 3: Streaming inference on Test transactions..."
        )

        for chunk_idx, chunk in enumerate(
            loader.stream_transactions(split="test", merge_identity=True)
        ):
            logger.info(f"Processing Test Chunk {chunk_idx + 1}...")

            transformed_chunk = transformer.transform(chunk)
            X_test = transformed_chunk[self.model_features]

            preds = self.booster.predict(X_test)

            sub_chunk = pd.DataFrame(
                {
                    "TransactionID": transformed_chunk["TransactionID"],
                    "isFraud": preds,
                }
            )
            submission_list.append(sub_chunk)

        final_submission = pd.concat(submission_list, axis=0)
        final_submission.to_csv(output_filename, index=False)
        logger.info(
            f"Successfully generated prediction file: {output_filename}"
        )

        return final_submission
