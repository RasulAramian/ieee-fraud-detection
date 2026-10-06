import os
from typing import Generator
import pandas as pd
from src.utils.logger import get_logger
from src.utils.memory import reduce_mem_usage

logger = get_logger("data_loader")


class ChunkedDataLoader:
    """Streams large datasets in memory-bounded chunks with automated schema alignment."""

    def __init__(
        self, data_dir: str = "data/raw", chunk_size: int = 100000
    ) -> None:
        self.data_dir = data_dir
        self.chunk_size = chunk_size

    def load_identity(self, split: str = "train") -> pd.DataFrame:
        """Loads and standardizes identity metadata."""
        file_path = os.path.join(self.data_dir, f"{split}_identity.csv")
        logger.info(f"Loading identity data from {file_path}")

        df = pd.read_csv(file_path)
        df.columns = [col.replace("-", "_") for col in df.columns]
        return reduce_mem_usage(df)

    def stream_transactions(
        self, split: str = "train"
    ) -> Generator[pd.DataFrame, None, None]:
        """Generator that streams transaction chunks merged with identity data."""
        identity_df = self.load_identity(split)
        trans_path = os.path.join(self.data_dir, f"{split}_transaction.csv")

        logger.info(
            f"Streaming transactions in chunks of {self.chunk_size} from {trans_path}"
        )

        for chunk_idx, chunk in enumerate(
            pd.read_csv(trans_path, chunksize=self.chunk_size)
        ):
            merged_chunk = pd.merge(
                chunk, identity_df, on="TransactionID", how="left"
            )
            merged_chunk = reduce_mem_usage(merged_chunk)
            yield merged_chunk
