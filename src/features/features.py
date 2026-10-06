from typing import Any, Dict, List
import pandas as pd
from src.utils.memory import reduce_mem_usage


class FeatureTransformer:
    """Applies time-based derivations, aggregated Z-scores, frequency encodings,

    and label encodings to incoming transaction chunks.
    """

    def __init__(
        self,
        card1_mean: Dict[Any, float],
        card1_std: Dict[Any, float],
        global_freq: Dict[str, Dict[Any, int]],
        cat_mappings: Dict[str, Dict[str, int]],
    ) -> None:
        self.card1_mean = card1_mean
        self.card1_std = card1_std
        self.global_freq = global_freq
        self.cat_mappings = cat_mappings

    def transform(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a single DataFrame chunk using precomputed mappings."""
        df = chunk_df.copy()
        new_cols = {}

        # 1. Cyclical and Time-based Features
        new_cols["hour"] = (df["TransactionDT"] // 3600) % 24
        new_cols["day_cycle"] = (df["TransactionDT"] // (3600 * 24)) % 7

        # 2. Card1 Mean & Deviation Metrics
        c1_m = df["card1"].map(self.card1_mean)
        c1_s = df["card1"].map(self.card1_std)

        new_cols["TransactionAmt_zscore_card1"] = (
            df["TransactionAmt"] - c1_m
        ) / (c1_s + 1e-5)
        new_cols["TransactionAmt_to_mean_card1"] = df["TransactionAmt"] / (
            c1_m + 1e-5
        )
        new_cols["TransactionAmt_diff_mean_card1"] = df["TransactionAmt"] - c1_m

        # 3. Global Frequency Encodings
        for col, freq_dict in self.global_freq.items():
            if col in df.columns:
                new_cols[f"{col}_fq_enc"] = (
                    df[col].map(pd.Series(freq_dict)).fillna(0)
                )

        df = df.assign(**new_cols)

        # 4. Categorical Label Encodings using precomputed index maps
        for col, mapping in self.cat_mappings.items():
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype("int16")
                )

        # 5. Handle remaining object/categorical columns
        for c in df.select_dtypes(
            include=["object", "string", "category"]
        ).columns:
            df[c] = df[c].astype("category").cat.codes.astype("int16")

        return reduce_mem_usage(df)
