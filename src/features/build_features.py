import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.data.loader import ChunkedDataLoader
from src.utils.logger import get_logger

logger = get_logger("build_features")


class FeatureStatsBuilder:
    """Pass 1: Extracts global statistics, categorical mappings, and frequency distributions

    incrementally to prevent data leakage during out-of-core pipeline execution.
    """

    def __init__(
        self,
        freq_cols: List[str] = None,
        cat_cols: List[str] = None,
        output_dir: str = "data/processed",
    ) -> None:
        self.freq_cols = freq_cols or [
            "card1",
            "card2",
            "card3",
            "card5",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
        ]
        self.cat_cols = cat_cols or [
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
        ]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fit(
        self, loader: ChunkedDataLoader
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[Any, int]], Dict[str, Dict[str, int]]]:
        """Iterates over training chunks to extract global continuous and categorical stats."""
        logger.info("Starting Pass 1: Global statistics and mappings extraction...")

        card1_sum: Dict[float, float] = {}
        card1_sq_sum: Dict[float, float] = {}
        card1_counts: Dict[float, int] = {}

        global_freq: Dict[str, Dict[Any, int]] = {
            col: {} for col in self.freq_cols
        }
        cat_values: Dict[str, set] = {col: set() for col in self.cat_cols}

        for chunk in loader.stream_transactions(
            split="train"
        ):
            # Accumulate sum & sum of squares for card1
            valid_card1 = chunk[["card1", "TransactionAmt"]].dropna()
            for c1, amt in zip(
                valid_card1["card1"], valid_card1["TransactionAmt"]
            ):
                card1_sum[c1] = card1_sum.get(c1, 0.0) + amt
                card1_sq_sum[c1] = card1_sq_sum.get(c1, 0.0) + (amt**2)
                card1_counts[c1] = card1_counts.get(c1, 0) + 1

            # Accumulate global frequency encoding counts
            for col in self.freq_cols:
                if col in chunk.columns:
                    counts = chunk[col].value_counts(dropna=False).to_dict()
                    for val, count in counts.items():
                        global_freq[col][val] = (
                            global_freq[col].get(val, 0) + count
                        )

            # Accumulate unique categorical strings
            for col in self.cat_cols:
                if col in chunk.columns:
                    uniques = chunk[col].dropna().astype(str).unique()
                    cat_values[col].update(uniques)

        # Compute continuous statistics
        card1_mean = {
            c1: card1_sum[c1] / card1_counts[c1] for c1 in card1_sum
        }
        card1_std = {
            c1: (
                np.sqrt(
                    (card1_sq_sum[c1] / card1_counts[c1]) - (card1_mean[c1] ** 2)
                )
                if card1_counts[c1] > 1
                else 0.0
            )
            for c1 in card1_sum
        }

        # Build integer category mappings
        cat_mappings = {
            col: {val: idx for idx, val in enumerate(sorted(vals))}
            for col, vals in cat_values.items()
        }

        logger.info(
            "Pass 1 completed successfully. Saving artifacts to disk..."
        )
        self.save_artifacts(card1_mean, card1_std, global_freq, cat_mappings)

        return card1_mean, card1_std, global_freq, cat_mappings

    def save_artifacts(
        self,
        card1_mean: Dict[str, float],
        card1_std: Dict[str, float],
        global_freq: Dict[str, Dict[Any, int]],
        cat_mappings: Dict[str, Dict[str, int]],
    ) -> None:
        """Saves calculated statistical dictionaries to JSON artifacts."""
        artifacts = {
            "card1_mean": {str(k): v for k, v in card1_mean.items()},
            "card1_std": {str(k): v for k, v in card1_std.items()},
            "global_freq": {
                col: {str(k): v for k, v in freqs.items()}
                for col, freqs in global_freq.items()
            },
            "cat_mappings": cat_mappings,
        }

        for name, data in artifacts.items():
            path = os.path.join(self.output_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f)
            logger.info(f"Saved artifact: {path}")
