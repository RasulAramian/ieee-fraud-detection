import numpy as np
import pandas as pd
import pytest

from src.features.features import FeatureTransformer
from src.utils.memory import reduce_mem_usage


def test_reduce_mem_usage():
    """Test memory reduction downcasting logic."""
    df = pd.DataFrame(
        {
            "int_col": np.array([1, 2, 3], dtype=np.int64),
            "float_col": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }
    )
    reduced_df = reduce_mem_usage(df)

    assert reduced_df["int_col"].dtype == np.int8
    assert reduced_df["float_col"].dtype == np.float16


def test_feature_transformer():
    """Test feature engineering transformation outputs."""
    card1_mean = {1000: 50.0}
    card1_std = {1000: 10.0}
    global_freq = {"ProductCD": {"W": 100}}
    cat_mappings = {"ProductCD": {"W": 0}}

    transformer = FeatureTransformer(
        card1_mean, card1_std, global_freq, cat_mappings
    )

    sample_chunk = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [86400],
            "TransactionAmt": [60.0],
            "card1": [1000],
            "ProductCD": ["W"],
        }
    )

    transformed = transformer.transform(sample_chunk)

    assert "hour" in transformed.columns
    assert "day_cycle" in transformed.columns
    assert "TransactionAmt_zscore_card1" in transformed.columns
    assert transformed["ProductCD_fq_enc"].iloc[0] == 100
