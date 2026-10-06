import os
import gc
import logging
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.features.features import FeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def reduce_mem_usage(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

def run_training_pipeline(data_dir: str = "data/raw", model_dir: str = "models", chunk_size: int = 100000):
    os.makedirs(model_dir, exist_ok=True)
    
    train_transaction_path = os.path.join(data_dir, "train_transaction.csv")
    train_identity_path = os.path.join(data_dir, "train_identity.csv")
    
    if not os.path.exists(train_transaction_path):
        raise FileNotFoundError(f"Missing {train_transaction_path}")

    logger.info("Loading identity dataset...")
    train_id = pd.read_csv(train_identity_path) if os.path.exists(train_identity_path) else pd.DataFrame()
    if not train_id.empty:
        train_id = reduce_mem_usage(train_id)

    logger.info("==========================================")
    logger.info("PASS 1: Extracting Mappings & Statistics (Chunks 1 to 5 ONLY)...")
    logger.info("==========================================")

    freq_cols = ["card1", "card2", "P_emaildomain"]
    global_freq = {col: {} for col in freq_cols}
    card1_stats = {}

    sample_trans = pd.read_csv(train_transaction_path, nrows=100)
    non_num_trans = sample_trans.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    non_num_id = train_id.select_dtypes(include=["object", "string", "category"]).columns.tolist() if not train_id.empty else []
    cat_cols = list(set(non_num_trans + non_num_id))

    cat_uniques = {col: set() for col in cat_cols}
    if not train_id.empty:
        for col in cat_cols:
            if col in train_id.columns:
                cat_uniques[col].update(train_id[col].dropna().astype(str).unique())

    for chunk_idx, chunk in enumerate(pd.read_csv(train_transaction_path, chunksize=chunk_size)):
        if chunk_idx >= 5:
            break

        for col in freq_cols:
            if col in chunk.columns:
                vc = chunk[col].value_counts().to_dict()
                for k, v in vc.items():
                    global_freq[col][k] = global_freq[col].get(k, 0) + v

        if "card1" in chunk.columns and "TransactionAmt" in chunk.columns:
            for card_id, group in chunk.groupby("card1")["TransactionAmt"]:
                if card_id not in card1_stats:
                    card1_stats[card_id] = []
                card1_stats[card_id].extend(group.tolist())

        for col in cat_cols:
            if col in chunk.columns:
                cat_uniques[col].update(chunk[col].dropna().astype(str).unique())

    card1_global_mean = {k: np.mean(v) for k, v in card1_stats.items()}
    card1_global_std = {k: np.std(v) for k, v in card1_stats.items()}
    del card1_stats

    cat_mappings = {
        col: {val: idx for idx, val in enumerate(sorted(list(vals)))}
        for col, vals in cat_uniques.items()
    }
    del cat_uniques
    gc.collect()

    # Instantiate and save the FeatureTransformer artifact
    transformer = FeatureTransformer(
        card1_mean=card1_global_mean,
        card1_std=card1_global_std,
        global_freq=global_freq,
        cat_mappings=cat_mappings,
    )

    artifacts_path = os.path.join(model_dir, "artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(transformer, f)
    logger.info(f"FeatureTransformer artifacts successfully saved to {artifacts_path}")

    logger.info("PASS 1 Completed Successfully!")
    logger.info("==========================================")
    logger.info("PASS 2: Feature Engineering & Incremental LGBM Training...")
    logger.info("==========================================")

    lgb_booster = None
    val_chunk_df = None

    for chunk_idx, chunk in enumerate(pd.read_csv(train_transaction_path, chunksize=chunk_size)):
        logger.info(f"Processing Chunk {chunk_idx + 1}...")

        if not train_id.empty:
            chunk_df = pd.merge(chunk, train_id, on="TransactionID", how="left")
        else:
            chunk_df = chunk
            
        chunk_df = reduce_mem_usage(chunk_df)

        # Use unified FeatureTransformer instead of duplicated manual logic
        chunk_df = transformer.transform(chunk_df)

        if chunk_idx < 5:
            if "isFraud" not in chunk_df.columns:
                continue
            X_train = chunk_df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
            y_train = chunk_df["isFraud"]

            lgb_clf = lgb.LGBMClassifier(
                n_estimators=50,
                learning_rate=0.03,
                num_leaves=31,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
            )
            lgb_clf.fit(X_train, y_train, init_model=lgb_booster)
            lgb_booster = lgb_clf.booster_
            logger.info(f"Chunk {chunk_idx + 1} trained incrementally!")

            del chunk_df, X_train, y_train, lgb_clf
            gc.collect()
        else:
            logger.info("Chunk 6 held out strictly for Clean Validation.")
            val_chunk_df = chunk_df.copy()
            del chunk_df
            gc.collect()
            break

    if lgb_booster is None:
        raise ValueError("No valid training chunks processed.")

    model_path = os.path.join(model_dir, "lgb_model.txt")
    lgb_booster.save_model(model_path)
    logger.info(f"Incremental model successfully saved to {model_path}")

    if val_chunk_df is not None and "isFraud" in val_chunk_df.columns:
        logger.info("--- Evaluating LightGBM on Strictly Clean Validation Set ---")
        X_val = val_chunk_df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
        y_val = val_chunk_df["isFraud"]
        val_preds = lgb_booster.predict(X_val)
        val_auc = roc_auc_score(y_val, val_preds)
        logger.info(f"Clean Validation ROC-AUC Score: {val_auc:.5f}")

    return lgb_booster

if __name__ == "__main__":
    run_training_pipeline()
