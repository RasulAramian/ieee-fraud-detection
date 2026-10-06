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
        """
        Runs the training pipeline:
        Pass 1: Global statistics extraction
        Pass 2: Chunked model training
        """
