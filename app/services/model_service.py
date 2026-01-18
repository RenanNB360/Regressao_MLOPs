import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger("app.model_service")


class ModelService:
    def __init__(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading XGBoost regression model")

        model_path = Path("models/xgboost_model.joblib")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)

        logger.info("Model loaded successfully")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(features)
        return predictions
