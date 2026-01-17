import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("src.model_evaluation.evaluate_regression")

SPLIT_DIR = Path("data/splits")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_validation_data():
    logger.info("Loading validation data")

    X_val = pd.read_csv(SPLIT_DIR / "X_val.csv")
    y_val = pd.read_csv(SPLIT_DIR / "y_val.csv").squeeze()

    return X_val, y_val


def load_model():
    model_path = MODELS_DIR / "xgboost_model.joblib"
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def evaluate_model(model, X_val, y_val):
    logger.info("Evaluating model on validation set")

    y_pred = model.predict(X_val)

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "MAE": float(mean_absolute_error(y_val, y_pred)),
        "R2": float(r2_score(y_val, y_pred)),
    }

    return metrics


def save_metrics(metrics: dict):
    output_path = METRICS_DIR / "xgboost_validation_metrics.json"
    logger.info(f"Saving validation metrics to {output_path}")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    logger.info("Starting validation evaluation")

    X_val, y_val = load_validation_data()
    model = load_model()

    metrics = evaluate_model(model, X_val, y_val)
    save_metrics(metrics)

    logger.info(f"Validation metrics: {metrics}")
    logger.info("Validation evaluation completed successfully")


if __name__ == "__main__":
    main()
