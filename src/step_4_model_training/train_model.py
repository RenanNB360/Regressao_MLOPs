import logging
from pathlib import Path

import joblib
import pandas as pd
import mlflow
from xgboost import XGBRegressor

logger = logging.getLogger("src.model_training.train_xgboost")


SPLIT_DIR = Path("data/splits")
PARAMS_PATH = Path("params.yaml")

MODELS_DIR = Path("models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    logger.info("Loading train/validation splits")

    X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
    X_val = pd.read_csv(SPLIT_DIR / "X_val.csv")
    y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(SPLIT_DIR / "y_val.csv").squeeze()

    return X_train, X_val, y_train, y_val


def load_params():
    import yaml

    logger.info(f"Loading training parameters from {PARAMS_PATH}")

    with open(PARAMS_PATH) as f:
        params_yaml = yaml.safe_load(f)

    train_cfg = params_yaml["train"]

    assert train_cfg["model"] == "XGBRegressor", \
        "Model in params.yaml is not XGBRegressor"

    return train_cfg["parameters"]


def train_model(X_train, y_train, X_val, y_val, params):
    logger.info("Training XGBoost model")

    mlflow.set_experiment('ml_regression')
    mlflow.xgboost.autolog()

    with mlflow.start_run():
        mlflow.log_params(params=params)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        return model


def save_model(model: XGBRegressor) -> None:
    model_path = MODELS_DIR / "xgboost_model.joblib"
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)


def main():
    logger.info("Starting XGBoost training pipeline")

    X_train, X_val, y_train, y_val = load_splits()
    params = load_params()

    model = train_model(X_train, y_train, X_val, y_val, params)
    save_model(model)

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
