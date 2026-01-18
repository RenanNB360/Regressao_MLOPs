import logging
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import os
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

    is_experiment = os.getenv('DVC_EXP_NAME') is not None
    extra_args = {}
    if is_experiment:
        runs = mlflow.search_runs(
            experiment_ids=[os.getenv('MLFLOW_EXPERIMENT_ID')],
            filter_string='tags.dvc_exp = "True"',
            order_by=['start_time DESC'],
        )
        if runs.empty:
            with mlflow.start_run() as parent_run:
                mlflow.set_tag('dvc_exp', True)
                parent_run_id = parent_run.info.run_id
        else:
            parent_run_id.runs.iloc[0].run_id
        run_name = os.getenv('DVC_EXP_NAME')
        extra_args = {
            'parent_run_id': parent_run_id,
            'run_name': run_name,
            'nested': True,
        }

    with mlflow.start_run(**extra_args):
        mlflow.log_params(params=params)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

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
