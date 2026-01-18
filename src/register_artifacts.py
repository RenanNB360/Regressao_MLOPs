import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger("src.model_registry.register_artifacts")
logging.basicConfig(level=logging.INFO)

EXPERIMENT_NAME = "ml_regression"
MODEL_NAME = "xgboost_regression_model"
METRIC_NAME = "RMSE"


client = MlflowClient()


def get_best_run(experiment_id: str):
    logger.info("Searching for best run based on RMSE")

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{METRIC_NAME} ASC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No runs found for experiment")

    best_run = runs[0]
    logger.info(
        f"Best run found: {best_run.info.run_id} "
        f"with {METRIC_NAME}={best_run.data.metrics.get(METRIC_NAME)}"
    )

    return best_run


def register_model():
    logger.info("Starting model registration process")

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    best_run = get_best_run(experiment.experiment_id)
    run_id = best_run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    try:
        client.create_registered_model(MODEL_NAME)
        logger.info(f"Registered model '{MODEL_NAME}' created")
    except mlflow.exceptions.MlflowException:
        logger.info(f"Registered model '{MODEL_NAME}' already exists")

    client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id,
    )

    logger.info(
        f"Model '{MODEL_NAME}' registered successfully "
        f"from run {run_id}"
    )


def main():
    register_model()
    logger.info("Model registration completed successfully")


if __name__ == "__main__":
    main()
