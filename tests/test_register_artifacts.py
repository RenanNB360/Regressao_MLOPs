import pytest
import mlflow
from unittest.mock import patch, MagicMock
from src.register_artifacts import get_best_run, register_model


@pytest.fixture
def mock_run():
    run = MagicMock()
    run.info.run_id = "fake_run_id_123"
    run.data.metrics = {"RMSE": 0.85}
    return run

@patch("src.register_artifacts.client")
def test_get_best_run_success(mock_client, mock_run):
    mock_client.search_runs.return_value = [mock_run]
    
    best_run = get_best_run("experiment_id_001")
    
    assert best_run.info.run_id == "fake_run_id_123"
    mock_client.search_runs.assert_called_once()

@patch("src.register_artifacts.client")
def test_get_best_run_not_found(mock_client):
    mock_client.search_runs.return_value = []
    
    with pytest.raises(RuntimeError, match="No runs found for experiment"):
        get_best_run("empty_exp")

@patch("src.register_artifacts.client")
@patch("src.register_artifacts.get_best_run")
def test_register_model_new_model(mock_get_best, mock_client, mock_run):
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp_id"
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_get_best.return_value = mock_run
    
    mock_client.create_registered_model.return_value = None
    
    register_model()
    
    mock_client.create_registered_model.assert_called_with("xgboost_regression_model")
    mock_client.create_model_version.assert_called_once()

@patch("src.register_artifacts.client")
@patch("src.register_artifacts.get_best_run")
def test_register_model_already_exists(mock_get_best, mock_client, mock_run):
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp_id"
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_get_best.return_value = mock_run
    
    mock_client.create_registered_model.side_effect = mlflow.exceptions.MlflowException("Already exists")
    
    register_model()
    
    assert mock_client.create_model_version.called

@patch("src.register_artifacts.client")
def test_register_model_experiment_not_found(mock_client):
    mock_client.get_experiment_by_name.return_value = None
    
    with pytest.raises(RuntimeError, match="Experiment 'ml_regression' not found"):
        register_model()

@patch("src.register_artifacts.register_model")
def test_main_execution(mock_register):
    from src.register_artifacts import main
    main()
    assert mock_register.called