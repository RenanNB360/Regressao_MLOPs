import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from src.step_5_model_evaluation.evaluate_regression import (
    load_validation_data,
    load_model,
    evaluate_model,
    main
)

@pytest.fixture
def dummy_val_data():
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([10.0, 20.0, 30.0])
    return X, y

@patch("pandas.read_csv")
def test_load_validation_data(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({"col": [1, 2, 3]})
    
    X_val, y_val = load_validation_data()
    
    assert mock_read_csv.call_count == 2
    assert isinstance(y_val, pd.Series)

@patch("joblib.load")
def test_load_model(mock_joblib):
    mock_joblib.return_value = "fake_model"
    model = load_model()
    assert model == "fake_model"
    assert mock_joblib.called

@patch("mlflow.search_runs")
@patch("mlflow.start_run")
@patch("mlflow.log_metrics")
@patch("mlflow.set_experiment")
def test_evaluate_model(mock_set_exp, mock_log, mock_start, mock_search, dummy_val_data):
    X, y = dummy_val_data
    
    mock_runs = MagicMock()
    mock_runs.iloc = [MagicMock(run_id="fake_run_id")]
    mock_search.return_value = mock_runs
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([11.0, 19.0, 31.0])
    
    with patch("builtins.open", mock_open()):
        metrics = evaluate_model(mock_model, X, y)
    
    assert "RMSE" in metrics
    assert metrics["MAE"] == pytest.approx(1.0)
    assert mock_log.called
    assert mock_set_exp.called

@patch("src.step_5_model_evaluation.evaluate_regression.load_validation_data")
@patch("src.step_5_model_evaluation.evaluate_regression.load_model")
@patch("src.step_5_model_evaluation.evaluate_regression.evaluate_model")
def test_main_evaluation_orchestration(mock_eval, mock_load_m, mock_load_d):
    # Setup mocks
    mock_load_d.return_value = (None, None)
    mock_load_m.return_value = MagicMock()
    mock_eval.return_value = {"MAE": 0.1}
    
    main()
    
    assert mock_load_d.called
    assert mock_load_m.called
    assert mock_eval.called