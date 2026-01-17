import pytest
import pandas as pd
import numpy as np
import yaml
from unittest.mock import patch, mock_open, MagicMock
from src.step_4_model_training.train_model import (
    load_splits,
    load_params,
    train_model,
    save_model,
    main
)

@pytest.fixture
def dummy_train_data():
    X = pd.DataFrame({"feat1": [1, 2, 3, 4, 5], "feat2": [10, 20, 30, 40, 50]})
    y = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
    return X, y

def test_load_params():
    mock_yaml_content = {
        "train": {
            "model": "XGBRegressor",
            "parameters": {
                "n_estimators": 100,
                "learning_rate": 0.1
            }
        }
    }
    
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml_content))):
        params = load_params()
        
        assert params["n_estimators"] == 100
        assert params["learning_rate"] == 0.1

def test_load_params_error():
    mock_yaml_content = {"train": {"model": "RandomForest", "parameters": {}}}
    
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml_content))):
        with pytest.raises(AssertionError, match="Model in params.yaml is not XGBRegressor"):
            load_params()

@patch("pandas.read_csv")
def test_load_splits(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({"col": [1, 2, 3]})
    
    X_train, X_val, y_train, y_val = load_splits()
    
    assert mock_read_csv.call_count == 4
    assert isinstance(y_train, pd.Series)

def test_train_model_logic(dummy_train_data):
    X, y = dummy_train_data
    params = {"n_estimators": 2, "max_depth": 3}
    
    model = train_model(X, y, X, y, params)
    
    from xgboost import XGBRegressor
    assert isinstance(model, XGBRegressor)
    assert model.n_estimators == 2
    assert hasattr(model, "feature_importances_")

@patch("joblib.dump")
def test_save_model(mock_joblib):
    mock_model = MagicMock()
    save_model(mock_model)
    
    assert mock_joblib.called
    args, _ = mock_joblib.call_args
    assert "xgboost_model.joblib" in str(args[1])

@patch("src.step_4_model_training.train_model.load_splits")
@patch("src.step_4_model_training.train_model.load_params")
@patch("src.step_4_model_training.train_model.train_model")
@patch("src.step_4_model_training.train_model.save_model")
def test_main_orchestration(mock_save, mock_train, mock_load_p, mock_load_s):
    
    mock_load_s.return_value = (None, None, None, None)
    mock_load_p.return_value = {}
    mock_train.return_value = MagicMock()
    
    main()
    
    assert mock_load_s.called
    assert mock_load_p.called
    assert mock_train.called
    assert mock_save.called