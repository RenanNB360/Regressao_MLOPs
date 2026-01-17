# ruff: noqa: PLR2004
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, mock_open
from src.step_3_data_splitting.split_time_series import (
    load_split_params,
    split_train_val_test,
    save_splits,
    load_data
)

@pytest.fixture
def mock_params_content():
    return """
    data_split:
      train_ratio: 0.7
      val_ratio: 0.15
      test_ratio: 0.15
      test_cutoff_date: '2022-01-01'
    """

@pytest.fixture
def sample_features_df():

    dates = pd.date_range(start="2021-10-01", periods=20, freq="W")
    df = pd.DataFrame({
        "Date": dates,
        "Store": [1] * 20,
        "Dept": [1] * 20,
        "Weekly_Sales": np.random.rand(20) * 1000,
        "Weekly_Sales_log": np.random.rand(20) * 7,
        "Feature1": np.random.rand(20),
        "Feature2": np.random.rand(20)
    })
    return df

def test_load_split_params(mock_params_content):
    with patch("builtins.open", mock_open(read_data=mock_params_content)):
        tr, vr, ter, cutoff = load_split_params()
        
        assert tr == 0.7
        assert vr == 0.15
        assert ter == 0.15
        assert cutoff == pd.Timestamp("2022-01-01")

def test_split_train_val_test(sample_features_df):

    with patch("src.step_3_data_splitting.split_time_series.load_split_params") as mock_params:
        mock_params.return_value = (0.7, 0.15, 0.15, pd.Timestamp("2021-12-15"))
        
        X_train, X_val, X_test, y_train, y_val, y_test, features = split_train_val_test(sample_features_df)
        
        assert "Date" not in features
        assert "Weekly_Sales" not in features
        assert "Weekly_Sales_log" not in features
        assert "Feature1" in features
        
        assert y_train.name == "Weekly_Sales_log"
        
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        
        assert (X_test.index.size > 0)
        assert len(X_train) + len(X_val) + len(X_test) == 20

@patch("pandas.Series.to_csv")
@patch("pandas.DataFrame.to_csv")
@patch("json.dump")
def test_save_splits(mock_json, mock_df_to_csv, mock_series_to_csv):
    X = pd.DataFrame({"a": [1]})
    y = pd.Series([1])
    feats = ["a"]
    
    with patch("builtins.open", mock_open()):
        save_splits(X, X, X, y, y, y, feats)
    
    assert mock_df_to_csv.call_count == 3
    assert mock_series_to_csv.call_count == 3
    assert mock_json.called

@patch("src.step_3_data_splitting.split_time_series.pd.read_csv")
def test_load_data(mock_read):
    mock_read.return_value = pd.DataFrame({
        "Date": ["2022-01-02", "2022-01-01"],
        "Val": [2, 1]
    })
    
    df = load_data()
    
    assert df["Date"].iloc[0] == "2022-01-01"