import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.step_2_feature_engineering.engineer_features import (
    create_time_features,
    create_holiday_features,
    create_lag_features,
    encode_categorical_features,
    save_features,
    load_clean_data
)

@pytest.fixture
def sample_clean_df():

    dates = pd.date_range(start="2020-01-01", periods=55, freq="W")
    data = {
        "Store": [1] * 55,
        "Dept": [1] * 55,
        "Date": dates,
        "Weekly_Sales": [100.0 + i for i in range(55)],
        "IsHoliday_x": [False] * 55,
        "Type": ["A"] * 55,
        "CPI": [200.0] * 55,
        "Unemployment": [8.0] * 55
    }
    return pd.DataFrame(data)

def test_create_time_features(sample_clean_df):
    df = create_time_features(sample_clean_df.copy())
    
    assert "year" in df.columns
    assert "month" in df.columns
    assert "week" in df.columns
    assert "week_sin" in df.columns
    assert "week_cos" in df.columns
    
    assert df["year"].iloc[0] == 2020
    assert df["month"].iloc[0] == 1
    
    assert -1 <= df["week_sin"].min() <= 1
    assert -1 <= df["week_cos"].max() <= 1

def test_create_holiday_features(sample_clean_df):
    df_input = sample_clean_df.copy()
    df_input.loc[0, "IsHoliday_x"] = True
    
    df = create_time_features(df_input)
    df = create_holiday_features(df)
    
    assert "is_holiday" in df.columns
    assert "holiday_week" in df.columns
    assert df["is_holiday"].iloc[0] == 1
    assert df["is_holiday"].iloc[1] == 0

    assert df["holiday_week"].iloc[0] == df["week"].iloc[0]
    assert df["holiday_week"].iloc[1] == 0

def test_create_lag_features(sample_clean_df):
    df = create_lag_features(sample_clean_df.copy())
    
    assert "rolling_mean_4" in df.columns
    assert "lag_1" in df.columns
    assert "lag_52" in df.columns
    assert len(df) == 3
    
    df_reset = df.reset_index(drop=True)
    
    assert df_reset["lag_1"].iloc[0] == sample_clean_df["Weekly_Sales"].iloc[51]
    assert df_reset["lag_52"].iloc[0] == sample_clean_df["Weekly_Sales"].iloc[0]

def test_encode_categorical_features():
    df_input = pd.DataFrame({
        "Type": ["A", "B", "C", "A"]
    })
    
    df = encode_categorical_features(df_input)
    
    assert "Type_B" in df.columns
    assert "Type_C" in df.columns
    assert "Type_A" not in df.columns
    assert df["Type_B"].iloc[1] == 1

@patch("pandas.read_csv")
def test_load_clean_data(mock_read_csv):
    
    mock_df = pd.DataFrame({"Date": ["2020-01-01"], "Weekly_Sales": [100]})
    mock_read_csv.return_value = mock_df
    
    df = load_clean_data()
    
    assert mock_read_csv.called
    assert isinstance(df, pd.DataFrame)

@patch("pandas.DataFrame.to_csv")
def test_save_features(mock_to_csv):
    df = pd.DataFrame({"test": [1]})
    
    save_features(df)
    
    assert mock_to_csv.called
    args, kwargs = mock_to_csv.call_args
    assert "df_features.csv" in str(args[0])