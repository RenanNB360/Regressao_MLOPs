from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.step_1_data_cleaning.clean_data import (
    clean_data,
    convert_date_columns,
    load_raw_data,
    merge_datasets,
    save_clean_data,
)


@pytest.fixture
def sample_train_df():
    return pd.DataFrame({
        "Store": [1, 1],
        "Date": ["2023-01-01", "2023-01-02"],
        "Weekly_Sales": [100.0, -50.0]
    })


@pytest.fixture
def sample_features_df():
    return pd.DataFrame({
        "Store": [1, 1],
        "Date": ["2023-01-01", "2023-01-02"],
        "MarkDown1": [10.0, np.nan],
        "CPI": [211.0, np.nan],
        "Unemployment": [8.1, np.nan]
    })


@pytest.fixture
def sample_stores_df():
    return pd.DataFrame({
        "Store": [1],
        "Type": ["A"],
        "Size": [150000]
    })


def test_convert_date_columns(sample_train_df):
    df = convert_date_columns(sample_train_df)

    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert df["Date"].iloc[0] == pd.Timestamp("2023-01-01")


def test_merge_datasets(sample_train_df, sample_features_df, sample_stores_df):
    train = convert_date_columns(sample_train_df)
    features = convert_date_columns(sample_features_df)

    df_merged = merge_datasets(train, features, sample_stores_df)

    assert "Weekly_Sales" in df_merged.columns
    assert "MarkDown1" in df_merged.columns
    assert "Type" in df_merged.columns
    assert len(df_merged) == 2


def test_clean_data_logic(sample_features_df):
    df_input = pd.DataFrame({
        "Store": [1, 1, 1],
        "Weekly_Sales": [100.0, 200.0, -10.0],
        "MarkDown1": [10.0, np.nan, np.nan],
        "CPI": [211.0, 211.0, np.nan],
        "Unemployment": [8.0, 8.0, np.nan]
    })

    df_cleaned = clean_data(df_input)

    assert len(df_cleaned) == 2
    assert (df_cleaned["Weekly_Sales"] >= 0).all()

    assert df_cleaned["MarkDown1"].isna().sum() == 0
    assert df_cleaned["MarkDown1"].iloc[1] == 0

    assert df_cleaned["CPI"].isna().sum() == 0
    assert df_cleaned["CPI"].iloc[0] == 211.0

    assert "Weekly_Sales_log" in df_cleaned.columns
    expected_log = np.log1p(100.0)
    assert df_cleaned["Weekly_Sales_log"].iloc[0] == pytest.approx(expected_log)


@patch("pandas.DataFrame.to_csv")
def test_save_clean_data(mock_to_csv, sample_train_df):
    save_clean_data(sample_train_df)

    args, kwargs = mock_to_csv.call_args
    assert "df_clean.csv" in str(args[0])
    assert kwargs["index"] is False


@patch("pandas.read_csv")
def test_load_raw_data(mock_read_csv):

    mock_read_csv.return_value = pd.DataFrame()

    train, features, stores = load_raw_data()

    assert mock_read_csv.call_count == 3
    assert isinstance(train, pd.DataFrame)
