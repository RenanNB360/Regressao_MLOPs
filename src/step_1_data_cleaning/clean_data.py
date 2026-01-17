import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("src.data_preprocessing.clean_data")


RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logger.info("Loading raw datasets")

    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    features = pd.read_csv(RAW_DATA_DIR / "features.csv")
    stores = pd.read_csv(RAW_DATA_DIR / "stores.csv")

    logger.info(
        "Datasets loaded | "
        f"train: {train.shape}, "
        f"features: {features.shape}, "
        f"stores: {stores.shape}"
    )

    return train, features, stores


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if col.lower().startswith("date") and df[col].notna().any():
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() >= df[col].notna().sum():
                df[col] = converted
                logger.info(f"Column converted to datetime: {col}")

    return df


def merge_datasets(
    train: pd.DataFrame,
    features: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Merging datasets")

    df = (
        train
        .merge(features, on=["Store", "Date"], how="left")
        .merge(stores, on="Store", how="left")
    )

    logger.info(f"Merged dataset shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Starting data cleaning")

    TARGET = "Weekly_Sales"

    initial_rows = len(df)
    df = df[df[TARGET] >= 0].copy()
    logger.info(f"Removed {initial_rows - len(df)} rows with negative sales")

    markdown_cols = [col for col in df.columns if "MarkDown" in col]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    df["CPI"] = (
        df.groupby("Store")["CPI"]
        .transform(lambda x: x.fillna(x.median()))
    )

    df["Unemployment"] = (
        df.groupby("Store")["Unemployment"]
        .transform(lambda x: x.fillna(x.median()))
    )

    df["Weekly_Sales_log"] = np.log1p(df["Weekly_Sales"])

    logger.info("Data cleaning completed successfully")
    return df


def save_clean_data(df: pd.DataFrame) -> None:

    output_path = OUTPUT_DIR / "df_clean.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")


def main() -> None:

    logger.info("Starting data cleaning pipeline")

    train, features, stores = load_raw_data()

    train = convert_date_columns(train)
    features = convert_date_columns(features)

    df_merged = merge_datasets(train, features, stores)
    df_clean = clean_data(df_merged)

    if df_clean.isna().sum().sum() > 0:
        logger.warning("Cleaned dataset still contains missing values")

    save_clean_data(df_clean)

    logger.info("Data cleaning pipeline finished successfully")


if __name__ == "__main__":
    main()
