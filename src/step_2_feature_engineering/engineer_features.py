import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("src.feature_engineering.engineer_features")


INPUT_PATH = Path("data/cleaned/df_clean.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_clean_data() -> pd.DataFrame:
    """
    Load cleaned dataset.
    """
    logger.info(f"Loading cleaned data from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal and seasonal features.
    """
    logger.info("Creating temporal features")

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)

    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    return df


def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create holiday-related features.
    """
    logger.info("Creating holiday features")

    df["is_holiday"] = df["IsHoliday_x"].astype(int)
    df["holiday_week"] = df["is_holiday"] * df["week"]

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling statistics and lag features.
    """
    logger.info("Creating rolling and lag features")

    df = df.sort_values(["Store", "Dept", "Date"])

    df["rolling_mean_4"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )

    df["lag_1"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .shift(1)
    )

    df["lag_52"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .shift(52)
    )

    # Remove rows without sufficient history
    before = len(df)
    df = df.dropna(subset=["lag_1", "lag_52"])
    logger.info(f"Removed {before - len(df)} rows due to lag NaNs")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables.
    """
    logger.info("Encoding categorical features")

    df = pd.get_dummies(
        df,
        columns=["Type"],
        drop_first=True
    )

    return df


def save_features(df: pd.DataFrame) -> None:
    """
    Save engineered dataset.
    """
    output_path = OUTPUT_DIR / "df_features.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Feature-engineered data saved to {output_path}")


def main() -> None:
    """
    Orchestrate feature engineering pipeline.
    """
    logger.info("Starting feature engineering pipeline")

    df = load_clean_data()

    df = create_time_features(df)
    df = create_holiday_features(df)
    df = create_lag_features(df)
    df = encode_categorical_features(df)

    save_features(df)

    logger.info("Feature engineering pipeline completed successfully")


if __name__ == "__main__":
    main()
