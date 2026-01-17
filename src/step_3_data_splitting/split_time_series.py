import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("src.data_splitting.split_time_series")

DATA_PATH = Path("data/processed/df_features.csv")
PARAMS_PATH = Path("params.yaml")
OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Weekly_Sales_log"
DROP_COLS = ["Weekly_Sales", "Weekly_Sales_log", "Date"]


def load_split_params():
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    split = params["data_split"]

    return (
        split["train_ratio"],
        split["val_ratio"],
        split["test_ratio"],
        pd.to_datetime(split["test_cutoff_date"]),
    )


def load_data() -> pd.DataFrame:
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df.sort_values("Date")


def split_train_val_test(df: pd.DataFrame):
    logger.info("Splitting train / validation / test")

    train_ratio, val_ratio, test_ratio, test_cutoff = load_split_params()

    df_train_val = df[df["Date"] <= test_cutoff].copy()
    df_test = df[df["Date"] > test_cutoff].copy()

    logger.info(
        f"Train+Val max date: {df_train_val['Date'].max()} | "
        f"Test min date: {df_test['Date'].min()}"
    )

    features = [c for c in df.columns if c not in DROP_COLS]

    X = df_train_val[features]
    y = df_train_val[TARGET_COL]

    val_fraction = val_ratio / (train_ratio + val_ratio)
    n_splits = round(1 / val_fraction)

    logger.info(f"Using TimeSeriesSplit with n_splits={n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, val_idx = list(tscv.split(X))[-1]

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    X_test = df_test[features]
    y_test = df_test[TARGET_COL]

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        features,
    )


def save_splits(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    features,
):
    logger.info("Saving data splits")

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    with open(OUTPUT_DIR / "feature_list.json", "w") as f:
        json.dump(features, f, indent=2)


def main():
    logger.info("Starting time series split pipeline")

    df = load_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        features,
    ) = split_train_val_test(df)

    save_splits(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        features,
    )

    logger.info("Time series split completed successfully")


if __name__ == "__main__":
    main()
