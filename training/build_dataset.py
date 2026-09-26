"""Build the multi-stock training dataset for the prediction model."""

import pandas as pd

from src.features import FEATURE_COLUMNS, build_features
from src.market_data import get_stock_history


TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "INTC", "AVGO",
    "NFLX", "CRM", "ORCL", "ADBE", "QCOM",
    "JPM", "BAC", "GS", "MS", "V",
    "MA", "AXP", "C", "WFC", "SCHW",
    "WMT", "COST", "HD", "NKE", "SBUX",
    "DIS", "MCD", "KO", "PEP", "PG",
    "XOM", "CVX", "CAT", "BA", "GE",
    "UNH", "JNJ", "PFE", "LLY", "MRK",
    "UBER", "ABNB", "SHOP", "PLTR", "COIN",
]


def build_training_dataset(
    tickers: list[str],
    period: str = "5y",
) -> pd.DataFrame:
    """Build one feature dataset from multiple stocks."""
    spy_data = get_stock_history("SPY", period=period)
    datasets = []

    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")

            stock_data = get_stock_history(ticker, period=period)
            features = build_features(stock_data, spy_data)

            features = features[
                FEATURE_COLUMNS + ["target"]
            ].copy()

            features["ticker"] = ticker
            features["date"] = features.index

            datasets.append(features)

        except Exception as error:
            print(f"Skipping {ticker}: {error}")

    return pd.concat(datasets, ignore_index=True)


if __name__ == "__main__":
    dataset = build_training_dataset(TICKERS)

    print()
    print("Rows:", len(dataset))
    print("Stocks:", dataset["ticker"].nunique())
    print("Abnormal moves:", dataset["target"].sum())
    print("Target rate:", dataset["target"].mean())

    dataset.to_csv(
        "training/training_data.csv",
        index=False,
    )

    print("Saved training/training_data.csv")
